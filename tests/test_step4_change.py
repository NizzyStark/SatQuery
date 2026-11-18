"""Unit and integration tests for Step 4: change_analyze tool.

Tests use synthetic bi-temporal raster pairs with known changed regions to
verify:
- Pair alignment (size, band count mismatch handling)
- Change mask computation (Otsu and manual threshold)
- Morphological cleanup and tiny blob removal
- Overlay generation
- IoU of detected change mask against ground truth (must exceed 0.8)
- VLM narration call structure (mocked — no GPU required)
- ToolResult field correctness
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from satquery.tools import (
    ToolResult,
    _align_pair,
    _compute_change_mask,
    change_analyze,
)
from satquery.vlm import VLMClient


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pair_with_change(
    h: int = 256,
    w: int = 256,
    bands: int = 3,
    change_rect: tuple[int, int, int, int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic bi-temporal raster pair with a known changed square.

    The before image has a uniform low-intensity background.  The after image
    is identical everywhere *except* inside ``change_rect``, which gets a
    very large intensity boost to guarantee separation.

    Returns
    -------
    arr_before : np.ndarray  (bands, H, W)
    arr_after  : np.ndarray  (bands, H, W)
    gt_mask    : np.ndarray  (H, W) bool — ground-truth change mask.
    """
    rng = np.random.default_rng(seed)

    if change_rect is None:
        # Default: a central square covering ~15–25% of the image
        margin_r = h // 5
        margin_c = w // 5
        change_rect = (margin_r, margin_c, h - margin_r, w - margin_c)

    r0, c0, r1, c1 = change_rect

    # Before image: uniform low values with tiny noise
    arr_before = rng.uniform(20, 40, size=(bands, h, w)).astype(np.float64)

    # After image: identical to before...
    arr_after = arr_before.copy()
    # ...except the change rectangle gets a huge jump (200+ units)
    arr_after[:, r0:r1, c0:c1] = rng.uniform(200, 255, size=(bands, r1 - r0, c1 - c0))

    # Ground-truth mask
    gt_mask = np.zeros((h, w), dtype=bool)
    gt_mask[r0:r1, c0:c1] = True

    return arr_before, arr_after, gt_mask


def _make_image_dict(
    arr: np.ndarray, modality: str = "optical", path: str = "synthetic.png"
) -> dict[str, Any]:
    """Wrap a numpy array as a minimal image_dict compatible with tools."""
    return {
        "array": arr,
        "modality": modality,
        "path": path,
        "meta": {
            "bands": int(arr.shape[0]),
            "dtype": str(arr.dtype),
            "crs": None,
            "transform": None,
            "size": (int(arr.shape[2]), int(arr.shape[1])),
            "bounds": None,
            "format": "synthetic",
        },
    }


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for two boolean masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.sum(pred_b & gt_b)
    union = np.sum(pred_b | gt_b)
    return float(intersection / union) if union > 0 else 0.0


# ── Test: _align_pair ────────────────────────────────────────────────────────


class TestAlignPair:
    """Tests for the _align_pair helper function."""

    def test_same_size_no_warnings(self):
        a = np.zeros((3, 100, 100), dtype=np.float32)
        b = np.zeros((3, 100, 100), dtype=np.float32)
        a_out, b_out, warnings = _align_pair(a, b)
        assert a_out.shape == b_out.shape == (3, 100, 100)
        assert len(warnings) == 0

    def test_size_mismatch_resize(self):
        a = np.ones((3, 200, 200), dtype=np.float32)
        b = np.ones((3, 100, 150), dtype=np.float32)
        a_out, b_out, warnings = _align_pair(a, b)
        assert a_out.shape == b_out.shape
        assert a_out.shape[1] == 100  # min height
        assert a_out.shape[2] == 150  # min width
        assert any("size" in w.lower() or "mismatch" in w.lower() for w in warnings)

    def test_band_mismatch_truncates(self):
        a = np.ones((4, 100, 100), dtype=np.float32)
        b = np.ones((3, 100, 100), dtype=np.float32)
        a_out, b_out, warnings = _align_pair(a, b)
        assert a_out.shape[0] == b_out.shape[0] == 3
        assert any("band" in w.lower() for w in warnings)

    def test_both_mismatch(self):
        a = np.ones((4, 200, 300), dtype=np.float32)
        b = np.ones((3, 150, 250), dtype=np.float32)
        a_out, b_out, warnings = _align_pair(a, b)
        assert a_out.shape == b_out.shape == (3, 150, 250)
        assert len(warnings) == 2  # band + size


# ── Test: _compute_change_mask ───────────────────────────────────────────────


class TestComputeChangeMask:
    """Tests for the classical change detection pipeline."""

    def test_known_change_otsu(self):
        """Otsu-based change detection finds the known changed square."""
        arr_before, arr_after, gt_mask = _make_pair_with_change()
        mask, diff_map, thr = _compute_change_mask(arr_before, arr_after)

        assert mask.shape == (256, 256)
        assert mask.dtype == bool
        assert diff_map.shape == (256, 256)
        assert 0.0 <= thr <= 1.0

        iou = _iou(mask, gt_mask)
        assert iou > 0.8, f"IoU {iou:.3f} is below 0.8 threshold"

    def test_known_change_manual_threshold(self):
        """Manual threshold also finds the changed square."""
        arr_before, arr_after, gt_mask = _make_pair_with_change()
        mask, diff_map, thr = _compute_change_mask(
            arr_before, arr_after, diff_threshold=0.3
        )

        iou = _iou(mask, gt_mask)
        assert iou > 0.7, f"IoU {iou:.3f} with manual threshold"
        assert abs(thr - 0.3) < 0.01

    def test_no_change(self):
        """Identical images → near-zero change percentage."""
        arr = np.random.default_rng(99).uniform(
            80, 120, size=(3, 128, 128)
        ).astype(np.float64)
        mask, diff_map, thr = _compute_change_mask(arr, arr.copy())

        change_pct = 100.0 * np.sum(mask) / mask.size
        assert change_pct < 2.0, f"Identical images should show <2% change, got {change_pct:.1f}%"

    def test_full_change(self):
        """Completely different images → high change percentage."""
        a = np.zeros((3, 128, 128), dtype=np.float64)
        b = np.full((3, 128, 128), 255.0, dtype=np.float64)
        mask, diff_map, thr = _compute_change_mask(a, b)

        change_pct = 100.0 * np.sum(mask) / mask.size
        assert change_pct > 80.0, f"Fully changed images should show >80% change, got {change_pct:.1f}%"

    def test_tiny_blob_removal(self):
        """Tiny changed blobs below min_blob_area are removed."""
        rng = np.random.default_rng(42)
        h, w, bands = 128, 128, 3
        arr_before = np.full((bands, h, w), 30.0, dtype=np.float64)
        arr_after = arr_before.copy()

        # Add a tiny 3×3 changed blob (9 pixels, below default 50-pixel threshold)
        arr_after[:, 10:13, 10:13] = 250.0

        # Also add a large changed region (should be retained)
        arr_after[:, 50:100, 50:100] = 250.0

        mask, _, _ = _compute_change_mask(arr_before, arr_after, min_blob_area=50)

        # The tiny 3×3 blob region should be mostly removed
        tiny_region = mask[10:13, 10:13]
        assert np.sum(tiny_region) < 5, "Tiny blob should be removed by min_blob_area filter"

        # The large region should largely survive
        large_region = mask[55:95, 55:95]
        assert np.sum(large_region) > 100, "Large change region should be detected"

    def test_threshold_clamping(self):
        """diff_threshold outside 0.1–0.5 is clamped."""
        arr_before, arr_after, _ = _make_pair_with_change(
            h=64, w=64, change_rect=(15, 15, 50, 50)
        )

        _, _, thr_low = _compute_change_mask(arr_before, arr_after, diff_threshold=0.01)
        assert abs(thr_low - 0.1) < 0.01, f"Threshold should be clamped to 0.1, got {thr_low}"

        _, _, thr_high = _compute_change_mask(arr_before, arr_after, diff_threshold=0.9)
        assert abs(thr_high - 0.5) < 0.01, f"Threshold should be clamped to 0.5, got {thr_high}"

    def test_single_band_pair(self):
        """Change detection works on single-band (grayscale) pairs."""
        arr_before = np.full((1, 100, 100), 30.0, dtype=np.float64)
        arr_after = arr_before.copy()
        arr_after[:, 30:70, 30:70] = 240.0

        gt = np.zeros((100, 100), dtype=bool)
        gt[30:70, 30:70] = True

        mask, _, _ = _compute_change_mask(arr_before, arr_after)
        iou = _iou(mask, gt)
        assert iou > 0.7, f"Single-band IoU {iou:.3f}"


# ── Test: change_analyze (full tool, VLM mocked) ────────────────────────────


class TestChangeAnalyzeTool:
    """Integration tests for the change_analyze tool with mocked VLM."""

    @staticmethod
    def _mock_vlm() -> MagicMock:
        mock = MagicMock(spec=VLMClient)
        mock.generate.return_value = {
            "text": "New construction detected in the central area.",
            "confidence": 0.82,
            "token_count": 12,
        }
        return mock

    def test_describe_mode(self, tmp_path):
        """Describe mode produces valid ToolResult with overlay and stats."""
        arr_before, arr_after, gt_mask = _make_pair_with_change()
        dict_before = _make_image_dict(arr_before, path="before.png")
        dict_after = _make_image_dict(arr_after, path="after.png")
        mock = self._mock_vlm()

        result = change_analyze(
            dict_before, dict_after,
            mode="describe",
            vlm=mock,
            output_dir=str(tmp_path),
        )

        # Check ToolResult structure
        assert isinstance(result, ToolResult)
        assert result.tool == "change_analyze"
        assert len(result.text) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_s > 0
        assert result.params_used["mode"] == "describe"

        # Check stats
        assert "change_pct" in result.stats
        assert "changed_pixels" in result.stats
        assert "total_pixels" in result.stats
        assert "threshold_used" in result.stats
        assert result.stats["change_pct"] > 5.0  # Should detect the big changed square

        # Check overlay file exists
        assert result.overlay_png is not None
        assert Path(result.overlay_png).exists()
        overlay_img = Image.open(result.overlay_png)
        assert overlay_img.mode == "RGB"

        # VLM was called with 2 images
        mock.generate.assert_called_once()
        call_args = mock.generate.call_args
        images_arg = call_args.kwargs.get("images", call_args[1].get("images", []))
        assert len(images_arg) == 2

    def test_vqa_mode(self, tmp_path):
        """VQA mode includes the user question in the prompt."""
        arr_before, arr_after, _ = _make_pair_with_change()
        dict_before = _make_image_dict(arr_before)
        dict_after = _make_image_dict(arr_after)
        mock = self._mock_vlm()

        result = change_analyze(
            dict_before, dict_after,
            query="Has the built-up area increased?",
            mode="vqa",
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert result.params_used["mode"] == "vqa"

        # Verify the prompt includes the user question
        call_args = mock.generate.call_args
        prompt_arg = call_args.kwargs.get("prompt", call_args[1].get("prompt", ""))
        assert "built-up area increased" in prompt_arg

    def test_manual_threshold(self, tmp_path):
        """Manual diff_threshold is used and recorded in params_used."""
        arr_before, arr_after, _ = _make_pair_with_change()
        dict_before = _make_image_dict(arr_before)
        dict_after = _make_image_dict(arr_after)
        mock = self._mock_vlm()

        result = change_analyze(
            dict_before, dict_after,
            diff_threshold=0.25,
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert result.params_used["diff_threshold"] == 0.25
        assert abs(result.params_used["threshold_actual"] - 0.25) < 0.01

    def test_invalid_mode_fallback(self, tmp_path):
        """Invalid mode defaults to 'describe'."""
        arr_before, arr_after, _ = _make_pair_with_change(h=64, w=64, change_rect=(10, 10, 50, 50))
        dict_before = _make_image_dict(arr_before)
        dict_after = _make_image_dict(arr_after)
        mock = self._mock_vlm()

        result = change_analyze(
            dict_before, dict_after,
            mode="invalid_mode",
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert result.params_used["mode"] == "describe"

    def test_iou_above_threshold(self, tmp_path):
        """The detected change mask has IoU > 0.8 with the ground truth."""
        arr_before, arr_after, gt_mask = _make_pair_with_change(
            h=256, w=256, change_rect=(80, 80, 180, 180)
        )

        # Directly test the mask quality
        mask, _, _ = _compute_change_mask(arr_before, arr_after)
        iou = _iou(mask, gt_mask)
        assert iou > 0.8, (
            f"Change mask IoU with ground truth is {iou:.3f}, "
            f"which is below the required 0.8 threshold."
        )

    def test_mismatched_sizes_still_works(self, tmp_path):
        """Pairs with different spatial sizes are auto-aligned."""
        arr_before = np.full((3, 200, 200), 30.0, dtype=np.float64)
        arr_after = np.full((3, 150, 180), 30.0, dtype=np.float64)
        arr_after[:, 30:80, 30:80] = 240.0  # Add change

        dict_before = _make_image_dict(arr_before)
        dict_after = _make_image_dict(arr_after)
        mock = self._mock_vlm()

        result = change_analyze(
            dict_before, dict_after,
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, ToolResult)
        assert result.stats["change_pct"] > 0
        assert "alignment_warnings" in result.stats
        assert len(result.stats["alignment_warnings"]) > 0

    def test_confidence_boost_for_reasonable_change(self, tmp_path):
        """Confidence gets a slight boost when change_pct is in a normal range."""
        arr_before, arr_after, _ = _make_pair_with_change()
        dict_before = _make_image_dict(arr_before)
        dict_after = _make_image_dict(arr_after)

        mock = self._mock_vlm()
        # VLM returns 0.82 confidence
        result = change_analyze(
            dict_before, dict_after,
            vlm=mock,
            output_dir=str(tmp_path),
        )

        # With reasonable change_pct (>1%, <95%) confidence should be boosted
        assert result.confidence >= 0.82
