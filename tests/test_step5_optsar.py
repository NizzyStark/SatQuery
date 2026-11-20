"""Unit and integration tests for Step 5: optsar_analyze tool.

Tests verify:
- Optical feature extraction (NDWI with 4-band/NIR and RGB fallback)
- SAR feature extraction (specular low backscatter for water, high backscatter + texture for built-up)
- Multi-sensor fusion logic and agreement statistics
- Evidence overlay rendering (blue for water, orange for built-up)
- Targets parameter filtering ("water", "built_up", "both", and fallback on unknown)
- Complete optsar_analyze execution with mocked VLM
- Synthetic optical/SAR array with a known water region and correct fusion label
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from satquery.tools import (
    ToolResult,
    _extract_optical_features,
    _extract_sar_features,
    _fuse_optsar,
    optsar_analyze,
)
from satquery.vlm import VLMClient


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_optical_sar_pair(
    h: int = 128,
    w: int = 128,
    water_rect: tuple[int, int, int, int] | None = None,
    buildup_rect: tuple[int, int, int, int] | None = None,
    optical_bands: int = 4,
    sar_bands: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic optical and SAR co-registered arrays with known features.

    Returns
    -------
    arr_opt : np.ndarray  (bands, H, W)
    arr_sar : np.ndarray  (bands, H, W)
    gt_water : np.ndarray (H, W) bool
    gt_buildup : np.ndarray (H, W) bool
    """
    rng = np.random.default_rng(seed)

    if water_rect is None:
        water_rect = (int(0.15 * h), int(0.15 * w), int(0.45 * h), int(0.45 * w))
    if buildup_rect is None:
        buildup_rect = (int(0.55 * h), int(0.55 * w), int(0.85 * h), int(0.85 * w))

    wr0, wc0, wr1, wc1 = water_rect
    br0, bc0, br1, bc1 = buildup_rect

    arr_opt = np.zeros((optical_bands, h, w), dtype=np.float32)

    if optical_bands >= 4:
        # Multispectral: B2=0 Blue, B3=1 Green, B4=2 Red, B8=3 NIR
        # Background: vegetation (high NIR, moderate Green)
        arr_opt[0] = rng.uniform(30, 50, size=(h, w))     # Blue
        arr_opt[1] = rng.uniform(60, 90, size=(h, w))     # Green
        arr_opt[2] = rng.uniform(40, 70, size=(h, w))     # Red
        arr_opt[3] = rng.uniform(150, 200, size=(h, w))   # NIR (vegetation canopy)

        # Water: Green > NIR, very low NIR absorption
        arr_opt[0, wr0:wr1, wc0:wc1] = rng.uniform(80, 110, size=(wr1 - wr0, wc1 - wc0))  # Blue
        arr_opt[1, wr0:wr1, wc0:wc1] = rng.uniform(90, 120, size=(wr1 - wr0, wc1 - wc0))  # Green
        arr_opt[2, wr0:wr1, wc0:wc1] = rng.uniform(10, 30, size=(wr1 - wr0, wc1 - wc0))   # Red
        arr_opt[3, wr0:wr1, wc0:wc1] = rng.uniform(5, 20, size=(wr1 - wr0, wc1 - wc0))    # NIR (NDWI > 0.5)

        # Built-up: high visible brightness, NIR slightly higher than Green
        arr_opt[0, br0:br1, bc0:bc1] = rng.uniform(170, 220, size=(br1 - br0, bc1 - bc0))
        arr_opt[1, br0:br1, bc0:bc1] = rng.uniform(160, 200, size=(br1 - br0, bc1 - bc0))
        arr_opt[2, br0:br1, bc0:bc1] = rng.uniform(160, 210, size=(br1 - br0, bc1 - bc0))
        arr_opt[3, br0:br1, bc0:bc1] = rng.uniform(210, 240, size=(br1 - br0, bc1 - bc0))
    elif optical_bands == 3:
        # Standard RGB: Band 0=Red, Band 1=Green, Band 2=Blue
        # Background: vegetation / soil
        arr_opt[0] = rng.uniform(40, 70, size=(h, w))     # Red
        arr_opt[1] = rng.uniform(70, 100, size=(h, w))    # Green
        arr_opt[2] = rng.uniform(30, 50, size=(h, w))     # Blue

        # Water in RGB: dominant Blue, very low Red
        arr_opt[0, wr0:wr1, wc0:wc1] = rng.uniform(10, 25, size=(wr1 - wr0, wc1 - wc0))   # Red low
        arr_opt[1, wr0:wr1, wc0:wc1] = rng.uniform(50, 80, size=(wr1 - wr0, wc1 - wc0))   # Green
        arr_opt[2, wr0:wr1, wc0:wc1] = rng.uniform(140, 180, size=(wr1 - wr0, wc1 - wc0)) # Blue dominant

        # Built-up in RGB: high overall brightness
        arr_opt[0, br0:br1, bc0:bc1] = rng.uniform(180, 230, size=(br1 - br0, bc1 - bc0))
        arr_opt[1, br0:br1, bc0:bc1] = rng.uniform(180, 230, size=(br1 - br0, bc1 - bc0))
        arr_opt[2, br0:br1, bc0:bc1] = rng.uniform(180, 230, size=(br1 - br0, bc1 - bc0))
    else:
        # Grayscale
        arr_opt[0] = rng.uniform(80, 150, size=(h, w))
        arr_opt[0, wr0:wr1, wc0:wc1] = rng.uniform(10, 30, size=(wr1 - wr0, wc1 - wc0))
        arr_opt[0, br0:br1, bc0:bc1] = rng.uniform(200, 240, size=(br1 - br0, bc1 - bc0))

    # 2. SAR image: Sentinel-1 VV (dB-like: background around -12 dB)
    arr_sar = rng.uniform(-14.0, -10.0, size=(sar_bands, h, w)).astype(np.float32)

    # Water in SAR: specular reflection → very low backscatter (-25 to -20 dB)
    arr_sar[0, wr0:wr1, wc0:wc1] = rng.uniform(-25.0, -20.0, size=(wr1 - wr0, wc1 - wc0))

    # Built-up in SAR: corner reflection → high backscatter + spatial texture
    buildup_tex = rng.uniform(-3.0, 3.0, size=(br1 - br0, bc1 - bc0))
    buildup_tex[::2, ::2] += 6.0
    arr_sar[0, br0:br1, bc0:bc1] = buildup_tex

    gt_water = np.zeros((h, w), dtype=bool)
    gt_water[wr0:wr1, wc0:wc1] = True

    gt_buildup = np.zeros((h, w), dtype=bool)
    gt_buildup[br0:br1, bc0:bc1] = True

    return arr_opt, arr_sar, gt_water, gt_buildup


def _make_dict(arr: np.ndarray, modality: str) -> dict[str, Any]:
    return {
        "array": arr,
        "modality": modality,
        "path": f"synthetic_{modality}.png",
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union > 0 else 0.0


# ── Feature Extraction Tests ─────────────────────────────────────────────────


class TestFeatureExtraction:
    """Tests for optical and SAR feature extractors."""

    def test_optical_multispectral_water(self):
        """NDWI on 4-band imagery successfully detects water region."""
        arr_opt, _, gt_water, _ = _make_optical_sar_pair(optical_bands=4)
        water_mask, _, fallbacks = _extract_optical_features(arr_opt)

        assert not fallbacks["optical_water_fallback"]
        iou = _iou(water_mask, gt_water)
        assert iou > 0.8, f"Optical NDWI water IoU {iou:.3f} below 0.8"

    def test_optical_rgb_fallback_water(self):
        """RGB colour-based fallback detects water when NIR is missing."""
        arr_opt, _, gt_water, _ = _make_optical_sar_pair(optical_bands=3)
        water_mask, _, fallbacks = _extract_optical_features(arr_opt)

        assert fallbacks["optical_water_fallback"]
        iou = _iou(water_mask, gt_water)
        assert iou > 0.7, f"Optical RGB water fallback IoU {iou:.3f} below 0.7"

    def test_sar_water_detection(self):
        """SAR low backscatter threshold detects water region."""
        _, arr_sar, gt_water, _ = _make_optical_sar_pair()
        water_mask, _ = _extract_sar_features(arr_sar)

        iou = _iou(water_mask, gt_water)
        assert iou > 0.75, f"SAR water IoU {iou:.3f} below 0.75"

    def test_sar_buildup_detection(self):
        """SAR double-bounce + texture detects built-up region."""
        _, arr_sar, _, gt_buildup = _make_optical_sar_pair()
        _, buildup_mask = _extract_sar_features(arr_sar)

        # Built-up region should be detected with substantial overlap
        overlap = np.sum(buildup_mask & gt_buildup)
        assert overlap > 100, "SAR built-up detector should detect corner reflectors"


# ── Sensor Fusion Tests ──────────────────────────────────────────────────────


class TestSensorFusion:
    """Tests for the rule-based optical-SAR fusion."""

    def test_fusion_water_agreement(self):
        """Both sensors agree on the known water square."""
        arr_opt, arr_sar, gt_water, _ = _make_optical_sar_pair()
        opt_w, opt_b, _ = _extract_optical_features(arr_opt)
        sar_w, sar_b = _extract_sar_features(arr_sar)

        final_w, final_b, stats = _fuse_optsar(opt_w, sar_w, opt_b, sar_b, targets="both")

        assert stats["water_both_pct"] > 0
        assert stats["water_agreement_ratio"] > 0.5
        iou_water = _iou(final_w, gt_water)
        assert iou_water > 0.8, f"Fused water IoU {iou_water:.3f} below 0.8"

    def test_fusion_targets_filter(self):
        """Target parameter filters active masks."""
        arr_opt, arr_sar, _, _ = _make_optical_sar_pair()
        opt_w, opt_b, _ = _extract_optical_features(arr_opt)
        sar_w, sar_b = _extract_sar_features(arr_sar)

        # Target water only: built-up mask should be empty
        final_w, final_b, _ = _fuse_optsar(opt_w, sar_w, opt_b, sar_b, targets="water")
        assert np.any(final_w)
        assert not np.any(final_b)

        # Target built-up only: water mask should be empty
        final_w, final_b, _ = _fuse_optsar(opt_w, sar_w, opt_b, sar_b, targets="built_up")
        assert not np.any(final_w)
        assert np.any(final_b)


# ── Full optsar_analyze Tool Tests (VLM Mocked) ──────────────────────────────


class TestOptSarAnalyzeTool:
    """Integration tests for optsar_analyze with mocked VLM."""

    @staticmethod
    def _mock_vlm() -> MagicMock:
        mock = MagicMock(spec=VLMClient)
        mock.generate.return_value = {
            "text": (
                "Optical imagery shows high NDWI water bodies and high visible "
                "reflectance urban areas. SAR confirms calm specular water surface "
                "and high-texture corner reflections from built infrastructure."
            ),
            "confidence": 0.86,
            "token_count": 35,
        }
        return mock

    def test_full_optsar_pipeline(self, tmp_path):
        """Complete optsar_analyze pipeline produces valid ToolResult."""
        arr_opt, arr_sar, gt_water, _ = _make_optical_sar_pair()
        dict_opt = _make_dict(arr_opt, "optical")
        dict_sar = _make_dict(arr_sar, "sar")
        mock = self._mock_vlm()

        result = optsar_analyze(
            dict_opt, dict_sar,
            targets="both",
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, ToolResult)
        assert result.tool == "optsar_analyze"
        assert len(result.text) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_s > 0

        # Stats
        assert "water_pct" in result.stats
        assert "built_up_pct" in result.stats
        assert "water_agreement_ratio" in result.stats
        assert result.stats["water_pct"] > 5.0

        # Evidence overlay
        assert result.overlay_png is not None
        assert Path(result.overlay_png).exists()
        overlay = Image.open(result.overlay_png)
        assert overlay.mode == "RGB"

        # VLM received both images
        mock.generate.assert_called_once()
        images_passed = mock.generate.call_args.kwargs.get(
            "images", mock.generate.call_args[1].get("images", [])
        )
        assert len(images_passed) == 2

    def test_invalid_target_fallback(self, tmp_path):
        """Invalid targets parameter defaults to 'both'."""
        arr_opt, arr_sar, _, _ = _make_optical_sar_pair(h=64, w=64)
        dict_opt = _make_dict(arr_opt, "optical")
        dict_sar = _make_dict(arr_sar, "sar")
        mock = self._mock_vlm()

        result = optsar_analyze(
            dict_opt, dict_sar,
            targets="invalid_category",
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert result.params_used["targets"] == "both"

    def test_spatial_size_mismatch_auto_aligned(self, tmp_path):
        """Optical and SAR with different grid resolutions are auto-aligned."""
        arr_opt, _, _, _ = _make_optical_sar_pair(h=150, w=150)
        _, arr_sar, _, _ = _make_optical_sar_pair(h=100, w=120)

        dict_opt = _make_dict(arr_opt, "optical")
        dict_sar = _make_dict(arr_sar, "sar")
        mock = self._mock_vlm()

        result = optsar_analyze(
            dict_opt, dict_sar,
            vlm=mock,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, ToolResult)
        assert "alignment_warnings" in result.stats
        assert len(result.stats["alignment_warnings"]) > 0
