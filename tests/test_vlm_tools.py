"""Smoke tests for VLM, VQA, and caption tools.

These tests are split into two categories:

1. **Unit tests** (always run) — verify module imports, data structures,
   helper functions, and the VLM client construction without loading the model.
2. **Integration tests** (require the model) — marked with
   ``@pytest.mark.slow`` so they can be skipped in quick CI runs.  Run with
   ``pytest -m slow`` to include them.

The integration tests create a synthetic satellite-like PNG, send it through
VQA and caption, and save the baseline outputs to ``docs/baseline.json``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Unit tests (no model download required) ─────────────────────────────


def test_vlm_client_construction():
    """VLMClient can be instantiated without loading the model."""
    from satquery.vlm import VLMClient

    client = VLMClient(device="cpu", use_4bit=False)
    assert not client.is_loaded
    assert client.device == "cpu"
    assert client.model_name == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_vlm_resize_helper():
    """_resize_for_vlm respects max_pixels cap."""
    from satquery.vlm import _resize_for_vlm

    big = Image.new("RGB", (1000, 1000))
    small = _resize_for_vlm(big, max_pixels=448 * 448)
    assert small.size[0] * small.size[1] <= 448 * 448 + 10  # tolerance

    tiny = Image.new("RGB", (100, 100))
    same = _resize_for_vlm(tiny, max_pixels=448 * 448)
    assert same.size == (100, 100)


def test_tool_result_dataclass():
    """ToolResult can be constructed with all fields."""
    from satquery.tools import ToolResult

    r = ToolResult(
        tool="vqa",
        text="Test answer",
        overlay_png=None,
        stats={"token_count": 5},
        confidence=0.85,
        params_used={"max_new_tokens": 64},
        latency_s=1.23,
    )
    assert r.tool == "vqa"
    assert r.confidence == 0.85


def test_confidence_computation():
    """_compute_confidence returns valid probability."""
    from satquery.vlm import VLMClient

    # Simulate logits: one token, high-confidence (peaked distribution)
    logits = np.zeros((1, 100), dtype=np.float32)
    logits[0, 42] = 10.0  # One token has high logit
    scores = (logits,)

    conf = VLMClient._compute_confidence(scores)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.9  # Should be very confident

    # Low-confidence: uniform distribution
    uniform = np.zeros((1, 1000), dtype=np.float32)
    scores_low = (uniform,)
    conf_low = VLMClient._compute_confidence(scores_low)
    assert conf_low < 0.01

    # Empty scores
    assert VLMClient._compute_confidence(()) == 0.0


def test_caption_prompt_variants():
    """Both caption prompt variants exist."""
    from satquery.tools import _CAPTION_PROMPTS

    assert "brief" in _CAPTION_PROMPTS
    assert "detailed" in _CAPTION_PROMPTS
    assert "remote-sensing" in _CAPTION_PROMPTS["brief"].lower() or \
           "remote-sensing" in _CAPTION_PROMPTS["detailed"].lower()


# ── Integration tests (require model download) ──────────────────────────


def _make_sample_image(tmp_path: Path) -> tuple[Path, dict]:
    """Create a synthetic 3-band satellite-like PNG for testing."""
    np.random.seed(42)
    h, w = 256, 256
    # Simulate a scene: green vegetation area + blue water + grey urban
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Vegetation (green)
    arr[20:120, 20:120] = [34, 139, 34]
    # Water (blue)
    arr[140:220, 30:180] = [30, 80, 180]
    # Urban (grey)
    arr[50:180, 160:240] = [160, 160, 160]
    # Background (light brown / bare soil)
    mask = np.all(arr == 0, axis=-1)
    arr[mask] = [180, 150, 100]

    img_path = tmp_path / "sample_satellite.png"
    Image.fromarray(arr).save(img_path)

    from satquery.io import load_image
    img_dict = load_image(str(img_path))
    return img_path, img_dict


@pytest.mark.slow
def test_vqa_smoke(tmp_path):
    """Smoke test: VQA produces a non-empty answer with confidence."""
    from satquery.tools import vqa, reset_vlm

    _, img_dict = _make_sample_image(tmp_path)

    result = vqa(
        img_dict,
        question="What types of land cover are visible in this image?",
        max_new_tokens=64,
    )

    assert result.tool == "vqa"
    assert len(result.text) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_s > 0
    assert result.params_used["max_new_tokens"] == 64

    reset_vlm()


@pytest.mark.slow
def test_caption_smoke(tmp_path):
    """Smoke test: caption produces a non-empty description."""
    from satquery.tools import caption, reset_vlm

    _, img_dict = _make_sample_image(tmp_path)

    for detail in ("brief", "detailed"):
        result = caption(img_dict, detail=detail)
        assert result.tool == "caption"
        assert len(result.text) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.params_used["detail"] == detail

    reset_vlm()


@pytest.mark.slow
def test_save_baseline_outputs(tmp_path):
    """Run VQA and caption, save outputs to docs/baseline.json."""
    from satquery.tools import vqa, caption, reset_vlm

    _, img_dict = _make_sample_image(tmp_path)

    baseline = {}

    # VQA
    vqa_q = "Describe the land-cover and major objects visible in this image."
    vqa_result = vqa(img_dict, question=vqa_q, max_new_tokens=128)
    baseline["vqa"] = {
        "question": vqa_q,
        "answer": vqa_result.text,
        "confidence": vqa_result.confidence,
        "latency_s": vqa_result.latency_s,
    }

    # Caption (brief)
    cap_brief = caption(img_dict, detail="brief")
    baseline["caption_brief"] = {
        "text": cap_brief.text,
        "confidence": cap_brief.confidence,
        "latency_s": cap_brief.latency_s,
    }

    # Caption (detailed)
    cap_detailed = caption(img_dict, detail="detailed")
    baseline["caption_detailed"] = {
        "text": cap_detailed.text,
        "confidence": cap_detailed.confidence,
        "latency_s": cap_detailed.latency_s,
    }

    # Save to docs/
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    assert out_path.exists()
    print(f"\nBaseline outputs saved to: {out_path}")
    print(json.dumps(baseline, indent=2))

    reset_vlm()
