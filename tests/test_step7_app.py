"""Unit tests for Step 7: app module (GUI helpers and execution pipeline).

Tests verify:
- Compatibility pre-check for uploaded single images, bi-temporal pairs, and optical-SAR pairs
- Query analysis runner interfacing with SatQueryAgent
- Report generation triggers from the UI execution loop
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from app import analyze_query, check_compatibility


def _create_sample_png(path: Path, color=(100, 150, 200)) -> str:
    arr = np.full((64, 64, 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def test_check_compatibility_empty():
    """Compatibility check handles empty inputs gracefully."""
    msg = check_compatibility(None, None)
    assert "Upload at least one" in msg


def test_check_compatibility_single_image(tmp_path: Path):
    """Compatibility check passes on a single valid image."""
    img1 = _create_sample_png(tmp_path / "img1.png")
    msg = check_compatibility(img1, None, mode="auto")
    assert "PASSED" in msg
    assert "single" in msg.lower()


def test_check_compatibility_bitemporal_pair(tmp_path: Path):
    """Compatibility check handles bi-temporal pair."""
    img1 = _create_sample_png(tmp_path / "before.png")
    img2 = _create_sample_png(tmp_path / "after.png")
    msg = check_compatibility(img1, img2, mode="bitemporal")
    assert "PASSED" in msg
    assert "bitemporal" in msg.lower()


@patch("app.SatQueryAgent")
def test_analyze_query_pipeline(mock_agent_cls, tmp_path: Path):
    """Test analyze_query execution and return signature."""
    img1 = _create_sample_png(tmp_path / "img.png")
    overlay = _create_sample_png(tmp_path / "overlay.png", color=(255, 0, 0))

    mock_agent = MagicMock()
    mock_agent.run.return_value = {
        "task": "caption",
        "text": "A forested area with a river.",
        "overlay": overlay,
        "confidence": 0.91,
        "stats": {"token_count": 8},
        "trace": {
            "query": "Describe scene",
            "task": "caption",
            "latency_s": 0.45,
            "validation": {"passed": True, "messages": []},
            "steps": [],
            "outputs": {"text": "A forested area with a river.", "overlay": overlay, "confidence": 0.91},
            "stats": {"token_count": 8},
        },
    }
    mock_agent_cls.return_value = mock_agent

    text, out_overlay, conf, summary, report_html, trace_json = analyze_query(
        query="Describe scene",
        file1=img1,
        file2=None,
        mode="single",
    )

    assert "forested area" in text
    assert conf == 0.91
    assert out_overlay is not None
    assert report_html is not None
    assert Path(report_html).exists()
    assert trace_json is not None
    assert Path(trace_json).exists()
