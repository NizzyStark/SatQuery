"""Unit tests for Step 8: satquery.report module.

Tests verify:
- HTML report generation from Spec §6.3 compliant execution traces
- Base64 data URI embedding of visual evidence overlays (ensuring standalone portability)
- Metric tables, query/task metadata, latency, and confidence meters in HTML
- Formatted raw JSON trace export
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from satquery.report import _image_to_base64_uri, export_json_trace, generate_html_report


def test_image_to_base64_uri(tmp_path: Path):
    """Test image conversion to base64 Data URI."""
    img_path = tmp_path / "test.png"
    Image.new("RGB", (10, 10), color="blue").save(img_path)

    uri = _image_to_base64_uri(img_path)
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")

    # Non-existent path returns None
    assert _image_to_base64_uri(tmp_path / "non_existent.png") is None
    assert _image_to_base64_uri(None) is None


def test_generate_html_report(tmp_path: Path):
    """Test self-contained HTML report generation with embedded overlay."""
    overlay_path = tmp_path / "overlay.png"
    Image.new("RGB", (32, 32), color="red").save(overlay_path)

    trace = {
        "query": "What changed between these two dates?",
        "inputs": [
            {"name": "t1.png", "modality": "optical", "size": [256, 256]},
            {"name": "t2.png", "modality": "optical", "size": [256, 256]},
        ],
        "task": "change_analyze",
        "validation": {"passed": True, "messages": ["Validation passed"]},
        "steps": [
            {"tool": "change_analyze", "params": {"mode": "describe"}, "latency_s": 1.25}
        ],
        "outputs": {
            "text": "Detected 14.5% change in central agricultural parcel.",
            "overlay": str(overlay_path),
            "confidence": 0.88,
        },
        "stats": {
            "change_pct": 14.5,
            "changed_pixels": 9500,
            "total_pixels": 65536,
        },
        "latency_s": 1.35,
    }

    report_path = tmp_path / "report.html"
    html_content = generate_html_report(trace, output_path=report_path)

    assert report_path.exists()
    assert "<!DOCTYPE html>" in html_content
    assert "What changed between these two dates?" in html_content
    assert "Detected 14.5% change" in html_content
    assert "data:image/png;base64," in html_content  # embedded base64 image
    assert "change_analyze" in html_content
    assert "88.0%" in html_content or "0.88" in html_content


def test_export_json_trace(tmp_path: Path):
    """Test exporting raw execution trace conforming to Spec §6.3."""
    trace = {
        "query": "Describe scene",
        "inputs": [{"name": "scene.png", "modality": "optical", "size": [512, 512]}],
        "task": "caption",
        "validation": {"passed": True, "messages": []},
        "steps": [{"tool": "caption", "params": {"detail": "detailed"}, "latency_s": 0.5}],
        "outputs": {"text": "A coastal city.", "overlay": None, "confidence": 0.92},
        "stats": {"token_count": 5},
    }

    out_file = tmp_path / "trace.json"
    saved_path = export_json_trace(trace, out_file)

    assert Path(saved_path).exists()
    loaded = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert loaded["task"] == "caption"
    assert loaded["outputs"]["confidence"] == 0.92
