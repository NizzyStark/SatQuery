"""Report export module for SatQuery AI.

Produces self-contained HTML reports and structured JSON traces containing:
- Query and target task
- Input imagery metadata
- Answer text and confidence metrics
- Embedded base64 visual evidence overlays
- Quantitative geospatial stats (change, water, built-up percentages)
- Complete execution trace steps and latencies
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


def _image_to_base64_uri(image_path: str | Path | None) -> str | None:
    """Convert an image file to a base64 Data URI for standalone HTML embedding."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return None


def generate_html_report(trace: dict[str, Any], output_path: str | Path | None = None) -> str:
    """Generate a self-contained, high-polish HTML report from an execution trace.

    Parameters
    ----------
    trace : dict
        Execution trace adhering to Spec §6.3.
    output_path : str | Path | None
        Optional file path to write the HTML report to.

    Returns
    -------
    html_content : str
    """
    query = html.escape(str(trace.get("query", "N/A")))
    task = html.escape(str(trace.get("task", "unknown")))
    latency = trace.get("latency_s", 0.0)

    outputs = trace.get("outputs", {})
    text = html.escape(str(outputs.get("text", "")))
    confidence = float(outputs.get("confidence", 0.0))
    conf_pct = round(confidence * 100.0, 1)

    overlay_path = outputs.get("overlay")
    overlay_uri = _image_to_base64_uri(overlay_path)

    stats = trace.get("stats", {})
    inputs = trace.get("inputs", [])
    validation = trace.get("validation", {})
    val_passed = validation.get("passed", True)
    val_msgs = validation.get("messages", [])

    steps = trace.get("steps", [])

    # Format stats table rows
    stats_rows = ""
    for k, v in stats.items():
        if isinstance(v, (str, int, float, bool)):
            stats_rows += f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"

    # Format inputs list
    inputs_html = ""
    for inp in inputs:
        name = html.escape(str(inp.get("name", "Unknown")))
        mod = html.escape(str(inp.get("modality", "unknown")))
        size = inp.get("size", "N/A")
        inputs_html += f"<li><strong>{name}</strong> — Modality: <code>{mod}</code> | Size: <code>{size}</code></li>"

    # Format validation messages
    val_badge = '<span class="badge badge-success">Passed</span>' if val_passed else '<span class="badge badge-danger">Failed</span>'
    val_msgs_html = "".join(f"<li>{html.escape(str(m))}</li>" for m in val_msgs) or "<li>None</li>"

    # Format overlay preview
    overlay_html = ""
    if overlay_uri:
        overlay_html = f"""
        <div class="card">
            <h3>Visual Evidence Overlay</h3>
            <div class="img-container">
                <img src="{overlay_uri}" alt="Visual Evidence Overlay" />
            </div>
        </div>
        """

    # Format execution steps
    steps_rows = ""
    for s in steps:
        tool_name = html.escape(str(s.get("tool", "")))
        step_lat = s.get("latency_s", 0.0)
        params_str = html.escape(json.dumps(s.get("params", {})))
        steps_rows += f"<tr><td><code>{tool_name}</code></td><td><code>{params_str}</code></td><td>{step_lat}s</td></tr>"

    trace_json_pretty = html.escape(json.dumps(trace, indent=2))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SatQuery AI Analysis Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1f2937;
        background: #f9fafb;
        margin: 0;
        padding: 2rem;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}
    header {{
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }}
    h1 {{
        color: #111827;
        margin: 0 0 0.5rem 0;
        font-size: 1.875rem;
    }}
    .meta-bar {{
        display: flex;
        gap: 1.5rem;
        font-size: 0.875rem;
        color: #4b5563;
    }}
    .card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }}
    .card h3 {{
        margin-top: 0;
        color: #0f172a;
        font-size: 1.1rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
    }}
    .badge-success {{ background: #dcfce7; color: #166534; }}
    .badge-danger {{ background: #fee2e2; color: #991b1b; }}
    .progress-bar {{
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .progress-fill {{
        height: 100%;
        background: #2563eb;
        width: {conf_pct}%;
    }}
    .answer-box {{
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 1.05rem;
        color: #1e3a8a;
        margin: 1rem 0;
    }}
    .img-container {{
        text-align: center;
        background: #0f172a;
        border-radius: 8px;
        padding: 1rem;
    }}
    .img-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }}
    th, td {{
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.875rem;
    }}
    th {{ color: #475569; width: 40%; font-weight: 600; }}
    pre {{
        background: #0f172a;
        color: #f8fafc;
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 0.8rem;
    }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>SatQuery AI — Remote Sensing Analysis Report</h1>
        <div class="meta-bar">
            <span><strong>Task:</strong> <code>{task}</code></span>
            <span><strong>Validation:</strong> {val_badge}</span>
            <span><strong>Total Latency:</strong> {latency}s</span>
        </div>
    </header>

    <div class="card">
        <h3>User Query</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">"{query}"</p>
    </div>

    <div class="card">
        <h3>Analysis Output</h3>
        <div class="answer-box">
            {text}
        </div>
        <div>
            <strong>Confidence Score:</strong> {conf_pct}% ({confidence:.4f})
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
    </div>

    {overlay_html}

    <div class="card">
        <h3>Input Imagery &amp; Modality</h3>
        <ul>{inputs_html}</ul>
        <h4 style="margin: 1rem 0 0.25rem 0; font-size: 0.95rem;">Validation Log:</h4>
        <ul style="font-size: 0.875rem; color: #4b5563;">{val_msgs_html}</ul>
    </div>

    {"<div class='card'><h3>Extracted Geospatial Metrics</h3><table>" + stats_rows + "</table></div>" if stats_rows else ""}

    {"<div class='card'><h3>Execution Steps</h3><table><tr><th>Tool</th><th>Parameters</th><th>Latency</th></tr>" + steps_rows + "</table></div>" if steps_rows else ""}

    <details class="card">
        <summary style="cursor: pointer; font-weight: 600;">View Raw Execution Trace (JSON)</summary>
        <pre>{trace_json_pretty}</pre>
    </details>
</div>
</body>
</html>
"""

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    return html_content


def export_json_trace(trace: dict[str, Any], output_path: str | Path) -> str:
    """Export raw JSON trace conforming to Spec §6.3."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return str(p)
