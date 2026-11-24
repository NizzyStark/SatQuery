"""SatQuery AI — Interactive Web Application (Gradio).

Implements Step 7:
- Upload 1-2 remote-sensing files (GeoTIFF, TIFF, PNG, JPEG).
- Mode dropdown: Auto, Single, Optical-SAR, Bi-temporal.
- Per-file tags for modality and date.
- Real-time compatibility and alignment pre-check on file upload.
- Example query buttons for one-click demonstration.
- Visual evidence overlay viewer with side-by-side or layered rendering.
- Confidence score meter and execution trace audit summary.
- One-click download of self-contained HTML analysis reports and JSON traces.
- Supports --share for cloud hosting on Kaggle/Colab.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from satquery.agent import SatQueryAgent
from satquery.io import load_image, render_for_vlm
from satquery.report import export_json_trace, generate_html_report
from satquery.validator import validate_inputs

logger = logging.getLogger(__name__)

# Preset demonstration queries
EXAMPLE_QUERIES = [
    "Describe the land-cover and major objects visible in this image.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Is there water or standing floodwater present in this scene?",
    "Provide a detailed description of terrain features and vegetation distribution.",
]


def check_compatibility(
    file1: str | None,
    file2: str | None,
    mode: str = "auto",
    modality1: str = "auto",
    modality2: str = "auto",
) -> str:
    """Perform instant compatibility pre-check on uploaded imagery."""
    if not file1 and not file2:
        return "Upload at least one satellite/aerial image to perform a compatibility check."

    files = [f for f in (file1, file2) if f]
    mod_tags = [modality1 if modality1 != "auto" else None, modality2 if modality2 != "auto" else None]

    try:
        images: list[dict[str, Any]] = []
        for i, f_path in enumerate(files):
            tag = mod_tags[i] if i < len(mod_tags) else None
            img_dict = load_image(f_path, modality=tag)
            images.append(img_dict)

        val_result = validate_inputs(images, mode=mode.lower().replace("-", "_"))

        status = "PASSED" if val_result["passed"] else "FAILED"
        status_color = "#16a34a" if val_result["passed"] else "#dc2626"
        inferred_mode = val_result.get("inferred_mode", mode)

        lines = [
            f"<div style='border-left: 4px solid {status_color}; padding: 0.5rem 1rem; background: #f8fafc; border-radius: 4px;'>",
            f"<h4 style='margin: 0; color: {status_color};'>Compatibility Pre-Check: {status} (Mode: {inferred_mode})</h4>",
            "<ul style='margin: 0.5rem 0 0 0; padding-left: 1.25rem; font-size: 0.875rem;'>",
        ]

        for img in images:
            p_name = Path(img.get("path", "file")).name
            mod = img.get("modality", "unknown")
            size = img.get("meta", {}).get("size", "N/A")
            bands = img.get("meta", {}).get("bands", "N/A")
            lines.append(f"<li><strong>{p_name}</strong>: Modality <code>{mod}</code>, {bands} bands, size {size}</li>")

        for msg in val_result.get("messages", []):
            lines.append(f"<li>{msg}</li>")

        lines.append("</ul></div>")
        return "".join(lines)

    except Exception as e:
        return f"<div style='color: #dc2626;'>Error reading file: {str(e)}</div>"


def analyze_query(
    query: str,
    file1: str | None,
    file2: str | None,
    mode: str = "auto",
    modality1: str = "auto",
    modality2: str = "auto",
    date1: str = "",
    date2: str = "",
) -> tuple[str, Image.Image | None, float, str, str | None, str | None]:
    """Execute SatQuery analysis pipeline via SatQueryAgent."""
    if not query.strip():
        return (
            "Please enter a natural-language query.",
            None,
            0.0,
            "No query provided.",
            None,
            None,
        )

    files = [f for f in (file1, file2) if f]
    if not files:
        return (
            "Please upload at least one image file.",
            None,
            0.0,
            "No images uploaded.",
            None,
            None,
        )

    out_dir = Path(tempfile.mkdtemp(prefix="satquery_ui_"))
    agent = SatQueryAgent(output_dir=out_dir)

    # Load images with tags
    mod_tags = [modality1 if modality1 != "auto" else None, modality2 if modality2 != "auto" else None]
    image_dicts: list[dict[str, Any]] = []
    for i, f_path in enumerate(files):
        tag = mod_tags[i] if i < len(mod_tags) else None
        img_dict = load_image(f_path, modality=tag)
        image_dicts.append(img_dict)

    clean_mode = mode.lower().replace("-", "_")

    # Run agent
    result = agent.run(
        query=query,
        images=image_dicts,
        mode=clean_mode,
        output_dir=out_dir,
    )

    answer_text = result.get("text", "")
    confidence = float(result.get("confidence", 0.0))
    overlay_path = result.get("overlay")

    overlay_img: Image.Image | None = None
    if overlay_path and Path(overlay_path).exists():
        try:
            overlay_img = Image.open(overlay_path)
        except Exception:
            overlay_img = None

    trace = result.get("trace", {})
    summary_lines = [
        f"**Selected Task:** `{trace.get('task', 'N/A')}`",
        f"**Latency:** `{trace.get('latency_s', 0.0)}s`",
        f"**Validation:** `{'Passed' if trace.get('validation', {}).get('passed') else 'Failed'}`",
    ]
    stats = result.get("stats", {})
    if stats:
        summary_lines.append("**Extracted Metrics:**")
        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)):
                summary_lines.append(f"- {k}: `{v}`")

    summary_text = "\n\n".join(summary_lines)

    # Export report & trace files for download
    report_html_path = out_dir / "satquery_report.html"
    generate_html_report(trace, report_html_path)

    trace_json_path = out_dir / "trace.json"
    export_json_trace(trace, trace_json_path)

    return (
        answer_text,
        overlay_img,
        confidence,
        summary_text,
        str(report_html_path),
        str(trace_json_path),
    )


def build_app():
    """Construct Gradio Interface."""
    import gradio as gr

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="SatQuery AI — Remote Sensing Assistant") as demo:
        gr.Markdown(
            """
            # 🛰️ SatQuery AI — Agentic Remote Sensing Assistant
            **Analyze Optical, Multispectral, and Synthetic Aperture Radar (SAR) Earth Observation Imagery**
            """
        )

        with gr.Row():
            # Left Column: Inputs & Compatibility
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Imagery")
                with gr.Row():
                    file1 = gr.File(label="Image 1 (Base / Earlier / Optical)", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
                    file2 = gr.File(label="Image 2 (Later / SAR)", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])

                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        label="Operational Mode",
                        choices=["auto", "single", "optical_sar", "bitemporal"],
                        value="auto",
                    )

                with gr.Accordion("Optional Per-File Tags & Metadata", open=False):
                    with gr.Row():
                        modality1 = gr.Dropdown(label="Image 1 Modality", choices=["auto", "optical", "sar"], value="auto")
                        date1 = gr.Textbox(label="Date 1 (YYYY-MM-DD)", placeholder="e.g. 2023-01-15")
                    with gr.Row():
                        modality2 = gr.Dropdown(label="Image 2 Modality", choices=["auto", "optical", "sar"], value="auto")
                        date2 = gr.Textbox(label="Date 2 (YYYY-MM-DD)", placeholder="e.g. 2023-08-20")

                compat_box = gr.HTML(
                    value="<div style='color: #64748b; font-size: 0.875rem;'>Upload images to preview format compatibility.</div>",
                    label="Compatibility Status",
                )

                # Event listener for live compatibility check
                for trigger in [file1.change, file2.change, mode_dropdown.change, modality1.change, modality2.change]:
                    trigger(
                        fn=check_compatibility,
                        inputs=[file1, file2, mode_dropdown, modality1, modality2],
                        outputs=[compat_box],
                    )

            # Right Column: Query & Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 2. Natural Language Query")
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter a remote-sensing question or click an example below...",
                    lines=2,
                )

                gr.Markdown("**Example Prompts:**")
                with gr.Row():
                    btn_desc = gr.Button("Describe Scene", size="sm")
                    btn_chg = gr.Button("Change Analysis", size="sm")
                    btn_sar = gr.Button("Optical-SAR Fusion", size="sm")
                    btn_water = gr.Button("Water Detection", size="sm")

                run_btn = gr.Button("🚀 Run SatQuery AI", variant="primary", size="lg")

                gr.Markdown("### 3. Analysis Results")
                answer_output = gr.Markdown(label="Analysis Output")
                conf_slider = gr.Slider(label="Confidence Score", minimum=0.0, maximum=1.0, value=0.0, interactive=False)
                overlay_output = gr.Image(label="Visual Evidence Overlay", type="pil")

                with gr.Accordion("Execution Summary & Trace", open=False):
                    summary_output = gr.Markdown()

                with gr.Row():
                    download_html = gr.File(label="Download HTML Report")
                    download_json = gr.File(label="Download JSON Trace")

        # Preset query button handlers
        btn_desc.click(fn=lambda: EXAMPLE_QUERIES[0], outputs=[query_input])
        btn_chg.click(fn=lambda: EXAMPLE_QUERIES[1], outputs=[query_input])
        btn_sar.click(fn=lambda: EXAMPLE_QUERIES[2], outputs=[query_input])
        btn_water.click(fn=lambda: EXAMPLE_QUERIES[4], outputs=[query_input])

        # Execution handler
        run_btn.click(
            fn=analyze_query,
            inputs=[query_input, file1, file2, mode_dropdown, modality1, modality2, date1, date2],
            outputs=[
                answer_output,
                overlay_output,
                conf_slider,
                summary_output,
                download_html,
                download_json,
            ],
        )

    return demo


def main() -> None:
    """Launch Gradio application."""
    parser = argparse.ArgumentParser(description="SatQuery AI Gradio Interface")
    parser.add_argument("--share", action="store_true", help="Generate public URL (Kaggle/Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name or IP")
    args = parser.parse_args()

    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    share_url = args.share or is_kaggle

    print("=" * 60)
    print("SatQuery AI — Gradio Web Interface")
    print(f"  Mode: {'Cloud / Kaggle Public Link' if share_url else 'Local'}")
    print(f"  Host: {args.server_name}:{args.port}")
    print("=" * 60)

    demo = build_app()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=share_url,
    )


if __name__ == "__main__":
    main()
