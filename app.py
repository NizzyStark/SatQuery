"""SatQuery AI — Web Application (Gradio).

Provides an interactive user interface for uploading satellite imagery (GeoTIFF, PNG, JPEG),
performing automatic compatibility checks, executing queries via SatQueryAgent,
displaying visual evidence overlays and confidence scores, and downloading analysis reports.
"""


import argparse
import os


def main() -> None:
    """Launch Gradio application.

    Supports local execution and Kaggle cloud execution with public URL sharing.
    Placeholder implementation to be completed in Step 7.
    """
    parser = argparse.ArgumentParser(description="SatQuery AI Gradio Interface")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL (useful on Kaggle/Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Local port to bind to")
    args = parser.parse_args()

    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    share_url = args.share or is_kaggle

    print(f"SatQuery AI Gradio Web App (Step 7 placeholder)")
    print(f"  Execution mode: {'Cloud / Kaggle (share=True)' if share_url else 'Local'}")
    print(f"  Port: {args.port}")


if __name__ == "__main__":
    main()
