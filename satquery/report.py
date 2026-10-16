"""Report export module for SatQuery AI.

Generates self-contained HTML reports and structured JSON traces containing
input metadata, query, tool reasoning steps, visual evidence overlays, and confidence metrics.
"""

from typing import Any


def generate_html_report(trace: dict[str, Any], output_path: str) -> str:
    """Generate an HTML report document from an execution trace.

    Placeholder implementation to be completed in Step 8.
    """
    raise NotImplementedError("Report generator will be implemented in Step 8.")


def export_json_trace(trace: dict[str, Any], output_path: str) -> None:
    """Export execution trace as a JSON file.

    Placeholder implementation to be completed in Step 8.
    """
    raise NotImplementedError("Trace export will be implemented in Step 8.")
