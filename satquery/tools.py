"""Tool implementations for SatQuery AI.

Contains tools:
- vqa: Single-image visual question answering using fine-tuned VLM.
- caption: Remote-sensing scene captioning and description.
- change_analyze: Bi-temporal image difference mapping and VLM change analysis (Step 4).
- optsar_analyze: Complementary optical-SAR feature extraction (Step 5).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from satquery.io import render_for_vlm
from satquery.vlm import VLMClient

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized result returned by all SatQuery tools."""

    tool: str
    text: str
    overlay_png: str | None  # path to evidence image or None
    stats: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0.0 to 1.0
    params_used: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


# ── Shared VLM singleton ────────────────────────────────────────────────────

_vlm_instance: VLMClient | None = None


def get_vlm(
    adapter_path: str | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> VLMClient:
    """Return (and lazily create) the shared VLM instance.

    The singleton is reused across tool calls to avoid reloading the heavy
    model. Call ``reset_vlm()`` to force a fresh load.
    """
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = VLMClient(
            model_name=model_name or "Qwen/Qwen2.5-VL-3B-Instruct",
            adapter_path=adapter_path,
            **kwargs,
        )
    return _vlm_instance


def reset_vlm() -> None:
    """Unload the shared VLM and clear the singleton."""
    global _vlm_instance
    if _vlm_instance is not None:
        _vlm_instance.unload()
        _vlm_instance = None


# ═══════════════════════════════════════════════════════════════════════════
#  Tool 1 — VQA
# ═══════════════════════════════════════════════════════════════════════════


def vqa(
    image_dict: dict[str, Any],
    question: str,
    max_new_tokens: int = 128,
    vlm: VLMClient | None = None,
) -> ToolResult:
    """Visual Question Answering on a single remote-sensing image.

    Parameters
    ----------
    image_dict : dict
        An image dict returned by ``satquery.io.load_image``.
    question : str
        Natural-language question about the image.
    max_new_tokens : int
        Maximum tokens for the answer (clamped to 16-128).
    vlm : VLMClient | None
        Optional pre-initialised VLM. Uses shared singleton if *None*.

    Returns
    -------
    ToolResult
    """
    t0 = time.perf_counter()

    max_new_tokens = max(16, min(128, int(max_new_tokens)))
    vlm_client = vlm or get_vlm()

    # Render image for the VLM
    pil_image = render_for_vlm(image_dict)

    # Build prompt — frame it for remote-sensing context
    prompt = (
        "You are an expert remote-sensing analyst. "
        "Answer the following question about the satellite/aerial image.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    result = vlm_client.generate(
        images=[pil_image],
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    latency = time.perf_counter() - t0

    return ToolResult(
        tool="vqa",
        text=result["text"],
        overlay_png=None,
        stats={"token_count": result["token_count"]},
        confidence=result["confidence"],
        params_used={"max_new_tokens": max_new_tokens},
        latency_s=round(latency, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Tool 2 — Caption
# ═══════════════════════════════════════════════════════════════════════════


_CAPTION_PROMPTS = {
    "brief": (
        "You are an expert remote-sensing analyst. "
        "Provide a brief one-sentence caption describing the main land-cover "
        "types and notable features visible in this satellite/aerial image."
    ),
    "detailed": (
        "You are an expert remote-sensing analyst. "
        "Provide a detailed description of this satellite/aerial image. "
        "Describe the land-cover types, major objects, spatial layout, "
        "any notable features such as water bodies, vegetation, built-up areas, "
        "roads, and agricultural fields. Be specific about locations (e.g. "
        "upper-left, centre, along the bottom edge)."
    ),
}


def caption(
    image_dict: dict[str, Any],
    detail: str = "detailed",
    vlm: VLMClient | None = None,
) -> ToolResult:
    """Generate a caption / scene description for a remote-sensing image.

    Parameters
    ----------
    image_dict : dict
        An image dict returned by ``satquery.io.load_image``.
    detail : str
        ``"brief"`` for a one-liner or ``"detailed"`` for a full description.
    vlm : VLMClient | None
        Optional pre-initialised VLM. Uses shared singleton if *None*.

    Returns
    -------
    ToolResult
    """
    t0 = time.perf_counter()

    detail = detail.strip().lower()
    if detail not in _CAPTION_PROMPTS:
        logger.warning("Unknown detail level '%s'; defaulting to 'detailed'.", detail)
        detail = "detailed"

    vlm_client = vlm or get_vlm()
    pil_image = render_for_vlm(image_dict)
    prompt = _CAPTION_PROMPTS[detail]

    # Use more tokens for detailed captions
    max_tokens = 64 if detail == "brief" else 128

    result = vlm_client.generate(
        images=[pil_image],
        prompt=prompt,
        max_new_tokens=max_tokens,
    )

    latency = time.perf_counter() - t0

    return ToolResult(
        tool="caption",
        text=result["text"],
        overlay_png=None,
        stats={"token_count": result["token_count"]},
        confidence=result["confidence"],
        params_used={"detail": detail},
        latency_s=round(latency, 2),
    )
