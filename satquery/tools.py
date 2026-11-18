"""Tool implementations for SatQuery AI.

Contains tools:
- vqa: Single-image visual question answering using fine-tuned VLM.
- caption: Remote-sensing scene captioning and description.
- change_analyze: Bi-temporal image difference mapping and VLM change analysis (Step 4).
- optsar_analyze: Complementary optical-SAR feature extraction (Step 5).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as ndimage_label
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk

from satquery.io import render_for_vlm, render_overlay, percentile_stretch
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
        latency_s=max(0.01, round(latency, 2)),
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
        latency_s=max(0.01, round(latency, 2)),
    )



# ═══════════════════════════════════════════════════════════════════════════
#  Tool 3 — Change Analyze (Bi-temporal)
# ═══════════════════════════════════════════════════════════════════════════


def _align_pair(
    arr1: np.ndarray, arr2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resize / align two raster arrays to the same (bands, H, W) grid.

    Both arrays are expected in (bands, H, W) format.
    Returns the aligned pair and a list of warning messages.
    """
    warnings: list[str] = []

    # Match band count — take the minimum number of bands
    b1, h1, w1 = arr1.shape
    b2, h2, w2 = arr2.shape
    min_bands = min(b1, b2)
    if b1 != b2:
        warnings.append(
            f"Band count mismatch ({b1} vs {b2}); using first {min_bands} bands."
        )
        arr1 = arr1[:min_bands]
        arr2 = arr2[:min_bands]

    # Resize to smaller spatial extent if sizes differ
    if (h1, w1) != (h2, w2):
        warnings.append(
            f"Spatial size mismatch ({w1}×{h1} vs {w2}×{h2}); "
            f"resizing to ({min(w1, w2)}×{min(h1, h2)})."
        )
        target_h, target_w = min(h1, h2), min(w1, w2)

        def _resize_array(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
            bands = arr.shape[0]
            resized_bands = []
            for b in range(bands):
                pil_band = Image.fromarray(arr[b].astype(np.float32), mode="F")
                pil_resized = pil_band.resize((tw, th), resample=Image.Resampling.BILINEAR)
                resized_bands.append(np.array(pil_resized))
            return np.stack(resized_bands, axis=0)

        if (h1, w1) != (target_h, target_w):
            arr1 = _resize_array(arr1, target_h, target_w)
        if (h2, w2) != (target_h, target_w):
            arr2 = _resize_array(arr2, target_h, target_w)

    return arr1, arr2, warnings


def _compute_change_mask(
    arr1: np.ndarray,
    arr2: np.ndarray,
    diff_threshold: float | None = None,
    min_blob_area: int = 50,
    sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a binary change mask from two aligned arrays.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Aligned arrays of shape (bands, H, W).
    diff_threshold : float | None
        Fixed threshold (0.0–1.0) on the normalised difference.
        If *None*, Otsu's method is used.
    min_blob_area : int
        Blobs smaller than this (in pixels) are removed.
    sigma : float
        Gaussian blur sigma applied before thresholding.

    Returns
    -------
    mask : np.ndarray
        Binary change mask (H, W), dtype bool.
    diff_map : np.ndarray
        Single-channel normalised difference image (H, W), range [0, 1].
    threshold_used : float
        The threshold value that was applied.
    """
    # Convert to float for safe arithmetic
    a1 = arr1.astype(np.float64)
    a2 = arr2.astype(np.float64)

    # Joint per-band normalisation to [0, 1]
    # Both images are normalised using the same min/max per band so that
    # real intensity changes are preserved in the difference.
    a1_norm = np.empty_like(a1)
    a2_norm = np.empty_like(a2)
    for b in range(a1.shape[0]):
        lo = min(float(np.nanmin(a1[b])), float(np.nanmin(a2[b])))
        hi = max(float(np.nanmax(a1[b])), float(np.nanmax(a2[b])))
        if hi - lo > 1e-8:
            a1_norm[b] = (a1[b] - lo) / (hi - lo)
            a2_norm[b] = (a2[b] - lo) / (hi - lo)
        else:
            a1_norm[b] = np.zeros_like(a1[b])
            a2_norm[b] = np.zeros_like(a2[b])

    # Absolute difference, then mean across bands → single channel
    abs_diff = np.abs(a2_norm - a1_norm)
    diff_map = np.mean(abs_diff, axis=0)  # (H, W) in [0, 1]

    # Gaussian blur to suppress noise
    diff_smooth = gaussian_filter(diff_map, sigma=sigma)

    # Threshold
    if diff_threshold is not None:
        thr = float(np.clip(diff_threshold, 0.1, 0.5))
    else:
        # Otsu on the smoothed difference
        # Guard against constant images (no variance → Otsu fails or returns 0)
        diff_range = float(np.max(diff_smooth) - np.min(diff_smooth))
        if diff_range < 1e-8:
            # Constant diff → no change at all
            thr = 1.0  # nothing exceeds this → empty mask
        else:
            try:
                thr = float(threshold_otsu(diff_smooth))
            except ValueError:
                thr = 0.2  # fallback

    mask = diff_smooth >= thr

    # Morphological open then close to clean up
    selem = disk(2)
    mask = opening(mask.astype(np.uint8), selem).astype(bool)
    mask = closing(mask.astype(np.uint8), selem).astype(bool)

    # Remove tiny blobs
    if min_blob_area > 0:
        labelled, num_features = ndimage_label(mask)
        for region_id in range(1, num_features + 1):
            if np.sum(labelled == region_id) < min_blob_area:
                mask[labelled == region_id] = False

    return mask, diff_map, thr


def change_analyze(
    image_dict_before: dict[str, Any],
    image_dict_after: dict[str, Any],
    query: str = "",
    mode: str = "describe",
    diff_threshold: float | None = None,
    vlm: VLMClient | None = None,
    output_dir: str | None = None,
) -> ToolResult:
    """Bi-temporal change analysis: difference mask + VLM description.

    Implements Spec §8.2:
    1. Align the pair to the same grid.
    2. Normalise per band; compute absolute difference → single channel.
    3. Gaussian blur, threshold (Otsu or diff_threshold), morphological
       open/close, remove tiny blobs.
    4. Overlay change mask in red on the later image. Report change_pct.
    5. Prompt VLM with both images for change description / VQA.

    Parameters
    ----------
    image_dict_before : dict
        Image dict (from ``satquery.io.load_image``) for the earlier date.
    image_dict_after : dict
        Image dict for the later date.
    query : str
        User question (used in ``vqa`` mode).
    mode : str
        ``"describe"`` for free-form change description,
        ``"vqa"`` to append the user query.
    diff_threshold : float | None
        Manual threshold (clamped to 0.1–0.5).  None → Otsu.
    vlm : VLMClient | None
        Pre-initialised VLM.  Uses shared singleton if *None*.
    output_dir : str | None
        Directory to save the overlay PNG.  Defaults to a temp directory.

    Returns
    -------
    ToolResult
    """
    t0 = time.perf_counter()

    # ── Validate & clamp params ──────────────────────────────────────────
    mode = mode.strip().lower()
    if mode not in ("describe", "vqa"):
        logger.warning("Unknown change mode '%s'; defaulting to 'describe'.", mode)
        mode = "describe"

    if diff_threshold is not None:
        diff_threshold = float(np.clip(diff_threshold, 0.1, 0.5))

    # ── 1. Align the pair ────────────────────────────────────────────────
    arr_before = image_dict_before["array"]
    arr_after = image_dict_after["array"]
    arr_before, arr_after, align_warnings = _align_pair(arr_before, arr_after)

    for w in align_warnings:
        logger.warning("change_analyze alignment: %s", w)

    # ── 2–3. Compute change mask ─────────────────────────────────────────
    change_mask, diff_map, threshold_used = _compute_change_mask(
        arr_before, arr_after, diff_threshold=diff_threshold
    )

    total_pixels = change_mask.size
    changed_pixels = int(np.sum(change_mask))
    change_pct = round(100.0 * changed_pixels / total_pixels, 2) if total_pixels > 0 else 0.0

    # ── 4. Overlay change mask on the later image ────────────────────────
    overlay_img = render_overlay(image_dict_after, change_mask, color=(255, 0, 0), alpha=0.45)

    # Save overlay
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="satquery_change_")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"change_overlay_{uuid.uuid4().hex[:8]}.png"
    overlay_img.save(str(overlay_path))

    # ── 5. VLM narration ─────────────────────────────────────────────────
    vlm_client = vlm or get_vlm()

    pil_before = render_for_vlm(image_dict_before)
    pil_after = render_for_vlm(image_dict_after)

    if mode == "describe":
        vlm_prompt = (
            "You are an expert remote-sensing change analyst. "
            "Image 1 is the earlier image and Image 2 is the later image. "
            f"Approximately {change_pct}% of the area shows change. "
            "Describe what changed and where the changes occurred. "
            "Be specific about locations (upper-left, centre, etc.) and "
            "types of change (new construction, deforestation, flooding, etc.)."
        )
    else:  # vqa mode
        vlm_prompt = (
            "You are an expert remote-sensing change analyst. "
            "Image 1 is the earlier image and Image 2 is the later image. "
            f"Approximately {change_pct}% of the area shows change. "
            f"Answer the following question about the changes:\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    vlm_result = vlm_client.generate(
        images=[pil_before, pil_after],
        prompt=vlm_prompt,
        max_new_tokens=128,
    )

    # ── Confidence ───────────────────────────────────────────────────────
    # Base confidence from VLM token probability
    confidence = vlm_result["confidence"]
    # Boost slightly if the change_pct is in a reasonable range (not 0% or 100%)
    if 1.0 < change_pct < 95.0:
        confidence = min(1.0, confidence + 0.05)
    # Lower if alignment had warnings (quality degradation)
    if align_warnings:
        confidence = max(0.0, confidence - 0.1)

    latency = time.perf_counter() - t0

    return ToolResult(
        tool="change_analyze",
        text=vlm_result["text"],
        overlay_png=str(overlay_path),
        stats={
            "change_pct": change_pct,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "threshold_used": round(threshold_used, 4),
            "alignment_warnings": align_warnings,
            "token_count": vlm_result["token_count"],
        },
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        params_used={
            "mode": mode,
            "diff_threshold": diff_threshold,
            "threshold_actual": round(threshold_used, 4),
        },
        latency_s=max(0.01, round(latency, 2)),
    )
