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
from scipy.ndimage import gaussian_filter, uniform_filter, label as ndimage_label
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



# ═══════════════════════════════════════════════════════════════════════════
#  Tool 4 — Optical–SAR Analysis (Multi-sensor Fusion)
# ═══════════════════════════════════════════════════════════════════════════


def _extract_optical_features(
    arr_opt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
    """Extract water and built-up masks from optical raster data.

    Parameters
    ----------
    arr_opt : np.ndarray
        Optical array of shape (bands, H, W).

    Returns
    -------
    water_mask : np.ndarray (H, W) bool
    buildup_mask : np.ndarray (H, W) bool
    fallbacks : dict[str, bool]
        Flags indicating whether fallbacks were used.
    """
    bands, h, w = arr_opt.shape
    arr = arr_opt.astype(np.float64)
    fallbacks = {"optical_water_fallback": False, "optical_buildup_fallback": False}

    if bands >= 4:
        # Multispectral (e.g. Sentinel-2: B2=0, B3=1 Green, B4=2 Red, B8=3 NIR)
        green = arr[1]
        red = arr[2]
        nir = arr[3]

        # NDWI = (Green - NIR) / (Green + NIR)
        denom_water = green + nir + 1e-8
        ndwi = (green - nir) / denom_water
        water_mask = ndwi > 0.0

        if bands >= 5:
            # SWIR exists (band 4) -> NDBI = (SWIR - NIR) / (SWIR + NIR)
            swir = arr[4]
            ndbi = (swir - nir) / (swir + nir + 1e-8)
            buildup_mask = (ndbi > 0.0) & (~water_mask)
        else:
            # 4-band fallback: low NDVI + high visible brightness
            fallbacks["optical_buildup_fallback"] = True
            ndvi = (nir - red) / (nir + red + 1e-8)
            vis_brightness = (arr[0] + green + red) / 3.0
            p70 = float(np.percentile(vis_brightness, 70))
            buildup_mask = (vis_brightness > p70) & (ndvi < 0.2) & (~water_mask)
    elif bands == 3:
        # RGB colour-based fallback
        fallbacks["optical_water_fallback"] = True
        fallbacks["optical_buildup_fallback"] = True

        r, g, b = arr[0], arr[1], arr[2]
        total = r + g + b + 1e-8
        blue_ratio = b / total

        # Water: dominant blue reflectance and blue > red
        water_mask = (blue_ratio > 0.38) & (b > r * 1.1)

        # Built-up: high overall brightness + moderate/low saturation
        brightness = total / 3.0
        p70 = float(np.percentile(brightness, 70))
        buildup_mask = (brightness > p70) & (~water_mask)
    else:
        # Single-band grayscale fallback
        fallbacks["optical_water_fallback"] = True
        fallbacks["optical_buildup_fallback"] = True

        gray = arr[0]
        lo = float(np.percentile(gray, 20))
        hi = float(np.percentile(gray, 80))
        water_mask = gray < lo
        buildup_mask = gray > hi

    # Morphological cleanup
    selem = disk(1)
    water_mask = opening(water_mask.astype(np.uint8), selem).astype(bool)
    buildup_mask = opening(buildup_mask.astype(np.uint8), selem).astype(bool)

    return water_mask, buildup_mask, fallbacks


def _extract_sar_features(
    arr_sar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract water and built-up masks from SAR raster data.

    Parameters
    ----------
    arr_sar : np.ndarray
        SAR array of shape (bands, H, W). Band 0 is VV backscatter.

    Returns
    -------
    water_mask : np.ndarray (H, W) bool
    buildup_mask : np.ndarray (H, W) bool
    """
    vv = arr_sar[0].astype(np.float64)

    # Water has specular reflection → very low backscatter
    vv_range = float(np.nanmax(vv) - np.nanmin(vv))
    if vv_range > 1e-5:
        try:
            # Otsu on the lower mode of backscatter separates specular water from terrain/urban
            med = float(np.median(vv))
            lower_mode = vv[vv < med]
            if len(lower_mode) > 10 and float(np.nanmax(lower_mode) - np.nanmin(lower_mode)) > 1e-5:
                thr_otsu = float(threshold_otsu(lower_mode))
            else:
                thr_otsu = float(threshold_otsu(vv))
            water_mask = vv < thr_otsu
        except ValueError:
            water_mask = vv < float(np.percentile(vv, 20))
    else:
        water_mask = np.zeros_like(vv, dtype=bool)

    # Built-up has double-bounce corner reflection → high backscatter + high local texture
    mean_vv = uniform_filter(vv, size=5)
    sqr_mean = uniform_filter(vv**2, size=5)
    var_vv = np.maximum(0.0, sqr_mean - mean_vv**2)
    std_vv = np.sqrt(var_vv)

    hi_backscatter = vv > float(np.percentile(vv, 75))
    hi_texture = std_vv > float(np.percentile(std_vv, 70))
    buildup_mask = hi_backscatter & hi_texture & (~water_mask)

    # Morphological cleanup
    selem = disk(1)
    water_mask = opening(water_mask.astype(np.uint8), selem).astype(bool)
    buildup_mask = opening(buildup_mask.astype(np.uint8), selem).astype(bool)

    return water_mask, buildup_mask


def _fuse_optsar(
    opt_water: np.ndarray,
    sar_water: np.ndarray,
    opt_buildup: np.ndarray,
    sar_buildup: np.ndarray,
    targets: str = "both",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse optical and SAR detections and compute agreement statistics.

    Parameters
    ----------
    opt_water, sar_water : np.ndarray (H, W) bool
    opt_buildup, sar_buildup : np.ndarray (H, W) bool
    targets : str
        "built_up", "water", or "both".

    Returns
    -------
    final_water : np.ndarray (H, W) bool
    final_buildup : np.ndarray (H, W) bool
    stats : dict[str, Any]
    """
    total_pixels = opt_water.size

    # Water fusion
    water_both = opt_water & sar_water
    water_any = opt_water | sar_water
    water_agreement = (
        float(np.sum(water_both) / np.sum(water_any)) if np.sum(water_any) > 0 else 1.0
    )

    # Built-up fusion (excluding water)
    buildup_both = (opt_buildup & sar_buildup) & (~water_any)
    buildup_any = (opt_buildup | sar_buildup) & (~water_any)
    buildup_agreement = (
        float(np.sum(buildup_both) / np.sum(buildup_any)) if np.sum(buildup_any) > 0 else 1.0
    )

    final_water = water_any if targets in ("water", "both") else np.zeros_like(opt_water, dtype=bool)
    final_buildup = buildup_any if targets in ("built_up", "both") else np.zeros_like(opt_buildup, dtype=bool)

    water_pct = round(100.0 * float(np.sum(water_any)) / total_pixels, 2)
    water_both_pct = round(100.0 * float(np.sum(water_both)) / total_pixels, 2)
    buildup_pct = round(100.0 * float(np.sum(buildup_any)) / total_pixels, 2)
    buildup_both_pct = round(100.0 * float(np.sum(buildup_both)) / total_pixels, 2)

    stats = {
        "water_pct": water_pct,
        "water_both_pct": water_both_pct,
        "water_agreement_ratio": round(water_agreement, 4),
        "built_up_pct": buildup_pct,
        "built_up_both_pct": buildup_both_pct,
        "built_up_agreement_ratio": round(buildup_agreement, 4),
        "total_pixels": total_pixels,
    }

    return final_water, final_buildup, stats


def optsar_analyze(
    image_dict_optical: dict[str, Any],
    image_dict_sar: dict[str, Any],
    targets: str = "both",
    query: str = "",
    vlm: VLMClient | None = None,
    output_dir: str | None = None,
) -> ToolResult:
    """Complementary optical–SAR analysis and feature extraction.

    Implements Spec §8.3:
    1. Optical water: NDWI when NIR exists, else colour-based fallback.
    2. SAR water: low backscatter (Otsu threshold on dB VV).
    3. SAR built-up: high backscatter combined with local texture.
    4. Optical built-up: NDBI if SWIR exists, else brightness/texture fallback.
    5. Fusion: both agree (high confidence), one only (medium confidence).
    6. Overlay: blue = water, orange = built-up. Report area percentages.
    7. VLM narration: summary of what each sensor contributes.

    Parameters
    ----------
    image_dict_optical : dict
        Image dict from ``satquery.io.load_image`` (optical/multispectral).
    image_dict_sar : dict
        Image dict from ``satquery.io.load_image`` (SAR).
    targets : str
        Target classes to detect: ``"built_up"``, ``"water"``, or ``"both"``.
    query : str
        Optional user query to include in the narration prompt.
    vlm : VLMClient | None
        Pre-initialised VLM singleton.
    output_dir : str | None
        Output directory for the overlay PNG.

    Returns
    -------
    ToolResult
    """
    t0 = time.perf_counter()

    # ── Validate & clamp parameters ──────────────────────────────────────
    targets = targets.strip().lower()
    valid_targets = ("built_up", "water", "both")
    if targets not in valid_targets:
        logger.warning(
            "Unknown target '%s' for optsar_analyze; defaulting to 'both'.", targets
        )
        targets = "both"

    # ── 1. Spatial alignment ─────────────────────────────────────────────
    arr_opt = image_dict_optical["array"]
    arr_sar = image_dict_sar["array"]

    h_opt, w_opt = arr_opt.shape[1], arr_opt.shape[2]
    h_sar, w_sar = arr_sar.shape[1], arr_sar.shape[2]
    align_warnings: list[str] = []

    if (h_opt, w_opt) != (h_sar, w_sar):
        target_h, target_w = min(h_opt, h_sar), min(w_opt, w_sar)
        align_warnings.append(
            f"Optical-SAR size mismatch ({w_opt}×{h_opt} vs {w_sar}×{h_sar}); "
            f"aligning to ({target_w}×{target_h})."
        )

        def _resize_arr(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
            res_bands = []
            for b in range(arr.shape[0]):
                p = Image.fromarray(arr[b].astype(np.float32), mode="F")
                p_res = p.resize((tw, th), resample=Image.Resampling.BILINEAR)
                res_bands.append(np.array(p_res))
            return np.stack(res_bands, axis=0)

        if (h_opt, w_opt) != (target_h, target_w):
            arr_opt = _resize_arr(arr_opt, target_h, target_w)
        if (h_sar, w_sar) != (target_h, target_w):
            arr_sar = _resize_arr(arr_sar, target_h, target_w)

    # ── 2–4. Feature extraction ──────────────────────────────────────────
    opt_water, opt_buildup, fallbacks = _extract_optical_features(arr_opt)
    sar_water, sar_buildup = _extract_sar_features(arr_sar)

    # ── 5. Sensor fusion ─────────────────────────────────────────────────
    final_water, final_buildup, fusion_stats = _fuse_optsar(
        opt_water, sar_water, opt_buildup, sar_buildup, targets=targets
    )

    # ── 6. Evidence overlay (Blue = water, Orange = built-up) ────────────
    # Base image is the rendered optical image
    base_pil = render_for_vlm(image_dict_optical)
    # Ensure base image matches aligned dimensions
    if base_pil.size != (final_water.shape[1], final_water.shape[0]):
        base_pil = base_pil.resize(
            (final_water.shape[1], final_water.shape[0]),
            resample=Image.Resampling.BILINEAR,
        )

    overlay_img = base_pil
    if targets in ("water", "both") and np.any(final_water):
        # Blue overlay for water: (30, 144, 255)
        overlay_img = render_overlay(
            overlay_img, final_water, color=(30, 144, 255), alpha=0.45
        )
    if targets in ("built_up", "both") and np.any(final_buildup):
        # Orange overlay for built-up: (255, 140, 0)
        overlay_img = render_overlay(
            overlay_img, final_buildup, color=(255, 140, 0), alpha=0.45
        )

    # Save overlay PNG
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="satquery_optsar_")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"optsar_overlay_{uuid.uuid4().hex[:8]}.png"
    overlay_img.save(str(overlay_path))

    # ── 7. VLM narration ─────────────────────────────────────────────────
    vlm_client = vlm or get_vlm()

    pil_opt = render_for_vlm(image_dict_optical)
    pil_sar = render_for_vlm(image_dict_sar)

    water_pct = fusion_stats["water_pct"]
    buildup_pct = fusion_stats["built_up_pct"]
    water_both_pct = fusion_stats["water_both_pct"]
    buildup_both_pct = fusion_stats["built_up_both_pct"]

    query_context = f"\nUser Query: {query}" if query else ""
    vlm_prompt = (
        "You are an expert remote-sensing analyst specializing in optical and SAR fusion. "
        "Image 1 is the optical image and Image 2 is the co-registered Synthetic Aperture Radar (SAR) image. "
        f"Multi-sensor feature extraction yields:\n"
        f"- Water coverage: {water_pct}% (both sensors agree on {water_both_pct}%)\n"
        f"- Built-up coverage: {buildup_pct}% (both sensors agree on {buildup_both_pct}%)\n"
        f"- Target analysis focus: {targets}.{query_context}\n\n"
        "Provide a concise summary explaining:\n"
        "1. What each sensor contributes (optical spectral reflectance vs SAR backscatter and physical structure).\n"
        "2. The spatial distribution of water bodies and built-up infrastructure."
    )

    vlm_result = vlm_client.generate(
        images=[pil_opt, pil_sar],
        prompt=vlm_prompt,
        max_new_tokens=128,
    )

    # ── Confidence calculation ───────────────────────────────────────────
    confidence = vlm_result["confidence"]
    # Boost if high sensor agreement
    if fusion_stats["water_agreement_ratio"] > 0.5 or fusion_stats["built_up_agreement_ratio"] > 0.5:
        confidence = min(1.0, confidence + 0.05)
    # Lower if fallbacks were used
    if fallbacks["optical_water_fallback"] or fallbacks["optical_buildup_fallback"]:
        confidence = max(0.0, confidence - 0.05)
    if align_warnings:
        confidence = max(0.0, confidence - 0.05)

    stats = {
        **fusion_stats,
        **fallbacks,
        "alignment_warnings": align_warnings,
        "token_count": vlm_result["token_count"],
    }

    latency = time.perf_counter() - t0

    return ToolResult(
        tool="optsar_analyze",
        text=vlm_result["text"],
        overlay_png=str(overlay_path),
        stats=stats,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        params_used={"targets": targets},
        latency_s=max(0.01, round(latency, 2)),
    )
