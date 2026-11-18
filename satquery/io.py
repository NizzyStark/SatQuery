"""I/O module for SatQuery AI.

Provides image loading, metadata extraction, modality auto-detection,
rendering for Vision-Language Models (VLM), and evidence overlay rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageColor

try:
    import rasterio
except (ImportError, Exception):
    rasterio = None


def detect_modality(array: np.ndarray, user_tag: str | None = None) -> str:
    """Detect image modality based on user tag or array heuristics.

    Heuristics:
    - If user_tag provided: normalize and return user tag.
    - 1-2 bands with values suggesting dB (negative values, typical SAR range) -> 'sar'
    - 1-2 bands with non-negative values -> 'optical'
    - 3 bands -> 'optical'
    - >3 bands -> 'multispectral'
    - Otherwise -> 'unknown'
    """
    if user_tag:
        tag = user_tag.strip().lower()
        if tag in ("optical", "sar", "multispectral", "unknown"):
            return tag
        return tag

    bands = array.shape[0] if array.ndim == 3 else 1
    finite_mask = np.isfinite(array)
    finite_vals = array[finite_mask]

    if bands in (1, 2):
        if len(finite_vals) > 0 and np.min(finite_vals) < 0:
            return "sar"
        return "optical"
    elif bands == 3:
        return "optical"
    elif bands > 3:
        return "multispectral"

    return "unknown"


def load_image(path: str | Path, modality: str | None = None) -> dict[str, Any]:
    """Load an image (GeoTIFF, TIFF, PNG, JPEG) and extract metadata and modality.

    Args:
        path: Path to the image file.
        modality: Optional explicit modality tag ('optical', 'sar', 'multispectral').

    Returns:
        dict with keys:
            - 'array': np.ndarray of shape (bands, height, width)
            - 'meta': dict with 'bands', 'dtype', 'crs', 'transform', 'size', 'bounds', 'format'
            - 'modality': detected or specified modality string
            - 'path': absolute path to image file
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in (".tif", ".tiff", ".geotiff"):
        if rasterio is None:
            raise ImportError(
                "rasterio is required to load GeoTIFF files but could not be imported. "
                "Install it with: pip install rasterio"
            )
        with rasterio.open(file_path) as src:
            array = src.read()  # Shape: (count, height, width)
            crs_str = src.crs.to_string() if src.crs else None
            transform_tuple = tuple(src.transform) if src.transform else None
            bounds_tuple = tuple(src.bounds) if src.bounds else None
            meta = {
                "bands": int(src.count),
                "dtype": str(src.dtypes[0]),
                "crs": crs_str,
                "transform": transform_tuple,
                "size": (int(src.width), int(src.height)),
                "bounds": bounds_tuple,
                "format": src.driver or "GTiff",
            }
    elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        with Image.open(file_path) as pil_img:
            format_name = pil_img.format or suffix.strip(".").upper()
            width, height = pil_img.size
            if pil_img.mode in ("RGB", "L", "RGBA"):
                raw_arr = np.array(pil_img)
            else:
                raw_arr = np.array(pil_img.convert("RGB"))

            if raw_arr.ndim == 2:
                # Grayscale: (H, W) -> (1, H, W)
                array = raw_arr[np.newaxis, ...]
            elif raw_arr.ndim == 3:
                if raw_arr.shape[2] == 4:
                    raw_arr = raw_arr[:, :, :3]
                # (H, W, C) -> (C, H, W)
                array = np.transpose(raw_arr, (2, 0, 1))
            else:
                array = raw_arr[np.newaxis, ...]

            meta = {
                "bands": int(array.shape[0]),
                "dtype": str(array.dtype),
                "crs": None,
                "transform": None,
                "size": (int(width), int(height)),
                "bounds": None,
                "format": format_name,
            }
    else:
        # Fallback using rasterio first, then Pillow
        try:
            with rasterio.open(file_path) as src:
                array = src.read()
                meta = {
                    "bands": int(src.count),
                    "dtype": str(src.dtypes[0]),
                    "crs": src.crs.to_string() if src.crs else None,
                    "transform": tuple(src.transform) if src.transform else None,
                    "size": (int(src.width), int(src.height)),
                    "bounds": tuple(src.bounds) if src.bounds else None,
                    "format": src.driver or "Raster",
                }
        except Exception:
            with Image.open(file_path) as pil_img:
                raw_arr = np.array(pil_img.convert("RGB"))
                array = np.transpose(raw_arr, (2, 0, 1))
                meta = {
                    "bands": int(array.shape[0]),
                    "dtype": str(array.dtype),
                    "crs": None,
                    "transform": None,
                    "size": (int(pil_img.width), int(pil_img.height)),
                    "bounds": None,
                    "format": pil_img.format or "Image",
                }

    detected_modality = detect_modality(array, user_tag=modality)

    return {
        "array": array,
        "meta": meta,
        "modality": detected_modality,
        "path": str(file_path),
    }


def percentile_stretch(
    band: np.ndarray,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
) -> np.ndarray:
    """Stretch band values to 0..255 uint8 using lower and upper percentiles."""
    finite_mask = np.isfinite(band)
    if not np.any(finite_mask):
        return np.zeros(band.shape, dtype=np.uint8)

    valid_vals = band[finite_mask]
    p_low, p_high = np.percentile(valid_vals, (lower_pct, upper_pct))

    if p_high <= p_low:
        p_min = np.min(valid_vals)
        p_max = np.max(valid_vals)
        if p_max > p_min:
            scaled = np.clip((band - p_min) / (p_max - p_min), 0.0, 1.0)
            return (scaled * 255.0).astype(np.uint8)
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255.0).astype(np.uint8)


def render_for_vlm(image_dict: dict[str, Any]) -> Image.Image:
    """Render an image dictionary as a 3-channel 8-bit RGB PIL Image for VLM consumption.

    Rendering logic:
    - Sentinel-2 / multispectral (>=4 bands): true colour using bands 4, 3, 2
      (0-indexed: 3, 2, 1) with 2-98% percentile stretch.
    - Optical with 3 bands: bands 0, 1, 2 with 2-98% percentile stretch.
    - SAR with >=2 bands: R=VV (band 0), G=VH (band 1), B=VV-VH (cross-ratio in dB),
      with 2-98% percentile stretch.
    - SAR or Optical single band: replicate to 3 channels with 2-98% percentile stretch.
    """
    array = image_dict["array"]
    modality = image_dict.get("modality", "optical")

    if array.ndim == 2:
        array = array[np.newaxis, ...]

    bands, height, width = array.shape

    if modality == "sar":
        if bands >= 2:
            vv = array[0].astype(np.float32)
            vh = array[1].astype(np.float32)
            diff = vv - vh
            r = percentile_stretch(vv)
            g = percentile_stretch(vh)
            b = percentile_stretch(diff)
        else:
            gray = percentile_stretch(array[0])
            r = g = b = gray
    else:
        # Optical / multispectral
        if bands >= 4:
            # Sentinel-2 true colour: B4 (Red, index 3), B3 (Green, index 2), B2 (Blue, index 1)
            r = percentile_stretch(array[3])
            g = percentile_stretch(array[2])
            b = percentile_stretch(array[1])
        elif bands == 3:
            r = percentile_stretch(array[0])
            g = percentile_stretch(array[1])
            b = percentile_stretch(array[2])
        elif bands == 2:
            r = percentile_stretch(array[0])
            g = percentile_stretch(array[1])
            b = percentile_stretch(array[0])
        else:
            gray = percentile_stretch(array[0])
            r = g = b = gray

    rgb_array = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb_array, mode="RGB")


def render_overlay(
    base_image: Image.Image | dict[str, Any] | np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] | str = (255, 0, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a binary mask onto a base image with specified color and transparency.

    Args:
        base_image: PIL Image, image dictionary from load_image, or numpy array.
        mask: 2D boolean or numeric array (H, W).
        color: RGB tuple (0..255) or color name (e.g., 'red', 'blue', 'orange').
        alpha: Transparency factor (0.0 = only base, 1.0 = solid overlay color).

    Returns:
        PIL.Image in RGB mode with the overlay applied.
    """
    if isinstance(base_image, dict):
        base_pil = render_for_vlm(base_image)
    elif isinstance(base_image, Image.Image):
        base_pil = base_image.convert("RGB")
    elif isinstance(base_image, np.ndarray):
        if base_image.ndim == 2:
            gray = percentile_stretch(base_image)
            base_pil = Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")
        elif base_image.ndim == 3:
            if base_image.shape[0] in (1, 3, 4):
                base_dict = {"array": base_image, "modality": "optical"}
                base_pil = render_for_vlm(base_dict)
            else:
                base_pil = Image.fromarray(base_image.astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"Unsupported array shape for base_image: {base_image.shape}")
    else:
        raise TypeError(f"Unsupported type for base_image: {type(base_image)}")

    # Resolve RGB color tuple
    if isinstance(color, str):
        color_rgb = ImageColor.getrgb(color)[:3]
    else:
        color_rgb = tuple(int(c) for c in color[:3])

    base_arr = np.array(base_pil, dtype=np.float32)
    h, w, _ = base_arr.shape

    # Match mask size to base image if necessary
    mask_arr = np.squeeze(np.asarray(mask))
    if mask_arr.shape != (h, w):
        mask_pil = Image.fromarray(mask_arr.astype(np.uint8)).resize((w, h), resample=Image.Resampling.NEAREST)
        mask_arr = np.array(mask_pil)

    mask_bool = mask_arr > 0
    alpha_clamped = max(0.0, min(1.0, float(alpha)))

    # Blend overlay where mask is positive
    color_layer = np.zeros_like(base_arr)
    color_layer[..., 0] = color_rgb[0]
    color_layer[..., 1] = color_rgb[1]
    color_layer[..., 2] = color_rgb[2]

    blended = base_arr.copy()
    blended[mask_bool] = (1.0 - alpha_clamped) * base_arr[mask_bool] + alpha_clamped * color_layer[mask_bool]
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")
