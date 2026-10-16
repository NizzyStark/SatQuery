"""Unit tests for satquery.io and satquery.validator.

Uses synthetic raster data (GeoTIFF via rasterio, PNG via Pillow) to verify:
- GeoTIFF and PNG loading with metadata and modality extraction
- SAR vs optical detection heuristics
- VLM rendering (3-channel 8-bit RGB output with percentile stretch)
- Overlay rendering with masks
- Input validation across modes (single, bitemporal, optical_sar, auto)
- Spatial mismatch warnings (size, CRS, extent)
- Tool parameter validation and clipping against registry
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.io import (
    detect_modality,
    load_image,
    percentile_stretch,
    render_for_vlm,
    render_overlay,
)
from satquery.validator import validate_inputs, validate_params


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for synthetic raster files."""
    return tmp_path


def create_synthetic_geotiff(
    file_path: Path,
    shape: tuple[int, int, int] = (3, 64, 64),
    dtype: str = "uint8",
    crs: str | None = "EPSG:4326",
    origin: tuple[float, float] = (10.0, 50.0),
    pixel_size: float = 0.001,
    data: np.ndarray | None = None,
) -> Path:
    """Create a synthetic GeoTIFF file."""
    bands, height, width = shape
    if data is None:
        if dtype == "float32":
            data = np.random.uniform(0.0, 1.0, size=(bands, height, width)).astype(np.float32)
        else:
            data = np.random.randint(0, 255, size=(bands, height, width), dtype=np.uint8)

    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


def create_synthetic_png(
    file_path: Path,
    size: tuple[int, int] = (64, 64),
    mode: str = "RGB",
    color: tuple[int, int, int] | int = (100, 150, 200),
) -> Path:
    """Create a synthetic PNG image."""
    img = Image.new(mode, size, color=color)
    img.save(file_path)
    return file_path


# =========================================================================
# I/O Module Tests
# =========================================================================

def test_load_geotiff_optical(temp_workspace: Path) -> None:
    """Test loading a 3-band optical GeoTIFF."""
    tif_path = temp_workspace / "optical_3band.tif"
    create_synthetic_geotiff(tif_path, shape=(3, 64, 64), dtype="uint8", crs="EPSG:4326")

    img_dict = load_image(tif_path)
    assert img_dict["modality"] == "optical"
    assert img_dict["meta"]["bands"] == 3
    assert img_dict["meta"]["size"] == (64, 64)
    assert img_dict["meta"]["crs"] == "EPSG:4326"
    assert img_dict["array"].shape == (3, 64, 64)
    assert img_dict["path"] == str(tif_path.resolve())


def test_load_geotiff_multispectral(temp_workspace: Path) -> None:
    """Test loading a 4-band Sentinel-2 style multispectral GeoTIFF."""
    tif_path = temp_workspace / "sentinel2_4band.tif"
    create_synthetic_geotiff(tif_path, shape=(4, 64, 64), dtype="uint16")

    img_dict = load_image(tif_path)
    assert img_dict["modality"] == "multispectral"
    assert img_dict["meta"]["bands"] == 4
    assert img_dict["array"].shape == (4, 64, 64)


def test_load_png_optical(temp_workspace: Path) -> None:
    """Test loading standard RGB and Grayscale PNG files."""
    png_path = temp_workspace / "sample_optical.png"
    create_synthetic_png(png_path, size=(80, 50), mode="RGB")

    img_dict = load_image(png_path)
    assert img_dict["modality"] == "optical"
    assert img_dict["meta"]["size"] == (80, 50)
    assert img_dict["array"].shape == (3, 50, 80)
    assert img_dict["meta"]["format"] == "PNG"

    # Test Grayscale PNG
    gray_path = temp_workspace / "sample_gray.png"
    create_synthetic_png(gray_path, size=(40, 40), mode="L", color=120)
    gray_dict = load_image(gray_path)
    assert gray_dict["array"].shape == (1, 40, 40)
    assert gray_dict["meta"]["bands"] == 1


def test_sar_detection_heuristics(temp_workspace: Path) -> None:
    """Test detection heuristics for SAR imagery (negative dB values)."""
    # 2-band float SAR data with typical backscatter dB values (-30 to -5 dB)
    sar_data = np.random.uniform(-30.0, -5.0, size=(2, 64, 64)).astype(np.float32)
    sar_tif = temp_workspace / "sentinel1_sar.tif"
    create_synthetic_geotiff(sar_tif, shape=(2, 64, 64), dtype="float32", data=sar_data)

    img_dict = load_image(sar_tif)
    assert img_dict["modality"] == "sar"
    assert img_dict["meta"]["bands"] == 2

    # Single-band float SAR data with negative dB values
    single_sar_data = np.random.uniform(-25.0, -2.0, size=(1, 64, 64)).astype(np.float32)
    single_sar_tif = temp_workspace / "sar_single.tif"
    create_synthetic_geotiff(single_sar_tif, shape=(1, 64, 64), dtype="float32", data=single_sar_data)

    single_dict = load_image(single_sar_tif)
    assert single_dict["modality"] == "sar"

    # Explicit user tag override
    tagged_dict = load_image(sar_tif, modality="optical")
    assert tagged_dict["modality"] == "optical"


def test_render_for_vlm_optical_and_sar(temp_workspace: Path) -> None:
    """Test render_for_vlm produces valid 3-channel 8-bit RGB PIL Images."""
    # 1. 4-band multispectral image (uses bands 3, 2, 1 for B4, B3, B2)
    ms_data = np.random.randint(100, 3000, size=(4, 64, 64), dtype=np.uint16)
    ms_tif = temp_workspace / "ms.tif"
    create_synthetic_geotiff(ms_tif, shape=(4, 64, 64), dtype="uint16", data=ms_data)
    ms_dict = load_image(ms_tif)
    pil_ms = render_for_vlm(ms_dict)

    assert isinstance(pil_ms, Image.Image)
    assert pil_ms.mode == "RGB"
    assert pil_ms.size == (64, 64)

    # 2. 2-band SAR image (R=VV, G=VH, B=VV-VH)
    sar_data = np.random.uniform(-25.0, -5.0, size=(2, 64, 64)).astype(np.float32)
    sar_tif = temp_workspace / "sar.tif"
    create_synthetic_geotiff(sar_tif, shape=(2, 64, 64), dtype="float32", data=sar_data)
    sar_dict = load_image(sar_tif)
    pil_sar = render_for_vlm(sar_dict)

    assert isinstance(pil_sar, Image.Image)
    assert pil_sar.mode == "RGB"
    assert pil_sar.size == (64, 64)

    # 3. Single-band image (replicate to 3 channels)
    single_data = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    single_tif = temp_workspace / "single.tif"
    create_synthetic_geotiff(single_tif, shape=(1, 32, 32), dtype="uint8", data=single_data)
    single_dict = load_image(single_tif)
    pil_single = render_for_vlm(single_dict)

    assert pil_single.mode == "RGB"
    assert pil_single.size == (32, 32)
    single_arr = np.array(pil_single)
    # Check that channels are identical (replicated grayscale)
    np.testing.assert_array_equal(single_arr[..., 0], single_arr[..., 1])
    np.testing.assert_array_equal(single_arr[..., 1], single_arr[..., 2])


def test_percentile_stretch() -> None:
    """Test percentile stretch behavior with uniform and normal data."""
    # Data with outliers
    arr = np.array([0.0, 10.0, 50.0, 90.0, 1000.0])
    stretched = percentile_stretch(arr, 10.0, 90.0)
    assert stretched.dtype == np.uint8
    assert stretched[0] == 0
    assert stretched[-1] == 255

    # Uniform constant data
    const_arr = np.ones((10, 10), dtype=np.float32) * 42.0
    stretched_const = percentile_stretch(const_arr)
    assert np.all(stretched_const == 0)


def test_render_overlay(temp_workspace: Path) -> None:
    """Test render_overlay blends binary masks onto images."""
    base_img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1

    blended = render_overlay(base_img, mask, color="red", alpha=0.5)
    assert isinstance(blended, Image.Image)
    assert blended.size == (64, 64)

    blended_arr = np.array(blended)
    # Outside mask: should remain black (0, 0, 0)
    assert blended_arr[0, 0, 0] == 0
    # Inside mask: 0.5 * 0 + 0.5 * 255 ≈ 127
    assert 120 <= blended_arr[20, 20, 0] <= 135
    assert blended_arr[20, 20, 1] == 0
    assert blended_arr[20, 20, 2] == 0


# =========================================================================
# Validator Module Tests
# =========================================================================

def test_validate_inputs_wrong_image_count() -> None:
    """Test wrong image count detection for each mode."""
    dummy_img = {"array": np.zeros((3, 32, 32)), "meta": {"size": (32, 32)}, "modality": "optical"}

    # Single mode requires 1 image
    res_zero = validate_inputs([], mode="single")
    assert not res_zero["passed"]
    assert "Expected 1 image" in res_zero["messages"][0]

    res_two = validate_inputs([dummy_img, dummy_img], mode="single")
    assert not res_two["passed"]
    assert "Expected 1 image" in res_two["messages"][0]

    # Bitemporal mode requires 2 images
    res_bi_one = validate_inputs([dummy_img], mode="bitemporal")
    assert not res_bi_one["passed"]
    assert "Expected 2 images" in res_bi_one["messages"][0]

    # Optical-SAR mode requires 2 images
    res_optsar_three = validate_inputs([dummy_img, dummy_img, dummy_img], mode="optical_sar")
    assert not res_optsar_three["passed"]
    assert "Expected 2 images" in res_optsar_three["messages"][0]


def test_validate_inputs_size_mismatch() -> None:
    """Test detection and warning for image size mismatch in pairs."""
    img1 = {"array": np.zeros((3, 64, 64)), "meta": {"size": (64, 64)}, "modality": "optical"}
    img2 = {"array": np.zeros((3, 128, 128)), "meta": {"size": (128, 128)}, "modality": "optical"}

    res = validate_inputs([img1, img2], mode="bitemporal")
    assert res["passed"]  # Pair passes, but size difference is warned
    warning_found = any("Image sizes differ" in msg for msg in res["messages"])
    assert warning_found


def test_validate_inputs_crs_and_extent_mismatch() -> None:
    """Test CRS and extent mismatch warnings for GeoTIFF pairs."""
    img1 = {
        "array": np.zeros((3, 64, 64)),
        "meta": {"size": (64, 64), "crs": "EPSG:4326", "bounds": (10.0, 50.0, 11.0, 51.0)},
        "modality": "optical",
    }
    img2 = {
        "array": np.zeros((3, 64, 64)),
        "meta": {"size": (64, 64), "crs": "EPSG:3857", "bounds": (1000.0, 2000.0, 1100.0, 2100.0)},
        "modality": "optical",
    }

    res = validate_inputs([img1, img2], mode="bitemporal")
    assert res["passed"]
    crs_warn = any("CRS mismatch" in msg for msg in res["messages"])
    extent_warn = any("Spatial extents differ" in msg for msg in res["messages"])
    assert crs_warn
    assert extent_warn


def test_modality_validation_optical_sar() -> None:
    """Test modality validation for optical_sar mode."""
    opt_img = {"array": np.zeros((3, 64, 64)), "meta": {"size": (64, 64)}, "modality": "optical"}
    ms_img = {"array": np.zeros((4, 64, 64)), "meta": {"size": (64, 64)}, "modality": "multispectral"}
    sar_img = {"array": np.zeros((2, 64, 64)), "meta": {"size": (64, 64)}, "modality": "sar"}

    # Optical + SAR -> Valid
    res_valid = validate_inputs([opt_img, sar_img], mode="optical_sar")
    assert res_valid["passed"]

    # Multispectral + SAR -> Valid
    res_ms_sar = validate_inputs([ms_img, sar_img], mode="optical_sar")
    assert res_ms_sar["passed"]

    # Optical + Optical -> Invalid for optical_sar
    res_invalid = validate_inputs([opt_img, ms_img], mode="optical_sar")
    assert not res_invalid["passed"]
    assert "optical_sar mode requires one optical" in res_invalid["messages"][0]

    # SAR + SAR -> Invalid for optical_sar
    res_two_sar = validate_inputs([sar_img, sar_img], mode="optical_sar")
    assert not res_two_sar["passed"]


def test_modality_validation_bitemporal() -> None:
    """Test modality validation for bitemporal mode."""
    opt_img = {"array": np.zeros((3, 64, 64)), "meta": {"size": (64, 64)}, "modality": "optical"}
    ms_img = {"array": np.zeros((4, 64, 64)), "meta": {"size": (64, 64)}, "modality": "multispectral"}
    sar_img = {"array": np.zeros((2, 64, 64)), "meta": {"size": (64, 64)}, "modality": "sar"}

    # Both optical/multispectral -> Valid
    res_optical = validate_inputs([opt_img, ms_img], mode="bitemporal")
    assert res_optical["passed"]

    # Both SAR -> Valid
    res_sar = validate_inputs([sar_img, sar_img], mode="bitemporal")
    assert res_sar["passed"]

    # Optical + SAR -> Invalid for bitemporal
    res_mixed = validate_inputs([opt_img, sar_img], mode="bitemporal")
    assert not res_mixed["passed"]
    assert "bitemporal mode requires both images to be of the same modality" in res_mixed["messages"][0]


def test_validate_inputs_auto_mode() -> None:
    """Test auto mode inference based on input counts and modalities."""
    opt_img = {"array": np.zeros((3, 64, 64)), "meta": {"size": (64, 64)}, "modality": "optical"}
    sar_img = {"array": np.zeros((2, 64, 64)), "meta": {"size": (64, 64)}, "modality": "sar"}

    # Single image -> inferred as single
    res_single = validate_inputs([opt_img], mode="auto")
    assert res_single["passed"]
    assert res_single["inferred_mode"] == "single"

    # Optical + SAR -> inferred as optical_sar
    res_optsar = validate_inputs([opt_img, sar_img], mode="auto")
    assert res_optsar["passed"]
    assert res_optsar["inferred_mode"] == "optical_sar"

    # Optical + Optical -> inferred as bitemporal
    res_bitemp = validate_inputs([opt_img, opt_img], mode="auto")
    assert res_bitemp["passed"]
    assert res_bitemp["inferred_mode"] == "bitemporal"


def test_param_validation() -> None:
    """Test parameter validation against registry constraints (in-range & clipping)."""
    # Test vqa max_new_tokens clipping: range [16, 128]
    val_normal, msgs_normal = validate_params("vqa", {"max_new_tokens": 64})
    assert val_normal["max_new_tokens"] == 64
    assert len(msgs_normal) == 0

    val_high, msgs_high = validate_params("vqa", {"max_new_tokens": 256})
    assert val_high["max_new_tokens"] == 128
    assert any("clipped to maximum 128" in m for m in msgs_high)

    val_low, msgs_low = validate_params("vqa", {"max_new_tokens": 5})
    assert val_low["max_new_tokens"] == 16
    assert any("clipped to minimum 16" in m for m in msgs_low)

    # Test caption detail categorical choices: [brief, detailed]
    val_cap, msgs_cap = validate_params("caption", {"detail": "detailed"})
    assert val_cap["detail"] == "detailed"
    assert len(msgs_cap) == 0

    val_cap_bad, msgs_cap_bad = validate_params("caption", {"detail": "super_long"})
    assert val_cap_bad["detail"] == "brief"
    assert any("fallback to 'brief'" in m for m in msgs_cap_bad)

    # Test change_analyze diff_threshold: range [0.1, 0.5] and mode: [describe, vqa]
    val_chg, msgs_chg = validate_params(
        "change_analyze",
        {"diff_threshold": 0.8, "mode": "vqa"},
    )
    assert val_chg["diff_threshold"] == 0.5
    assert val_chg["mode"] == "vqa"
    assert any("clipped to maximum 0.5" in m for m in msgs_chg)

    # Test optsar_analyze targets: [built_up, water, both]
    val_optsar, msgs_optsar = validate_params("optsar_analyze", {"targets": "water"})
    assert val_optsar["targets"] == "water"

    val_optsar_bad, msgs_optsar_bad = validate_params("optsar_analyze", {"targets": "vegetation"})
    assert val_optsar_bad["targets"] == "built_up"  # First choice fallback
    assert len(msgs_optsar_bad) > 0

    # Test unknown tool
    val_unknown, msgs_unknown = validate_params("unknown_tool", {})
    assert "not defined in registry" in msgs_unknown[0]
