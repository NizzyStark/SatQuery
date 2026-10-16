"""Validator module for SatQuery AI.

Validates input imagery compatibility, mode constraints, spatial alignment,
and tool parameter boundaries against configs/registry.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
import yaml


class ParamValidationResult(NamedTuple):
    """Result of parameter validation and clipping."""
    params: dict[str, Any]
    messages: list[str]


def _load_registry(registry: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    """Load tool registry from dict, path, or default configs/registry.yaml."""
    if isinstance(registry, dict):
        return registry

    if registry is None:
        reg_path = Path(__file__).resolve().parent.parent / "configs" / "registry.yaml"
    else:
        reg_path = Path(registry).resolve()

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry configuration not found: {reg_path}")

    with open(reg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_inputs(images: list[dict[str, Any]], mode: str = "auto") -> dict[str, Any]:
    """Validate image inputs against the specified operational mode.

    Args:
        images: List of loaded image dictionaries (from satquery.io.load_image).
        mode: Mode to validate against: 'single', 'bitemporal', 'optical_sar', or 'auto'.

    Returns:
        dict with:
            - 'passed': bool indicating validation success
            - 'messages': list of informational, warning, or error messages
            - 'inferred_mode': the effective mode evaluated
    """
    messages: list[str] = []
    norm_mode = mode.lower().strip()

    valid_modes = {"single", "bitemporal", "optical_sar", "auto"}
    if norm_mode not in valid_modes:
        return {
            "passed": False,
            "messages": [f"Unsupported mode '{mode}'. Choose from: {sorted(valid_modes)}."],
            "inferred_mode": norm_mode,
        }

    if not isinstance(images, list):
        return {
            "passed": False,
            "messages": ["Input 'images' must be a list of image dictionaries."],
            "inferred_mode": norm_mode,
        }

    num_images = len(images)
    effective_mode = norm_mode

    # Auto mode resolution
    if norm_mode == "auto":
        if num_images == 1:
            effective_mode = "single"
            messages.append("Auto-detected mode: 'single' (1 image provided).")
        elif num_images == 2:
            m1 = images[0].get("modality", "unknown").lower()
            m2 = images[1].get("modality", "unknown").lower()
            opt_set = {"optical", "multispectral"}

            has_optical = (m1 in opt_set) or (m2 in opt_set)
            has_sar = (m1 == "sar") or (m2 == "sar")

            if has_optical and has_sar:
                effective_mode = "optical_sar"
                messages.append("Auto-detected mode: 'optical_sar' (optical and SAR images provided).")
            else:
                effective_mode = "bitemporal"
                messages.append("Auto-detected mode: 'bitemporal' (pair of same modality provided).")
        else:
            return {
                "passed": False,
                "messages": [f"Auto mode requires 1 or 2 images, got {num_images}."],
                "inferred_mode": "auto",
            }

    # Single image mode validation
    if effective_mode == "single":
        if num_images != 1:
            return {
                "passed": False,
                "messages": [f"Expected 1 image for 'single' mode, got {num_images}."],
                "inferred_mode": effective_mode,
            }
        messages.append("Single image validation passed.")
        return {
            "passed": True,
            "messages": messages,
            "inferred_mode": effective_mode,
        }

    # Pair mode validation (bitemporal or optical_sar)
    if effective_mode in ("bitemporal", "optical_sar"):
        if num_images != 2:
            return {
                "passed": False,
                "messages": [f"Expected 2 images for '{effective_mode}' mode, got {num_images}."],
                "inferred_mode": effective_mode,
            }

        img1, img2 = images[0], images[1]
        m1 = img1.get("modality", "unknown").lower()
        m2 = img2.get("modality", "unknown").lower()
        opt_set = {"optical", "multispectral"}

        # Modality check
        if effective_mode == "optical_sar":
            is_m1_opt = m1 in opt_set
            is_m2_opt = m2 in opt_set
            is_m1_sar = m1 == "sar"
            is_m2_sar = m2 == "sar"

            if not ((is_m1_opt and is_m2_sar) or (is_m1_sar and is_m2_opt)):
                return {
                    "passed": False,
                    "messages": [
                        f"optical_sar mode requires one optical/multispectral image and one SAR image, "
                        f"got '{m1}' and '{m2}'."
                    ],
                    "inferred_mode": effective_mode,
                }
        elif effective_mode == "bitemporal":
            both_optical = (m1 in opt_set) and (m2 in opt_set)
            both_sar = (m1 == "sar") and (m2 == "sar")

            if not (both_optical or both_sar):
                return {
                    "passed": False,
                    "messages": [
                        f"bitemporal mode requires both images to be of the same modality, "
                        f"got '{m1}' and '{m2}'."
                    ],
                    "inferred_mode": effective_mode,
                }

        # Size consistency check (Warning)
        size1 = img1.get("meta", {}).get("size")
        size2 = img2.get("meta", {}).get("size")
        if size1 and size2 and size1 != size2:
            messages.append(
                f"Warning: Image sizes differ ({size1} vs {size2}). Images will be resized during analysis."
            )

        # CRS consistency check for GeoTIFF pairs (Warning)
        crs1 = img1.get("meta", {}).get("crs")
        crs2 = img2.get("meta", {}).get("crs")
        if crs1 and crs2 and crs1 != crs2:
            messages.append(
                f"Warning: CRS mismatch detected ({crs1} vs {crs2}). Images may not be co-registered."
            )

        # Extent consistency check for GeoTIFF pairs (Warning)
        bounds1 = img1.get("meta", {}).get("bounds")
        bounds2 = img2.get("meta", {}).get("bounds")
        if bounds1 and bounds2 and bounds1 != bounds2:
            messages.append(
                f"Warning: Spatial extents differ ({bounds1} vs {bounds2})."
            )

        messages.append(f"{effective_mode} image pair validation passed.")
        return {
            "passed": True,
            "messages": messages,
            "inferred_mode": effective_mode,
        }

    return {
        "passed": False,
        "messages": [f"Unhandled mode '{effective_mode}'."],
        "inferred_mode": effective_mode,
    }


def validate_params(
    tool_name: str,
    params: dict[str, Any] | None = None,
    registry: dict[str, Any] | str | Path | None = None,
) -> ParamValidationResult:
    """Validate and clip parameters against tool registry constraints.

    Args:
        tool_name: Name of tool ('vqa', 'caption', 'change_analyze', 'optsar_analyze').
        params: Dictionary of parameters supplied by caller/user.
        registry: Optional registry dict or path to registry.yaml.

    Returns:
        ParamValidationResult (named tuple with .params and .messages, also supports 2-element unpack).
    """
    reg_dict = _load_registry(registry)
    tools_config = reg_dict.get("tools", {})

    if tool_name not in tools_config:
        return ParamValidationResult(
            params={},
            messages=[f"Tool '{tool_name}' not defined in registry."],
        )

    tool_def = tools_config[tool_name]
    param_rules = tool_def.get("params", {})
    user_params = params or {}
    validated: dict[str, Any] = {}
    messages: list[str] = []

    for param_name, rule in param_rules.items():
        if isinstance(rule, list):
            # Check if rule is numerical range [min_val, max_val]
            is_num_range = (
                len(rule) == 2
                and all(isinstance(x, (int, float)) for x in rule)
            )

            if is_num_range:
                min_val, max_val = rule[0], rule[1]
                is_int = isinstance(min_val, int) and isinstance(max_val, int)

                if param_name in user_params:
                    raw_val = user_params[param_name]
                    try:
                        num_val = int(raw_val) if is_int else float(raw_val)
                    except (ValueError, TypeError):
                        num_val = max_val
                        messages.append(
                            f"Invalid numeric value '{raw_val}' for '{param_name}'; defaulted to {num_val}."
                        )

                    if num_val < min_val:
                        clipped = min_val
                        messages.append(
                            f"Parameter '{param_name}' ({num_val}) clipped to minimum {min_val}."
                        )
                    elif num_val > max_val:
                        clipped = max_val
                        messages.append(
                            f"Parameter '{param_name}' ({num_val}) clipped to maximum {max_val}."
                        )
                    else:
                        clipped = num_val

                    validated[param_name] = clipped
                else:
                    # Default: use max_val for tokens, midpoint for floats
                    default_val = max_val if is_int else round((min_val + max_val) / 2, 2)
                    validated[param_name] = default_val

            else:
                # Categorical choices: [choice1, choice2, ...]
                choices = [str(c) for c in rule]
                if param_name in user_params:
                    raw_val = str(user_params[param_name]).strip()
                    matched = [c for c in choices if c.lower() == raw_val.lower()]
                    if matched:
                        validated[param_name] = matched[0]
                    else:
                        default_val = choices[0]
                        messages.append(
                            f"Invalid value '{raw_val}' for '{param_name}'. "
                            f"Allowed: {choices}; fallback to '{default_val}'."
                        )
                        validated[param_name] = default_val
                else:
                    validated[param_name] = choices[0]

    # Preserve any extra user params not in schema, noting them
    for k, v in user_params.items():
        if k not in validated and k not in param_rules:
            validated[k] = v

    return ParamValidationResult(params=validated, messages=messages)
