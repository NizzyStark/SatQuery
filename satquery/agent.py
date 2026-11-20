"""Agentic orchestration module for SatQuery AI.

Implements:
- Query router: rule-based intent router matching queries and inputs to tools.
- Conflict detection: identifies mismatched queries and input configurations.
- Tool executor: validates parameters against configs/registry.yaml and dispatches execution.
- Multi-sensor fuser: combines visual overlays, numerical statistics, text answers, and confidence.
- Execution trace: writes transparent audit trace (trace.json) per Spec §6.3.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from satquery.io import load_image
from satquery.tools import (
    ToolResult,
    caption,
    change_analyze,
    optsar_analyze,
    vqa,
)
from satquery.validator import (
    _load_registry,
    validate_inputs,
    validate_params,
)
from satquery.vlm import VLMClient

logger = logging.getLogger(__name__)


# ── Routing Keywords ────────────────────────────────────────────────────────

CHANGE_KEYWORDS = [
    r"\bchange\b",
    r"\bchanged\b",
    r"\bchanges\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bbetween dates\b",
    r"\bbetween these two\b",
    r"\bincrease\b",
    r"\bincreased\b",
    r"\bdecrease\b",
    r"\bdecreased\b",
    r"\bunchanged\b",
    r"\bdifference\b",
    r"\bdifferences\b",
    r"\bdeforestation\b",
    r"\bflood\b",
]

OPTSAR_KEYWORDS = [
    r"\bsar\b",
    r"\bradar\b",
    r"\boptical together\b",
    r"\bboth sensors\b",
    r"\boptical and sar\b",
    r"\bsar and optical\b",
    r"\bcomplementary\b",
    r"\bbackscatter\b",
    r"\bbuilt-up and water\b",
    r"\bwater and built-up\b",
    r"\bsentinel-1\b",
]

CAPTION_KEYWORDS = [
    r"\bdescribe\b",
    r"\bdescription\b",
    r"\bcaption\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\boverview\b",
    r"\bwhat is visible\b",
    r"\btell me about\b",
]


class SatQueryAgent:
    """Agent orchestrator for routing queries, running tools, and fusing results."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        vlm: VLMClient | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config_path = config_path or "configs/registry.yaml"
        self.registry = _load_registry(self.config_path)
        self.vlm = vlm
        self.output_dir = Path(output_dir) if output_dir else None

    # ── 1. Query Router ─────────────────────────────────────────────────────

    @classmethod
    def route_query(
        cls,
        query: str,
        num_images: int = 1,
        mode: str = "auto",
        modalities: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Determine target tool and detect input/query conflicts.

        Ordered rules (Spec §9 Step 6):
        1. Mode is bitemporal, or query mentions change/before/after/increase/decrease
           → change_analyze (requires 2 images).
        2. Mode is optical_sar, or query mentions SAR/radar/optical together/both sensors
           → optsar_analyze (requires 2 images, optical + SAR).
        3. Single image and query asks to describe/caption/summarise
           → caption (requires 1 image).
        4. Otherwise single image
           → vqa (requires 1 image).

        Returns
        -------
        tool_name : str | None
            The selected tool ('vqa', 'caption', 'change_analyze', 'optsar_analyze')
            or None if a conflict is detected.
        conflict_error : str | None
            Clear error explanation if query conflicts with input configuration.
        """
        q_lower = query.strip().lower()
        mode_lower = mode.strip().lower()
        mods = [m.lower() for m in (modalities or [])]

        has_change_kw = any(re.search(pat, q_lower) for pat in CHANGE_KEYWORDS)
        has_optsar_kw = any(re.search(pat, q_lower) for pat in OPTSAR_KEYWORDS)
        has_caption_kw = any(re.search(pat, q_lower) for pat in CAPTION_KEYWORDS)

        # Rule 1: Change Analysis
        if mode_lower == "bitemporal" or has_change_kw:
            if num_images == 1:
                return (
                    None,
                    "Conflict: Query requires bi-temporal change analysis, but only 1 image "
                    "was provided. Change analysis requires two images of the same area from different dates.",
                )
            if mode_lower == "optical_sar":
                # User forced optical_sar mode but query asks about temporal change
                return (
                    None,
                    "Conflict: Bi-temporal change queries require two images of the same modality "
                    "at different dates, but operational mode was set to 'optical_sar'.",
                )
            return "change_analyze", None

        # Rule 2: Optical-SAR Analysis
        if mode_lower == "optical_sar" or has_optsar_kw:
            if num_images == 1:
                return (
                    None,
                    "Conflict: Query requires multi-sensor optical and SAR analysis, but only 1 image "
                    "was provided. Optical-SAR analysis requires one optical and one SAR image.",
                )
            if mode_lower == "bitemporal":
                return (
                    None,
                    "Conflict: Optical-SAR queries require one optical and one SAR image, "
                    "but operational mode was set to 'bitemporal'.",
                )
            return "optsar_analyze", None

        # Two images provided with neutral query
        if num_images == 2:
            opt_set = {"optical", "multispectral"}
            if len(mods) == 2 and ((mods[0] in opt_set and mods[1] == "sar") or (mods[0] == "sar" and mods[1] in opt_set)):
                return "optsar_analyze", None
            return "change_analyze", None

        # Rule 3: Single image captioning
        if has_caption_kw:
            return "caption", None

        # Rule 4: Otherwise single image VQA
        return "vqa", None

    # ── 2. Execution Pipeline ───────────────────────────────────────────────

    def run(
        self,
        query: str,
        images: list[str | Path | dict[str, Any]],
        mode: str = "auto",
        output_dir: str | Path | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Execute the agentic pipeline on input imagery and user query.

        Steps:
        1. Load & normalize input images.
        2. Route query to appropriate tool and check for conflicts.
        3. Validate input compatibility against target tool.
        4. Validate and clamp parameters against registry schema.
        5. Execute tool and catch exceptions.
        6. Assemble outputs and save execution trace (trace.json).

        Returns
        -------
        dict with:
            - 'task': selected tool name
            - 'text': analysis response
            - 'overlay': evidence overlay image path (or None)
            - 'stats': quantitative metrics dict
            - 'confidence': confidence score (0.0–1.0)
            - 'trace': transparent execution trace dictionary
            - 'trace_path': path to saved trace.json
        """
        t_start = time.perf_counter()
        out_dir = Path(output_dir or self.output_dir or tempfile.mkdtemp(prefix="satquery_agent_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load images if paths were provided
        image_dicts: list[dict[str, Any]] = []
        for item in images:
            if isinstance(item, dict) and "array" in item:
                image_dicts.append(item)
            elif isinstance(item, (str, Path)):
                image_dicts.append(load_image(item))
            else:
                raise TypeError(f"Unsupported image input type: {type(item)}")

        num_images = len(image_dicts)
        modalities = [img.get("modality", "unknown") for img in image_dicts]

        # 2. Route query and detect conflicts
        tool_name, conflict_error = self.route_query(
            query=query,
            num_images=num_images,
            mode=mode,
            modalities=modalities,
        )

        # Prepare trace input descriptors
        inputs_meta = [
            {
                "name": Path(img.get("path", f"image_{i}")).name,
                "modality": img.get("modality", "unknown"),
                "size": list(img.get("meta", {}).get("size", (0, 0))),
            }
            for i, img in enumerate(image_dicts)
        ]

        if conflict_error is not None:
            # Report clear error without guessing
            trace = {
                "query": query,
                "inputs": inputs_meta,
                "task": "conflict_error",
                "validation": {"passed": False, "messages": [conflict_error]},
                "steps": [],
                "outputs": {"text": conflict_error, "overlay": None, "confidence": 0.0},
                "latency_s": round(time.perf_counter() - t_start, 2),
            }
            trace_path = out_dir / "trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)

            return {
                "task": "error",
                "text": conflict_error,
                "overlay": None,
                "stats": {},
                "confidence": 0.0,
                "trace": trace,
                "trace_path": str(trace_path),
            }

        # 3. Input validation
        val_mode = "single" if tool_name in ("vqa", "caption") else ("bitemporal" if tool_name == "change_analyze" else "optical_sar")
        val_res = validate_inputs(image_dicts, mode=val_mode)

        if not val_res["passed"]:
            error_msg = "; ".join(val_res["messages"])
            trace = {
                "query": query,
                "inputs": inputs_meta,
                "task": tool_name,
                "validation": {"passed": False, "messages": val_res["messages"]},
                "steps": [],
                "outputs": {"text": error_msg, "overlay": None, "confidence": 0.0},
                "latency_s": round(time.perf_counter() - t_start, 2),
            }
            trace_path = out_dir / "trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)

            return {
                "task": tool_name,
                "text": f"Input validation failed: {error_msg}",
                "overlay": None,
                "stats": {},
                "confidence": 0.0,
                "trace": trace,
                "trace_path": str(trace_path),
            }

        # 4. Parameter validation against registry
        param_res = validate_params(tool_name, params=params, registry=self.registry)
        validated_params = param_res.params

        # 5. Tool execution
        tool_start = time.perf_counter()
        tool_result: ToolResult

        try:
            if tool_name == "vqa":
                max_tokens = validated_params.get("max_new_tokens", 128)
                tool_result = vqa(
                    image_dicts[0],
                    question=query,
                    max_new_tokens=max_tokens,
                    vlm=self.vlm,
                )
            elif tool_name == "caption":
                detail = validated_params.get("detail", "detailed")
                tool_result = caption(
                    image_dicts[0],
                    detail=detail,
                    vlm=self.vlm,
                )
            elif tool_name == "change_analyze":
                c_mode = validated_params.get("mode", "describe")
                diff_thr = validated_params.get("diff_threshold")
                tool_result = change_analyze(
                    image_dict_before=image_dicts[0],
                    image_dict_after=image_dicts[1],
                    query=query,
                    mode=c_mode,
                    diff_threshold=diff_thr,
                    vlm=self.vlm,
                    output_dir=str(out_dir),
                )
            elif tool_name == "optsar_analyze":
                targets = validated_params.get("targets", "both")
                # Identify optical vs SAR dicts
                opt_set = {"optical", "multispectral"}
                if image_dicts[0].get("modality") in opt_set:
                    opt_dict, sar_dict = image_dicts[0], image_dicts[1]
                else:
                    opt_dict, sar_dict = image_dicts[1], image_dicts[0]

                tool_result = optsar_analyze(
                    image_dict_optical=opt_dict,
                    image_dict_sar=sar_dict,
                    targets=targets,
                    query=query,
                    vlm=self.vlm,
                    output_dir=str(out_dir),
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.exception("Error executing tool %s: %s", tool_name, e)
            err_text = f"Tool execution failed for '{tool_name}': {str(e)}"
            trace = {
                "query": query,
                "inputs": inputs_meta,
                "task": tool_name,
                "validation": {"passed": True, "messages": val_res["messages"]},
                "steps": [{"tool": tool_name, "params": validated_params, "latency_s": round(time.perf_counter() - tool_start, 2)}],
                "outputs": {"text": err_text, "overlay": None, "confidence": 0.0},
                "latency_s": round(time.perf_counter() - t_start, 2),
            }
            trace_path = out_dir / "trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)

            return {
                "task": tool_name,
                "text": err_text,
                "overlay": None,
                "stats": {},
                "confidence": 0.0,
                "trace": trace,
                "trace_path": str(trace_path),
            }

        # 6. Build execution trace per Spec §6.3
        trace = {
            "query": query,
            "inputs": inputs_meta,
            "task": tool_name,
            "validation": {
                "passed": val_res["passed"],
                "messages": val_res["messages"] + param_res.messages,
            },
            "steps": [
                {
                    "tool": tool_name,
                    "params": tool_result.params_used,
                    "latency_s": tool_result.latency_s,
                }
            ],
            "outputs": {
                "text": tool_result.text,
                "overlay": tool_result.overlay_png,
                "confidence": tool_result.confidence,
            },
            "stats": tool_result.stats,
            "latency_s": round(time.perf_counter() - t_start, 2),
        }

        trace_path = out_dir / "trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

        return {
            "task": tool_name,
            "text": tool_result.text,
            "overlay": tool_result.overlay_png,
            "stats": tool_result.stats,
            "confidence": tool_result.confidence,
            "trace": trace,
            "trace_path": str(trace_path),
        }
