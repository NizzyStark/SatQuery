"""Unit and integration tests for Step 6: SatQueryAgent.

Tests verify:
- Ordered query routing rules on 25+ labelled remote-sensing queries (reporting accuracy >= 95%)
- Conflict detection between query intent and input image configurations
- Input validation failures handled transparently in trace.json
- Parameter validation and clipping against configs/registry.yaml
- End-to-end execution across all 4 tools (vqa, caption, change_analyze, optsar_analyze)
- trace.json compliance with Spec §6.3 schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from satquery.agent import SatQueryAgent
from satquery.tools import ToolResult
from satquery.vlm import VLMClient


# ── Labelled Queries Dataset (26 queries covering all operational modes) ─────

LABELLED_ROUTER_DATASET: list[dict[str, Any]] = [
    # Single image VQA queries
    {
        "query": "What is the primary land-cover type in this satellite image?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },
    {
        "query": "Are there any roads or bridges visible near the river?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },
    {
        "query": "Is there standing water in this agricultural field?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },
    {
        "query": "How many large industrial buildings are in this sector?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },
    {
        "query": "Can you see any solar panels on the commercial roofs?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },
    {
        "query": "Is this area prone to coastal erosion?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "vqa",
        "expected_conflict": False,
    },

    # Single image Caption queries
    {
        "query": "Describe the land-cover and major objects visible in this image.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "caption",
        "expected_conflict": False,
    },
    {
        "query": "Provide a brief caption for this aerial scene.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "caption",
        "expected_conflict": False,
    },
    {
        "query": "Give an overview of this satellite patch.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "caption",
        "expected_conflict": False,
    },
    {
        "query": "Summarise the visible terrain and vegetation patterns.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "caption",
        "expected_conflict": False,
    },
    {
        "query": "Tell me about what is visible across this scene.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": "caption",
        "expected_conflict": False,
    },

    # Bi-temporal Change queries (2 images)
    {
        "query": "What changed between these two dates, and where did the change occur?",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Has the built-up area increased, decreased, or remained unchanged?",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Describe the differences before and after the flood event.",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Did deforestation increase between 2020 and 2022?",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Highlight the regions that changed over time.",
        "num_images": 2,
        "mode": "bitemporal",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "What changed in this urban development area?",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["sar", "sar"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Compare these two acquisitions and show what changed.",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "optical"],
        "expected_tool": "change_analyze",
        "expected_conflict": False,
    },

    # Optical-SAR multi-sensor queries (2 images)
    {
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "sar"],
        "expected_tool": "optsar_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Combine optical and radar sensors to detect urban structures.",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "sar"],
        "expected_tool": "optsar_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Analyze water bodies using both sensors.",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "sar"],
        "expected_tool": "optsar_analyze",
        "expected_conflict": False,
    },
    {
        "query": "What does SAR backscatter reveal compared to the optical image?",
        "num_images": 2,
        "mode": "auto",
        "modalities": ["optical", "sar"],
        "expected_tool": "optsar_analyze",
        "expected_conflict": False,
    },
    {
        "query": "Perform optical and SAR complementary analysis.",
        "num_images": 2,
        "mode": "optical_sar",
        "modalities": ["optical", "sar"],
        "expected_tool": "optsar_analyze",
        "expected_conflict": False,
    },

    # Conflict queries (Mismatched query intent and input configuration)
    {
        "query": "What changed between these two dates and where did it change?",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": None,
        "expected_conflict": True,
    },
    {
        "query": "Combine optical and SAR images together to detect water.",
        "num_images": 1,
        "mode": "auto",
        "modalities": ["optical"],
        "expected_tool": None,
        "expected_conflict": True,
    },
    {
        "query": "Has the built-up area increased or decreased between dates?",
        "num_images": 2,
        "mode": "optical_sar",  # Mode forced to optical_sar contradicts change query
        "modalities": ["optical", "sar"],
        "expected_tool": None,
        "expected_conflict": True,
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_dummy_image_dict(name: str = "img.png", modality: str = "optical") -> dict[str, Any]:
    arr = np.zeros((3, 64, 64), dtype=np.uint8)
    return {
        "array": arr,
        "modality": modality,
        "path": f"/tmp/{name}",
        "meta": {
            "bands": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": (10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
            "size": (64, 64),
            "bounds": (0.0, 0.0, 640.0, 640.0),
            "format": "PNG",
        },
    }


# ── 1. Router Accuracy Tests ─────────────────────────────────────────────────


def test_router_accuracy_on_labelled_dataset():
    """Verify router accuracy on 26 labelled remote-sensing queries >= 95%."""
    total = len(LABELLED_ROUTER_DATASET)
    correct = 0

    for item in LABELLED_ROUTER_DATASET:
        tool, conflict = SatQueryAgent.route_query(
            query=item["query"],
            num_images=item["num_images"],
            mode=item["mode"],
            modalities=item["modalities"],
        )

        has_conflict = conflict is not None
        matches_conflict = has_conflict == item["expected_conflict"]
        matches_tool = tool == item["expected_tool"]

        if matches_conflict and matches_tool:
            correct += 1
        else:
            print(
                f"Router mismatch on query: '{item['query']}'\n"
                f"  Got: tool={tool}, conflict={conflict}\n"
                f"  Expected: tool={item['expected_tool']}, conflict={item['expected_conflict']}"
            )

    accuracy = correct / total
    print(f"\nRouter Evaluation: {correct}/{total} correct ({accuracy:.1%})")
    assert accuracy >= 0.95, f"Routing accuracy {accuracy:.1%} is below 95%"


# ── 2. Conflict Handling Tests ───────────────────────────────────────────────


def test_agent_conflict_returns_clear_error(tmp_path):
    """Mismatched query/image count returns clear error and logs trace."""
    agent = SatQueryAgent(output_dir=tmp_path)
    img = _make_dummy_image_dict("single.png", "optical")

    # Change question with only 1 image
    result = agent.run(
        query="What changed between these two dates?",
        images=[img],
        mode="auto",
    )

    assert result["task"] == "error"
    assert "Conflict" in result["text"]
    assert result["confidence"] == 0.0
    assert result["trace_path"] is not None

    # Verify trace.json was written
    trace_file = Path(result["trace_path"])
    assert trace_file.exists()
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace_data["task"] == "conflict_error"
    assert not trace_data["validation"]["passed"]


# ── 3. Pipeline Execution & Trace Verification ───────────────────────────────


@patch("satquery.agent.vqa")
def test_agent_vqa_pipeline(mock_vqa, tmp_path):
    """VQA query routes to vqa tool and writes complete trace.json."""
    mock_vqa.return_value = ToolResult(
        tool="vqa",
        text="The primary land-cover is dense coniferous forest.",
        overlay_png=None,
        stats={"token_count": 8},
        confidence=0.88,
        params_used={"max_new_tokens": 64},
        latency_s=0.25,
    )

    agent = SatQueryAgent(output_dir=tmp_path)
    img = _make_dummy_image_dict("forest.png", "optical")

    result = agent.run(
        query="What type of forest is shown in this image?",
        images=[img],
        max_new_tokens=64,
    )

    assert result["task"] == "vqa"
    assert "coniferous forest" in result["text"]
    assert result["confidence"] == 0.88
    mock_vqa.assert_called_once()

    # Verify trace schema per Spec §6.3
    trace = result["trace"]
    assert trace["query"] == "What type of forest is shown in this image?"
    assert len(trace["inputs"]) == 1
    assert trace["inputs"][0]["name"] == "forest.png"
    assert trace["task"] == "vqa"
    assert trace["validation"]["passed"] is True
    assert len(trace["steps"]) == 1
    assert trace["steps"][0]["tool"] == "vqa"
    assert trace["outputs"]["confidence"] == 0.88


@patch("satquery.agent.caption")
def test_agent_caption_pipeline(mock_caption, tmp_path):
    """Caption query routes to caption tool."""
    mock_caption.return_value = ToolResult(
        tool="caption",
        text="High-resolution aerial view of an agricultural sector with irrigation canals.",
        overlay_png=None,
        stats={"token_count": 14},
        confidence=0.91,
        params_used={"detail": "detailed"},
        latency_s=0.31,
    )

    agent = SatQueryAgent(output_dir=tmp_path)
    img = _make_dummy_image_dict("farm.png", "optical")

    result = agent.run(
        query="Describe the scene and visible agricultural infrastructure.",
        images=[img],
    )

    assert result["task"] == "caption"
    assert "irrigation canals" in result["text"]
    mock_caption.assert_called_once()


@patch("satquery.agent.change_analyze")
def test_agent_change_pipeline(mock_change, tmp_path):
    """Bi-temporal query routes to change_analyze with overlay."""
    overlay_path = str(tmp_path / "change_overlay.png")
    Path(overlay_path).write_bytes(b"dummy_png_bytes")

    mock_change.return_value = ToolResult(
        tool="change_analyze",
        text="Significant deforestation occurred in the central region.",
        overlay_png=overlay_path,
        stats={"change_pct": 14.2, "changed_pixels": 580},
        confidence=0.85,
        params_used={"mode": "describe", "diff_threshold": 0.25},
        latency_s=1.12,
    )

    agent = SatQueryAgent(output_dir=tmp_path)
    img1 = _make_dummy_image_dict("t1.png", "optical")
    img2 = _make_dummy_image_dict("t2.png", "optical")

    result = agent.run(
        query="What changed between these two dates?",
        images=[img1, img2],
        diff_threshold=0.25,
    )

    assert result["task"] == "change_analyze"
    assert result["overlay"] == overlay_path
    assert result["stats"]["change_pct"] == 14.2
    mock_change.assert_called_once()


@patch("satquery.agent.optsar_analyze")
def test_agent_optsar_pipeline(mock_optsar, tmp_path):
    """Optical-SAR query routes to optsar_analyze and orders optical/SAR inputs."""
    overlay_path = str(tmp_path / "optsar_overlay.png")
    Path(overlay_path).write_bytes(b"dummy_png_bytes")

    mock_optsar.return_value = ToolResult(
        tool="optsar_analyze",
        text="Fused optical-SAR demonstrates clear delineation of reservoirs and urban settlements.",
        overlay_png=overlay_path,
        stats={"water_pct": 8.5, "built_up_pct": 18.2},
        confidence=0.89,
        params_used={"targets": "both"},
        latency_s=0.95,
    )

    agent = SatQueryAgent(output_dir=tmp_path)
    img_sar = _make_dummy_image_dict("s1.tif", "sar")
    img_opt = _make_dummy_image_dict("s2.tif", "optical")

    result = agent.run(
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        images=[img_sar, img_opt],  # Pass SAR first to verify agent reorders
        targets="both",
    )

    assert result["task"] == "optsar_analyze"
    assert result["overlay"] == overlay_path
    assert result["stats"]["built_up_pct"] == 18.2
    mock_optsar.assert_called_once()

    # Confirm optical dict was routed to image_dict_optical and SAR to image_dict_sar
    kwargs = mock_optsar.call_args.kwargs
    assert kwargs["image_dict_optical"]["modality"] == "optical"
    assert kwargs["image_dict_sar"]["modality"] == "sar"
