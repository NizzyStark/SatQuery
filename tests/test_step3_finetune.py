"""Unit tests for Step 3: BigEarthNet dataset preparation and QLoRA fine-tuning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from data.prepare_bigearthnet_txt import (
    BIGEARTHNET_19_CLASSES,
    CLASS_COLOR_MAP,
    prepare_dataset,
)
from training.finetune_lora import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    MAX_IMAGE_PIXELS,
    TARGET_LINEAR_MODULES,
    FinetuneConfig,
    evaluate_after_finetune,
    train,
)


def test_bigearthnet_schema_constants():
    """Verify BigEarthNet 19-class schema constants and color mappings."""
    assert len(BIGEARTHNET_19_CLASSES) >= 19
    assert "Urban fabric" in BIGEARTHNET_19_CLASSES
    assert "Inland waters" in BIGEARTHNET_19_CLASSES
    assert "Broad-leaved forest" in BIGEARTHNET_19_CLASSES
    assert "Arable land" in BIGEARTHNET_19_CLASSES

    for cls_name in BIGEARTHNET_19_CLASSES:
        assert cls_name in CLASS_COLOR_MAP
        color = CLASS_COLOR_MAP[cls_name]
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_data_preparation_chat_format(tmp_path: Path):
    """Verify data preparation outputs valid chat-format samples."""
    out_jsonl = tmp_path / "test_chat.jsonl"
    images_dir = tmp_path / "images"

    stats = prepare_dataset(
        data_dir=None,
        output_path=str(out_jsonl),
        max_samples=30,
        images_dir=str(images_dir),
        seed=123,
    )

    assert out_jsonl.exists()
    assert stats["total_samples"] == 30
    assert len(stats["task_distribution"]) > 0

    # Read and inspect samples
    with open(out_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 30
    for s in lines:
        assert "id" in s
        assert "patch_id" in s
        assert "image" in s
        assert "task_type" in s
        assert "labels" in s
        assert "messages" in s
        assert "conversations" in s

        # Verify messages format
        assert len(s["messages"]) == 2
        assert s["messages"][0]["role"] == "user"
        assert s["messages"][1]["role"] == "assistant"

        # Verify conversations format
        assert len(s["conversations"]) == 2
        assert s["conversations"][0]["from"] == "human"
        assert s["conversations"][1]["from"] == "gpt"

        # Verify image file exists
        img_path = Path(s["image"])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        assert img_path.exists(), f"Image {img_path} does not exist"


def test_finetune_config_specification_compliance():
    """Verify FinetuneConfig strictly complies with Step 3 specifications."""
    cfg = FinetuneConfig()
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert set(cfg.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert cfg.max_image_pixels == 448 * 448
    assert cfg.gradient_checkpointing is True


def test_finetune_small_run_and_artifacts(tmp_path: Path):
    """Run a small fine-tuning job and verify generated LoRA adapter and logs."""
    # Prepare mini dataset
    data_file = tmp_path / "mini_chat.jsonl"
    prepare_dataset(output_path=str(data_file), max_samples=10, images_dir=str(tmp_path / "imgs"))

    adapter_out = tmp_path / "test_lora_adapter"
    log_file = tmp_path / "test_training_log.json"

    cfg = FinetuneConfig(
        train_file=str(data_file),
        output_dir=str(adapter_out),
        log_file=str(log_file),
        epochs=1,
        batch_size=2,
        max_samples=10,
    )

    log = train(cfg)

    # 1. Check training log
    assert log["status"] == "COMPLETED"
    assert log_file.exists()
    assert log["total_steps"] > 0
    assert len(log["history"]) > 0

    # 2. Check saved adapter
    assert adapter_out.exists()
    adapter_cfg_file = adapter_out / "adapter_config.json"
    adapter_weights_file = adapter_out / "adapter_model.safetensors"
    readme_file = adapter_out / "README.md"

    assert adapter_cfg_file.exists()
    assert adapter_weights_file.exists()
    assert readme_file.exists()

    with open(adapter_cfg_file, "r", encoding="utf-8") as f:
        acfg = json.load(f)
    assert acfg.get("r") == 16
    assert acfg.get("lora_alpha") == 32


def test_vlm_client_with_adapter(monkeypatch):
    """Verify VLMClient recognizes the fine-tuned adapter and applies domain adaptation."""
    monkeypatch.setenv("SATQUERY_MOCK_VLM", "1")
    from satquery.vlm import VLMClient

    # Baseline client without adapter
    client_base = VLMClient(adapter_path=None)
    assert not client_base.has_adapter

    # Fine-tuned client with adapter
    client_adapted = VLMClient(adapter_path="models/lora_adapter")
    assert client_adapted.has_adapter

    # Run query through adapted client
    dummy_img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    res_base = client_base.generate([dummy_img], prompt="Describe land-cover classes.")
    res_adapted = client_adapted.generate([dummy_img], prompt="Describe land-cover classes.")

    assert res_base["confidence"] > 0
    assert res_adapted["confidence"] > 0
    # Adapted response references BigEarthNet domain concepts
    assert "BigEarthNet" in res_adapted["text"] or "CORINE" in res_adapted["text"]


def test_after_finetune_file_exists_and_matches_schema():
    """Verify docs/after_finetune.json matches baseline structure and reflects adaptation."""
    after_path = Path("docs/after_finetune.json")
    baseline_path = Path("docs/baseline.json")

    assert after_path.exists(), "docs/after_finetune.json was not generated."
    assert baseline_path.exists(), "docs/baseline.json does not exist."

    with open(after_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    for key in ("vqa", "caption_brief", "caption_detailed"):
        assert key in after
        assert key in baseline
        assert "confidence" in after[key]
        assert after[key]["confidence"] >= 0.0
        assert after[key]["latency_s"] >= 0.0

    assert "_adapter_metadata" in after
    assert after["_adapter_metadata"]["fine_tuned"] is True
