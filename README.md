# SatQuery AI

SatQuery AI is an agentic vision-language assistant for remote-sensing imagery analysis.

## Overview
SatQuery AI processes optical, multispectral, and Synthetic Aperture Radar (SAR) Earth observation data. It supports:
- Single-image Visual Question Answering (VQA)
- Remote-sensing captioning
- Bi-temporal change analysis (description, change VQA, change overlay)
- Optical–SAR multi-sensor fusion and analysis
- Automated tool routing with validation and traceable execution

## Project Structure
```
satquery/
├── configs/registry.yaml              # Tool and parameter definitions
├── data/prepare_bigearthnet_txt.py    # Remote-sensing VLM dataset preparation
├── training/finetune_lora.py          # QLoRA fine-tuning for vision-language models
├── satquery/
│   ├── __init__.py
│   ├── io.py                          # GeoTIFF / PNG loading and VLM rendering
│   ├── validator.py                   # Input compatibility & parameter validator
│   ├── vlm.py                         # Vision-language model wrapper
│   ├── tools.py                       # VQA, caption, change, and optical-SAR tools
│   ├── agent.py                       # Router, executor, fuser, and trace logger
│   └── report.py                      # HTML / JSON report generator
├── app.py                             # Gradio web interface
├── eval/quick_eval.py                 # Evaluation pipeline
├── tests/                             # Unit tests with synthetic data
└── demo/                              # Demo sample imagery
```

## Quick Start
```bash
pip install -r requirements.txt
pytest
python app.py
```

## Remote-Sensing VLM Adaptation (Step 3)

### 1. BigEarthNet Schema Inspection
SatQuery adapts vision-language models using the **BigEarthNet / BigEarthNet.txt** remote sensing schema:
- **Nomenclature**: 19-class harmonized CORINE Land Cover (CLC 2018) classes including *Urban fabric, Industrial/commercial units, Arable land, Broad-leaved / Coniferous / Mixed forest, Inland / Marine waters, Wetlands*.
- **Data Schema**: Co-registered multi-spectral Sentinel-2 and Sentinel-1 SAR tiles with patch metadata (`*_labels_metadata.json`), geospatial coordinates, and multi-label annotations.
- **Loader & Synthetic Fallback**: `data/prepare_bigearthnet_txt.py` loads real BigEarthNet/BigEarthNet.txt archives when available, or synthesizes realistic remote-sensing image-text pairs adhering strictly to the 19-class schema when large raw archives are not downloaded locally.

### 2. Dataset Preparation
```bash
python data/prepare_bigearthnet_txt.py --max-samples 5000 --output data/bigearthnet_chat.jsonl
```
Builds chat-format samples covering:
- Remote-sensing brief and detailed captions
- Detailed scene descriptions
- Land-cover VQA (water presence, urban infrastructure, forest canopy, agriculture)
- RSVQA / VRSBench spatial reasoning queries

### 3. QLoRA Fine-Tuning
```bash
python training/finetune_lora.py --train-file data/bigearthnet_chat.jsonl --output-dir models/lora_adapter --epochs 1
```
Fine-tuning configuration:
- **Base model**: Qwen2.5-VL-3B-Instruct (4-bit NF4 with bitsandbytes on CUDA / CPU fallback)
- **LoRA parameters**: Rank $r=16$, $\alpha=32$, dropout $0.05$
- **Target modules**: All linear layers of the LLM (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Resolution**: Images capped near 448 px (`max_pixels = 448 * 448`)
- **Memory safety**: bf16 / fp16 mixed precision and gradient checkpointing
- **Outputs**: Trained adapter saved to `models/lora_adapter/` and training log to `training/training_log.json`.

### 4. Baseline vs. Fine-Tuned Comparison
- Baseline outputs: [`docs/baseline.json`](docs/baseline.json)
- Fine-tuned adapter outputs: [`docs/after_finetune.json`](docs/after_finetune.json)
- The fine-tuned adapter demonstrates domain terminology aligned with CORINE/BigEarthNet classes, precise hydrology/infrastructure distinction, and higher confidence.

