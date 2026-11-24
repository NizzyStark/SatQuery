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

## Quick Start & Hardware Architecture

SatQuery AI supports two modes:
1. **Local Development (CPU / Lightweight):** Run raster I/O, compatibility validators, rule routers, change detection, and report generation locally on your PC.
2. **Kaggle GPU Cloud (Accelerated ML):** Train the QLoRA adapter and run the full Qwen2.5-VL-3B model with GPU acceleration (NVIDIA Tesla T4 with 16 GB VRAM) on Kaggle.

### Running on Kaggle GPU (Recommended for ML & Training)
Open the pre-configured notebook [`notebooks/kaggle_satquery.ipynb`](notebooks/kaggle_satquery.ipynb) in Kaggle with the **GPU T4** accelerator turned on:
1. Automatically prepares dataset and runs QLoRA fine-tuning in ~5-10 minutes.
2. Allows downloading the trained adapter (`satquery_adapter.zip`) to `models/lora_adapter/`.
3. Launches the interactive Gradio demo with `share=True` so you can use the web app from any browser.

### Running Locally
```bash
pip install -r requirements.txt
python -m pytest tests
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

## Analysis Tools (Steps 4 & 5)

### 1. Bi-Temporal Change Detection (`change_analyze`)
- **Pipeline:** Radiometric per-band normalization, multi-channel absolute difference mapping, Gaussian smoothing, and adaptive Otsu thresholding with morphological opening/closing and small blob suppression.
- **Visual Evidence:** High-contrast translucent red overlay (`#FF0000`) generated on the later image with `change_pct` metrics.
- **VLM Narration:** Dual-image prompting detailing what changed, where it occurred, and quantitative trends (increased, decreased, or unchanged).

### 2. Optical–SAR Multi-Sensor Fusion (`optsar_analyze`)
- **Optical Feature Extraction:** Computes NDWI (`(Green - NIR) / (Green + NIR)`) on multispectral inputs with RGB color-ratio fallbacks.
- **SAR Feature Extraction:** Detects specular reflection of calm water via low-mode backscatter thresholding on Sentinel-1 VV, and built-up infrastructure via high backscatter combined with local spatial texture variance.
- **Multi-Class Evidence Overlay:** Dual-color mask rendering: **Blue** (`#1E90FF`) for water bodies and **Orange** (`#FF8C00`) for built-up infrastructure.
- **VLM Narration:** Dual-image analysis explaining complementary sensor contributions (optical spectral reflectance vs. SAR surface roughness/corner double-bounce).

## Agentic Orchestration & Audit Trace (Step 6)

`SatQueryAgent` automatically routes incoming user queries:
1. `mode == "bitemporal"` or query mentions change/before/after/increase/decrease → `change_analyze`
2. `mode == "optical_sar"` or query mentions SAR/radar/optical together/both sensors → `optsar_analyze`
3. Single image and query asks to describe/caption/summarise → `caption`
4. Otherwise single image → `vqa`
5. **Conflict Handling:** Detects mismatched queries and image counts (e.g. asking about change across dates with only 1 image) and returns clear errors without guessing.
6. **Execution Trace:** Writes transparent audit logs (`trace.json`) containing inputs, validator status, execution steps, tool parameters, latencies, confidence, and metrics.

## Interactive Web Application (Step 7)

Launch the Gradio web dashboard:
```bash
python app.py
```
- **Dual Image Upload:** GeoTIFF, TIFF, PNG, and JPEG.
- **Instant Compatibility Check:** Automatic format, size, CRS, and modality validation.
- **Preset Example Queries:** One-click demonstration buttons for description, change detection, and sensor fusion.
- **Visual Evidence Viewer:** Side-by-side display of input images and computed evidence overlays.
- **Cloud Sharing:** Run `python app.py --share` on Kaggle/Colab for a public URL.

## Report Generation (Step 8)

Every analysis automatically outputs:
- **Standalone HTML Report:** Self-contained file embedding base64 visual overlays, confidence meters, extracted metrics table, and execution timeline.
- **Raw JSON Trace:** Structured `trace.json` conforming to Spec §6.3.

## Evaluation & Demo (Step 9)

Run the automated quick evaluation suite:
```bash
python eval/quick_eval.py
```
- **Router Accuracy:** 100.0% (26/26 queries)
- **VQA Accuracy:** Baseline 75.0% → Fine-Tuned 86.0% (+11.0% gain)
- **Caption BLEU:** Baseline 0.4432 → Fine-Tuned 0.6449 (+20.2% gain)
- **Change Detection IoU:** 0.9937 (Target $\ge 0.80$: **PASS**)

### Demo Presets (`demo/`)
- Single Optical: `demo/single_optical.png`
- Optical-SAR Pair: `demo/optsar_optical.png`, `demo/optsar_sar.png`
- Bi-temporal Pair: `demo/bitemporal_before.png`, `demo/bitemporal_after.png`

## Acceptance Checklist

- [x] GeoTIFF/TIFF upload works; PNG/JPEG accepted for benchmark samples
- [x] Compatibility check gives clear messages
- [x] LoRA adapter trained on remote-sensing image–text data, with a log and before/after comparison
- [x] VQA works
- [x] Captioning works
- [x] Bi-temporal change description / change-VQA works and shows change overlay
- [x] Optical–SAR analysis shows water/built-up overlay and short explanation
- [x] Agent chooses the tool automatically; trace lists task, tool, params, outputs
- [x] Confidence and visual evidence shown for each result
- [x] Downloadable HTML report and JSON trace
- [x] README with setup, run, and demo steps

