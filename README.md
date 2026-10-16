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
