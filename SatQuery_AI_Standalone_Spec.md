# SatQuery AI — Standalone Prototype Spec

This document is self-contained. It describes the problem, the prototype scope, the design, and the build steps. No other file is needed.

**Instruction to the coding agent:** build the prototype step by step in the order of section 9. After each step, run its tests and confirm they pass before continuing. Verify every model name, checkpoint, dataset link, and licence before using it. Do not invent APIs; read the library docs first.

---

## 1. Problem summary

Build **SatQuery AI**, an agentic vision-language assistant that answers natural-language queries about remote-sensing imagery. Inputs can be:

- **Single image:** one optical/multispectral or SAR image.
- **Optical–SAR pair:** co-registered images of the same area.
- **Bi-temporal pair:** two images of the same area at different dates.

Supported formats: GeoTIFF/TIFF for real data; PNG/JPEG only for public benchmark samples.

Required capabilities (all must be demonstrable):

1. **Remote-sensing adaptation:** at least one vision or vision-language component is fine-tuned on remote-sensing data (BigEarthNet.txt or another open dataset). A generic, un-adapted VLM does not qualify.
2. **Single-image VQA** (mandatory) plus **one more single-image task** (captioning or text-guided grounding).
3. **Change analysis:** change description or change-based VQA from a bi-temporal pair. A change map is a bonus.
4. **Optical–SAR analysis:** extract complementary information from a co-registered pair.
5. **Agentic orchestration:** the system automatically selects and runs the right tool from a predefined registry, with permitted parameters only, and produces an observable execution trace (task, tools, parameters, outputs).
6. **GUI/web app** with upload, compatibility check, visual evidence, confidence, execution summary, and a downloadable report.

Example queries the demo must handle:

- "Describe the land-cover and major objects visible in this image."
- "What changed between these two dates, and where did the change occur?"
- "Use the optical and SAR images together to identify built-up and water-covered regions."
- "Has the built-up area increased, decreased, or remained unchanged?"
- "Highlight the water body referred to in the query." *(optional in this prototype)*

Evaluation (later, by organisers): public benchmark test subsets (RSVQA, VRSBench, CDVQA) and a held-out GeoTIFF dataset. Metrics are normalised and combined.

---

## 2. Prototype scope

This is a **prototype**, not the full system. Keep it small and working.

### In scope

| Feature | Approach |
|---|---|
| Adaptation | Short QLoRA fine-tune of one VLM on a small BigEarthNet.txt subset |
| VQA | Fine-tuned VLM |
| Captioning (the extra single-image task) | Same VLM, caption prompt |
| Change analysis | Classical difference map + VLM description / change-VQA with both images |
| Optical–SAR analysis | NDWI + SAR backscatter threshold + simple fusion; VLM narrates the stats |
| Agent | Rule-based router, tool registry, param validation, JSON trace |
| GUI | Gradio |
| Report | HTML or JSON download |

### Out of scope (do not build)

- Text-guided grounding and SAM
- Separate controller LLM
- Pretrained change-detection networks
- Optical–SAR foundation encoders (DOFA, CROMA, etc.)
- PDF reports
- Full benchmark runs (use small subsets)

---

## 3. Models

| Role | Model |
|---|---|
| The only neural model | **Qwen2.5-VL-3B-Instruct** (4-bit, LoRA adapter). Use 7B only if the GPU has 24 GB or more |
| Fallbacks if it fails to load | PaliGemma 2 (3B), InternVL3-2B |

Everything else is classical image processing (NumPy, rasterio, scikit-image, OpenCV). Optional references for prompt and data formats only (do not integrate): GeoChat, EarthDial.

---

## 4. Architecture

```
Gradio UI ──▶ Agent: router → validator → executor → fuser → trace
                         │
        ┌────────────────┼───────────────────────┐
        ▼                ▼                       ▼
   vqa / caption    change_analyze         optsar_analyze
   (VLM + LoRA)     (diff map + VLM)       (NDWI + SAR thr + VLM narration)
```

---

## 5. Project structure

```
satquery-proto/
├── README.md
├── requirements.txt
├── configs/registry.yaml
├── data/prepare_bigearthnet_txt.py
├── training/finetune_lora.py
├── satquery/
│   ├── io.py            # loading + rendering
│   ├── validator.py     # compatibility checks
│   ├── vlm.py           # model wrapper
│   ├── tools.py         # four tools
│   ├── agent.py         # router, executor, fuser, trace
│   └── report.py        # HTML/JSON export
├── app.py               # Gradio UI
├── eval/quick_eval.py
├── tests/
└── demo/                # 1 single image, 1 optical–SAR pair, 1 bi-temporal pair
```

---

## 6. Core interfaces

### 6.1 Tool registry (`configs/registry.yaml`)

```yaml
tools:
  vqa:
    inputs: single_image
    params: {max_new_tokens: [16, 128]}
  caption:
    inputs: single_image
    params: {detail: [brief, detailed]}
  change_analyze:
    inputs: bitemporal
    params: {mode: [describe, vqa], diff_threshold: [0.1, 0.5]}
  optsar_analyze:
    inputs: optical_sar
    params: {targets: [built_up, water, both]}
```

Parameters outside the listed ranges or values must be rejected or clipped, and the clip logged in the trace.

### 6.2 Tool result (every tool returns this)

```python
@dataclass
class ToolResult:
    tool: str
    text: str
    overlay_png: str | None   # path to evidence image
    stats: dict               # e.g. {"change_pct": 12.4, "water_pct": 8.1}
    confidence: float         # 0..1
    params_used: dict
    latency_s: float
```

### 6.3 Trace (`trace.json`)

```json
{
  "query": "...",
  "inputs": [{"name": "a.tif", "modality": "optical", "size": [512, 512]}],
  "task": "change_analyze",
  "validation": {"passed": true, "messages": []},
  "steps": [{"tool": "change_analyze", "params": {"mode": "describe"}, "latency_s": 3.2}],
  "outputs": {"text": "...", "overlay": "...", "confidence": 0.74}
}
```

---

## 7. Input handling

- **GeoTIFF/TIFF:** rasterio. Read band count, dtype, CRS, transform, size.
- **PNG/JPEG:** Pillow; treat as RGB optical unless the user tags it as SAR.
- **Modality detection:** user tag first; otherwise heuristics (e.g., 1–2 bands with negative dB-like values → SAR; 3+ bands → optical/multispectral).
- **Rendering for the VLM (always 3-channel 8-bit):**
  - Sentinel-2 true colour: B4, B3, B2, percentile stretch (2–98).
  - Sentinel-1: R=VV, G=VH, B=VV−VH or VV/VH in dB, percentile stretch.
- **Keep raw arrays** for index and threshold calculations.
- **Validator checks:** image count matches mode; both images same size (else resize with a warning); CRS and extent match for GeoTIFF pairs (else warn); optical–SAR mode needs one optical and one SAR; bi-temporal mode needs two images of the same modality. Return `{passed, messages}` with clear text.

---

## 8. Tool logic

### 8.1 `vqa` and `caption`
Send the rendered image and a prompt to the VLM. Confidence = mean token probability of the generated answer.

### 8.2 `change_analyze`
1. Resize/align the pair to the same grid.
2. Normalise each image per band; compute absolute difference, then a single channel (mean across bands).
3. Gaussian blur, threshold (Otsu, or `diff_threshold` if set), morphological open/close, remove tiny blobs.
4. Overlay the change mask in red on the later image. Report `change_pct`.
5. Prompt the VLM with both images: "Image 1 is the earlier image and Image 2 is the later image. Describe what changed and where." In `vqa` mode, append the user question.
6. Answer rules for "increased/decreased/unchanged" questions: use the index-based area statistic for the target class in each image (e.g., built-up proxy or water index) and compare with a small tolerance; let the VLM explain.

### 8.3 `optsar_analyze`
1. **Optical water:** NDWI = (Green − NIR) / (Green + NIR) when NIR exists; otherwise a colour-based fallback and a lower confidence.
2. **SAR water:** low backscatter (Otsu threshold on dB VV).
3. **SAR built-up:** high backscatter (upper percentile) combined with local texture.
4. **Optical built-up:** NDBI if SWIR exists; otherwise brightness/texture fallback.
5. **Fusion:** water = both agree (high confidence), one only (medium). Built-up similarly.
6. Overlay: blue = water, orange = built-up. Report area percentages.
7. VLM narration: give it both rendered images and the stats and ask for a short summary of what each sensor contributes.

---

## 9. Build steps (in order)

### Step 1 — Scaffold, I/O, validator
Create the structure, `requirements.txt` (pinned), registry, `io.py`, `validator.py`.
*Tests:* synthetic rasters for wrong count, size mismatch, CRS mismatch, SAR vs optical detection.

### Step 2 — Baseline VLM
Implement `vlm.py`: load Qwen2.5-VL-3B (4-bit), `generate(images, prompt, adapter=None)` supporting 1–2 images, returning text and mean token probability. Implement `vqa` and `caption` tools.
*Tests:* a smoke test on a sample image; save baseline outputs to `docs/baseline.json`.

### Step 3 — Fine-tune
1. Inspect the actual BigEarthNet.txt schema before writing any loader. If it cannot be obtained, use BigEarthNet-MM or another open remote-sensing image–text set and state that in the README.
2. `data/prepare_bigearthnet_txt.py`: build 5k–20k chat-format samples (caption, scene description, simple label-derived VQA like "Is there water?"). Optionally add ~2k samples from VRSBench/RSVQA.
3. `training/finetune_lora.py`: QLoRA (r=16, alpha=32, dropout 0.05, all linear layers of the LLM), 1–2 epochs, images capped near 448 px, bf16, gradient checkpointing. Use LLaMA-Factory, ms-swift, or TRL+PEFT.
4. Save the adapter and the training log. Re-run the baseline prompts with the adapter and store `docs/after_finetune.json`.

*Run training in the background while building later steps.*

### Step 4 — Change tool
Implement per 8.2. *Tests:* synthetic pair with a known changed square; assert the mask overlaps it with IoU above 0.8.

### Step 5 — Optical–SAR tool
Implement per 8.3. *Tests:* synthetic optical/SAR arrays with a known water region; assert detection and correct fusion label.

### Step 6 — Agent
`agent.py`:

- **Router rules (checked in order):**
  1. Input mode is bi-temporal, or query mentions change/changed/before/after/between dates/increase/decrease → `change_analyze`.
  2. Input mode is optical–SAR, or query mentions SAR/radar/optical together/both sensors → `optsar_analyze`.
  3. Single image and query asks to describe/caption/summarise → `caption`.
  4. Otherwise single image → `vqa`.
- If the query and input configuration conflict (e.g., a change question with one image), return a clear error, not a guess.
- **Executor:** validate params against the registry, run the tool, catch exceptions.
- **Fuser:** assemble text, overlay, stats, confidence. Confidence = VLM token probability, raised slightly when the index/threshold evidence agrees with the answer, lowered when it conflicts or when fallbacks were used.
- **Trace:** write `trace.json` as in 6.3.
- *Optional stretch:* when no rule matches, ask the VLM to output exactly one tool name from the registry.

*Tests:* 20–30 labelled queries; report routing accuracy.

### Step 7 — GUI
`app.py` (Gradio):

- Upload 1–2 files; mode dropdown (Auto, Single, Optical–SAR, Bi-temporal); optional per-file tags (modality, date).
- Show the compatibility check immediately after upload.
- Query box with the example queries as buttons.
- Output: answer text, overlay image, confidence, execution summary (from the trace), and a button to download the report.

### Step 8 — Report
`report.py`: produce a single HTML file with inputs, query, answer, overlay, confidence, and trace; also save the raw JSON.

### Step 9 — Quick evaluation and demo
- `eval/quick_eval.py`: 100–200 test samples from RSVQA and VRSBench (whatever is available); compare baseline vs fine-tuned accuracy (VQA) and BLEU/ROUGE or CIDEr (captioning).
- A small CDVQA sample run through the change tool.
- Router test from step 6.
- Put three demo inputs in `demo/` and write the README with setup, run, and demo instructions.

---

## 10. Acceptance checklist

- [ ] GeoTIFF/TIFF upload works; PNG/JPEG accepted for benchmark samples
- [ ] Compatibility check gives clear messages
- [ ] LoRA adapter trained on remote-sensing image–text data, with a log and a before/after comparison
- [ ] VQA works
- [ ] Captioning works
- [ ] Bi-temporal change description / change-VQA works and shows a change overlay
- [ ] Optical–SAR analysis shows water/built-up overlay and a short explanation
- [ ] Agent chooses the tool automatically; trace lists task, tool, params, outputs
- [ ] Confidence and visual evidence shown for each result
- [ ] Downloadable report
- [ ] README with setup and demo steps

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| GPU too small | Use the 3B model, 4-bit, lower image resolution, fewer samples; or train on Colab/Kaggle |
| BigEarthNet.txt format differs from expectations | Write the loader against the real schema; fall back to BigEarthNet-MM with label-derived captions |
| VLM mishandles two-image prompts | Concatenate the two images side by side into one image with labels "Before" and "After" |
| Missing NIR/SWIR bands | Use colour-based fallbacks and lower the confidence, noting it in the trace |
| Time pressure | Drop in this order: HTML report polish, router stretch, caption polish. Never drop the fine-tune, change tool, optical–SAR tool, or the trace |

---

## 12. Prompts to give the coding agent (one at a time)

1. "Read `SatQuery_AI_Standalone_Spec.md`. Do Step 1: create the scaffold, pinned `requirements.txt`, `configs/registry.yaml`, `satquery/io.py`, and `satquery/validator.py`, with tests."
2. "Do Step 2: implement `satquery/vlm.py` (Qwen2.5-VL-3B, 4-bit, optional LoRA adapter, 1–2 images, token probability) and the `vqa` and `caption` tools, with a CLI and a smoke test."
3. "Do Step 3: inspect the BigEarthNet.txt schema, write the data preparation script and the QLoRA training script, and run a small training job. Save the adapter, logs, and before/after outputs."
4. "Do Steps 4 and 5: implement `change_analyze` and `optsar_analyze` in `satquery/tools.py` with overlays, stats, confidence, and the synthetic-data tests."
5. "Do Steps 6 to 8: implement the agent (router, executor, fuser, trace), the Gradio app, and the HTML/JSON report."
6. "Do Step 9: write the quick evaluation scripts, the router test set, the demo inputs, and the README. Then walk through the acceptance checklist and list anything missing."
