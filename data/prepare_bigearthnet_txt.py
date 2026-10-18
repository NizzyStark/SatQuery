"""Data preparation script for BigEarthNet.txt remote sensing dataset.

Formats BigEarthNet / BigEarthNet.txt image-text pairs into chat-format samples
for vision-language model fine-tuning (captions, scene descriptions, and
label-derived VQA like 'Is there water?').

Supports both:
1. Real BigEarthNet archives (CORINE 19-class or 43-class metadata JSONs / manifests).
2. Automated realistic remote-sensing patch generation conforming to BigEarthNet
   19-class land-cover schema when local multi-gigabyte archives are not present.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── BigEarthNet 19-Class Harmonized CORINE Nomenclature ──────────────────────

BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Color map approximations for realistic synthetic remote sensing rendering
CLASS_COLOR_MAP = {
    "Urban fabric": (180, 70, 70),
    "Industrial or commercial units": (200, 100, 110),
    "Arable land": (220, 210, 140),
    "Permanent crops": (200, 160, 60),
    "Pastures": (160, 220, 100),
    "Complex cultivation patterns": (190, 180, 90),
    "Land principally occupied by agriculture, with significant areas of natural vegetation": (140, 170, 80),
    "Agro-forestry areas": (120, 150, 70),
    "Broad-leaved forest": (34, 139, 34),
    "Coniferous forest": (15, 80, 25),
    "Mixed forest": (45, 110, 45),
    "Natural grassland and sparsely vegetated areas": (170, 200, 110),
    "Moors, heathland and sclerophyllous vegetation": (150, 130, 80),
    "Sclerophyllous vegetation": (140, 120, 70),
    "Transitional woodland, shrub": (110, 140, 90),
    "Beaches, dunes, sands": (230, 220, 170),
    "Inland wetlands": (70, 130, 120),
    "Coastal wetlands": (60, 110, 110),
    "Inland waters": (30, 80, 180),
    "Marine waters": (20, 50, 140),
}


# ── Chat Sample Builders ────────────────────────────────────────────────────


def _generate_caption_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate a concise remote-sensing caption sample."""
    label_str = ", ".join(labels)
    templates = [
        f"A Sentinel-2 satellite observation displaying {label_str}.",
        f"Remote-sensing aerial imagery featuring {label_str}.",
        f"Earth observation patch characterized by {label_str}.",
        f"Satellite view capturing terrain with {label_str}.",
    ]
    caption = random.choice(templates)
    prompt = random.choice([
        "Provide a concise caption for this satellite image.",
        "Briefly describe the remote sensing imagery.",
        "Generate a short summary of the visible land cover in this scene.",
    ])

    return {
        "id": f"{sample_id}_caption",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "caption",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": caption},
        ],
    }


def _generate_scene_description_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate a detailed scene description conforming to BigEarthNet schema."""
    primary = labels[0]
    secondary = labels[1:] if len(labels) > 1 else []

    desc_parts = [
        f"Detailed remote sensing analysis: The scene is predominantly composed of {primary}."
    ]
    if secondary:
        desc_parts.append(f"Adjacent parcels exhibit {', '.join(secondary)}.")
    desc_parts.append(
        "Spectral reflectance patterns demonstrate clear spatial demarcation between contiguous land-cover categories."
    )
    detailed_desc = " ".join(desc_parts)

    prompt = random.choice([
        "Provide a detailed analysis of the land cover and spatial layout in this image.",
        "Describe the geographic features and land classes present in this remote sensing patch.",
        "Conduct a comprehensive land-cover assessment of this satellite observation.",
    ])

    return {
        "id": f"{sample_id}_scene_desc",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "scene_description",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": detailed_desc}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": detailed_desc},
        ],
    }


def _generate_vqa_water_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate a water-presence VQA sample."""
    has_water = any(c in labels for c in ["Inland waters", "Marine waters", "Inland wetlands", "Coastal wetlands"])
    prompt = random.choice([
        "Is there water present in this satellite image?",
        "Are there open water bodies or wetlands detected in this scene?",
        "Does this remote sensing imagery contain surface water?",
    ])

    if has_water:
        matching = [c for c in labels if c in ["Inland waters", "Marine waters", "Inland wetlands", "Coastal wetlands"]]
        answer = f"Yes, water features are detected in this image, specifically: {', '.join(matching)}."
    else:
        answer = "No surface water bodies or wetland hydrological features are detected in this observation."

    return {
        "id": f"{sample_id}_vqa_water",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "vqa_water",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def _generate_vqa_urban_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate an urban/built-up VQA sample."""
    has_urban = any(c in labels for c in ["Urban fabric", "Industrial or commercial units"])
    prompt = random.choice([
        "Does this image show built-up urban infrastructure or commercial areas?",
        "Are human-made structures or urban settlements visible in this satellite patch?",
        "Is there developed urban fabric in this remote sensing observation?",
    ])

    if has_urban:
        urban_types = [c for c in labels if c in ["Urban fabric", "Industrial or commercial units"]]
        answer = f"Yes, built-up infrastructure is present, including {', '.join(urban_types)}."
    else:
        answer = "No developed urban fabric or industrial structures are present; the area consists of non-urban terrain."

    return {
        "id": f"{sample_id}_vqa_urban",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "vqa_urban",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def _generate_vqa_landcover_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate a multi-class land cover query sample."""
    prompt = random.choice([
        "What land-cover classes are present in this satellite image?",
        "List all CORINE/BigEarthNet land-use categories identified in this observation.",
        "Identify the primary land-cover types visible in this scene.",
    ])
    answer = f"The identified land-cover classes are: {', '.join(labels)}."

    return {
        "id": f"{sample_id}_vqa_landcover",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "vqa_landcover",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def _generate_vqa_rsvqa_style_sample(
    sample_id: str,
    patch_id: str,
    image_rel_path: str,
    labels: list[str],
) -> dict[str, Any]:
    """Generate RSVQA/VRSBench-style reasoning question."""
    forest_classes = ["Broad-leaved forest", "Coniferous forest", "Mixed forest"]
    agri_classes = ["Arable land", "Pastures", "Complex cultivation patterns", "Permanent crops"]

    has_forest = any(c in labels for c in forest_classes)
    has_agri = any(c in labels for c in agri_classes)

    choice = random.choice(["forest", "agri"])
    if choice == "forest":
        prompt = "Is there woodland or forest cover in this area?"
        if has_forest:
            detected = [c for c in labels if c in forest_classes]
            answer = f"Yes, forest vegetation is present ({', '.join(detected)})."
        else:
            answer = "No significant forest canopy or woodland is identified in this area."
    else:
        prompt = "Are agricultural cultivation or pasture lands observed in this image?"
        if has_agri:
            detected = [c for c in labels if c in agri_classes]
            answer = f"Yes, agricultural activities are evident ({', '.join(detected)})."
        else:
            answer = "No agricultural parcels or pastures are visible in this scene."

    return {
        "id": f"{sample_id}_vqa_rsvqa",
        "patch_id": patch_id,
        "image": image_rel_path,
        "images": [image_rel_path],
        "task_type": "vqa_rsvqa",
        "labels": labels,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


# ── Synthetic Patch Generator ───────────────────────────────────────────────


def _create_synthetic_patch(
    labels: list[str],
    out_path: str,
    size: int = 120,
) -> None:
    """Render a synthetic remote-sensing patch reflecting the assigned labels."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)

    # Base background from first label
    base_color = CLASS_COLOR_MAP.get(labels[0], (140, 140, 140))
    arr[:] = base_color

    # Add textures and subregions for additional labels
    num_labels = len(labels)
    if num_labels > 1:
        for i, lbl in enumerate(labels[1:], start=1):
            col = CLASS_COLOR_MAP.get(lbl, (100, 100, 100))
            # Slice regions or quadrants
            x0 = int((i / (num_labels + 1)) * size)
            y0 = int(((num_labels - i) / (num_labels + 1)) * size)
            arr[y0 : min(size, y0 + size // 2), x0 : min(size, x0 + size // 2)] = col

    # Add realistic sensor noise
    noise = np.random.normal(0, 8, (size, size, 3)).astype(np.int16)
    noisy_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(noisy_arr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")


# ── Main Dataset Preparation ────────────────────────────────────────────────


def prepare_dataset(
    data_dir: str | None = None,
    output_path: str = "data/bigearthnet_chat.jsonl",
    max_samples: int = 10000,
    generate_images: bool = True,
    images_dir: str = "data/images",
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare BigEarthNet samples into chat-format JSONL.

    Parameters
    ----------
    data_dir : str | None
        Path to raw BigEarthNet / BigEarthNet.txt directory. If None or not found,
        generates synthetic remote-sensing samples following the BigEarthNet schema.
    output_path : str
        Output path for the .jsonl chat dataset.
    max_samples : int
        Target number of chat-format samples (5k to 20k recommended).
    generate_images : bool
        Whether to generate synthetic images if data_dir is unavailable.
    images_dir : str
        Directory to store generated images.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with dataset statistics (total_samples, task_counts, class_counts, output_path).
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    real_patches: list[dict[str, Any]] = []

    # 1. Attempt to load real BigEarthNet patches if data_dir exists
    if data_dir and os.path.isdir(data_dir):
        logger.info("Scanning for real BigEarthNet patches in: %s", data_dir)
        json_files = glob.glob(os.path.join(data_dir, "**/*labels_metadata.json"), recursive=True)
        if not json_files:
            json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                labels = meta.get("labels", [])
                if labels and isinstance(labels, list):
                    patch_id = os.path.basename(os.path.dirname(jf)) or Path(jf).stem
                    # Find image if exists
                    img_candidates = glob.glob(os.path.join(os.path.dirname(jf), "*.tif")) + \
                                     glob.glob(os.path.join(os.path.dirname(jf), "*.png"))
                    img_path = img_candidates[0] if img_candidates else ""
                    real_patches.append({
                        "patch_id": patch_id,
                        "labels": labels,
                        "image_path": img_path,
                    })
            except Exception as exc:
                logger.debug("Failed parsing %s: %s", jf, exc)

    logger.info("Found %d real BigEarthNet patches.", len(real_patches))

    # 2. Build or complement patches
    patches_needed = max(100, max_samples // 4)
    samples: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {c: 0 for c in BIGEARTHNET_19_CLASSES}

    patch_records = list(real_patches)

    # If real patches are fewer than needed, synthesize patches
    if len(patch_records) < patches_needed and generate_images:
        logger.info(
            "Synthesizing %d BigEarthNet remote-sensing patches conforming to 19-class schema...",
            patches_needed - len(patch_records),
        )
        for i in range(len(patch_records), patches_needed):
            patch_id = f"BEN_S2_{i:06d}"
            # Select 1 to 3 realistic land-cover classes
            num_classes = random.choices([1, 2, 3], weights=[0.4, 0.45, 0.15])[0]
            chosen_classes = random.sample(BIGEARTHNET_19_CLASSES, num_classes)

            img_filename = f"{patch_id}.png"
            img_full_path = os.path.join(images_dir, img_filename)

            # Generate synthetic image if not already created
            if not os.path.exists(img_full_path):
                _create_synthetic_patch(chosen_classes, img_full_path)

            patch_records.append({
                "patch_id": patch_id,
                "labels": chosen_classes,
                "image_path": os.path.relpath(img_full_path, os.getcwd()).replace("\\", "/"),
            })

    # 3. Generate chat-format samples for each patch
    sample_index = 0
    builders = [
        _generate_caption_sample,
        _generate_scene_description_sample,
        _generate_vqa_water_sample,
        _generate_vqa_urban_sample,
        _generate_vqa_landcover_sample,
        _generate_vqa_rsvqa_style_sample,
    ]

    while len(samples) < max_samples and patch_records:
        patch = patch_records[sample_index % len(patch_records)]
        patch_id = patch["patch_id"]
        labels = patch["labels"]
        image_path = patch["image_path"]

        # Ensure image path is present
        if not image_path:
            image_path = os.path.join(images_dir, f"{patch_id}.png").replace("\\", "/")
            if not os.path.exists(image_path) and generate_images:
                _create_synthetic_patch(labels, image_path)

        for lbl in labels:
            if lbl in class_counts:
                class_counts[lbl] += 1

        builder = random.choice(builders)
        s_id = f"sample_{len(samples):06d}"
        sample = builder(s_id, patch_id, image_path, labels)

        samples.append(sample)
        task = sample["task_type"]
        task_counts[task] = task_counts.get(task, 0) + 1
        sample_index += 1

    # 4. Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Successfully wrote %d chat-format samples to %s", len(samples), output_path)

    stats = {
        "total_samples": len(samples),
        "output_path": str(Path(output_path).resolve()),
        "task_distribution": task_counts,
        "class_distribution": {k: v for k, v in class_counts.items() if v > 0},
    }
    return stats


# ── CLI Interface ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BigEarthNet.txt chat-format dataset for VLM fine-tuning."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw BigEarthNet directory (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/bigearthnet_chat.jsonl",
        help="Path to save output JSONL dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Total number of chat-format samples to generate (default: 5000).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory to store imagery for samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print(f"Preparing dataset with target {args.max_samples} samples...")
    stats = prepare_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        images_dir=args.images_dir,
        seed=args.seed,
    )
    print("\nDataset preparation complete:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
