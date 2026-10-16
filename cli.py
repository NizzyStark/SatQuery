"""Command-line interface for SatQuery VQA and captioning.

Usage
-----
    # VQA
    python cli.py vqa path/to/image.tif "What land cover is visible?"

    # Caption
    python cli.py caption path/to/image.tif --detail detailed

    # Caption (brief)
    python cli.py caption path/to/image.png --detail brief

    # With LoRA adapter
    python cli.py vqa path/to/image.tif "Is there water?" --adapter path/to/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from satquery.io import load_image
from satquery.tools import vqa, caption, get_vlm, reset_vlm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SatQuery AI — VQA and caption CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── VQA subcommand ──────────────────────────────────────────────────
    vqa_p = sub.add_parser("vqa", help="Visual question answering")
    vqa_p.add_argument("image", help="Path to the image file")
    vqa_p.add_argument("question", help="Question about the image")
    vqa_p.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max new tokens (16-128, default 128)",
    )
    vqa_p.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    vqa_p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Force device (default: auto)",
    )

    # ── Caption subcommand ──────────────────────────────────────────────
    cap_p = sub.add_parser("caption", help="Image captioning / scene description")
    cap_p.add_argument("image", help="Path to the image file")
    cap_p.add_argument(
        "--detail", default="detailed", choices=["brief", "detailed"],
        help="Caption detail level (default: detailed)",
    )
    cap_p.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    cap_p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Force device (default: auto)",
    )

    args = parser.parse_args()

    # ── Load image ──────────────────────────────────────────────────────
    print(f"Loading image: {args.image}")
    img_dict = load_image(args.image)
    print(
        f"  Modality: {img_dict['modality']}  |  "
        f"Bands: {img_dict['meta']['bands']}  |  "
        f"Size: {img_dict['meta']['size']}"
    )

    # ── Initialise VLM ──────────────────────────────────────────────────
    vlm_kwargs = {}
    if args.device:
        vlm_kwargs["device"] = args.device
    if args.adapter:
        vlm_kwargs["adapter_path"] = args.adapter
    vlm_client = get_vlm(**vlm_kwargs)

    # ── Run tool ────────────────────────────────────────────────────────
    if args.command == "vqa":
        print(f"\nQuestion: {args.question}")
        print("Running VQA...\n")
        result = vqa(
            img_dict, question=args.question,
            max_new_tokens=args.max_tokens, vlm=vlm_client,
        )
    elif args.command == "caption":
        print(f"\nDetail level: {args.detail}")
        print("Generating caption...\n")
        result = caption(img_dict, detail=args.detail, vlm=vlm_client)
    else:
        parser.print_help()
        sys.exit(1)

    # ── Print result ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Tool: {result.tool}")
    print(f"Answer:\n  {result.text}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Latency: {result.latency_s:.2f}s")
    print(f"Params: {json.dumps(result.params_used)}")
    print(f"Stats: {json.dumps(result.stats)}")
    print("=" * 60)

    reset_vlm()


if __name__ == "__main__":
    main()
