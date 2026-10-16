"""Data preparation script for BigEarthNet.txt remote sensing dataset.

Formats BigEarthNet.txt image-text pairs into chat-format samples for vision-language
model fine-tuning (captions, scene descriptions, and label-derived VQA).
"""


def prepare_dataset(data_dir: str, output_path: str, max_samples: int = 10000) -> None:
    """Prepare BigEarthNet samples into chat-format JSONL.

    Placeholder implementation to be completed in Step 3.
    """
    raise NotImplementedError("Dataset preparation will be implemented in Step 3.")


if __name__ == "__main__":
    print("Run data preparation for BigEarthNet.txt")
