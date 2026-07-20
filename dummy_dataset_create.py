"""
Create dummy pathology datasets for local testing.

Generates N datasets, each with M fake WSI images (small random TIFFs)
and a manifest.csv matching the real data format:

    patient_id,wsi_path,cancer
    PAT_001,wsis/PAT_001.tiff,0

Usage examples:
    # Default: 2 datasets x 50 WSIs each
    python dummy_dataset_create.py

    # 3 datasets with 20 WSIs each
    python dummy_dataset_create.py --datasets 3 --wsis-per-dataset 20

    # 1 dataset with 100 WSIs (replaces existing data/)
    python dummy_dataset_create.py --datasets 1 --wsis-per-dataset 100

    # Keep existing datasets, just add more
    python dummy_dataset_create.py --datasets 1 --wsis-per-dataset 30 --no-clean
"""

import argparse
import csv
import io
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args():
    p = argparse.ArgumentParser(
        description="Create dummy pathology datasets for local testing.",
    )
    p.add_argument(
        "--datasets", "-d",
        type=int, default=2,
        help="Number of dataset folders to create (default: 2)",
    )
    p.add_argument(
        "--wsis-per-dataset", "-w",
        type=int, default=50,
        help="Number of WSI images per dataset (default: 50)",
    )
    p.add_argument(
        "--data-dir",
        type=str, default="test_data",
        help="Parent directory for the datasets (default: test_data/)",
    )
    p.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip removing existing dummy_dataset_* folders before creating new ones",
    )
    p.add_argument(
        "--img-size",
        type=int, default=128,
        help="Size of each dummy WSI image in pixels (default: 128x128)",
    )
    return p.parse_args()


def _clean_existing(data_dir: Path):
    """Remove old dummy_dataset_* folders from data/."""
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("dummy_dataset_"):
            shutil.rmtree(entry)
            print(f"  Removed existing {entry.name}/")


def _generate_wsi(img_size: int, rng: np.random.Generator) -> bytes:
    """Generate a small random TIFF image."""
    arr = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


def create_dummy_datasets(
    num_datasets: int = 2,
    wsis_per_dataset: int = 50,
    data_dir: Path = Path("test_data"),
    seed: int = 42,
    img_size: int = 128,
    clean: bool = True,
):
    rng = np.random.default_rng(seed)

    if clean and data_dir.exists():
        _clean_existing(data_dir)

    for ds_idx in range(num_datasets):
        ds_name = f"dummy_dataset_{ds_idx + 1}"
        ds_path = data_dir / ds_name
        wsis_path = ds_path / "wsis"
        wsis_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(wsis_per_dataset):
            patient_id = f"PAT_{uuid.uuid4().hex[:8]}"
            tiff_name = f"{patient_id}.tiff"
            cancer = int(rng.random() > 0.5)
            wsi_path = wsis_path / tiff_name

            wsi_path.write_bytes(_generate_wsi(img_size, rng))
            rows.append((patient_id, str(wsi_path.relative_to(ds_path)), cancer))

        manifest_path = ds_path / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "wsi_path", "cancer"])
            writer.writerows(rows)

        print(f"  Created {ds_name}/ ({len(rows)} samples)")


def main():
    args = _parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Creating {args.datasets} dataset(s) with {args.wsis_per_dataset} WSIs each...")
    create_dummy_datasets(
        num_datasets=args.datasets,
        wsis_per_dataset=args.wsis_per_dataset,
        data_dir=data_dir,
        seed=args.seed,
        img_size=args.img_size,
        clean=not args.no_clean,
    )
    print(f"[*] Done. Data in: {data_dir.resolve()}")


if __name__ == "__main__":
    main()
