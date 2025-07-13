import shutil
import sys
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
# Computing mean/std over 58462 images ...
# Computed mean: [0.5620089173316956, 0.5811969041824341, 0.7454625368118286]
# Computed std : [0.21154886484146118, 0.19519373774528503, 0.20036040246486664] 
# Configuration – adjust these if your folder names change
TRAIN_DIR = Path(__file__).parent / "dataset" / "512x512-dataset-melanoma"
TEST_DIR  = Path(__file__).parent / "dataset" / "512x512-test"
OUT_ROOT  = Path(__file__).parent / "dataset_norm"
EXTS      = {".png", ".jpg", ".jpeg"}


def list_and_dedupe(folder: Path):
    """
    Recursively find all image files under `folder`, dedupe by base ID,
    preferring non-downsampled files over downsampled versions.
    """
    all_imgs = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
    chosen = {}
    for p in sorted(all_imgs):
        stem = p.stem
        if stem.endswith("_downsampled"):
            base = stem[:-len("_downsampled")]
            is_down = True
        else:
            base = stem
            is_down = False
        if base not in chosen or not is_down:
            chosen[base] = p
    return list(chosen.values())


def compute_mean_std(files):
    """
    Compute global per-channel mean & std over a list of image files.
    Uses axis-sums to avoid flattening.
    """
    sum_     = np.zeros(3, dtype=np.float64)
    sum_sq   = np.zeros(3, dtype=np.float64)
    total_px = 0

    for p in files:
        arr = (np.array(Image.open(p).convert("RGB"), dtype=np.float32)
               / 255.0)
        h, w, _ = arr.shape
        N = h * w
        total_px += N

        # sum over H×W for each channel
        sum_   += arr.sum(axis=(0, 1))
        sum_sq += (arr * arr).sum(axis=(0, 1))

    mean = (sum_ / total_px).astype(np.float32)
    var  = (sum_sq / total_px - (sum_ / total_px) ** 2)
    std  = np.sqrt(var).astype(np.float32)
    return mean, std


def normalize_and_save(files, dst_dir: Path, mean: np.ndarray, std: np.ndarray):
    """
    Normalize images and save as float32 .npy, one per base ID.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        clean = p.stem.replace("_downsampled", "")
        out_path = dst_dir / (clean + ".npy")

        arr = (np.array(Image.open(p).convert("RGB"), dtype=np.float32)
               / 255.0)
        norm = ((arr - mean) / std).astype(np.float32)
        np.save(out_path, norm)


def main():
    # 1) Clear old outputs
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    # 2) Gather & dedupe inputs
    train_files = list_and_dedupe(TRAIN_DIR)
    test_files  = list_and_dedupe(TEST_DIR)

    print(f"Unique training images: {len(train_files)}")
    print(f"Unique testing images:  {len(test_files)}")

    if not train_files:
        raise RuntimeError("No training images found - check TRAIN_DIR path.")

    # 3) Compute mean/std
    print(f"\nComputing mean/std over {len(train_files)} images ...")
    mean, std = compute_mean_std(train_files)
    print("Computed mean:", mean.tolist())
    print("Computed std :", std.tolist(), "\n")

    # 4) Normalize & save
    normalize_and_save(train_files, OUT_ROOT / "train", mean, std)
    normalize_and_save(test_files,  OUT_ROOT / "test",  mean, std)
    print("Normalization complete. Files saved in", OUT_ROOT)

    # 5) Run tests
    test_script = Path(__file__).parent / "test_normalization.py"
    print("\nRunning normalization tests...")
    result = subprocess.run(
        [sys.executable, str(test_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Tests failed. Exiting with error.")
        sys.exit(result.returncode)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
