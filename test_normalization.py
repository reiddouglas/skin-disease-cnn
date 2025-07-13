import numpy as np
import random
from pathlib import Path
from PIL import Image

# Settings - adjust these if your paths change
TRAIN_DIR = Path(__file__).parent / "dataset" / "512x512-dataset-melanoma"
NORM_DIR  = Path(__file__).parent / "dataset_norm" / "train"
MEAN      = np.array([0.56057299, 0.58049109, 0.74380801], dtype=np.float32)
STD       = np.array([0.21038430, 0.19394357, 0.19913281], dtype=np.float32)
EXTS      = {".jpg"}

def seed_random():
    """Make random.sample deterministic for repeatable CI runs."""
    random.seed(42)

def check_file_names(train_dir, norm_dir):
    inp = {
        p.stem.replace("_downsampled", "")
        for p in train_dir.rglob("*")
        if p.suffix.lower() in EXTS
    }
    out = {p.stem for p in norm_dir.rglob("*.npy")}
    missing = inp - out
    extra   = out - inp
    if missing or extra:
        print("Warning: file-name mapping is not exact:")
        if missing:
            print("  Missing normalized files for:", sorted(missing)[:5], "...")
        if extra:
            print("  Extra files for:", sorted(extra)[:5], "...")
    else:
        print("File name test passed")

def check_shapes(norm_dir):
    for p in norm_dir.rglob("*.npy"):
        arr = np.load(p)
        assert arr.dtype == np.float32, f"Wrong dtype for {p.name}"
        assert arr.ndim == 3 and arr.shape[2] == 3, f"Wrong shape for {p.name}"
        assert np.isfinite(arr).all(), f"Non-finite values in {p.name}"
        mn, mx = arr.min(), arr.max()
        assert -10 < mn < 10 and -10 < mx < 10, (
            f"Out-of-range values in {p.name}: min={mn}, max={mx}"
        )
    print("Shape and dtype test passed")

def roundtrip_test(orig_dir, norm_dir, mean, std, samples=100):
    seed_random()
    npy_files = list(norm_dir.rglob("*.npy"))
    n = min(samples, len(npy_files))
    print(f"Roundtrip test: sampling {n} files")
    selected = random.sample(npy_files, n)

    for p in selected:
        arr = np.load(p)
        rec = (arr * std + mean).clip(0, 1)

        # find original anywhere under orig_dir
        orig_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            matches = list(orig_dir.rglob(p.stem + ext))
            if matches:
                orig_path = matches[0]
                break

        if orig_path is None:
            print(f"  Warning: no original found for {p.stem}, skipping")
            continue

        # load original and convert both to uint8
        orig_img   = Image.open(orig_path).convert("RGB")
        orig_uint8 = np.array(orig_img, dtype=np.uint8)
        rec_uint8  = (rec * 255.0).round().astype(np.uint8)

        # allow off-by-one due to rounding noise
        diff = np.abs(rec_uint8.astype(int) - orig_uint8.astype(int))
        max_diff = diff.max()
        assert max_diff <= 1, (
            f"Pixel difference >1 for {p.stem}: max diff = {max_diff}"
        )

    print("Roundtrip test passed")




def synthetic_test():
    arr = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
         [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]],
        dtype=np.float32
    )
    mean = float(arr.mean())
    std  = float(arr.std())
    norm = (arr - mean) / std
    assert abs(norm.mean()) < 1e-6, "Synthetic norm mean not zero"
    assert abs(norm.std() - 1) < 1e-6,  "Synthetic norm std not one"
    rec = norm * std + mean
    assert np.allclose(rec, arr, atol=1e-6), "Synthetic roundtrip failed"
    print("Synthetic test passed")

if __name__ == "__main__":
    check_file_names(TRAIN_DIR, NORM_DIR)
    check_shapes(NORM_DIR)
    roundtrip_test(TRAIN_DIR, NORM_DIR, MEAN, STD, samples=100)
    synthetic_test()
    print("All tests completed")
