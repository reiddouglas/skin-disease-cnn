import csv
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

# ---------------- Configuration ----------------
DATA_ROOT   = Path(__file__).parent / "dataset"
TRAIN_DIR   = DATA_ROOT / "train"
TEST_DIR    = DATA_ROOT / "test"
TRAIN_CSV   = DATA_ROOT / "train.csv"        # CSV with image_name,target
OUT_ROOT    = Path(__file__).parent / "dataset_b4_batches"

EXTS        = {".jpg"}        
BATCH_SIZE  = 3000
MAX_BATCHES = 12              

# EfficientNet-B4 normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SRC_SIZE   = 384   # expected original size
CROP_SIZE  = 380   # final size after center crop
PROGRESS_INTERVAL = 1000  # images per progress print

# Optional version tag stored in each .npz (set to None to skip)
VERSION_TAG: Optional[str] = "b4_center380_imagenet_v1"

# ---------------- Helper Functions ----------------

def load_train_targets(train_csv: Path) -> Dict[str, int]:
    """
    Load mapping image_name -> target (0/1) from train.csv.
    Assumes 'image_name' and 'target' columns exist.
    """
    mapping = {}
    with open(train_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'image_name' not in reader.fieldnames or 'target' not in reader.fieldnames:
            raise ValueError("train.csv must contain 'image_name' and 'target' columns.")
        for row in reader:
            name = row['image_name'].strip()
            if not name:
                continue
            tgt = row['target'].strip()
            if tgt == '':
                continue
            try:
                mapping[name] = int(tgt)
            except ValueError:
                raise ValueError(f"Invalid target value '{tgt}' for image '{name}'")
    return mapping


def center_crop_380_from_384(arr_384: np.ndarray) -> np.ndarray:
    """Center crop 384x384 -> 380x380 (no interpolation)."""
    assert arr_384.shape[0] == SRC_SIZE and arr_384.shape[1] == SRC_SIZE, \
        f"Expected {SRC_SIZE}x{SRC_SIZE}, got {arr_384.shape[:2]}"
    off = (SRC_SIZE - CROP_SIZE) // 2  # 2
    return arr_384[off:off + CROP_SIZE, off:off + CROP_SIZE, :]


def preprocess_image(path: Path) -> np.ndarray:
    """
    Load a 384x384 RGB image, center-crop to 380x380,
    normalize (ImageNet), return channel-first float16 tensor (3,380,380).
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    if arr.shape[0] != SRC_SIZE or arr.shape[1] != SRC_SIZE:
        raise ValueError(f"Image {path.name} is not {SRC_SIZE}x{SRC_SIZE}, got {arr.shape[:2]}")
    arr = arr / 255.0
    arr = center_crop_380_from_384(arr)                # (380,380,3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr_chw = np.transpose(arr, (2, 0, 1)).astype(np.float16)  # (3,380,380) float16
    return arr_chw


def batch_write(out_path: Path, images: List[np.ndarray], names: List[str], targets: List[int] = None):
    """
    Write a batch .npz file.
    - images: list of (3,380,380) float16 arrays
    - names : list of stems (strings)
    - targets: optional list of ints (0/1)
    Stored keys:
        train: image, name, target (, version)
        test : image, name (, version)
    """
    img_stack = np.stack(images, axis=0)          # (N,3,380,380)
    
    max_len = max(len(n) for n in names) if names else 1
    name_arr = np.array(names, dtype=f'<U{max_len}')
    arrays = {
        "image": img_stack,
        "name": name_arr
    }
    if targets is not None:
        arrays["target"] = np.array(targets, dtype=np.uint8)
    if VERSION_TAG is not None:
        arrays["version"] = np.array(VERSION_TAG)
    np.savez(out_path, **arrays)


def process_train(train_dir: Path, mapping: Dict[str, int]):
    """
    Process training images in CSV order, batching into .npz files.
    Stops after MAX_BATCHES * BATCH_SIZE images.
    """
    out_dir = OUT_ROOT / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    batch_index = 0
    images, names, targets = [], [], []
    skipped_missing_img = 0
    total_limit = BATCH_SIZE * MAX_BATCHES

    for name, tgt in mapping.items():
        if used >= total_limit:
            break

        img_path = train_dir / f"{name}.jpg"
        if not img_path.exists():
            print(f"[WARN] Missing image file '{name}.jpg'")
            skipped_missing_img += 1
            continue

        try:
            im = preprocess_image(img_path)
        except Exception as e:
            print(f"[WARN] Skipping {img_path.name}: {e}")
            skipped_missing_img += 1
            continue

        images.append(im)
        names.append(name)
        targets.append(tgt)
        used += 1

        if used % PROGRESS_INTERVAL == 0:
            print(f"[TRAIN] Processed {used} images (current batch size {len(images)})")

        if len(images) == BATCH_SIZE:
            out_path = out_dir / f"train_batch_{batch_index:03d}.npz"
            batch_write(out_path, images, names, targets)
            print(f"[TRAIN] Wrote batch {batch_index} with {len(images)} images -> {out_path.name}")
            batch_index += 1
            images.clear(); names.clear(); targets.clear()

            if batch_index >= MAX_BATCHES:
                break

    # Flush remainder
    if images and batch_index < MAX_BATCHES:
        out_path = out_dir / f"train_batch_{batch_index:03d}.npz"
        batch_write(out_path, images, names, targets)
        print(f"[TRAIN] Wrote final partial batch {batch_index} with {len(images)} images")

    print(f"[TRAIN SUMMARY] kept={used}, skipped_missing_img={skipped_missing_img}, "
          f"batches_written={min(batch_index + (1 if images else 0), MAX_BATCHES)}")


def process_test(test_dir: Path):
    """
    Process test images (no targets) in filename-sorted order.
    Limit to MAX_BATCHES * BATCH_SIZE images.
    """
    out_dir = OUT_ROOT / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in EXTS)
    total_limit = BATCH_SIZE * MAX_BATCHES
    used = 0
    batch_index = 0
    images, names = [], []

    for p in all_imgs:
        if used >= total_limit:
            break

        name = p.stem
        try:
            im = preprocess_image(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        images.append(im)
        names.append(name)
        used += 1

        if used % PROGRESS_INTERVAL == 0:
            print(f"[TEST] Processed {used} images (current batch size {len(images)})")

        if len(images) == BATCH_SIZE:
            out_path = out_dir / f"test_batch_{batch_index:03d}.npz"
            batch_write(out_path, images, names)
            print(f"[TEST] Wrote batch {batch_index} with {len(images)} images -> {out_path.name}")
            batch_index += 1
            images.clear(); names.clear()

            if batch_index >= MAX_BATCHES:
                break

    # Flush remainder
    if images and batch_index < MAX_BATCHES:
        out_path = out_dir / f"test_batch_{batch_index:03d}.npz"
        batch_write(out_path, images, names)
        print(f"[TEST] Wrote final partial batch {batch_index} with {len(images)} images")

    print(f"[TEST SUMMARY] kept={used}, batches_written={min(batch_index + (1 if images else 0), MAX_BATCHES)}")


def main():
    
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    print("Loading train targets...")
    mapping = load_train_targets(TRAIN_CSV)
    print(f"Loaded {len(mapping)} rows with targets.")

    print("\nProcessing TRAIN batches...")
    process_train(TRAIN_DIR, mapping)

    print("\nProcessing TEST batches...")
    process_test(TEST_DIR)

    print("\nDone. Batched EfficientNet-B4 files in:", OUT_ROOT)
    print("Train batch keys: image (N,3,380,380 float16), name (<U), target (uint8)", 
          f"+ version" if VERSION_TAG else "")
    print("Test  batch keys: image (N,3,380,380 float16), name (<U)",
          f"+ version" if VERSION_TAG else "")


if __name__ == "__main__":
    main()
