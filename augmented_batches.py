import csv
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image

# =========================================================
# -------------------- CONFIGURATION ----------------------
# =========================================================

DATA_ROOT  = Path(__file__).parent / "dataset"     
TRAIN_DIR  = DATA_ROOT / "train"
TRAIN_CSV  = DATA_ROOT / "train.csv"

OUT_ROOT   = Path(__file__).parent / "dataset_b4_batches_aug_only7"
BATCH_SIZE = 3000
MAX_BATCHES = 10                       
TOTAL_LIMIT = BATCH_SIZE * MAX_BATCHES 

# EXACT seven lossless dihedral variants (no original kept)
AUG_VARIANTS = [
    "rot90", "rot180", "rot270",
    "hflip", "vflip",
    "rot90_hflip", "rot270_hflip"
]  

# Image / normalization
SRC_SIZE   = 384
CROP_SIZE  = 380
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PROGRESS_INTERVAL = 250
VERBOSE = True

# =========================================================
# -------------------- UTILITIES --------------------------
# =========================================================

def log(msg: str):
    if VERBOSE:
        print(msg)

def load_train_targets(csv_path: Path) -> Dict[str, int]:
    mapping: Dict[str,int] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if 'image_name' not in r.fieldnames or 'target' not in r.fieldnames:
            raise ValueError("train.csv must contain image_name,target")
        for row in r:
            name = row['image_name'].strip()
            tgt  = row['target'].strip()
            if name and tgt != '':
                mapping[name] = int(tgt)
    return mapping

def center_crop(arr384: np.ndarray) -> np.ndarray:
    off = (SRC_SIZE - CROP_SIZE) // 2
    return arr384[off:off+CROP_SIZE, off:off+CROP_SIZE, :]

def normalize_to_chw(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img, dtype=np.float32)
    if arr.shape[:2] != (SRC_SIZE, SRC_SIZE):
        raise ValueError(f"Unexpected size {arr.shape[:2]}")
    arr = arr / 255.0
    arr = center_crop(arr)
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2,0,1)).astype(np.float16)  # (3,380,380)

def dihedral_augments(img: Image.Image):
    """Yield exactly 7 lossless variants (tags & image)."""
    rot90  = img.transpose(Image.ROTATE_90)
    rot180 = img.transpose(Image.ROTATE_180)
    rot270 = img.transpose(Image.ROTATE_270)
    hflip  = img.transpose(Image.FLIP_LEFT_RIGHT)
    vflip  = img.transpose(Image.FLIP_TOP_BOTTOM)
    rot90_hflip  = rot90.transpose(Image.FLIP_LEFT_RIGHT)
    rot270_hflip = rot270.transpose(Image.FLIP_LEFT_RIGHT)
    pool = {
        "rot90": rot90, "rot180": rot180, "rot270": rot270,
        "hflip": hflip, "vflip": vflip,
        "rot90_hflip": rot90_hflip, "rot270_hflip": rot270_hflip
    }
    return [(pool[tag], tag) for tag in AUG_VARIANTS]

def batch_write(path: Path, images: List[np.ndarray], names: List[str]):
    img_stack = np.stack(images, axis=0)
    max_len = max(len(n) for n in names)
    name_arr = np.array(names, dtype=f'<U{max_len}')
    target_arr = np.ones(len(names), dtype=np.uint8)  
    np.savez(path, image=img_stack, name=name_arr, target=target_arr)

# =========================================================
# -------------------- TRAIN PIPELINE ---------------------
# =========================================================

def process_train(mapping: Dict[str,int]):
    out_dir = OUT_ROOT / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    pos_count = 0
    batch_idx = 0

    imgs: List[np.ndarray] = []
    names: List[str] = []

    def flush():
        nonlocal imgs, names, batch_idx
        if not imgs: return
        out_path = out_dir / f"train_batch_{batch_idx:03d}.npz"
        batch_write(out_path, imgs, names)
        log(f"[TRAIN] wrote batch {batch_idx} size={len(imgs)}")
        batch_idx += 1
        imgs.clear(); names.clear()

    for name, tgt in mapping.items():
        if tgt != 1:
            # skip negatives completely
            continue
        if batch_idx >= MAX_BATCHES or used >= TOTAL_LIMIT:
            break

        img_path = TRAIN_DIR / f"{name}.jpg"
        if not img_path.exists():
            log(f"[WARN] missing {name}.jpg")
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            log(f"[WARN] loading {name}: {e}")
            continue

        # seven aug variants (no original)
        for aug_img, tag in dihedral_augments(pil):
            if batch_idx >= MAX_BATCHES or used >= TOTAL_LIMIT:
                break
            arr = normalize_to_chw(aug_img)
            imgs.append(arr)
            names.append(f"{name}__{tag}")
            used += 1
            pos_count += 1
            if len(imgs) == BATCH_SIZE:
                flush()

        if used % PROGRESS_INTERVAL == 0:
            log(f"[TRAIN] progress used={used} positives={pos_count} (current batch {len(imgs)})")

    if imgs and batch_idx < MAX_BATCHES:
        flush()

    log(f"[TRAIN SUMMARY] total_augmented_positives={pos_count} batches={batch_idx}")

# =========================================================
# ----------------------------- MAIN ----------------------
# =========================================================

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mapping = load_train_targets(TRAIN_CSV)
    log(f"Loaded {len(mapping)} train rows.")

    log("\nGenerating ONLY 7 aug variants per positive (negatives skipped)...")
    process_train(mapping)

    log("\nDone. Output batches in:")
    log(f"  {OUT_ROOT}")
    log("Each positive appears 7x; no negatives or originals stored.")

if __name__ == "__main__":
    main()
