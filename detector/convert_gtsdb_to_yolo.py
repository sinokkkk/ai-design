import os
import shutil
from pathlib import Path
import cv2


GTSDB_ROOT = Path("F:/ai-design/dataset/gtsdb/TrainIJCNN2013/TrainIJCNN2013")
OUT_ROOT   = Path("F:/ai-design/dataset/gtsdb_yolo")

GT_FILE = GTSDB_ROOT / "gt.txt"
IMG_DIR = GTSDB_ROOT

DET_CLASS_ID = 0

def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def read_gt_lines():
    lines = GT_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [ln for ln in lines if ln.strip()]

def main():
    ensure_dirs()
    lines = read_gt_lines()

    boxes_by_img = {}
    for ln in lines:
        parts = ln.split(";")
        img_name = parts[0]
        left, top, right, bottom = map(int, parts[1:5])
        boxes_by_img.setdefault(img_name, []).append((left, top, right, bottom))

    img_names = sorted(boxes_by_img.keys())

    split_idx = int(0.8 * len(img_names))
    train_imgs = set(img_names[:split_idx])
    val_imgs   = set(img_names[split_idx:])

    for img_name in img_names:
        split = "train" if img_name in train_imgs else "val"

        src_img = IMG_DIR / img_name
        if not src_img.exists():
            raise FileNotFoundError(f"Image not found: {src_img}")

        dst_img = OUT_ROOT / "images" / split / img_name
        shutil.copy2(src_img, dst_img)

        im = cv2.imread(str(dst_img))
        if im is None:
            raise RuntimeError(f"OpenCV could not read: {dst_img}")
        h, w = im.shape[:2]

        label_path = OUT_ROOT / "labels" / split / (Path(img_name).stem + ".txt")
        yolo_lines = []
        for (l, t, r, b) in boxes_by_img[img_name]:
            x_c = ((l + r) / 2.0) / w
            y_c = ((t + b) / 2.0) / h
            bw  = (r - l) / float(w)
            bh  = (b - t) / float(h)
            yolo_lines.append(f"{DET_CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    print("Converted to YOLO format at:", OUT_ROOT)

if __name__ == "__main__":
    main()