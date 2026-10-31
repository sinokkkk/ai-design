"""把 GTSDB 的标注转成 YOLO 单类格式，并生成 detector/data.yaml。

用法：
    python detector/convert_gtsdb_to_yolo.py
    python detector/convert_gtsdb_to_yolo.py --dataset-root /root/autodl-tmp/dataset
    python detector/convert_gtsdb_to_yolo.py --gtsdb-root /data/gtsdb/TrainIJCNN2013/TrainIJCNN2013
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import dataset_root, PROJECT_ROOT  # noqa: E402

DET_CLASS_ID = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=None, help="数据集根目录，默认读环境变量或项目同级的 dataset/")
    p.add_argument("--gtsdb-root", default=None,
                   help="GTSDB 标注和图片所在目录，默认 <dataset-root>/gtsdb/TrainIJCNN2013/TrainIJCNN2013")
    p.add_argument("--out-root", default=None, help="输出目录，默认 <dataset-root>/gtsdb_yolo")
    p.add_argument("--val-ratio", type=float, default=0.2)
    args = p.parse_args()

    ds = dataset_root(args.dataset_root)
    gtsdb_root = Path(args.gtsdb_root) if args.gtsdb_root else ds / "gtsdb/TrainIJCNN2013/TrainIJCNN2013"
    out_root = Path(args.out_root) if args.out_root else ds / "gtsdb_yolo"

    gt_file = gtsdb_root / "gt.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"找不到标注文件: {gt_file}")

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in gt_file.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]

    boxes_by_img = {}
    for ln in lines:
        parts = ln.split(";")
        img_name = parts[0]
        left, top, right, bottom = map(int, parts[1:5])
        boxes_by_img.setdefault(img_name, []).append((left, top, right, bottom))

    img_names = sorted(boxes_by_img.keys())
    split_idx = int((1 - args.val_ratio) * len(img_names))
    train_imgs = set(img_names[:split_idx])

    for img_name in img_names:
        split = "train" if img_name in train_imgs else "val"

        src_img = gtsdb_root / img_name
        if not src_img.exists():
            raise FileNotFoundError(f"图片不存在: {src_img}")

        dst_img = out_root / "images" / split / img_name
        shutil.copy2(src_img, dst_img)

        im = cv2.imread(str(dst_img))
        if im is None:
            raise RuntimeError(f"OpenCV 读不了: {dst_img}")
        h, w = im.shape[:2]

        yolo_lines = []
        for (l, t, r, b) in boxes_by_img[img_name]:
            x_c = ((l + r) / 2.0) / w
            y_c = ((t + b) / 2.0) / h
            bw = (r - l) / float(w)
            bh = (b - t) / float(h)
            yolo_lines.append(f"{DET_CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_root / "labels" / split / (Path(img_name).stem + ".txt")
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    yaml_path = PROJECT_ROOT / "detector" / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: traffic_sign\n",
        encoding="utf-8",
    )

    print(f"训练图 {len(train_imgs)} 张，验证图 {len(img_names) - len(train_imgs)} 张")
    print(f"输出: {out_root}")
    print(f"配置: {yaml_path}")


if __name__ == "__main__":
    main()
