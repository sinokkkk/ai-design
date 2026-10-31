"""把 YOLO 数据集里的 .ppm 转成同名 .png，因为 ultralytics 不支持 ppm。

用法：
    python detector/convert_ppm_to_png.py
    python detector/convert_ppm_to_png.py --yolo-root /root/autodl-tmp/dataset/gtsdb_yolo
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import dataset_root  # noqa: E402


def convert_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        print(f"目录不存在: {folder}")
        return 0

    files = list(folder.glob("*.ppm"))
    for f in files:
        Image.open(f).save(f.with_suffix(".png"))
    print(f"{folder}: 转换 {len(files)} 个文件")
    return len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--yolo-root", default=None, help="默认 <dataset-root>/gtsdb_yolo")
    args = p.parse_args()

    root = Path(args.yolo_root) if args.yolo_root else dataset_root(args.dataset_root) / "gtsdb_yolo"
    for split in ("train", "val"):
        convert_folder(root / "images" / split)


if __name__ == "__main__":
    main()
