"""按类别分层划分训练集/验证集/测试集，产出 CSV 清单，供报告里的指标使用。

用法：
    python pipeline/make_splits.py
    python pipeline/make_splits.py --data-root /root/autodl-tmp/dataset/tt100k_cls --out-dir splits_tt100k

输出：
    splits/train.csv、val.csv、test.csv，列为 path,label
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import dataset_root  # noqa: E402

IMG_EXTS = (".ppm", ".png", ".jpg", ".jpeg", ".bmp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None,
                   help="类别根目录，默认 <dataset-root>/gtsrb_train/GTSRB/Training")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--out-dir", default="splits")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.data_root) if args.data_root else dataset_root(args.dataset_root) / "gtsrb_train/GTSRB/Training"
    if not root.exists():
        raise FileNotFoundError(f"找不到数据目录: {root}")
    paths, labels = [], []
    for idx, d in enumerate(sorted([x for x in root.iterdir() if x.is_dir()])):
        for f in d.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
                labels.append(idx)

    paths = np.array(paths)
    labels = np.array(labels)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=1 - args.train - args.val,
        random_state=args.seed, stratify=labels,
    )
    val_share = args.val / (args.train + args.val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_share, random_state=args.seed, stratify=y_tmp,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, Xs, ys in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(zip(Xs, ys))
        print(f"{name}: {len(Xs)} 张")


if __name__ == "__main__":
    main()
