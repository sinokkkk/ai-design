"""把 TT100K 转成本项目要的两种格式。

1. 检测：单类 YOLO 数据集（images/labels 的 train/val）
2. 分类：按类别码裁好的小图，供 train_classifier_torch.py 直接使用

前提：已解压 data.zip，目录里有 annotations.json 和 train/ test/ other/ 三个图像文件夹。

用法：
    python detector/prepare_tt100k.py --tt100k-root F:/ai-design/dataset/tt100k/data
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import dataset_root  # noqa: E402

# 按类别码前缀生成中文名，无法确定具体含义的保留编码
FAMILY = {
    "pl": ("限速{}km/h", "pl"),
    "il": ("最低限速{}km/h", "il"),
    "ph": ("限高{}米", "ph"),
    "pw": ("限宽{}米", "pw"),
    "pm": ("限重{}吨", "pm"),
    "p": ("禁令标志", None),
    "w": ("警告标志", None),
    "i": ("指示标志", None),
}


def code_to_name(code):
    for prefix in ("pl", "il", "ph", "pw", "pm"):
        if code.startswith(prefix) and code != prefix:
            num = code[len(prefix):]
            fmt = FAMILY[prefix][0]
            return fmt.format(num)
    head = code[0]
    if head in FAMILY:
        return f"{FAMILY[head][0]}-{code}"
    return code


def link_or_copy(src, dst):
    """优先硬链接，跨盘或失败时复制，省磁盘空间。"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tt100k-root", default=None,
                   help="包含 annotations.json 的 data 目录，默认 <dataset-root>/tt100k/data")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--out", default=None, help="检测集输出，默认 <dataset-root>/tt100k_yolo")
    p.add_argument("--cls-out", default=None, help="分类图输出，默认 <dataset-root>/tt100k_cls")
    p.add_argument("--min-count", type=int, default=100, help="少于这个数量的类别丢弃")
    p.add_argument("--val-ratio", type=float, default=0.1, help="从 train 里划多少做验证集")
    p.add_argument("--neg-ratio", type=float, default=0.2,
                   help="负样本（无标志图）占训练正样本的比例，0 表示不用")
    p.add_argument("--yaml-out", default=None,
                   help="检测配置输出路径，默认 detector/data_tt100k.yaml")
    p.add_argument("--skip-cls", action="store_true",
                   help="只重生成检测集和配置，跳过分类裁剪图（已裁过时用）")
    args = p.parse_args()

    ds = dataset_root(args.dataset_root)
    root = Path(args.tt100k_root) if args.tt100k_root else ds / "tt100k/data"
    if not (root / "annotations.json").exists():
        raise FileNotFoundError(f"{root} 下没有 annotations.json")
    ann = json.loads((root / "annotations.json").read_text(encoding="utf-8"))

    counts = Counter()
    for info in ann["imgs"].values():
        for o in info.get("objects", []):
            counts[o["category"]] += 1
    keep = {c for c, n in counts.items() if n >= args.min_count}
    print(f"类别总数 {len(counts)}，保留 {len(keep)} 类（>= {args.min_count} 个样本）")

    out = Path(args.out) if args.out else ds / "tt100k_yolo"
    cls_out = Path(args.cls_out) if args.cls_out else ds / "tt100k_cls"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 按文件编号排序后取前 val_ratio 做验证，保证每次划分一致
    ids = sorted(ann["imgs"].keys(), key=lambda k: int(k))
    train_ids = [i for i in ids if ann["imgs"][i]["path"].startswith("train/")]
    n_val = int(len(train_ids) * args.val_ratio)
    val_set = set(train_ids[:n_val])

    stat = Counter()

    for img_id in tqdm(ids, desc="生成检测集"):
        info = ann["imgs"][img_id]
        rel = info["path"]

        # 有标注的图全部参与，包括 other/ 里那批带框的图，否则稀疏类别裁不出图
        objects = info.get("objects", [])
        if not objects:
            continue

        src = root / rel
        if not src.exists():
            stat["missing_img"] += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            stat["unreadable"] += 1
            continue
        h, w = img.shape[:2]

        split = "val" if (rel.startswith("test/") or img_id in val_set) else "train"
        stem = Path(rel).stem
        name = f"{stem}.jpg"

        link_or_copy(src, out / "images" / split / name)

        lines = []
        for k, o in enumerate(objects):
            b = o["bbox"]
            x1, y1 = max(0.0, b["xmin"]), max(0.0, b["ymin"])
            x2, y2 = min(float(w), b["xmax"]), min(float(h), b["ymax"])
            if x2 <= x1 or y2 <= y1:
                continue
            # 单类检测，全部记作 0
            lines.append(
                f"0 {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )

            # 分类裁剪图只保留样本足够的类别
            if args.skip_cls or o["category"] not in keep:
                continue
            code = o["category"]
            cdir = cls_out / code
            cdir.mkdir(parents=True, exist_ok=True)
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size:
                cv2.imwrite(str(cdir / f"{stem}_{k}.jpg"), crop)
                stat["crops"] += 1

        if not lines:
            continue
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stat[f"imgs_{split}"] += 1

    # 负样本：other/ 里没有标注的图，标签留空，教模型别乱框
    if args.neg_ratio > 0:
        pos_train = stat["imgs_train"]
        limit = int(pos_train * args.neg_ratio)
        neg_count = 0
        for img_id in ids:
            info = ann["imgs"][img_id]
            rel = info["path"]
            if not rel.startswith("other/") or info.get("objects"):
                continue
            if neg_count >= limit:
                break
            src = root / rel
            if not src.exists():
                continue
            stem = Path(rel).stem
            link_or_copy(src, out / "images" / "train" / f"{stem}.jpg")
            (out / "labels" / "train" / f"{stem}.txt").write_text("", encoding="utf-8")
            neg_count += 1
        stat["neg_train"] = neg_count
        print(f"负样本: {neg_count} 张（{neg_count / max(pos_train, 1):.0%} 正样本）")

    # 类别名表
    actual = sorted([d.name for d in cls_out.iterdir() if d.is_dir() and any(d.iterdir())]) if cls_out.exists() else []
    rows = ["ClassId,Code,SignName"]
    for i, code in enumerate(actual):
        rows.append(f'{i},{code},"{code_to_name(code)}"')
    (out / "signnames_tt100k.csv").write_text("\n".join(rows), encoding="utf-8")

    yaml_text = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: traffic_sign\n"
    )
    yaml_path = Path(args.yaml_out) if args.yaml_out else Path(__file__).resolve().parent / "data_tt100k.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("完成：", dict(stat))
    print(f"检测集：{out}")
    print(f"分类图：{cls_out}（实际 {len(actual)} 类，裁剪 {stat['crops']} 张）")
    print(f"配置：{yaml_path}")
    print(f"类别表：{out / 'signnames_tt100k.csv'}")


if __name__ == "__main__":
    main()
