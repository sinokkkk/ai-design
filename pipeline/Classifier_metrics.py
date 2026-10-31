"""用 PyTorch 版分类模型在 GTSRB 上算准确率，划分方式和训练一致（80/20，seed=42）。

用法：
    python pipeline/Classifier_metrics.py
    python pipeline/Classifier_metrics.py --data-root /root/autodl-tmp/dataset/gtsrb_train/GTSRB/Training
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.model_torch import TrafficSignNet
from config import dataset_root, pick_device

MODEL_PATH = Path("output/trafficsignnet_v2.pth")

IMG_SIZE = (64, 64)
SEED = 42
BATCH = 256
SPEED_LIMIT_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}
IMG_EXTS = (".ppm", ".png", ".jpg", ".jpeg", ".bmp")


def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_gtsrb(data_root):
    X, y = [], []
    for class_dir in sorted(Path(data_root).glob("*")):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        class_id = int(class_dir.name)
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = clahe_rgb(img)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            X.append(img)
            y.append(class_id)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def predict(model, X, device):
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            batch = torch.from_numpy(X[i:i + BATCH]).permute(0, 3, 1, 2).to(device)
            probs_all.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probs_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None,
                   help="GTSRB 类别根目录，默认 <dataset-root>/gtsrb_train/GTSRB/Training")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--device", default="auto")
    p.add_argument("--num-classes", type=int, default=43)
    args = p.parse_args()

    data_root = Path(args.data_root) if args.data_root else dataset_root(args.dataset_root) / "gtsrb_train/GTSRB/Training"
    if not Path(data_root).exists():
        raise FileNotFoundError(f"找不到数据目录: {data_root}")

    device = torch.device(pick_device(args.device))
    print(f"设备: {device}")
    print(f"数据: {data_root}")

    X, y = load_gtsrb(data_root)
    print(f"数据集: {X.shape}, {y.shape}")

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"验证集: {X_val.shape}")

    model = TrafficSignNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device)

    probs = predict(model, X_val, device)
    y_pred = probs.argmax(1)

    acc = (y_pred == y_val).mean()
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.array([t in top3[i] for i, t in enumerate(y_val)]).mean()

    mask = np.isin(y_val, list(SPEED_LIMIT_CLASSES))
    speed_acc = (y_pred[mask] == y_val[mask]).mean() if mask.any() else None

    print("\n===== 分类指标 =====")
    print(f"准确率: {acc:.4f}")
    print(f"Top-3 准确率: {top3_acc:.4f}")
    if speed_acc is not None:
        print(f"限速子集准确率: {speed_acc:.4f}")

    print("\n===== 分类报告 =====")
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_val, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    worst = np.argsort(per_class)[:5]
    print("表现最差的 5 类:")
    for c in worst:
        print(f"  类 {c}: {per_class[c] * 100:.1f}% ({cm.sum(axis=1)[c]} 张)")


if __name__ == "__main__":
    main()
