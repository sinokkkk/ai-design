"""用 PyTorch 版分类模型在 GTSRB 上算准确率，划分方式和训练一致（80/20，seed=42）。

用法：python pipeline/Classifier_metrics.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.model_torch import TrafficSignNet

DATA_ROOT = Path("F:/ai-design/dataset/gtsrb_train/GTSRB/Training")
MODEL_PATH = Path("output/trafficsignnet_v2.pth")

IMG_SIZE = (64, 64)
SEED = 42
BATCH = 256
SPEED_LIMIT_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_gtsrb():
    X, y = [], []
    for class_dir in sorted(DATA_ROOT.glob("*")):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        for img_path in class_dir.glob("*.ppm"):
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
    device = get_device()
    print(f"设备: {device}")

    X, y = load_gtsrb()
    print(f"数据集: {X.shape}, {y.shape}")

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"验证集: {X_val.shape}")

    model = TrafficSignNet(num_classes=43)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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
