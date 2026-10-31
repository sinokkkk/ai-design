"""PyTorch 版分类训练脚本，适用于任意「按类别分文件夹」的数据集（德国 GTSRB、中国数据集通用）。

目录要求：
    <data-root>/00000/*.ppm
    <data-root>/00001/*.ppm
    ...

用法：
    python classifier/train_classifier_torch.py --data-root F:/ai-design/dataset/gtsrb_train/GTSRB/Training
    python classifier/train_classifier_torch.py --data-root F:/ai-design/dataset/china --out output/china.pth

类数、类别名都从数据目录读，不用改代码。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.model_torch import TrafficSignNet

IMG_EXTS = (".ppm", ".png", ".jpg", ".jpeg", ".bmp")


def get_device(prefer="auto"):
    if prefer != "auto":
        return torch.device(prefer)
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


def preprocess(img_bgr, size):
    img_bgr = clahe_rgb(img_bgr)
    img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img


def random_augment(img):
    """轻度增强：小角度旋转 + 平移缩放 + 亮度对比度 + 模糊。不做水平翻转。"""
    h, w = img.shape[:2]
    angle = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.92, 1.08)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += np.random.uniform(-0.08, 0.08) * w
    m[1, 2] += np.random.uniform(-0.08, 0.08) * h
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=np.random.uniform(0.8, 1.2), beta=np.random.uniform(-18, 18))
    if np.random.rand() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def load_dataset(root, size):
    """读所有类别子目录，返回 图 和 标签，以及 索引→类别名 的映射。"""
    root = Path(root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"{root} 下没有类别文件夹")

    X, y, names = [], [], {}
    for idx, d in enumerate(class_dirs):
        names[idx] = d.name
        files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            X.append(preprocess(img, size))
            y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="output/trafficsignnet_torch.pth")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--augment", type=int, default=1, help="1 开启增强，0 关闭")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"设备: {device}")

    size = (args.img_size, args.img_size)
    X, y, names = load_dataset(args.data_root, size)
    num_classes = len(names)
    print(f"数据集: {X.shape}, 类别数: {num_classes}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed, stratify=y
    )

    if args.augment:
        aug_X = [random_augment((x * 255).astype(np.uint8)) for x in X_tr]
        aug_X = np.array([preprocess(a, size) for a in aug_X], dtype=np.float32)
        X_tr = np.concatenate([X_tr, aug_X], axis=0)
        y_tr = np.concatenate([y_tr, y_tr], axis=0)

    def to_loader(Xa, ya, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xa).permute(0, 3, 1, 2),
            torch.from_numpy(ya),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr, True)
    val_loader = to_loader(X_val, y_val, False)

    model = TrafficSignNet(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()

        acc = correct / len(y_val)
        sched.step(acc)
        print(f"epoch {epoch}/{args.epochs} | loss {total_loss / len(y_tr):.4f} | val_acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> 保存 {args.out} (val_acc {acc:.4f})")

    print(f"完成，最佳验证准确率 {best_acc:.4f}")


if __name__ == "__main__":
    main()
