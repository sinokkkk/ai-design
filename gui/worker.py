"""推理封装：检测 + 分类，界面层只调用这里，不直接碰 torch。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classifier.model_torch import TrafficSignNet  # noqa: E402


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


class Recognizer:
    def __init__(self, det_weights=None, cls_weights=None, signnames=None):
        det_weights = det_weights or ROOT / "runs/detect/train2/weights/best.pt"
        cls_weights = cls_weights or ROOT / "output/trafficsignnet_v2.pth"
        signnames = signnames or ROOT / "classifier/signnames.csv"

        self.device = get_device()
        self.detector = YOLO(str(det_weights))

        model = TrafficSignNet(num_classes=43)
        model.load_state_dict(torch.load(cls_weights, map_location="cpu"))
        self.classifier = model.eval().to(self.device)

        self.labels = pd.read_csv(signnames).set_index("ClassId")["SignName"].to_dict()
        self.size = (64, 64)

    def _classify(self, crop_bgr):
        crop = clahe_rgb(crop_bgr)
        crop = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.classifier(tensor), dim=1)[0].cpu().numpy()
        cid = int(probs.argmax())
        return cid, float(probs[cid])

    def run(self, frame_bgr, conf=0.25):
        """输入 BGR 图，返回 (带框的图, 结果列表)。结果每项含 bbox/label/det/cls。"""
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        results = self.detector.predict(source=img, conf=conf, iou=0.5, verbose=False)
        items = []

        for r in results:
            if r.boxes is None:
                continue
            for box, det_conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cid, cls_conf = self._classify(crop)
                name = self.labels.get(cid, f"Class {cid}")
                items.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cid,
                    "label": name,
                    "det": float(det_conf),
                    "cls": cls_conf,
                })

                text = f"{name} {cls_conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img, items
