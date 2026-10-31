import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.model_torch import TrafficSignNet


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


def preprocess_for_classifier(crop_bgr, size, device):
    crop_bgr = clahe_rgb(crop_bgr)
    crop = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test_images/test_image_1.png")
    parser.add_argument("--save", default=None, help="结果图保存路径，默认 output/pred_<文件名>")
    parser.add_argument("--det", default="runs/detect/train2/weights/best.pt")
    parser.add_argument("--cls", default="output/trafficsignnet_v2.pth")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    device = get_device()
    det = YOLO(args.det)

    model = TrafficSignNet(num_classes=43)
    model.load_state_dict(torch.load(args.cls, map_location="cpu"))
    model.eval().to(device)
    size = (64, 64)

    signnames = pd.read_csv("classifier/signnames.csv")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    results = det.predict(source=img, conf=args.conf, iou=0.5, verbose=False)

    found_any = False
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inp = preprocess_for_classifier(crop, size, device)
            with torch.no_grad():
                probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()

            class_id = int(probs.argmax())
            score = float(probs[class_id])
            name = signnames.loc[signnames["ClassId"] == class_id, "SignName"].values[0]

            print("Top-3 predictions:")
            for idx in np.argsort(probs)[-3:][::-1]:
                n = signnames.loc[signnames["ClassId"] == int(idx), "SignName"].values[0]
                print(f"  {n}: {probs[idx]:.3f}")
            print(f"Traffic sign: {name} | det={conf:.2f} | cls={score:.2f}")

            label = f"{name} det={conf:.2f} cls={score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            found_any = True

    if not found_any:
        print("No traffic sign detected.")

    save_path = args.save or f"output/pred_{Path(args.image).stem}.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"结果已保存: {save_path}")


if __name__ == "__main__":
    main()
