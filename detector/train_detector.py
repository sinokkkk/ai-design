"""YOLOv8 检测训练。数据集用 --data 指定，默认德国 GTSDB。

用法：
    python detector/train_detector.py
    python detector/train_detector.py --data detector/data_tt100k.yaml
    python detector/train_detector.py --data detector/data_tt100k.yaml --device xpu:0 --epochs 100
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="detector/data.yaml", help="数据集 yaml 路径")
    p.add_argument("--weights", default="yolov8n.pt", help="预训练权重")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="xpu:0", help="xpu:0 / 0 / cpu")
    p.add_argument("--name", default=None, help="本次训练的输出目录名")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
