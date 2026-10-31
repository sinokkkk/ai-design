from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
metrics = model.val(data="detector/data.yaml", split="val")

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)