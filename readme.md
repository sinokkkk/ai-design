# 交通标志检测与识别

基于 YOLOv8 与 CNN 的两阶段交通标志识别系统。先用 YOLOv8 在道路图像中定位交通标志，再把检测框裁出来交给 CNN 判断具体类别，最后在原图上标出位置和名称。

检测在 GTSDB 上训练，分类在 GTSRB 上训练。全部模型用 PyTorch 运行。

## 运行环境

Python 3.11，PyTorch。本机使用 Intel 核显加速：

```bash
E:/Anaconda/envs/torch-xpu/python.exe gui/app.py
```

依赖：`torch`、`ultralytics`、`opencv-python`、`numpy`、`pandas`、`scikit-learn`、`pillow`。

## 目录结构

```
German-Traffic_sign_Detection/
├── classifier/
│   ├── model_torch.py              # CNN 网络结构（4 卷积块 + 全连接）
│   ├── train_classifier_torch.py   # 分类训练脚本
│   ├── convert_keras_to_torch.py   # 旧 Keras 权重转 PyTorch
│   └── signnames.csv               # 43 类名称表
├── detector/
│   ├── convert_gtsdb_to_yolo.py    # GTSDB 标注转 YOLO 格式
│   ├── convert_ppm_to_png.py       # ppm 转 png（ultralytics 不支持 ppm）
│   ├── data.yaml                   # 检测数据集配置
│   └── train_detector.py           # 检测训练脚本
├── pipeline/
│   ├── infer_detect_classify.py    # 命令行单图推理
│   ├── metrics_extraction.py       # 检测指标
│   ├── Classifier_metrics.py       # 分类指标
│   └── make_splits.py              # 按类别分层划分数据集
├── gui/
│   ├── app.py                      # 图形界面
│   └── worker.py                   # 推理封装
├── output/
│   ├── trafficsignnet_v2.pth       # 分类权重
│   └── upstream/                   # 上游原始 Keras 权重（仅存档）
├── runs/detect/train2/weights/best.pt  # 检测权重
└── test_images/                    # 测试图片
```

## 使用

单张图片：

```bash
E:/Anaconda/envs/torch-xpu/python.exe pipeline/infer_detect_classify.py --image test_images/test_image_1.png
```

结果保存在 `output/pred_<文件名>.png`。

图形界面：

```bash
E:/Anaconda/envs/torch-xpu/python.exe gui/app.py
```

支持选择图片或视频、调整置信度、查看每个标志的类别和置信度、保存结果。

## 训练

### 检测

```bash
python detector/convert_gtsdb_to_yolo.py
python detector/convert_ppm_to_png.py
python detector/train_detector.py
```

### 分类

数据需按类别分文件夹：`<data-root>/00000/*.ppm`、`<data-root>/00001/*.ppm` …

```bash
python classifier/train_classifier_torch.py --data-root F:/ai-design/dataset/gtsrb_train/GTSRB/Training
```

换数据集只需改 `--data-root`，类别数从目录自动读取：

```bash
python classifier/train_classifier_torch.py --data-root F:/ai-design/dataset/china --out output/china.pth
```

## 数据集

- GTSDB：900 张 1360×800 道路全景图，训练集 600 张带标注，标注格式 `文件名;左;上;右;下;类别`。
- GTSRB：43 类交通标志局部图。

本机数据集放在 `F:/ai-design/dataset`。

当前使用的 GTSRB 是 IJCNN 2011 竞赛训练子集，共 26640 张，不是官方 Final Training 的 39209 张。

## 指标

检测（102 张验证图、163 个框）：

| Precision | Recall | mAP50 | mAP50-95 |
| --------- | ------ | ----- | -------- |
| 0.953     | 0.987  | 0.991 | 0.883    |

分类的验证结果见 `pipeline/Classifier_metrics.py`。当前分类权重来自上游，其训练数据与本地验证集高度重合，准确率偏乐观，重训后应重新统计。

## 已删除的上游文件

原仓库是 TensorFlow/Keras 实现，本项目改为纯 PyTorch，删除了以下脚本和文件。为便于溯源，保留记录：

| 原文件 | 说明 | 替代 |
| ------ | ---- | ---- |
| `classifier/train_classifier.py` | Keras 分类训练（初版，3 层卷积） | `classifier/train_classifier_torch.py` |
| `classifier/train_classifier_v2.py` | Keras 分类训练（改进版，4 卷积块） | 同上 |
| `pipeline/detect_image_sign.py` | Keras 推理，写死 32×32 输入 | `pipeline/infer_detect_classify.py` |
| `pipeline/classifier_alone test.py` | Keras 单图分类，同上问题 | 同上 |
| `output/trafficsignnet.keras` | 初版分类权重 | 移至 `output/upstream/` |
| `output/trafficsignnet_v2.keras` | 改进版分类权重，现有 `.pth` 的来源 | 移至 `output/upstream/` |
| `runs/detect/train/` | 与 `train2` 完全相同的权重副本 | 保留 `train2` |
| `debug_crop.png` | 调试时输出的裁剪图 | 无 |
