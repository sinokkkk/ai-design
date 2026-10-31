# 交通标志检测与识别

基于 YOLOv8 与 CNN 的两阶段交通标志识别系统。先用 YOLOv8 在道路图像中定位交通标志，再把检测框裁出来交给 CNN 判断具体类别，最后在原图上标出位置和名称。

检测在 GTSDB 上训练，分类在 GTSRB 上训练。全部模型用 PyTorch 运行。

## 运行环境

Python 3.11，PyTorch。设备和数据路径都从命令行传入，不写死在代码里。

安装依赖（conda 环境，以 AutoDL 上为例）：

```bash
conda create -n aidedesign python=3.11 -y
conda activate aidedesign
# AutoDL 镜像一般已带 CUDA 版 torch，没带时按官网命令单独装
pip install ultralytics opencv-python numpy pandas scikit-learn pillow
```

本机用 Intel 核显加速：

```bash
E:/Anaconda/envs/torch-xpu/python.exe gui/app.py
```

## 数据路径

优先级从高到低：

1. 命令行 `--data-root` / `--dataset-root`
2. 环境变量 `DATASET_ROOT`
3. 项目上一级的 `dataset/` 目录

所以本机不用改代码，AutoDL 上只要设一次：

```bash
export DATASET_ROOT=/root/autodl-tmp/dataset
```

## 设备

训练和推理脚本统一用 `--device` 指定，默认 `auto`：

| 参数 | 用途 |
| ---- | ---- |
| `auto` | 自动选 xpu → cuda → cpu |
| `--device 0` | 检测训练用，交给 ultralytics 识别 GPU 序号 |
| `--device cuda:0` | 分类训练和推理用 |
| `--device xpu:0` | 本机 Intel 核显 |
| `--device cpu` | 纯 CPU |

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

# 中国数据集
python detector/prepare_tt100k.py
python detector/train_detector.py --data detector/data_tt100k.yaml --name tt100k_run1
```

`train_detector.py` 支持 `--data` `--weights` `--imgsz` `--epochs` `--batch` `--device` `--name`。

### 分类

数据需按类别分文件夹：`<data-root>/00000/*.ppm`、`<data-root>/00001/*.ppm` …

```bash
python classifier/train_classifier_torch.py                # 默认 GTSRB
python classifier/train_classifier_torch.py --data-root /root/autodl-tmp/dataset/tt100k_cls --out output/tt100k.pth
```

类别数从目录自动读取，换数据集只改 `--data-root`。

## 数据集

- GTSDB：900 张 1360×800 道路全景图，训练集 600 张带标注，标注格式 `文件名;左;上;右;下;类别`。
- GTSRB：43 类交通标志局部图。
- TT100K：清华-腾讯中国交通标志数据集，16811 张，182 类细类。

本机放在 `F:/ai-design/dataset`，AutoDL 上通过 `DATASET_ROOT` 指定。

当前使用的 GTSRB 是 IJCNN 2011 竞赛训练子集，共 26640 张，不是官方 Final Training 的 39209 张。

### TT100K 下载

```bash
mkdir -p ~/autodl-tmp/dataset/tt100k
curl -L -o ~/autodl-tmp/dataset/tt100k/data.zip \
  https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip
unzip -q ~/autodl-tmp/dataset/tt100k/data.zip -d ~/autodl-tmp/dataset/tt100k
```

解压后目录为 `tt100k/data/`，含 `annotations.json` 和 `train/ test/ other/`。

`prepare_tt100k.py` 会生成单类 YOLO 检测集（含负样本）和按类别裁剪的分类图：

```bash
python detector/prepare_tt100k.py --neg-ratio 0.2 --min-count 100
```

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
