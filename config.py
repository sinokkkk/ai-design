"""统一的数据集根目录和设备选择，避免把本机路径写死在脚本里。

数据集根目录按下面的顺序确定：
1. 命令行传入的参数
2. 环境变量 DATASET_ROOT
3. 项目根目录下的 dataset/（与 German-Traffic_sign_Detection 同级）
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent


def dataset_root(cli_value=None):
    if cli_value:
        return Path(cli_value)
    env = os.environ.get("DATASET_ROOT")
    if env:
        return Path(env)
    return PARENT_ROOT / "dataset"


def pick_device(prefer="auto"):
    """返回可用的 torch 设备字符串：xpu / cuda / cpu。"""
    import torch

    if prefer and prefer != "auto":
        return prefer
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def yolo_device(prefer=None):
    """ultralytics 的设备参数，未指定时交给它自己选。"""
    if prefer:
        return prefer
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu:0"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"
