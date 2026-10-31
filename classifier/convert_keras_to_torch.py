"""把 output/trafficsignnet_v2.keras 转成 PyTorch 权重 trafficsignnet_v2.pth。

不需要 TensorFlow，直接读 .keras 里的 model.weights.h5。
用法：python classifier/convert_keras_to_torch.py
"""
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.model_torch import TrafficSignNet

KERAS_PATH = Path("output/trafficsignnet_v2.keras")
OUT_PATH = Path("output/trafficsignnet_v2.pth")


def load_weights_from_keras(path):
    """按 Keras 层名读出权重，返回 {层名: [数组...]}。"""
    with zipfile.ZipFile(path) as z:
        raw = z.read("model.weights.h5")

    weights = {}
    with h5py.File(io.BytesIO(raw), "r") as f:
        if "layers" not in f:
            raise RuntimeError("权重文件里没有 layers 组")

        def collect(name, obj):
            # 路径形如 layers/<层名>/vars/<序号>
            parts = name.split("/")
            if isinstance(obj, h5py.Dataset) and len(parts) == 4 and parts[0] == "layers" and parts[2] == "vars":
                weights.setdefault(parts[1], {})[int(parts[3])] = np.array(obj)

        f.visititems(collect)

    return {k: [v[i] for i in sorted(v)] for k, v in weights.items()}


def keras_to_torch(w):
    """Keras Conv2D 是 HWIO，PyTorch 要 OIHW；Dense 是 (in, out)，要转置。"""
    return torch.from_numpy(np.transpose(w, (3, 2, 0, 1))).contiguous()


def main():
    if not KERAS_PATH.exists():
        raise FileNotFoundError(KERAS_PATH)

    keras_w = load_weights_from_keras(KERAS_PATH)
    model = TrafficSignNet(num_classes=43)
    sd = {}

    # 卷积：keras 里 conv2d / conv2d_1 ... 按出现顺序，PyTorch 里 block1~4 各两个
    conv_names = ["conv2d"] + [f"conv2d_{i}" for i in range(1, 8)]
    conv_targets = [
        "block1.0", "block1.3",
        "block2.0", "block2.3",
        "block3.0", "block3.3",
        "block4.0", "block4.3",
    ]
    for kname, tname in zip(conv_names, conv_targets):
        sd[f"{tname}.weight"] = keras_to_torch(keras_w[kname][0])

    # BN：batch_normalization ... _7，vars 顺序 gamma/beta/moving_mean/moving_variance
    bn_names = ["batch_normalization"] + [f"batch_normalization_{i}" for i in range(1, 8)]
    bn_targets = [
        "block1.1", "block1.4",
        "block2.1", "block2.4",
        "block3.1", "block3.4",
        "block4.1", "block4.4",
    ]
    for kname, tname in zip(bn_names, bn_targets):
        g, b, m, v = keras_w[kname]
        sd[f"{tname}.weight"] = torch.from_numpy(g)
        sd[f"{tname}.bias"] = torch.from_numpy(b)
        sd[f"{tname}.running_mean"] = torch.from_numpy(m)
        sd[f"{tname}.running_var"] = torch.from_numpy(v)
        sd[f"{tname}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # 全连接：dense (256,256) -> fc1，dense_1 (256,43) -> fc2
    w1, b1 = keras_w["dense"]
    w2, b2 = keras_w["dense_1"]
    sd["fc1.weight"] = torch.from_numpy(w1.T).contiguous()
    sd["fc1.bias"] = torch.from_numpy(b1)
    sd["fc2.weight"] = torch.from_numpy(w2.T).contiguous()
    sd["fc2.bias"] = torch.from_numpy(b2)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("缺少:", missing)
        print("多余:", unexpected)
        raise RuntimeError("权重没有完全对上，检查层顺序")

    model.eval()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
