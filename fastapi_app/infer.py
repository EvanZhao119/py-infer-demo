# fastapi_app/infer.py
import torch
from torchvision import models
from PIL import Image
from typing import List, Tuple

# 加载与 DJL 相同的 ResNet 系列（默认 resnet18，轻量便于演示）
# 你也可以改成 models.resnet50，注意显存与性能差异
_MODEL = None
_TRANSFORM = None
_CLASSES = None  # ImageNet 1000 类

def _load_model():
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        weights_path = "/Users/hongnan/EvanShan-Tech/py-infer-demo/fastapi_app/models/resnet18-f37072fd.pth"
        _MODEL = models.resnet18()
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        _MODEL.load_state_dict(state_dict, strict=False)
        _MODEL.eval()

        default_weights = models.ResNet18_Weights.DEFAULT
        _TRANSFORM = default_weights.transforms()
    return _MODEL, _TRANSFORM

def preprocess(img: Image.Image) -> torch.Tensor:
    _, transform = _load_model()
    return transform(img).unsqueeze(0)  # [1,3,H,W]

@torch.inference_mode()
def predict(img: Image.Image, topk: int = 5) -> List[Tuple[str, float]]:
    model, _ = _load_model()
    x = preprocess(img)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    topk_probs, topk_idxs = torch.topk(probs, k=topk)
    # 类名从权重中拿（与 torchvision 权重一致）
    classes = models.ResNet18_Weights.DEFAULT.meta["categories"]
    return [(classes[i], float(topk_probs[j])) for j, i in enumerate(topk_idxs)]
