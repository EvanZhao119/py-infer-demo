# grpc_app/infer.py
import torch
from torchvision import models
from PIL import Image
import io

_MODEL = None
_TRANSFORM = None
_CLASSES = None

def _load_model():
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        weights_path = "/Users/hongnan/EvanShan-Tech/py-infer-demo/grpc_app/models/resnet18-f37072fd.pth"
        _MODEL = models.resnet18()
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        _MODEL.load_state_dict(state_dict, strict=False)
        _MODEL.eval()
        default_weights = models.ResNet18_Weights.DEFAULT
        _TRANSFORM = default_weights.transforms()
    return _MODEL, _TRANSFORM

def _img_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def preprocess_bytes(data: bytes) -> torch.Tensor:
    _, transform = _load_model()
    img = _img_from_bytes(data)
    return transform(img).unsqueeze(0)

@torch.inference_mode()
def predict_bytes(data: bytes, topk: int = 5):
    model, _ = _load_model()
    x = preprocess_bytes(data)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    topk_probs, topk_idxs = torch.topk(probs, k=topk)
    classes = models.ResNet18_Weights.DEFAULT.meta["categories"]
    return [(classes[i], float(topk_probs[j])) for j, i in enumerate(topk_idxs)]
