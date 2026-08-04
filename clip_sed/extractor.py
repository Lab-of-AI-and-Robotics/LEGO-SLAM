"""Online CLIP-SED + HR language feature extractor (for --feature_mode online).

Produces a dense (768, 192, 192) CLIP feature map from an RGB image in real time,
using the SED language model + the high-resolution module from Online Language
Splatting (https://github.com/rpng/online_lang_splatting).

All paths are resolved relative to this file, so the repo is self-contained.
Place the model weights under  clip_sed/weights/  (see README).
"""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# make the vendored `language` package importable (needed to unpickle the SED model)
if _HERE not in sys.path:
    sys.path.append(_HERE)
from language.supervisedNet import LangSupervisedNet

_SED_CKPT = os.path.join(_HERE, "weights", "seg_clip_model_l.pth")
_HR_CKPT = os.path.join(_HERE, "weights", "high_res_71_indoor.ckpt")

_SED_MODEL = None
_HR_MODEL = None


def get_models(device="cuda:0"):
    """Lazy-load SED + HR models once per process."""
    global _SED_MODEL, _HR_MODEL
    if _SED_MODEL is None:
        _SED_MODEL = torch.load(_SED_CKPT, map_location=device)
        _SED_MODEL.eval()
        _HR_MODEL = LangSupervisedNet.load_from_checkpoint(_HR_CKPT).to("cuda").eval()
    return _SED_MODEL, _HR_MODEL


@torch.no_grad()
def extract(image):
    """image: HxWx3 RGB numpy array -> (768, 192, 192) float16 CPU tensor."""
    sed_model, hr_model = get_models()
    H, W = image.shape[:2]
    img_t = torch.from_numpy(image).float().permute(2, 0, 1).cuda()
    dense = sed_model([{"image": img_t, "height": H, "width": W}])[1]
    feat = hr_model(dense["clip_vis_dense"], dense["res3"], dense["res2"])
    return feat.squeeze(0).cpu().half()
