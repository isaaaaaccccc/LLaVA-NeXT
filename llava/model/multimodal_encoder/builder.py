import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(config, delay_load=False, **kwargs):
    vision_tower = getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    vision_tower_cfg = getattr(config, "mm_vision_tower_cfg", None) or config

    tower_type = getattr(config, "mm_vision_tower_type", None) or getattr(config, "vision_tower_type", None)

    if isinstance(tower_type, str):
        tower_type = tower_type.lower()

    # Resolve relative local paths safely
    if isinstance(vision_tower, str):
        if os.path.exists(vision_tower):
            vision_tower = os.path.abspath(vision_tower)
        else:
            rel = os.path.join(os.path.dirname(__file__), "..", "..", vision_tower)
            rel = os.path.abspath(rel)
            if os.path.exists(rel):
                vision_tower = rel

    if tower_type == "siglip":
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, delay_load=delay_load, **kwargs)

    if isinstance(vision_tower, str) and "siglip" in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, delay_load=delay_load, **kwargs)

    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, delay_load=delay_load, **kwargs)

