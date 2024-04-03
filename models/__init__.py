import torch

from models.backbone import build_backbone
from models.backbone2 import build_backbone2
from models.counter import get_counter
from models.transformer import build_transformer
from models.epf_extractor import build_epf_extractor

from models.matcher import build_matcher
from models.class_agnostic_counting_model import CACModel


def build_model(cfg):
    backbone = build_backbone(cfg)
    backbone2 = build_backbone2(cfg)
    epf_extractor = build_epf_extractor(cfg)
    transformer = build_transformer(cfg)

    matcher = build_matcher(cfg)
    counter = get_counter(cfg)
    model = CACModel(backbone, backbone2, epf_extractor, transformer, matcher, counter, cfg.MODEL.hidden_dim)

    return model


