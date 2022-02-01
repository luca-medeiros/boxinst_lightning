from modeling.fcos.fcos import FCOS
from layers import ShapeSpec

def build_proposal_generator(cfg):
    shape_spec = [ShapeSpec(channels=256)] * len(cfg.model.rpn.fcos.in_features)
    fcos = FCOS(cfg, shape_spec)
    return fcos