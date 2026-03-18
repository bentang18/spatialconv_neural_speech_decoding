"""Config-driven model assembly.

Takes a YAML config dict and patient grid shapes, returns
(backbone, head, {patient_id: read_in}).
"""
from __future__ import annotations

import torch.nn as nn

from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.models.linear_readin import LinearReadIn
from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.flat_head import FlatCTCHead
from speech_decoding.models.articulatory_head import ArticulatoryCTCHead


def assemble_model(
    config: dict,
    patients: dict[str, tuple[int, int]],
) -> tuple[SharedBackbone, nn.Module, dict[str, nn.Module]]:
    """Build model components from config.

    Args:
        config: Parsed YAML config with 'model' section.
        patients: {patient_id: (grid_h, grid_w)} for each patient.

    Returns:
        (backbone, head, read_ins) where read_ins maps patient_id to module.
    """
    mc = config["model"]

    # Backbone
    backbone = SharedBackbone(
        D=mc["d_shared"],
        H=mc["hidden_size"],
        temporal_stride=mc["temporal_stride"],
        gru_layers=mc["gru_layers"],
        gru_dropout=mc["gru_dropout"],
    )

    # Head
    input_dim = mc["hidden_size"] * 2  # bidirectional
    if mc["head_type"] == "flat":
        head: nn.Module = FlatCTCHead(input_dim=input_dim, num_classes=mc["num_classes"])
    elif mc["head_type"] == "articulatory":
        head = ArticulatoryCTCHead(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown head_type: {mc['head_type']}")

    # Per-patient read-ins
    read_ins: dict[str, nn.Module] = {}
    for pid, (grid_h, grid_w) in patients.items():
        if mc["readin_type"] == "linear":
            lc = mc["linear"]
            read_ins[pid] = LinearReadIn(
                d_padded=lc["d_padded"],
                d_shared=mc["d_shared"],
            )
        elif mc["readin_type"] == "spatial_conv":
            sc = mc["spatial_conv"]
            read_ins[pid] = SpatialConvReadIn(
                grid_h=grid_h,
                grid_w=grid_w,
                C=sc["channels"],
                num_layers=sc["num_layers"],
                kernel_size=sc["kernel_size"],
                pool_h=sc["pool_h"],
                pool_w=sc["pool_w"],
            )
        else:
            raise ValueError(f"Unknown readin_type: {mc['readin_type']}")

    return backbone, head, read_ins
