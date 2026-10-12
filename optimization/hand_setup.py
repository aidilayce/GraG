"""Build MANO and align DynHaMR hand tensors to the RGB timeline."""
from __future__ import annotations

import numpy as np
import torch
import yaml

from third_party.Dyn_HaMR.dyn_hamr.body_model import MANO
from grag_common.project_paths import (
    MANO_CFG_PATH,
    MANO_MEAN_PARAMS_PATH,
    MANO_MODEL_PATH,
)
from grag_common.runtime_config import device
from optimization.data_loading import (
    densify_temporal_tensor,
    despike_translation_sequence,
)


def prepare_hand_resources(
    dynhamr_res,
    dynhamr_frame_ids: np.ndarray,
    total_frames: int,
    despike_trans: bool = True,
) -> tuple[MANO, dict]:
    with open(MANO_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    mano_cfg_raw = cfg.get("MANO", {})
    mano_cfg = {k.lower(): v for k, v in mano_cfg_raw.items()}
    mano_cfg["model_path"] = str(MANO_MODEL_PATH)
    if "mean_params" in mano_cfg or MANO_MEAN_PARAMS_PATH.exists():
        mano_cfg["mean_params"] = str(MANO_MEAN_PARAMS_PATH)
    hand_model = MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)

    is_right_idx = 1

    hand_res_tensors: dict[str, torch.Tensor] = {}
    for key, value in dynhamr_res.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim >= 1 and value.shape[0] == 2:
            tensor = torch.from_numpy(value[is_right_idx]).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(value).float()
        tensor = tensor.to(device)
        if key == "betas" and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        hand_res_tensors[key] = tensor

    if "betas" in hand_res_tensors and hand_res_tensors["betas"].ndim == 1:
        hand_res_tensors["betas"] = hand_res_tensors["betas"].unsqueeze(0)

    # Densify kinematics to the RGB timeline.
    temporal_keys = ["is_right", "root_orient", "pose_body", "trans", "cam_R", "cam_t", "latent_pose", "init_body_pose"]
    for key in temporal_keys:
        if key not in hand_res_tensors:
            continue
        # Keep discrete kinematics nearest-neighbor; interpolate only auxiliary features.
        use_linear = key not in {"is_right", "root_orient", "pose_body", "trans", "cam_R", "cam_t"}
        hand_res_tensors[key] = densify_temporal_tensor(
            hand_res_tensors[key],
            dynhamr_frame_ids,
            total_frames,
            is_binary=(key == "is_right"),
            use_linear=use_linear,
        )
    if despike_trans and "trans" in hand_res_tensors:
        trans_before = hand_res_tensors["trans"]
        trans_after = despike_translation_sequence(trans_before)
        delta = torch.linalg.norm(trans_after - trans_before, dim=-1)
        changed = int((delta > 1e-6).sum().item())
        if changed > 0:
            print(f"[INFO] Despiked DynHaMR hand translations on {changed} frames.")
        hand_res_tensors["trans"] = trans_after
    return hand_model, hand_res_tensors
