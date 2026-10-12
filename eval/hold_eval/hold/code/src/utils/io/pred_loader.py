import os
import os.path as op
from glob import glob
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from hold.common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers


import argparse
from pathlib import Path

import numpy as np
import torch
import sys


from common.xdict import xdict
from src.utils.io import gt as gt_io
from src.utils.eval_modules import (
    convert_to_tensors,
    compute_bounding_box_centers,
    eval_cd_f_ra,
    eval_cd_f_right,
    eval_mpjpe_right,
)

def load_data(pred_dir: str):
    pred_dir = Path(pred_dir)

    v3d_c_object = np.load(pred_dir / "pred_v3d_c_object.npy") #[:89]   # (T,Vo,3)
    v3d_c_right  = np.load(pred_dir / "pred_v3d_c_right.npy") #[:89]    # (T,Vh,3)
    j3d_c_right  = np.load(pred_dir / "pred_j3d_c_right.npy") #[:89]    # (T,J,3)

    data_pred = xdict()
    data_pred["v3d_c.object"] = torch.from_numpy(v3d_c_object).float()
    data_pred["v3d_c.right"]  = torch.from_numpy(v3d_c_right).float()
    data_pred["j3d_c.right"]  = torch.from_numpy(j3d_c_right).float()
    data_pred["full_seq_name"] = str(pred_dir).split('/')[-1] # when pred dir is like '/path/to/scratch'

    faces = {}
    faces["object"] = torch.from_numpy(np.load(pred_dir / "pred_faces_object.npy")).long()
    faces["right"]  = torch.from_numpy(np.load(pred_dir / "pred_faces_right.npy")).long()
    data_pred['faces'] = faces

    # --- roots & root-relative, same convention as HOLD ---

    # object root = bbox center per frame
    root_object_np = compute_bounding_box_centers(v3d_c_object)    # (T,3)
    root_object = torch.from_numpy(root_object_np).float()
    data_pred["root.object"]   = root_object
    data_pred["v3d_ra.object"] = data_pred["v3d_c.object"] - root_object[:, None, :]

    # hand root = first joint (wrist)
    root_right_np = j3d_c_right[:, 0, :]                           # (T,3)
    root_right = torch.from_numpy(root_right_np).float()
    data_pred["root.right"]    = root_right
    data_pred["j3d_ra.right"]  = data_pred["j3d_c.right"] - data_pred["j3d_c.right"][:, :1, :]

    # object vertices in hand-root frame
    data_pred["v3d_right.object"] = data_pred["v3d_c.object"] - data_pred["root.right"][:, None, :]

    return convert_to_tensors(data_pred)
