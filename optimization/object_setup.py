"""Load the canonical MV-SAM3D object (Gaussians and mesh) for Stage 3."""
from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from third_party.SAM3DO.sam3d_objects.model.io import load_model_from_checkpoint
from grag_common.project_paths import SAM3DO_CKPT
from grag_common.runtime_config import device
from optimization.data_loading import ExperimentPaths


def decode_mesh(slat):
    """Decode the object mesh from its structured latent with the SAM3D mesh decoder."""
    cfg_path = Path(SAM3DO_CKPT)
    cfg = OmegaConf.load(cfg_path)
    model = instantiate(OmegaConf.load(cfg_path.parent / cfg["slat_decoder_mesh_config_path"]))
    model = load_model_from_checkpoint(
        model, str(cfg_path.parent / cfg["slat_decoder_mesh_ckpt_path"]),
        strict=True, device="cpu", freeze=True, eval=True, state_dict_key=None,
    ).to(device)
    return model(slat.to(device))[0]


def load_object(paths: ExperimentPaths, num_surface_samples: int = 30_000):
    """Return the object Gaussians, mesh, and surface samples with normals."""
    data = torch.load(paths.slat_coords_init, map_location="cpu", weights_only=False)
    gaussians = data["gs_4"]
    if hasattr(gaussians, "convert_to_fp32"):
        gaussians.convert_to_fp32()
    mesh = data.get("mesh")
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        mesh = decode_mesh(data["slat"])
    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    with torch.no_grad():
        points, normals = sample_points_from_meshes(
            Meshes(verts=[verts], faces=[faces]), num_samples=num_surface_samples, return_normals=True,
        )
    return gaussians, verts, faces, points[0], torch.nn.functional.normalize(normals[0], dim=-1)
