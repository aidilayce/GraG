"""Render solid mesh overlays directly from exported camera-space arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


HAND_COLOR = (0.75, 0.80, 0.90)
OBJECT_COLOR = (0.90, 0.35, 0.30)


def binary_mask(mask: np.ndarray, *, object_mask: bool) -> np.ndarray:
    """Decode binary masks and HO3D's legacy 0/50/150 label masks."""
    values = set(np.unique(mask).tolist())
    if object_mask and 50 in values and values.issubset({0, 50, 150}):
        return (mask == 50).astype(np.uint8) * 255
    return (mask > 0).astype(np.uint8) * 255


class SolidRenderer:
    def __init__(self, faces_h: np.ndarray, faces_o: np.ndarray,
                 image_size: tuple[int, int]) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height, self.width = image_size
        self.faces_h = torch.as_tensor(faces_h, device=self.device, dtype=torch.int64)
        self.faces_o = torch.as_tensor(faces_o, device=self.device, dtype=torch.int64)
        lights = DirectionalLights(
            device=self.device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((1.0, 1.0, 1.0),),
            direction=((0.0, 0.0, -1.0),),
        )
        raster = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
            bin_size=32 if self.device.type == "cuda" else 0,
            max_faces_per_bin=len(faces_h) + len(faces_o),
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(raster_settings=raster),
            HardPhongShader(
                device=self.device, lights=lights,
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4),
            ),
        )

    def camera(self, K: np.ndarray):
        rotation = torch.eye(3, device=self.device)[None]
        translation = torch.zeros(1, 3, device=self.device)
        matrix = torch.as_tensor(K, device=self.device, dtype=torch.float32)[None]
        size = torch.tensor([[self.height, self.width]], device=self.device).float()
        return cameras_from_opencv_projection(rotation, translation, matrix, size)

    def mesh(self, vertices: torch.Tensor, faces: torch.Tensor,
             color: tuple[float, float, float]) -> Meshes:
        features = torch.tensor(color, device=self.device)[None, None]
        features = features.expand(1, len(vertices), 3).contiguous()
        return Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(features))

    def render(self, hand: np.ndarray, obj: np.ndarray, K: np.ndarray,
               background: np.ndarray | None, side: bool = False) -> np.ndarray:
        hand_t = torch.as_tensor(hand, device=self.device, dtype=torch.float32)
        obj_t = torch.as_tensor(obj, device=self.device, dtype=torch.float32)
        if side:
            center = torch.cat([hand_t, obj_t]).mean(0, keepdim=True)
            rotation = torch.tensor(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                device=self.device,
            )
            hand_t = (hand_t - center) @ rotation.T + center
            obj_t = (obj_t - center) @ rotation.T + center
            points = torch.cat([hand_t, obj_t])
            xy_center = points[:, :2].mean(0)
            hand_t[:, :2] -= xy_center
            obj_t[:, :2] -= xy_center
            points = torch.cat([hand_t, obj_t])
            fx, fy = float(K[0, 0]), float(K[1, 1])
            fit_z = max(
                0.02,
                fx * points[:, 0].abs().max().item() / (0.44 * self.width),
                fy * points[:, 1].abs().max().item() / (0.44 * self.height),
            )
            shift = fit_z - points[:, 2].min().item()
            hand_t[:, 2] += shift
            obj_t[:, 2] += shift
        scene = join_meshes_as_scene([
            self.mesh(obj_t, self.faces_o, OBJECT_COLOR),
            self.mesh(hand_t, self.faces_h, HAND_COLOR),
        ])
        with torch.no_grad():
            rgba = self.renderer(scene, cameras=self.camera(K))[0].clamp(0, 1).cpu().numpy()
        base = np.zeros_like(rgba[..., :3]) if background is None else background / 255.0
        result = rgba[..., :3] * rgba[..., 3:] + base * (1.0 - rgba[..., 3:])
        return (result.clip(0, 1) * 255).astype(np.uint8)


def main(args: argparse.Namespace) -> None:
    pred = Path(args.pred_dir) if args.pred_dir else Path(args.artifact_root) / args.dataset / args.seq_name
    rgb_dir = Path(args.demo_root) / args.seq_name / "rgb"
    vertices_h = np.load(pred / "pred_v3d_c_right.npy")
    vertices_o = np.load(pred / "pred_v3d_c_object.npy")
    faces_h = np.load(pred / "pred_faces_right.npy")
    faces_o = np.load(pred / "pred_faces_object.npy")
    pointmaps = torch.load(
        Path(args.demo_root) / args.seq_name / "da3_pointmaps_intrinsics_extrinsics.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    K = np.asarray(pointmaps["intrinsics"])
    rgb = sorted(rgb_dir.glob("*.png"))
    obj_masks = {p.stem: p for p in (Path(args.demo_root) / args.seq_name / "masks").glob("*.png")}
    hand_masks = {p.stem: p for p in (Path(args.demo_root) / args.seq_name / "hand_masks").glob("*.png")}
    if not rgb:
        raise RuntimeError(f"No input frames found in {rgb_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first_frame = cv2.imread(str(rgb[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read {rgb[0]}")
    renderer = SolidRenderer(faces_h, faces_o, first_frame.shape[:2])
    input_writer = imageio.get_writer(
        out / f"{args.seq_name}_input_camera_overlay.mp4", fps=args.fps
    )
    side_writer = imageio.get_writer(out / f"{args.seq_name}_side_view.mp4", fps=args.fps)
    try:
        start_frame = max(0, int(args.start_frame))
        frame_count = len(vertices_h) - start_frame
        if args.max_frames is not None:
            frame_count = min(frame_count, max(0, int(args.max_frames)))
        for t in range(start_frame, start_frame + max(0, frame_count)):
            path = rgb[t]
            frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            frame = renderer.render(vertices_h[t], vertices_o[t], K[t], frame)
            for masks, color, is_object in (
                (hand_masks, (80, 255, 80), False),
                (obj_masks, (255, 255, 80), True),
            ):
                if path.stem not in masks:
                    continue
                mask = cv2.imread(str(masks[path.stem]), cv2.IMREAD_GRAYSCALE)
                mask = binary_mask(mask, object_mask=is_object)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)
            input_writer.append_data(frame)
            side_writer.append_data(
                renderer.render(vertices_h[t], vertices_o[t], K[t], None, side=True)
            )
    finally:
        input_writer.close()
        side_writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq_name", required=True)
    p.add_argument("--pred_dir", type=Path)
    p.add_argument("--artifact_root", type=Path,
                   help="Hugging Face bundle root; replaces --pred_dir")
    p.add_argument("--dataset", choices=["HO3D", "HOT3D"],
                   help="Dataset subdirectory under --artifact_root")
    p.add_argument("--demo_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_frames", type=int, default=None,
                   help="optional frame cap for quick diagnostic renders")
    p.add_argument("--start_frame", type=int, default=0,
                   help="first frame to render (for interval diagnostics)")
    args = p.parse_args()
    if not args.pred_dir and not (args.artifact_root and args.dataset):
        p.error("provide --pred_dir or both --artifact_root and --dataset")
    main(args)
