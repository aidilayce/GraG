# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Transform3d, quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, quaternion_invert

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area
from sam3d_objects.utils.visualization import SceneVisualizer



def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap_fromVideo(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)

        # converted below to take from function args
        # # NOTE: change strength for sparse structure sampling
        # self.ss_cfg_strength = 1.0 # 7.0 originally
        # self.ss_cfg_strength_pm = 0.0 # 0.0 originally
        # self.ss_inference_steps = 25 # 25 originally
        # self.ss_cfg_interval = [0, 500] # [0, 500] originally


    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        ss_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _get_canonical_object_points(self, outputs) -> torch.Tensor:
        """
        Extract a canonical object point cloud (in object space) from the decoded Gaussians.
        Assumes outputs["gaussian"] is a Gaussians object with get_xyz.
        Returns (M, 3) on self.device.
        """
        gaussian = outputs["gaussian"][0]  # adjust key if needed
        xyz = gaussian.get_xyz  # typically (N, 3) or (1, N, 3)

        if xyz.dim() == 3:
            # assume (B, N, 3)
            xyz = xyz[0]
        xyz = xyz.to(self.device)
        return xyz 
    
    def _refine_pose_with_pointmap(
        self,
        X_cam: torch.Tensor,          # (M, 3), posed object points (Gaussian means) in pointmap/cam space
        t_init: torch.Tensor,         # (1, 3)
        R_init: torch.Tensor,         # (1, 4) quaternion OR (1, 3, 3) matrix
        scale: torch.Tensor,          # (1, 3) or scalar tensor
        pointmap: torch.Tensor,       # (3, H, W) in camera coords
        mask_np: np.ndarray,          # (H, W) numpy mask for object
        icp_iterations: int = 5,
    ):
        from pytorch3d.ops import knn_points
        device = pointmap.device
        X_cam = X_cam.to(device)

        # --- 1. Build Y_cam from pointmap and mask ---
        # pointmap: (3, H, W) -> (H, W, 3)
        pm_hwc = pointmap.permute(1, 2, 0)  # (H, W, 3)
        mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        if mask_t.shape != pm_hwc.shape[:2]:
            # If sizes differ, resize mask to match pointmap
            mask_t = torchvision.transforms.functional.resize(
                mask_t.float().unsqueeze(0).unsqueeze(0),
                pm_hwc.shape[:2],
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ).squeeze(0).squeeze(0).bool()

        Y_cam = pm_hwc[mask_t]  # (K, 3)
        Y_cam = Y_cam[torch.isfinite(Y_cam).all(dim=-1)]

        M = X_cam.shape[0]
        if M > Y_cam.shape[0]:
            idx = torch.randperm(M, device=device)[:Y_cam.shape[0]]
            X_cam = X_cam[idx]

        if R_init.dim() == 2 and R_init.shape[-1] == 4:
            R_current = quaternion_to_matrix(R_init)[0]  # (3, 3)
        elif R_init.dim() == 3:
            R_current = R_init[0]
        else:
            raise ValueError(f"Unexpected rotation shape {R_init.shape}")

        t_current = t_init.squeeze()
        X_curr = X_cam.clone()

        for iter in range(icp_iterations):
            nn_result = knn_points(X_curr.unsqueeze(0), Y_cam.unsqueeze(0), K=1)
            nn_idx = nn_result.idx[0, :, 0]  # (M,)
            Y_matched = Y_cam[nn_idx]  # (M, 3)

            mu_X = X_curr.mean(dim=0, keepdim=True)
            mu_Y = Y_matched.mean(dim=0, keepdim=True)

            Xc = X_curr - mu_X
            Yc = Y_matched - mu_Y

            H = Xc.T @ Yc
            U, _, Vt = torch.linalg.svd(H)
            R_delta = Vt.T @ U.T

            if torch.linalg.det(R_delta) < 0:
                Vt[-1, :] *= -1
                R_delta = Vt.T @ U.T

            t_delta = mu_Y.squeeze() - (R_delta @ mu_X.T).squeeze()
            X_curr = (R_delta @ X_curr.T).T + t_delta

            R_current = R_delta @ R_current
            t_current = (R_delta @ t_current.unsqueeze(-1)).squeeze() + t_delta

        # Pack back to shapes (1, 3, 3) / (1, 4)
        R_final = R_current.unsqueeze(0)  # (1, 3, 3)
        t_final = t_current.unsqueeze(0)  # (1, 3)

        # If original rotation was quaternion, convert back
        if R_init.dim() == 2 and R_init.shape[-1] == 4:
            q_final = matrix_to_quaternion(R_final)[0].unsqueeze(0)  # (1, 4)
            return t_final, q_final
        else:
            return t_final, R_final

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)

        return item

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def pose_gaussians(self, gaussian, rotation, translation, scale, in_place=False):
        if not in_place:
            gaussian = deepcopy(gaussian)

        minimum_kernel_size = float("inf")
        if translation.ndim == 1:
            translation = translation.unsqueeze(0)
        if rotation.ndim == 1:
            rotation = rotation.unsqueeze(0)
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)

        PC = SceneVisualizer.object_pointcloud(
                points_local=gaussian.get_xyz.unsqueeze(0),
                quat_l2c=rotation,
                trans_l2c=translation,
                scale_l2c=scale,
            )
        gaussian.from_xyz(PC.points_list()[0])
        # must ... ROTATE
        gaussian.from_rotation(
            quaternion_multiply(
                quaternion_invert(rotation),
                gaussian.get_rotation,
            )
        )
        gaus_scale = gaussian.get_scaling
        
        adjusted_scale = gaus_scale * scale
        assert (scale[0, 0].item() == scale[0, 1].item() == scale[0, 2].item())
        gaussian.mininum_kernel_size *= scale[0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                gaussian.mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        gaussian.from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            gaussian.mininum_kernel_size,
        )
        gaussian.mininum_kernel_size = minimum_kernel_size
        return gaussian

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None

        points_tensor = points_tensor.permute(2, 0, 1).to(torch.float32)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask)
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

        return point_map_tensor

    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal
        revised_quat, revised_t, revised_scale, final_iou, _, _ = (
            self.layout_post_optimization_method(
                mesh_glb,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_pointmap"][0].permute(1, 2, 0),
                intrinsics,
                min_size=518,
            )
        )
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }
    
    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image
    
    def run_selected_view(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])
            
            if estimate_plane:
                return self.estimate_plane(pointmap_dict, image)

            # NOTE: this one should run per-frame to get pointmap and pointmap scale/shift
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor, pointmap=pointmap)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor) # pointmap, scale, shift stuff
            if seed is not None:
                torch.manual_seed(seed)

            # NOTE: this one has shape, coords, and downsample factor. should be sampled ONCE per video using selected view
            ss_return_dict = self.sample_sparse_structure( # shape features
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            ref_pose_dict = self.pose_decoder(
                ss_return_dict,
                scene_scale=ss_input_dict.get("pointmap_scale", None),
                scene_shift=ss_input_dict.get("pointmap_shift", None),
            )
            ref_scale_raw = ref_pose_dict["scale"]
            ref_scale = ref_scale_raw * ss_return_dict["downsample_factor"]

            # NOTE: ONCE per video using selected view
            coords = ss_return_dict["coords"]
            slat = self.sample_slat( # texture features
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )

            return ss_return_dict, outputs, slat, ref_scale, pts.cpu().permute((1, 2, 0)),  pts_colors.cpu().permute((1, 2, 0))
    
    def get_pose_per_frame(
        self,
        images: Union[None, Image.Image, np.ndarray],
        masks: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        pointmaps=None,
        selected_view_idx=0,
        sam3d_pose_inference_steps: int = 25,
        ss_cfg_strength: float = 1.0,
        ss_cfg_strength_pm: float = 0.0,
        temporal_guidance_lambda: float = 0.0,
        decode_formats=['gaussian_4','gaussian', 'mesh'],
        refine_w_icp=False,
        only_key_frame=False,
        shape_mode='fixed', # 'fixed', 'trajectory', 'sam3d'
        use_prev_pose=False,
        scale_mode: str = "video_median",  # 'video_median', 'per_frame', 'first_frame', 'mv_sam3d'
        lock_scale_during_pose: bool = False,
        calibrate_scale_with_selected_view: bool = False,
        mv_sam3d_ss_path=None,
        mv_sam3d_slat_path=None,
        debug=False,
    ) -> dict:
        # DEBUGGING DIR
        if debug:
            debug_dir = mv_sam3d_slat_path.replace('slat.pth', '') + '_debug_poses'
            os.makedirs(debug_dir, exist_ok=True)

        self.ss_inference_steps = sam3d_pose_inference_steps
        self.ss_cfg_strength = ss_cfg_strength
        # Pointmap-only CFG is a separate control from generic CFG.
        self.ss_cfg_strength_pm = ss_cfg_strength_pm

        trajectory = None
        if only_key_frame:
            geo_output = self.run_selected_view(image=images[selected_view_idx], mask=masks[selected_view_idx], pointmap=pointmaps[selected_view_idx], decode_formats=decode_formats)
            ss_return_dict0, outputs, slat, ref_scale, _, _ = geo_output
        else:
            if not mv_sam3d_ss_path or not mv_sam3d_slat_path:
                raise ValueError("MV-SAM3D result paths are required for full-video inference")
            ss_return_dict0 = torch.load(mv_sam3d_ss_path)
            slat = torch.load(mv_sam3d_slat_path)
            mv_sam3d_res_dir = mv_sam3d_slat_path.replace('/slat.pth','')
            trajectory_dict = torch.load(f'{mv_sam3d_res_dir}/ss_sampling_trajectory.pth')
            trajectory = trajectory_dict['traj']
            t_steps = trajectory_dict['traj_t'] # for debugging
            
        coords = ss_return_dict0["coords"]
        ref_scale = ss_return_dict0['scale']
        shape_latent_star = ss_return_dict0['shape'] if shape_mode == 'fixed' else None
        freeze_shape = True if shape_mode == 'fixed' else False
        match_shape_tracjectory = True if shape_mode == 'trajectory' else False
        # ss_return_dict --> (['6drotation_normalized', 'scale', 'shape', 'translation', 'translation_scale', 'coords_original', 'coords', 'downsample_factor'])

        if scale_mode not in {"video_median", "per_frame", "first_frame", "mv_sam3d"}:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        poses = {}
        first_frame_scale = None
        selected_frame_scale = None
        frame_scales = []
        locked_scale_latent = None
        if lock_scale_during_pose and isinstance(ss_return_dict0, dict) and ("scale" in ss_return_dict0):
            locked_scale_latent = ss_return_dict0["scale"]
            if torch.is_tensor(locked_scale_latent):
                locked_scale_latent = locked_scale_latent.detach().clone()
                if locked_scale_latent.ndim == 1:
                    locked_scale_latent = locked_scale_latent.unsqueeze(0)
        prev_pose_dict_latents = None
        # run per-frame using the same shape latents
        for i, (image, mask, pointmap) in enumerate(zip(images, masks, pointmaps)):
            image = self.merge_mask_to_rgba(image, mask)
            with self.device: 
                pointmap_dict = self.compute_pointmap(image, pointmap)
                pointmap = pointmap_dict["pointmap"]
                pts = type(self)._down_sample_img(pointmap)
                pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])
                
                ss_input_dict_i = self.preprocess_image(image, self.ss_preprocessor, pointmap=pointmap) # pointmap, scale, shift stuff
                if seed is not None:
                    torch.manual_seed(seed)

                pointmap_scale = ss_input_dict_i.get("pointmap_scale", None)
                pointmap_shift = ss_input_dict_i.get("pointmap_shift", None)

                # ### vel guidance and save denoising steps
                # bb = self.models["ss_generator"].reverse_fn.backbone
                # bb.x_trace = None
                # bb.t_trace = None
                # chosen_frames = {29, 30, 31, 61, 62, 63}  # whatever you want
                # bb.record_x_trace = (i in chosen_frames)
                # bb.vel_trace = None
                # bb.record_vel_trace = True
                # # hybrid guidance only after first frame has prev trace
                # bb.temporal_guidance = (i > 0 and bb.prev_vel_trace is not None)
                # bb.temporal_guidance_lambda = 1.0      # alpha multiplier (start at 1.0)
                # bb.temporal_guidance_keys = ["translation", "6drotation_normalized"]
                # bb.hybrid_prev_steps = 3
                # bb.hybrid_total_steps = self.ss_inference_steps
                # bb.hybrid_mode = "hard"
                ### end
                temporal_guidance = False
                if temporal_guidance_lambda > 0.0:
                    use_prev_pose = True
                    temporal_guidance = True
                ss_return_dict_i = self.sample_sparse_structure(
                    ss_input_dict_i,
                    inference_steps=self.ss_inference_steps, # NOTE: changed from None
                    shape_latent_star=shape_latent_star, 
                    freeze_shape=freeze_shape,
                    use_prev_pose=use_prev_pose,
                    prev_pose_dict=prev_pose_dict_latents,
                    temporal_guidance=temporal_guidance,
                    temporal_guidance_lambda=temporal_guidance_lambda,
                    temporal_guidance_keys=["translation","6drotation_normalized"],
                    match_shape_tracjectory=match_shape_tracjectory,
                    trajectory=trajectory,
                    record_vel_trace=False,
                    )

                if lock_scale_during_pose and "scale" in ss_return_dict_i:
                    if locked_scale_latent is None:
                        locked_scale_latent = ss_return_dict_i["scale"].detach().clone()
                        if locked_scale_latent.ndim == 1:
                            locked_scale_latent = locked_scale_latent.unsqueeze(0)
                    else:
                        ss_return_dict_i["scale"] = locked_scale_latent.to(
                            device=ss_return_dict_i["scale"].device,
                            dtype=ss_return_dict_i["scale"].dtype,
                        )
                
                prev_pose_dict_latents = {
                    '6drotation_normalized': ss_return_dict_i['6drotation_normalized'],
                    'scale': ss_return_dict_i['scale'],
                    'translation': ss_return_dict_i['translation'],
                    'translation_scale': ss_return_dict_i['translation_scale'],
                }
                # self.ss_inference_steps = 10  # for next frames, only 5 step to refine
                
                pose_dict = self.pose_decoder(ss_return_dict_i, scene_scale=pointmap_scale, scene_shift=pointmap_shift)
                
                # pose_dict = self.pose_decoder(ss_return_dict0, scene_scale=pointmap_scale, scene_shift=pointmap_shift)
                t_i, R_i = pose_dict["translation"], pose_dict["rotation"]
                s_i = pose_dict["scale"]
                frame_scales.append(s_i.detach().clone())
                if first_frame_scale is None:
                    first_frame_scale = s_i.detach().clone()
                if i == selected_view_idx:
                    selected_frame_scale = s_i.detach().clone()
                
                # if refine_w_icp:
                #     canon_obj_gaussian = outputs['gaussian_4'][0]
                #     posed_obj_gaussian = self.pose_gaussians(canon_obj_gaussian, R_i, t_i, s_i)
                #     X_obj_cam = posed_obj_gaussian.get_xyz.to(self.device)
                #     t_i, R_i = self._refine_pose_with_pointmap(X_obj_cam, t_i, R_i, s_i, pointmap, mask)

                poses[f'{i}'] = {
                    "translation": t_i,
                    "rotation": R_i,
                    "scale": s_i,
                }

        if len(frame_scales) > 0:
            shared_scale = None
            if scale_mode == "first_frame":
                shared_scale = first_frame_scale
            elif scale_mode == "mv_sam3d":
                shared_scale = ss_return_dict0.get("scale", None)
                if shared_scale is None:
                    shared_scale = first_frame_scale
                elif shared_scale.ndim == 1:
                    shared_scale = shared_scale.unsqueeze(0)
            elif scale_mode == "video_median":
                stacked_scales = torch.cat(
                    [s if s.ndim == 2 else s.unsqueeze(0) for s in frame_scales], dim=0
                )
                # Robust aggregation: reject invalid/outlier frames and use a conservative quantile.
                valid_rows = torch.isfinite(stacked_scales).all(dim=1) & (stacked_scales > 0).all(dim=1)
                candidate_scales = stacked_scales[valid_rows]

                if candidate_scales.shape[0] > 0:
                    scalar_scales = candidate_scales.mean(dim=1)  # isotropic scale proxy

                    # Anchor-based filtering around selected-view frame when available.
                    if selected_frame_scale is not None and torch.isfinite(selected_frame_scale).all():
                        anchor = selected_frame_scale.mean().clamp_min(1e-6)
                        ratio = scalar_scales / anchor
                        anchor_inliers = (ratio >= 0.6) & (ratio <= 1.8)
                        if anchor_inliers.any():
                            candidate_scales = candidate_scales[anchor_inliers]
                            scalar_scales = scalar_scales[anchor_inliers]

                    # MAD-based outlier filtering.
                    med = scalar_scales.median()
                    mad = (scalar_scales - med).abs().median().clamp_min(1e-6)
                    mad_inliers = (scalar_scales - med).abs() <= 3.0 * mad
                    if mad_inliers.any():
                        candidate_scales = candidate_scales[mad_inliers]

                    # Conservative estimate to avoid over-scaling bias.
                    q = 0.4
                    shared_scale = torch.quantile(candidate_scales, q=q, dim=0, keepdim=True)
                else:
                    shared_scale = first_frame_scale

            if shared_scale is not None:
                for k in poses:
                    poses[k]["scale"] = shared_scale

        # breakpoint() # check if the slat from mv-sam3d can be decoded properly and posed properly with the given ref_scale and poses, might need to modify it to take DA3 pointmaps (should be easy, do it anyway)

        outputs = self.decode_slat(
            slat, self.decode_formats if decode_formats is None else decode_formats
        )

        if (
            calibrate_scale_with_selected_view
            and "gaussian_4" in outputs
            and f"{selected_view_idx}" in poses
            and pointmaps is not None
            and masks is not None
            and len(pointmaps) > selected_view_idx
            and len(masks) > selected_view_idx
        ):
            try:
                canon = outputs["gaussian_4"][0].get_xyz.to(self.device)

                def frame_correction(idx):
                    # Ratio of the observed (masked pointmap) to the posed object extent.
                    pose = poses[f"{idx}"]
                    R = quaternion_to_matrix(pose["rotation"].reshape(1, 4))[0]
                    src = (canon * pose["scale"].reshape(-1, 3)[0].mean()) @ R.T + pose["translation"].reshape(-1, 3)[0]
                    pm = pointmaps[idx]
                    pm = (torch.from_numpy(pm) if isinstance(pm, np.ndarray) else pm).to(self.device, dtype=torch.float32)
                    pm = pm if (pm.ndim == 3 and pm.shape[-1] == 3) else pm.permute(1, 2, 0)
                    mask = np.array(masks[idx])
                    mask = torch.from_numpy((mask[..., -1] if mask.ndim == 3 else mask) > 0).to(self.device)
                    if mask.shape != pm.shape[:2]:
                        mask = torchvision.transforms.functional.resize(
                            mask.float()[None, None], pm.shape[:2],
                            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                        )[0, 0].bool()
                    # An object cut off by the image border only shows part of its extent.
                    if mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any():
                        return None
                    tgt = pm[mask]
                    tgt = tgt[torch.isfinite(tgt).all(dim=-1)]
                    src = src[torch.isfinite(src).all(dim=-1)]
                    if src.shape[0] <= 128 or tgt.shape[0] <= 128:
                        return None
                    src = src[torch.randperm(src.shape[0], device=src.device)[:10000]]
                    tgt = tgt[torch.randperm(tgt.shape[0], device=tgt.device)[:10000]]
                    src_r = torch.linalg.norm(src - src.median(dim=0).values, dim=-1).quantile(0.8).clamp_min(1e-6)
                    tgt_r = torch.linalg.norm(tgt - tgt.median(dim=0).values, dim=-1).quantile(0.8).clamp_min(1e-6)
                    return float(tgt_r / src_r)

                # Median over all fully visible frames; the selected view alone can be truncated.
                frames = [i for i in range(min(len(pointmaps), len(masks))) if f"{i}" in poses]
                corrs = [c for c in (frame_correction(i) for i in frames) if c is not None]
                if not corrs:
                    corrs = [c for c in [frame_correction(selected_view_idx)] if c is not None]
                if corrs:
                    corr = float(np.clip(np.median(corrs), 0.25, 2.0))
                    for k in poses:
                        poses[k]["scale"] = poses[k]["scale"] * corr
            except Exception as e:
                print(f"[scale calibration] skipped: {e}")

        return poses, outputs, slat, coords

    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        # img_3chw: (3, H, W)
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0

        # heuristics
        if max_side > 3800:
            scale_factor = 0.125
        if max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # -> (1, 3, H/4, W/4)
        return x.squeeze(0)

    def estimate_plane(self, pointmap_dict, image, ground_area_threshold=0.25, min_points=100):
        assert image.shape[-1] == 4  # rgba format
        # Extract mask from alpha channel
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # Get all points in 3D space (H, W, 3)
        pts_hwc = pts.cpu().permute((1, 2, 0))

        valid_mask_points = floor_mask.cpu().numpy()
        # Extract points that fall within the mask
        if valid_mask_points.any():
            # Get points within mask
            masked_points = pts_hwc[valid_mask_points]
            # Filter out invalid points (zero points from depth estimation failures)
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.numpy()
        else:
            points = np.array([]).reshape(0, 3)
     
        # Calculate area coverage and check num of points
        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")
        if overlap_area > ground_area_threshold and has_enough_points:
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh")
            except Exception as e:
                logger.error(f"Failed to estimate plane: {e}")
                mesh = None
        else:
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
