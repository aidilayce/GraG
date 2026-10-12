# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
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


class InferencePipeline_fromVideo(InferencePipeline):

    def __init__(
        self, *args, depth_model, **kwargs
    ):
        super().__init__(*args, **kwargs)

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
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)

            ss_return_dict = self.sample_sparse_structure( # shape features
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            ref_pose_dict = self.pose_decoder(
                ss_return_dict
            )
            ref_scale_raw = ref_pose_dict["scale"]
            ref_scale = ref_scale_raw * ss_return_dict["downsample_factor"]

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

            return ss_return_dict, outputs, slat, ref_scale
    
    def get_pose_per_frame(
        self,
        images: Union[None, Image.Image, np.ndarray],
        masks: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        pointmaps=None,
        selected_view_idx=0,
        decode_formats=['gaussian_4','gaussian', 'mesh'],
        refine_w_icp=False,
        only_key_frame=False,
        mv_sam3d_ss_path=None,
        mv_sam3d_slat_path=None,
    ) -> dict:
        if only_key_frame:
            geo_output = self.run_selected_view(image=images[selected_view_idx], mask=masks[selected_view_idx], pointmap=None, decode_formats=decode_formats)
            ss_return_dict0, outputs, slat, ref_scale, _, _ = geo_output
        else:
            if not mv_sam3d_ss_path or not mv_sam3d_slat_path:
                raise ValueError("MV-SAM3D result paths are required for full-video inference")
            ss_return_dict0 = torch.load(mv_sam3d_ss_path)
            slat = torch.load(mv_sam3d_slat_path)
            
        coords = ss_return_dict0["coords"]
        shape_latent_star = ss_return_dict0['shape']
        ref_scale = ss_return_dict0['scale']
        
        poses = {}
        s_i = None
        prev_pose_dict_latents = None
        # run per-frame using the same shape latents
        for i, (image, mask) in enumerate(zip(images, masks)):
            image = self.merge_mask_to_rgba(image, mask)
            with self.device: 
                ss_input_dict_i = self.preprocess_image(image, self.ss_preprocessor) # pointmap, scale, shift stuff
                if seed is not None:
                    torch.manual_seed(seed)

                ss_return_dict_i = self.sample_sparse_structure(
                    ss_input_dict_i,
                    shape_latent_star=shape_latent_star,
                    freeze_shape=True,
                    use_prev_pose=True,
                    prev_pose_dict=prev_pose_dict_latents,
                    )
                
                prev_pose_dict_latents = {
                    '6drotation_normalized': ss_return_dict_i['6drotation_normalized'],
                    'scale': ss_return_dict_i['scale'],
                    'translation': ss_return_dict_i['translation'],
                    'translation_scale': ss_return_dict_i['translation_scale'],
                }
                
                pose_dict = self.pose_decoder(ss_return_dict_i)
                t_i, R_i = pose_dict["translation"], pose_dict["rotation"]
                if s_i is None:
                    s_i = pose_dict['scale']
                
                poses[f'{i}'] = {
                    "translation": t_i,
                    "rotation": R_i,
                    "scale": s_i,
                }
        
        # breakpoint() # check if the slat from mv-sam3d can be decoded properly and posed properly with the given ref_scale and poses, might need to modify it to take DA3 pointmaps (should be easy, do it anyway)

        outputs = self.decode_slat(
            slat, self.decode_formats if decode_formats is None else decode_formats
        )

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
