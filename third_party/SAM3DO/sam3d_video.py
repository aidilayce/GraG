import sys
import argparse
from typing import Union, Optional, List, Callable
from pathlib import Path
# import inference code
THIS_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = THIS_DIR / "notebook"
note_str = str(NOTEBOOK_DIR)
if note_str not in sys.path:
    sys.path.append(note_str)
from inference_sam3d import ready_gaussian_for_video_rendering, render_video, load_image, load_mask, display_image, make_scene, interactive_visualizer
import imageio
import os 
import glob
import builtins
import shutil
from copy import deepcopy
import subprocess
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
from PIL import Image
import trimesh
import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_invert, quaternion_to_matrix
import sam3d_objects 
from sam3d_objects.pipeline.video_inference_pipeline import InferencePipelinePointMap_fromVideo
from sam3d_objects.pipeline.video_inference_pipeline_onlyimg import InferencePipeline_fromVideo
from sam3d_objects.utils.visualization import SceneVisualizer
from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer

CUDA_LAUNCH_BLOCKING=1

renderer = GaussianRenderer(
        {
            "resolution": None,  # to be set later
            "near": 0.1,
            "far": 100,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
            "backend": "gsplat",
            # "backend": "inria",
        }
    )

opencv_to_pytorch3d = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
])

pytorch3d_to_opengl = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
])

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]

def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )

def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))

class Inference:
    ''' Inference class that also takes pointmap for conditioning'''
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False, onlyimg_cond: bool = False, pose_only: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.pose_only = pose_only
        config.workspace_dir = os.path.dirname(config_file)
        # Be defensive here: the shipped video config has drifted before and pointed
        # at the non-video pipeline, which does not implement get_pose_per_frame().
        if onlyimg_cond:
            config._target_ = "sam3d_objects.pipeline.video_inference_pipeline_onlyimg.InferencePipeline_fromVideo"
        else:
            config._target_ = "sam3d_objects.pipeline.video_inference_pipeline.InferencePipelinePointMap_fromVideo"
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self.onlyimg_cond = onlyimg_cond
        
        if onlyimg_cond:
            self._pipeline: InferencePipeline_fromVideo = instantiate(config)
        else:
            self._pipeline: InferencePipelinePointMap_fromVideo = instantiate(config)
        print(f"Initialized inference pipeline from config: {config_file}")

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray],
        masks: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmaps=None,
        selected_view_idx=0,
        sam3d_pose_inference_steps: int = 25,
        ss_cfg_strength: float = 1.0,
        ss_cfg_strength_pm: float = 0.0,
        temporal_guidance_lambda: float = 0.0,
        shape_mode='fixed', # 'fixed', 'trajectory', 'sam3d'
        use_prev_pose=True,
        scale_mode: str = "video_median",  # 'video_median', 'per_frame', 'first_frame'
        lock_scale_during_pose: bool = False,
        calibrate_scale_with_selected_view: bool = False,
        mv_sam3d_ss_path: Optional[str] = None,
        mv_sam3d_slat_path: Optional[str] = None,
        decode_formats: Optional[List[str]] = None,
    ) -> dict:
        
        if shape_mode == 'fixed':
            print('Using fixed shape latents from MV-SAM3D for all frames.')
        elif shape_mode == 'trajectory':
            print('Using shape latents from MV-SAM3D trajectory for each frame.')
        elif shape_mode == 'sam3d':
            print('Using SAM3D inferred shape latents for each frame.')
        else:
            raise ValueError(f'Unknown shape_mode: {shape_mode}')

        if use_prev_pose:
            print('Using previous frame pose as initialization for current frame pose estimation.')
        else:
            print('Not using previous frame pose for initialization.')
        
        output = self._pipeline.get_pose_per_frame(
            images,
            masks,
            seed,
            pointmaps=pointmaps,
            selected_view_idx=selected_view_idx,
            sam3d_pose_inference_steps=sam3d_pose_inference_steps,
            ss_cfg_strength=ss_cfg_strength,
            ss_cfg_strength_pm=ss_cfg_strength_pm,
            temporal_guidance_lambda=temporal_guidance_lambda,
            decode_formats=decode_formats or ["gaussian_4","gaussian", "mesh"],
            shape_mode=shape_mode,
            use_prev_pose=use_prev_pose,
            scale_mode=scale_mode,
            lock_scale_during_pose=lock_scale_during_pose,
            calibrate_scale_with_selected_view=calibrate_scale_with_selected_view,
            mv_sam3d_ss_path=mv_sam3d_ss_path,
            mv_sam3d_slat_path=mv_sam3d_slat_path,
        )
        return output

def align_gaussian_to_opengl(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    scene_gs.from_xyz(scene_gs.get_xyz @ torch.tensor(pytorch3d_to_opengl, device=scene_gs._xyz.device, dtype=scene_gs._xyz.dtype))
    return scene_gs

def pose_gaussians(gaussian, rotation, translation, scale, in_place=False):
    if not in_place:
        gaussian = deepcopy(gaussian)

    minimum_kernel_size = float("inf")
    if translation.ndim > 2:
        translation = translation.reshape(-1, translation.shape[-1])
    if rotation.ndim > 2:
        rotation = rotation.reshape(-1, rotation.shape[-1])
    if scale.ndim > 2:
        scale = scale.reshape(-1, scale.shape[-1])
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
    kernel_scale = float(scale.reshape(-1).min().item())
    gaussian.mininum_kernel_size *= kernel_scale
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="hold_ABF12_ho3d", help="name of the sequence folder")
    parser.add_argument("--onlyimg_cond", action='store_true', help="whether to use only image conditioning or also pointmap conditioning")
    args = parser.parse_args()

    NUM_INF_STEPS = 10
    ss_cfg_strength = 1.0
    MAIN_PATH = str(Path(os.environ.get("GRAG_WORK_DIR", THIS_DIR.parent.parent / "demo_data")))

    onlyimg_cond = args.onlyimg_cond  # whether to use only image conditioning or also pointmap conditioning
    seq_name = args.seq_name
    DEMO_PATH = f"{MAIN_PATH}/{seq_name}"
    exp_name = 'sam3d_fixed_shape'
    PATH = f"{DEMO_PATH}/sam3d_onlyimg" if onlyimg_cond else f"{DEMO_PATH}/{exp_name}"
    os.makedirs(PATH, exist_ok=True)

    # load image and mask
    imagedir = f'{MAIN_PATH}/{seq_name}/rgb'
    maskdir = f'{MAIN_PATH}/{seq_name}/masks'
    image_files = sorted(glob.glob(os.path.join(imagedir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(maskdir, "*.png")))
    images = [load_image(f) for f in image_files]
    masks = [load_mask(f) for f in mask_files]

    da3_result = torch.load(f'{MAIN_PATH}/{seq_name}/da3_pointmaps_intrinsics_extrinsics.pt')
    pointmap = da3_result['pointmaps'] # T, H, W, 3
    # pointmap_colors = da3_result['colors']
    Ks = da3_result['intrinsics']
    Xs = da3_result['extrinsics']

    mv_sam3d_ss_path = DEMO_PATH + f'/{exp_name}/ss_return_dict.pth'
    mv_sam3d_slat_path = DEMO_PATH + f'/{exp_name}/slat.pth'
    
    # load model
    if onlyimg_cond:
        config_path = f"checkpoints/hf/pipeline_video_onlyimg.yaml"
        inference = Inference(config_path, onlyimg_cond=onlyimg_cond, compile=False)
    else:
        config_path = f"checkpoints/hf/pipeline_video.yaml"
        inference = Inference(config_path, onlyimg_cond=onlyimg_cond, compile=False)
    selected_view_idx = 0
    output = inference(
        images, masks, seed=42, pointmaps=pointmap, 
        # shape_mode='trajectory', 
        use_prev_pose=True,
        ss_cfg_strength=ss_cfg_strength, sam3d_pose_inference_steps=NUM_INF_STEPS, 
        selected_view_idx=selected_view_idx, 
        mv_sam3d_ss_path=mv_sam3d_ss_path, mv_sam3d_slat_path=mv_sam3d_slat_path
        )
    poses = output[0]
    geo_outputs = output[1]
    slat = output[2]
    coords = output[3]
    H, W = pointmap.shape[1], pointmap.shape[2]
    renderer.rendering_options['resolution'] = (H, W)

    # # fix da3 predicted intrinsics to resized image size
    # H_old, W_old = images[0].shape[0], images[0].shape[1]
    # scale_w = W / W_old
    # scale_h = H / H_old
    # Ks = Ks.clone()
    # Ks[:, 0, 0] *= scale_w
    # Ks[:, 1, 1] *= scale_h
    # Ks[:, 0, 2] *= scale_w
    # Ks[:, 1, 2] *= scale_h

    gs4 = geo_outputs['gaussian_4'][0]
    gs = geo_outputs['gaussian'][0]
    mesh = geo_outputs['mesh'][0] # MeshExtract class with vertices and faces

    # save the poses, generated object coords and slat for optimization later
    # first convert poses into ob_in_cam format where in a folder, we have .txt per frame in 000000.txt format and pose is 4x4 pytorch3d format
    if onlyimg_cond:
        pose_output_dir = os.path.join(f"{PATH}/sam3d_poses_onlyimg")
        os.makedirs(pose_output_dir, exist_ok=True)
        torch.save(poses, os.path.join(PATH, f"poses_sam3d_raw.pt"))
    else:
        pose_output_dir = os.path.join(f"{PATH}/sam3d_poses")
        os.makedirs(pose_output_dir, exist_ok=True)
        torch.save(poses, os.path.join(PATH, f"poses_sam3d_raw.pt"))
    for frame_idx, pose in enumerate(poses.values()):
        trans = pose['translation']
        quat_rot = pose['rotation']
        scale_obj = pose['scale']
        # get 4x4 opencv w2c
        R_obj = quaternion_to_matrix(quat_rot)  # (3, 3)
        T_obj = torch.eye(4)
        T_obj[:3, :3] = R_obj
        T_obj[:3, 3] = trans
        # save as txt
        np.savetxt(os.path.join(pose_output_dir, f"{frame_idx:06d}.txt"), T_obj.cpu().numpy(), fmt="%.6f")
    # save slat and coords
    model_output_dir = os.path.join(f"{PATH}/sam3d_guided") if onlyimg_cond else os.path.join(f"{PATH}/sam3d_model")
    os.makedirs(model_output_dir, exist_ok=True)
    torch.save({
        'slat': slat,
        'coords': coords,
        'gs_4': gs4,
        'gs': gs,
        'mesh': mesh,
    }, os.path.join(model_output_dir, f"object_slat_coords.pt"))        
    

    from pytorch3d.renderer import look_at_view_transform
    upside_down = torch.tensor([[1,  0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
    # pose the gaussians for visualization
    device = gs4._xyz.device
    frames = []
    overlayed_frames = []
    for frame_idx, pose in enumerate(poses.values()):
        trans = pose['translation']
        quat_rot = pose['rotation']
        scale_obj = pose['scale']
        gs4_posed = pose_gaussians(gs4, quat_rot, trans, scale_obj)
        gs4_posed_opengl = align_gaussian_to_opengl(gs4_posed, in_place=True) # take from pytorch3d to opengl

        # breakpoint() # check wo pointmaps, where are these gaussians? which cam can i use to render them?
        if not onlyimg_cond:
            # render img with the estimated camera pose from da3
            X_pt3d_R = Xs[frame_idx][:3, :3] @ opencv_to_pytorch3d @ pytorch3d_to_opengl
            X_pt3d_t = Xs[frame_idx][:3, 3:]
            X_pt3d = torch.eye(4)
            X_pt3d[:3, :3] = X_pt3d_R
            X_pt3d[:3, 3:] = X_pt3d_t
        else:
            R, T = look_at_view_transform(dist=2.0, elev=15.0, azim=0.0)
            X_pt3d = torch.eye(4)
            X_pt3d[:3, :3] = R[0] @ pytorch3d_to_opengl # NOTE: CHECK if this one gives the correct non-mirrored view
            # X_pt3d[:3, :3] = upside_down @ R[0]
            X_pt3d[:3, 3]  = T[0]
            
        # normalize Ks (for Gaussian renderer, both inria and gsplat backends)
        Ks[frame_idx][0, 0] /= W
        Ks[frame_idx][1, 1] /= H
        Ks[frame_idx][0, 2] /= W
        Ks[frame_idx][1, 2] /= H

        color = renderer.render(gs4_posed_opengl, X_pt3d.to(device), Ks[frame_idx].to(device))['color']
        frames.append((color.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        
        overlayed = color.permute(1,2,0).cpu().numpy() * 255
        img_orig = imageio.imread(image_files[frame_idx])
        if img_orig.max() <= 1.0:
            img_orig = (img_orig * 255).astype(np.uint8)
        alpha = 0.5
        overlayed = (alpha * overlayed + (1 - alpha) * img_orig).astype(np.uint8)
        overlayed_frames.append(overlayed)

    # save video as gif
    imageio.mimsave(
        os.path.join(f"{model_output_dir}/gaussians_da3_aligned.gif"),
        frames,
        format="GIF",
        duration=1000 / 10,  # default assuming 30fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )
    print(f'Saved aligned gaussian gif to {model_output_dir}/gaussians_da3_aligned.gif')
    
    imageio.mimsave(
        os.path.join(f"{model_output_dir}/gaussians_da3_aligned_overlayed.gif"),
        overlayed_frames,
        format="GIF",
        duration=1000 / 10,  # default assuming 30fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )
    print(f'Saved overlayed gaussian gif to {model_output_dir}/overlayed_gaussians_da3_aligned.gif')
    
    imageio.mimsave(
        os.path.join(f"{model_output_dir}/gaussians_da3_aligned_overlayed.mp4"),
        overlayed_frames,
        format="mp4",
    )
    print(f'Saved overlayed gaussian mp4 to {model_output_dir}/overlayed_gaussians_da3_aligned.mp4')
    # imageio.mimsave(
    #     os.path.join(f"{PATH}/gaussians_da3_aligned.gif"),
    #     frames,
    #     format="GIF",
    #     duration=1000 / 30,  # default assuming 30fps from the input MP4
    #     loop=0,  # 0 means loop indefinitely
    # )
    # scene_gs = make_scene(output) # puts the object in pytorch3d scene
    
    # # save point map
    # pcd = trimesh.points.PointCloud(pointmap.reshape(-1, 3), colors=pointmap_colors.reshape(-1, 3))
    # pcd.export(os.path.join(f"{PATH}/pointmap.ply"))
    # else:
    #     # run model
    #     output = inference(image, mask, seed=42)
    #     # export gaussian splat
    #     output["gs"].save_ply(f"{PATH}/splat.ply")
    #     # render gaussian splat
    #     scene_gs = make_scene(output)
    #     scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    #     video = render_video(
    #         scene_gs,
    #         r=1,
    #         fov=60,
    #         pitch_deg=15,
    #         yaw_start_deg=-45,
    #         resolution=512,
    #     )["color"]
    #     # save video as gif
    #     imageio.mimsave(
    #         os.path.join(f"{PATH}/gaussians_single.gif"),
    #         video,
    #         format="GIF",
    #         duration=1000 / 30,  # default assuming 30fps from the input MP4
    #         loop=0,  # 0 means loop indefinitely
    #     )
    #     interactive_visualizer(f"{MAIN_PATH}/hold_ABF12_ho3d/sam3d/splat.ply")
