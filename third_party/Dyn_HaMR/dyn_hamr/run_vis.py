import os
import glob

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from body_model import MANO
from omegaconf import DictConfig, OmegaConf

from data import get_dataset_from_cfg, expand_source_paths
from optim.output import (
    get_results_paths,
    load_result,
    save_input_frames,
    save_input_poses,
)
from util.loaders import load_config_from_log, resolve_cfg_paths
from util.tensor import get_device, move_to, detach_all, to_torch
from vis.output import prep_result_vis, animate_scene, make_video_grid_2x2
from vis.tools import vis_keypoints
from vis.viewer import init_viewer
import time
from geometry.mesh import save_mesh_scenes, vertices_to_trimesh
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def save_meshes_all(cfg, dataset, res_dicts, dev_id, mesh_dirs, num_steps=-1):
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)
    device = get_device(dev_id)
    obs_data = move_to(next(iter(loader)), device)

    # load models
    cfg = resolve_cfg_paths(cfg)
    # Instantiate MANO model
    mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
    print('initializing MANO model with cfgs:', mano_cfg, 'B*T', B*T)
    hand_model = MANO(batch_size=B*T, pose2rot=True, **mano_cfg).to(device)

    for res_dict, mesh_dir in zip(res_dicts, mesh_dirs):
        res_dict = move_to(res_dict, device)
        scene_dict = move_to(
            prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                hand_model,
                temporal_smooth=cfg.temporal_smooth,
                smooth_trans=True  # For mesh export, smooth everything
            ),
            "cpu",
        )

        scene_dir = mesh_dir
        verts, joints, colors, l_faces, r_faces, is_right, bounds = scene_dict["geometry"]
        T = len(verts)
        print(f"{T} mesh frames for ", mesh_dir)
        times = list(range(0, T, 1))
        flag = False
        for t in times:
            if len(is_right[t]) > 1:
                flag = True
                vv = t

        if flag:
            init_trans = (joints[vv][0][9].clone() + joints[vv][1][9].clone()) / 2
        else:
            init_trans = joints[0][0][9].clone()

        for t in times:
            if len(is_right[t]) > 1:
                assert (is_right[t].cpu().numpy().tolist() == [0,1])
                # l_meshes = make_batch_mesh(verts[t][0][None], l_faces[t], colors[t][0][None])
                # r_meshes = make_batch_mesh(verts[t][1][None], r_faces[t], colors[t][1][None])
                # assert len(l_meshes) == 1
                # assert len(r_meshes) == 1

                verts[t][0] -= init_trans
                joints[t][0] -= init_trans
                tmesh = vertices_to_trimesh(verts[t][0].detach().cpu().numpy(), l_faces[t].detach().cpu().numpy(), LIGHT_BLUE, is_right=0)
                tmesh.export(os.path.join(scene_dir, f'{str(t).zfill(6)}_0.obj'))
                # np.save(f'{str(t).zfill(6)}_0.npy', joints[t][0].detach().cpu().numpy())

                verts[t][1] -= init_trans
                joints[t][1] -= init_trans
                tmesh = vertices_to_trimesh(verts[t][1].detach().cpu().numpy(), r_faces[t].detach().cpu().numpy(), LIGHT_BLUE, is_right=1)
                tmesh.export(os.path.join(scene_dir, f'{str(t).zfill(6)}_1.obj'))
                # np.save(f'{str(t).zfill(6)}_1.npy', joints[t][1].detach().cpu().numpy())

            else:
                assert len(is_right[t]) == 1
                if is_right[t] == 0:
                    verts[t][0] -= init_trans
                    joints[t][0] -= init_trans
                    tmesh = vertices_to_trimesh(verts[t][0].detach().cpu().numpy(), l_faces[t].detach().cpu().numpy(), LIGHT_BLUE, is_right=0)
                    tmesh.export(os.path.join(scene_dir, f'{str(t).zfill(6)}_0.obj'))
                    # np.save(f'{str(t).zfill(6)}_0.npy', joints[t][0].detach().cpu().numpy())

                elif is_right[t] == 1:
                    verts[t][0] -= init_trans
                    joints[t][0] -= init_trans
                    tmesh = vertices_to_trimesh(verts[t][0].detach().cpu().numpy(), r_faces[t].detach().cpu().numpy(), LIGHT_BLUE, is_right=1)
                    tmesh.export(os.path.join(scene_dir, f'{str(t).zfill(6)}_1.obj'))
                    # np.save(f'{str(t).zfill(6)}_1.npy', joints[t][0].detach().cpu().numpy())


def run_vis(
    cfg,
    dataset,
    out_dir,
    dev_id,
    phases=["smooth_fit"],
    render_views=["src_cam", "above", "side"],
    make_grid=True,
    overwrite=False,
    save_dir=None,
    render_kps=False,
    render_layers=False,
    save_frames=False,
    **kwargs
):
    save_dir = out_dir if save_dir is None else save_dir
    print("OUT_DIR", out_dir)
    print("SAVE_DIR", save_dir)
    print("VISUALIZING PHASES", phases)
    print("RENDERING VIEWS", render_views)
    print("RENDER_KPS", render_kps)
    print("OVERWRITE", overwrite)

    # save input frames
    inp_vid_path = save_input_frames(
        dataset,
        f"{save_dir}/{dataset.seq_name}_input.mp4",
        fps=cfg.fps,
        overwrite=True,
    )

    if render_kps:
        render_keypoints_2d(dataset, save_dir, overwrite=overwrite)

    if len(render_views) < 1:
        return

    out_ext = "/" if render_layers or save_frames else ".mp4"
    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        res_dir = os.path.join(out_dir, phase)
        if phase == "input":
            res = get_input_dict(dataset)
            it = f"{0:06d}"

        elif os.path.isdir(res_dir):
            res_path_dict = get_results_paths(res_dir)
            print(f"FOUND {len(res_path_dict)} results in {res_dir}")
            it = sorted(res_path_dict.keys())[-1]
            res = load_result(res_path_dict[it])["world"]

        else:
            print(f"{res_dir} does not exist, skipping")
            continue

        out_name = f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}"
        mesh_dir = f"{save_dir}/{phase}/{dataset.seq_name}_{it}_meshes"
        os.makedirs(mesh_dir, exist_ok=True)
        phase_max_iters[phase] = it

        out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        if not overwrite and all(os.path.exists(p) for p in out_paths):
            print("FOUND OUT PATHS", out_paths)
            continue

        phase_results[phase] = out_name, mesh_dir, res

    if len(phase_results) > 0:
        out_names, mesh_dir, res_dicts = zip(*phase_results.values())
        render_results(
            cfg,
            dataset,
            dev_id,
            res_dicts,
            out_names,
            render_views=render_views,
            render_layers=render_layers,
            save_frames=save_frames,
            **kwargs,
        )
        save_meshes_all(cfg, dataset, res_dicts, dev_id, mesh_dir)

    if make_grid:
        for phase, it in phase_max_iters.items():
            grid_path = f"{save_dir}/{dataset.seq_name}_{phase}_grid.mp4"
            vid_paths = [
                f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_src_cam.mp4",
                f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_front.mp4",
                f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_above.mp4",
                f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_side.mp4",
            ]
            make_video_grid_2x2(
                grid_path,
                vid_paths,
                fps=cfg.fps,
                overwrite=True,
            )


def get_input_dict(dataset):
    dataset.load_data(interp_input=False)
    d = dataset.data_dict
    input_params = {
        "pose_body": np.stack(d["init_body_pose"], axis=0),
        "trans": np.stack(d["init_trans"], axis=0),
        "root_orient": np.stack(d["init_root_orient"], axis=0),
    }
    input_params = to_torch(input_params)
    print({k: v.shape for k, v in input_params.items()})
    return input_params


def render_keypoints_2d(dataset, save_dir, overwrite=False):
    """
    render 2d keypoints for each track
    """
    dataset.load_data()
    out_dir = f"{save_dir}/{dataset.seq_name}_joints2d"
    B, T = dataset.n_tracks, dataset.seq_len
    if not overwrite and (os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= B * T):
        print(f"Keypoints already rendered in {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for i, tid in enumerate(dataset.track_ids):
        joints2d = dataset.data_dict["joints2d"][i]  # (T, J, 3)
        for t, sel_img_name in enumerate(dataset.sel_img_names):
            img = vis_keypoints(joints2d[t : t + 1], dataset.img_size)
            out_path = f"{out_dir}/{sel_img_name}_{tid}.png"
            imageio.imwrite(out_path, img)



def render_results(cfg, dataset, dev_id, res_dicts, out_names, **kwargs):
    """
    render results for all selected phases
    """
    assert len(res_dicts) == len(out_names)
    if len(res_dicts) < 1:
        print("no results to render, skipping")
        return

    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    device = get_device(dev_id)
    obs_data = move_to(next(iter(loader)), device)
    cam_data = dataset.get_camera_data()

    # load models
    cfg = resolve_cfg_paths(cfg)
    # Instantiate MANO model
    mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
    print('initializing MANO model with cfgs:', mano_cfg, 'B*T', B*T)
    hand_model = MANO(batch_size=B*T, pose2rot=True, **mano_cfg).to(device)
    vis = init_viewer(
        dataset.img_size,
        cam_data["intrins"][0],
        vis_scale=1.0,
        bg_paths=dataset.sel_img_paths,
        fps=cfg.fps,
    )

    # Set 2D keypoints for overlay visualization if render_keypoints is enabled
    if kwargs.get('render_keypoints', False):
        # Load 2D keypoints from dataset - assume single track for now
        # joints2d is list of (T, J, 3) arrays, one per track
        dataset.load_data()
        if len(dataset.data_dict["joints2d"]) > 0:
            joints2d = dataset.data_dict["joints2d"][0]  # First track: (T, J, 3)
            keypoints_seq = [joints2d[t] for t in range(T)]  # List of (J, 3) arrays
            vis.set_keypoints_seq(keypoints_seq)
            print(f"Loaded {T} frames of 2D keypoints with {joints2d.shape[1]} joints")

    save_paths_all = []
    render_views = kwargs.get('render_views', ['src_cam', 'above', 'side'])
    
    # Separate src_cam from other views
    src_cam_views = [v for v in render_views if v == 'src_cam']
    other_views = [v for v in render_views if v != 'src_cam']
    
    for res_dict, out_name in zip(res_dicts, out_names):
        print(f'preparing results for rendering {out_name}')
        res_dict = move_to(res_dict, device)
        
        # Render src_cam WITH temporal smoothing but WITHOUT trans smoothing
        if src_cam_views:
            print(f"Rendering src_cam view WITH temporal smoothing (excluding trans)")
            scene_dict_no_smooth = prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                hand_model,
                temporal_smooth=cfg.temporal_smooth,  # Apply smoothing
                smooth_trans=False  # But don't smooth translation
            )
            
            # Compute predicted 2D keypoints using THE SAME function as optimization
            if kwargs.get('render_keypoints', False):
                from geometry import camera as cam_util
                from body_model import run_mano
                
                # Get the same data as optimization
                joints3d_op = run_mano(
                    hand_model,
                    res_dict["trans"],
                    res_dict["root_orient"],
                    res_dict["pose_body"],
                    res_dict["is_right"],
                    res_dict.get("betas", None),
                )["joints"]  # (B, T, J, 3)
                
                # Get cameras (same as optimization)
                cam_R = res_dict["cam_R"]  # (B, T, 3, 3)
                cam_t = res_dict["cam_t"]  # (B, T, 3)
                intrins = res_dict["intrins"]  # (4,) or (T, 4) or (B, T, 4)
                
                # Ensure intrins is (B, T, 4)
                if intrins.ndim == 1:  # (4,)
                    intrins = intrins[None, None].expand(cam_R.shape[0], cam_R.shape[1], -1)  # (B, T, 4)
                elif intrins.ndim == 2:  # (T, 4)
                    intrins = intrins[None].expand(cam_R.shape[0], -1, -1)  # (B, T, 4)
                
                cam_f = intrins[:, :, :2]  # (B, T, 2)
                cam_center = intrins[:, :, 2:]  # (B, T, 2)
                
                # Reproject using EXACTLY the same function as optimization
                joints2d_pred = cam_util.reproject(joints3d_op, cam_R, cam_t, cam_f[0], cam_center[0])  # (B, T, J, 2)
                
                # Convert to list for first track
                pred_keypoints_seq = [joints2d_pred[0, t].cpu().numpy() for t in range(joints2d_pred.shape[1])]
                vis.set_pred_keypoints_seq(pred_keypoints_seq)
                print(f"Computed {len(pred_keypoints_seq)} frames of predicted 2D keypoints using reproject()")
            
            # Render only src_cam view
            kwargs_src = kwargs.copy()
            kwargs_src['render_views'] = src_cam_views
            save_paths_src = animate_scene(
                vis, scene_dict_no_smooth, out_name, seq_name=dataset.seq_name, **kwargs_src
            )
            save_paths_all.append(save_paths_src)
        
        # Render other views WITH temporal smoothing (including trans)
        if other_views:
            print(f"Rendering {other_views} views WITH temporal smoothing (including trans)")
            scene_dict_smooth = prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                hand_model,
                temporal_smooth=cfg.temporal_smooth,  # Apply smoothing for other views
                smooth_trans=True  # Smooth translation for other views
            )
            
            # Render other views
            kwargs_other = kwargs.copy()
            kwargs_other['render_views'] = other_views
            kwargs_other['render_keypoints'] = False  # No keypoints on other views
            save_paths_other = animate_scene(
                vis, scene_dict_smooth, out_name, seq_name=dataset.seq_name, **kwargs_other
            )
            if not src_cam_views:
                save_paths_all.append(save_paths_other)

    vis.close()

    return save_paths_all


def visualize_log(log_dir, dev_id, phases, save_dir=None, **kwargs):
    print(log_dir)
    cfg = load_config_from_log(log_dir)
    print(cfg)
    print(cfg.data)
    print(cfg.data.sources)

    # make sure we get all necessary inputs
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)
    # cfg.data.track_ids = "001"
    dataset = get_dataset_from_cfg(cfg)
    if len(dataset) < 1:
        print(f"No tracks in dataset, skipping")
        return

    run_vis(cfg, dataset, log_dir, dev_id, phases=phases, save_dir=save_dir, **kwargs)


def launch_vis(i, args):
    log_dir = args.log_dirs[i]
    dev_id = args.gpus[i % len(args.gpus)]
    os.environ["EGL_DEVICE_ID"] = str(dev_id)
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    if args.save_root is not None:
        path_name = log_dir.split(args.log_root)[-1].strip("/")
        exp_name = "-".join(path_name.split("/")[:2])
        save_dir = f"{args.save_root}/{exp_name}"
        os.makedirs(save_dir, exist_ok=True)

    print('args.save_root: ', args.save_root)
    visualize_log(
        log_dir,
        dev_id,
        phases=args.phases,
        save_dir=save_dir,
        overwrite=args.overwrite,
        accumulate=args.accumulate,
        render_kps=args.render_kps,
        render_layers=args.render_layers,
        render_views=args.render_views,
        save_frames=args.save_frames,
        make_grid=args.grid,
    )


def main(args):
    """
    visualize all runs in root
    """
    OmegaConf.register_new_resolver("eval", eval)
    log_dirs = []
    for root, subd, files in os.walk(args.log_root):
        if ".hydra" in subd:
            log_dirs.append(root)
    args.log_dirs = log_dirs
    print(f"FOUND {len(args.log_dirs)} TO RENDER")

    if len(args.gpus) > 1:
        from torch.multiprocessing import Pool

        torch.multiprocessing.set_start_method("spawn")

        with Pool(processes=len(args.gpus)) as pool:
            res = pool.starmap(
                launch_vis, [(i, args) for i in range(len(args.log_dirs))]
            )
        return

    for i in range(len(args.log_dirs)):
        launch_vis(i, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--save_root", default='./')
    parser.add_argument("--phases", nargs="*", default=["smooth_fit"])
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument(
        "-rv",
        "--render_views",
        nargs="*",
        default=["src_cam", "front", "above", "side"],
    )
    parser.add_argument("-g", "--grid", action="store_true")
    parser.add_argument("-rl", "--render_layers", action="store_true")
    parser.add_argument("-kp", "--render_kps", action="store_true")
    parser.add_argument("-sf", "--save_frames", action="store_true")
    parser.add_argument("-ra", "--accumulate", action="store_true")
    parser.add_argument("-y", "--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)
