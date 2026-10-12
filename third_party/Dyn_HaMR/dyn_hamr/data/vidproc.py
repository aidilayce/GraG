import os
import numpy as np
import subprocess
import cv2
import torch

import preproc.launch_hamer as hamer
from preproc.launch_slam import split_frames_shots, get_command, check_intrins
from preproc.extract_frames import video_to_frames, split_frame


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def preprocess_frames(img_dir, src_path, overwrite=False, **kwargs):
    if not overwrite and is_nonempty(img_dir):
        print(f"FOUND {len(os.listdir(img_dir))} FRAMES in {img_dir}")
        return
    print(f"EXTRACTING FRAMES FROM {src_path} TO {img_dir}")
    print(kwargs)

    # out = video_to_frames(src_path, img_dir, overwrite=overwrite, **kwargs)
    out = split_frame(src_path, img_dir, overwrite=overwrite, **kwargs)
    assert out == 0, "FAILED FRAME EXTRACTION"


def preprocess_tracks(datatype, img_dir, track_dir, shot_dir, gpu, overwrite=False):
    """
    :param img_dir
    :param track_dir, expected format: res_root/track_name/sequence
    :param shot_dir, expected format: res_root/shot_name/sequence
    """
    if not overwrite and is_nonempty(track_dir):
        print(f"FOUND TRACKS IN {track_dir}")
        return

    print(f"RUNNING HAMER ON {img_dir}")
    track_root, seq = os.path.split(track_dir.rstrip("/"))
    res_root, track_name = os.path.split(track_root)
    shot_name = shot_dir.rstrip("/").split("/")[-2]

    hamer.process_seq(
        [gpu],
        res_root,
        seq,
        img_dir,
        track_name=track_name,
        shot_name=shot_name,
        datatype=datatype,
        overwrite=overwrite,
    )

# NOTE: modified: DA3 instead of VIPE
def load_da3_cameras(da3_path):
    """
    Load DA3 camera outputs and convert to DROID-SLAM format
    
    Args:
        da3_path: Path to DA3 results file
        seq_name: Sequence name
        img_dir: Image directory to get image size
        start: Start frame index
        end: End frame index
        
    Returns:
        w2c: (N, 4, 4) world-to-camera matrices
        intrins_full: (N, 6) intrinsics [fx, fy, cx, cy, W, H]
    """
    if not os.path.exists(da3_path):
        raise FileNotFoundError(f"DA3 results file not found: {da3_path}")
    
    print(f"Loading DA3 cameras from {da3_path}")
    da3_data = torch.load(da3_path)

    w2c = da3_data['extrinsics'].cpu().numpy()  # (N, 4, 4) world-to-camera
    
    # convert NX3X3 intrinsics to NX4 [fx, fy, cx, cy]
    intrins = da3_data['intrinsics'] 
    intrins_4 = torch.zeros((intrins.shape[0], 4), dtype=intrins.dtype)
    intrins_4[:, 0] = intrins[:, 0, 0]  # fx
    intrins_4[:, 1] = intrins[:, 1, 1]  # fy
    intrins_4[:, 2] = intrins[:, 0, 2]  # cx
    intrins_4[:, 3] = intrins[:, 1, 2]  # cy
    intrins = intrins_4.numpy()
    
    pointmaps = da3_data['pointmaps'].cpu().numpy()  # (N, H, W, 3)
    img_height, img_width = pointmaps.shape[1], pointmaps.shape[2]

    # Add width and height to intrinsics
    N = len(w2c)
    intrins_full = np.zeros((N, 6), dtype=np.float32)
    intrins_full[:, :4] = intrins
    intrins_full[:, 4] = img_width
    intrins_full[:, 5] = img_height
    
    print(f"Loaded {N} DA3 camera poses")
    return w2c, intrins_full


def save_da3_cameras_as_droid(output_dir, w2c, intrins_full):
    """
    Save DA3 cameras in DROID-SLAM format
    
    Args:
        output_dir: Output directory
        w2c: (N, 4, 4) world-to-camera matrices
        intrins_full: (N, 6) intrinsics [fx, fy, cx, cy, W, H]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters
    W, H = intrins_full[0, 4], intrins_full[0, 5]
    focal = intrins_full[:, :2].mean()
    
    print(f"Saving DA3 cameras to {output_dir}")
    print(f"Image size: {int(W)}x{int(H)}, focal: {focal:.2f}")
    
    # Save cameras.npz (main format used by Dyn-HaMR)
    np.savez(
        f"{output_dir}/cameras.npz",
        height=H,
        width=W,
        focal=focal,
        intrins=intrins_full[:, :4],
        w2c=w2c,
    )
    print(f"Saved cameras.npz with {len(w2c)} frames")

def preprocess_cameras(cfg, overwrite=False):
    use_da3 = cfg.get("use_da3", False)
    da3_path = cfg.get("da3_path", None)
    map_dir = cfg.sources.cameras

    if use_da3 and da3_path is not None:
        # Hand fitting and refinement must use the same DA3 cameras.
        w2c, intrins_full = load_da3_cameras(da3_path)
        save_da3_cameras_as_droid(map_dir, w2c, intrins_full)
        return

    if not overwrite and is_nonempty(cfg.sources.cameras):
        print(f"FOUND CAMERAS IN {cfg.sources.cameras}")
        return

    # # Default: use DROID-SLAM
    # print(f"RUNNING SLAM ON {cfg.seq}")
    # img_dir = cfg.sources.images
    # map_dir = cfg.sources.cameras
    # subseqs, shot_idcs = split_frames_shots(cfg.sources.images, cfg.sources.shots)
    # print(shot_idcs, cfg.shot_idx, np.where(shot_idcs == cfg.shot_idx), cfg.sources.images, cfg.sources.shots)
    # print(subseqs)
    # shot_idx = np.where(shot_idcs == cfg.shot_idx)[0][0]
    # # run on selected shot
    # start, end = subseqs[shot_idx]
    # if not cfg.split_cameras:
    #     # only run on specified segment within shot
    #     end = start + cfg.end_idx
    #     start = start + cfg.start_idx
    # intrins_path = cfg.sources.get("intrins", None)
    # if intrins_path is not None:
    #     intrins_path = check_intrins(cfg.type, cfg.root, intrins_path, cfg.seq, cfg.split)

    # print('img_dir, map_dir, start, end, intrins_path', img_dir, map_dir, start, end, intrins_path)
    # # raise ValueERRROR
    # cmd = get_command(
    #     img_dir,
    #     map_dir,
    #     start=start,
    #     end=end,
    #     intrins_path=intrins_path,
    #     overwrite=overwrite,
    # )
    # print(cmd)
    # gpu = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    # out = subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True)
    # assert out == 0, "SLAM FAILED"
