import os
import glob
import typing

import imageio
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from body_model import MANO_JOINTS
assert len(MANO_JOINTS) == 16
import time

from util.logger import Logger
from geometry.camera import invert_camera

from .tools import read_keypoints, read_mask_path, load_mano_preds, load_keypoints_with_interp
from .vidproc import preprocess_cameras, preprocess_frames, preprocess_tracks


"""
Define data-related constants
"""
# DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])

# XXX: TEMPORARY CONSTANTS
SHOT_PAD = 0
MIN_SEQ_LEN = 10
MAX_NUM_TRACKS = 12
MIN_TRACK_LEN = 10 #40 #60
MIN_KEYP_CONF = 0.4


def get_dataset_from_cfg(cfg):
    args = cfg.data
    if not args.use_cams:
        args.sources.cameras = ""

    args.sources = expand_source_paths(args.sources)
    print("DATA SOURCES", args.sources)
    a = time.time()
    check_data_sources(args, cfg)
    print('end of check_data_sources, time: ', time.time() - a)

    return MultiPeopleDataset(
        args.sources,
        args.seq,
        tid_spec=args.track_ids,
        shot_idx=args.shot_idx,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        is_static=cfg.is_static,
        split_cameras=args.get("split_cameras", True),
    )


def expand_source_paths(data_sources):
    return {k: get_data_source(v) for k, v in data_sources.items()}


def get_data_source(source):
    matches = glob.glob(source)
    if len(matches) < 1:
        print(f"{source} does not exist")
        return source  # return anyway for default values
    if len(matches) > 1:
        raise ValueError(f"{source} is not unique")
    return matches[0]


def check_data_sources(args, cfg):
    print()
    a = time.time()
    if args.type == "video":
        preprocess_frames(args.sources.images, args.src_path, **args.frame_opts)
    b = time.time()
    preprocess_tracks(cfg.datatype, args.sources.images, args.sources.tracks, args.sources.shots, gpu=cfg.gpu)
    c = time.time()
    preprocess_cameras(args, overwrite=args.get("overwrite_cams", False))
    d = time.time()
    print('frame, hand track, slam camera: ', b-a, c-b, d-c)


class MultiPeopleDataset(Dataset):
    def __init__(
        self,
        data_sources: typing.Dict,
        seq_name,
        tid_spec="all",
        shot_idx=0,
        start_idx=0,
        end_idx=-1,
        is_static=False,
        pad_shot=False,
        split_cameras=True,
    ):
        self.is_static = is_static
        self.seq_name = seq_name
        self.data_sources = data_sources
        self.split_cameras = split_cameras

        # select only images in the desired shot
        img_files, _ = get_shot_img_files(
            self.data_sources["shots"], shot_idx, pad_shot
        )
        end_idx = end_idx if end_idx > 0 else len(img_files)
        self.data_start, self.data_end = start_idx, end_idx
        img_files = img_files[start_idx:end_idx]
        self.img_names = [get_name(f) for f in img_files]
        self.num_imgs = len(self.img_names)

        img_dir = self.data_sources["images"]
        assert os.path.isdir(img_dir)
        self.img_paths = [os.path.join(img_dir, f) for f in img_files]
        img_h, img_w = imageio.imread(self.img_paths[0]).shape[:2]
        self.img_size = img_w, img_h
        print(f"USING TOTAL {self.num_imgs} {img_w}x{img_h} IMGS")

        # find the tracks in the video
        track_root = self.data_sources["tracks"]
        if tid_spec == "all" or tid_spec.startswith("longest"):
            n_tracks = MAX_NUM_TRACKS
            if tid_spec.startswith("longest"):
                n_tracks = int(tid_spec.split("-")[1])
            # get the longest tracks in the selected shot
            track_ids = sorted(os.listdir(track_root))
            track_paths = [
                [f"{track_root}/{tid}/{name}_keypoints.json" for name in self.img_names]
                for tid in track_ids
            ]
            track_lens = [
                len(list(filter(os.path.isfile, paths))) for paths in track_paths
            ]
            print("raw TRACK IDS and LENGTHS", track_ids, track_lens)
            track_ids = [
                track_ids[i]
                for i in range(len(track_lens))# np.argsort(track_lens)[::-1]
                if track_lens[i] > MIN_TRACK_LEN
            ]
            print("TRACK LENGTHS", track_ids, track_lens)
            track_ids = track_ids[:n_tracks]
        else:
            track_ids = [f"{int(tid):03d}" for tid in tid_spec.split("-")]

        print("TRACK IDS", track_ids)

        self.track_ids = track_ids
        self.n_tracks = len(track_ids)
        self.track_dirs = [os.path.join(track_root, tid) for tid in track_ids]

        # keep a list of frame index masks of whether a track is available in a frame
        sidx = np.inf
        eidx = -1
        self.track_vis_masks = []
        for pred_dir in self.track_dirs:
            kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
            has_kp = [os.path.isfile(x) for x in kp_paths]

            # keep track of which frames this track is visible in
            vis_mask = np.array(has_kp)
            idcs = np.where(vis_mask)[0]
            if len(idcs) > 0:
                si, ei = min(idcs), max(idcs)
                sidx = min(sidx, si)
                eidx = max(eidx, ei)
            self.track_vis_masks.append(vis_mask)

        eidx = max(eidx + 1, 0)
        sidx = min(sidx, eidx)
        print("START", sidx, "END", eidx)
        self.start_idx = sidx
        self.end_idx = eidx
        self.seq_len = eidx - sidx
        self.seq_intervals = [(sidx, eidx) for _ in track_ids]

        self.sel_img_paths = self.img_paths[sidx:eidx]
        self.sel_img_names = self.img_names[sidx:eidx]

        # used to cache data
        self.data_dict = {}
        self.cam_data = None

    def __len__(self):
        return self.n_tracks

    def load_data(self, interp_input=True):
        if len(self.data_dict) > 0:
            return

        # load camera data
        self.load_camera_data()
        # get data for each track
        data_out = {
            "mask_paths": [],
            "floor_plane": [],
            "joints2d": [],
            "vis_mask": [],
            "track_interval": [],
            "init_body_pose": [],
            "init_body_shape": [],
            "init_root_orient": [],
            "init_trans": [],
            "is_right": []
        }

        # create batches of sequences
        # each batch is a track for a person
        T = self.seq_len
        sidx, eidx = self.start_idx, self.end_idx
        for i, tid in enumerate(self.track_ids):
            if len(self.track_ids) == 2:
                assert int(i) == int(tid), f'{i}/{tid}/{self.track_ids}'
            # load mask of visible frames for this track
            vis_mask = self.track_vis_masks[i][sidx:eidx]  # (T)
            vis_idcs = np.where(vis_mask)[0]
            track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
            data_out["track_interval"].append([track_s, track_e])

            vis_mask = get_ternary_mask(vis_mask)
            data_out["vis_mask"].append(vis_mask)

            # load 2d keypoints for visible frames with interpolation
            kp_paths = [
                f"{self.track_dirs[i]}/{x}_keypoints.json" for x in self.sel_img_names
            ]
            # (T, J, 3) (x, y, conf) - with interpolation for missing frames
            joints2d_data = load_keypoints_with_interp(kp_paths, interp=interp_input)
            # print('joints2d_data: ', joints2d_data.shape)
            # print('joints2d_data[:, :, [2]]: ', np.unique(joints2d_data[:, :, [2]]))
            # raise ValueError
            # Discard bad ViTPose detections
            joints2d_data[:, :, 2] = 1.0
            joints2d_data[
                np.repeat(joints2d_data[:, :, [2]] < MIN_KEYP_CONF, 3, axis=2)
            ] = 0
            # Set all confidence values to 1.0 for optimization
            data_out["joints2d"].append(joints2d_data)

            # load single image mano predictions
            pred_paths = [
                f"{self.track_dirs[i]}/{x}_mano.json" for x in self.sel_img_names
            ]
            pose_init, orient_init, trans_init, betas_init, is_right = load_mano_preds(
                pred_paths, tid=tid, interp=interp_input
            )

            n_joints = len(MANO_JOINTS) - 1 # 15
            data_out["init_body_pose"].append(pose_init)
            data_out["init_body_shape"].append(betas_init)
            data_out["init_root_orient"].append(orient_init)
            data_out["init_trans"].append(trans_init)
            data_out['is_right'].append(is_right)
            
            # DEBUG: Print hand type for each track
            print(f"DEBUG: Track {i} (tid={tid}): is_right={is_right[0] if len(is_right) > 0 else 'empty'}, {len(pred_paths)} frames")

            # data_out["floor_plane"].append(DEFAULT_GROUND[:3] * DEFAULT_GROUND[3:])

        self.data_dict = data_out
        # save joints2d (observed from ViTPose) for 2d joint loss in optimization
        dynhamr_output_dir = os.environ.get('DYNHAMR_OUTPUT_DIR', '.')
        torch.save(self.data_dict['joints2d'], os.path.join(dynhamr_output_dir, 'joints2d.pt'))
        print(f"Saved joints2d to {os.path.join(dynhamr_output_dir, 'joints2d.pt')}")

    def __getitem__(self, idx):
        if len(self.data_dict) < 1:
            self.load_data()

        obs_data = dict()

        # 2D keypoints
        joint2d_data = self.data_dict["joints2d"][idx]
        obs_data["joints2d"] = torch.Tensor(joint2d_data)

        # single frame predictions
        obs_data["init_body_pose"] = torch.Tensor(self.data_dict["init_body_pose"][idx])
        obs_data["init_body_shape"] = torch.Tensor(self.data_dict["init_body_shape"][idx])
        obs_data["init_root_orient"] = torch.Tensor(
            self.data_dict["init_root_orient"][idx]
        )
        obs_data["init_trans"] = torch.Tensor(self.data_dict["init_trans"][idx])
        obs_data["is_right"] = torch.Tensor(self.data_dict["is_right"][idx])

        # floor plane
        # obs_data["floor_plane"] = torch.Tensor(self.data_dict["floor_plane"][idx])

        # the frames the track is visible in
        obs_data["vis_mask"] = torch.Tensor(self.data_dict["vis_mask"][idx])

        # the frames used in this subsequence
        obs_data["seq_interval"] = torch.Tensor(list(self.seq_intervals[idx])).to(
            torch.int
        )
        # the start and end interval of available keypoints
        obs_data["track_interval"] = torch.Tensor(
            self.data_dict["track_interval"][idx]
        ).int()

        print(len(self.track_ids), self.track_ids[idx], obs_data["is_right"].shape, obs_data["seq_interval"].shape)

        obs_data["track_id"] = int(self.track_ids[idx])
        # print(self.track_ids[idx], obs_data["is_right"])
        # print(self.track_ids[idx] == obs_data["is_right"])
        # is_right should be constant across time for each track
        assert torch.all(obs_data["is_right"] == obs_data["is_right"][0]), "is_right should be constant"
        assert int(self.track_ids[idx]) == int(obs_data["is_right"][0]), f"Track ID {self.track_ids[idx]} != is_right {obs_data['is_right'][0]}"
        obs_data["seq_name"] = self.seq_name
        return obs_data

    def load_camera_data(self):
        cam_dir = self.data_sources["cameras"]
        data_interval = 0, -1
        if self.split_cameras:
            data_interval = self.data_start, self.data_end
        track_interval = self.start_idx, self.end_idx
        self.cam_data = CameraData(
            cam_dir, self.seq_len, self.img_size, self.is_static, data_interval, track_interval
        )

    def get_camera_data(self):
        if self.cam_data is None:
            raise ValueError
        return self.cam_data.as_dict()


class CameraData(object):
    def __init__(
        self, cam_dir, seq_len, img_size, is_static, data_interval=[0, -1], track_interval=[0, -1]
    ):
        self.img_size = img_size
        self.cam_dir = cam_dir
        self.is_static = is_static

        # inclusive exclusive
        data_start, data_end = data_interval
        if data_end < 0:
            data_end += seq_len + 1
        data_len = data_end - data_start

        # start and end indices are with respect to the data interval
        sidx, eidx = track_interval
        if eidx < 0:
            eidx += data_len + 1

        self.sidx, self.eidx = sidx + data_start, eidx + data_start
        print(self.sidx, self.eidx, data_start, data_end, data_interval, track_interval)
#
        self.seq_len = self.eidx - self.sidx

        self.load_data()

    def load_data(self):
        # camera info
        sidx, eidx = self.sidx, self.eidx
        img_w, img_h = self.img_size
        fpath = os.path.join(self.cam_dir, "cameras.npz")
        if os.path.isfile(fpath) and not self.is_static:
            Logger.log(f"Loading cameras from {fpath}...")
            cam_R, cam_t, intrins, width, height = load_cameras_npz(fpath, self.seq_len)
            scale = img_w / width
            self.intrins = scale * intrins[sidx:eidx]
            # move first camera to origin
            #             R0, t0 = invert_camera(cam_R[sidx], cam_t[sidx])
            #             self.cam_R = torch.einsum("ij,...jk->...ik", R0, cam_R[sidx:eidx])
            #             self.cam_t = t0 + torch.einsum("ij,...j->...i", R0, cam_t[sidx:eidx])
#             t0 = -cam_t[sidx:eidx].mean(dim=0) + torch.randn(3) * 0.1
            
            # t0 = -cam_t[sidx:sidx+1] + torch.randn(3) * 0.1
            
            self.cam_R = cam_R[sidx:eidx]
            self.cam_t = cam_t[sidx:eidx] #- t0
        else:
            # raise ValueError
            Logger.log(f"WARNING: {fpath} does not exist, using static cameras...")
            default_focal = 0.5 * (img_h + img_w)
            self.intrins = torch.tensor(
                [default_focal, default_focal, img_w / 2, img_h / 2]
            )[None].repeat(self.seq_len, 1)

            self.cam_R = torch.eye(3)[None].repeat(self.seq_len, 1, 1)
            self.cam_t = torch.zeros(self.seq_len, 3)

        Logger.log(f"Images have {img_w}x{img_h}, intrins {self.intrins[0]}")
        print("CAMERA DATA", self.cam_R.shape, self.cam_t.shape, self.intrins[0])

    def world2cam(self):
        return self.cam_R, self.cam_t

    def cam2world(self):
        # print('dataset.py, cam2world: ', self.cam_R)
        R = self.cam_R.transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, self.cam_t)
        # print('dataset.py, cam2world: ', R)
        return R, t

    def as_dict(self):
        return {
            "cam_R": self.cam_R,  # (T, 3, 3)
            "cam_t": self.cam_t,  # (T, 3)
            "intrins": self.intrins,  # (T, 4)
            "static": self.is_static,  # bool
        }


def get_ternary_mask(vis_mask):
    # get the track start and end idcs relative to the filtered interval
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
    # -1 = track out of scene, 0 = occlusion, 1 = visible
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1
    vis_mask[track_e:] = -1
    return vis_mask


def get_shot_img_files(shots_path, shot_idx, shot_pad=SHOT_PAD):
    assert os.path.isfile(shots_path)
    with open(shots_path, "r") as f:
        shots_dict = json.load(f)
    img_names = sorted(shots_dict.keys())
    N = len(img_names)
    shot_mask = np.array([shots_dict[x] == shot_idx for x in img_names])

    idcs = np.where(shot_mask)[0]
    if shot_pad > 0:  # drop the frames before/after shot change
        raise ValueError
        if min(idcs) > 0:
            idcs = idcs[shot_pad:]
        if len(idcs) > 0 and max(idcs) < N - 1:
            idcs = idcs[:-shot_pad]
        if len(idcs) < MIN_SEQ_LEN:
            raise ValueError("shot is too short for optimization")

        shot_mask = np.zeros(N, dtype=bool)
        shot_mask[idcs] = 1

    sel_paths = [img_names[i] for i in idcs]
    print(f"FOUND {len(idcs)}/{len(shots_dict)} FRAMES FOR SHOT {shot_idx}")
    return sel_paths, idcs


def load_cameras_npz(camera_path, v):
    assert os.path.splitext(camera_path)[-1] == ".npz"

    cam_data = np.load(camera_path)
    height, width, focal = (
        int(cam_data["height"]),
        int(cam_data["width"]),
        float(cam_data["focal"]),
    )

    w2c = torch.from_numpy(cam_data["w2c"])  # (N, 4, 4)
    cam_R = w2c[:, :3, :3]  # (N, 3, 3)
    cam_t = w2c[:, :3, 3]  # (N, 3)
    N = len(w2c)

    if "intrins" in cam_data:
        intrins = torch.from_numpy(cam_data["intrins"].astype(np.float32))
    else:
        intrins = torch.tensor([focal, focal, width / 2, height / 2])[None].repeat(N, 1)

    print(f"Loaded {N} cameras")
    return cam_R, cam_t, intrins, width, height


def is_image(x):
    return (x.endswith(".png") or x.endswith(".jpg")) and not x.startswith(".")


def get_name(x):
    return os.path.splitext(os.path.basename(x))[0]


def split_name(x, suffix):
    return os.path.basename(x).split(suffix)[0]


def get_names_in_dir(d, suffix):
    files = [split_name(x, suffix) for x in glob.glob(f"{d}/*{suffix}")]
    return sorted(files)


def batch_join(parent, names, suffix=""):
    return [os.path.join(parent, f"{n}{suffix}") for n in names]
