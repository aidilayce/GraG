import os
import glob
import json
import subprocess
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """Inference-safe fallback when the optional TensorBoard package is absent."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from data import get_dataset_from_cfg, expand_source_paths

# from humor.humor_model import HumorModel
from optim.base_scene import BaseSceneModel

from optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer
)
from optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from body_model import MANO
from util.loaders import (
    # load_vposer,
    # load_state,
    # load_gmm,
    resolve_cfg_paths,
)
from util.logger import Logger
from util.tensor import get_device, move_to, detach_all, to_torch

import hydra
from omegaconf import DictConfig, OmegaConf
import time

N_STAGES = 3

import sys
# sys.path.append('src/human_body_prior')
# sys.path.append('HMP/')
# print(sys.path)
# print(torch.cuda.memory_summary())
# from HMP.fitting import run_prior

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_opt(cfg, dataset, out_dir, device):
    a = time.time()
    args = cfg.data
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    obs_data = move_to(next(iter(loader)), device)
    cam_data = move_to(dataset.get_camera_data(), device)
    print("Batch size (dataset_length), T (dataset.seq_len): ", B, len(dataset), T)
    # print("OBS DATA", obs_data.keys())
    # print("CAM DATA", cam_data.keys(), 'cam_R', cam_data['cam_R'])

    # save cameras
    cam_R, cam_t = dataset.cam_data.cam2world()
    intrins = dataset.cam_data.intrins
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins)

    # check whether the cameras are static
    # if static, cannot optimize scale
    cfg.model.opt_scale &= not dataset.cam_data.is_static
    Logger.log(f"OPT SCALE {cfg.model.opt_scale}")

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]
    max_loss_weights = {k: max(wts) for k, wts in all_loss_weights.items()}

    # load models
    cfg = resolve_cfg_paths(cfg)
    cfg.paths.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    # pose_prior, _ = load_model(paths.vposer, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
    # pose_prior = pose_prior.to(device)
    pose_prior = None

    # load models
    cfg = resolve_cfg_paths(cfg)
    paths = cfg.paths
    Logger.log(f"Loading hand model")
    # Instantiate MANO model
    mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
    print('initializing MANO model with cfgs:', mano_cfg)
    hand_model = MANO(batch_size=B*T, pose2rot=True, **mano_cfg).to(device)

    ################################################################
    ######################## optimization ##########################
    ################################################################
    margs = cfg.model
    base_model = BaseSceneModel(
        B, T, hand_model, pose_prior, **margs
    )

    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_input_poses(dataset, os.path.join(out_dir, "hamer"), args.seq)
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)

    opts = cfg.optim.options
    vis_scale = 0.25
    vis = None
    if opts.vis_every > 0:
        from vis.viewer import init_viewer

        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
            fps=cfg.fps,
        )
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir)

    print('start optimization')
    a = time.time()
    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)

    args = cfg.optim.smooth
    print(args)
    b = time.time()
    print('root optimization time: ', b - a)
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data, args.num_iters, out_dir, vis, writer)
    c = time.time()
    print('Smooth optimization time: ', c - b)

    # # HMP
    # if cfg.run_prior and not os.path.exists(os.path.join(out_dir, 'prior')):
    #     run_prior(cfg, dataset, out_dir, device, ['smooth_fit'], \
    #     obs_data, hand_model, cfg, cfg.data, os.path.join(out_dir, 'prior'))
    d = time.time()
    print('prior optimization time: ', d-c)


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    print('run_opt.py: ', cfg)

    # Set random seed
    set_seed(cfg.get('seed', 42))

    out_dir = os.getcwd()
    print("out_dir", out_dir)
    Logger.init(f"{out_dir}/opt_log.txt")

    # make sure we get all necessary inputs
    print("init SOURCES", cfg.data.sources)
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)

    dataset = get_dataset_from_cfg(cfg)
    save_track_info(dataset, out_dir)
    # GPU indices are local to the devices exposed by Slurm or the caller.
    device_id = cfg.get("gpu")
    # breakpoint()

     # run optimization
    if cfg.run_opt:
        device = get_device(device_id)
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        from run_vis import run_vis

        run_vis(
            cfg, dataset, out_dir, device_id, **cfg.get("vis", dict())
        )


if __name__ == "__main__":
    main()
