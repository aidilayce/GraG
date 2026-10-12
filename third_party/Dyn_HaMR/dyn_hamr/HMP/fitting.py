import os
import cv2
import copy
import glob
import time
import torch
# import ffmpeg
import joblib
import random
import shutil
import subprocess
import numpy as np
import torch.nn as nn
from loguru import logger 
from argparse import Namespace
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

# import scenepic_viz
# from utils import slerp
# import open3d_viz_overlay
# from datasets.amass import *

from arguments import Arguments
from argparse import ArgumentParser
from nemf.generative import Architecture
from nemf.fk import ForwardKinematicsLayer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from nemf.losses import GeodesicLoss, pos_smooth_loss, rot_smooth_loss
from rotations import (axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_rotation_6d,
                        quaternion_to_matrix, rotation_6d_to_matrix, axis_angle_to_quaternion, quaternion_to_axis_angle, quat_to_aa)
from fitting_utils import (process_gt, gmof, perspective_projection, get_joints2d, run_pymafx, run_metro, process_pymafx_mano, compute_seq_intervals,
                        save_quantitative_evaluation, get_seqname_ho3d_v3, get_seqname_arctic_data, get_seqname_in_the_wild, get_seqname_dexycb,
                        RIGHT_WRIST_BASE_LOC, joints2d_loss, # map_openpose_joints_to_mano, map_mano_joints_to_openpose,
                        export_pymafx_json, blend_keypoints, BMCLoss)
from utils import estimate_angular_velocity, estimate_linear_velocity, render_openpose
bmc = BMCLoss(lambda_bl=1, lambda_rb=1, lambda_a=1)

HAND_JOINT_NUM = 16
# mano2openpose = map_mano_joints_to_openpose()
# openpose2mano = map_openpose_joints_to_mano()

IGNORE_KEYS = ['cam_f', 'cam_center', 'img_height', 'img_width', 'img_dir',
                'save_path', 'frame_id', 'config_type', 'rh_verts', 'handedness']
POSSIBLE_KEYP_SOURCES = ["mmpose", "mediapipe", "mediapipe_std", "mediapipe_multiview", 
                                    "pymafx", "pymafx_std", "gt", "metro", "blend",
                                    "blend_mediapipe_std_mmpose",
                                      "blend_std", "blend_smooth"]

MANO_JOINTS = {
    'wrist': 0,
    'index1': 1,
    'index2': 2,
    'index3': 3,
    'middle1': 4,
    'middle2': 5,
    'middle3': 6,
    'pinky1': 7,
    'pinky2': 8,
    'pinky3': 9,
    'ring1': 10,
    'ring2': 11,
    'ring3': 12,
    'thumb1': 13,
    'thumb2': 14,
    'thumb3': 15,
}
def run_mano(body_model, trans, root_orient, body_pose, is_right, betas=None, only_right=False):
    """
    Forward pass of the MANO model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    B, T, _ = trans.shape
    bm_batch_size = body_model.batch_size
    assert bm_batch_size % B == 0
    # assert (is_right[0]==is_right[0][0]).all(), f'{is_right}'
    # assert (is_right[1]==is_right[1][0]).all(), f'{is_right}'

    seq_len = bm_batch_size // B
    bm_num_betas = body_model.num_betas
    J_BODY = len(MANO_JOINTS) - 1  # all joints except root
    # print('utils.py, seq_len, T: ', seq_len, T, bm_batch_size, B) # 1, 120, 1, 1
    # print('trans, root_orient, body_pose:', trans.shape, root_orient.shape, body_pose.shape)
    if T == 1:
        raise ValueError
        # must expand to use with body model
        trans = trans.expand(B, seq_len, 3)
        root_orient = root_orient.expand(B, seq_len, 3)
        body_pose = body_pose.expand(B, seq_len, J_BODY * 3)
    elif T != seq_len:
        # print(trans.shape, root_orient.shape, body_pose.shape)
        trans, root_orient, body_pose = zero_pad_tensors(
            [trans, root_orient, body_pose], seq_len - T
        )
    if betas is None:
        betas = torch.zeros(B, bm_num_betas, device=trans.device)
    betas = betas.reshape((B, 1, bm_num_betas)).expand((B, seq_len, bm_num_betas))
    # print('body_pose: ', body_pose.reshape((B * seq_len, -1)).shape)

    mano_output = body_model(
        hand_pose=body_pose.reshape((B * seq_len, -1)),
        betas=betas.reshape((B * seq_len, -1)),
        global_orient=root_orient.reshape((B * seq_len, -1)),
        transl=trans.reshape((B * seq_len, -1)),
    )
    joints = mano_output.joints
    verts = mano_output.vertices

    joints = joints.reshape(B, seq_len, -1, 3)[:, :T]
    verts = verts.reshape(B, seq_len, -1, 3)[:, :T]
    is_right = is_right.unsqueeze(-1)

    if not only_right:    
        joints[:, :, :, 0] = (2*is_right-1)*joints[:, :, :, 0]
        verts[:, :, :, 0] = (2*is_right-1)*verts[:, :, :, 0]

    return {
        "joints": joints,
        "vertices": verts,
        "l_faces": body_model.faces_tensor[:,[0,2,1]],
        "r_faces": body_model.faces_tensor,
        "is_right": is_right.squeeze(-1),
        'body_pose': body_pose.clone()
    }

def zero_pad_tensors(pad_list, pad_size):
    """
    Assumes tensors in pad_list are B x T x D and pad temporal dimension
    """
    B = pad_list[0].size(0)
    new_pad_list = []
    for pad_idx, pad_tensor in enumerate(pad_list):
        padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
        new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
    return new_pad_list

def pred_mano(hand_model, trans, root_orient, body_pose, is_right, betas, only_right=False):
    """
    Forward pass of the MANO model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : B x D
    """
    mano_out = run_mano(hand_model, trans, root_orient, body_pose, is_right, betas=betas, only_right=only_right)
    joints3d, points3d = mano_out["joints"], mano_out["vertices"]
    verts3d = points3d

    return {
        "is_right": is_right,
        "points3d": points3d,  # all vertices
        "verts3d": verts3d,  # keypoint vertices
        "joints3d": joints3d,  # smpl joints
        "joints3d_op": joints3d,  # OP joints
        "l_faces": mano_out["l_faces"],  # index array of faces
        "r_faces": mano_out["r_faces"],  # index array of faces
        "body_pose": mano_out["body_pose"]
    }

def reproject(points3d, cam_R, cam_t, cam_f, cam_center):
    """
    reproject points3d into the scene cameras
    :param points3d (B, T, N, 3)
    :param cam_R (B, T, 3, 3)
    :param cam_t (B, T, 3)
    :param cam_f (T, 2)
    :param cam_center (T, 2)
    """
    B, T, N, _ = points3d.shape
    points3d = torch.einsum("btij,btnj->btni", cam_R, points3d)
    points3d = points3d + cam_t[..., None, :]  # (B, T, N, 3)
    points2d = points3d[..., :2] / points3d[..., 2:3]
    points2d = cam_f[None, :, None] * points2d + cam_center[None, :, None]
    return points2d

def get_stage2_res(base_dir, device, npz_init_dict, hand_model):
    args = Arguments(base_dir, os.path.dirname(__file__), filename='amass.yaml')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fk = ForwardKinematicsLayer(args)
    to_th = lambda x: torch.from_numpy(x).to(device)
    
    cdata = npz_init_dict

    gender = 'neutral' 
    if gender.startswith('b'):
        gender = str(cdata['gender'], encoding='utf-8')

    N = len(cdata['poses'])
    print(f'Sequence has {N} frames')

    # no matter what the keypoint source is, we need to read the bbox from mmpose
    # joint2d_data = to_th(np.stack(keyp_frames, axis=0)) # T x J x 3 (x,y,conf)
    joint2d_data = npz_init_dict['keyp2d']
    is_right = npz_init_dict['is_right']
    vis_mask = npz_init_dict['vis_mask']

    assert (len(np.where(is_right==1)[0])==0) or (len(np.where(is_right==0)[0])==0), f'{len(np.where(is_right==0)[0]), len(np.where(is_right==1)[0])}'

    # ONLY FOR RIGHT HAND
    root_orient_aa = cdata['root_orient']
    pose_body = cdata['poses']
    data_poses = np.concatenate((root_orient_aa, pose_body), axis=1)
 
    pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
    pose = pose.view(-1, 15 + 1, 3)  # axis-angle (T, J, 3)
    assert len(pose) == 128

    trans = torch.from_numpy(np.asarray(cdata['trans'], np.float32)).to(device)  # global translation (T, 3)

    # Compute necessary data for model training.
    rotmat = axis_angle_to_matrix(pose)  # rotation matrix (T, J, 3, 3)
    root_orient = rotmat[:, 0].clone()
    root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (T, 6)     

    # print(pose.shape, trans.shape, rotmat.shape, root_orient.shape)
    # raise ValueError
    # defined in amass.yaml. Set as True
    if args.unified_orientation:
        identity = torch.eye(3).cuda()
        identity = identity.view(1, 3, 3).repeat(rotmat.shape[0], 1, 1)
        rotmat[:, 0] = identity
    rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (T, J, 6)

    rot_seq = rotmat.clone()
    angular = estimate_angular_velocity(rot_seq.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # angular velocity of all the joints (T, J, 3)

    pos, global_xform = fk(rot6d)  # local joint positions (T, J, 3), global transformation matrix for each joint (T, J, 4, 4)

    pos = pos.contiguous()
    global_xform = global_xform.contiguous()
    velocity = estimate_linear_velocity(pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of all the joints (T, J, 3)

    # defined in amass.yaml. Set as True
    if args.unified_orientation:
        root_rotation = rotation_6d_to_matrix(root_orient)  # (T, 3, 3)
        root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)
        global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        height = global_pos + trans.unsqueeze(1)
    else:
        height = pos + trans.unsqueeze(1)
    height = height[..., 'xyz'.index(args.data.up)]  # (T, J)
    
    root_vel = estimate_linear_velocity(trans.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of the root joint (T, 3)

    global_xform = global_xform[:, :, :3, :3]  # (T, J, 3, 3)
    global_xform = matrix_to_rotation_6d(global_xform)  # (T, J, 6)

    betas = to_th(cdata["betas"])
    out_mano = pred_mano(hand_model, \
    trans[None], torch.tensor(root_orient_aa).to(device)[None], \
    torch.tensor(pose_body).to(device)[None], torch.tensor(is_right).to(device)[None], \
    torch.tensor(cdata["betas"]).to(device)[None], only_right=True)

    assert len(out_mano['joints3d']) == 1
    joints3d, vertices3d = out_mano['joints3d'][0], out_mano['verts3d'][0]
    # print(npz_init_dict.keys())
    # for i in npz_init_dict.keys():
    #     print(i, npz_init_dict[i].shape)

    data = {'rotmat': rotmat,
            'pos': pos,
            'trans': trans,
            'root_vel': root_vel,
            'height': height,
            'rot6d': rot6d,
            'angular': angular,
            'betas': betas,
            'global_xform': global_xform,
            'velocity': velocity,
            'root_orient': root_orient,
            'joints2d': torch.tensor(joint2d_data).to(device),
            'joints3d': joints3d,
            'vertices': vertices3d,
            'cam_t': torch.tensor(npz_init_dict['cam_t']).to(device),
            "cam_R": torch.tensor(npz_init_dict['cam_R']).to(device),
            "cam_center": torch.tensor(npz_init_dict['cam_center']).to(device),
            "cam_f": torch.tensor(npz_init_dict['cam_f']).to(device),
            'is_right': torch.tensor(is_right).to(device),
            'vis_mask': torch.tensor(vis_mask).to(device),
            }
    # for i in data.keys():
    #     print(i, data[i].shape)

    return data

def forward_mano(output):
    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    B, T, J, _, _ = rotmat.size()
    
    # b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.view(-1, J, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(B, -1, J, 3, 3)

    root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
    
    local_rotmat[:, :, 0] = root_orient

    poses = matrix_to_axis_angle(local_rotmat)  # (T, J, 3)
    poses = poses.view(-1, J * 3)

    # no_shift is the flag for not shifting the wrist location to the origin
    mano_out = args.hand_model(input_dict={"betas":output['betas'].view(-1, 10),
                            "global_orient":poses[..., :3].view(-1, 3),
                            "hand_pose":poses[..., 3:].view(-1, 45),
                            "no_shift":True,
                            "return_finger_tips": True,
                            "transl":output['trans'].view(-1, 3)})
 
    return mano_out


def L_pose_prior(output):

    hposer.eval()

    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    B, T, J, _, _ = rotmat.size()
    
    # b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.view(-1, J, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(B, -1, J, 3, 3)

    local_rotmat_casted = local_rotmat[:, :, 1:, ...].view(B*T, -1)
  
    pose_latent_code = hposer.encode(local_rotmat_casted)
    pose_prior_mean_squared = (pose_latent_code.mean ** 2).mean(-1)
    
    loss = pose_prior_mean_squared.mean()
    
    return loss


def L_rot(pred, gt, T, conf=None):
    """
    Args:
        source, target: rotation matrices in the shape B x T x J x 3 x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    if conf is not None:
        criterion_rec = nn.L1Loss(reduction='none') if args.l1_loss else nn.MSELoss(reduction='none')
    else:
        criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()
    
    B, seqlen, J, _, _ = pred.shape
   
    if args.geodesic_loss:
        if conf is not None:
            loss = (conf.squeeze(-1) ** 2) *  criterion_geo(pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3), reduction='none').reshape(B, seqlen, J)
            loss = loss.mean()
        else:
            loss = criterion_geo(pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3))
    else:
        if conf is not None:
            loss = (conf.unsqueeze(-1) ** 2) *  criterion_rec(pred[:, T], gt[:, T])
            loss = loss.mean()
        else:
            loss = criterion_rec(pred[:, T], gt[:, T])

    return loss

def pose_prior_loss(latent_pose_pred, mask=None):
    """
    :param latent_pose_pred (B, T, D)
    :param mask (optional) (B, T)
    """
    # prior is isotropic gaussian so take L2 distance from 0
    loss = latent_pose_pred**2
    if mask is not None:
        loss = loss[mask.bool()]
    loss = torch.sum(loss)
    return loss


def L_PCA(pose):
    
    bs = pose.shape[0]
    
    # convert aa to rotmat 
    pose_rotmat_global = axis_angle_to_matrix(pose)
    
    # convert global to local 
    pose_rotmat_local = fk.global_to_local(pose_rotmat_global.reshape(-1, 16, 3, 3))
    
    # convert back to aa 
    pose_aa_local = matrix_to_axis_angle(pose_rotmat_local)
    
    pose_reshaped = pose_aa_local[:, 1:, :].reshape(-1, args.data.clip_length * 15 * 3)
    
    pose_reshaped_centered = pose_reshaped - pca_mean
    
    pose_projected = pose_reshaped_centered @ pca_pc.T  
    
    normalized_pose_loss = abs(pose_projected) / pca_sv 
     
    mp_loss = normalized_pose_loss.mean(0).mean()

    return mp_loss

def L_GMM(pose):
    
    bs = pose.shape[0]
        
    # convert aa to rotmat 
    pose_rotmat_global = axis_angle_to_matrix(pose)
    
    # convert global to local 
    pose_rotmat_local = fk.global_to_local(pose_rotmat_global.reshape(-1, 16, 3, 3))
    
    # convert back to aa 
    pose_aa_local = matrix_to_axis_angle(pose_rotmat_local).reshape(bs, args.data.clip_length, -1, 3)
    
    mp_loss_list = []
    loss_tot = 0
    
    for i in range(bs):
        loss_i = gmm_aa.log_likelihood(pose_aa_local[i, :, 1:].reshape(1, -1).cpu())
        loss_tot += loss_i
        
    mp_loss = loss_tot / bs     
    return mp_loss.to("cuda")

def L_orient(source, target, T, bbox_conf=None):
    """
    Args:
        source: predicted root orientation in the shape B x T x 6.
        target: root orientation in the shape of B x T x 6.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the root orientation.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()

    source = rotation_6d_to_matrix(source)  # (B, T, 3, 3)
    target = rotation_6d_to_matrix(target)  # (B, T, 3, 3)

    if args.geodesic_loss:
        
        if bbox_conf is not None:
            
            loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3), reduction='none')
            bbox_conf_coef = bbox_conf.reshape(-1)
            loss = ((bbox_conf_coef ** 2) * loss).mean()
            
        else:
            loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
            
    else:
        loss = criterion_rec(source[:, T], target[:, T])
    
    return loss


def L_trans(source, target, T, bbox_conf=None):
    """
    Args:
        source: predict global translation in the shape B x T x 3 (the origin is (0, 0, height)).
        target: global translation of the root joint in the shape B x T x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the global translation.
    """
 
    trans = source
    trans_gt = target
     
    # dont make reduction and weight by bbox_conf
    if bbox_conf is not None:
        criterion_pred = nn.L1Loss(reduction='none') if args.l1_loss else nn.MSELoss(reduction='none')
        
        # reshape to (T * N, 3)
        loss = criterion_pred(trans[:, T].reshape(-1, 3), trans_gt[:, T].reshape(-1, 3)).mean(1)
        bbox_conf_coef = bbox_conf.reshape(-1)
        loss = ((bbox_conf_coef **2) * loss).mean()
        
    else:
        criterion_pred = nn.L1Loss() if args.l1_loss else nn.MSELoss()
        loss = criterion_pred(trans[:, T], trans_gt[:, T])

    return loss


def L_pos(source, target, T):
    """
    Args:
        source, target: joint local positions in the shape B x T x J x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the joint local positions.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    loss = criterion_rec(source[:, T], target[:, T])

    return loss

 
def contact_vel_loss(contacts_conf, joints3d):
    '''
    Velocity should be zero at predicted contacts
    '''
    delta_pos = (joints3d[:,1:] - joints3d[:,:-1])**2
    cur_loss = delta_pos.sum(dim=-1) * contacts_conf[:,1:]
    cur_loss = 0.5 * torch.mean(cur_loss)

    return cur_loss

def motion_prior_loss(latent_motion_pred):
    # assume standard normal
    loss = latent_motion_pred**2
    loss = torch.mean(loss)
    
    return loss


def motion_reconstruction(hand_model, target, output_dir, steps, T=None, idx=0):
 
    z_l, _, _ = model.encode_local()
    z_g, _, _ = model.encode_global()
     
    # optimize pose directly 
    pose_aa = matrix_to_axis_angle(model.input_data["rotmat"].detach().clone()) if motion_prior_type in ['gmm', 'pca'] else None

    z_l, z_g, opt_betas, opt_root_orient, opt_trans, cam_R, cam_t, cam_f, cam_center, opt_pose_aa = latent_optimization(hand_model, target, T=T, z_l=z_l, z_g=z_g, pose=pose_aa)

    for step in steps:
    
        fps = int(args.data.fps / step)

        with torch.no_grad():
            B, seqlen, _ = opt_trans.shape
                
            if motion_prior_type in ["pca", "gmm"]:
                output = {"rotmat": axis_angle_to_matrix(opt_pose_aa)}    
            else:
                output = model.decode(z_l, z_g=z_g, length=args.data.clip_length, step=step)

            rotmat = output['rotmat']  # (B, T, J, 3, 3)
            rotmat_gt = target['rotmat']  # (B, T, J, 3, 3)
            b_size, _, n_joints = rotmat.shape[:3]
            
            local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
            local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
            # local_rotmat_gt = fk.global_to_local(rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
            # local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)

            root_orient = rotation_6d_to_matrix(opt_root_orient)  # (B, T, 3, 3)
            # root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
            
            global_trans = opt_trans.clone()
            B = global_trans.shape[0] 
        
            output['betas'] = opt_betas[:, None, None, :].repeat(1, B, seqlen, 1)
            output['trans'] = opt_trans.clone().reshape(B*args.nsubject, seqlen, 3)
            output['root_orient'] = opt_root_orient

            ####################
            body_pose = matrix_to_axis_angle(local_rotmat)[:, :, 1:] # matrix_to_axis_angle(local_rotmat)
            is_right = target['is_right'].clone()
            ####################

            R = matrix_to_axis_angle(rotation_6d_to_matrix(opt_root_orient))
            T = output['trans']
            P = body_pose.reshape(1, 128, 45)
            Be = opt_betas

        DR = matrix_to_axis_angle(local_rotmat[:, :, 0]).clone()
        if args.data.root_transform:
            local_rotmat[:, :, 0] = root_orient
            # local_rotmat_gt[:, :, 0] = root_orient_gt

        trans = global_trans 
        trans_gt = target['trans']  # (B, T, 3)

        if trans.shape[0] > 1:
            raise ValueError

        assert local_rotmat.shape[0] == 1
        return R, T, P, Be, DR
        # for i in range(local_rotmat.shape[0]): # (1, T. J, 3, 3)
            
        #     poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
        #     poses_gt = c2c(matrix_to_axis_angle(local_rotmat_gt[i]))  # (T, J, 3)

        #     poses = poses.reshape((poses.shape[0], -1))  # (T, 48)
        #     poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, 66)

        #     limit = poses.shape[0]
            
        #     pred_save_path = os.path.join(output_dir, f'recon_{offset + i:03d}_{fps}fps.npz')
        #     gt_save_path = pred_save_path if args.raw_config else os.path.join(output_dir, f'recon_{offset + i:03d}_gt.npz') 

            # save opt results if not _pymafx_raw or _metro_raw 
            # save_keys = ['root_orient', 'trans', 'latent_pose', 'is_right', 'init_body_pose', \
            # 'world_scale', 'betas', 'pose_body', 'cam_R', 'cam_t', 'intrins']

            # if not args.raw_config:
            #     np.savez(pred_save_path,
            #             poses=poses[:limit].detach().cpu().numpy(), 
            #             trans=c2c(trans[i][:limit]), 
            #             betas=opt_betas.detach().cpu().numpy()[:limit], 
            #             gender=args.data.gender, 
            #             mocap_framerate=fps,
            #             cam_R=c2c(target['cam_R'][i][0]),
            #             cam_t=c2c(target["cam_t"][i][0]),
            #             cam_f=c2c(target['cam_f'].unsqueeze(0)),
            #             cam_center=c2c(target['cam_center'].unsqueeze(0)),
            #             joints_2d=c2c(joints2d_pred[i][:limit]),
            #             keypoints_2d=c2c(target['joints2d'][i][:limit]),  # detected keypoints with confidence
            #             vertices=c2c(vertices_pred[i][:limit]),
            #             save_path=pred_save_path,
            #             handedness=target["handedness"].detach().cpu().numpy(),
            #             # img_dir=str(target['img_dir']),
            #             # frame_id=target['frame_id'],
            #             # img_height=target['img_height'],
            #             # img_width=target['img_width'],
            #             # config_type=target['config_type'],
            #             joints_3d=c2c(joints3d_pred)[i][:limit])
             
            # if ("_encode_decode" in gt_save_path) or args.raw_config: 
            #     np.savez(gt_save_path,
            #             poses=poses_gt[:limit].detach().cpu().numpy(), 
            #             trans=c2c(trans_gt[i][:limit]), 
            #             betas=target["betas"][0].detach().cpu().numpy()[:limit], 
            #             vertices=target["rh_verts"].detach().cpu().numpy()[:limit], 
            #             config_type=target['config_type'] if args.raw_config else None,
            #             gender=args.data.gender, 
            #             save_path=gt_save_path,
            #             mocap_framerate=args.data.fps,
            #             # frame_id=target['frame_id'],
            #             # img_height=target['img_height'],
            #             # img_width=target['img_width'],
            #             cam_R=c2c(target['cam_R'][i][0]),
            #             cam_t=c2c(target['cam_t'][i][0]),
            #             cam_f=c2c(target['cam_f'].unsqueeze(0)),
            #             cam_center=c2c(target['cam_center'].unsqueeze(0)),
            #             joints2d=c2c(target['joints2d'][i, ...,])[:limit], 
            #             joints_2d=c2c(target['joints2d'][i, ...,])[:limit], 
            #             keypoints_2d=c2c(target['joints2d'][i, ...,])[:limit], # duplicated, because alignment reads that key value
            #             joints_3d=c2c(target['joints3d'][i])[:limit])

def get_gt_path(expname):
    if args.dataname == "DexYCB":
        gt_path = args.vid_path
        subjectname = args.vid_path.split("/")[4]
    elif args.dataname == "HO3D_v3":
        gt_path = os.path.join(os.path.dirname(args.vid_path), "meta")
        subjectname = args.vid_path.split("/")[-2]
    elif args.dataname == "arctic_data":
        subjectname = args.vid_path.split("/")[-3]
        split_path = os.path.join(f"./external/arctic/data/arctic_data/data/splits/p1_{expname}.npy")
        meta_path = os.path.join(f"./external/arctic/data/arctic_data/data/meta/misc.json")
        gt_path = {"gt_path": args.vid_path, "split": split_path, "meta_path": meta_path}
    else:  
        args.dataname = "in_the_wild"
        subjectname = args.vid_path.split("/")[-2]
        gt_path = os.path.join(os.path.dirname(args.vid_path), "meta")

    return gt_path, subjectname 


def multi_stage_opt(opt, device, obs_data, res_dict, hand_model, config_f, exp_setup_name, init_method_name):
    
    logger.info(f"Running reconstruction with prior")

    config_type = config_f.split('/')[-1].split('.')[:-1]
    config_type = '.'.join(config_type)
    args.raw_config = config_type in ["_pymafx_raw", "_metro_raw"]
    
    keypoint_blend_weight = 1.0
    vid_path = args.vid_path
    abs_video_path = os.path.join(os.getcwd(), args.vid_path)
 
    args.dataname = args.vid_path.split("/")[3]
    args.N_frames = len(glob.glob(os.path.join(abs_video_path, "*.jpg")))
    
    if args.N_frames == 0:
        args.N_frames = len(glob.glob(os.path.join(abs_video_path, "*.png")))
    
    args.hand_model = hand_model
    gt_path, subjectname = get_gt_path(exp_setup_name)
    init_method_out_path = os.path.join(os.path.dirname(vid_path), f"{init_method_name}_out")

    print('loading data...')
    assert len(res_dict) == 1
    res_dict = res_dict[0]
    print('obs_data: ', obs_data.keys())
    print('res_dict: ', res_dict.keys())
    for i in res_dict.keys():
        res_dict[i] = res_dict[i].cpu().detach().numpy()
        print(i, res_dict[i].shape)

    for i in obs_data.keys():
        try:
            obs_data[i] = obs_data[i].cpu().detach().numpy()
        except:
            pass

    R_list = []
    T_list = []
    P_list = []
    Be_list = []
    DR_list = []
    for idx in range(len(res_dict['pose_body'])):

        assert (res_dict['is_right'][idx] == (obs_data['is_right'][idx])).all()
        rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, is_right = \
                            res_dict['root_orient'][idx], res_dict['betas'][idx], res_dict['trans'][idx], res_dict['pose_body'][idx], res_dict['is_right'][idx]

        cam_center = torch.tensor(res_dict['intrins'][2:][None]).repeat(128, 1)  # (T, 2)
        cam_f = torch.tensor(res_dict['intrins'][:2][None]).repeat(128, 1)  # (T, 2)

        init_dict = {
                    "keyp2d": obs_data['joints2d'][idx],
                    "betas": rhand_betas_padded,
                    "trans": rhand_trans_padded,
                    "root_orient": rhand_orient_padded,
                    "poses": rhand_pose_padded.reshape(-1, 45),
                    "cam_R": res_dict['cam_R'][idx],
                    "cam_t": res_dict['cam_t'][idx],
                    "img_dir":abs_video_path,
                    "is_right": res_dict['is_right'][idx],
                    "cam_f": cam_f,
                    "cam_center": cam_center,
                    'vis_mask': obs_data['vis_mask'][idx]
                    }

        # save the results, expand the length by padding so that it matches the number of frames in the video	
        args.pkl_output_dir = os.path.join(args.save_path, "pkls")
        os.makedirs(args.pkl_output_dir, exist_ok=True)

        # run stage3 optimization
        os.makedirs(args.save_path, exist_ok=True)
        data = get_stage2_res(opt.paths.base_dir, device, init_dict, hand_model)

        data['save_path'] =  os.path.join(args.save_path, 'pymaf_output.npz') 
        data['config_type'] = config_type
        args.orig_seq_len = data['trans'].shape[0]

        shutil.copy2(config_f, args.save_path)
        
        data['betas'] = data['betas'][None].repeat(128, 1)

        for k, v in data.items():
            if k in IGNORE_KEYS:
                continue        
            else:
                if k == 'betas':
                    print(k, v.shape)
                if v.shape[0] > 128:
                    
                    # in case of batch optimization, we need to split the data into chunks of 128 frames
                    if args.overlap_len > 0:
                        # compute start and end indices
                        seq_intervals = compute_seq_intervals(v.shape[0], 128, args.overlap_len)
                        data_split = []
                        for seq_s, seq_e in seq_intervals:
                            data_split.append(v[seq_s:seq_e])
                    else:
                        data_split = list(torch.split(v, 128))
                        
                    if data_split[-1].shape[0] == 128:
                        data[k] = torch.stack(data_split, dim=0)
                    else:
                        pad_repeat = 128 - data_split[-1].shape[0]
                        last_el = data_split[-1]
                        last_el = torch.cat([last_el, last_el[-1:].repeat_interleave(pad_repeat, 0)])
                        data_split[-1] = last_el
                        data[k] = torch.stack(data_split, dim=0)
                else:
                    pad_repeat = 128 - v.shape[0]
                    data[k] = torch.cat([v, v[-1:].repeat_interleave(pad_repeat, 0)]).unsqueeze(0) # BxTxJxD

        args.data.clip_length = data['pos'].shape[1]
        model.set_input(data)

        target = dict()
        target['pos'] = data['pos'].to(model.device)
        target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
        target['trans'] = data['trans'].to(model.device)  
        target['root_orient'] = data['root_orient'].to(model.device)
        target['cam_R'] = data['cam_R'].to(model.device) 
        target['cam_t'] = data['cam_t'].to(model.device)
        target['cam_f'] = data['cam_f'].to(model.device).squeeze(0)
        target['cam_center'] = data['cam_center'].to(model.device).squeeze(0)
        target['joints2d'] = data['joints2d'].to(model.device)
        target['joints3d'] = data['joints3d'].to(model.device)
        # target['verts3d'] = data['verts3d'].to(model.device)
        target['betas'] = data['betas'].to(model.device)
        target['save_path'] = data['save_path']
        target['config_type'] = data['config_type']
        target['handedness'] = data['is_right']
        target['vis_mask'] = data['vis_mask']
        target['is_right'] = data['is_right']

        ####################
        # vis for debugging
        # rhand_trans_padded = target['trans']
        # rhand_orient_padded = matrix_to_axis_angle(rotation_6d_to_matrix(target['root_orient']))
        # rhand_betas_padded = target['betas'][:, 0, ]
        # is_right = target['is_right']
        # rhand_pose_padded = fk.global_to_local(target['rotmat'].view(-1, HAND_JOINT_NUM, 3, 3))  # (B x T, J, 3, 3)
        # rhand_pose_padded = rhand_pose_padded.view(1*args.nsubject, -1, HAND_JOINT_NUM, 3, 3) # (B x T, J, 3, 3)
        # rhand_pose_padded = matrix_to_axis_angle(rhand_pose_padded)[:, :, 1:]
        # print(rhand_pose_padded.shape)

        # # print(rhand_trans_padded.shape, rhand_orient_padded.shape, rhand_pose_padded.reshape(1, 128, 45).shape, is_right.shape, rhand_betas_padded.shape)
        # rh_mano_out = pred_mano(hand_model, rhand_trans_padded, rhand_orient_padded, rhand_pose_padded.reshape(1, 128, 45), is_right, rhand_betas_padded, only_right=False) 

        # joints3d_pred = rh_mano_out['joints3d'].view(1, 128, -1, 3)
        # vertices_pred = rh_mano_out['verts3d'].view(1, 128, -1, 3)
        # print(joints3d_pred.shape, vertices_pred.shape)

        # cam_R = torch.tensor(res_dict['cam_R'][idx]).cuda()[None]
        # cam_t = torch.tensor(res_dict['cam_t'][idx]).cuda()[None]
        # cam_center = torch.tensor(res_dict['intrins'][2:][None]).repeat(128, 1).cuda()  # (T, 2)
        # cam_f = torch.tensor(res_dict['intrins'][:2][None]).repeat(128, 1).cuda()  # (T, 2)

        # print(joints3d_pred.shape, cam_R.shape, cam_t.shape, cam_f.shape, cam_center.shape)
        # joints2d_pred = reproject(
        #         joints3d_pred, cam_R, cam_t, cam_f, cam_center
        #     )

        # # 2d keypoints debugging.
        # openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # gt_indices = openpose_indices
        # num_hand = 1
        # num_frames = joints2d_pred.shape[1]
        # for x in range(num_frames):
            
        #     vit_img = cv2.imread(f'/data/home/x/code/hand/Dyn-HaMR/test/images/dance_old/' + str(x+1).zfill(6) + '.jpg')
        #     print(f'/data/home/x/code/hand/Dyn-HaMR/test/images/dance_old/' + str(x+1).zfill(6) + '.jpg')
        #     all_vit_2d = joints2d_pred[:, x].detach().cpu().numpy() # observed_data["joints2d"][:, x].cpu().numpy()
        #     v = np.ones((len(all_vit_2d), 21, 1))
        #     all_vit_2d = np.concatenate((all_vit_2d, v), axis=-1)
        #     for i in range(num_hand):
        #         body_keypoints_2d = all_vit_2d[i, :21].copy()
        #         for op, gt in zip(openpose_indices, gt_indices):
        #             if all_vit_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
        #                 body_keypoints_2d[op] = all_vit_2d[i, gt]
        #         vit_img = render_openpose(vit_img, body_keypoints_2d)
        #     cv2.imwrite(f'/data/home/x/code/hand/Dyn-HaMR/' + str(x+1).zfill(6) + '.jpg', vit_img)

        R, T, P, Be, DR = motion_reconstruction(hand_model, target, args.save_path, steps=[1.0], idx=idx)

        # vis
        is_right = target['is_right'].clone()
        rh_mano_out = pred_mano(hand_model, T, R, P, is_right, Be, only_right=False) 
        joints3d_pred = rh_mano_out['joints3d'].view(1, 128, -1, 3)
        vertices_pred = rh_mano_out['verts3d'].view(1, 128, -1, 3)
        joints2d_pred = reproject(
                joints3d_pred, target['cam_R'], target['cam_t'], target['cam_f'], target['cam_center']
            )

        R_list.append(R.detach().cpu().numpy())
        T_list.append(T.detach().cpu().numpy())
        P_list.append(P.detach().cpu().numpy())
        Be_list.append(Be.detach().cpu().numpy())
        DR_list.append(DR.detach().cpu().numpy())

    save_keys = ['root_orient', 'trans', 'latent_pose', 'is_right', 'init_body_pose', 'world_scale', 'betas', 'pose_body', 'cam_R', 'cam_t', 'intrins']
    R_list = np.vstack(R_list)
    T_list = np.vstack(T_list)
    P_list = np.vstack(P_list)
    Be_list = np.vstack(Be_list)
    DR_list = np.vstack(DR_list)
    res_dict['root_orient'] = R_list
    res_dict['trans'] = T_list
    res_dict['latent_pose'] = P_list
    num_hands = len(P_list)
    res_dict['pose_body'] = P_list.reshape(num_hands, 128, 15, 3)
    res_dict['betas'] = Be_list
    res_dict['decode_root'] = DR_list
    pred_save_path = os.path.join(args.save_path, os.path.basename(args.vid_path).split('.')[0] + f'_000000_world_results.npz')

    for i in res_dict.keys():
        print(i, res_dict[i].shape)
    np.savez(pred_save_path, **res_dict)

def run_quantitative_evaluation(pred_npz_path, gt_dict, pymafx_npz_path, viz_flag, misc={}):

    from eval import alignment

    out_dir = "/".join(pred_npz_path.split("/")[:-1])
    evaluator_object = alignment.Evaluator(out_dir)
 
    # load predictions 
    pred_dict = dict(np.load(pred_npz_path))
    
    # change keynames in gt_dict for quant evaluation
    gt_dict["joints_3d"] = torch.tensor(np.array(gt_dict["joints_3d"]))
    gt_dict["joints_2d"] = torch.tensor(np.array(gt_dict["joints_2d"]))
    

    pred_dict, gt_dict = process_gt(pred_dict, gt_dict, args.dataname)
    
    if pymafx_npz_path is not None:
        pymafx_dict = dict(np.load(pymafx_npz_path, allow_pickle=True))
        pymafx_dict, gt_dict = process_gt(pymafx_dict, gt_dict, args.dataname)
    else:
        pymafx_dict = None
        
    
    # MPJPE
    no_align = alignment.PointError(alignment_object=alignment.NoAlignment(), return_aligned=True)
    procrustes_align = alignment.PointError(alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    root_align = alignment.PointError(alignment_object=alignment.RootAlignment(), return_aligned=True)
    
    # ACCELERATION
    no_align_accel = alignment.AccelError(alignment_object=alignment.NoAlignment(), return_aligned=True)
    root_align_accel = alignment.AccelError(alignment_object=alignment.RootAlignment(), return_aligned=True)
    procrustes_align_accel = alignment.AccelError(alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    
    # F-SCORE
    f_thresholds = np.array([5/1000, 15/1000])
    root_align_f = alignment.FScores(thresholds=f_thresholds, alignment_object=alignment.RootAlignment(), return_aligned=True)
    procrustes_align_f = alignment.FScores(thresholds=f_thresholds, alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    
    # scale alignment and procrustes alignment are the same. Only input shapes are different. SO, there is no need to use. 
    align_dict_3d = {"no_align": no_align, "root_align": root_align, "procrustes_align": procrustes_align}  
    align_dict_fscore = {"root_align": root_align_f, "procrustes_align": procrustes_align_f}  
    align_dict_2d = {"no_align": no_align}

    align_dict_accel_score = {"no_align": no_align_accel, "root_align": root_align_accel, "procrustes_align": procrustes_align_accel}
 
    metrics = {"mpjpe_3d": align_dict_3d, "mpjpe_2d": align_dict_2d, "acc_err": align_dict_accel_score, "f_score": align_dict_fscore}

    save_quantitative_evaluation(evaluator_object, pred_dict, pymafx_dict=pymafx_dict, gt_dict=gt_dict, metrics=metrics, viz_flag=viz_flag, misc=misc)
    
    return  


def latent_optimization(hand_model, target, T=None, z_l=None, z_g=None, pose=None):

    if T is None:
        T = torch.arange(args.data.clip_length)
 
    cam_R = target['cam_R'].clone()
    cam_t = target['cam_t'].clone()
    cam_f = target['cam_f'].clone()
    cam_center = target['cam_center'].clone()
    is_right = target['is_right'].clone()
  
    optim_trans = target['trans'].clone()
    optim_root_orient = target["root_orient"].clone()
    
    # mp_bbox_conf = target["mediapipe_bbox_conf"]
    vis_mask = torch.tensor(target["vis_mask"])

    B, seqlen, _ = optim_trans.shape
    optim_trans.requires_grad = True 
    optim_root_orient.requires_grad = True
    
    if not pose is None:
        optim_pose = pose.clone()
        optim_pose.requires_grad = False
    else:
        optim_pose = pose

    z_global = torch.zeros_like(z_g).to(z_g)
    z_global.requires_grad = False
    
    init_z_l = z_l.clone().detach()
    init_z_l.requires_grad = False

    full_cam_R = cam_R
    full_cam_t = cam_t
    
    # take pymafx mean as starting point
    mean_betas = target["betas"].mean(dim=1).mean(dim=0)

    if args.opt_betas:
        betas = Variable(mean_betas.clone().unsqueeze(0).repeat_interleave(args.nsubject, 0), requires_grad=True)
        logger.info(f'Optimizing betas: {betas}')
    else:
        betas = mean_betas.clone().unsqueeze(0).repeat_interleave(args.nsubject, 0)
        betas.requires_grad = False
    
    # print(target["betas"].shape)
    # print('args.opt_betas: ', args.opt_betas, betas.shape, target["betas"].shape, mean_betas.shape)
    stg_configs = [args.stg1]

    if hasattr(args, 'stg2'):
        stg_configs.append(args.stg2)
    
    if hasattr(args, 'stg3'):
        stg_configs.append(args.stg3)
    
    if hasattr(args, 'stg4'):
        stg_configs.append(args.stg4)
        
    if hasattr(args, 'stg5'):
        stg_configs.append(args.stg5)

    stg_int_results = (full_cam_R, optim_trans, optim_root_orient, z_l, z_global, betas, 
                       target, B, seqlen, mean_betas, T, full_cam_R, full_cam_t)
    
    joblib.dump(stg_int_results, f'{args.pkl_output_dir}/stg_0.pkl')
    logger.info(f'Saved intermediate results to {args.pkl_output_dir}/stg_0.pkl')
            
    is_nan_loss = False
    iter = 0
    
    # iterate over different optimization steps 
    while iter < len(stg_configs):
        stg_conf = stg_configs[iter]
        logger.info(f'Stage {iter+1}: Learning rate: {stg_conf.lr}')
        stg_id = iter
        
        # break is better than continue here
        if stg_conf.niters == 0 and iter!=0:
            break    
        # this corresponds to encode-decode stage, we need to calculate joints2d, joints3d etc. 
        elif stg_conf.niters == 0 and iter == 0:
            logger.info('Encode-Decode case')
            # cannot plot loss this case 
            args.plot_loss = False
            break 
        
        if is_nan_loss:
            # will give error here
            prev_stg_results = joblib.load(f'{args.pkl_output_dir}/stg_{stg_id}.pkl')       
            _, optim_trans, optim_root_orient, z_l, z_global, betas, target, B, seqlen, mean_betas, T, \
            full_cam_R, full_cam_t = prev_stg_results
 
        stg_results = optim_step(hand_model, stg_conf, stg_id, z_l, z_global, betas, target,
                                 B, seqlen, optim_trans, optim_root_orient, init_z_l, mean_betas,
                                 T, full_cam_R, full_cam_t, cam_f, cam_center, is_right, vis_mask=vis_mask.to("cuda"), pose=optim_pose)

        if isinstance(stg_results, int):
            is_nan_loss = True
            logger.error(f'[Stage {stg_id+1}] NaN loss detected, restarting stage {stg_id+1}')
            logger.warning(f'Decreasing learning rate by 0.5 for the current stage')
            stg_configs[stg_id].lr *= 0.5
          
        else:
            z_l, z_global, optim_cam_R, optim_cam_t, optim_trans, optim_root_orient, betas, optim_pose = stg_results
            
            full_cam_R = optim_cam_R # matrix_to_rotation_6d(optim_cam_R.detach())
            full_cam_t = optim_cam_t.detach()
            
            stg_int_results = (full_cam_R, optim_trans, optim_root_orient,
                               z_l, z_global, betas, target, B, seqlen,
                               mean_betas, T, full_cam_R, full_cam_t)
            
            joblib.dump(stg_int_results, f'{args.pkl_output_dir}/stg_{stg_id+1}.pkl')
            logger.info(f'Saved intermediate results to {args.pkl_output_dir}/stg_{stg_id+1}.pkl')
            
            iter += 1

    if args.plot_loss:
        plot_list = []
        
        for num in range(iter):  
            loss_i = joblib.load(f'{args.pkl_output_dir}/stage_{num}_loss.pkl')
            plt.figure()
            
            for k, v in loss_i.items():
                if not v == []:
                    plt.plot(v, label=k)
            plt.legend()
            plt.title(f'Stage {num}')
            plt.savefig(f'{args.pkl_output_dir}/stage_{num}_loss.jpg')
        
            # concatenate all the losses
            plot_list.append(cv2.imread(f'{args.pkl_output_dir}/stage_0_loss.jpg'))
        
        plt_concat = np.concatenate(plot_list, axis=0)
        cv2.imwrite(f'{args.pkl_output_dir}/all_stages_loss.jpg', plt_concat)

    return z_l, z_global, betas, optim_root_orient, optim_trans, full_cam_R, full_cam_t, cam_f, cam_center, optim_pose


def optim_step_new(hand_model, stg_conf, stg_id, z_l, z_g, betas, target, B,
               seqlen, trans, root_orient, init_z_l, mean_betas, T, cam_R, cam_t, cam_f, cam_center, is_right, vis_mask=None, pose=None):
    def closure():
        optimizer.zero_grad()

        if motion_prior_type == "hmp":
            output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

            for k, v in output.items():
                if torch.isnan(v).any():
                    logger.warning(f'{k} in output is NaN, skipping this stage...')
                    return 0
      
        # instead of latent code, work with pose, it is in global coordinates  
        else:
            raise ValueError
            output = {"rotmat": axis_angle_to_matrix(pose)}       

        output['betas'] = betas[:, None, None, :].repeat(1, B, seqlen, 1)
        global_trans = trans.clone().reshape(args.nsubject, B, seqlen, 3) 
        global_trans = global_trans.reshape(B*args.nsubject, seqlen, 3)
        
        output['trans'] = global_trans
        output['root_orient'] = root_orient

        local_rotmat = fk.global_to_local(output['rotmat'].view(-1, HAND_JOINT_NUM, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(B*args.nsubject, -1, HAND_JOINT_NUM, 3, 3) # (B x T, J, 3, 3)
        body_pose = matrix_to_axis_angle(local_rotmat)[:, :, 1:]

        rh_mano_out = pred_mano(hand_model, output['trans'], matrix_to_axis_angle(rotation_6d_to_matrix(root_orient)), body_pose.reshape(1, 128, 45), is_right, betas, only_right=False) 

        joints3d = rh_mano_out['joints3d'].view(B, seqlen, -1, 3)
        vertices3d = rh_mano_out['verts3d'].view(B, seqlen, -1, 3)

        # print(joints3d.shape, cam_R.shape)
        joints2d_pred = reproject(
                joints3d, cam_R, cam_t, cam_f, cam_center
            )
        output['joints2d'] = joints2d_pred
        output['joints3d'] = joints3d

        ############################
        # 2d keypoints debugging.
        # openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # gt_indices = openpose_indices
        # num_hand = 1
        # num_frames = joints2d_pred.shape[1]
        # print(target['joints2d'].shape)
        # for x in range(num_frames):

        # raise ValueError
        ###########################

        _bbox_conf_ = None
        # _bbox_conf_[_bbox_conf_<0.6] = 0.0
        
        local_rotmat_gt = target['rotmat'] 
        loss_dict = {}

        if stg_conf.lambda_rot > 0:    
            mano_joint_conf = torch.zeros_like(target['rotmat'][..., :1, 0])
            
            for si in range(16):
                op_conf = [target['joints2d'][:, :, si, 2]]
                max_conf = torch.stack(op_conf, dim=0).max(0).values
                mano_joint_conf[:, :, si] = max_conf.unsqueeze(-1)
            
            # use bbox conf if that is the case 
            if _bbox_conf_ is not None:
                bbox_coef = torch.repeat_interleave(_bbox_conf_[..., None], dim=2, repeats=15) 
                
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=bbox_coef)
            else:
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=mano_joint_conf[:, :, 1:])
            
            loss_dict['rot'] = stg_conf.lambda_rot * rot_loss

        if stg_conf.lambda_bio > 0:
            assert len(output['joints3d']) == 1
            joints = output['joints3d'][0]  # (128,21,3)
            loss_total, _ = bmc.compute_loss(joints)
            # print("loss_total=", loss_total)
            # print("loss_dict=", loss_dict)
            loss_dict['bio'] = stg_conf.lambda_bio * loss_total

        if stg_conf.lambda_consistency > 0:
            cur_loss = pose_prior_loss(body_pose)
            loss_dict['pose_prior'] = stg_conf.lambda_reproj * cur_loss

        if stg_conf.lambda_reproj > 0:
            reproj_loss = joints2d_loss(joints2d_obs=target['joints2d'], joints2d_pred=joints2d_pred, bbox_conf=_bbox_conf_) 
            loss_dict['reproj'] = stg_conf.lambda_reproj * reproj_loss
            
        if stg_conf.lambda_orient > 0:
            orient_loss = L_orient(output['root_orient'], target['root_orient'], T, bbox_conf=_bbox_conf_)
            loss_dict['orient'] = stg_conf.lambda_orient * orient_loss     
        
        if stg_conf.lambda_trans > 0:
            trans_loss = L_trans(output['trans'], target['trans'], T, bbox_conf=_bbox_conf_)
            loss_dict['trans'] = stg_conf.lambda_trans * trans_loss

        if stg_conf.lambda_rot_smooth > 0:
            rot_smooth_l = rot_smooth_loss(local_rotmat)
            loss_dict['rot_sm'] = stg_conf.lambda_rot_smooth * rot_smooth_l   

        if stg_conf.lambda_orient_smooth > 0: 
            matrot_root_orient = rotation_6d_to_matrix(root_orient)            
            orient_smooth_l = rot_smooth_loss(matrot_root_orient)
            loss_dict['orient_sm'] = stg_conf.lambda_orient_smooth * orient_smooth_l
                
        # Smoothness objectives
        if stg_conf.lambda_j3d_smooth > 0:
            joints3d = output['joints3d']

            j3d_smooth_l = pos_smooth_loss(joints3d)
            loss_dict['j3d_sm'] = stg_conf.lambda_j3d_smooth * j3d_smooth_l   
        
        if stg_conf.lambda_trans_smooth > 0:
            # tr = mask_data(output['trans'], mask)
            tr = output['trans']
            tr = tr.reshape(args.nsubject, B, seqlen, 3)
            trans_smooth_l = 0
            for sid in range(args.nsubject):
                trans_smooth_l += pos_smooth_loss(tr[sid])
            loss_dict['trans_sm'] = stg_conf.lambda_trans_smooth * trans_smooth_l
        
        if stg_conf.lambda_motion_prior > 0:
            
            if motion_prior_type == "pca":
                mp_local_loss = L_PCA(pose)
            elif motion_prior_type == "gmm":
                mp_local_loss = L_GMM(pose)                 
            else:    
                mp_local_loss = motion_prior_loss(z_l)
            loss_dict['mot_prior'] = stg_conf.lambda_motion_prior * mp_local_loss
            
            
        if stg_conf.lambda_init_z_prior > 0:
            zl_init_prior_l = F.mse_loss(z_l, init_z_l)
            loss_dict['init_z_prior'] = stg_conf.lambda_init_z_prior * (zl_init_prior_l)

        if stg_conf.lambda_pose_prior > 0 and opt.HMP.use_hposer:
            raise NotImplementedError
            # loss_dict['pose_prior'] = stg_conf.lambda_pose_prior * L_pose_prior(output)
        
        if hasattr(stg_conf, 'lambda_batch_cs'):
            if stg_conf.lambda_batch_cs > 0:
                if args.overlap_len == 0:
                    logger.warning('Batch consistency won\'t be effective since overlap_len is 0')
                if B > 1:
                    # joints3d = mask_data(output['joints3d'], mask)
                    joints3d = joints3d.reshape(args.nsubject, B, seqlen, -1, 3)
                    batch_cs_l = 0
                    for sid in range(args.nsubject):
                        batch_cs_l += L_pos(joints3d[sid, :-1, -args.overlap_len:], joints3d[sid, 1:, :args.overlap_len], T)
                    loss_dict['batch_cs'] = stg_conf.lambda_batch_cs * batch_cs_l
                else:
                    if i < 5:
                        logger.warning('Batch consistency won\'t be effective since batch size is 1')
                
        if hasattr(stg_conf, 'betas_prior'):
            if stg_conf.betas_prior > 0:
                if betas is None:
                    logger.error('Cannot compute betas prior since args.opt_betas is False')
                betas_prior_l = torch.pow(betas - mean_betas, 2).mean()
                loss_dict['betas_prior'] = stg_conf.betas_prior * betas_prior_l
            

        loss = sum(loss_dict.values())
        loss_dict['loss'] = loss
        
        # copy loss values to loss_dict_by_step
        for k, v in loss_dict.items():
            loss_dict_by_step[k].append(v.detach().item()) 

        if not torch.isnan(loss):
            loss.backward()
            return loss
        else:
            logger.warning('Loss is NaN, skipping this stage')
            raise ValueError

        print(stg_id, i, stg_conf.niters)
        loss_log_str = f'Stage {stg_id+1} [{i:03d}/{stg_conf.niters}]'
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
            loss_log_str += f'{k}: {v.item():.3f}\t'
        logger.info(loss_log_str)

    logger.info(f'Running optimization stage {stg_id+1} ...')
    opt_params = []
    for param in stg_conf.opt_params:
        if param == 'root_orient':
            opt_params.append(root_orient)
        elif param == 'trans':
            opt_params.append(trans)
        elif param == 'z_l':
            opt_params.append(z_l)
        elif param == 'pose':
            opt_params.append(pose)
        elif param == 'betas':
            if betas is None:
                logger.error('Cannot optimize betas if args.opt_betas is False')
            opt_params.append(betas)
        else:
            raise ValueError(f'Unknown parameter {param}')
    
    for param in opt_params:
        param.requires_grad = True
        
    # optimizer = torch.optim.Adam(opt_params, lr=stg_conf.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)
    LINE_SEARCH = "strong_wolfe"
    lr = 1.0
    lbfgs_max_iter = 20
    optimizer = torch.optim.LBFGS(
            opt_params, max_iter=lbfgs_max_iter, lr=lr, line_search_fn=LINE_SEARCH
    )
    
    def mask_data(data, mask):
        ml = len(mask.shape)
        dl = len(data.shape)
        for _ in range(dl-ml):
            mask = mask[..., None]
        return data * mask
      
    loss_dict_by_step = {"rot": [], "reproj": [], "rot_sm": [], "orient_sm": [], 'betas_prior': [], "j3d_sm": [], "pose_prior": [],
                    "trans_sm": [], "mot_prior": [], "init_z_prior": [], "orient": [], "trans": [], "loss": [], "bio": []} 
    
    # optimize the z_l and root_orient, pos, trans, 2d kp objectives
    start_time = time.time()
    print('start latent optimization...')
    for i in range(stg_conf.niters):
        optimizer.step(closure)
        
    end_time = time.time()
    
    # save the loss dict. 
    joblib.dump(loss_dict_by_step, open(os.path.join(args.pkl_output_dir, f'stage_{stg_id}_loss.pkl'), 'wb'))
    
    print(f'Stage {stg_id+1} finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    
    if not betas is None:
        logger.info(f'mean_betas: {mean_betas.detach().cpu().numpy()}')
        logger.info(f'betas: {betas.detach().cpu().numpy()}')
    
    return z_l, z_g, cam_R, cam_t, trans, root_orient, betas, pose

def optim_step(hand_model, stg_conf, stg_id, z_l, z_g, betas, target, B,
               seqlen, trans, root_orient, init_z_l, mean_betas, T, cam_R, cam_t, cam_f, cam_center, is_right, vis_mask=None, pose=None):

    logger.info(f'Running optimization stage {stg_id+1} ...')
    opt_params = []
    for param in stg_conf.opt_params:
        if param == 'root_orient':
            opt_params.append(root_orient)
        elif param == 'trans':
            opt_params.append(trans)
        elif param == 'z_l':
            opt_params.append(z_l)
        elif param == 'pose':
            opt_params.append(pose)
        elif param == 'betas':
            if betas is None:
                logger.error('Cannot optimize betas if args.opt_betas is False')
            opt_params.append(betas)
        else:
            raise ValueError(f'Unknown parameter {param}')
    
    for param in opt_params:
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(opt_params, lr=stg_conf.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)

    def mask_data(data, mask):
        ml = len(mask.shape)
        dl = len(data.shape)
        for _ in range(dl-ml):
            mask = mask[..., None]
        return data * mask
      
    loss_dict_by_step = {"rot": [], "reproj": [], "rot_sm": [], "orient_sm": [], 'betas_prior': [], "j3d_sm": [], "pose_prior": [],
                    "trans_sm": [], "mot_prior": [], "init_z_prior": [], "orient": [], "trans": [], "loss": [], "bio": []} 
    
    # optimize the z_l and root_orient, pos, trans, 2d kp objectives
    start_time = time.time()
    print(f'start latent optimization... {stg_conf.niters} iters')
    for i in range(stg_conf.niters):
        optimizer.zero_grad()

        if motion_prior_type == "hmp":
            output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

            for k, v in output.items():
                if torch.isnan(v).any():
                    logger.warning(f'{k} in output is NaN, skipping this stage...')
                    return 0
      
        # instead of latent code, work with pose, it is in global coordinates  
        else:
            raise ValueError
            output = {"rotmat": axis_angle_to_matrix(pose)}       

        output['betas'] = betas[:, None, None, :].repeat(1, B, seqlen, 1)
        global_trans = trans.clone().reshape(args.nsubject, B, seqlen, 3) 
        global_trans = global_trans.reshape(B*args.nsubject, seqlen, 3)
        
        output['trans'] = global_trans
        output['root_orient'] = root_orient

        local_rotmat = fk.global_to_local(output['rotmat'].view(-1, HAND_JOINT_NUM, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(B*args.nsubject, -1, HAND_JOINT_NUM, 3, 3) # (B x T, J, 3, 3)
        body_pose = matrix_to_axis_angle(local_rotmat)[:, :, 1:]

        rh_mano_out = pred_mano(hand_model, output['trans'], matrix_to_axis_angle(rotation_6d_to_matrix(root_orient)), body_pose.reshape(1, 128, 45), is_right, betas, only_right=False) 

        joints3d = rh_mano_out['joints3d'].view(B, seqlen, -1, 3)
        vertices3d = rh_mano_out['verts3d'].view(B, seqlen, -1, 3)

        # print(joints3d.shape, cam_R.shape)
        joints2d_pred = reproject(
                joints3d, cam_R, cam_t, cam_f, cam_center
            )
        output['joints2d'] = joints2d_pred
        output['joints3d'] = joints3d

        ############################
        # 2d keypoints debugging.
        # openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # gt_indices = openpose_indices
        # num_hand = 1
        # num_frames = joints2d_pred.shape[1]
        # print(target['joints2d'].shape)
        # for x in range(num_frames):

        # raise ValueError
        ###########################

        _bbox_conf_ = None
        # _bbox_conf_[_bbox_conf_<0.6] = 0.0
        
        local_rotmat_gt = target['rotmat'] 
        loss_dict = {}

        if stg_conf.lambda_rot > 0:    
            mano_joint_conf = torch.zeros_like(target['rotmat'][..., :1, 0])
            
            for si in range(16):
                op_conf = [target['joints2d'][:, :, si, 2]]
                max_conf = torch.stack(op_conf, dim=0).max(0).values
                mano_joint_conf[:, :, si] = max_conf.unsqueeze(-1)
            
            # use bbox conf if that is the case 
            if _bbox_conf_ is not None:
                bbox_coef = torch.repeat_interleave(_bbox_conf_[..., None], dim=2, repeats=15) 
                
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=bbox_coef)
            else:
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=mano_joint_conf[:, :, 1:])
            
            loss_dict['rot'] = stg_conf.lambda_rot * rot_loss

        if stg_conf.lambda_bio > 0:
            assert len(output['joints3d']) == 1
            joints = output['joints3d'][0]  # (128,21,3)
            loss_total, _ = bmc.compute_loss(joints)
            # print("loss_total=", loss_total)
            # print("loss_dict=", loss_dict)
            loss_dict['bio'] = stg_conf.lambda_bio * loss_total

        if stg_conf.lambda_consistency > 0:
            cur_loss = pose_prior_loss(body_pose)
            loss_dict['pose_prior'] = stg_conf.lambda_reproj * cur_loss

        if stg_conf.lambda_reproj > 0:
            reproj_loss = joints2d_loss(joints2d_obs=target['joints2d'], joints2d_pred=joints2d_pred, bbox_conf=_bbox_conf_) 
            loss_dict['reproj'] = stg_conf.lambda_reproj * reproj_loss
            
        if stg_conf.lambda_orient > 0:
            orient_loss = L_orient(output['root_orient'], target['root_orient'], T, bbox_conf=_bbox_conf_)
            loss_dict['orient'] = stg_conf.lambda_orient * orient_loss     
        
        if stg_conf.lambda_trans > 0:
            trans_loss = L_trans(output['trans'], target['trans'], T, bbox_conf=_bbox_conf_)
            loss_dict['trans'] = stg_conf.lambda_trans * trans_loss

        if stg_conf.lambda_rot_smooth > 0:
            rot_smooth_l = rot_smooth_loss(local_rotmat)
            loss_dict['rot_sm'] = stg_conf.lambda_rot_smooth * rot_smooth_l   

        if stg_conf.lambda_orient_smooth > 0: 
            matrot_root_orient = rotation_6d_to_matrix(root_orient)            
            orient_smooth_l = rot_smooth_loss(matrot_root_orient)
            loss_dict['orient_sm'] = stg_conf.lambda_orient_smooth * orient_smooth_l
                
        # Smoothness objectives
        if stg_conf.lambda_j3d_smooth > 0:
            joints3d = output['joints3d']

            j3d_smooth_l = pos_smooth_loss(joints3d)
            loss_dict['j3d_sm'] = stg_conf.lambda_j3d_smooth * j3d_smooth_l   
        
        if stg_conf.lambda_trans_smooth > 0:
            # tr = mask_data(output['trans'], mask)
            tr = output['trans']
            tr = tr.reshape(args.nsubject, B, seqlen, 3)
            trans_smooth_l = 0
            for sid in range(args.nsubject):
                trans_smooth_l += pos_smooth_loss(tr[sid])
            loss_dict['trans_sm'] = stg_conf.lambda_trans_smooth * trans_smooth_l
        
        if stg_conf.lambda_motion_prior > 0:
            
            if motion_prior_type == "pca":
                mp_local_loss = L_PCA(pose)
            elif motion_prior_type == "gmm":
                mp_local_loss = L_GMM(pose)                 
            else:    
                mp_local_loss = motion_prior_loss(z_l)
            loss_dict['mot_prior'] = stg_conf.lambda_motion_prior * mp_local_loss
            
            
        if stg_conf.lambda_init_z_prior > 0:
            zl_init_prior_l = F.mse_loss(z_l, init_z_l)
            loss_dict['init_z_prior'] = stg_conf.lambda_init_z_prior * (zl_init_prior_l)

        if stg_conf.lambda_pose_prior > 0 and opt.HMP.use_hposer:
            raise NotImplementedError
            # loss_dict['pose_prior'] = stg_conf.lambda_pose_prior * L_pose_prior(output)
        
        if hasattr(stg_conf, 'lambda_batch_cs'):
            if stg_conf.lambda_batch_cs > 0:
                if args.overlap_len == 0:
                    logger.warning('Batch consistency won\'t be effective since overlap_len is 0')
                if B > 1:
                    # joints3d = mask_data(output['joints3d'], mask)
                    joints3d = joints3d.reshape(args.nsubject, B, seqlen, -1, 3)
                    batch_cs_l = 0
                    for sid in range(args.nsubject):
                        batch_cs_l += L_pos(joints3d[sid, :-1, -args.overlap_len:], joints3d[sid, 1:, :args.overlap_len], T)
                    loss_dict['batch_cs'] = stg_conf.lambda_batch_cs * batch_cs_l
                else:
                    if i < 5:
                        logger.warning('Batch consistency won\'t be effective since batch size is 1')
                
        if hasattr(stg_conf, 'betas_prior'):
            if stg_conf.betas_prior > 0:
                if betas is None:
                    logger.error('Cannot compute betas prior since args.opt_betas is False')
                betas_prior_l = torch.pow(betas - mean_betas, 2).mean()
                loss_dict['betas_prior'] = stg_conf.betas_prior * betas_prior_l
            

        loss = sum(loss_dict.values())
        loss_dict['loss'] = loss
        
        # copy loss values to loss_dict_by_step
        for k, v in loss_dict.items():
            loss_dict_by_step[k].append(v.detach().item()) 

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 5.0)
            optimizer.step()
        else:
            logger.warning('Loss is NaN, skipping this stage')
            return 0

        scheduler.step()
        loss_log_str = f'Stage {stg_id+1} [{i:03d}/{stg_conf.niters}]'
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
            loss_log_str += f'{k}: {v.item():.3f}\t'
        logger.info(loss_log_str)
        
    end_time = time.time()
    
    # save the loss dict. 
    joblib.dump(loss_dict_by_step, open(os.path.join(args.pkl_output_dir, f'stage_{stg_id}_loss.pkl'), 'wb'))
    
    print(f'Stage {stg_id+1} finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    
    if not betas is None:
        logger.info(f'mean_betas: {mean_betas.detach().cpu().numpy()}')
        logger.info(f'betas: {betas.detach().cpu().numpy()}')
    
    return z_l, z_g, cam_R, cam_t, trans, root_orient, betas, pose

def fitting_prior(obs_data, res_dict, hand_model, opt, data_args, out_dir, device):
    global model, fk, ngpu, hposer, motion_prior_type, pca_aa, gmm_aa, args
    args = Arguments(opt.paths.base_dir, os.path.dirname(__file__), filename=opt.HMP.config)

    opt.HMP.exp_name = data_args.seq
    
    cfg_name = opt.HMP.config.split(".")[0]
    opt.HMP.use_hposer = False
    
    args.save_path = out_dir
    args.plot_loss = True

    args.vid_path = opt.HMP.vid_path

    args.root = opt.paths.base_dir
    args.dataset_dir = os.path.join(opt.paths.base_dir, '_DATA/hmp_model')
    args.save_dir = os.path.join(opt.paths.base_dir, '_DATA/hmp_model')

    init_method = args.init_method if hasattr(args, 'init_method') else "pymafx"
    assert init_method in ["metro", "pymafx"]
    
    if hasattr(args, 'motion_prior_type'):
        raise ValueError
    else:
        motion_prior_type = "hmp"

    assert motion_prior_type in ["hmp"]

    # load hposer 
    if opt.HMP.use_hposer:
        raise ValueError
        # hposer, _ = load_hposer()
        # hposer.to("cuda")
        # hposer.eval()
    else:
        hposer = None

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    ngpu = 1

    model = Architecture(args, ngpu)
    model.load(optimal=True)
    model.eval()

    fk = ForwardKinematicsLayer(args)
    multi_stage_opt(opt, device, obs_data, res_dict, hand_model, os.path.join(os.path.dirname(__file__), opt.HMP.config), opt.HMP.exp_name, init_method)


def run_prior(
    cfg,
    dataset,
    out_dir,
    device,
    phases, 
    obs_data, hand_model, opt, data_args, prior_out, 
    save_dir=None
):

    def load_result(res_path_dict):
        def to_torch(obj):
            if isinstance(obj, np.ndarray):
                return torch.from_numpy(obj).float()
            if isinstance(obj, dict):
                return {k: to_torch(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_torch(x) for x in obj]
            return obj
        """
        load all saved results for a given iteration
        :param res_path_dict (dict) paths to relevant results
        returns dict of results
        """
        res_dict = {}
        for name, path in res_path_dict.items():
            res = np.load(path)
            res_dict[name] = to_torch({k: res[k] for k in res.files})
        return res_dict

    def get_results_paths(res_dir):
        """
        get the iterations of all saved results in res_dir
        :param res_dir (str) result dir
        returns a dict of iter to result path
        """
        res_files = sorted(glob.glob(f"{res_dir}/*_results.npz"))
        print(f"found {len(res_files)} results in {res_dir}")

        path_dict = {}
        for res_file in res_files:
            it, name, _ = os.path.basename(res_file).split("_")[-3:]
            assert name in ["world", "prior"]
            if it not in path_dict:
                path_dict[it] = {}
            path_dict[it][name] = res_file
        return path_dict

    save_dir = out_dir if save_dir is None else save_dir
    print("OUT_DIR", out_dir)
    print("SAVE_DIR", save_dir)
    print("VISUALIZING PHASES", phases)
    print("PRIOR OUT", prior_out)

    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        res_dir = os.path.join(out_dir, phase)
        if phase == "input":
            res = get_input_dict(dataset)
            it = f"{0:06d}"

        elif os.path.isdir(res_dir):
            res_path_dict = get_results_paths(res_dir)
            it = sorted(res_path_dict.keys())[-1]
            res = load_result(res_path_dict[it])["world"]

        else:
            print(f"{res_dir} does not exist, skipping")
            continue

        out_name = f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}"
        phase_max_iters[phase] = it

        # out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        # if not overwrite and all(os.path.exists(p) for p in out_paths):
        #     print("FOUND OUT PATHS", out_paths)
        #     continue

        phase_results[phase] = out_name, res

    if len(phase_results) > 0:
        out_names, res_dicts = zip(*phase_results.values())

    assert len(res_dicts) == 1
    assert len(res_dicts) == len(out_names)
    fitting_prior(obs_data, res_dicts, hand_model, opt, data_args, prior_out, device)
