import numpy as np
import os
import torch
from scipy.spatial import cKDTree as KDTree
from pytorch3d.loss import chamfer_distance
import sys

# sys.path = [".."] + sys.path
from tqdm import tqdm
import hold.common.metrics as metrics


def compute_bounding_box_centers(vertices):
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers


def convert_to_tensors(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint32:
                data[key] = torch.from_numpy(value.astype(np.int64))
            elif value.dtype.kind == "f":  # Check if it's a floating-point type
                data[key] = torch.from_numpy(
                    value.astype(np.float32)
                )  # Convert to Float32
            else:
                data[key] = torch.from_numpy(value)
    return data


def eval_icp_first_frame(data_pred, data_gt, metric_dict):
    faces = data_pred["faces"]["object"]
    from src.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    v3d_o_ra = Vector3dVector(data_pred["v3d_ra.object"][0].numpy())
    faces_o = Vector3iVector(faces.cpu().numpy())
    v3d_o_ra_gt = Vector3dVector(data_gt["v3d_ra.object"][0].numpy())
    faces_o_gt = Vector3iVector(data_gt["faces.object"].numpy())
    source_mesh = TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)
    
    # best_cd, best_f5, best_f10 = compute_icp_metrics(
    #     target_mesh, source_mesh, num_iters=1200
    # )
    icp_iters = int(os.environ.get("HOLD_ICP_ITERS", "600"))
    best_cd, best_f5, best_f10 = compute_icp_metrics(
        target_mesh, source_mesh, num_iters=icp_iters, is_sqrt=True
    )
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    # metric_dict["cd_icp_no_scale"] = best_cd_no_scale
    # metric_dict["f5_icp_no_scale"] = best_f5_no_scale * 100.0
    # metric_dict["f10_icp_no_scale"] = best_f10_no_scale * 100.0
    # metric_dict["scale"] = convert_to_absolute_scale(scale)
    return metric_dict


def eval_icp_every_frame(data_pred, data_gt, metric_dict):
    from src.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    is_valid = data_gt["is_valid"]

    num_frames = len(data_pred["v3d_o_ra"])
    cd_list = []
    f5_list = []
    f10_list = []
    assert num_frames == len(data_gt["v3d_o_ra"])
    assert num_frames == len(is_valid)

    for idx in tqdm(range(num_frames)):
        if is_valid[idx]:
            v3d_o_ra = Vector3dVector(data_pred["v3d_o_ra"][idx].numpy())
            faces_o = Vector3iVector(data_pred["faces_o"][idx].numpy())
            v3d_o_ra_gt = Vector3dVector(data_gt["v3d_o_ra"][idx].numpy())
            faces_o_gt = Vector3iVector(data_gt["faces_o"].numpy())
            source_mesh = TriangleMesh(v3d_o_ra, faces_o)
            target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)

            cd, f5, f10 = compute_icp_metrics(
                target_mesh, source_mesh, num_iters=10, no_tqdm=True, is_sqrt=True
            )
        else:
            cd = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        cd_list.append(cd)
        f5_list.append(f5)
        f10_list.append(f10)

    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)
    mean_cd = np.nanmean(cd_list)
    mean_f5 = np.nanmean(f5_list)
    mean_f10 = np.nanmean(f10_list)

    metric_dict["cd_icp"] = mean_cd
    metric_dict["f5_icp"] = mean_f5 * 100.0
    metric_dict["f10_icp"] = mean_f10 * 100.0
    return metric_dict


def eval_mrrpe_ho_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred = data_pred["j3d_c.right"]
    root_o_pred = data_pred["root.object"]

    j3d_h_c_gt = data_gt["j3d_c.right"]
    root_o_gt = data_gt["root.object"]
    is_valid = data_gt["is_valid"]

    if j3d_h_c_gt.shape != j3d_h_c_pred.shape:
        j3d_h_c_gt = j3d_h_c_gt[: j3d_h_c_pred.shape[0], :, :]
        root_o_gt = root_o_gt[: j3d_h_c_pred.shape[0], :]
        is_valid = is_valid[: j3d_h_c_pred.shape[0]]
    
    root_h_gt = j3d_h_c_gt[:, 0]
    root_h_pred = j3d_h_c_pred[:, 0]
    mrrpe_ho = (
        metrics.compute_mrrpe(
            root_h_gt,
            root_o_gt,
            root_h_pred,
            root_o_pred,
            is_valid,
        )
        * 1000
    )
    not_valid = (1 - is_valid).numpy().astype(bool)
    mrrpe_ho[not_valid] = np.nan

    metric_dict["mrrpe_ho"] = mrrpe_ho
    return metric_dict


def calculate_chamfer_f_scores(vertices_source, vertices_target, is_sqrt=True):
    vertices_source = vertices_source * 100
    vertices_target = vertices_target * 100

    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)
    if is_sqrt:
        gt_to_gen_chamfer = np.mean((one_distances))
    else:
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)
    if is_sqrt:
        gen_to_gt_chamfer = np.mean((two_distances))
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return array_like


def _resolve_faces(data, node_key):
    faces = None
    if "faces" in data:
        faces_container = data["faces"]
        if isinstance(faces_container, dict) and node_key in faces_container:
            faces = faces_container[node_key]
        elif hasattr(faces_container, "get"):
            faces = faces_container.get(node_key, None)
    faces_key = f"faces.{node_key}"
    if faces is None and faces_key in data:
        faces = data[faces_key]
    if faces is None:
        return None
    return _to_numpy(faces).astype(np.int64)


def _load_pointmap_data(pointmap_path):
    try:
        return torch.load(pointmap_path, map_location="cpu")
    except Exception as exc:
        print(f"[eval_modules] Failed to load pointmap data: {pointmap_path} ({exc})")
        return None


def _extract_pointmap(pointmap_data, frame_idx, pointmap_key=None):
    if pointmap_data is None:
        return None
    pointmap = None
    if torch.is_tensor(pointmap_data) or isinstance(pointmap_data, np.ndarray):
        pointmap = pointmap_data
    elif isinstance(pointmap_data, dict):
        if pointmap_key and pointmap_key in pointmap_data:
            pointmap = pointmap_data[pointmap_key]
        else:
            key_candidates = [
                "pointmap",
                "pointmaps",
                "point_map",
                "point_maps",
                "points",
                "pointcloud",
                "point_cloud",
                "xyz",
                "pts",
                "pc",
            ]
            for key in key_candidates:
                if key in pointmap_data:
                    pointmap = pointmap_data[key]
                    break
        if pointmap is None and len(pointmap_data) == 1:
            pointmap = next(iter(pointmap_data.values()))
    elif isinstance(pointmap_data, (list, tuple)) and len(pointmap_data) > 0:
        pointmap = pointmap_data[0]

    if pointmap is None:
        print("[eval_modules] Could not find pointmap tensor in pointmap data.")
        return None

    if torch.is_tensor(pointmap):
        pointmap = pointmap.detach().cpu()

    if pointmap.ndim == 4:
        if frame_idx >= pointmap.shape[0]:
            print(
                f"[eval_modules] Frame {frame_idx} out of range for pointmap shape {tuple(pointmap.shape)}"
            )
            return None
        pointmap = pointmap[frame_idx]
    return pointmap


def _pointmap_to_xyz(pointmap):
    if pointmap is None:
        return None
    pointmap_np = _to_numpy(pointmap)
    if pointmap_np.ndim == 3 and pointmap_np.shape[-1] == 3:
        xyz = pointmap_np.reshape(-1, 3)
    elif pointmap_np.ndim == 2 and pointmap_np.shape[-1] == 3:
        xyz = pointmap_np
    else:
        print(
            f"[eval_modules] Unsupported pointmap shape for export: {tuple(pointmap_np.shape)}"
        )
        return None
    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask]


def save_eval_alignment_meshes(
    data_pred,
    data_gt,
    frame_idx,
    v3d_pred_obj,
    v3d_gt_obj,
    rand_pred_idx,
    rand_gt_idx,
    save_dir,
    prefix="",
    pointmap_xyz=None,
):
    import open3d as o3d

    os.makedirs(save_dir, exist_ok=True)

    pred_obj = _to_numpy(v3d_pred_obj)
    gt_obj = _to_numpy(v3d_gt_obj)

    faces_obj_pred = _resolve_faces(data_pred, "object")
    faces_obj_gt = _resolve_faces(data_gt, "object")
    if faces_obj_pred is None and faces_obj_gt is not None:
        faces_obj_pred = faces_obj_gt
    if faces_obj_gt is None and faces_obj_pred is not None:
        faces_obj_gt = faces_obj_pred

    tag = f"{prefix}frame_{frame_idx:04d}"
    if faces_obj_pred is not None:
        mesh_pred = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pred_obj),
            o3d.utility.Vector3iVector(faces_obj_pred),
        )
        o3d.io.write_triangle_mesh(
            os.path.join(save_dir, f"{tag}_pred_object.ply"), mesh_pred
        )
    if faces_obj_gt is not None:
        mesh_gt = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(gt_obj),
            o3d.utility.Vector3iVector(faces_obj_gt),
        )
        o3d.io.write_triangle_mesh(
            os.path.join(save_dir, f"{tag}_gt_object.ply"), mesh_gt
        )

    pred_pc = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pred_obj[rand_pred_idx])
    )
    gt_pc = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(gt_obj[rand_gt_idx])
    )
    o3d.io.write_point_cloud(
        os.path.join(save_dir, f"{tag}_pred_object_sample.ply"), pred_pc
    )
    o3d.io.write_point_cloud(
        os.path.join(save_dir, f"{tag}_gt_object_sample.ply"), gt_pc
    )
    if pointmap_xyz is not None and len(pointmap_xyz) > 0:
        pm_pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pointmap_xyz)
        )
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"{tag}_pred_pointmap.ply"), pm_pc
        )

    if "v3d_c.right" in data_pred and "root.right" in data_pred:
        pred_hand = _to_numpy(
            data_pred["v3d_c.right"][frame_idx] - data_pred["root.right"][frame_idx][None, :]
        )
        faces_hand_pred = _resolve_faces(data_pred, "right")
        if faces_hand_pred is not None:
            mesh_pred_h = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(pred_hand),
                o3d.utility.Vector3iVector(faces_hand_pred),
            )
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"{tag}_pred_hand.ply"), mesh_pred_h
            )

    if "v3d_c.right" in data_gt and "root.right" in data_gt:
        gt_hand = _to_numpy(
            data_gt["v3d_c.right"][frame_idx] - data_gt["root.right"][frame_idx][None, :]
        )
        faces_hand_gt = _resolve_faces(data_gt, "right")
        if faces_hand_gt is not None:
            mesh_gt_h = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(gt_hand),
                o3d.utility.Vector3iVector(faces_hand_gt),
            )
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"{tag}_gt_hand.ply"), mesh_gt_h
            )


def compute_iou_per_frame(insta_map_pred, insta_map_gt):
    classes = [0, 100, 200]
    ious = []

    for frame_idx in range(insta_map_pred.shape[0]):
        iou_per_class = []
        for cls in classes:
            pred_mask = insta_map_pred[frame_idx] == cls
            gt_mask = insta_map_gt[frame_idx] == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0
            iou_per_class.append(iou)
        ious.append(
            np.mean(iou_per_class)
        )  # Assuming you want the mean IoU for all classes per frame

    return np.array(ious)


def eval_cd_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c_ra"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c_ra"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_ra = (
        chamfer_distance(
            v3d_o_c_pred_ra[:, rand_pred_idx],
            v3d_o_c_gt_ra[:, rand_gt_idx],
            batch_reduction=None,
        )[0]
        * 1000
    )

    metric_dict["cd_ra"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    return metric_dict


def eval_cd_f(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(v3d_o_c_pred_ra.shape[0]):
        cd_error, f5, f10 = calculate_chamfer_f_scores(
            v3d_o_c_pred_ra[idx, rand_pred_idx].numpy(),
            v3d_o_c_gt_ra[idx, rand_gt_idx].numpy(),
        )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd"] = cd_list
    metric_dict["f5"] = f5_list * 100.0
    metric_dict["f10"] = f10_list * 100.0
    return metric_dict


def _compute_scale_from_origin(v3d_pred, v3d_gt, eps=1e-8):
    pred_sq_norm = np.sum(np.square(v3d_pred))
    gt_sq_norm = np.sum(np.square(v3d_gt))
    if pred_sq_norm < eps or gt_sq_norm < eps:
        return 1.0
    return np.sqrt(gt_sq_norm / pred_sq_norm)


def eval_cd_f_right_scale_invariant(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_right.object"]
    v3d_o_c_gt_ra = data_gt["v3d_right.object"]
    is_valid = data_gt["is_valid"]

    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]

    t_pred = j3d_h_c_pred_ra.shape[0]
    if j3d_h_c_gt_ra.shape[0] != t_pred:
        j3d_h_c_gt_ra = j3d_h_c_gt_ra[:t_pred]
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[:t_pred]
        is_valid = is_valid[:t_pred]
    if v3d_o_c_pred_ra.shape[0] != t_pred:
        v3d_o_c_pred_ra = v3d_o_c_pred_ra[:t_pred]

    if v3d_o_c_gt_ra.shape != v3d_o_c_pred_ra.shape:
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[: len(v3d_o_c_pred_ra), :, :]
        is_valid = is_valid[: len(v3d_o_c_pred_ra)]

    cd_list = []
    f5_list = []
    f10_list = []
    scale_list = []

    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]
        v3d_gt = v3d_o_c_gt_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
            scale = float("nan")
        else:
            v3d_pred_np = v3d_pred.detach().cpu().numpy()
            v3d_gt_np = v3d_gt.detach().cpu().numpy()
            scale = _compute_scale_from_origin(v3d_pred_np, v3d_gt_np)
            v3d_pred_scaled = v3d_pred_np * scale

            num_pts_pred = min(3000, v3d_pred_scaled.shape[0])
            num_pts_gt = min(3000, v3d_gt_np.shape[0])
            rand_pred_idx = torch.randperm(v3d_pred_scaled.shape[0])[:num_pts_pred]
            rand_gt_idx = torch.randperm(v3d_gt_np.shape[0])[:num_pts_gt]

            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred_scaled[rand_pred_idx.numpy()],
                v3d_gt_np[rand_gt_idx.numpy()],
            )

        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
        scale_list.append(scale)

    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)
    scale_list = np.array(scale_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan
    scale_list[not_valid] = np.nan

    metric_dict["cd_right_si"] = cd_list
    metric_dict["f5_right_si"] = f5_list * 100.0
    metric_dict["f10_right_si"] = f10_list * 100.0
    metric_dict["right_obj_scale_factor"] = scale_list
    return metric_dict


def eval_cd_f_right(
    data_pred,
    data_gt,
    metric_dict,
    save_mesh_dir=None,
    save_frame_indices=None,
    save_max_frames=1,
    pointmap_path=None,
    pointmap_key=None,
    normalize_scale=False,
):
    v3d_o_c_pred_ra = data_pred["v3d_right.object"]
    v3d_o_c_gt_ra = data_gt["v3d_right.object"]
    is_valid = data_gt["is_valid"]
    
    # get hand alignment first, use the same alignment
    # 1. Get hand joints to compute alignment
    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]

    # # take until half (NOTE: need to fix the rest)
    # window_started_at = 11
    # until_idx = len(j3d_h_c_pred_ra) // 2 + window_started_at
    # j3d_h_c_pred_ra = j3d_h_c_pred_ra[window_started_at: until_idx]
    # j3d_h_c_gt_ra = j3d_h_c_gt_ra[window_started_at: until_idx]
    # v3d_o_c_pred_ra = data_pred["v3d_right.object"][window_started_at: until_idx]
    # v3d_o_c_gt_ra = data_gt["v3d_right.object"][window_started_at: until_idx]
    # is_valid = data_gt["is_valid"][window_started_at: until_idx]

    
    # 2. Handle potential length mismatches (common issue)
    T_pred = j3d_h_c_pred_ra.shape[0]
    if j3d_h_c_gt_ra.shape[0] != T_pred:
        j3d_h_c_gt_ra = j3d_h_c_gt_ra[:T_pred]
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[:T_pred]
        is_valid = is_valid[:T_pred]
    if v3d_o_c_pred_ra.shape[0] != T_pred:
        v3d_o_c_pred_ra = v3d_o_c_pred_ra[:T_pred]

    # _pred_joints_aligned, s, R, _t = root_procrustes_align_batch(j3d_h_c_pred_ra[0][None,...], j3d_h_c_gt_ra[0][None,...], is_valid)
    # # _pred_joints_aligned, s, R, _t = root_procrustes_align_batch(j3d_h_c_pred_ra, j3d_h_c_gt_ra, is_valid)

    # T, Vo, _ = v3d_o_c_pred_ra.shape
    # pred_verts_flat = v3d_o_c_pred_ra.reshape(-1, 3) # (T*Vo, 3)
    
    # pred_verts_aligned_flat = s * (pred_verts_flat @ R.T) + _t
    # # pred_verts_aligned_flat = s * (pred_verts_flat @ R.T)
    
    # v3d_o_c_pred_ra = pred_verts_aligned_flat.view(T, Vo, 3)

    if v3d_o_c_gt_ra.shape != v3d_o_c_pred_ra.shape:
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[: len(v3d_o_c_pred_ra), :, :]
        is_valid = is_valid[: len(v3d_o_c_pred_ra)]

    if normalize_scale and j3d_h_c_pred_ra.shape[0] > 0:
        is_valid_np_s = _to_numpy(is_valid).astype(bool) if is_valid is not None else None
        first_valid_s = 0
        if is_valid_np_s is not None:
            valid_frames_s = np.where(is_valid_np_s)[0]
            if len(valid_frames_s) > 0:
                first_valid_s = int(valid_frames_s[0])
        t0 = min(first_valid_s, j3d_h_c_pred_ra.shape[0] - 1)

        # Estimate scale as depth ratio in camera space.
        # The relative depth estimator produces all coordinates at a wrong global
        # scale; the z-depth of the hand joints at frame 0 captures this factor.
        # Translation cancels in wrist-relative subtraction, so only sf matters.
        pred_j3d_c = data_pred["j3d_c.right"]
        gt_j3d_c = data_gt["j3d_c.right"]
        pred_j0 = pred_j3d_c[t0]  # (J, 3)
        gt_j0 = gt_j3d_c[min(t0, gt_j3d_c.shape[0] - 1)]
        pred_depth = pred_j0[:, 2].mean()
        gt_depth = gt_j0[:, 2].mean()
        sf = gt_depth / (pred_depth + 1e-8)
        print(f"[normalize_scale] pred_depth={pred_depth:.4f} gt_depth={gt_depth:.4f} sf={sf:.4f} (cam-space depth ratio, frame {t0})")

        v3d_o_c_pred_ra = v3d_o_c_pred_ra * sf
        j3d_h_c_pred_ra = j3d_h_c_pred_ra * sf

    # torch.manual_seed(1)

    cd_list = []
    f5_list = []
    f10_list = []
    saved_frames = 0
    save_frame_set = None
    if save_frame_indices is not None:
        save_frame_set = set(save_frame_indices)
    pointmap_data = None
    if pointmap_path:
        pointmap_data = _load_pointmap_data(pointmap_path)
    debug_break = os.environ.get("HOLD_BREAK_CD_RIGHT", "0") == "1"
    debug_once = False
    is_valid_np = _to_numpy(is_valid).astype(bool) if is_valid is not None else None
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]
        v3d_gt = v3d_o_c_gt_ra[idx]

        # breakpoint() # check the mesh alignment here!!
        # _pred_joints_aligned, s, R, _t = procrustes_align_batch(j3d_h_c_pred_ra, j3d_h_c_gt_ra, is_valid)

        # _pred_joints_aligned, s, R, _t = procrustes_align_batch(j3d_h_c_pred_ra[idx][None,...], j3d_h_c_gt_ra[idx][None,...], None)
        # Vo, _ = v3d_pred.shape
        # pred_verts_flat = v3d_pred.reshape(-1, 3) # (T*Vo, 3)
        # pred_verts_aligned_flat = s * (pred_verts_flat @ R.T) + _t        
        # v3d_pred = pred_verts_aligned_flat.view(Vo, 3)

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:3000]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]

            # Optional interactive debug hook for checking hand/object alignment.
            # Enable with: HOLD_BREAK_CD_RIGHT=1
            if debug_break and (not debug_once) and (
                is_valid_np is None or (idx < len(is_valid_np) and is_valid_np[idx])
            ):
                import trimesh

                hand_mesh_pred_dbg = None
                hand_mesh_gt_dbg = None
                hand_mesh_pred_abs_dbg = None
                hand_mesh_gt_abs_dbg = None
                hand_pred_dbg = None
                hand_gt_dbg = None
                faces_hand_pred_dbg = _resolve_faces(data_pred, "right")
                faces_hand_gt_dbg = _resolve_faces(data_gt, "right")
                if "v3d_c.right" in data_pred and "root.right" in data_pred:
                    hand_pred_abs_dbg = _to_numpy(data_pred["v3d_c.right"][idx])
                    root_pred_dbg = _to_numpy(data_pred["root.right"][idx])[None, :]
                    hand_pred_dbg = _to_numpy(
                        hand_pred_abs_dbg - root_pred_dbg
                    )
                    if faces_hand_pred_dbg is not None:
                        hand_mesh_pred_abs_dbg = trimesh.Trimesh(
                            vertices=hand_pred_abs_dbg,
                            faces=faces_hand_pred_dbg,
                            process=False,
                        )
                        hand_mesh_pred_dbg = trimesh.Trimesh(
                            vertices=hand_pred_dbg,
                            faces=faces_hand_pred_dbg,
                            process=False,
                        )
                if "v3d_c.right" in data_gt and "root.right" in data_gt:
                    hand_gt_abs_dbg = _to_numpy(data_gt["v3d_c.right"][idx])
                    root_gt_dbg = _to_numpy(data_gt["root.right"][idx])[None, :]
                    hand_gt_dbg = _to_numpy(
                        hand_gt_abs_dbg - root_gt_dbg
                    )
                    if faces_hand_gt_dbg is not None:
                        hand_mesh_gt_abs_dbg = trimesh.Trimesh(
                            vertices=hand_gt_abs_dbg,
                            faces=faces_hand_gt_dbg,
                            process=False,
                        )
                        hand_mesh_gt_dbg = trimesh.Trimesh(
                            vertices=hand_gt_dbg,
                            faces=faces_hand_gt_dbg,
                            process=False,
                        )

                obj_pred_dbg = _to_numpy(v3d_pred)
                obj_gt_dbg = _to_numpy(v3d_gt)
                obj_pred_abs_dbg = None
                obj_gt_abs_dbg = None
                if "v3d_c.object" in data_pred:
                    obj_pred_abs_dbg = _to_numpy(data_pred["v3d_c.object"][idx])
                if "v3d_c.object" in data_gt:
                    obj_gt_abs_dbg = _to_numpy(data_gt["v3d_c.object"][idx])

                # Quick distance stats to avoid ambiguity in relative-vs-absolute checks.
                if hand_gt_dbg is not None:
                    h_c = hand_gt_dbg.mean(axis=0)
                    o_c = obj_gt_dbg.mean(axis=0)
                    print(
                        f"[DEBUG][cd_f_right] frame={idx} "
                        f"gt_rel_centroid_dist={float(np.linalg.norm(o_c - h_c)):.4f}m"
                    )
                    d2 = ((hand_gt_dbg[:, None, :] - obj_gt_dbg[None, :, :]) ** 2).sum(axis=2)
                    print(
                        f"[DEBUG][cd_f_right] frame={idx} "
                        f"gt_rel_min_surface_dist={float(np.sqrt(d2.min())):.4f}m"
                    )
                debug_once = True
                breakpoint()

            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
            # breakpoint()
            if (
                save_mesh_dir
                and saved_frames < save_max_frames
                and (save_frame_set is None or idx in save_frame_set)
            ):
                pointmap_xyz = None
                if pointmap_data is not None:
                    pointmap = _extract_pointmap(pointmap_data, idx, pointmap_key)
                    pointmap_xyz = _pointmap_to_xyz(pointmap)
                save_eval_alignment_meshes(
                    data_pred,
                    data_gt,
                    idx,
                    v3d_pred,
                    v3d_gt,
                    rand_pred_idx,
                    rand_gt_idx,
                    save_mesh_dir,
                    prefix="cd_right_",
                    pointmap_xyz=pointmap_xyz,
                )
                saved_frames += 1
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # breakpoint()

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_right"] = cd_list
    metric_dict["f5_right"] = f5_list * 100.0
    metric_dict["f10_right"] = f10_list * 100.0
    return metric_dict

def eval_cd_f_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_ra.object"]
    v3d_o_c_gt_ra = data_gt["v3d_ra.object"]
    is_valid = data_gt["is_valid"]

    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]

    # # debuggingL
    # j3d_h_c_pred_ra = j3d_h_c_pred_ra[:10]
    # j3d_h_c_gt_ra = j3d_h_c_gt_ra[:10]

    # 2. Handle potential length mismatches (common issue)
    T_pred = j3d_h_c_pred_ra.shape[0]
    if j3d_h_c_gt_ra.shape[0] != T_pred:
        j3d_h_c_gt_ra = j3d_h_c_gt_ra[:T_pred]
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[:T_pred]
        is_valid = is_valid[:T_pred]
    if v3d_o_c_pred_ra.shape[0] != T_pred: # Ensure object vertices also match
        v3d_o_c_pred_ra = v3d_o_c_pred_ra[:T_pred]

    # # 3. Align pred hand joints to gt hand joints to get s, R, t
    # # [CHANGE HERE]
    # _pred_joints_aligned, s, R, _t = root_procrustes_align_batch(
    #     j3d_h_c_pred_ra, j3d_h_c_gt_ra, is_valid)

    # # 4. Apply this same s, R, t to the predicted *object vertices*
    # T, Vo, _ = v3d_o_c_pred_ra.shape
    # pred_verts_flat = v3d_o_c_pred_ra.reshape(-1, 3) # (T*Vo, 3)
    
    # # Apply alignment: P_aligned = (s * (P_full @ R.T)) + t
    # pred_verts_aligned_flat = (s * (pred_verts_flat @ R.T))
    
    # # Reshape back and update the variable
    # v3d_o_c_pred_ra = pred_verts_aligned_flat.view(T, Vo, 3)
    
    # --- END FIX ---

    if v3d_o_c_gt_ra.shape != v3d_o_c_pred_ra.shape:
        v3d_o_c_gt_ra = v3d_o_c_gt_ra[: len(v3d_o_c_pred_ra), :, :]
        is_valid = is_valid[: len(v3d_o_c_pred_ra)]

    torch.manual_seed(1)
    # rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    # rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            v3d_gt = v3d_o_c_gt_ra[idx]

            num_pts = min(3000, v3d_pred.shape[0])
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:num_pts]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan
    
    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_ra"] = cd_list
    metric_dict["f5_ra"] = f5_list * 100.0
    metric_dict["f10_ra"] = f10_list * 100.0
    return metric_dict

# ----- MINE -----
def procrustes_align_batch(pred, gt, valid=None):
    """
    pred, gt: (T, J, 3) predicted & GT joints (same space, e.g. root-relative)
    valid: (T,) or (T,1) bool/0-1 tensor; if None, all frames valid.

    Returns:
        pred_aligned: (T, J, 3) = s R pred + t, aligned to gt
    """
    if pred.shape != gt.shape:
        gt = gt[: pred.shape[0], :, :]

    T, J, _ = pred.shape

    if valid is None:
        valid = torch.ones(T, device=pred.device, dtype=torch.bool)
    valid = valid[:T]
    valid = (valid.reshape(T) > 0.5).to(dtype=torch.bool)

    # mask invalid frames
    pred_v = pred[valid]   # (Tv, J, 3)
    gt_v   = gt[valid]     # (Tv, J, 3)

    # flatten over frames + joints: (N,3)
    P = pred_v.reshape(-1, 3)
    G = gt_v.reshape(-1, 3)

    # 1) subtract centroids
    mu_P = P.mean(dim=0, keepdim=True)   # (1,3)
    mu_G = G.mean(dim=0, keepdim=True)

    P0 = P - mu_P
    G0 = G - mu_G

    # 2) covariance & SVD
    H = P0.T @ G0   # (3,3)
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T  # (3,3)

    # ensure right-handed (det=+1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3) global scale
    var_P = (P0 ** 2).sum()
    s = (S.sum() / var_P)

    # 4) translation
    t = mu_G - s * (mu_P @ R.T)   # (1,3)

    # apply to full sequence
    P_full = pred.reshape(-1, 3)
    P_aligned = (s * (P_full @ R.T)) + t
    pred_aligned = P_aligned.view(T, J, 3)

    return pred_aligned, s, R, t

def root_scale_align_batch(pred_ra, gt_ra, valid=None):
    """
    pred_ra, gt_ra: (T, J, 3) root-relative joints (wrist at 0).
    valid: (T,) or (T,1) bool/0-1 tensor.

    Returns:
        pred_ra_aligned: (T, J, 3) = s * pred_ra, aligned in scale to gt_ra.
        s: scalar scale factor
    """
    if pred_ra.shape != gt_ra.shape:
        gt_ra = gt_ra[: pred_ra.shape[0], :, :]
    T, J, _ = pred_ra.shape
    
    if valid is None:
        valid = torch.ones(T, device=pred_ra.device, dtype=torch.bool)
    valid = valid[:T]
    valid = (valid.reshape(T) > 0.5).to(dtype=torch.bool)

    P = pred_ra[valid].reshape(-1, 3)   # (N,3)
    G = gt_ra[valid].reshape(-1, 3)

    # LS solution for s: argmin_s || sP - G ||^2
    num = (P * G).sum()
    den = (P * P).sum() + 1e-8
    s = num / den

    pred_ra_aligned = pred_ra * s

    return pred_ra_aligned, s

def root_procrustes_align_batch(pred, gt, valid=None):
    """
    Aligns pred to gt by finding s, R that minimize || s * P @ R.T - G ||^2.
    Assumes P and G are root-relative (centered at origin).
    
    pred, gt: (T, J, 3) predicted & GT joints (root-relative)
    valid: (T,) or (T,1) bool/0-1 tensor; if None, all frames valid.

    Returns:
        pred_aligned: (T, J, 3) = s R pred, aligned to gt
        s: scalar scale
        R: (3,3) rotation matrix
        t: (1,3) zero tensor (for API compatibility)
    """
    if pred.shape != gt.shape:
        gt = gt[: pred.shape[0], :, :]

    T, J, _ = pred.shape

    if valid is None:
        valid = torch.ones(T, device=pred.device, dtype=torch.bool)
    valid = valid[:T]
    valid = (valid.reshape(T) > 0.5).to(dtype=torch.bool)

    # mask invalid frames
    pred_v = pred[valid]   # (Tv, J, 3)
    gt_v   = gt[valid]     # (Tv, J, 3)

    # flatten over frames + joints: (N,3)
    P = pred_v.reshape(-1, 3)
    G = gt_v.reshape(-1, 3)

    # --- DO NOT SUBTRACT CENTROIDS ---
    # We assume data is already root-relative

    # 2) covariance & SVD (on non-centered data)
    H = P.T @ G   # (3,3)
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T  # (3,3)

    # ensure right-handed (det=+1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3) global scale (formula derived from s = trace(R @ H) / trace(P.T @ P))
    var_P = (P ** 2).sum()  # Variance of non-centered data
    s = (S.sum() / var_P)
    
    # 4) NO translation
    t = torch.zeros(1, 3, device=pred.device)

    # apply to full sequence
    P_full = pred.reshape(-1, 3)
    P_aligned = (s * (P_full @ R.T)) # NO "+ t"
    pred_aligned = P_aligned.view(T, J, 3)

    return pred_aligned, s, R, t

def icp_after_align(P_pred, P_gt, max_corr=0.01, max_iter=50):
    """Returns fitness, inlier_rmse, and the found extra transform (R_icp,t_icp). 
       Set max_corr to ~1% of object size (in your units)."""
    import open3d as o3d
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(P_pred)
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(P_gt)
    threshold = max_corr
    init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    T = reg.transformation
    R_icp, t_icp = T[:3,:3], T[:3,3]
    return reg.fitness, reg.inlier_rmse, R_icp, t_icp
# ---- MINE -----

def eval_mpjpe_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]
    is_valid = data_gt["is_valid"]

    # added
    j3d_h_c_pred_ra, s, R, t = procrustes_align_batch(j3d_h_c_pred_ra, j3d_h_c_gt_ra, is_valid)

    mpjpe_ra_r = metrics.compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra, is_valid)
    mpjpe_ra_r = mpjpe_ra_r.mean(axis=1) * 1000  # Use dim instead of axis for PyTorch

    metric_dict["mpjpe_ra_r"] = mpjpe_ra_r
    return metric_dict


def eval_ious(data_pred, data_gt, metric_dict):
    masks_pred = data_pred["masks_pred"].long().numpy()
    masks_gt = data_gt["masks_gt"].long().numpy()
    is_valid = data_gt["is_valid"]
    ious = compute_iou_per_frame(masks_pred, masks_gt)
    not_valid = (1 - is_valid).numpy().astype(bool)
    ious[not_valid] = np.nan
    metric_dict["ious"] = ious * 100.0
    return metric_dict
