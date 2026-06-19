import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


# def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
#     if mask is None:
#         return trimmed_l1_loss(pred, gt, quantile)
#     else:
#         sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
#         quantile_mask = (
#             (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
#             if quantile < 1
#             else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
#         )
#         ndim = sum_loss.shape[-1]
#         if normalize:
#             return torch.sum((sum_loss * mask)[quantile_mask]) / (
#                 ndim * torch.sum(mask[quantile_mask]) + 1e-8
#             )
#         else:
#             return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        # sum_loss.shape 
        # block     [218255, 1]
        # apple     [36673, 475, 1]     17,419,675
        # creeper   [37587, 360, 1]     13,531,320
        # backpack  [37828, 180, 1]     6,809,040
        # quantile_mask = (
        #     (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
        #     if quantile < 1
        #     else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        # )
        # use torch.sort instead of torch.quantile when input too large
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])

def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    if loss.numel() == 0 or quantile >= 1:
        return loss.mean()
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss <= loss_at_quantile]
    if trimmed_loss.numel() == 0:
        return loss.mean()
    return trimmed_loss.mean()


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    if loss.numel() == 0 or quantile >= 1:
        return loss.mean()
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss <= loss_at_quantile]
    if trimmed_loss.numel() == 0:
        return loss.mean()
    return trimmed_loss.mean()


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_ray_local_isometry_loss(
    means_t: torch.Tensor,
    means0: torch.Tensor,
    w2cs: torch.Tensor,
    knn_k: int,
    radius_mult: float,
    huber_beta: float,
    edge_weight_temp: float,
):
    """
    Preserve local foreground structure with separate ray/perpendicular components.
    :param means_t: (G, B, 3)
    :param means0: (G, 3)
    :param w2cs: (B, 4, 4)
    return ray_loss, perp_loss, dist_loss, num_edges
    """
    zero = means_t.new_zeros(())
    num_gaussians = means0.shape[0]
    if num_gaussians <= 1 or knn_k <= 0:
        return zero, zero, zero, zero

    k = min(knn_k, num_gaussians - 1)
    eps = 1e-6

    with torch.no_grad():
        means0_np = means0.detach().float().cpu().numpy()
        knn_model = NearestNeighbors(
            n_neighbors=k + 1, algorithm="auto", metric="euclidean"
        ).fit(means0_np)
        dists_np, inds_np = knn_model.kneighbors(means0_np)
        nbr_inds = torch.from_numpy(inds_np[:, 1:]).to(
            device=means_t.device, dtype=torch.long
        )
        cand_dists = torch.from_numpy(dists_np[:, 1:]).to(
            device=means_t.device, dtype=means_t.dtype
        )
        valid = cand_dists > eps
        if radius_mult > 0 and valid.any():
            radius = cand_dists[valid].median() * radius_mult
            valid = valid & (cand_dists <= radius)
        if not valid.any():
            return zero, zero, zero, zero

        src_inds = (
            torch.arange(num_gaussians, device=means_t.device)[:, None]
            .expand(-1, k)
        )
        src_inds = src_inds[valid]
        nbr_inds = nbr_inds[valid]
        rest_len = cand_dists[valid].clamp_min(eps)

        if edge_weight_temp > 0:
            weight_scale = rest_len.median().clamp_min(eps)
            edge_weights = torch.exp(
                -((rest_len / weight_scale) ** 2) / edge_weight_temp
            )
        else:
            edge_weights = torch.ones_like(rest_len)

    e_t = means_t[src_inds] - means_t[nbr_inds]  # (E, B, 3)
    e0 = means0.detach()[src_inds] - means0.detach()[nbr_inds]  # (E, 3)
    rest_len = rest_len[:, None]
    edge_weights = edge_weights[:, None]

    camera_centers = torch.linalg.inv(w2cs.to(dtype=means_t.dtype))[:, :3, 3]  # (B, 3)
    edge_midpoints = 0.5 * (means_t[src_inds] + means_t[nbr_inds])
    ray_dirs = F.normalize(
        edge_midpoints.detach() - camera_centers[None], p=2.0, dim=-1
    )  # (E, B, 3)

    dot_t = (e_t * ray_dirs).sum(dim=-1)
    dot0 = (e0[:, None] * ray_dirs).sum(dim=-1)
    ray_err = (dot_t.abs() - dot0.abs()) / rest_len

    perp_t = e_t - dot_t[..., None] * ray_dirs
    perp0 = e0[:, None] - dot0[..., None] * ray_dirs
    perp_err = (perp_t.norm(dim=-1) - perp0.norm(dim=-1)) / rest_len

    dist_err = (e_t.norm(dim=-1) - rest_len) / rest_len

    def robust_mean(err):
        if huber_beta > 0:
            vals = F.huber_loss(
                err, torch.zeros_like(err), delta=huber_beta, reduction="none"
            )
        else:
            vals = err.pow(2)
        return (vals * edge_weights).sum() / (edge_weights.sum() * err.shape[1] + eps)

    ray_loss = robust_mean(ray_err)
    perp_loss = robust_mean(perp_err)
    dist_loss = robust_mean(dist_err)
    num_edges = means_t.new_tensor(float(src_inds.shape[0]))
    return ray_loss, perp_loss, dist_loss, num_edges


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss
