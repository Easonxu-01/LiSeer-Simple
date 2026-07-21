import torch, torch.nn as nn, torch.nn.functional as F
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmengine.model import BaseModule
from copy import deepcopy
from projects.unilidar_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.unilidar_plugin.utils.lovasz_softmax import lovasz_softmax
import numpy as np
from projects.unilidar_plugin.utils.nusc_param import nusc_class_frequencies, nusc_class_names
from projects.unilidar_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names
import torch.distributed as dist
from .Spectraldistiller import TPVHighFreqSpectralDistiller
import torch.nn.functional as F

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Movable objects', 'Other Ground']


def _cb_adc_consistency_loss(
    logits_p,
    logits_q,
    voxel_label,
    num_classes=8,
    T=1.5,
    consistency_loss_weight=1.0,
    outer_scale=2.0,

    # Label / empty setting
    ignore_index=0,
    empty_class=0,

    # Additive structural terms
    boundary_weight=1.0,
    minority_classes=(5, 6),
    minority_weight=5.0,
    continuity_weight=0.001,

    # Asymmetric empty-pull term
    # Trigger: GT non-empty valid voxel, exactly one branch predicts empty_class.
    # Direction: non-empty branch.detach() -> empty branch.
    empty_pull_weight=0.03,
    teacher_conf_threshold=0.0,

    eps=1e-8,
    return_dict=False,
):
    """
    Final compact CBA-CL loss with asymmetric empty-pull.

    Components:
        1. L_base:
            Original entropy-weighted symmetric JS consistency.

        2. L_boundary:
            Extra JS consistency on one-voxel-thin GT semantic boundary voxels.

        3. L_minority:
            Extra JS consistency on GT minority-class voxels.

        4. L_continuity:
            Single-view same-category continuity.
            For adjacent valid voxel pairs with the same GT label:
                TV(p_i, p_j) + TV(q_i, q_j)

        5. L_empty_asym:
            If one branch predicts empty_class and the other predicts non-empty,
            the non-empty branch is detached as teacher and pulls the empty
            branch through KL. This prevents the non-empty prediction from being
            dragged toward empty by symmetric JS.

    Total:
        L = L_base
          + boundary_weight   * L_boundary
          + minority_weight   * L_minority
          + continuity_weight * L_continuity
          + empty_pull_weight * L_empty_asym

        L = L * T^2 * consistency_loss_weight * outer_scale

    Notes:
        - No boundary maxpool / dilation.
        - No multiplicative modulator.
        - Confidence weight in JS terms is not detached, matching the original
          optimizable baseline.
        - Empty-pull uses prediction-based pseudo routing because no explicit
          per-view occupancy is available.
    """

    # 0) Checks
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"logits_p and logits_q must have the same shape, "
            f"got {logits_p.shape} and {logits_q.shape}"
        )

    if logits_p.dim() != 5:
        raise ValueError(
            f"expected logits shape [B, C, D, H, W], got {logits_p.shape}"
        )

    if voxel_label is None:
        raise ValueError("voxel_label is required for this supervised CBA-CL loss.")

    B, C, D, H, W = logits_p.shape

    if C != num_classes:
        raise ValueError(
            f"num_classes={num_classes}, but logits channel dim is {C}"
        )

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    if empty_class is None:
        empty_pull_weight = 0.0
    else:
        if not (0 <= empty_class < C):
            raise ValueError(
                f"empty_class={empty_class} is out of range [0, {C})"
            )

    device = logits_p.device
    dtype = logits_p.dtype

    if voxel_label.dim() == 3:
        label = voxel_label.unsqueeze(0)
    else:
        label = voxel_label

    if label.shape != (B, D, H, W):
        raise ValueError(
            f"voxel_label shape should be [B,D,H,W] or [D,H,W], "
            f"got {voxel_label.shape}, expected {(B, D, H, W)}"
        )

    label = label.to(device=device)
    valid = (label != ignore_index).to(dtype=dtype)

    valid_count = valid.sum()
    if valid_count.item() == 0:
        zero = logits_p.sum() * 0.0
        if return_dict:
            return {
                "loss_total": zero,
                "L_base_raw": zero.detach(),
                "L_boundary_raw": zero.detach(),
                "L_minority_raw": zero.detach(),
                "L_continuity_raw": zero.detach(),
                "L_empty_asym_raw": zero.detach(),
                "valid_count": valid_count.detach(),
                "valid_ratio": valid.detach().mean(),
            }
        return zero

    denom_valid = valid_count.clamp_min(1.0)

    # 1) Temperature-scaled distributions
    log_p = F.log_softmax(logits_p / T, dim=1)
    log_q = F.log_softmax(logits_q / T, dim=1)

    p = torch.exp(log_p)
    q = torch.exp(log_q)

    # 2) Symmetric JS divergence
    #    Keep the same style as the original working baseline.
    m = 0.5 * (p + q)
    log_m = torch.log(m + eps)

    kl_pm = F.kl_div(log_m, p, reduction="none").sum(dim=1)
    kl_qm = F.kl_div(log_m, q, reduction="none").sum(dim=1)

    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = js_div.clamp_min(0.0)

    # 3) Entropy confidence weight
    #    Not detached. This preserves the original self-pressure mechanism.
    entropy_p = -(p * log_p).sum(dim=1)
    entropy_q = -(q * log_q).sum(dim=1)

    mean_entropy = 0.5 * (entropy_p + entropy_q)
    weight_conf = torch.exp(-mean_entropy)

    weighted_js = js_div * weight_conf

    # 4) GT boundary and minority masks
    #    Boundary is one-voxel-thin. No maxpooling / dilation.
    with torch.no_grad():
        is_boundary = torch.zeros_like(valid)

        # D-axis boundary
        if D > 1:
            diff = (
                (label[:, 1:, :, :] != label[:, :-1, :, :]).to(dtype)
                * valid[:, 1:, :, :]
                * valid[:, :-1, :, :]
            )
            is_boundary[:, 1:, :, :] = torch.maximum(
                is_boundary[:, 1:, :, :], diff
            )
            is_boundary[:, :-1, :, :] = torch.maximum(
                is_boundary[:, :-1, :, :], diff
            )

        # H-axis boundary
        if H > 1:
            diff = (
                (label[:, :, 1:, :] != label[:, :, :-1, :]).to(dtype)
                * valid[:, :, 1:, :]
                * valid[:, :, :-1, :]
            )
            is_boundary[:, :, 1:, :] = torch.maximum(
                is_boundary[:, :, 1:, :], diff
            )
            is_boundary[:, :, :-1, :] = torch.maximum(
                is_boundary[:, :, :-1, :], diff
            )

        # W-axis boundary
        if W > 1:
            diff = (
                (label[:, :, :, 1:] != label[:, :, :, :-1]).to(dtype)
                * valid[:, :, :, 1:]
                * valid[:, :, :, :-1]
            )
            is_boundary[:, :, :, 1:] = torch.maximum(
                is_boundary[:, :, :, 1:], diff
            )
            is_boundary[:, :, :, :-1] = torch.maximum(
                is_boundary[:, :, :, :-1], diff
            )

        is_boundary = is_boundary * valid

        is_minority = torch.zeros_like(valid)
        if minority_classes is not None and len(minority_classes) > 0:
            for cls in minority_classes:
                if cls < 0 or cls >= C:
                    raise ValueError(
                        f"minority class {cls} out of range [0, {C})"
                    )
                is_minority = torch.maximum(
                    is_minority,
                    (label == cls).to(dtype),
                )

            is_minority = is_minority * valid

    # 5) Base JS consistency
    L_base = (weighted_js * valid).sum() / denom_valid

    # 6) Boundary JS extra
    #    Normalize by valid_count to keep it as a mild additive term.
    if boundary_weight is not None and boundary_weight > 0.0:
        L_boundary = (weighted_js * is_boundary).sum() / denom_valid
    else:
        L_boundary = logits_p.sum() * 0.0

    # 7) Minority JS extra
    if minority_weight is not None and minority_weight > 0.0:
        L_minority = (weighted_js * is_minority).sum() / denom_valid
    else:
        L_minority = logits_p.sum() * 0.0

    # 8) Same-category single-view continuity
    def tv_pair(a, b):
        return 0.5 * (a - b).abs().sum(dim=1)

    L_continuity = logits_p.sum() * 0.0
    continuity_pair_count = logits_p.new_tensor(0.0)

    if continuity_weight is not None and continuity_weight > 0.0:
        cont_sum = logits_p.sum() * 0.0

        # D-axis same-category pairs
        if D > 1:
            with torch.no_grad():
                same_d = (
                    (label[:, 1:, :, :] == label[:, :-1, :, :]).to(dtype)
                    * valid[:, 1:, :, :]
                    * valid[:, :-1, :, :]
                )

            tv_p_d = tv_pair(p[:, :, 1:, :, :], p[:, :, :-1, :, :])
            tv_q_d = tv_pair(q[:, :, 1:, :, :], q[:, :, :-1, :, :])

            cont_sum = cont_sum + (0.5 * (tv_p_d + tv_q_d) * same_d).sum()
            continuity_pair_count = continuity_pair_count + same_d.sum()

        # H-axis same-category pairs
        if H > 1:
            with torch.no_grad():
                same_h = (
                    (label[:, :, 1:, :] == label[:, :, :-1, :]).to(dtype)
                    * valid[:, :, 1:, :]
                    * valid[:, :, :-1, :]
                )

            tv_p_h = tv_pair(p[:, :, :, 1:, :], p[:, :, :, :-1, :])
            tv_q_h = tv_pair(q[:, :, :, 1:, :], q[:, :, :, :-1, :])

            cont_sum = cont_sum + (0.5 * (tv_p_h + tv_q_h) * same_h).sum()
            continuity_pair_count = continuity_pair_count + same_h.sum()

        # W-axis same-category pairs
        if W > 1:
            with torch.no_grad():
                same_w = (
                    (label[:, :, :, 1:] == label[:, :, :, :-1]).to(dtype)
                    * valid[:, :, :, 1:]
                    * valid[:, :, :, :-1]
                )

            tv_p_w = tv_pair(p[:, :, :, :, 1:], p[:, :, :, :, :-1])
            tv_q_w = tv_pair(q[:, :, :, :, 1:], q[:, :, :, :, :-1])

            cont_sum = cont_sum + (0.5 * (tv_p_w + tv_q_w) * same_w).sum()
            continuity_pair_count = continuity_pair_count + same_w.sum()

        # Normalize by valid_count, not pair_count.
        L_continuity = cont_sum / denom_valid

    # 9) Asymmetric empty-pull
    #
    # Since true per-view occupancy is unavailable, we use prediction-based
    # pseudo routing:
    #   q predicts empty, p predicts non-empty -> p.detach() teaches q
    #   p predicts empty, q predicts non-empty -> q.detach() teaches p
    #
    # This term is additive. It does not replace the symmetric JS base.
    L_empty_asym = logits_p.sum() * 0.0

    with torch.no_grad():
        pull_q_raw = torch.zeros_like(valid)
        pull_p_raw = torch.zeros_like(valid)
        pull_q_mask = torch.zeros_like(valid)
        pull_p_mask = torch.zeros_like(valid)

    if empty_pull_weight is not None and empty_pull_weight > 0.0:
        with torch.no_grad():
            pred_p = p.argmax(dim=1)
            pred_q = q.argmax(dim=1)

            p_empty = (pred_p == empty_class).to(dtype)
            q_empty = (pred_q == empty_class).to(dtype)

            p_non_empty = 1.0 - p_empty
            q_non_empty = 1.0 - q_empty

            # Trigger only on GT-valid non-empty voxels.
            # q empty, p non-empty: p teaches q.
            pull_q_raw = q_empty * p_non_empty * valid

            # p empty, q non-empty: q teaches p.
            pull_p_raw = p_empty * q_non_empty * valid

            # Optional teacher confidence gate.
            p_teacher_conf = p.detach().max(dim=1).values
            q_teacher_conf = q.detach().max(dim=1).values

            if teacher_conf_threshold is not None and teacher_conf_threshold > 0.0:
                pull_q_raw = pull_q_raw * (
                    p_teacher_conf >= teacher_conf_threshold
                ).to(dtype)
                pull_p_raw = pull_p_raw * (
                    q_teacher_conf >= teacher_conf_threshold
                ).to(dtype)

            # Reliability weight for teacher side.
            # Detached by construction: this term should route gradients only
            # into the empty student branch.
            reliability_p_teacher = torch.exp(-entropy_p.detach())
            reliability_q_teacher = torch.exp(-entropy_q.detach())

            pull_q_mask = pull_q_raw * reliability_p_teacher
            pull_p_mask = pull_p_raw * reliability_q_teacher

        # p.detach() -> q
        kl_p_to_q = (
            p.detach() * (torch.log(p.detach() + eps) - log_q)
        ).sum(dim=1)

        # q.detach() -> p
        kl_q_to_p = (
            q.detach() * (torch.log(q.detach() + eps) - log_p)
        ).sum(dim=1)

        # Normalize by valid_count, not trigger count, to keep this term mild.
        L_empty_asym = (
            (kl_p_to_q * pull_q_mask).sum()
            +
            (kl_q_to_p * pull_p_mask).sum()
        ) / denom_valid

    # 10) Total
    total_unscaled = (
        L_base
        + (boundary_weight or 0.0) * L_boundary
        + (minority_weight or 0.0) * L_minority
        + (continuity_weight or 0.0) * L_continuity
        + (empty_pull_weight or 0.0) * L_empty_asym
    )

    total = total_unscaled * (T ** 2) * consistency_loss_weight * outer_scale

    # 11) Diagnostics
    if return_dict:
        with torch.no_grad():
            scale = (T ** 2) * consistency_loss_weight * outer_scale
            valid_count_safe = valid_count.clamp_min(1.0)

            diagnostics = {
                "loss_total": total,
                "loss_unscaled_before_T": total_unscaled.detach(),

                # raw components before local weights and global scale
                "L_base_raw": L_base.detach(),
                "L_boundary_raw": L_boundary.detach(),
                "L_minority_raw": L_minority.detach(),
                "L_continuity_raw": L_continuity.detach(),
                "L_empty_asym_raw": L_empty_asym.detach(),

                # actual contributions to final scaled loss
                "L_base_contrib": L_base.detach() * scale,
                "L_boundary_contrib": (
                    L_boundary.detach() * (boundary_weight or 0.0) * scale
                ),
                "L_minority_contrib": (
                    L_minority.detach() * (minority_weight or 0.0) * scale
                ),
                "L_continuity_contrib": (
                    L_continuity.detach() * (continuity_weight or 0.0) * scale
                ),
                "L_empty_asym_contrib": (
                    L_empty_asym.detach() * (empty_pull_weight or 0.0) * scale
                ),

                # raw consistency
                "js_div_valid_mean": (
                    js_div.detach() * valid
                ).sum() / valid_count_safe,
                "weighted_js_valid_mean": (
                    weighted_js.detach() * valid
                ).sum() / valid_count_safe,

                # entropy / confidence
                "entropy_p_valid": (
                    entropy_p.detach() * valid
                ).sum() / valid_count_safe,
                "entropy_q_valid": (
                    entropy_q.detach() * valid
                ).sum() / valid_count_safe,
                "weight_conf_valid_mean": (
                    weight_conf.detach() * valid
                ).sum() / valid_count_safe,

                # mask coverage
                "boundary_count": is_boundary.detach().sum(),
                "boundary_ratio": is_boundary.detach().sum() / valid_count_safe,
                "minority_count": is_minority.detach().sum(),
                "minority_ratio": is_minority.detach().sum() / valid_count_safe,
                "continuity_pair_count": continuity_pair_count.detach(),
                "continuity_pair_per_valid": (
                    continuity_pair_count.detach() / valid_count_safe
                ),

                # empty pull diagnostics
                "pull_q_trigger_count": pull_q_raw.detach().sum(),
                "pull_p_trigger_count": pull_p_raw.detach().sum(),
                "pull_q_trigger_ratio": (
                    pull_q_raw.detach().sum() / valid_count_safe
                ),
                "pull_p_trigger_ratio": (
                    pull_p_raw.detach().sum() / valid_count_safe
                ),
                "pull_q_weighted_ratio": (
                    pull_q_mask.detach().sum() / valid_count_safe
                ),
                "pull_p_weighted_ratio": (
                    pull_p_mask.detach().sum() / valid_count_safe
                ),

                "valid_count": valid_count.detach(),
                "valid_ratio": valid.detach().mean(),
            }

        return diagnostics

    return total

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    trailing_shape = shape[ndim:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.reshape(-1, *trailing_shape)
    return ret

@VOXEL_ENCODERS.register_module()
class TPVAggregator_Occ(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, grid_size_occ, coarse_ratio, loss_weight=[1,1,1,1],
        nbr_classes=20, in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False, dual = False,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.loss_weight = loss_weight
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.coarse_ratio = coarse_ratio
        out_dims = in_dims if out_dims is None else out_dims
        self.dual = dual

        if self.dual:
            self.decoder_nu = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.decoder_sk = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.classifier_nu = nn.Linear(out_dims, nbr_classes[0])
            self.classes_nu = nbr_classes[0]
            self.classifier_sk = nn.Linear(out_dims, nbr_classes[1])
            self.classes_sk = nbr_classes[1]
        else:
            self.decoder = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.classifier = nn.Linear(out_dims, nbr_classes)
            self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        self.ce_loss_func = CE_ssc_loss
        self.lovasz_loss_func = lovasz_softmax

        # Running per-class frequencies used by the seesaw re-weighting.
        self.register_buffer(
            "class_counts", torch.ones(self.classes)  # start at 1 to avoid division by zero
        )

    def seesaw_loss(self, logits, labels, p=0.8, eps=1e-6, ignore_index=0):
        """Seesaw loss over arbitrary spatial shapes.

        Args:
            logits: any shape, provided dim=1 is the channel axis (C).
            labels: semantic labels whose spatial dims match ``logits``.
        """
        # ---- reshape: (B, C, ...) → (N, C) ----
        B, C = logits.shape[:2]
        logits_flat = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        # logits_flat shape = (N, C)
        # ---- labels reshape: (B, ...) → (N,) ----
        labels_flat = labels.reshape(-1)
        # ---- mask ignore_index ----
        valid = labels_flat != ignore_index
        if not valid.any():
            # No valid point in this batch.
            return logits.sum() * 0
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
        # ---- one-hot ----
        targets = F.one_hot(labels_flat, num_classes=C).float()
        def distribution_agnostic_seesaw_loss(logits, targets, p=0.8, eps=1e-6):
            # Accumulate class frequencies.
            batch_class_count = targets.sum(dim=0)
            self.class_counts += batch_class_count
            cc = self.class_counts
            # Pairwise mitigation matrix.
            conditions = cc[:, None] > cc[None, :]
            trues = (cc[None, :] / cc[:, None]).pow(p)
            falses = torch.ones_like(trues)
            s = torch.where(conditions, trues, falses)
            # Shift logits for numerical stability.
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            numerator = torch.exp(logits)
            denominator = (
                (1 - targets)[:, None, :] * s[None, :, :] * torch.exp(logits)[:, None, :]
            ).sum(dim=-1) + torch.exp(logits)
            sigma = numerator / (denominator + eps)
            loss = (-targets * torch.log(sigma + eps)).sum(dim=-1)
            return loss.mean()
        # ---- seesaw loss ----
        return distribution_agnostic_seesaw_loss(logits_flat, targets, p=p, eps=eps)

        if not self.dual:
            if nbr_classes == 17:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
                self.class_names = nusc_class_names
            elif nbr_classes == 20:
                self.class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))# FIXME hardcode
                self.class_names = kitti_class_names
            else:
                self.class_weights = torch.ones(nbr_classes)/nbr_classes  # FIXME hardcode
                self.class_names = CLASS_NAMES_DG
        else:
            self.class_weights_nu = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
            self.class_weights_sk = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))

    def forward(self, tpv_list, voxels=None, voxels_coarse=None, voxel_label=None, dataset_flag=None, return_loss=True):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        if not self.dual or (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
            tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
            tpv_hw = tpv_xy.permute(0, 1, 3, 2)
            tpv_wz = tpv_zx.permute(0, 1, 3, 2)
            tpv_zh = tpv_yz.permute(0, 1, 3, 2)
            bs, c, _, _ = tpv_hw.shape

            if self.scale_h != 1 or self.scale_w != 1:
                tpv_hw = F.interpolate(
                    tpv_hw,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
            if self.scale_z != 1 or self.scale_h != 1:
                tpv_zh = F.interpolate(
                    tpv_zh,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
            if self.scale_w != 1 or self.scale_z != 1:
                tpv_wz = F.interpolate(
                    tpv_wz,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )

            # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
            _, n, _ = voxels_coarse.shape
            voxels_coarse = voxels_coarse.reshape(bs, 1, n, 3)
            voxels_coarse[..., 0] = voxels_coarse[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse[..., 1] = voxels_coarse[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse[..., 2] = voxels_coarse[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc_vox = voxels_coarse[:, :, :, [0, 1]]
            tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox = voxels_coarse[:, :, :, [1, 2]]
            tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
            sample_loc_vox = voxels_coarse[:, :, :, [2, 0]]
            tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
            fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox

            fused = fused.permute(0, 2, 1)   # bs, whz, c
        else:
            tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
            tpv_hw = tpv_xy.permute(0, 1, 3, 2)
            tpv_wz = tpv_zx.permute(0, 1, 3, 2)
            tpv_zh = tpv_yz.permute(0, 1, 3, 2)

            tpv_hw_nu, tpv_wz_nu, tpv_zh_nu = tpv_hw[::2], tpv_wz[::2], tpv_zh[::2]
            tpv_hw_sk, tpv_wz_sk, tpv_zh_sk = tpv_hw[1::2], tpv_wz[1::2], tpv_zh[1::2]

            bs_nu, c, _, _ = tpv_hw_nu.shape
            bs_sk, c, _, _ = tpv_hw_sk.shape

            if self.scale_h != 1 or self.scale_w != 1:
                tpv_hw_nu = F.interpolate(
                    tpv_hw_nu,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
                tpv_hw_sk = F.interpolate(
                    tpv_hw_sk,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
            if self.scale_z != 1 or self.scale_h != 1:
                tpv_zh_nu = F.interpolate(
                    tpv_zh_nu,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
                tpv_zh_sk = F.interpolate(
                    tpv_zh_sk,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
            if self.scale_w != 1 or self.scale_z != 1:
                tpv_wz_nu = F.interpolate(
                    tpv_wz_nu,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )
                tpv_wz_sk = F.interpolate(
                    tpv_wz_sk,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )

            # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
            if isinstance(voxels_coarse, list):
                voxels_coarse_nu = torch.cat(voxels_coarse[::2], dim=0)
                voxels_coarse_sk = torch.cat(voxels_coarse[1::2], dim=0)
                _, n_nu, _ = voxels_coarse_nu.shape
                _, n_sk, _ = voxels_coarse_sk.shape
            else:
                voxels_coarse_nu = voxels_coarse[::2]
                voxels_coarse_sk = voxels_coarse[1::2]
                _, n_nu, _ = voxels_coarse_nu.shape
                _, n_sk, _ = voxels_coarse_sk.shape

            voxels_coarse_nu = voxels_coarse_nu.reshape(bs_nu, 1, n_nu, 3)
            voxels_coarse_nu[..., 0] = voxels_coarse_nu[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse_nu[..., 1] = voxels_coarse_nu[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse_nu[..., 2] = voxels_coarse_nu[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            voxels_coarse_sk = voxels_coarse_sk.reshape(bs_sk, 1, n_sk, 3)
            voxels_coarse_sk[..., 0] = voxels_coarse_sk[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse_sk[..., 1] = voxels_coarse_sk[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse_sk[..., 2] = voxels_coarse_sk[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [0, 1]]
            tpv_hw_vox_nu = F.grid_sample(tpv_hw_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [1, 2]]
            tpv_zh_vox_nu = F.grid_sample(tpv_zh_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2)
            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [2, 0]]
            tpv_wz_vox_nu = F.grid_sample(tpv_wz_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2)
            fused_nu_input = tpv_hw_vox_nu + tpv_zh_vox_nu + tpv_wz_vox_nu
            fused_nu_input = fused_nu_input.permute(0, 2, 1)   # bs_nu, whz, c

            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [0, 1]]
            tpv_hw_vox_sk = F.grid_sample(tpv_hw_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [1, 2]]
            tpv_zh_vox_sk = F.grid_sample(tpv_zh_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2)
            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [2, 0]]
            tpv_wz_vox_sk = F.grid_sample(tpv_wz_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2)
            fused_sk_input = tpv_hw_vox_sk + tpv_zh_vox_sk + tpv_wz_vox_sk
            fused_sk_input = fused_sk_input.permute(0, 2, 1)   # bs_sk, whz, c

        if self.use_checkpoint:
            if not self.dual:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    fused_nu = torch.utils.checkpoint.checkpoint(self.decoder_nu, fused)
                    logits_nu = torch.utils.checkpoint.checkpoint(self.classifier_nu, fused_nu)
                if flag.item() == 2:
                    fused_sk = torch.utils.checkpoint.checkpoint(self.decoder_sk, fused)
                    logits_sk = torch.utils.checkpoint.checkpoint(self.classifier_sk, fused_sk)
            else:
                fused_nu = torch.utils.checkpoint.checkpoint(self.decoder_nu, fused_nu_input)
                fused_sk = torch.utils.checkpoint.checkpoint(self.decoder_sk, fused_sk_input)
                logits_nu = torch.utils.checkpoint.checkpoint(self.classifier_nu, fused_nu)
                logits_sk = torch.utils.checkpoint.checkpoint(self.classifier_sk, fused_sk)
        else:
            if not self.dual:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    fused_nu = self.decoder_nu(fused)
                    logits_nu = self.classifier_nu(fused_nu)
                if flag.item() == 2:
                    fused_sk = self.decoder_sk(fused)
                    logits_sk = self.classifier_sk(fused_sk)
            else:
                fused_nu = self.decoder_nu(fused_nu_input)
                fused_sk = self.decoder_sk(fused_sk_input)
                logits_nu = self.classifier_nu(fused_nu)
                logits_sk = self.classifier_sk(fused_sk)

        if not self.dual:
            W, H, D = int(self.grid_size_occ[0]/self.coarse_ratio), int(self.grid_size_occ[1]/self.coarse_ratio), int(self.grid_size_occ[2]/self.coarse_ratio)
            logits = logits.permute(0, 2, 1)
            B, C, N = logits.shape
            logits = logits.reshape(B, C, W, H ,D)
        elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
            if flag.item() == 1:
                W1_, H1_, D1_ = int(self.grid_size_occ[0][0]/self.coarse_ratio), int(self.grid_size_occ[0][1]/self.coarse_ratio), int(self.grid_size_occ[0][2]/self.coarse_ratio)
                logits_nu = logits_nu.permute(0, 2, 1)
                B1_, C1_, N1_ = logits_nu.shape
                logits_nu = logits_nu.reshape(B1_, C1_, W1_, H1_, D1_)
            if flag.item() == 2:
                W2_, H2_, D2_ = int(self.grid_size_occ[1][0]/self.coarse_ratio), int(self.grid_size_occ[1][1]/self.coarse_ratio), int(self.grid_size_occ[1][2]/self.coarse_ratio)
                logits_sk = logits_sk.permute(0, 2, 1)
                B2_, C2_, N2_ = logits_sk.shape
                logits_sk = logits_sk.reshape(B2_, C2_, W2_, H2_, D2_)
        else:
            W1_, H1_, D1_ = int(self.grid_size_occ[0][0]/self.coarse_ratio), int(self.grid_size_occ[0][1]/self.coarse_ratio), int(self.grid_size_occ[0][2]/self.coarse_ratio)
            logits_nu = logits_nu.permute(0, 2, 1)
            B1_, C1_, N1_ = logits_nu.shape
            logits_nu = logits_nu.reshape(B1_, C1_, W1_, H1_, D1_)
            W2_, H2_, D2_ = int(self.grid_size_occ[1][0]/self.coarse_ratio), int(self.grid_size_occ[1][1]/self.coarse_ratio), int(self.grid_size_occ[1][2]/self.coarse_ratio)
            logits_sk = logits_sk.permute(0, 2, 1)
            B2_, C2_, N2_ = logits_sk.shape
            logits_sk = logits_sk.reshape(B2_, C2_, W2_, H2_, D2_)

        if return_loss:
            if not self.dual:
                # resize gt
                ratio = voxel_label.shape[2] // H
                if ratio != 1:
                    voxel_label_coarse = voxel_label.reshape(B, W, ratio, H, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, W, H, D, ratio**3)
                    empty_mask = voxel_label_coarse.sum(-1) == 0
                    voxel_label_coarse = voxel_label_coarse.to(torch.int64)
                    occ_space = voxel_label_coarse[~empty_mask]
                    occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
                    voxel_label_coarse[~empty_mask] = occ_space
                    voxel_label_coarse = torch.mode(voxel_label_coarse, dim=-1)[0]
                    voxel_label_coarse[voxel_label_coarse<0] = 255
                    voxel_label_coarse = voxel_label_coarse.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = logits.float()
                        loss_dict = {}

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights.type_as(logits), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)

                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = logits.float()
                        loss_dict = {}

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights.type_as(logits), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)

            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                if flag.item() == 1:
                    ratio = voxel_label.shape[2] // H1_
                elif flag.item() == 2:
                    ratio = voxel_label.shape[2] // H2_

                if ratio != 1:
                    if flag.item() == 1:
                        voxel_label_coarse = voxel_label.reshape(B1_, W1_, ratio, H1_, ratio, D1_, ratio).permute(0,1,3,5,2,4,6).reshape(B1_, W1_, H1_, D1_, ratio**3)
                    if flag.item() == 2:
                        voxel_label_coarse = voxel_label.reshape(B2_, W2_, ratio, H2_, ratio, D2_, ratio).permute(0,1,3,5,2,4,6).reshape(B2_, W2_, H2_, D2_, ratio**3)
                    empty_mask = voxel_label_coarse.sum(-1) == 0
                    voxel_label_coarse = voxel_label_coarse.to(torch.int64)
                    occ_space = voxel_label_coarse[~empty_mask]
                    occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
                    voxel_label_coarse[~empty_mask] = occ_space
                    voxel_label_coarse = torch.mode(voxel_label_coarse, dim=-1)[0]
                    voxel_label_coarse[voxel_label_coarse<0] = 255
                    voxel_label_coarse = voxel_label_coarse.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        if flag.item() == 1:
                            logits = logits_nu.float()
                            loss_dict = {}

                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights_nu.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)
                        if flag.item() == 2:
                            logits = logits_sk.float()
                            loss_dict = {}

                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights_sk.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        if flag.item() == 1:
                            logits = logits_nu.float()
                            loss_dict = {}
                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label, self.class_weights_nu.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)
                        if flag.item() == 2:
                            logits = logits_sk.float()
                            loss_dict = {}
                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label, self.class_weights_sk.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)
            else:
                if isinstance(voxel_label, list):
                    voxel_label_nu = torch.cat(voxel_label[::2], dim=0)
                    voxel_label_sk = torch.cat(voxel_label[1::2], dim=0)
                else:
                    voxel_label_nu = voxel_label[::2]
                    voxel_label_sk = voxel_label[1::2]
                ratio_nu = voxel_label_nu.shape[2] // H1_
                ratio_sk = voxel_label_sk.shape[2] // H2_
                loss_dict = {}
                if ratio_nu != 1:
                    voxel_label_coarse_nu = voxel_label_nu.reshape(B1_, W1_, ratio_nu, H1_, ratio_nu, D1_, ratio_nu).permute(0,1,3,5,2,4,6).reshape(B1_, W1_, H1_, D1_, ratio_nu**3)
                    empty_mask_nu = voxel_label_coarse_nu.sum(-1) == 0
                    voxel_label_coarse_nu = voxel_label_coarse_nu.to(torch.int64)
                    occ_space_nu = voxel_label_coarse_nu[~empty_mask_nu]
                    occ_space_nu[occ_space_nu==0] = -torch.arange(len(occ_space_nu[occ_space_nu==0])).to(occ_space_nu.device) - 1
                    voxel_label_coarse_nu[~empty_mask_nu] = occ_space_nu
                    voxel_label_coarse_nu = torch.mode(voxel_label_coarse_nu, dim=-1)[0]
                    voxel_label_coarse_nu[voxel_label_coarse_nu<0] = 255
                    voxel_label_coarse_nu = voxel_label_coarse_nu.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_nu = logits_nu.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits_nu, voxel_label_coarse_nu, self.class_weights_nu.type_as(logits_nu), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits_nu, voxel_label_coarse_nu, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits_nu, voxel_label_coarse_nu, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_nu, dim=1), voxel_label_coarse_nu, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_nu = logits_nu.float()
                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits_nu, voxel_label_nu, self.class_weights_nu.type_as(logits_nu), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits_nu, voxel_label_nu, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits_nu, voxel_label_nu, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_nu, dim=1), voxel_label_nu, ignore=255)

                if ratio_sk != 1:
                    voxel_label_coarse_sk = voxel_label_sk.reshape(B2_, W2_, ratio_sk, H2_, ratio_sk, D2_, ratio_sk).permute(0,1,3,5,2,4,6).reshape(B2_, W2_, H2_, D2_, ratio_sk**3)
                    empty_mask_sk = voxel_label_coarse_sk.sum(-1) == 0
                    voxel_label_coarse_sk = voxel_label_coarse_sk.to(torch.int64)
                    occ_space_sk = voxel_label_coarse_sk[~empty_mask_sk]
                    occ_space_sk[occ_space_sk==0] = -torch.arange(len(occ_space_sk[occ_space_sk==0])).to(occ_space_sk.device) - 1
                    voxel_label_coarse_sk[~empty_mask_sk] = occ_space_sk
                    voxel_label_coarse_sk = torch.mode(voxel_label_coarse_sk, dim=-1)[0]
                    voxel_label_coarse_sk[voxel_label_coarse_sk<0] = 255
                    voxel_label_coarse_sk = voxel_label_coarse_sk.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_sk = logits_sk.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits_sk, voxel_label_coarse_sk, self.class_weights_sk.type_as(logits_sk), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits_sk, voxel_label_coarse_sk, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits_sk, voxel_label_coarse_sk, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_sk, dim=1), voxel_label_coarse_sk, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_sk = logits_sk.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits_sk, voxel_label_sk, self.class_weights_sk.type_as(logits_sk), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits_sk, voxel_label_sk, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits_sk, voxel_label_sk, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_sk, dim=1), voxel_label_sk, ignore=255)
            return loss_dict
        else:
            if not self.dual:
                B_, W_, H_, D_ = voxel_label.shape
                pred = F.interpolate(logits, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels': pred,
                }
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    B_, W_, H_, D_ = voxel_label.shape
                    pred_nu = F.interpolate(logits_nu, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                    res = {
                        'output_voxels_1': pred_nu,
                    }
                if flag.item() == 2:
                    B_, W_, H_, D_ = voxel_label.shape
                    pred_sk = F.interpolate(logits_sk, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                    res = {
                        'output_voxels_2': pred_sk,
                    }
            else:
                voxel_label_nu = voxel_label[::2]
                voxel_label_sk = voxel_label[1::2]
                B1_, W1_, H1_, D1_ = voxel_label_nu.shape
                pred_nu = F.interpolate(logits_nu, size=[W1_, H1_, D1_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels_1': pred_nu,
                }
                B2_, W2_, H2_, D2_ = voxel_label_sk.shape
                pred_sk = F.interpolate(logits_sk, size=[W2_, H2_, D2_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels_2': pred_sk,
                }
            return res

class SensorEncoder(nn.Module):
    """Encode raw sensor parameters into an embedding vector."""
    def __init__(self, in_dim=5, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sensor_params):
        # sensor_params: [B, in_dim]
        return self.mlp(sensor_params)  # [B, emb_dim]

class SensorHead(nn.Module):
    """Regress sensor parameters from a global feature vector."""
    def __init__(self, feat_dim, out_dim=5, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, global_feat):
        # global_feat: [B, feat_dim]
        return self.mlp(global_feat)  # [B, out_dim]

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation. A sensor embedding e produces (gamma, beta)
    that affinely transform the feature map channel-wise:

        F' = (1 + gamma(e)) * F + beta(e)

    Accepts x of shape [B, C, L] or [B, C, H, W].
    """
    def __init__(self, sensor_emb_dim, num_channels, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(sensor_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        self.num_channels = num_channels

    def forward(self, x, sensor_emb):
        """
        x: [B, C, L] or [B, C, H, W]
        sensor_emb: [B, sensor_emb_dim]
        """
        B, C = x.shape[0], x.shape[1]
        assert C == self.num_channels, "FiLM channel count must match the input"

        gamma_beta = self.fc(sensor_emb)           # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # each [B, C]

        if x.dim() == 3:
            # [B, C, L]
            gamma = gamma.unsqueeze(-1)  # [B, C, 1]
            beta = beta.unsqueeze(-1)
        elif x.dim() == 4:
            # [B, C, H, W]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError(f"unsupported input rank: {x.shape}")

        return x * (1 + gamma) + beta

@VOXEL_ENCODERS.register_module()
class TPVAggregator_Seg(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=20,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False,
        loss_weight=[1,1], dual = False,
        # Sensor-conditioning hyper-parameters (S-FiLM).
        sensor_in_dim=5,
        sensor_emb_dim=256,
        sensor_loss_weight=0.6,
        consistency = False,
        consistency_loss_weight = 0.0,
        dense_train = False,
        d2skd = False,
        outside_d2skd = False,
        sensor_film = False,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.loss_weight = loss_weight
        self.dual = dual
        self.consistency = consistency
        self.dense_train = dense_train
        self.d2skd = d2skd
        self.outside_d2skd = outside_d2skd
        self.consistency_loss_weight = consistency_loss_weight
        out_dims = in_dims if out_dims is None else out_dims
        if self.outside_d2skd:
            self.spectral_distiller = TPVHighFreqSpectralDistiller()
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )
        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.lovasz_loss_func = lovasz_softmax

        # Running per-class frequencies used by the seesaw re-weighting.
        self.register_buffer("class_counts",
            torch.ones(self.classes)  # start at 1 to avoid division by zero
        )
        # Sensor encoder / regression head / FiLM layer.
        self.sensor_in_dim = sensor_in_dim
        self.sensor_emb_dim = sensor_emb_dim
        self.sensor_loss_weight = sensor_loss_weight
        self.sensor_film = sensor_film
        if not self.dense_train and self.sensor_film:
            # Raw sensor vector -> embedding.
            self.sensor_encoder = SensorEncoder(
                in_dim=sensor_in_dim,
                emb_dim=sensor_emb_dim,
                hidden_dim=sensor_emb_dim
            )
            # Predict sensor parameters from the pooled global feature; its width
            # is in_dims, i.e. the channel count of the fused TPV feature.
            self.sensor_head = SensorHead(
                feat_dim=in_dims,
                out_dim=sensor_in_dim,
                hidden_dim=sensor_emb_dim
            )
            # Modulates the fused feature [B, C, L].
            self.film_layer = FiLMLayer(
                sensor_emb_dim=sensor_emb_dim,
                num_channels=in_dims,
                hidden_dim=sensor_emb_dim
            )
        # FiLM mixing coefficient: 0 = ground-truth sensor vector, 1 = predicted.
        self.film_alpha = 0.0
        # Initialised for the first curriculum phase (epochs 0-8); the curriculum hook
        # refreshes these in before_train_epoch. Phase 0 ranges are beams(32,64),
        # height(1.7,1.9), H_res(900,2400), theta_low(-25,-20), theta_up(0,10), with
        # mean=(a+b)/2 and std=(b-a)/√12.
        self.register_buffer('sensor_mean', torch.tensor([48.0, 1.8, 1650.0, -22.5, 5.0], dtype=torch.float32))
        self.register_buffer('sensor_std', torch.tensor([9.24, 0.058, 433.0, 1.44, 2.89], dtype=torch.float32))

    # Lets a training hook ramp alpha from 0 to 1 across epochs.
    def set_film_alpha(self, alpha: float):
        self.film_alpha = float(alpha)

    def update_sensor_stats(self, mean, std):
        self.sensor_mean = mean.to(self.sensor_mean.device)
        self.sensor_std = std.to(self.sensor_std.device)

    def seesaw_loss(self, logits, labels, p=0.8, eps=1e-6, ignore_index=0):
        """Seesaw loss over arbitrary spatial shapes.

        Args:
            logits: any shape, provided dim=1 is the channel axis (C).
            labels: semantic labels whose spatial dims match ``logits``.
        """
        # ---- reshape: (B, C, ...) → (N, C) ----
        B, C = logits.shape[:2]
        logits_flat = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        # logits_flat shape = (N, C)
        # ---- labels reshape: (B, ...) → (N,) ----
        labels_flat = labels.reshape(-1)
        # ---- mask ignore_index ----
        valid = labels_flat != ignore_index
        if not valid.any():
            # No valid point in this batch.
            return logits.sum() * 0
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
        # ---- one-hot ----
        targets = F.one_hot(labels_flat, num_classes=C).float()
        def distribution_agnostic_seesaw_loss(logits, targets, p=0.8, eps=1e-6):
            # Accumulate class frequencies.
            batch_class_count = targets.sum(dim=0)
            self.class_counts += batch_class_count
            cc = self.class_counts
            # Pairwise mitigation matrix.
            conditions = cc[:, None] > cc[None, :]
            trues = (cc[None, :] / cc[:, None]).pow(p)
            falses = torch.ones_like(trues)
            s = torch.where(conditions, trues, falses)
            # Shift logits for numerical stability.
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            numerator = torch.exp(logits)
            denominator = (
                (1 - targets)[:, None, :] * s[None, :, :] * torch.exp(logits)[:, None, :]
            ).sum(dim=-1) + torch.exp(logits)
            sigma = numerator / (denominator + eps)
            loss = (-targets * torch.log(sigma + eps)).sum(dim=-1)
            return loss.mean()
        # ---- seesaw loss ----
        return distribution_agnostic_seesaw_loss(logits_flat, targets, p=p, eps=eps)

    def forward(self, tpv_list, points=None, voxel_label=None, point_labels=None, dataset_flag=None, return_loss=True, sensor_vec=None, logits_vox_t=None, logits_pts_t=None, skip_consistency=False, return_logits_for_consistency=False):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w

        Args:
            logits_vox_t: teacher voxel predictions, used for distillation.
            logits_pts_t: teacher point predictions, used for distillation.
            skip_consistency: skip the internal consistency branch when the caller
                already evaluates it group by group.
        """
        # In cross-density distillation the aggregator only emits predictions.
        if self.d2skd:
            return_loss = False

        # The consistency branch only runs during training (sensor_vec is not None).
        if self.consistency and sensor_vec is not None and not skip_consistency:
            if isinstance(points, list):
                points_list = points
            else:
                # Split a batched tensor so that each entry is one sample.
                if isinstance(points, torch.Tensor) and points.dim() >= 2 and points.size(0) > 1:
                    points_list = [points[i:i+1] for i in range(points.size(0))]
                else:
                    points_list = [points]
            num_samples = len(points_list)

            # Run sample by sample, collecting voxel logits for the consistency term.
            losses, voxel_logits, point_logits = [], [], []
            sensor_outs = []
            voxel_labels_for_consistency = []

            def _get(container, idx):
                if container is None:
                    return None
                if isinstance(container, list):
                    return container[idx]
                if isinstance(container, torch.Tensor) and container.dim() > 0 and container.size(0) > idx:
                    return container[idx:idx+1]
                return container

            for i in range(num_samples):
                # Views arrive in pairs (0,1), (2,3), (4,5), ... The first sample of
                # each pair is the main view and carries the supervised losses; the
                # second contributes only to the consistency regulariser.
                is_main_view = (i % 2 == 0)

                tpv_xy_i, tpv_yz_i, tpv_zx_i = [
                    feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in tpv_list
                ]
                pts_i = points_list[i]
                # Align the point batch with the per-view TPV features (batch size 1).
                if isinstance(pts_i, torch.Tensor) and pts_i.dim() >= 2 and pts_i.size(0) > 1:
                    pts_i = pts_i[0:1]
                elif isinstance(pts_i, torch.Tensor) and pts_i.dim() == 2:
                    pts_i = pts_i.unsqueeze(0)
                voxel_label_i = _get(voxel_label, i)
                point_labels_i = _get(point_labels, i)
                sensor_vec_i = _get(sensor_vec, i)
                dataset_flag_i = _get(dataset_flag, i)

                # Teacher predictions are only consumed by the main view.
                logits_vox_t_i = None
                logits_pts_t_i = None
                if is_main_view:
                    if logits_vox_t is not None:
                        if isinstance(logits_vox_t, list):
                            logits_vox_t_i = logits_vox_t[i] if i < len(logits_vox_t) else None
                        else:
                            logits_vox_t_i = logits_vox_t[i:i+1] if logits_vox_t is not None else None
                    if logits_pts_t is not None:
                        if isinstance(logits_pts_t, list):
                            logits_pts_t_i = logits_pts_t[i] if i < len(logits_pts_t) else None
                        else:
                            logits_pts_t_i = logits_pts_t[i:i+1] if logits_pts_t is not None else None

                if is_main_view:
                    # Main view: full supervised losses.
                    out_i = self._forward_single(
                        [tpv_xy_i, tpv_yz_i, tpv_zx_i],
                        points=pts_i,
                        voxel_label=voxel_label_i,
                        point_labels=point_labels_i,
                        dataset_flag=dataset_flag_i,
                        return_loss=return_loss,
                        sensor_vec=sensor_vec_i,
                        return_logits=True,
                        logits_vox_t=logits_vox_t_i,
                        logits_pts_t=logits_pts_t_i,
                    )

                    if return_loss:
                        loss_i, vox_i, pts_out_i, sensor_out_i = out_i
                        losses.append(loss_i)
                        voxel_logits.append(vox_i)
                        point_logits.append(pts_out_i)
                        sensor_outs.append(sensor_out_i)
                        voxel_labels_for_consistency.append(voxel_label_i)
                    else:
                        vox_i, pts_out_i, sensor_out_i = out_i
                        voxel_logits.append(vox_i)
                        point_logits.append(pts_out_i)
                        sensor_outs.append(sensor_out_i)
                else:
                    # Auxiliary view: no supervised loss, but CBA-CL propagates
                    # gradients through both views, so no torch.no_grad here.
                    out_i = self._forward_single(
                        [tpv_xy_i, tpv_yz_i, tpv_zx_i],
                        points=pts_i,
                        voxel_label=None,          # no supervised voxel loss
                        point_labels=None,         # no supervised point loss
                        dataset_flag=dataset_flag_i,
                        return_loss=False,         # logits only
                        sensor_vec=sensor_vec_i,
                        return_logits=True,
                        logits_vox_t=None,
                        logits_pts_t=None,
                    )
                    vox_i, pts_out_i, sensor_out_i = out_i
                    voxel_logits.append(vox_i)
                    point_logits.append(pts_out_i)
                    sensor_outs.append(sensor_out_i)

            if not return_loss:
                return voxel_logits, point_logits, sensor_outs

            if len(losses) == 0:
                agg_loss = {}
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                all_keys = set()
                for ld in losses:
                    all_keys.update(ld.keys())
                all_keys.add('loss_consistency')

                # Sort the keys so every rank inserts them in the same order; otherwise
                # dict.values() may be misordered across ranks and corrupt all_reduce.
                sorted_keys = sorted(list(all_keys))

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                agg_loss = {}
                device = losses[0][list(losses[0].keys())[0]].device

                for k in sorted_keys:
                    if k == 'loss_consistency':
                        # Placeholder, overwritten below; requires_grad keeps DDP happy.
                        agg_loss[k] = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        sample_tensor = None
                        for ld in losses:
                            if k in ld:
                                sample_tensor = ld[k]
                                break
                        if sample_tensor is not None:
                            agg_loss[k] = sample_tensor.new_zeros(())
                        else:
                            agg_loss[k] = torch.zeros((), device=device)

                for ld in losses:
                    for k in sorted_keys:
                        if k != 'loss_consistency' and k in ld:
                            agg_loss[k] = agg_loss[k] + ld[k]

                # Normalise by the number of main views (i.e. of losses), not by the
                # total view count, so repeated sampling does not shrink the loss.
                for k in agg_loss:
                    if k != 'loss_consistency':
                        agg_loss[k] = agg_loss[k] / (len(losses) + 1e-8)
            # Softmax temperature for the consistency term. T > 1 flattens the
            # distributions and emphasises agreement on the overall shape.
            T = 1.5
            # Always define the consistency key so every rank reduces the same dict.
            if len(voxel_logits) > 0:
                device = voxel_logits[0].device
                loss_consistency_val = voxel_logits[0].sum() * 0.0
            else:
                # No sample: emit a differentiable zero so DDP does not flag unused parameters.
                if 'device' not in locals():
                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                loss_consistency_val = torch.tensor(0.0, device=device, requires_grad=True)
            if num_samples == 1 and len(voxel_logits) > 0:
                voxel_logits.append(voxel_logits[0].detach())
                voxel_labels_for_consistency.append(voxel_labels_for_consistency[0])
                num_samples = 2
            if len(voxel_logits) > 1:
                cl_weight = float(self.consistency_loss_weight)
                cl_total = loss_consistency_val.new_zeros(())
                pairs = 0
                # Compare within each pair only: (0,1), (2,3), (4,5), ...
                for i in range(0, len(voxel_logits) - 1, 2):
                    j = i + 1
                    # Only the main view carries a label; the pair shares it.
                    group_idx = i // 2
                    vi = voxel_labels_for_consistency[group_idx] if group_idx < len(voxel_labels_for_consistency) else None
                    if vi is None:
                        continue
                    cl_total = cl_total + cl_weight * _cb_adc_consistency_loss(
                        voxel_logits[i],
                        voxel_logits[j],
                        vi,
                        num_classes=self.classes,
                        T=T,
                        ignore_index=0,
                        outer_scale=40.0,
                    )
                    pairs += 1

                if pairs > 0:
                    loss_consistency_val = loss_consistency_val + cl_total / pairs

            agg_loss['loss_consistency'] = loss_consistency_val
            return agg_loss

        # Collapse a list of point tensors into a single batched tensor.
        if isinstance(points, list):
            if len(points) == 0:
                raise ValueError("points list is empty")
            if all(pl.shape == points[0].shape for pl in points):
                points = torch.cat([pl[None, ...] if pl.dim() == 2 else pl for pl in points], dim=0)
            else:
                points = points[0].unsqueeze(0) if points[0].dim() == 2 else points[0]

        return self._forward_single(
            tpv_list, points, voxel_label, point_labels, dataset_flag, return_loss, sensor_vec,
            return_logits=return_logits_for_consistency, logits_vox_t=logits_vox_t, logits_pts_t=logits_pts_t
        )

    def _forward_single(self, tpv_list, points, voxel_label, point_labels, dataset_flag, return_loss, sensor_vec, return_logits=False, logits_vox_t=None, logits_pts_t=None):
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)
        bs, c, _, _ = tpv_hw.shape

        # Broadcast batch-1 labels produced by the data pipeline up to the feature batch.
        if voxel_label is not None and voxel_label.shape[0] != bs:
            if voxel_label.shape[0] == 1:
                voxel_label = voxel_label.expand(bs, *voxel_label.shape[1:])
            else:
                raise ValueError(f"voxel_label batch ({voxel_label.shape[0]}) != feature batch ({bs})")
        if point_labels is not None and point_labels.shape[0] != bs:
            if point_labels.shape[0] == 1:
                point_labels = point_labels.expand(bs, *point_labels.shape[1:])
            else:
                raise ValueError(f"point_labels batch ({point_labels.shape[0]}) != feature batch ({bs})")

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                mode='bilinear'
            )

        # points: (bs, n, 3); a bare (n, 3) tensor gets a batch axis.
        if points.dim() == 2:
            points = points.unsqueeze(0)
        points_bs, n, _ = points.shape

        # Reconcile the point batch with the TPV feature batch.
        if points_bs > bs:
            points = points[0:1]
            points_bs = 1
        elif points_bs < bs:
            if points_bs == 1:
                points = points.expand(bs, -1, -1)
                points_bs = bs
            else:
                raise ValueError(f"points batch ({points_bs}) < tpv_hw batch ({bs}) and cannot be expanded")

        points = points.unsqueeze(1)  # bs,1,n,3
        voxels = torch.unique(torch.floor(points), dim=2)
        # Keep batch dimension; remove only the singleton channel dimension
        voxels_ind = deepcopy(voxels).type(torch.long).squeeze(1)

        voxels += 0.5
        points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1
        sample_loc = points[:, :, :, [0, 1]]
        tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc, padding_mode="border").squeeze(2) # bs, c, n
        sample_loc = points[:, :, :, [1, 2]]
        tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc, padding_mode="border").squeeze(2)
        sample_loc = points[:, :, :, [2, 0]]
        tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc, padding_mode="border").squeeze(2)

        voxels[..., 0] = voxels[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        voxels[..., 1] = voxels[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        voxels[..., 2] = voxels[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

        sample_loc_vox = voxels[:, :, :, [0, 1]]
        tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
        sample_loc_vox = voxels[:, :, :, [1, 2]]
        tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
        sample_loc_vox = voxels[:, :, :, [2, 0]]
        tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
        fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox

        fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
        fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
        # ---------------- S-FiLM: sensor head + FiLM modulation ----------------
        sensor_loss = fused.new_tensor(0.0)
        pred_sensor_output = None                    # returned at test time
        sensor_mae = fused.new_tensor(0.0)           # MAE in raw units
        sensor_mae_norm = fused.new_tensor(0.0)      # MAE in normalised space
        sensor_mae_per_dim = fused.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        if hasattr(self, 'sensor_head') and hasattr(self, 'sensor_encoder') and not self.dense_train:
            # Training: ground-truth sensor_vec is available.
            if sensor_vec is not None:
                # Normalise the ground-truth sensor parameters.
                sensor_gt = sensor_vec.to(self.sensor_mean.device).float()        # [B, D]
                if sensor_gt.dim() == 1:  # shape is [D]
                    sensor_gt = sensor_gt.unsqueeze(0)  # → [1, D]
                sensor_gt_norm = (sensor_gt - self.sensor_mean) / self.sensor_std # [B, D]
                # Pool fused [B, C, L] into a global [B, C] descriptor.
                global_feat = F.adaptive_avg_pool1d(fused, output_size=1).squeeze(-1)  # [B, C]
                # Regress the normalised parameters.
                pred_sensor_norm = self.sensor_head(global_feat)  # [B, D]
                # Denormalise back to raw units for the reported error.
                pred_sensor_raw = pred_sensor_norm * self.sensor_std + self.sensor_mean  # [B, D]
                abs_err = (pred_sensor_raw - sensor_gt).abs()
                sensor_mae = abs_err.mean()
                sensor_mae_norm = (pred_sensor_norm - sensor_gt_norm).abs().mean()
                sensor_mae_per_dim = abs_err.mean(dim=0)  # [D]
                # Regress in normalised space so per-dimension gradients stay balanced.
                per_dim_loss = F.smooth_l1_loss(pred_sensor_norm, sensor_gt_norm, reduction='none')  # [B, D]
                sensor_loss = per_dim_loss.mean()
                # Progressively blend ground-truth and predicted parameters.
                alpha = float(self.film_alpha)
                alpha = max(0.0, min(1.0, alpha))
                sensor_mix_norm = (1.0 - alpha) * sensor_gt_norm + alpha * pred_sensor_norm.detach()
                sensor_emb = self.sensor_encoder(sensor_mix_norm)  # [B, sensor_emb_dim]
                fused = self.film_layer(fused, sensor_emb)
                pred_sensor_output = pred_sensor_raw  # [B, D]
            # Inference: no ground truth, condition FiLM on the prediction itself.
            else:
                global_feat = fused.mean(dim=-1)  # [B, C]
                pred_sensor_norm = self.sensor_head(global_feat)  # [B, D]
                pred_sensor_raw = pred_sensor_norm * self.sensor_std + self.sensor_mean  # [B, D]
                sensor_emb = self.sensor_encoder(pred_sensor_norm)  # [B, sensor_emb_dim]
                fused = self.film_layer(fused, sensor_emb)
                pred_sensor_output = pred_sensor_raw  # [B, D]
        # ---------------- Segmentation head ----------------
        fused = fused.permute(0, 2, 1)
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            logits = self.classifier(fused)
        logits = logits.permute(0, 2, 1)

        # Preserve batch dimension; shape: (bs, whz, classes)
        feats_vox = logits[:, :, :(-n)].permute(0,2,1)
        output_shape = [int(self.tpv_w*self.scale_w), int(self.tpv_h*self.scale_h), int(self.tpv_z*self.scale_z), feats_vox.shape[-1]]
        # Scatter per batch to build (bs, C, W, H, Z)
        batch_logits_vox = []
        for b in range(bs):
            scattered = scatter_nd(indices=voxels_ind[b], updates=feats_vox[b], shape=output_shape)  # (W,H,Z,C)
            batch_logits_vox.append(scattered.permute(3,0,1,2))  # (C,W,H,Z)
        logits_vox = torch.stack(batch_logits_vox, dim=0)
        logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)

        if return_loss:
            # Always return fixed keys with Tensor values to keep DDP collectives aligned
            zero_point = logits_pts.sum() * 0
            zero_voxel = logits_vox.sum() * 0

            point_loss = zero_point
            voxel_loss = zero_voxel
            seesaw_loss_points = zero_point
            seesaw_loss_voxels = zero_voxel

            # Point-level loss
            if point_labels is not None:
                valid_mask = point_labels != 0  # Ignore label 0
                if valid_mask.any():
                    point_loss = self.lovasz_loss_func(torch.softmax(logits_pts, dim=1), point_labels, ignore=0)
                    seesaw_loss_points = self.seesaw_loss(logits_pts, point_labels, ignore_index=0)

            # Voxel-level loss
            if voxel_label is not None:
                valid_voxel_mask = voxel_label != 0
                if valid_voxel_mask.any():
                    # Lovasz-Softmax replaces CE as the main voxel-level loss.
                    voxel_probs = torch.softmax(logits_vox, dim=1)
                    voxel_loss = self.lovasz_loss_func(voxel_probs, voxel_label, ignore=0)
                    seesaw_loss_voxels = self.seesaw_loss(logits_vox, voxel_label, ignore_index=0)

            sensor_loss_term = sensor_loss * self.sensor_loss_weight

            # Spectral distillation against the teacher, when one is provided.
            distill_loss_vox = zero_voxel
            if logits_vox_t is not None and not self.dense_train:
                distill_loss_vox = self.spectral_distiller(logits_vox, logits_vox_t)

            loss_dict = {
                'loss_point_lovasz': self.loss_weight[0] * point_loss,
                'loss_point_seesaw': self.loss_weight[0] * seesaw_loss_points,
                'loss_voxel_ce': self.loss_weight[1] * 0.5 * voxel_loss,
                'loss_voxel_seesaw': self.loss_weight[1] * seesaw_loss_voxels,
                'loss_sensor': sensor_loss_term,
                'sensor_mae_total': sensor_mae_norm.detach(),
            }

            if distill_loss_vox != 0:
                loss_dict['loss_distill_vox'] = 500.0 * distill_loss_vox
            # Per-dimension sensor MAE, logged for the curriculum hook.
            if sensor_mae_per_dim is not None:
                dim_names = ['beam', 'height', 'H_res', 'theta_low', 'theta_up']
                for i, name in enumerate(dim_names):
                    loss_dict[f'sensor_mae_{name}'] = sensor_mae_per_dim[i].detach()

            if return_logits:
                return loss_dict, logits_vox, logits_pts, pred_sensor_output
            return loss_dict

        return logits_vox, logits_pts, pred_sensor_output
