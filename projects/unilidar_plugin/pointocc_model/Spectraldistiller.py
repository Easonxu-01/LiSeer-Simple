'''
Author: EASON XU
Date: 2026-01-12 15:43:49
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-01-24 23:11:58
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/pointocc_model/Spectraldistiller.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

class TPVLowFreqSpectralDistiller(nn.Module):
    """
    SI-STJD 低频蒸馏融合版：集成多尺度扩散响应与狄利克雷能量对齐。
    
    理论映射：
    - TPV 表征被视为规则 2D 网格图上的信号 F。
    - 使用稳定的扩散算子 (I - alpha L) 提取多尺度低频响应 G_b。
    - L 是归一化图拉普拉斯算子，描述信号在空间上的“震荡”。
    - (I - alpha L) 是低通滤波器，保留宏观语义（几何骨架）。
    - 蒸馏损失 L = L_resp (形态一致性) + L_eng (几何平滑能量一致性)。
    """

    def __init__(
        self,
        num_scales: int = 8,            # B: 扩散尺度的数量（滤波器组大小）
        alpha: float = 0.25,            # 扩散步长，对应离散化的时间步长 t
        eta: Optional[List[float]] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],   # 各个尺度的损失权重
        lambda_energy: float = 0.5,     # 狄利克雷能量对齐的相对权重
        num_moments: int = 8,
        lambda_moment: float = 0.5,
        eps: float = 1e-6,
        detach_teacher: bool = True,
        pad_mode: str = "replicate"     # 边界处理：使用 replicate 保持流形边界连续性
    ):
        super().__init__()
        self.num_scales = num_scales
        self.alpha = alpha
        self.lambda_energy = lambda_energy
        self.eps = eps
        self.detach_teacher = detach_teacher
        self.pad_mode = pad_mode
        self.num_moments = num_moments
        self.lambda_moment = lambda_moment
        
        # 默认权重：低频（b 越大）的权重略高，因为低频包含更稳健的语义骨架
        if eta is None:
            eta = [0.5 + 0.25 * b for b in range(self.num_scales)]
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))

        # [理论核心] 定义归一化图拉普拉斯卷积核 (L = I - NeighborAvg)
        # 对应图理论中规则网格的拉普拉斯算子实现
        K_lap = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        K_lap[0, 0, 1, 1] = 1.0
        K_lap[0, 0, 0, 1] = -0.25
        K_lap[0, 0, 1, 0] = -0.25
        K_lap[0, 0, 1, 2] = -0.25
        K_lap[0, 0, 2, 1] = -0.25
        self.register_buffer("K_lap", K_lap)

    def _apply_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """计算 L*x (拉普拉斯算子响应)"""
        B, C, H, W = x.shape
        k = self.K_lap.expand(C, 1, 3, 3)
        x_pad = F.pad(x, (1, 1, 1, 1), mode=self.pad_mode)
        # 使用 depthwise 卷积模拟图信号处理
        return F.conv2d(x_pad, k, bias=None, stride=1, padding=0, groups=C)

    def _diffuse_step(self, x: torch.Tensor) -> torch.Tensor:
        """执行一步热传导扩散: D(x) = x - alpha * L(x)"""
        return x - self.alpha * self._apply_laplacian(x)

    def _get_dirichlet_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算每通道的狄利克雷能量: E = tr(x^T * L * x)
        这是衡量几何平滑性的本征物理量。
        """
        Lx = self._apply_laplacian(x)
        # 在空间维度 (H, W) 求内积
        energy = torch.sum(x * Lx, dim=(2, 3)) # [B, C]
        return energy

    def _spectral_moments(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute per-channel spectral moments:
            E_k(x) = sum_u x(u) * (L^k x)(u)  -> [B, C], for k=1..K

        Implementation:
          - iterative application of Laplacian: Lx, L^2 x, ...
          - inner product with original x (not with L^k x) matches tr(x^T L^k x)
        """
        moments: List[torch.Tensor] = []
        Lkx = x
        for k in range(1, self.num_moments + 1):
            Lkx = self._apply_laplacian(Lkx)  # now L^k x
            Ek = torch.sum(x * Lkx, dim=(2, 3))  # [B, C]
            Ek = torch.clamp(Ek, min=0.0)
            moments.append(Ek)
        return moments

    def _normalize_response(self, x: torch.Tensor) -> torch.Tensor:
        """
        响应形态归一化：
        消除稀疏采样导致的绝对幅值偏差，只对齐信号的“空间分布形状”。
        """
        denom = torch.sqrt(torch.sum(x * x, dim=(1, 2, 3), keepdim=True) + self.eps)
        return x / denom

    def _match_view(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """对单个 TPV 视图执行蒸馏"""
        # 初始状态
        gs, gt = s, t
        loss = s.new_tensor(0.0)
        
        for b in range(self.num_scales):
            # 1. 响应形态对齐 (低通形态)
            ns = self._normalize_response(gs)
            nt = self._normalize_response(gt)
            loss_resp = torch.mean((ns - nt) ** 2)

            # 2. 能量谱对齐 (狄利克雷一致性)
            Es = self._get_dirichlet_energy(gs)
            Et = self._get_dirichlet_energy(gt)
            # 对通道维度进行归一化，学习特征间的相对能量分布
            Es_n = F.normalize(Es, p=2, dim=1)
            Et_n = F.normalize(Et, p=2, dim=1)
            loss_eng = F.mse_loss(Es_n, Et_n)

            # 累加损失，并结合当前尺度的权重 eta
            loss += self.eta[b].to(loss.dtype) * (loss_resp + self.lambda_energy * loss_eng)

            # 准备下一尺度的扩散 (通过迭代增加扩散半径)
            if b < self.num_scales - 1:
                gs = self._diffuse_step(gs)
                gt = self._diffuse_step(gt)
            
        mom_s = self._spectral_moments(s)  # list of [B,C]
        mom_t = self._spectral_moments(t)  # list of [B,C]

        loss_mom = s.new_tensor(0.0)
        for k in range(self.num_moments):
            ms = F.normalize(mom_s[k], p=2, dim=1)  # normalize across channels -> compare energy distribution
            mt = F.normalize(mom_t[k], p=2, dim=1)
            lk = F.mse_loss(ms, mt)
            loss_mom = loss_mom + lk

        # 归一化多尺度加权项，避免 num_scales 变化时整体梯度被额外放大
        loss = loss / (self.eta.sum().to(loss.dtype) + self.eps)
        # 对谱矩对齐按阶数取平均，避免 num_moments 增加时权重被线性放大
        loss = loss + (self.lambda_moment / float(self.num_moments)) * loss_mom
        return loss

    def forward(self, tpv_s: List[torch.Tensor], tpv_t: List[torch.Tensor]):
        """
        tpv_s/t: [tpv_xy, tpv_yz, tpv_zx]
        """
        if self.detach_teacher:
            tpv_t = [x.detach() for x in tpv_t]

        l_xy = self._match_view(tpv_s[0], tpv_t[0])
        l_yz = self._match_view(tpv_s[1], tpv_t[1])
        l_zx = self._match_view(tpv_s[2], tpv_t[2])

        return (3.0 * l_xy + l_yz + l_zx) / 3.0 #focous more on xy



class TPVHighFreqSpectralDistiller(nn.Module):
    """
    SI-STJD logits/high-frequency distillation for TPV 3-view tensors + ECC alignment.

    This class is NEW and does NOT modify your existing TPVLowFreqSpectralDistiller.

    Input format (same as your TPV lists):
      tpv_s/t: [tpv_xy, tpv_yz, tpv_zx]
      shapes:  [B,C,H,W], [B,C,H,Z], [B,C,W,Z]  (all treated as 2D grids)

    What it does:
      (1) High-frequency (high-pass) spectral-invariant distillation:
          - Use stable diffusion (low-pass) operator: G <- G - alpha * L(G)
          - Define high-pass residual at scale b: H_b = X - G_b
          - Align teacher/student high-pass responses across multiple scales b

      (2) ECC (Euler Characteristic Curve) alignment per-channel (class-wise):
          - For each channel c independently, build a soft superlevel set at thresholds tau
          - Compute soft Euler characteristic chi_c(tau) on the 2D grid
          - Align ECC curves for all channels and all thresholds

    Notes:
      - This module assumes channel dimension C corresponds to "per-class logit field"
        (so ECC is meaningful). If you feed intermediate features (C=256), ECC still runs
        but its semantic interpretation is not intended.
    """

    def __init__(
        self,
        num_scales: int = 8,                     # number of high-pass scales (b=0..B-1)
        alpha: float = 0.25,                     # diffusion step size
        eta: Optional[List[float]] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],       # per-scale weights
        lambda_energy: float = 0.5,              # weight for Dirichlet-energy alignment on high-pass
        num_moments: int = 8,                    # spectral moment orders computed on final high-pass
        lambda_moment: float = 0.5,              # weight for spectral-moment alignment
        eps: float = 1e-6,
        detach_teacher: bool = True,
        pad_mode: str = "replicate",

        # ECC
        lambda_ecc: float = 0.2,
        ecc_taus: Optional[List[float]] = None,  # threshold list for ECC curve
        ecc_kappa: float = 10.0,                 # softness of thresholding (bigger -> harder)
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.alpha = float(alpha)
        self.lambda_energy = float(lambda_energy)
        self.num_moments = int(num_moments)
        self.lambda_moment = float(lambda_moment)
        self.eps = float(eps)
        self.detach_teacher = bool(detach_teacher)
        self.pad_mode = pad_mode

        # scale weights
        if eta is None:
            eta = [0.5 + 0.25 * b for b in range(self.num_scales)]
        assert len(eta) == self.num_scales
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))

        # ECC settings
        self.lambda_ecc = float(lambda_ecc)
        if ecc_taus is None:
            ecc_taus = [0.2, 0.35, 0.5, 0.65, 0.8]
        self.register_buffer("ecc_taus", torch.tensor(ecc_taus, dtype=torch.float32))
        self.ecc_kappa = float(ecc_kappa)

        # Normalized grid Laplacian kernel (4-neighbor): L = I - NeighborAvg
        K_lap = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        K_lap[0, 0, 1, 1] = 1.0
        K_lap[0, 0, 0, 1] = -0.25
        K_lap[0, 0, 1, 0] = -0.25
        K_lap[0, 0, 1, 2] = -0.25
        K_lap[0, 0, 2, 1] = -0.25
        self.register_buffer("K_lap", K_lap)

    # ---------------------- Core operators ----------------------

    def _apply_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute L*x on regular grids.

        - If x is 4D [B,C,H,W], use the original 2D Laplacian (kept for backward compatibility).
        - If x is 5D [B,C,H,W,Z], use a 3D 6-neighbor Laplacian:
            L = I - (1/6) * sum_{6-neighbors}
        """
        if x.dim() == 4:
            # ---- original 2D branch ----
            B, C, H, W = x.shape
            k = self.K_lap.expand(C, 1, 3, 3).to(dtype=x.dtype)
            x_pad = F.pad(x, (1, 1, 1, 1), mode=self.pad_mode)
            return F.conv2d(x_pad, k, bias=None, stride=1, padding=0, groups=C)

        assert x.dim() == 5, f"Expect 5D logits [B,C,H,W,Z], got {x.shape}"
        B, C, H, W, Z = x.shape

        # ---- build (or reuse) 3D Laplacian kernel ----
        if not hasattr(self, "K_lap_3d"):
            K = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32, device=self.K_lap.device)
            K[0, 0, 1, 1, 1] = 1.0
            w = -1.0 / 6.0
            K[0, 0, 0, 1, 1] = w
            K[0, 0, 2, 1, 1] = w
            K[0, 0, 1, 0, 1] = w
            K[0, 0, 1, 2, 1] = w
            K[0, 0, 1, 1, 0] = w
            K[0, 0, 1, 1, 2] = w
            self.register_buffer("K_lap_3d", K)

        k3 = self.K_lap_3d.expand(C, 1, 3, 3, 3).to(dtype=x.dtype)
        # pad order for 5D is (Z_left, Z_right, W_left, W_right, H_left, H_right)
        x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode=self.pad_mode)
        return F.conv3d(x_pad, k3, bias=None, stride=1, padding=0, groups=C)

    def _diffuse_step(self, x: torch.Tensor) -> torch.Tensor:
        """One stable diffusion (low-pass) step: G <- (I - alpha L)G."""
        return x - self.alpha * self._apply_laplacian(x)

    def _normalize_response(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample Frobenius normalization (supports both 2D and 3D grids)."""
        if x.dim() == 4:
            denom = torch.sqrt(torch.sum(x * x, dim=(1, 2, 3), keepdim=True) + self.eps)
            return x / denom
        assert x.dim() == 5
        denom = torch.sqrt(torch.sum(x * x, dim=(1, 2, 3, 4), keepdim=True) + self.eps)
        return x / denom

    def _get_dirichlet_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-channel Dirichlet energy:
        E = sum_u x(u) * (Lx)(u)
        Supports:
        - 2D: x [B,C,H,W] -> E [B,C]
        - 3D: x [B,C,H,W,Z] -> E [B,C]
        """
        Lx = self._apply_laplacian(x)
        if x.dim() == 4:
            return torch.sum(x * Lx, dim=(2, 3))          # [B,C]
        assert x.dim() == 5
        return torch.sum(x * Lx, dim=(2, 3, 4))           # [B,C]

    def _spectral_moments(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Per-channel spectral moments:
        M_k(x) = sum_u x(u) * (L^k x)(u),  k=1..K
        Works for both 2D and 3D grids.
        """
        moments: List[torch.Tensor] = []
        Lkx = x
        for _ in range(1, self.num_moments + 1):
            Lkx = self._apply_laplacian(Lkx)
            if x.dim() == 4:
                Mk = torch.sum(x * Lkx, dim=(2, 3))      # [B,C]
            else:
                Mk = torch.sum(x * Lkx, dim=(2, 3, 4))   # [B,C]
            Mk = torch.clamp(Mk, min=0.0)
            moments.append(Mk)
        return moments

    # ---------------------- ECC (Euler Characteristic Curve) ----------------------

    def _pad_to_multiple_3d(self, x: torch.Tensor, block: Tuple[int,int,int], pad_value: float = 0.0) -> Tuple[torch.Tensor, Tuple[int,int,int]]:
        """
        Pad [B,C,H,W,Z] to multiples of block sizes (H,W,Z).
        Returns padded tensor and (Hp, Wp, Zp).
        """
        assert x.dim() == 5
        B, C, H, W, Z = x.shape
        bh, bw, bz = block

        Hp = ((H + bh - 1) // bh) * bh
        Wp = ((W + bw - 1) // bw) * bw
        Zp = ((Z + bz - 1) // bz) * bz

        pad_h = Hp - H
        pad_w = Wp - W
        pad_z = Zp - Z

        # F.pad for 5D: pad order is (Z_left, Z_right, W_left, W_right, H_left, H_right)
        x_pad = F.pad(x, (0, pad_z, 0, pad_w, 0, pad_h), mode="constant", value=pad_value)
        return x_pad, (Hp, Wp, Zp)


    def _block_view_3d(self, x: torch.Tensor, block: Tuple[int,int,int]) -> torch.Tensor:
        """
        Convert padded [B,C,Hp,Wp,Zp] into block view:
        [B, C, Nh, Nw, Nz, bh, bw, bz]
        where Nh=Hp/bh, etc.

        Non-overlapping blocks only.
        """
        B, C, Hp, Wp, Zp = x.shape
        bh, bw, bz = block
        assert Hp % bh == 0 and Wp % bw == 0 and Zp % bz == 0
        Nh, Nw, Nz = Hp // bh, Wp // bw, Zp // bz

        # reshape then permute so block indices come before within-block indices
        x = x.view(B, C, Nh, bh, Nw, bw, Nz, bz)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
        # [B,C,Nh,Nw,Nz,bh,bw,bz]
        return x


    def soft_euler_char_3d_local(self, occ: torch.Tensor, block: Tuple[int,int,int] = (32,32,16)) -> torch.Tensor:
        """
        Localized soft Euler characteristic for 3D union-of-cubes, computed per block.

        occ: [B,C,H,W,Z] in [0,1]
        block: (bh,bw,bz)

        Returns:
        chi_local_mean: [B,C]  (mean chi over all blocks)
        chi_blocks:     [B,C,Nh,Nw,Nz]  (optional; can be used for diagnostics)

        Definition inside each block (no cross-block counting):
        chi = N_cubes - N_faces + N_edges - N_vertices
        where:
        N_faces  = faces along H/W/Z (pairwise adjacencies)
        N_edges  = edges along H/W/Z (2x2 blocks in orthogonal planes)
        N_vertices = 2x2x2 blocks (8-way product)
        """
        assert occ.dim() == 5
        bh, bw, bz = block
        assert bh >= 2 and bw >= 2 and bz >= 2, "block size must be >=2 in each dim for 3D Euler components."

        occ_pad, (Hp, Wp, Zp) = self._pad_to_multiple_3d(occ, block, pad_value=0.0)
        x = self._block_view_3d(occ_pad, block)  # [B,C,Nh,Nw,Nz,bh,bw,bz]
        B, C, Nh, Nw, Nz, _, _, _ = x.shape

        # Convenience aliases for within-block dims
        # cubes: sum over (bh,bw,bz)
        n_cubes = x.sum(dim=(5, 6, 7))  # [B,C,Nh,Nw,Nz]

        # faces: pairwise adjacencies within block
        # along h: (h,h+1)
        n_face_h = (x[..., :-1, :, :] * x[..., 1:, :, :]).sum(dim=(5, 6, 7))  # [B,C,Nh,Nw,Nz]
        # along w: (w,w+1)
        n_face_w = (x[..., :, :-1, :] * x[..., :, 1:, :]).sum(dim=(5, 6, 7))
        # along z: (z,z+1)
        n_face_z = (x[..., :, :, :-1] * x[..., :, :, 1:]).sum(dim=(5, 6, 7))
        n_faces = n_face_h + n_face_w + n_face_z

        # edges: 4-way intersections (2x2 blocks) within block
        # edges parallel to h -> 2x2 in (w,z)
        e_h = (
            x[..., :, :-1, :-1] *
            x[..., :,  1:, :-1] *
            x[..., :, :-1,  1:] *
            x[..., :,  1:,  1:]
        ).sum(dim=(5, 6, 7))

        # edges parallel to w -> 2x2 in (h,z)
        e_w = (
            x[..., :-1, :, :-1] *
            x[...,  1:, :, :-1] *
            x[..., :-1, :,  1:] *
            x[...,  1:, :,  1:]
        ).sum(dim=(5, 6, 7))

        # edges parallel to z -> 2x2 in (h,w)
        e_z = (
            x[..., :-1, :-1, :] *
            x[...,  1:, :-1, :] *
            x[..., :-1,  1:, :] *
            x[...,  1:,  1:, :]
        ).sum(dim=(5, 6, 7))

        n_edges = e_h + e_w + e_z

        # vertices: 8-way intersections (2x2x2 blocks)
        n_vertices = (
            x[..., :-1, :-1, :-1] *
            x[...,  1:, :-1, :-1] *
            x[..., :-1,  1:, :-1] *
            x[...,  1:,  1:, :-1] *
            x[..., :-1, :-1,  1:] *
            x[...,  1:, :-1,  1:] *
            x[..., :-1,  1:,  1:] *
            x[...,  1:,  1:,  1:]
        ).sum(dim=(5, 6, 7))

        chi_blocks = n_cubes - n_faces + n_edges - n_vertices  # [B,C,Nh,Nw,Nz]
        chi_local_mean = chi_blocks.mean(dim=(2, 3, 4))        # [B,C]
        return chi_local_mean, chi_blocks


    def ecc_curve_3d_local(self,
        logits: torch.Tensor,
        taus: List[float],
        kappa: float = 10.0,
        block: Tuple[int,int,int] = (32,32,16),
    ) -> torch.Tensor:
        """
        Localized 3D ECC per-channel on voxel logits.

        logits: [B,C,H,W,Z] (per-class logits)
        taus: thresholds in [0,1]
        kappa: softness of thresholding
        block: local block size for localized Euler

        Returns:
        ecc: [B,C,M] where M=len(taus)
        """
        if logits.dim() == 4:
            logits = logits.reshape(1, logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3])
        assert logits.dim() == 5
        p = torch.sigmoid(logits)  # [B,C,H,W,Z] in [0,1]
        ecc_list = []
        for tau in taus:
            tau_t = torch.tensor(tau, device=logits.device, dtype=logits.dtype)
            # soft superlevel set {p > tau}
            occ = torch.sigmoid(kappa * (p - tau_t))
            chi_mean, _ = self.soft_euler_char_3d_local(occ, block=block)  # [B,C]
            ecc_list.append(chi_mean)
        return torch.stack(ecc_list, dim=-1)  # [B,C,M]

    # ---------------------- final loss ----------------------

    def _match_view(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        High-frequency distillation + 3D localized ECC on voxel logits.

        Expected:
        s,t: [B,C,H,W,Z]  (per-class voxel logits)
        """
        assert s.dim() == 5 and t.dim() == 5, f"Expect [B,C,H,W,Z], got {s.shape} and {t.shape}"

        # (A) Multi-scale high-pass alignment: H_b = X - G_b, where G_b is low-pass via diffusion
        gs, gt = s, t
        loss_hf = s.new_tensor(0.0)

        for b in range(self.num_scales):
            gs = self._diffuse_step(gs)
            gt = self._diffuse_step(gt)

            hs = s - gs
            ht = t - gt

            ns = self._normalize_response(hs)
            nt = self._normalize_response(ht)
            loss_resp = torch.mean((ns - nt) ** 2)

            Es = self._get_dirichlet_energy(hs)
            Et = self._get_dirichlet_energy(ht)
            Es_n = F.normalize(Es, p=2, dim=1)
            Et_n = F.normalize(Et, p=2, dim=1)
            loss_eng = F.mse_loss(Es_n, Et_n)

            loss_hf = loss_hf + self.eta[b].to(loss_hf.dtype) * (loss_resp + self.lambda_energy * loss_eng)

        loss_hf = loss_hf / (self.eta.sum().to(loss_hf.dtype) + self.eps)

        # (B) Spectral moments on the final (strongest) high-pass residual
        h_final_s = s - gs
        h_final_t = t - gt
        mom_s = self._spectral_moments(h_final_s)
        mom_t = self._spectral_moments(h_final_t)

        loss_mom = s.new_tensor(0.0)
        for k in range(self.num_moments):
            ms = F.normalize(mom_s[k], p=2, dim=1)
            mt = F.normalize(mom_t[k], p=2, dim=1)
            loss_mom = loss_mom + F.mse_loss(ms, mt)

        # (C) 3D localized ECC alignment (per-channel / per-threshold)
        taus = [float(v) for v in self.ecc_taus.detach().cpu().tolist()]
        ecc_s = F.normalize(self.ecc_curve_3d_local(s, taus=taus, kappa=self.ecc_kappa), p=2, dim=-1)  # [B,C,M]
        ecc_t = F.normalize(self.ecc_curve_3d_local(t, taus=taus, kappa=self.ecc_kappa), p=2, dim=-1)  # [B,C,M]
        loss_ecc = torch.mean(torch.abs(ecc_s - ecc_t))

        # 将谱矩对齐按阶数取平均，避免 num_moments 增加时梯度线性放大
        loss_mom = loss_mom / float(self.num_moments)

        return loss_hf + self.lambda_moment * loss_mom + self.lambda_ecc * loss_ecc

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor):
        """
        logits_s/t: [B,C,H,W,Z] (per-class voxel logits)
        """
        if self.detach_teacher:
            logits_t = logits_t.detach()
        return self._match_view(logits_s, logits_t)