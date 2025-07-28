import math
from typing import Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class FrequencyModule(nn.Module):

    def __init__(self, dim):
        super(FrequencyModule, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        rdim = self.get_reduction_dim(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 2, 1, bias=False),
        )
        # Define learnable parameters for gating
        self.alpha_h = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_w = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        f_high, f_low = self.fft(x)
        return f_high, f_low

    def get_reduction_dim(self, dim):
        if dim < 8:  # 最小维度保护
            return max(2, dim)
        log_dim = math.log2(dim)
        reduction = max(2, int(dim // log_dim))
        return reduction

    def shift(self, x):
        """shift FFT feature map to center"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        # 这个阈值用于确定频谱中心的大小，即决定多大范围的频率被认为是低频。
        # 加入了两个可学习参数帮助确定h和w
        blended_threshold_h = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        blended_threshold_w = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        # Calculate the dimensions of the mask based on the blended thresholds
        for i in range(mask.shape[0]):
            h_ = (h // 2 * blended_threshold_h[i]).round().int()  # Convert to int after rounding
            w_ = (w // 2 * blended_threshold_w[i]).round().int()  # Convert to int after rounding

            # Apply the mask based on blended h and w
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        # 对于mask的每个元素，根据阈值在频谱的中心位置创建一个正方形窗口，窗口内的值设为1，表示这部分是低频区域。
        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)
        # 对x执行FFT变换，得到频谱，并通过shift方法将低频分量移动到中心。
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)  # 丢弃相位信息

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)  # 丢弃相位信息

        return high, low


def _create_normalized_distance_grid(h: int, w: int) -> torch.Tensor:
    """Returns an (h×w) tensor whose values ∈[0,1] encode the radial distance
    from the spectrum centre after *fftshift*.
    """
    # Coordinates after an fftshift: (0,0) is the centre.
    # Build meshgrid with range [‑floor(h/2), ..., ceil(h/2)‑1] etc.
    yy = torch.arange(h) - h // 2
    xx = torch.arange(w) - w // 2
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    dist = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    dist /= dist.max()  # normalise to [0,1]
    return dist  # (H,W)


class FreqDecoupleSoft(nn.Module):
    """
    Soft frequency decoupling with per-channel learnable radius R and
    learnable temperature τ.  Optionally leaves a 'gap' (delta) so that
    low/high bands do not overlap.
    Args:
        h, w  : spatial size of the feature map fed to FFT
        dim   : channel dim (C) used to create per-channel R
        tau_0 : initial temperature (positive)
        delta : width of the buffer zone between low & high (0-0.1 recommended)
    """

    def __init__(self,
                 h: int,
                 w: int,
                 dim: int,
                 tau_0: float = 10.0,
                 delta: float = 0.05) -> None:
        super().__init__()

        # distance grid after rfft2 & fftshift (rows shifted, cols natural)
        self.register_buffer("dist", _create_normalized_distance_grid(h, w // 2 + 1))  # (H,Wc)

        # learnable radius per-channel (unconstrained → sigmoid → (0,1))
        self.r = nn.Parameter(torch.zeros(1, dim, 1, 1))  # (1,C,1,1)

        # learnable positive temperature
        self.tau = nn.Parameter(torch.tensor(tau_0))

        self.delta = delta

        # simple 3×3 conv to mix channels & stabilise spectrum statistics
        self.pre_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

    def _make_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create complementary soft masks, shape (1,1,H,Wc)."""
        # (1,1,H,Wc)
        dist = self.dist.unsqueeze(0).unsqueeze(0)

        # (1,C,1,1) broadcast
        R = torch.sigmoid(self.r)  # (1,C,1,1) ∈(0,1)
        tau = F.softplus(self.tau) + 1e-6  # keep τ > 0

        # high-frequency soft mask  σ((d - R) * τ)
        mask_high = torch.sigmoid((dist - R) * tau)

        # low-frequency soft mask   σ((R - δ - d) * τ)
        mask_low = torch.sigmoid((R - dist) * tau)

        return mask_low, mask_high  # (1,C,H,Wc)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (B,C,H,W): real-valued feature map
        Returns:
            complex_low  (B,C,H,Wc): low-freq complex spectrum
            complex_high (B,C,H,Wc): high-freq complex spectrum
            phase_all    (B,C,H,Wc): phase of the full spectrum
        """
        B, C, H, W = x.shape
        assert H == self.dist.shape[0], "height mismatch with pre-computed dist grid"

        x = self.pre_conv(x)  # simple spectral stabiliser

        # rFFT (orthonormal – preserves energy)  → (B,C,H,W//2+1)
        F_r = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        # shift rows so that radial distance grid对齐
        F_shift = torch.fft.fftshift(F_r, dim=-2)

        mask_low, mask_high = self._make_masks()  # (1,C,H,Wc)

        complex_low = F_shift * mask_low
        complex_high = F_shift * mask_high

        phase = torch.angle(F_shift)  # （备用，可解释）
        return complex_low, complex_high, phase


class FreqRecouple(nn.Module):
    def forward(self, F_complex: torch.Tensor) -> torch.Tensor:
        # inverse shift
        F_ishift = torch.fft.ifftshift(F_complex, dim=-2)
        x_rec = torch.fft.irfft2(F_ishift, dim=(-2, -1), norm="ortho")  # (B,C,H,W)
        return x_rec.real


class SplitFrequencySoft(nn.Module):
    """
    Combines FreqDecoupleSoft + FreqRecouple to return spatial-domain
    (x_high, x_low).  Drop-in replacement for previous SplitFrequency_soft.
    """

    def __init__(self,
                 height: int,
                 width: int,
                 dim: int,
                 tau_0: float = 10.0,
                 delta: float = 0.05):
        super().__init__()
        self.fd = FreqDecoupleSoft(height, width, dim, tau_0, delta)
        self.ifft = FreqRecouple()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        complex_low, complex_high, _ = self.fd(x)
        x_low = self.ifft(complex_low)  # (B,C,H,W)
        x_high = self.ifft(complex_high)
        return x_high, x_low


class SplitFrequency(nn.Module):

    def __init__(self, dim):
        super(SplitFrequency, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        rdim = self.get_reduction_dim(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 2, 1, bias=False),
        )
        # Define learnable parameters for gating
        self.alpha_h = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_w = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        f_high, f_low = self.fft(x)
        return f_high, f_low

    def get_reduction_dim(self, dim):
        if dim < 8:  # 最小维度保护
            return max(2, dim)
        log_dim = math.log2(dim)
        reduction = max(2, int(dim // log_dim))
        return reduction

    def shift(self, x):
        """shift FFT feature map to center"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        # 这个阈值用于确定频谱中心的大小，即决定多大范围的频率被认为是低频。
        # 加入了两个可学习参数帮助确定h和w
        blended_threshold_h = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        blended_threshold_w = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        # Calculate the dimensions of the mask based on the blended thresholds
        for i in range(mask.shape[0]):
            h_ = (h // 2 * blended_threshold_h[i]).round().int()  # Convert to int after rounding
            w_ = (w // 2 * blended_threshold_w[i]).round().int()  # Convert to int after rounding

            # Apply the mask based on blended h and w
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        # 对于mask的每个元素，根据阈值在频谱的中心位置创建一个正方形窗口，窗口内的值设为1，表示这部分是低频区域。
        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)
        # 对x执行FFT变换，得到频谱，并通过shift方法将低频分量移动到中心。
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)  # 丢弃相位信息

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)  # 丢弃相位信息

        return high, low


class FourierUnitHL(nn.Module):
    """Split the spectrum into high / low bands and bring each back to spatial
    domain, yielding **two** feature maps (x_high, x_low).

    The split radius R is learnable (scalar ∈(0,1)).  This is a minimal
    extension over the classic Fourier Unit in FFCNet.
    """

    def __init__(self, h, w, channels: int, tau_0: float = 10.0, delta: float = 0.05):
        super().__init__()

        self.register_buffer("dist", _create_normalized_distance_grid(h, w // 2 + 1))  # (H,Wc)

        # learnable radius per-channel (unconstrained → sigmoid → (0,1))
        self.r = nn.Parameter(torch.zeros(1, channels, 1, 1))  # (1,C,1,1)
        # learnable positive temperature
        self.tau = nn.Parameter(torch.tensor(tau_0))
        self.delta = delta

        # Light 1×1 "complex" conv implemented as two real convs on stacked R/I
        self.conv_re = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_im = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)

    def _make_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create complementary soft masks, shape (1,1,H,Wc)."""
        # (1,1,H,Wc)
        dist = self.dist.unsqueeze(0).unsqueeze(0)
        # (1,C,1,1) broadcast
        R = torch.sigmoid(self.r)  # (1,C,1,1) ∈(0,1)
        tau = F.softplus(self.tau) + 1e-6  # keep τ > 0
        # high-frequency soft mask  σ((d - R) * τ)
        mask_high = torch.sigmoid((dist - R) * tau)
        # low-frequency soft mask   σ((R - δ - d) * τ)
        mask_low = torch.sigmoid((R - self.delta - dist) * tau)

        return mask_low, mask_high  # (1,C,H,Wc)

    def _complex_conv(self, F_c: torch.Tensor) -> torch.Tensor:
        """Performs a 1×1 conv on real & imag parts separately, then re‑stacks
        back to complex tensor with shape like F_c."""
        B, C, H, Wc = F_c.shape
        # (B,C,H,Wc,2)
        F_ri = torch.view_as_real(F_c)
        # rearrange → (B, 2C, H, Wc)
        F_ri = F_ri.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, Wc)
        # conv + BN
        re = self.conv_re(F_ri[:, ::2])
        im = self.conv_im(F_ri[:, 1::2])
        out = torch.cat([re, im], dim=1)
        out = self.bn(out)
        # back to complex
        out = out.reshape(B, C, 2, H, Wc).permute(0, 1, 3, 4, 2).contiguous()
        return torch.view_as_complex(out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (B,C,H,W); Returns: x_high, x_low (each B,C,H,W)"""
        B, C, H, W = x.shape
        device = x.device
        # rFFT2
        F_r = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")  # (B,C,H,Wc)
        F_shift = torch.fft.fftshift(F_r, dim=-2)  # shift rows

        # radial masks
        mask_low, mask_high = self._make_masks()  # (1,C,H,Wc)

        # apply masks
        F_high = F_shift * mask_high
        F_low = F_shift * mask_low

        # small complex 1×1 conv (channel mixing) per band
        F_high = self._complex_conv(F_high)
        F_low = self._complex_conv(F_low)

        # inverse FFT back to spatial domain
        F_high = torch.fft.ifftshift(F_high, dim=-2)
        F_low = torch.fft.ifftshift(F_low, dim=-2)
        x_high = torch.fft.irfft2(F_high, s=(H, W), dim=(-2, -1), norm="ortho")
        x_low = torch.fft.irfft2(F_low, s=(H, W), dim=(-2, -1), norm="ortho")
        return x_high, x_low


class FFCBlockHL(nn.Module):
    """Local spatial conv + global Fourier branch, returning two outputs:
         (out_high, out_low).  Each output concatenates the local branch with
         the respective high/low global features.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 height: int,
                 width: int,
                 alpha: float = 0.5,
                 tau_0: float = 10.0,
                 delta: float = 0.05
                 ):
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha should be in (0,1)"
        self.c_global = int(round(in_channels * alpha))
        self.c_local = in_channels - self.c_global

        # simple 3×3 conv to mix channels & stabilise spectrum statistics (v1)
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        # # simple 3×3 conv to mix channels & stabilise spectrum statistics (v2)
        # self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)

        # Local (spatial) conv branch (v1)
        self.local_conv = nn.Sequential(
            nn.Conv2d(self.c_local, self.c_local, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c_local),
            nn.GELU()
        )
        # # High and low frequencies are Completely Decoupled (v2)
        # self.local_conv_high = nn.Sequential(
        #     nn.Conv2d(self.c_local, self.c_local, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.c_local),
        #     nn.GELU()
        # )
        # self.local_conv_low = nn.Sequential(
        #     nn.Conv2d(self.c_local, self.c_local, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.c_local),
        #     nn.GELU()
        # )

        # Global (frequency) branch
        self.fourier_unit = FourierUnitHL(height, width, self.c_global, tau_0, delta) if self.c_global > 0 else None

        # Final projection to match out_channels
        self.proj_high = nn.Conv2d(self.c_local + self.c_global, out_channels, 1, bias=False)
        self.proj_low = nn.Conv2d(self.c_local + self.c_global, out_channels, 1, bias=False)
        self.bn_high = nn.BatchNorm2d(out_channels)
        self.bn_low = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_conv(x)

        if self.c_global == 0:
            # Degenerates to normal conv block
            x_local = self.local_conv(x)
            out_high = out_low = self.bn_high(self.proj_high(x_local))
            return out_high, out_low

        x_local, x_global = torch.split(x, [self.c_local, self.c_global], dim=1)
        # Local (spatial) conv branch (v1)
        x_local = self.local_conv(x_local)  # (B,Cl,H,W)
        # # High and low frequencies are Completely Decoupled (v2)
        # x_local_high = self.local_conv_high(x_local)  # (B,Cl,H,W)
        # x_local_low = self.local_conv_low(x_local)  # (B,Cl,H,W)

        x_g_high, x_g_low = self.fourier_unit(x_global)  # each (B,Cg,H,W)

        # Local (spatial) conv branch (v1)
        out_high = torch.cat([x_local, x_g_high], dim=1)  # (B,Cl+Cg,H,W)
        out_low = torch.cat([x_local, x_g_low], dim=1)
        # # High and low frequencies are Completely Decoupled (v2)
        # out_high = torch.cat([x_local_high, x_g_high], dim=1)
        # out_low = torch.cat([x_local_low, x_g_low], dim=1)

        out_high = self.bn_high(self.proj_high(out_high))
        out_low = self.bn_low(self.proj_low(out_low))
        return out_high, out_low


class SpatialAttentionMean(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttentionMean, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(2, 2 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * reduction, 1, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True)  # B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True)  # B  1  H  W
        x_cat = torch.cat((x1_mean_out, x2_mean_out), dim=1)  # B 4 H W
        spatial_weights = self.mlp(x_cat)
        return spatial_weights


class SpatialAttentionMax(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttentionMax, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(2, 2 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * reduction, 1, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  # B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  # B  1  H  W
        x_cat = torch.cat((x1_max_out, x2_max_out), dim=1)  # B 4 H W
        spatial_weights = self.mlp(x_cat)
        return spatial_weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  # B  1  H  W
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True)  # B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  # B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True)  # B  1  H  W
        x_cat = torch.cat((x1_mean_out, x1_max_out, x2_mean_out, x2_max_out), dim=1)  # B 4 H W
        spatial_weights = self.mlp(x_cat)  # B 1 H W
        return spatial_weights


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2)  # B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)

        avg_se = self.mlp(avg_v).view(B, self.dim, 1)
        max_se = self.mlp(max_v).view(B, self.dim, 1)

        channel_weights = self.sigmoid(avg_se + max_se).view(B, self.dim, 1, 1)  # B C 1 1
        return channel_weights
