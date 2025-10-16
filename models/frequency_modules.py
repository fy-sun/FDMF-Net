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
        if dim < 8:
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
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        blended_threshold_h = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        blended_threshold_w = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        for i in range(mask.shape[0]):
            h_ = (h // 2 * blended_threshold_h[i]).round().int()
            w_ = (w // 2 * blended_threshold_w[i]).round().int()

            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)

        return high, low


def _create_normalized_distance_grid(h: int, w: int) -> torch.Tensor:
    yy = torch.arange(h) - h // 2
    xx = torch.arange(w) - w // 2
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    dist = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    dist /= dist.max()
    return dist


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
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True)
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True)
        x_cat = torch.cat((x1_mean_out, x1_max_out, x2_mean_out, x2_max_out), dim=1)
        spatial_weights = self.mlp(x_cat)
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
        avg_v = self.avg_pool(x).view(B, self.dim * 2)
        max_v = self.max_pool(x).view(B, self.dim * 2)

        avg_se = self.mlp(avg_v).view(B, self.dim, 1)
        max_se = self.mlp(max_v).view(B, self.dim, 1)

        channel_weights = self.sigmoid(avg_se + max_se).view(B, self.dim, 1, 1)
        return channel_weights
