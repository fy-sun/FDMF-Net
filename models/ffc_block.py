import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FourierUnit(nn.Module):
    """A very light‑weight 1×1 convolution in the complex frequency domain.
    Real and imaginary parts are concatenated along the channel dim.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        # We double the channel count because we concatenate (real, imag).
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Complex FFT  → (B, C, H, W//2+1)
        fft = torch.fft.rfft2(x, s=(H, W), norm="ortho")
        # Split into real / imag and treat them as channels
        real = fft.real
        imag = fft.imag
        fft_feat = torch.cat([real, imag], dim=1)  # (B, 2C, H, W/2+1)
        # 1×1 conv mixing across channels
        fft_feat = self.conv(fft_feat)
        # Split back
        real, imag = torch.chunk(fft_feat, 2, dim=1)
        fft_new = torch.complex(real, imag)
        # Inverse FFT back to spatial domain
        x_out = torch.fft.irfft2(fft_new, s=(H, W), norm="ortho")  # (B, C_out, H, W)
        return x_out


class FFC(nn.Module):
    """
    FFC layer as proposed in *Fast Fourier Convolution* (Chi et al., NeurIPS 2020).
    Combines local 3×3 conv and global Fourier Unit; two paths interact via
    learnable 1×1 convs.  Suitable drop‑in replacement for Conv2d.

    Args:
        in_channels   :  input channel count
        out_channels  :  output channel count
        ratio_g       :  fraction of channels allocated to the global branch
        kernel_size   :  local conv kernel size (default 3)
        stride / pad  :  same as nn.Conv2d
        bias          :  include bias in local convs
        gating        :  if True, apply simple SE‑style gating to global outputs
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ratio_g: float = 0.25,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            gating: bool = True,
    ) -> None:
        super().__init__()
        assert 0.0 <= ratio_g <= 1.0, "ratio_g must be in [0,1]"

        self.cg_in = int(in_channels * ratio_g)
        self.cl_in = in_channels - self.cg_in
        self.cg_out = int(out_channels * ratio_g)
        self.cl_out = out_channels - self.cg_out
        self.gating = gating and self.cg_out > 0

        # ---------- local branch ----------
        self.conv_l2l = nn.Conv2d(self.cl_in, self.cl_out, kernel_size, stride,
                                  padding, dilation, groups, bias=bias)
        # From global → local (1×1)
        self.conv_g2l = None
        if self.cg_in > 0 and self.cl_out > 0:
            self.conv_g2l = nn.Conv2d(self.cg_in, self.cl_out, 1, bias=False)

        # ---------- global branch ----------
        self.conv_l2g = None  # local → global (1×1)
        if self.cl_in > 0 and self.cg_out > 0:
            self.conv_l2g = nn.Conv2d(self.cl_in, self.cg_out, 1, bias=False)

        self.fourier_unit = None
        if self.cg_in > 0 and self.cg_out > 0:
            self.fourier_unit = FourierUnit(self.cg_in, self.cg_out)

        # Optional gating (SE‑like) on global output
        if self.gating:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.cg_out, self.cg_out // 2, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cg_out // 2, self.cg_out, 1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input along channels
        if self.cg_in > 0:
            x_l, x_g = torch.split(x, [self.cl_in, self.cg_in], dim=1)
        else:
            x_l, x_g = x, None

        # ----- local path -----
        y_l = self.conv_l2l(x_l)
        if self.cg_in > 0 and self.conv_g2l is not None:
            y_l = y_l + self.conv_g2l(x_g)

        # ----- global path -----
        if self.cg_out > 0:
            # local → global message
            if self.conv_l2g is not None:
                x_l2g = self.conv_l2g(x_l)
            else:
                x_l2g = 0.0
            # spectral conv on global branch
            x_g2g = self.fourier_unit(x_g) if self.fourier_unit is not None else 0.0
            y_g = x_l2g + x_g2g

            if self.gating:
                y_g = y_g * self.gate(y_g)
        else:
            y_g = None

        # Concatenate outputs
        if y_g is not None:
            out = torch.cat([y_l, y_g], dim=1)
        else:
            out = y_l
        return out


class FFCResBlock(nn.Module):
    """Residual block composed of two FFC layers."""

    def __init__(self, channels: int, ratio_g: float = 0.25):
        super().__init__()
        self.ffc1 = FFC(channels, channels, ratio_g=ratio_g, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.ffc2 = FFC(channels, channels, ratio_g=ratio_g, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.ffc1(x)))
        out = self.bn2(self.ffc2(out))
        return self.relu(out + x)
