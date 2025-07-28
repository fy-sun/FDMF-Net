import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MINEMean(nn.Module):
    def __init__(self, dim, hidden_size=512):
        super(MINEMean, self).__init__()
        # 用于把 [B,C,H,W] → [B,C]
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 两层 MLP：dim → hidden_size → 1
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def _vec(self, t: torch.Tensor) -> torch.Tensor:
        # GAP + flatten
        return self.pool(t).flatten(1)  # [B,C]

    def forward(self, x_samples, y_samples):
        """
        返回 MINE 下界：
            I(X;Y) ≥ E_{p(x,y)}[T(x,y)] – log E_{p(x)p(y)}[e^{T(x,y')}]
        """

        # 把 feature map 变成向量
        x = self._vec(x_samples)  # [B, x_dim]
        y = self._vec(y_samples)  # [B, y_dim]

        # 1) joint 对
        joint = torch.cat([x, y], dim=1)                # [B, x_dim+y_dim]
        t1 = self.net(joint)                         # [B, 1]

        # 2) marginal 对（打乱 y）
        y_perm = y[torch.randperm(y.size(0))]           # [B, y_dim]
        marginal = torch.cat([x, y_perm], dim=1)        # [B, x_dim+y_dim]
        t2 = self.net(marginal)                   # [B, 1]

        # 3) 计算下界
        # 注意数值稳定可以改成 logsumexp trick
        # mi_lb = torch.mean(t1) - torch.log(torch.mean(torch.exp(t2)))
        log_mean_exp = torch.logsumexp(t2, dim=0) - math.log(t2.numel())
        mi_lb = torch.mean(t1) - log_mean_exp

        return mi_lb


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 10, 512, 8, 8
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    mine = MINEMean(C, C)
    print(mine(x, y).item())
