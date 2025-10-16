import torch
import torch.nn as nn


class CLUBMean(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512):

        super(CLUBMean, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # → [B,C,1,1]

        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                      nn.ReLU(),
                                      nn.Linear(int(hidden_size), y_dim))

    def _vec(self, t: torch.Tensor) -> torch.Tensor:
        return self.pool(t).flatten(1)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        return mu, 0

    def forward(self, x_samples, y_samples):
        x_vec, y_vec = self._vec(x_samples), self._vec(y_samples)

        mu, logvar = self.get_mu_logvar(x_vec)

        positive = - (mu - y_vec) ** 2 / 2.

        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_vec.unsqueeze(0)

        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.
        positive = positive.sum(dim=-1)
        negative = negative.sum(dim=-1)
        return (positive - negative).mean()


    def loglikeli(self, x_samples, y_samples):
        x_vec, y_vec = self._vec(x_samples), self._vec(y_samples)  # [B,C]
        mu, logvar = self.get_mu_logvar(x_vec)
        return (-(mu - y_vec) ** 2).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 10, 512, 8, 8
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    club = CLUBMean(C, C)
    print(club(x, y).item())
    print(club.learning_loss(x, y))
