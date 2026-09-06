"""Separate actor and privileged critic; float32 MPS-compatible networks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def select_device(requested):
    if requested == 'auto':
        requested = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable; no silent CPU fallback')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(requested)


def as_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class Encoder(nn.Module):
    def __init__(self, channels, context_dim, hidden):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels,16,3,stride=2,padding=1),nn.SiLU(),
                                  nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),
                                  nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(128+context_dim,hidden),nn.LayerNorm(hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden),nn.Tanh())

    def forward(self, maps, context):
        return self.mlp(torch.cat((self.conv(maps),context),dim=-1))


class Actor(nn.Module):
    def __init__(self, context_dim, hidden=64, channels=2):
        super().__init__()
        self.encoder = Encoder(channels,context_dim,hidden)
        self.mean = nn.Linear(hidden,2)
        nn.init.orthogonal_(self.mean.weight, .01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((2,), -.5))

    def forward(self, maps, context):
        return self.mean(self.encoder(maps,context)), self.log_std.clamp(-5,1).expand(len(maps),-1)

    @staticmethod
    def log_prob(latent, mean, log_std):
        normal = -.5*((latent-mean)*torch.exp(-log_std))**2-log_std-.5*math.log(2*math.pi)
        # log(1-tanh(z)^2), computed without inverse tanh or cancellation.
        jacobian = 2*(math.log(2)-latent-F.softplus(-2*latent))
        return (normal-jacobian).sum(-1)

    def sample(self, maps, context, generator, deterministic=False):
        mean, log_std = self(maps, context)
        noise = torch.randn(mean.shape, generator=generator, device='cpu').to(mean.device) if not deterministic else 0
        latent = mean + torch.exp(log_std)*noise
        return torch.tanh(latent), latent, self.log_prob(latent,mean,log_std)


class Critic(nn.Module):
    def __init__(self, context_dim, hidden=64, channels=3):
        super().__init__()
        self.encoder = Encoder(channels,context_dim,hidden)
        self.value = nn.Linear(hidden,1)

    def forward(self, maps, context):
        return self.value(self.encoder(maps,context)).squeeze(-1)
