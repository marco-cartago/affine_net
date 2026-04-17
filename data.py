import torch
import torch.nn.functional as F
import math
from typing import Tuple


def make_spiral(
    n_per_class: int = 500,
    noise: float = 0.1,
    turns: float = 3.0,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2-D two-class spiral (the classic “two-spiral” toy problem).

    Args
    ----
    n_per_class: number of points per class (total = 2 * n_per_class).
    noise: standard deviation of Gaussian noise added to each point.
    turns: how many revolutions each spiral makes (default 3).
    device: torch device for the returned tensors.

    Returns
    -------
    x: Tensor of shape (2*n_per_class, 2) - the coordinates.
    y: Tensor of shape (2*n_per_class, 2) - one-hot class labels (float).

    """
    # Angles
    theta = torch.linspace(0, turns * math.pi, n_per_class, device=device)
    r = theta

    x0 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x1 = torch.stack([r * torch.cos(theta + math.pi), r *
                     torch.sin(theta + math.pi)], dim=1)

    x = torch.cat([x0, x1], dim=0)

    # Add Gaussian noise
    if noise > 0:
        x += torch.randn_like(x) * noise

    # Labels
    labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long, device=device),
        torch.ones(n_per_class, dtype=torch.long, device=device)
    ], dim=0)

    y = F.one_hot(labels, num_classes=2).float()

    return x, y