import torch
from typing import Tuple


def freeze_weights(
    tensor: torch.Tensor, mask: torch.Tensor
) -> torch.utils.hooks.RemovableHandle:

    if mask.shape != tensor.shape:
        if not mask.broadcast_to(tensor.shape).shape == tensor.shape:
            raise ValueError("mask must be broadcastable to tensor's shape")

    def hook(grad): return grad * mask.to(grad.device)
    handle = tensor.register_hook(hook)

    return handle


def triangular_xavier_norm_(
    W: torch.Tensor, *, upper: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:

    torch.nn.init.xavier_normal_(W)
    rows, cols = W.size()

    if upper:
        tri_rows, tri_cols = torch.tril_indices(rows, cols, device=W.device)
    else:
        tri_rows, tri_cols = torch.triu_indices(rows, cols, device=W.device)

    with torch.no_grad():
        W[tri_rows, tri_cols] = 0.0
        W += torch.eye(rows, device=W.device)

    mask = torch.ones_like(W)
    mask[tri_rows, tri_cols] = 0.0

    indices = torch.stack([tri_rows, tri_cols], dim=0)

    return mask, indices


# torch.nn.init.normal_(W)
# rows, cols = W.size()

# if upper:
#     fanouts = torch.arange(cols - 1, -1, -1).unsqueeze(0)
# else:
#     fanouts = torch.arange(0, cols).unsqueeze(0)

# fanins = torch.ones(rows) * rows
# var_estim = torch.sqrt(4 / (fanins + fanouts.T))
# W = W * var_estim
