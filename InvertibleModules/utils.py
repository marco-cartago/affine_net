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

def triang_QR_gen_(size: Tuple[int], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    W = torch.empty(size=size, dtype=dtype)
    rows, cols = W.size()

    torch.nn.init.xavier_normal_(W)
    Q, _ = torch.linalg.qr(W,mode='complete')
    _, L, U = torch.linalg.lu(Q)

    trl_rows, trl_cols = torch.tril_indices(rows, cols)
    tru_rows, tru_cols = torch.triu_indices(rows, cols)

    with torch.no_grad():
        L[tru_rows, tru_cols] = 0.0
        L += torch.eye(rows)
        U[tru_rows, tru_cols] = 0.0
        U += torch.eye(rows)

    L_mask = torch.ones_like(L)
    L_mask[tru_rows, tru_cols] = 0.0
    U_mask = torch.ones_like(U)
    U_mask[tru_rows, tru_cols] = 0.0

    return L, U, L_mask, U_mask


