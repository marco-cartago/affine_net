from InvertibleModules.inv_modules import *

from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineNet(nn.Module):

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        pad_dim: int = 8,
        num_blocks: int = 3,
        slope: float = 1e-1,
        dtype: torch.dtype = torch.float
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pad_dim = pad_dim
        self.num_blocks = num_blocks
        self.slope = slope

        # Initial dimensionality expansion
        ls = [ExtendDim(pad_dim, dtype=dtype)]

        # Invertible (upto machine epsilon sometimes!) LU blocks
        for _ in range(num_blocks):
            ls += [LUBlock(in_features + pad_dim, dtype=dtype),
                   I_LeakyReLU(negative_slope=slope)]

        # Final linear layer to extract logits
        ls += [nn.Linear(in_features + pad_dim, out_features, dtype=dtype)]

        self.network_modules = nn.ModuleList(ls)

    def forward(self, x: torch.Tensor, start: int = 0, end=None) -> torch.Tensor:
        module_list = list(self.network_modules)[start:]
        if end is not None and end < len(module_list):
            module_list = module_list[0:end]

        for module in module_list:
            x = module(x)

        return x

    def inverse(self, x: torch.Tensor, start: int = 1, end=None) -> torch.Tensor:
        module_list: List[nn.Module] = list(self.network_modules)[::-1]
        module_list = module_list[start:]

        if end is not None and end < len(module_list):
            module_list = module_list[:end]

        for module in module_list:
            x = module.inverse(x)  # type: ignore
        return x

class RecAffineNet(nn.Module):

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        pad_dim: int = 8,
        num_blocks: int = 3,
        slope: float = 1e-1,
        dtype: torch.dtype = torch.float
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pad_dim = pad_dim
        self.num_blocks = num_blocks
        self.slope = slope

        # Initial dimensionality expansion
        ls = [ExtendDim(pad_dim, dtype=dtype)]

        # Invertible (upto machine epsilon sometimes!) LU blocks
        ls += [LUBlock(in_features + pad_dim, dtype=dtype),
                I_LeakyReLU(negative_slope=slope)] * num_blocks

        # Final linear layer to extract logits
        ls += [nn.Linear(in_features + pad_dim, out_features, dtype=dtype)]

        self.network_modules = nn.ModuleList(ls)

    def forward(self, x: torch.Tensor, start: int = 0, end=None) -> torch.Tensor:
        module_list = list(self.network_modules)[start:]
        if end is not None and end < len(module_list):
            module_list = module_list[0:end]

        for module in module_list:
            x = module(x)

        return x

    def inverse(self, x: torch.Tensor, start: int = 1, end=None) -> torch.Tensor:
        module_list: List[nn.Module] = list(self.network_modules)[::-1]
        module_list = module_list[start:]

        if end is not None and end < len(module_list):
            module_list = module_list[:end]

        for module in module_list:
            x = module.inverse(x)  # type: ignore
        return x
