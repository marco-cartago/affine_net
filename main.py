from InvertibleModules.inv_modules import *

from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def train(
    max_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module | Callable = nn.CrossEntropyLoss(),
    device: torch.device = torch.device('cpu'),
) -> nn.Module:

    model.train()

    for epoch in range(max_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_loss += loss.item()

            if i == 0 or i % 10 == 0:
                print(
                    f'Epoch {epoch}, Batch {i}, Loss {loss.item():3.4f}',
                    end='\r'
                )

        print(
            f'Epoch {epoch}, Loss {total_loss/len(train_loader):3.4f}',
            end='\r'
        )

    print()

    return model


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


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> float:

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        predicted_labels = torch.argmax(output, dim=-1).cpu().numpy()
        true_labels = torch.argmax(y, dim=-1).cpu().numpy()

        total += y.size(0)
        correct += (true_labels == predicted_labels).sum().item()

        all_preds.extend(predicted_labels)
        all_labels.extend(true_labels)

    accuracy = correct / total * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')

    print("\nClassification Report:\n", classification_report(
        all_labels,
        all_preds,
        digits=4))

    return accuracy


def train_network_dummy(DEVICE):

    torch.manual_seed(0)

    print(f"Device {DEVICE}")

    # Parameters
    n = 10_000
    dim = 2
    scale = 1
    seed = 0
    a = 2

    pad_dim = 16
    n_blocks = 4
    sl = 0.125

    epochs = 500
    batch_size = 32
    test_ratio = 0.2
    lr = 2e-3

    # Network initialization
    luNet = AffineNet(
        in_features=dim,
        out_features=2,
        pad_dim=pad_dim,
        num_blocks=n_blocks,
        slope=sl,
        dtype=torch.float
    ).to(DEVICE)

    # Dataset and train test split initialization
    x, y = make_spiral(n_per_class=1000, noise=0.8, turns=8.0)
    dataset = TensorDataset(x, y)
    test_len = int(len(dataset) * test_ratio)
    train_len = len(dataset) - test_len

    train_set, test_set = random_split(
        dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model training
    optimizer = torch.optim.Adam(
        luNet.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=500,
        T_mult=2,
        eta_min=1e-6,
        last_epoch=-1
    )

    modl = train(
        epochs,
        luNet,
        train_loader,
        optimizer,
        scheduler,
        criterion=nn.CrossEntropyLoss(),
        device=DEVICE
    )

    # Performance evaluation
    test(luNet, test_loader, device=DEVICE)

    # Some plotting :)
    cols_tru = y.to('cpu').detach().numpy()
    cols_tru = 0.9 * np.hstack(
        (cols_tru, 0.2 * np.ones((1, cols_tru.shape[0])).T)
    )

    cols_pre = torch.sigmoid(
        luNet.forward(x.to(DEVICE)).to('cpu')
    ).detach().numpy()
    cols_pre = 0.9 * np.hstack(
        (cols_pre, 0.2 * np.ones((1, cols_pre.shape[0])).T)
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].scatter(x[:, 0], x[:, 1], c=cols_tru, alpha=0.5)
    ax[0].set_title("True dataset")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("x")

    ax[1].scatter(x[:, 0], x[:, 1], c=cols_pre, alpha=0.5)
    ax[1].set_title("Model boundary (treshold 0.5)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    plt.show()

    return luNet, dim, scale


def test_invertibility(luNet, dim, scale, DEVICE):

    n = 10_000

    x = scale*(2*torch.rand(n, dim)-1).to(DEVICE)
    y = luNet.forward(x, end=-1)
    x_inv = luNet.inverse(y, start=1)

    t, _ = torch.max(torch.abs(x - x_inv), dim=-1)

    m = (
        f"Mean maximum absolute errors on {n} samples from [{-scale}, {scale}]^{dim}:" +
        f"\n -> μ, σ     = {torch.mean(t).item():.2e} ± {torch.var(t).item():.2e}" +
        f"\n -> min, max = [{torch.min(t).item():.2e}, {torch.max(t).item():.2e}]"
    )

    print(m)


if __name__ == "__main__":

    print("="*90)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = train_network_dummy(DEVICE)
    test_invertibility(*params, DEVICE)

    print("="*90)
    exit()
