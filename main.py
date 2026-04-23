from InvertibleModules.inv_modules import *

from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

import sys
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import scipy
from scipy.stats import gaussian_kde

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

from networks import *
from data import *


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

    losses = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i == 0 or i % 10 == 0:
                print(
                    f'Epoch {str(epoch).zfill(3)}, Batch {str(i).zfill(3)}, Loss {loss.item():.4e}, η {scheduler.get_lr()[0]:.4e} ',
                    end='\r'
                )

        scheduler.step()
        mean_loss = total_loss/len(train_loader)
        losses[epoch] = mean_loss

        if epoch == max_epochs - 1:
            print(" "*90, end="\r")

        print(
            f'Epoch {str(epoch).zfill(3)}, Loss {mean_loss:.4e}',
            end='\r'
        )

    print()

    return losses


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

    torch.manual_seed(96)

    print(f"Device {DEVICE}")

    # Parameters
    n = 1_000
    dim = 2
    scale = 1
    seed = 0
    a = 4.5

    pad_dim = 4
    n_blocks = 4
    sl = 0.125

    epochs = 1500
    batch_size = 16
    test_ratio = 0.2
    lr = 2**-7

    # Network initialization
    luNet = AffineNet(
        in_features=dim,
        out_features=2,
        pad_dim=pad_dim,
        num_blocks=n_blocks,
        slope=sl,
        dtype=torch.double
    ).to(DEVICE)

    # Dataset and train test split initialization
    x, y = make_spiral(n_per_class=n, noise=0.4, turns=4.22)
    x /= (torch.amax(x) - torch.amin(x))

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

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.96875,
        last_epoch=-1
    )

    losses = train(
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

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    ax[0].scatter(x[:, 0], x[:, 1], c=cols_tru, alpha=0.5)
    ax[0].set_title("True dataset")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("x")

    ax[1].scatter(x[:, 0], x[:, 1], c=cols_pre, alpha=0.5)
    ax[1].set_title("Model prodiction")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    ax[2].semilogy(losses, c="black")
    ax[2].set_title("Train loss")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Loss")

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


def plot_inverse_path(net: AffineNet, dim, n=20):

    fig, ax = plt.subplots(dim, dim, figsize=(20, 20))
    for i in np.random.permutation(range(0, n)):
        for v1 in range(0, dim):
            for v2 in range(v1, dim):
                data = make_cross(n, dim, v1, v2, scale=5)
                # data = make_line(n, dim, v1, scale=5)
                back_map = net.inverse(data, start=7).detach().cpu().numpy()
                ax[v1, v2].scatter(
                    back_map[:, 0],
                    back_map[:, 1], alpha=0.5)

    plt.show()


def show_forward_path(net, dim, n=100, internal_dim=6):

    x = 2*torch.rand(n, dim) - 1
    x, _ = make_spiral(n, 0.5, 4)
    x /= (torch.amax(x) - torch.amin(x))

    layer_approx = []

    for l in net.network_modules:
        x = l(x)
        a = x.to('cpu').detach().numpy()
        layer_approx.append(a)

    traces = np.stack(layer_approx[0:-1])
    print(traces.shape)

    layers, _, _ = traces.shape
    fig, ax = plt.subplots(
        layers,
        internal_dim,
        figsize=(30, 20),
        constrained_layout=True)

    for l in range(layers):
        for v in range(internal_dim):
            try:
                data = traces[l, :, v]
                kde = gaussian_kde(data, bw_method=(
                    lambda obj, fac=1/6: np.power(obj.n, -1./(obj.d+4)) * fac)
                )
                x_grid = np.linspace(data.min(), data.max(), 500)
                ax[l, v].plot(x_grid, kde(x_grid), color='darkred', lw=2)

            except scipy.linalg.LinAlgError as e:
                continue

            ax[l, v].set_title(f'Layer {l}, Component {v}')

    plt.show()


def show_path(net: AffineNet, scale=1, dim=2):

    n = 500
    x, y = make_spiral(n_per_class=n, noise=0.8, turns=4.22)
    x /= (torch.amax(x) - torch.amin(x))

    cols_tru = y.to('cpu').detach().numpy()
    cols_tru = 0.9 * np.hstack(
        (cols_tru, 0.2 * np.ones((1, cols_tru.shape[0])).T)
    )

    layer_approx = []

    for l in net.network_modules:
        x = l(x)
        a = x.to('cpu').detach().numpy()
        layer_approx.append(a)

    traces = np.stack(layer_approx[0:-1])
    print(traces.shape)

    l, n, d = traces.shape

    fig, ax = plt.subplots(d, d, figsize=(15, 15))
    for i in np.random.permutation(range(0, n)):
        for v1 in range(0, d):
            for v2 in range(v1, d):
                ax[v1, v2].plot(
                    traces[:, i, v1],
                    traces[:, i, v2],
                    c=cols_tru[i], alpha=0.2)

    plt.show()


def show_path_pca(net: AffineNet, n_samples=200, pca_dim=2):

    x, y = make_spiral(n_per_class=n_samples, noise=0.5, turns=4.22)
    x = x / (torch.max(x) - torch.min(x))

    # colour for each point (RGBA)
    cols = y.cpu().numpy()
    cols = 0.9 * np.hstack((cols, 0.2 * np.ones((cols.shape[0], 1))))

    activations = []
    for layer in net.network_modules[0:-1]:
        x = layer(x)
        activations.append(x.detach().cpu().numpy())

    pcs = []
    for act in activations:
        pca = PCA(n_components=pca_dim)
        pcs.append(pca.fit_transform(act))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)  # projection='3d')

    # each sample follows a line through the layers
    for i in np.random.permutation(range(2*n_samples)):
        traj = np.stack([pc[i] for pc in pcs])
        ax.plot(traj[:, 0],
                traj[:, 1],
                # traj[:, 2],
                color=cols[i], alpha=0.2)

        ax.scatter(traj[-1, 0],
                   traj[-1, 1],
                   # traj[:, 2],
                   color=cols[i], alpha=0.8)

    plt.show()


if __name__ == "__main__":

    print("="*90)

    torch.manual_seed(96)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = "./networks/net_4d.p"

    argv = sys.argv
    if "train" in argv:
        net, dim, scale = train_network_dummy(DEVICE)
        torch.save(net.state_dict(), PATH)
        test_invertibility(net, 2, 1, DEVICE)

    net = AffineNet(
        in_features=2,
        out_features=2,
        pad_dim=4,
        num_blocks=4,
        slope=0.125
    )

    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.eval()

    # show_path_pca(net)
    show_forward_path(net, 2)
    #  plot_inverse_path(net, 6)

    print("="*90)
    exit()
