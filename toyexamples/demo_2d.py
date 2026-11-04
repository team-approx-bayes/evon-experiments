"""Compare two-dimensional logistic-regression posteriors and convergence.

Run with uv run demo_2d.py. 
"""

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from torch import nn

from vonsoap.optimizers import EVON, IVON
from utils import expected_loss, get_posterior_covariance, get_posterior_kl, set_seed

# --------------- configuration ---------------
RUN_IVON = True
RUN_SGD = True
RUN_EVON = True
RUN_VON = True  # VON with Hessian as "ground truth"
EVON_WHITENING = False  # True leads to a worse variational objective.

SEED = 0
N = 50
DELTA = 0.01  # Prior precision.
N_STEPS = 30_001
VON_STEPS = 2_500
MC_EVAL = 10_000  # Monte Carlo samples for the variational objective.
K_EVAL = 500  # Evaluate every K steps.
GRID_N = 1_000
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def toy_data(n=60, std1=1.0, std2=1.1):
    n1 = n // 2
    n2 = n - n1
    x1 = np.random.normal([1, 5], std1, size=(n1, 2))
    x2 = np.random.normal([-5, 1], std2, size=(n2, 2))
    return np.r_[x1, x2], np.r_[np.ones(n1), np.zeros(n2)]


def grid_density(weights, x, y, delta, dx, chunk=20_000):
    logp = np.empty(len(weights))
    for i in range(0, len(weights), chunk):
        w = weights[i:i + chunk]
        logits = w @ x.T
        loss = np.logaddexp(0, logits) - y[None, :] * logits
        logp[i:i + chunk] = -loss.sum(1) - 0.5 * delta * (w * w).sum(1)
    logp -= logp.max()
    p = np.exp(logp)
    return p / (p.sum() * dx**2)


def covariance_ellipse(ax, mean, cov, label, color, n_std=1.5):
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0), vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, width, height, angle=angle,
                         fill=False, edgecolor=color, lw=3))
    ax.plot(*mean, "o", color=color, label=label)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0, 2.0]))

    def forward(self, x):
        return x @ self.weight


def von_groundtruth(x, y, delta, ess, steps, mc=10, mc_eval=10_000):
    d = x.shape[1]
    eye = torch.eye(d, dtype=x.dtype, device=x.device)
    bce = nn.BCEWithLogitsLoss()

    def loss(w):
        return bce(x @ w, y)

    def grad_hess(w):
        w = w.detach().requires_grad_(True)
        g = torch.autograd.grad(loss(w), w)[0]
        h = torch.autograd.functional.hessian(loss, w)
        return g.detach(), h.detach()

    mean = torch.tensor([3.0, 2.0], dtype=x.dtype, device=x.device)
    hbar = 0.01 * eye
    gbar = torch.zeros(d, dtype=x.dtype, device=x.device)

    for t in range(steps):
        cov = torch.linalg.inv(ess * hbar + delta * eye)
        chol = torch.linalg.cholesky(cov)
        gs, hs = zip(*(grad_hess(mean + chol @ torch.randn(d)) for _ in range(mc)))
        gbar = 0.9 * gbar + 0.1 * torch.stack(gs).mean(0)
        hbar = 0.995 * hbar + 0.005 * torch.stack(hs).mean(0)
        mean -= (0.4 / (1 + 0.005 * t)) * cov @ (ess * gbar + delta * mean)

    cov = torch.linalg.inv(ess * hbar + delta * eye)
    chol = torch.linalg.cholesky(cov)
    samples = mean + torch.randn(mc_eval, d) @ chol.T
    logits = samples @ x.T
    expected_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y.expand_as(logits), reduction="none"
    ).mean()
    kl = 0.5 * (delta * (torch.trace(cov) + mean @ mean)
                - torch.linalg.slogdet(cov).logabsdet)
    return (ess * expected_loss + kl).item(), mean.detach(), cov.detach()


def train_sgd(model, x, y):
    opt = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9,
                          weight_decay=DELTA / N)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
    for _ in range(N_STEPS):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
        opt.step()
        sched.step()


def main():
    set_seed(SEED, deterministic=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, labels = toy_data(N)
    x = torch.tensor(data, dtype=torch.get_default_dtype())
    y = torch.tensor(labels, dtype=torch.get_default_dtype())
    loss_fn = nn.BCEWithLogitsLoss()

    sgd_model = ivon_model = evon_model = None
    ivon_opt = None
    evon_cov = None
    obj_ivon, obj_evon = [], []
    exact_obj = exact_mean = exact_cov = None

    if RUN_VON:
        exact_obj, exact_mean, exact_cov = von_groundtruth(
            x, y, DELTA, N, VON_STEPS, mc_eval=MC_EVAL
        )

    if RUN_IVON:
        ivon_model = LinearModel()
        ivon_opt = IVON(ivon_model.parameters(), lr=2.0, ess=N,
                        beta1=0.9, beta2=0.9999,
                        weight_decay=DELTA / N, hess_init=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(ivon_opt, N_STEPS)
        for step in range(N_STEPS):
            if step % K_EVAL == 0:
                loss = expected_loss(ivon_model, ivon_opt, x, y, loss_fn,
                                     MC_EVAL, preserve_rng=False)
                obj_ivon.append(N * loss + get_posterior_kl(ivon_opt, omit_constants=True))
            with ivon_opt.sampled_params(train=True):
                ivon_opt.zero_grad()
                loss_fn(ivon_model(x), y).backward()
            ivon_opt.step()
            sched.step()

    if RUN_EVON:
        evon_model = LinearModel()
        opt = EVON(evon_model.parameters(), lr=0.05, betas=(0.9, 0.9999),
                   precondition_frequency=5, correct_bias=True, eps=1e-12,
                   precondition_1d=True, weight_decay=DELTA / N,
                   shampoo_beta=0.999995, ess=N, hess_init=0.01,
                   sync=False, whiten_prec_grad=EVON_WHITENING)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
        for step in range(N_STEPS):
            if step % K_EVAL == 0:
                loss = expected_loss(evon_model, opt, x, y, loss_fn,
                                     MC_EVAL, preserve_rng=False)
                obj_evon.append(N * loss + get_posterior_kl(opt, omit_constants=True))
            with opt.sampled_params(train=True):
                opt.zero_grad()
                loss_fn(evon_model(x), y).backward()
            opt.step()
            sched.step()
        evon_cov = get_posterior_covariance(opt, evon_model)["weight"]

    if RUN_SGD:
        sgd_model = LinearModel()
        train_sgd(sgd_model, x, y)

    grid = np.linspace(-2, 30, GRID_N)
    w0, w1 = np.meshgrid(grid, grid)
    weights = np.c_[w0.ravel(), w1.ravel()]
    density = grid_density(weights, data, labels, DELTA, grid[1] - grid[0])
    density = density.reshape(GRID_N, GRID_N)

    # ---------------- posterior plot ----------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(grid, grid, density, levels=[4e-3], colors="black", linewidths=3)
    active_x = np.flatnonzero(density.max(0) > 2e-3)
    active_y = np.flatnonzero(density.max(1) > 2e-3)
    if len(active_x) and len(active_y):
        xmin, xmax = grid[active_x[[0, -1]]]
        ymin, ymax = grid[active_y[[0, -1]]]
        ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    ax.set_aspect("equal")
    ax.axis("off")

    if sgd_model is not None:
        ax.plot(*sgd_model.weight.detach().numpy(), "kx", label="SGD",
                markersize=12, markeredgewidth=3)
    if ivon_model is not None:
        covariance_ellipse(
            ax, ivon_model.weight.detach().numpy(),
            get_posterior_covariance(ivon_opt, ivon_model)["weight"].numpy(),
            "IVON", "gray",
        )
    if evon_model is not None:
        covariance_ellipse(ax, evon_model.weight.detach().numpy(),
                           np.asarray(evon_cov), "EVON", "red")
    if exact_mean is not None:
        covariance_ellipse(ax, exact_mean.numpy(), exact_cov.numpy(), "Exact", "blue")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)

    suffix = "_whitening" if EVON_WHITENING else ""
    filename = OUTPUT_DIR / f"demo_2d_posterior{suffix}.png"
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

    # ---------------- convergence plot ----------------
    if obj_ivon or obj_evon or exact_obj is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        if obj_ivon:
            ax.plot(np.arange(len(obj_ivon)) * K_EVAL, obj_ivon,
                    label="IVON", color="black")
        if obj_evon:
            ax.plot(np.arange(len(obj_evon)) * K_EVAL, obj_evon,
                    label="EVON", color="red")
        if exact_obj is not None:
            ax.axhline(exact_obj, linestyle="--", color="gray", label="Exact full")

        gray = "0.45"
        ax.set_xlabel("Step", color=gray)
        ax.set_ylabel("Variational objective", color=gray)
        ax.tick_params(axis="both", colors=gray)
        ax.spines["left"].set_color(gray)
        ax.spines["bottom"].set_color(gray)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=gray, alpha=0.2, linewidth=0.8)
        ax.set_ylim(-0.6, 2)
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(gray)

        filename = OUTPUT_DIR / f"demo_2d_convergence{suffix}.png"
        fig.tight_layout()
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
