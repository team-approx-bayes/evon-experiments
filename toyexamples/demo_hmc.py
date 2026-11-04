"""Iris logistic regression: HMC (gray), EVON (red), and IVON (black)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from utils import expected_loss, get_posterior_covariance, get_posterior_kl
from vonsoap.optimizers import EVON, IVON


PRIOR_STD = 100.0
STEPS = 100_000
SEED = 0
EVAL_EVERY = 2_500
EVAL_SAMPLES = 1_000
EVON_LR = 0.05
CHAINS = 8
WARMUP = 2_500
STAN_SAMPLES = 5_000
RECOMPUTE = False  # Set True to rerun after changing the experiment settings.

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
DATA_DIR = HERE / "outputs" / "demo_hmc"
RESULTS_FILE = DATA_DIR / "demo_hmc_variational.npz"
DRAWS_FILE = DATA_DIR / "demo_hmc_draws.npz"

STAN_MODEL = """
data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=2> K;
  matrix[N, D] X;
  array[N] int<lower=1, upper=K> y;
  real<lower=0> sigma;
}
parameters {
  vector[K] alpha;
  matrix[D, K] beta;
}
model {
  alpha ~ normal(0, sigma);
  to_vector(beta) ~ normal(0, sigma);
  target += categorical_logit_glm_lpmf(y | X, alpha, beta);
}
"""


class LinearModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_features + 1, n_classes))

    def forward(self, x):
        return torch.cat((x, x.new_ones(len(x), 1)), dim=1) @ self.weight


def make_optimizer(method, model, n):
    weight_decay = 1 / (n * PRIOR_STD**2)
    if method == "EVON":
        return EVON(
            model.parameters(), lr=EVON_LR, ess=n, hess_init=.01,
            weight_decay=weight_decay, betas=(.9, .99985),
            precondition_1d=True, precondition_frequency=25, correct_bias=False,
            whiten_prec_grad=False, sync=False,
        )
    if method == "IVON":
        return IVON(model.parameters(), lr=1., ess=n, hess_init=.05,
                    weight_decay=weight_decay, beta2=.9995, clip_radius=.1)
    raise ValueError(f"Unknown optimizer: {method}")


def train(method, x, y):
    torch.manual_seed(SEED)
    model = LinearModel(x.shape[1], int(y.max()) + 1)
    optimizer = make_optimizer(method, model, len(y))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)
    loss_fn = nn.CrossEntropyLoss()
    checkpoints, objectives = [], []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0 or step == STEPS:
            loss = expected_loss(model, optimizer, x, y, loss_fn, EVAL_SAMPLES)
            objective = len(y) * loss + get_posterior_kl(optimizer, omit_constants=True)
            checkpoints.append(step)
            objectives.append(objective)
            print(f"{method} step {step}: objective = {objective:.6f}", flush=True)
        if step == STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        with optimizer.sampled_params(train=True):
            loss_fn(model(x), y).backward()
        optimizer.step()
        scheduler.step()
    covariance = get_posterior_covariance(optimizer, model)["weight"]
    return {
        "mean": model.weight.detach().numpy().copy(),
        "std": covariance.diagonal().sqrt().reshape_as(model.weight).cpu().numpy(),
        "steps": np.array(checkpoints), "objectives": np.array(objectives),
    }


def sample_stan(x, y):
    from cmdstanpy import CmdStanModel

    stan_file = DATA_DIR / "demo_hmc.stan"
    stan_file.write_text(STAN_MODEL, encoding="utf-8")
    chain_dir = DATA_DIR / "stan"
    chain_dir.mkdir(exist_ok=True)
    fit = CmdStanModel(stan_file=str(stan_file)).sample(
        data=dict(N=len(y), D=x.shape[1], K=int(y.max()) + 1,
                  X=x, y=y + 1, sigma=PRIOR_STD),
        seed=SEED, chains=CHAINS, parallel_chains=CHAINS,
        iter_warmup=WARMUP, iter_sampling=STAN_SAMPLES,
        adapt_delta=.99, max_treedepth=12, output_dir=str(chain_dir),
    )
    (DATA_DIR / "demo_hmc_diagnostics.txt").write_text(fit.diagnose(), encoding="utf-8")
    return np.concatenate((fit.stan_variable("beta"),
                           fit.stan_variable("alpha")[:, None, :]), axis=1)


def normal_curve(grid, mean, std):
    # Refine around the Gaussian mean so narrow IVON peaks remain visible.
    mean = float(mean)
    std = max(float(std), 1e-12)
    points = np.unique(np.r_[grid, np.linspace(mean - 6 * std, mean + 6 * std, 401), mean])
    points = points[(points >= grid[0]) & (points <= grid[-1])]
    density = np.exp(-.5 * ((points - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
    return points, density


def plot(results, draws):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi = np.quantile(draws, [.005, .995])
    pad = .1 * (hi - lo + 1e-12)
    grid = np.linspace(lo - pad, hi + pad, 400)
    style = {"figure.figsize": (2.25, 1.55), "figure.constrained_layout.use": True,
             "font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        for row in range(draws.shape[1]):
            for k in range(draws.shape[2]):
                density = gaussian_kde(draws[:, row, k])(grid)
                fig, ax = plt.subplots()
                ax.plot(grid, density, color="gray", lw=5)
                for method, color in (("EVON", "red"), ("IVON", "black")):
                    result = results[method]
                    points, normal = normal_curve(grid, result["mean"][row, k],
                                                   result["std"][row, k])
                    ax.plot(points, normal, color=color, lw=1.5)
                ax.set(xlim=(grid[0], grid[-1]), ylim=(0, density.max() + .01))
                ax.axis("off")
                name = f"alpha_{k + 1}" if row == draws.shape[1] - 1 else f"beta_{row + 1}_{k + 1}"
                fig.savefig(FIGURES_DIR / f"demo_hmc_{name}.png", bbox_inches="tight")
                plt.close(fig)
        fig, ax = plt.subplots()
        for method, color in (("EVON", "red"), ("IVON", "black")):
            result = results[method]
            ax.plot(result["steps"], result["objectives"], color=color, label=method)
        ax.set(xlabel="Step", ylabel="Variational objective")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        fig.savefig(FIGURES_DIR / "demo_hmc_convergence.png", bbox_inches="tight")
        plt.close(fig)


def main():
    torch.set_num_threads(1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RECOMPUTE or not DRAWS_FILE.exists():
        from cmdstanpy import cmdstan_path

        cmdstan_path()
    iris = load_iris()
    data = StandardScaler().fit_transform(iris.data)
    labels = iris.target.astype(np.int64)
    if RECOMPUTE or not RESULTS_FILE.exists():
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.int64)
        results = {method: train(method, x, y) for method in ("EVON", "IVON")}
        np.savez(RESULTS_FILE, **{f"{method.lower()}_{key}": value
                 for method, result in results.items() for key, value in result.items()})
    else:
        with np.load(RESULTS_FILE) as saved:
            results = {method: {key: saved[f"{method.lower()}_{key}"]
                       for key in ("mean", "std", "steps", "objectives")}
                       for method in ("EVON", "IVON")}
    if RECOMPUTE or not DRAWS_FILE.exists():
        draws = sample_stan(data, labels)
        np.savez(DRAWS_FILE, beta=draws[:, :-1], alpha=draws[:, -1])
    else:
        with np.load(DRAWS_FILE) as saved:
            draws = np.concatenate((saved["beta"], saved["alpha"][:, None, :]), axis=1)
    plot(results, draws)
    print(f"Saved HMC plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
