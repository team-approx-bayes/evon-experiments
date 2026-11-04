""" Inbetween uncertainty
Red: posterior functions; gray: mean-weight function; crosses: observations.
"""

from pathlib import Path
import sys

import torch
from torch import nn

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))
from vonsoap.optimizers import EVON, IVON
from utils import sample_model_outputs, set_seed


SEED = 42
SAMPLES = 100
GRID_POINTS = 500
DEVICE = "cpu"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
NETWORKS = {"Large NN": (64, "elu", 4), "Small NN": (8, "tanh", 2)}
JOINT_BIASES = {"Large NN": False, "Small NN": True}
STEPS = {"Large NN": 10_000, "Small NN": 30_000}
# ESS controls posterior uncertainty and stays fixed throughout each fit.
POSTERIORS = {
    "Large NN": {
        "ivon": dict(lr=1., ess=100_000., hess_init=.01, beta2=.9995),
        "evon": dict(lr=.1, ess=10_000., weight_decay=1e-4,
                     betas=(.9, .999), shampoo_beta=.9999),
    },
    "Small NN": {
        "ivon": dict(lr=10., ess=10_000., hess_init=.05, beta2=.999),
        "evon": dict(lr=.01, ess=30_000., weight_decay=5e-7,
                     betas=(.9, .999), shampoo_beta=.99,
                     precondition_frequency=5, merge_dims=True,
                     price_clip_ratio=100),
    },
}
IVON_SETTINGS = dict(weight_decay=1e-5, clip_radius=.1, sync=False)
EVON_SETTINGS = dict(
    hess_init=.05, precondition_1d=True, whiten_prec_grad=True, sync=False,
)


def make_data(device):
    """Five noisy observations on each side of an unobserved interval."""
    x = torch.linspace(.35, .65, 20)
    noise = torch.logspace(-3, 0, 20) * torch.randn_like(x)
    y = 5 * torch.sin(10 * x) + noise
    x, y = (torch.cat((t[:5], t[-5:]))[:, None].to(device) for t in (x, y))
    return x, y


class JointLinear(nn.Module):
    """Store a layer's weights and biases in one block."""

    def __init__(self, linear):
        super().__init__()
        self.theta = nn.Parameter(torch.cat(
            (linear.weight.detach(), linear.bias.detach()[:, None]), dim=1,
        ))

    def forward(self, x):
        return nn.functional.linear(x, self.theta[:, :-1], self.theta[:, -1])


def make_model(hidden=8, activation="tanh", depth=2, *, joint_bias=False):
    nonlinearity = {"tanh": nn.Tanh, "elu": nn.ELU}[activation]
    layers = []
    for layer in range(depth):
        layers.extend((nn.Linear(1 if layer == 0 else hidden, hidden), nonlinearity()))
    layers.append(nn.Linear(hidden, 1))
    if joint_bias:
        layers = [JointLinear(layer) if isinstance(layer, nn.Linear) else layer for layer in layers]
    return nn.Sequential(*layers)


def make_optimizer(method, model, network="Small NN"):
    settings = POSTERIORS[network][method]
    if method == "evon":
        return EVON(model.parameters(), **(EVON_SETTINGS | settings))
    if method == "ivon":
        return IVON(model.parameters(), **IVON_SETTINGS, **settings)
    raise ValueError(f"Unknown optimizer: {method}")


def train(model, optimizer, x, y, steps):
    """One posterior sample and one optimizer update per cosine-scheduled step."""
    model.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for step in range(steps):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}.")
            loss.backward()
        optimizer.step()
        scheduler.step()


def sample_predictions(model, optimizer, x, samples):
    mean, draws = sample_model_outputs(model, optimizer, x, samples)
    return mean.squeeze(-1), draws.squeeze(-1)


def save_plots(results, x, y, grid, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for network in NETWORKS:
        size = network.split()[0].lower()
        for method in ("ivon", "evon"):
            fig = plt.figure(figsize=(3.1, 1.6))
            ax = fig.add_axes((0, 0, 1, 1))
            result = results[network][method]
            ax.plot(grid, result["draws"].T, color="red", alpha=.10, linewidth=.8)
            ax.plot(grid, result["mean_weights"], color="gray", linewidth=2.)
            ax.plot(x, y, "kx", markersize=5, markeredgewidth=1.2)
            ax.set(xlim=(0, 1), ylim=(-8, 4))
            ax.set_axis_off()
            path = output_dir / f"demo_inbetween_{size}_{method}.png"
            fig.savefig(path, transparent=True)
            print(f"Saved {path}")
            plt.close(fig)


def main():
    device = torch.device(DEVICE)
    if device.type == "cpu":
        torch.set_num_threads(1)
    results = {}
    for network, architecture in NETWORKS.items():
        results[network] = {}
        for method in ("ivon", "evon"):
            # Both methods start with the same data and network coefficients.
            set_seed(SEED)
            x, y = make_data(device)
            model = make_model(*architecture, joint_bias=JOINT_BIASES[network]).to(device)
            optimizer = make_optimizer(method, model, network)
            print(f"{network}: {method.upper()}", flush=True)
            train(model, optimizer, x, y, STEPS[network])
            grid = torch.linspace(0, 1, GRID_POINTS, device=device)[:, None]
            mean, draws = sample_predictions(model, optimizer, grid, SAMPLES)
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(x), y).item()
            results[network][method] = dict(mean_weights=mean, draws=draws)
            print(f"  mean-weight training MSE={mse:.5f}", flush=True)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, grid = (t.cpu().squeeze(-1) for t in (x, y, grid))
    save_plots(results, x, y, grid, output_dir)


if __name__ == "__main__":
    main()
