"""USPS 3-vs-5 

Attribution: SLANG by Aaron Mishkin, Frederik Kunstner, Didrik Nielsen,
Mark Schmidt and Mohammad Emtiyaz Khan (NeurIPS 2018), arXiv:1811.04504.
Code based on: https://github.com/aaronpmishkin/SLANG (2563276dedf4d18c77f3195e8645d3133be8d14c)
Files adapted or consulted (paths relative to that repository):
- Low-rank precision updates, sampling and momentum:
  code/python/libs/vi_lib/lib/optimizers/slang.py
- Experiment settings, prior and original USPS split:
  code/matlab/experiments/reproduce_log_reg_table.m
  code/matlab/lib/utils/experiment_runners/run_experiment.m
  code/matlab/lib/utils/experiment_runners/get_expt_params.m
  code/matlab/lib/log_reg/get_data_log_reg.m
  code/matlab/lib/utils/general_utilities/setSeed.m
- Full Gaussian Cholesky objective and L-BFGS reference setup:
  code/matlab/lib/log_reg/baseline_infer.m
  code/matlab/lib/log_reg/call_log_reg.m
  code/matlab/lib/log_reg/chol_log_reg.m
  This port uses SciPy L-BFGS-B instead of MATLAB minFunc.
- Logistic bound coefficients and truncated-Gaussian moment derivatives:
  code/matlab/lib/log_reg/E_log_p.m (written by Emtiyaz Khan).
- Covariance palette, color normalization and method titles:
  code/python/run_and_plot_scripts/logreg_covviz/logreg_covviz_plotter.py
  code/python/libs/vi_lib/lib/utilities/plotting.py
"""

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import get_evon_covariance, get_ivon_covariance
from vonsoap.optimizers.evon import EVON
from vonsoap.optimizers.ivon import IVON

PIXEL_PRIOR, BIAS_PRIOR = 25.0, 1e-5  # Precisions, not variances.
BATCH_SIZE = 64
MC_SAMPLES = 12
EPOCHS = 10000
RETRAIN = False
DIRECTORY = Path(__file__).resolve().parent
OUTPUT = DIRECTORY / "figures" / Path(__file__).stem
NAMES = ("Full Gaussian", "EVON", "IVON", "SLANG (Rank 1)", "SLANG (Rank 5)", "SLANG (Rank 10)")
RANKS = (None, None, None, 1, 5, 10)
FILENAMES = ("full_gaussian", "evon", "ivon", "slang_1", "slang_5", "slang_10")
COLORS = ("#242B36", "#D84A3A", "#9099A5", "#78B4E3", "#357FC0", "#143F78")
MARKERS = (None, "o", "s", "^", "D", "v")


class SLANG:
    """Paper Eqs. 7–11: low-rank precision, diagonal correction, momentum.

    Uses an exact small SVD in place of the original randomized eigensolver.
    Individual observation gradients, rather than the batch-average outer
    product, estimate the empirical Fisher.
    """

    def __init__(self, prior, rank, n):
        self.prior = torch.as_tensor(prior, dtype=torch.float64)
        self.mean = torch.zeros_like(self.prior)
        self.factor = torch.zeros(len(prior), rank, dtype=torch.float64)
        self.diagonal = torch.ones_like(self.mean)  # Authors' initial precision.
        self.momentum = torch.zeros_like(self.mean)
        self.rank, self.n = rank, n

    def solve(self, rhs):
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        scaled_factor = self.factor / self.diagonal[:, None]
        small = torch.eye(self.rank, dtype=self.mean.dtype) + self.factor.T @ scaled_factor
        result = rhs / self.diagonal[:, None]
        result = result - scaled_factor @ torch.linalg.solve(small, self.factor.T @ result)
        return result[:, 0] if vector else result

    def covariance(self):
        covariance = self.solve(torch.eye(len(self.mean), dtype=self.mean.dtype))
        return (covariance + covariance.T) / 2

    def step(self, x, y, step):
        # A random RHS with covariance equal to the precision gives posterior
        # noise after solving that precision system.
        rhs = self.diagonal.sqrt()[:, None] * torch.randn(len(self.mean), MC_SAMPLES, dtype=self.mean.dtype)
        rhs += self.factor @ torch.randn(self.rank, MC_SAMPLES, dtype=self.mean.dtype)
        theta = self.mean[:, None] + self.solve(rhs)
        residual = (x @ theta).sigmoid() - y[:, None]
        # This is exactly the MC-average individual-gradient Fisher, compressed
        # to one column per observation since every gradient is residual * x_i.
        fisher_factor = x.T * residual.square().mean(1).sqrt()[None, :]
        rate = 0.05 / (1 + step**0.51)  # Original alpha_t = beta_t.
        columns = torch.cat(((1 - rate)**0.5 * self.factor,
                             (rate * self.n / len(x))**0.5 * fisher_factor), dim=1)
        vectors, values, _ = torch.linalg.svd(columns, full_matrices=False)
        factor = vectors[:, :self.rank] * values[:self.rank]
        correction = (columns.square().sum(1) - factor.square().sum(1)).clamp_min(0)
        self.diagonal = (1 - rate) * self.diagonal + rate * self.prior + correction
        self.factor = factor
        gradient = self.n / len(x) * (x.T @ residual.mean(1))
        self.momentum = 0.9 * self.momentum + 0.1 * gradient
        direction = self.momentum / (1 - 0.9**step) + self.prior * self.mean
        self.mean -= rate * self.solve(direction)


def expected_softplus(mean, variance, bound, derivatives=False):
    """The authors' 20-piece quadratic logistic bound and analytic derivatives."""
    c, b, a, low, high = bound[:, :, None]
    m, v = np.asarray(mean)[None, :], np.asarray(variance)[None, :]
    std = np.sqrt(v)
    zl, zh = (low-m)/std, (high-m)/std
    pl = np.exp(-0.5*zl**2)/(np.sqrt(2*np.pi)*std)
    ph = np.exp(-0.5*zh**2)/(np.sqrt(2*np.pi)*std)
    mass = ndtr(zh)-ndtr(zl)
    low, high = low.copy(), high.copy()
    low[0], high[-1] = 0.0, 0.0  # Limits of infinity * Gaussian density.
    first = v*(pl-ph) + m*mass
    second = v*((low+m)*pl-(high+m)*ph) + (v+m*m)*mass
    value = np.sum(a*second+b*first+c*mass, axis=0)
    if not derivatives:
        return value
    dm = a*((low*low+2*v)*pl-(high*high+2*v)*ph)+2*a*m*mass
    dm += b*(low*pl-high*ph)+b*mass+c*(pl-ph)
    t1, t2 = 2*v*low+low**3-low*low*m, 2*v*high+high**3-high*high*m
    dv = a/(2*v)*(t1*pl-t2*ph)+a*mass
    dv += b/(2*v)*((low*low+v-low*m)*pl-(high*high+v-high*m)*ph)
    dv += c/(2*v)*((low-m)*pl-(high-m)*ph)
    return value, dm.sum(axis=0), dv.sum(axis=0)


def objective(x, y, mean, covariance, prior, bound):
    """Original negative ELBO; the authors omit the bias log-prior constant."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    logits = x @ mean
    variance = np.einsum("ij,ij->i", x @ covariance, x)
    kl = 0.5*(-np.linalg.slogdet(covariance)[1]-np.log(prior[:-1]).sum()
              + np.dot(np.diag(covariance), prior)+np.dot(mean*mean, prior)-mean.size)
    return float(kl + np.sum(expected_softplus(logits, variance, bound)-y*logits))


def full_gaussian(x, y, prior, bound):
    """Original full Gaussian L-BFGS reference, from zero mean/identity covariance."""
    dim = x.shape[1]
    row, col = np.triu_indices(dim)

    def fun(parameters):
        mean = parameters[:dim]
        upper = np.zeros((dim, dim))
        upper[row, col] = parameters[dim:]
        diagonal = np.diag(upper)
        projected = x @ upper.T
        variance = np.einsum("ij,ij->i", projected, projected)
        logits = x @ mean
        softplus, dm, dv = expected_softplus(logits, variance, bound, True)
        kl = 0.5*(-2*np.log(diagonal).sum()-np.log(prior[:-1]).sum()
                  + np.sum(upper*upper*prior)+np.dot(mean*mean, prior)-dim)
        value = kl + np.sum(softplus-y*logits)
        grad_mean = prior*mean + x.T@(dm-y)
        grad_upper = upper*prior + 2*upper@(x.T@(dv[:, None]*x))
        grad_upper[np.diag_indices(dim)] -= 1/diagonal
        return value, np.r_[grad_mean, grad_upper[row, col]]

    start = np.r_[np.zeros(dim), np.eye(dim)[row, col]]
    bounds = [(None, None)]*dim + [(1e-8, None) if r == c else (None, None) for r, c in zip(row, col)]
    result = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=1500, maxcor=30, ftol=1e-13, gtol=1e-7, maxls=30))
    upper = np.zeros((dim, dim))
    upper[row, col] = result.x[dim:]
    return result.x[:dim], upper.T@upper, result


def cosine_rate(step, start, end, warm=0):
    """Hold the initial rate, decay to step 2,500, then hold the final rate."""
    phase = np.clip((step-1-warm)/(2500-warm), 0, 1)
    return end + (start-end)*0.5*(1+np.cos(np.pi*phase))


def train():
    # Original first 50/50 USPS split; pixels in [-1, 1], bias column last.
    with np.load(DIRECTORY / "data" / "usps_3vs5.npz") as data:
        xn, yn = data["x_train"], data["y_train"]
    with np.load(DIRECTORY / "data" / "logisticbound.npz") as data:
        bound = data["coefficients"]  # Rows: constant, linear, quadratic, lower, upper.
    prior = np.r_[np.full(xn.shape[1]-1, PIXEL_PRIOR), BIAS_PRIOR]
    x, y = torch.from_numpy(xn), torch.from_numpy(yn)
    evaluate = lambda mean, covariance: objective(xn, yn, mean, covariance, prior, bound)
    steps = EPOCHS * len(x) // BATCH_SIZE
    logged_steps = sorted({step for step in
                           (0, 1, 2, 5, 10, steps, *range(25, 3001, 25), *range(1000, steps+1, 1000))
                           if step <= steps})
    log_steps = set(logged_steps)
    mean, covariance, reference_fit = full_gaussian(xn, yn, prior, bound)
    means, covariances, histories = [mean], [covariance], []
    reference = evaluate(mean, covariance)
    print(f"Full Gaussian: {reference:.6f} ({reference_fit.message})", flush=True)
    for name, rank in zip(NAMES[1:], RANKS[1:]):
        torch.manual_seed(2)
        batches = torch.Generator().manual_seed(1)
        if rank is not None:
            optimizer = SLANG(prior, rank, len(x))
            mean, covariance = optimizer.mean, optimizer.covariance()
        else:
            parameter = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=x.dtype))
            # Scalar weight decay supplies the bias prior. The residual pixel
            # prior is included in the sampled gradient, retaining cross terms.
            common = dict(ess=len(x), hess_init=(1-BIAS_PRIOR)/len(x),
                          weight_decay=BIAS_PRIOR/len(x), mc_samples=MC_SAMPLES, sync=False)
            if name == "EVON":
                optimizer = EVON([parameter], **common, lr=0.03, betas=(0.9, 0.99),
                                 shampoo_beta=0.999, debias_beta2=False, cast_dtype=None, eps=0,
                                 precondition_1d=True, precondition_frequency=25,
                                 correct_bias=False, whiten_prec_grad=False)
            else:
                optimizer = IVON([parameter], **common, lr=0.005, beta2=0.995,
                                 clip_radius=0.1, rescale_lr=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps) if name == "IVON" else None
            mean = parameter.detach()
            covariance = torch.eye(x.shape[1], dtype=x.dtype)
        history = [evaluate(mean, covariance)]
        for step in range(1, steps + 1):
            if name == "EVON":
                for group in optimizer.param_groups:
                    group.update(lr=cosine_rate(step, 0.03, 0.001),
                                 betas=(0.9, 1-cosine_rate(step, 0.01, 0.001, warm=250)))
            elif name == "IVON":
                for group in optimizer.param_groups:
                    group["beta2"] = 1-cosine_rate(step, 0.005, 0.0005, warm=250)
            batch = torch.randint(len(x), (BATCH_SIZE,), generator=batches)
            with torch.no_grad():
                if rank is not None:
                    optimizer.step(x[batch], y[batch], step)
                    mean = optimizer.mean
                else:
                    xb, yb = x[batch], y[batch]
                    for _ in range(MC_SAMPLES):
                        with optimizer.sampled_params(train=True):
                            # Exact gradient of mean NLL + residual pixel prior.
                            parameter.grad = xb.T@((xb@parameter).sigmoid()-yb)/BATCH_SIZE
                            parameter.grad[:-1] += (PIXEL_PRIOR-BIAS_PRIOR)/len(x)*parameter[:-1]
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    mean = parameter.detach()
                if step in log_steps:
                    if rank is not None:
                        covariance = optimizer.covariance()
                    elif name == "EVON":
                        covariance = get_evon_covariance(optimizer, eps=0)[parameter]
                    else:
                        covariance = get_ivon_covariance(optimizer)[parameter]
            if step in log_steps:
                value = evaluate(mean, covariance)
                history.append(value)
                if step % 10000 == 0 or step == steps:
                    print(f"{name}: step {step}, objective {value:.4f}", flush=True)
        means.append(mean.detach().numpy().copy())
        covariances.append(covariance.numpy())
        histories.append(history)
    return dict(means=np.stack(means), covariances=np.stack(covariances),
                steps=np.array(logged_steps), objectives=np.array(histories), reference=np.array(reference))


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "text.color": "#666666", "axes.labelcolor": "#666666",
                         "xtick.color": "#777777", "ytick.color": "#777777",
                         "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12, 5.1), facecolor="white")
    panels = fig.add_gridspec(2, 3, left=0.09, right=0.60, bottom=0.20,
                             top=0.94, hspace=0.38, wspace=0.10)
    # The paper crops seismic to the reference's off-diagonal range while
    # preserving symmetric color intensity about zero (plotting.py above).
    # Retain diagonals so IVON's structure is visible; they saturate at pink.
    matrices = results["covariances"][:, :-1, :-1].copy()
    off_diagonal = ~np.eye(matrices.shape[1], dtype=bool)
    low, high = matrices[0][off_diagonal].min(), matrices[0][off_diagonal].max()
    amplitude = max(abs(low), abs(high))
    colors = plt.get_cmap("seismic")(np.linspace(0.5+low/(2*amplitude), 0.5+high/(2*amplitude), 257))
    cmap = LinearSegmentedColormap.from_list("slang_covariance", colors)
    norm = Normalize(vmin=low, vmax=high)
    for index, (name, covariance) in enumerate(zip(NAMES, matrices)):
        panel = fig.add_subplot(panels[index // 3, index % 3])
        image = panel.imshow(covariance, origin="upper", cmap=cmap,
                             norm=norm, interpolation="nearest")
        panel.set(xticks=[], yticks=[])
        panel.set_title(name, fontsize=13, color="#666666", pad=7)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, -2))
    colorbar = fig.colorbar(image, cax=fig.add_axes([0.035, 0.20, 0.019, 0.74]),
                            ticks=np.arange(-0.004, 0.00101, 0.001), format=formatter)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_offset_position("left")
    colorbar.ax.tick_params(labelsize=11, length=3, width=0.6)
    colorbar.outline.set_visible(False)

    ax = fig.add_axes([0.685, 0.20, 0.295, 0.58])
    reference = float(results["reference"])
    end = min(2500, results["steps"][-1])
    visible = results["steps"] <= end
    steps = results["steps"][visible]
    marker_indices = sorted({int(np.abs(steps-step).argmin()) for step in np.linspace(0, end, 7)[1:-1]})
    ax.axhline(reference, color=COLORS[0], ls=(0, (5, 3)), lw=1.5, label="Full Gaussian")
    for name, values, color, marker in zip(NAMES[1:], results["objectives"], COLORS[1:], MARKERS[1:]):
        values = values[visible]
        ax.plot(steps, values, label=name, color=color, lw=2.0 if name != "IVON" else 1.6,
                marker=marker, markevery=marker_indices, ms=4.5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.0)
        ax.plot(steps[-1], values[-1], marker=marker, color=color, ms=7.5,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, clip_on=False)
    ax.set(xlabel="Training step", ylabel="Variational objective",
           xlim=(0, end), ylim=(140, 250))
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_yticks([150, 175, 200, 225, 250])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, labelsize=13)
    ax.grid(axis="y", color="#E8ECF1", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, 1.04, 1.18, 0.12),
              mode="expand", ncol=2, frameon=False, fontsize=13,
              handlelength=1.4, handletextpad=0.4, columnspacing=0.6, labelspacing=0.35)

    # Save untitled covariance PDFs for arranging the panels separately.
    for filename, covariance in zip(FILENAMES, matrices):
        raw = plt.figure(figsize=(2.5, 2.5))
        raw_ax = raw.add_axes([0, 0, 1, 1])
        raw_ax.imshow(covariance, origin="upper", cmap=cmap, norm=norm,
                      interpolation="nearest")
        raw_ax.set_axis_off()
        raw.savefig(OUTPUT.with_name(f"{OUTPUT.name}_covariance_{filename}.pdf"), dpi=300, pad_inches=0)
        plt.close(raw)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for filename, plot_ax in (("convergence", ax), ("colorbar", colorbar.ax)):
        bounds = plot_ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.04)
        fig.savefig(OUTPUT.with_name(f"{OUTPUT.name}_{filename}.pdf"), bbox_inches=bounds, dpi=300)
    for extension in ("pdf", "png"):
        path = OUTPUT.with_name(f"{OUTPUT.name}_panel.{extension}")
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved individual plots and combined panel to {OUTPUT.parent}")
    plt.close(fig)


def main():
    # Small matrix updates are faster with one CPU thread.
    torch.set_num_threads(1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = OUTPUT.with_suffix(".npz")
    if path.exists() and not RETRAIN:
        with np.load(path) as saved:
            results = dict(saved)
    else:
        results = train()
        np.savez_compressed(path, **results)
    plot(results)


if __name__ == "__main__":
    main()
