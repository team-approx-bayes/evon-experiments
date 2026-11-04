""" Shared helpers
"""

from contextlib import nullcontext

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """Seed NumPy and Torch, optionally requesting deterministic cuDNN kernels."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _ivon_parameters(optimizer):
    """Yield each trainable parameter and its slice of IVON's flat curvature."""
    for group in optimizer.param_groups:
        offset = 0
        for parameter in group["params"]:
            if parameter is None:
                continue
            end = offset + parameter.numel()
            if parameter.requires_grad:
                hessian = group.get("hess")
                curvature = (hessian[offset:end] if hessian is not None else
                             torch.full_like(parameter, group["hess_init"]).flatten())
                yield group, parameter, curvature
            offset = end


def get_ivon_covariance(optimizer, model=None, eps=0.0):
    """Return unit-temperature covariance blocks, with optional precision eps."""
    names = {} if model is None else {id(p): name for name, p in model.named_parameters()}
    return {
        names.get(id(parameter), parameter): torch.diag(
            1 / (group["ess"] * (curvature + group["weight_decay"]) + eps)
        )
        for group, parameter, curvature in _ivon_parameters(optimizer)
    }


def get_ivon_kl(optimizer, omit_constants=False):
    """KL of IVON's unit-temperature Gaussian against its isotropic prior."""
    return sum(
        compute_kl_term(curvature, parameter.detach(), group["ess"],
                        group["weight_decay"], omit_constants)
        for group, parameter, curvature in _ivon_parameters(optimizer)
        if group["ess"] != float("inf")
    )


def get_posterior_covariance(optimizer, model=None, eps=None):
    """Read EVON or IVON covariance blocks at unit temperature.

    With ``eps=None``, EVON retains its group precision epsilon and IVON uses 0.
    """
    from vonsoap.optimizers import EVON, IVON

    if isinstance(optimizer, EVON):
        return get_evon_covariance(optimizer, model, eps)
    if isinstance(optimizer, IVON):
        return get_ivon_covariance(optimizer, model, 0.0 if eps is None else eps)
    raise TypeError(f"Unsupported posterior optimizer: {type(optimizer).__name__}")


def get_posterior_kl(optimizer, omit_constants=False):
    """Read EVON or IVON KL at unit temperature, with matching constants."""
    from vonsoap.optimizers import EVON, IVON

    if isinstance(optimizer, EVON):
        return get_evon_kl(optimizer, omit_constants)
    if isinstance(optimizer, IVON):
        return get_ivon_kl(optimizer, omit_constants)
    raise TypeError(f"Unsupported posterior optimizer: {type(optimizer).__name__}")


@torch.no_grad()
def expected_loss(model, optimizer, x, y, loss_fn, samples, *, preserve_rng=True):
    """Estimate posterior loss without changing parameters or sampling statistics."""
    if samples < 1:
        raise ValueError("samples must be positive")
    devices = sorted({t.device.index for t in (*model.parameters(), *model.buffers())
                      if t.is_cuda})
    context = torch.random.fork_rng(devices=devices) if preserve_rng else nullcontext()
    total = 0.0
    with context:
        for _ in range(samples):
            with optimizer.sampled_params(train=False):
                total += loss_fn(model(x), y).item()
    return total / samples


@torch.no_grad()
def sample_model_outputs(model, optimizer, x, samples):
    """Return mean-weight outputs and posterior draws on CPU, preserving shape."""
    if samples < 1:
        raise ValueError("samples must be positive")
    model.eval()
    mean = model(x).detach().cpu().clone()
    draws = []
    for _ in range(samples):
        with optimizer.sampled_params(train=False):
            draws.append(model(x).detach().cpu().clone())
    draws = torch.stack(draws)
    if not torch.isfinite(mean).all() or not torch.isfinite(draws).all():
        raise FloatingPointError("Non-finite posterior predictions.")
    return mean, draws

def get_evon_covariance(optimizer: "EVON", model=None, eps: float | None = None) -> dict:
        """Full DxD posterior covariance for each parameter (or named parameter).

        Constructs ``Cov = P diag(1 / (ess * (h + wd))) P^T`` where ``P`` is the
        Kronecker product of the Shampoo eigenvector matrices.

        Args:
            model: Optional ``nn.Module``; keys in the returned dict become
                parameter names instead of parameter objects.
            eps: Small value added to the diagonal; defaults to group ``eps``.

        Returns:
            Dict mapping parameter (or name) to (D, D) covariance.
        """
        param_names: dict[int, str] = {}
        if model is not None:
            for name, param in model.named_parameters():
                param_names[id(param)] = name

        cov_dict: dict = {}
        for group in optimizer.param_groups:
            _eps = group["eps"] if eps is None else eps
            ess = group.get("ess", optimizer._ess)
            wd = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = optimizer.state[p]
                if "h_mom" not in state:
                    continue

                h = state["h_mom"]
                V = 1.0 / (ess * (h + wd) + _eps)

                matrices = state.get("Q")
                if matrices and not (len(matrices) == 1 and len(matrices[0]) == 0):
                    P = matrices[0].clone()
                    for mat in matrices[1:]:
                        P = torch.kron(P, mat)
                else:
                    P = None

                C_diag = torch.diag(V.flatten())
                if P is not None:
                    P = P.to(dtype=C_diag.dtype, device=C_diag.device)
                    C = P @ C_diag @ P.mT
                else:
                    C = C_diag

                cov_dict[param_names.get(id(p), p)] = C

        return cov_dict

def compute_kl_term(
    h: torch.Tensor,
    m: torch.Tensor,
    ess: float,
    wd: float,
    omit_constants: bool = False,
) -> float:
    """
    Compute KL divergence KL(q||p) for Gaussian posterior q vs isotropic Gaussian prior p.

    This function computes the KL divergence between:
    - q: Gaussian posterior with mean m and diagonal covariance Σ = diag(1 / (ess * (H + wd)))
    - p: Isotropic Gaussian prior N(0, 1 / (ess * wd * I))

    Args:
        h: Hessian diagonal or eigenvalues (precision parameters)
        m: Mean of the posterior distribution
        ess: Effective sample size
        wd: Weight decay (L2 regularization strength)
        omit_constants: If True, uses simplified formula from demo2d.py that omits additive constants. If False, uses exact KL derivation.

    Returns:
        KL divergence value (scalar)

    Notes:
        When omit_constants=True, uses:
            KL = 0.5 * delta * sum(m^2) + 0.5 * delta * sum(σ^2) - 0.5 * sum(log(σ^2))
            where delta = wd * ess and σ^2 = 1 / (ess * (h + wd))

        When omit_constants=False, uses exact KL:
            KL = 0.5 * [sum(log((h+wd)/wd) + wd/(h+wd) - 1) + ess * wd * sum(m^2)]
    """
    if omit_constants:
        delta = wd * ess

        # sigma^2 = 1 / (ess * (h + wd))
        precision = ess * (h + wd)
        sigma2 = 1.0 / precision

        # Term 1: 0.5 * delta * sum(m^2)
        term1 = 0.5 * delta * (m**2).sum().item()

        # Term 2: 0.5 * delta * sum(sigma^2)
        term2 = 0.5 * delta * sigma2.sum().item()

        # Term 3: -0.5 * sum(log(sigma^2))
        term3 = -0.5 * torch.log(sigma2).sum().item()

        return term1 + term2 + term3

    else:
        # Exact KL derivation
        h_plus_wd = h + wd
        curvature_term = (torch.log(h_plus_wd / wd) + wd / h_plus_wd - 1.0).sum().item()

        # Mean term
        mean_term = ess * wd * (m**2).sum().item()

        return 0.5 * (curvature_term + mean_term)

def get_evon_kl(optimizer, omit_constants: bool = False) -> float:
    """
    Compute the KL divergence of EVON's posterior to the prior.

    This correctly computes the exact KL divergence KL(q || p) natively in Shampoo's eigenspaces.
    - q: Gaussian posterior N(m, Q Σ Q^T) where Q is the orthogonal eigenvector matrix from Shampoo
            and Σ is the diagonal matrix of variances 1 / (ess * (h + wd)).
    - p: Isotropic Gaussian prior N(0, σ_p^2 I) with σ_p^2 = 1 / (ess * wd).

    Since the prior consists of an isotropic covariance matrix (it is a scalar multiple of the identity matrix), the KL divergence evaluation is entirely rotation-invariant to Q.
    Mathematically, the trace term tr(Σ_p^-1 Σ_q) = tr(1/σ_p^2 * I * Q Σ Q^T) simplifies cyclically to tr(1/σ_p^2 * Σ Q^T Q) = tr(1/σ_p^2 * Σ). Similarly, the determinant |Q Σ Q^T| resolves to |Σ|.
    Therefore, computing the KL penalty locally on the projected eigenspace variances matches the true parameter-space KL divergence perfectly.

    Args:
        omit_constants (bool): Whether to omit constant terms from the KL divergence.

    Returns:
        float: The KL divergence.
    """
    import math

    total_kl = 0.0
    for group in optimizer.param_groups:
        ess = group.get("ess", optimizer._ess)
        if ess == float("inf"):
            continue

        wd = group.get("weight_decay", 0.0)
        hess_init_val = group.get("hess_init", optimizer._hess_init)

        for p in group["params"]:
            if p.requires_grad:
                state = optimizer.state.get(p, {})
                if "h_mom" in state:
                    h = state["h_mom"]
                    total_kl += compute_kl_term(h, p.data, ess, wd, omit_constants)
                else:
                    if omit_constants:
                        delta = wd * ess
                        sigma2 = 1.0 / (ess * (hess_init_val + wd))
                        term1 = 0.5 * delta * (p.data**2).sum().item()
                        term2 = 0.5 * delta * (sigma2 * p.numel())
                        term3 = -0.5 * (math.log(sigma2) * p.numel())
                        total_kl += term1 + term2 + term3
                    else:
                        h_plus_wd = hess_init_val + wd
                        curvature_term = (
                            math.log(h_plus_wd / wd) + wd / h_plus_wd - 1.0
                        ) * p.numel()
                        mean_term = ess * wd * (p.data**2).sum().item()
                        total_kl += 0.5 * (curvature_term + mean_term)
    return total_kl
