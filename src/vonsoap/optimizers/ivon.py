from math import pow
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    hessian_approx_methods = (
        "price",
        "gradsq",
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError("Invalid number of MC samples: {}".format(mc_samples))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError("Invalid Hessian initialization: {}".format(hess_init))
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                f"Parameters are on different devices: {[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                f"Parameters are on different dtypes: {[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel  # sanity check

        if train:  # collect grad sample for training
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], grad_sample, count
            )
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(
                    self.state["avg_nxg"], noise * grad_sample, count
                )
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(
                    self.state["avg_gsq"], grad_sample.square(), count
                )

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self.sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]  # sanity check
        assert offset == self._numel  # sanity check

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self):
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat(
                [p.flatten() for p in group["params"] if p is not None], 0
            )

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"])
                if self.rescale_lr
                else lr,
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"],
            )

            # update params
            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]  # sanity check
            offset += group["numel"]
        assert offset == self._numel  # sanity check

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == "price":
            return avg_nxg[pg_slice] * hess
        elif method == "gradsq":
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd) -> Tensor:
        f = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return (
            beta2 * hess
            + (1.0 - beta2) * f
            + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)
        )

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )

    @torch.no_grad()
    def get_covariance(self, model=None, eps: float = 1e-10):
        """
        Computes the full covariance matrix for the flattened (block-wise) parameters with size DxD.
        Warning: This constructs a D x D matrix where D is the parameter size.

        Args:
            model: Optional nn.Module. If provided, returns a dict with parameter names as keys.
            eps: Small value to add to the diagonal of the covariance matrix to prevent it from being singular.

        Returns:
            Dictionary mapping parameter (or parameter name) to its DxD covariance matrix.

        Example:
            >>> optimizer = IVON(model.parameters(), lr=0.1, ess=100.0)
            >>> optimizer.step()
            >>> cov_dict = optimizer.get_covariance(model=model)
            >>> layer1_cov = cov_dict["layer1.weight"]
            >>> print(layer1_cov.shape)
            torch.Size([12, 12])
        """
        param_names = {}
        if model is not None:
            for name, param in model.named_parameters():
                param_names[id(param)] = name

        cov_dict = {}
        for group in self.param_groups:
            if "hess" not in group:
                continue

            hess = group["hess"]

            offset = 0
            for p in group["params"]:
                if p is None:
                    continue

                numel = p.numel()
                if not p.requires_grad:
                    offset += numel
                    continue

                h_mom = hess[offset : offset + numel]

                # Empirical variance
                V = 1.0 / (h_mom + eps)

                # Preconditioner is diagonal, so C is just a diagonal matrix
                C = torch.diag(V)

                key = param_names.get(id(p), p)
                cov_dict[key] = C

                offset += numel

        return cov_dict

    def get_kl(self, omit_constants: bool = False) -> float:
        """
        Compute the KL divergence.

        This computes the exact KL divergence KL(q || p) between the posterior q and the prior p:
        - q: Gaussian posterior N(m, Σ_q) with diagonal covariance Σ_q = diag(1 / (ess * (H + wd)))
        - p: Isotropic Gaussian prior N(0, Σ_p) with covariance Σ_p = (1 / (ess * wd)) * I

        The computation separates into:
        1. A mean term: 0.5 * ||m||^2 / σ_p^2
        2. A sum of trace and determinant terms over the parameters.

        Args:
            omit_constants (bool): Whether to omit constant terms from the KL divergence.

        Returns:
            float: The KL divergence.
        """
        from vonsoap.optimizers.utils import compute_kl_term
        import math

        total_kl = 0.0
        for group in self.param_groups:
            ess = group["ess"]
            if ess == float("inf"):
                continue

            wd = group["weight_decay"]
            hess = group.get("hess")
            hess_init_val = group.get("hess_init", 1.0)

            offset = 0
            for p in group["params"]:
                if p is None:
                    continue
                numel = p.numel()
                if p.requires_grad:
                    if hess is not None:
                        h = hess[offset : offset + numel]
                        m = p.data.flatten()
                        total_kl += compute_kl_term(h, m, ess, wd, omit_constants)
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
                            mean_term = 0.5 * ess * wd * (p.data**2).sum().item()
                            total_kl += 0.5 * (curvature_term + mean_term)
                offset += numel
        return total_kl
