import argparse
import sys
import time
from contextlib import nullcontext

import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(value).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_mc_samples_list(mc_samples_list_str, fallback_mc_samples):
    if not mc_samples_list_str.strip():
        if fallback_mc_samples < 1:
            raise ValueError("--mc_samples must be >= 1")
        return [fallback_mc_samples]

    values = []
    for tok in mc_samples_list_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = int(tok)
        if value < 1:
            raise ValueError("All MC sample counts must be >= 1")
        values.append(value)

    if not values:
        raise ValueError("--mc_samples_list did not contain valid values")

    deduped = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


class SimpleProgress:
    def __init__(self, total, desc, enabled=True, every=25):
        self.total = max(1, int(total))
        self.desc = desc
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.count = 0
        self.start = time.time()
        self._last_msg_len = 0

    def _emit(self, force=False):
        if not self.enabled:
            return
        if not force and self.count % self.every != 0 and self.count < self.total:
            return
        elapsed = max(1e-6, time.time() - self.start)
        frac = min(1.0, self.count / self.total)
        width = 24
        filled = int(round(width * frac))
        bar = "=" * filled + "." * (width - filled)
        rate = self.count / elapsed
        remaining = max(0.0, self.total - self.count)
        eta = int(remaining / max(rate, 1e-9))
        msg = (
            f"{self.desc} [{bar}] {self.count}/{self.total} "
            f"({100.0 * frac:5.1f}%) eta={eta:4d}s"
        )
        pad = max(0, self._last_msg_len - len(msg))
        sys.stdout.write("\r" + msg + (" " * pad))
        sys.stdout.flush()
        self._last_msg_len = len(msg)

    def update(self, n=1):
        self.count = min(self.total, self.count + int(n))
        self._emit(force=False)

    def close(self):
        self.count = self.total
        self._emit(force=True)
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


def sampled_params_context(optimizer, train=False):
    if optimizer is None or not hasattr(optimizer, "sampled_params"):
        return nullcontext()
    try:
        return optimizer.sampled_params(train=train)
    except TypeError:
        return optimizer.sampled_params()


def cast_dtype_from_string(value):
    if isinstance(value, torch.dtype):
        return value
    value = str(value).lower().strip()
    if value.startswith("torch."):
        value = value.split(".", 1)[1]
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported cast_dtype: {value}")


def move_optimizer_state_to_device(optimizer, device):
    def _move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            for k, v in value.items():
                value[k] = _move(v)
            return value
        if isinstance(value, list):
            for i, v in enumerate(value):
                value[i] = _move(v)
            return value
        if isinstance(value, tuple):
            return tuple(_move(v) for v in value)
        return value

    _move(optimizer.state)
    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if key == "params":
                continue
            group[key] = _move(value)

    if hasattr(optimizer, "_get_param_configs"):
        optimizer._numel, optimizer._device, optimizer._dtype = optimizer._get_param_configs()
