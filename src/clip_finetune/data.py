from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

_HF_CACHE_DIR: str | None = os.getenv("HF_DATASETS_CACHE")
_HF_TOKEN: str | None = os.getenv("HF_TOKEN")

import datasets as hf_datasets
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL")

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 128

DATASET_TO_FT_EPOCHS: dict[str, int] = {
    "eurosat": 12,
    "dtd": 76,
    "cars": 35,
    "sun397": 14,
    "svhn": 4,
    "resisc45": 15,
    "mnist": 5,
    "gtsrb": 11,
    "fer2013": 10,
    "pcam": 1,
    "cifar100": 6,
    "flowers102": 147,
    "oxfordiiitpet": 82,
    "stl10": 60,
    "kmnist": 5,
    "emnist": 2,
    "renderedsst2": 39,
    "fashionmnist": 5,
    "food101": 4,
    "cifar10": 6,
    "tinyimagenet": 5,
    "imagenet": 3,
}

DATASET_TO_CHECKPOINT_EPOCHS: dict[str, list[int]] = {
    "eurosat": [1, 6, 12],
    "dtd": [1, 38, 76],
    "cars": [1, 17, 35],
    "sun397": [1, 7, 14],
    "svhn": [1, 2, 4],
    "resisc45": [1, 7, 15],
    "mnist": [1, 2, 5],
    "gtsrb": [1, 5, 11],
    "cifar100": [1, 3, 6],
    "flowers102": [1, 73, 147],
    "oxfordiiitpet": [1, 41, 82],
    "stl10": [1, 30, 60],
    "fer2013": [1, 5, 10],
    "pcam": [1, 1, 1],
    "kmnist": [1, 2, 5],
    "emnist": [1, 2, 2],
    "fashionmnist": [1, 2, 5],
    "food101": [1, 2, 4],
    "cifar10": [1, 3, 6],
    "renderedsst2": [1, 19, 39],
    "tinyimagenet": [1, 3, 5],
    "imagenet": [1, 2, 3],
}

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

@dataclass
class _DatasetConfig:
    hf_path: str
    hf_name: str | None = None
    train_split: str = "train"
    test_split: str = "test"
    image_col: str = "image"
    label_col: str = "label"
    drop_cols: list[str] = field(default_factory=list)
    trust_remote_code: bool = False
    token: str | None = None

_REGISTRY: dict[str, _DatasetConfig] = {
    "imagenet": _DatasetConfig("ILSVRC/imagenet-1k", test_split="validation", token=_HF_TOKEN),
    "sun397": _DatasetConfig("tanganke/sun397"),
    "cars": _DatasetConfig("tanganke/stanford_cars"),
    "resisc45": _DatasetConfig("tanganke/resisc45"),
    "eurosat": _DatasetConfig("tanganke/eurosat"),
    "svhn": _DatasetConfig("ufldl-stanford/svhn", hf_name="cropped_digits"),
    "gtsrb": _DatasetConfig("tanganke/gtsrb"),
    "mnist": _DatasetConfig("ylecun/mnist"),
    "dtd": _DatasetConfig("tanganke/dtd"),
    "cifar100": _DatasetConfig(
        "uoft-cs/cifar100",
        image_col="img",
        label_col="fine_label",
        drop_cols=["coarse_label"],
    ),
    "flowers102": _DatasetConfig("dpdl-benchmark/oxford_flowers102"),
    "oxfordiiitpet": _DatasetConfig("timm/oxford-iiit-pet"),
    "pcam": _DatasetConfig("1aurent/PatchCamelyon"),
    "fer2013": _DatasetConfig(
        "clip-benchmark/wds_fer2013",
        image_col="jpg",
        label_col="cls",
        drop_cols=["__key__", "__url__"],
    ),
    "emnist": _DatasetConfig("tanganke/emnist_letters"),
    "cifar10": _DatasetConfig("uoft-cs/cifar10", image_col="img"),
    "food101": _DatasetConfig("ethz/food101", test_split="validation"),
    "fashionmnist": _DatasetConfig("zalando-datasets/fashion_mnist"),
    "renderedsst2": _DatasetConfig("nateraw/rendered-sst2"),
    "kmnist": _DatasetConfig("tanganke/kmnist"),
    "stl10": _DatasetConfig("tanganke/stl10"),
    "tinyimagenet": _DatasetConfig("zh-plus/tiny-imagenet", test_split="valid"),
}

def _get_default_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])

def _ensure_rgb_pil(img: Any) -> Image.Image:
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    if not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type {type(img)!r}; expected PIL.Image.Image or bytes")
    return img.convert("RGB")

def _ensure_3ch_tensor(img: Tensor) -> Tensor:
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.ndim != 3:
        raise RuntimeError(f"Expected image tensor with shape [C,H,W], got {tuple(img.shape)}")
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    if img.shape[0] != 3:
        raise RuntimeError(f"Expected image tensor with 1 or 3 channels, got shape {tuple(img.shape)}")
    return img

def _make_transform_fn(transform: Callable) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def fn(batch: dict[str, Any]) -> dict[str, Any]:
        images = []
        for img in batch["image"]:
            transformed = transform(_ensure_rgb_pil(img))
            if isinstance(transformed, Tensor):
                transformed = _ensure_3ch_tensor(transformed)
            images.append(transformed)
        batch["image"] = images
        return batch
    return fn

def _is_corrupt_hf_arrow_cache_error(exc: OSError) -> bool:
    msg = str(exc)
    return "Expected to be able to read" in msg and "message body" in msg

def _is_missing_hf_cache_file_error(exc: FileNotFoundError) -> bool:
    msg = str(exc)
    return "hf://" in msg or "datasets--" in msg

def _load_hf_dataset_with_cache_recovery(**load_kwargs: Any) -> hf_datasets.Dataset:
    try:
        return hf_datasets.load_dataset(**load_kwargs)
    except FileNotFoundError as exc:
        if not _is_missing_hf_cache_file_error(exc):
            raise
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["download_mode"] = "force_redownload"
        warnings.warn("Detected a missing HuggingFace cache file; retrying with force_redownload.")
        return hf_datasets.load_dataset(**retry_kwargs)
    except OSError as exc:
        if not _is_corrupt_hf_arrow_cache_error(exc):
            raise
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["download_mode"] = "force_redownload"
        warnings.warn("Detected a corrupted HuggingFace Arrow cache shard; retrying with force_redownload.")
        return hf_datasets.load_dataset(**retry_kwargs)

def _load_hf_split(
    cfg: _DatasetConfig,
    split: str,
    transform: Callable,
) -> hf_datasets.Dataset:
    load_kwargs: dict[str, Any] = {"path": cfg.hf_path, "split": split}
    if cfg.hf_name is not None:
        load_kwargs["name"] = cfg.hf_name
    if cfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if cfg.token is not None:
        load_kwargs["token"] = cfg.token
    if _HF_CACHE_DIR is not None:
        load_kwargs["cache_dir"] = _HF_CACHE_DIR

    ds: hf_datasets.Dataset = _load_hf_dataset_with_cache_recovery(**load_kwargs)

    if cfg.image_col != "image":
        ds = ds.rename_column(cfg.image_col, "image")
    if cfg.label_col != "label":
        ds = ds.rename_column(cfg.label_col, "label")

    cols_to_drop = [c for c in cfg.drop_cols if c in ds.column_names]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)

    ds = ds.with_transform(_make_transform_fn(transform))
    return ds

def _collate_fn(batch: list[dict]) -> tuple[Tensor, Tensor]:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels

def get_dataloaders(
    name: str,
    *,
    batch_size: int = 64,
    transform: Callable | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int | None = 2,
    shuffle_train: bool = True,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")

    cfg = _REGISTRY[name]

    if transform is None:
        transform = _get_default_transform()

    train_ds = _load_hf_split(cfg, cfg.train_split, transform)
    test_ds = _load_hf_split(cfg, cfg.test_split, transform)

    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, shuffle=shuffle_train, generator=generator, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

_CLASS_NAMES_OVERRIDE: dict[str, list[str]] = {
    "fer2013": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "pcam": ["normal", "tumor"],
    "svhn": [str(i) for i in range(10)],
}

def get_class_names(name: str) -> list[str]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")

    if name in _CLASS_NAMES_OVERRIDE:
        return _CLASS_NAMES_OVERRIDE[name]

    cfg = _REGISTRY[name]
    load_kwargs: dict[str, Any] = {"path": cfg.hf_path, "split": cfg.train_split}
    if cfg.hf_name is not None:
        load_kwargs["name"] = cfg.hf_name
    if cfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if cfg.token is not None:
        load_kwargs["token"] = cfg.token
    if _HF_CACHE_DIR is not None:
        load_kwargs["cache_dir"] = _HF_CACHE_DIR

    ds: hf_datasets.Dataset = _load_hf_dataset_with_cache_recovery(**load_kwargs)
    label_col = cfg.label_col

    feature = ds.features.get(label_col)
    if feature is not None and hasattr(feature, "names"):
        return list(feature.names)

    return sorted({str(v) for v in ds[label_col]})
