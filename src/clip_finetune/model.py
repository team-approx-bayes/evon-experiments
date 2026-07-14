from __future__ import annotations

import re
import timm
import torch
import torch.nn as nn
import open_clip
from tqdm import tqdm
from .templates import dataset_to_template

def canonicalize_timm_model_name(model_name: str) -> str:
    """Normalize legacy model names to timm's current naming."""
    if not model_name.endswith(".openai_clip"):
        return model_name

    arch = model_name[: -len(".openai_clip")]
    if "_clip_" not in arch:
        arch = re.sub(r"_(\d+)$", r"_clip_\1", arch)
    return f"{arch}.openai"

def _uses_clip_projection(model_name: str) -> bool:
    """Return True when timm model output should keep the CLIP projection head."""
    if "." not in model_name:
        return False
    arch, _tag = model_name.rsplit(".", 1)
    return "_clip_" in arch

def _model_output_dim(model: nn.Module) -> int:
    """Return the forward() output dimensionality used as embedding size."""
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.out_features)
    return int(model.num_features)

def _strip_text_tower_if_present(model: nn.Module) -> list[str]:
    """Remove text-side CLIP modules when present to reduce memory footprint."""
    removed: list[str] = []
    text_attrs = (
        "transformer",
        "token_embedding",
        "positional_embedding",
        "ln_final",
        "text_projection",
        "text",
    )
    for attr in text_attrs:
        if hasattr(model, attr):
            delattr(model, attr)
            removed.append(attr)
    return removed

def load_timm_model(
    model_name: str,
    device: torch.device,
    expected_embed_dim: int | None = None,
    strip_text_tower: bool = True,
) -> tuple[nn.Module, int, list[str]]:
    resolved_name = canonicalize_timm_model_name(model_name)
    if _uses_clip_projection(resolved_name):
        model = timm.create_model(resolved_name, pretrained=True)
    else:
        model = timm.create_model(resolved_name, pretrained=True, num_classes=0)

    removed_text_attrs: list[str] = []
    if strip_text_tower:
        removed_text_attrs = _strip_text_tower_if_present(model)

    embed_dim = _model_output_dim(model)
    if expected_embed_dim is not None and embed_dim != expected_embed_dim:
        raise RuntimeError(
            f"Model {resolved_name!r} has embed_dim={embed_dim}, but classifier head expects "
            f"embed_dim={expected_embed_dim}."
        )

    model = model.to(device)
    return model, embed_dim, removed_text_attrs

def freeze_for_finetune_fp(model: nn.Module, trainable_scope: str = "all") -> None:
    """Configure which model parameters are trainable during fine-tuning."""
    if trainable_scope not in {"all", "linear"}:
        raise ValueError(
            f"Unsupported trainable_scope={trainable_scope!r}. Use 'all' or 'linear'."
        )

    if trainable_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        # Step 1 — freeze everything.
        for p in model.parameters():
            p.requires_grad = False

        # Step 2 — unfreeze .weight of every nn.Linear.
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = True
    
    # freeze classifier head (if it exists).
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier.weight.requires_grad = False
        if model.classifier.bias is not None:
            model.classifier.bias.requires_grad = False

    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nPARAMETER FREEZE MAP:")
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"  [{status}]  {name}")

    print(
        f"  Trainable : {trainable_params:>12,} ({100 * trainable_params / total_params:.2f}%)\n"
        f"  Frozen    : {total_params - trainable_params:>12,}\n"
        f"  Total     : {total_params:>12,}"
    )

    if trainable_scope == "linear":
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name.endswith(".weight"), (
                    f"Unexpected trainable non-weight param: {name}"
                )
                parent_name = name[: -len(".weight")]
                parent = model.get_submodule(parent_name)
                assert isinstance(parent, nn.Linear), (
                    f"Unexpected trainable .weight of non-Linear module: {name}"
                )
        print("  [OK] Assertions passed: only nn.Linear weight matrices are trainable.\n")
    else:
        print("  [OK] Assertions passed: all parameters are trainable.\n")

def resolve_clip_text_model(model_name: str) -> tuple[str, str]:
    """Map timm model name to matching OpenCLIP text model architecture and pretrained tag."""
    model_name = canonicalize_timm_model_name(model_name)
    if "vit_base_patch16" in model_name:
        return "ViT-B-16", "openai"
    elif "vit_large_patch14" in model_name:
        if "336" in model_name:
            return "ViT-L-14-336", "openai"
        return "ViT-L-14", "openai"
    # Default fallback
    return "ViT-B-16", "openai"

def build_clip_head(
    clip_model_name: str,
    dataset_name: str,
    class_names: list[str],
    device: torch.device,
    clip_pretrained: str = "openai",
    show_progress: bool = True,
) -> torch.Tensor:
    """Build a CLIP zero-shot classification head locally."""
    dataset_name_map = {
        "cars": "Cars",
        "dtd": "DTD",
        "eurosat": "EuroSAT",
        "gtsrb": "GTSRB",
        "mnist": "MNIST",
        "resisc45": "RESISC45",
        "sun397": "SUN397",
        "svhn": "SVHN",
        "cifar100": "CIFAR100",
        "stl10": "STL10",
        "flowers102": "Flowers102",
        "oxfordiiitpet": "OxfordIIITPet",
        "pcam": "PCAM",
        "fer2013": "FER2013",
        "emnist": "EMNIST",
        "cifar10": "CIFAR10",
        "food101": "Food101",
        "fashionmnist": "FashionMNIST",
        "renderedsst2": "RenderedSST2",
        "kmnist": "KMNIST",
        "imagenet": "ImageNet",
        "tinyimagenet": "ImageNet",
    }
    pascal_name = dataset_name_map.get(dataset_name.lower())
    if not pascal_name:
        raise KeyError(f"Unsupported dataset: {dataset_name}")
    templates = dataset_to_template[pascal_name]

    # Load CLIP model text encoder
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    try:
        tokenizer = open_clip.get_tokenizer(clip_model_name)
    except AttributeError:
        tokenizer = open_clip.tokenize

    zeroshot_weights = []
    with torch.no_grad():
        iterable = tqdm(class_names, desc=f"CLIP head [{pascal_name}]", leave=False) if show_progress else class_names
        for classname in iterable:
            texts = [t(classname) for t in templates]
            tokens = tokenizer(texts).to(device)
            embeddings = clip_model.encode_text(tokens)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embedding = embeddings.mean(dim=0)
            embedding = embedding / embedding.norm()
            zeroshot_weights.append(embedding)

    weights = torch.stack(zeroshot_weights, dim=0).float()
    clip_model.cpu()
    del clip_model
    torch.cuda.empty_cache()
    return weights
