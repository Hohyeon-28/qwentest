from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


COMPONENT_SUFFIXES = {
    "q": ".qweight",
    "s": ".scales",
    "z": ".qzeros",
    "g": ".g_idx",
}


def validate_shared_quant_config(config: dict[str, Any]) -> None:
    shared = config["quantization"]["shared"]
    fake = config["quantization"]["fake"]
    real = config["quantization"]["real"]
    if shared.get("source") != "models.real_gptq":
        raise ValueError("Shared (q,s,z,g) source must be models.real_gptq")
    if fake.get("method") != "dequantize_shared_gptq":
        raise ValueError("Fake method must be dequantize_shared_gptq")
    if real.get("method") != "gptq":
        raise ValueError("Real method must be gptq")
    for key in ("bits", "group_size", "symmetric", "desc_act"):
        if fake.get(key) != real.get(key):
            raise ValueError(f"Fake and Real quantization setting mismatch: {key}")
    if int(shared["bits"]) != 4 or int(fake["bits"]) != 4:
        raise ValueError("This experiment requires 4-bit q")
    if int(shared["group_size"]) != 128:
        raise ValueError("This experiment requires group_size=128")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _tensor_digest(name: str, tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def fingerprint_quant_tensor_dict(
    tensors: Mapping[str, torch.Tensor],
    *,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fingerprint serialized GPTQ tensors defining the logical (q,s,z,g) tuple."""

    layers: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        component = next(
            (
                short_name
                for short_name, suffix in COMPONENT_SUFFIXES.items()
                if name.endswith(suffix)
            ),
            None,
        )
        if component is None:
            continue
        suffix = COMPONENT_SUFFIXES[component]
        layer_name = name[: -len(suffix)]
        tensor = tensors[name]
        layers.setdefault(layer_name, {})[component] = {
            "tensor_name": name,
            "sha256": _tensor_digest(name, tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }

    return _finalize_manifest(layers, checkpoint=checkpoint)


def _finalize_manifest(
    layers: dict[str, dict[str, Any]],
    *,
    checkpoint: str | None,
) -> dict[str, Any]:
    if not layers:
        raise ValueError("No GPTQ qweight/scales/qzeros/g_idx tensors were found")
    incomplete = {
        layer: sorted(set(COMPONENT_SUFFIXES) - set(components))
        for layer, components in layers.items()
        if set(components) != set(COMPONENT_SUFFIXES)
    }
    if incomplete:
        preview = dict(list(incomplete.items())[:8])
        raise ValueError(
            "Every quantized layer must expose the same (q,s,z,g) tuple; "
            f"incomplete layers (first 8): {preview}"
        )

    global_digest = hashlib.sha256()
    component_digests = {component: hashlib.sha256() for component in COMPONENT_SUFFIXES}
    total_bytes = 0
    for layer_name in sorted(layers):
        global_digest.update(layer_name.encode("utf-8"))
        for component in ("q", "s", "z", "g"):
            item = layers[layer_name][component]
            encoded = item["sha256"].encode("ascii")
            global_digest.update(component.encode("ascii"))
            global_digest.update(encoded)
            component_digests[component].update(layer_name.encode("utf-8"))
            component_digests[component].update(encoded)
            total_bytes += int(item["bytes"])

    return {
        "checkpoint": checkpoint,
        "definition": "(q,s,z,g) serialized in the standard GPTQ checkpoint",
        "quantized_layers": len(layers),
        "total_tuple_bytes": total_bytes,
        "tuple_sha256": global_digest.hexdigest(),
        "component_sha256": {
            component: digest.hexdigest()
            for component, digest in component_digests.items()
        },
        "layers": layers,
    }


def resolve_checkpoint_path(model_id_or_path: str, revision: str = "main") -> Path:
    local = Path(model_id_or_path).expanduser()
    if local.exists():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id_or_path,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors"],
        )
    ).resolve()


def fingerprint_gptq_checkpoint(
    model_id_or_path: str,
    *,
    revision: str = "main",
) -> dict[str, Any]:
    from safetensors import safe_open

    checkpoint_path = resolve_checkpoint_path(model_id_or_path, revision)
    files = sorted(checkpoint_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No safetensors files found in GPTQ checkpoint: {checkpoint_path}"
        )
    layers: dict[str, dict[str, Any]] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                component = next(
                    (
                        short_name
                        for short_name, suffix in COMPONENT_SUFFIXES.items()
                        if key.endswith(suffix)
                    ),
                    None,
                )
                if component is None:
                    continue
                suffix = COMPONENT_SUFFIXES[component]
                layer_name = key[: -len(suffix)]
                if component in layers.setdefault(layer_name, {}):
                    raise ValueError(f"Duplicate GPTQ tensor key across shards: {key}")
                tensor = handle.get_tensor(key)
                layers[layer_name][component] = {
                    "tensor_name": key,
                    "sha256": _tensor_digest(key, tensor),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
                del tensor
    manifest = _finalize_manifest(layers, checkpoint=model_id_or_path)
    manifest["resolved_checkpoint_path"] = str(checkpoint_path)
    manifest["safetensor_files"] = [path.name for path in files]
    return manifest


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Fake compute dtype: {name}")
    return mapping[normalized]


@torch.no_grad()
def dequantize_torch_quant_model_(
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """Replace GPTQModel TorchLinear modules with dense Linear modules.

    TorchLinear.dequantize_weight() evaluates exactly
    `s[g] * (q - z[g])` from the checkpoint's qweight, scales, qzeros and
    g_idx. The transpose converts GPTQ's [in, out] logical matrix to
    torch.nn.Linear's [out, in] storage.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    names = [
        name for name, module in model.named_modules() if isinstance(module, TorchLinear)
    ]
    if not names:
        raise RuntimeError(
            "No GPTQModel TorchLinear modules found. Load with BACKEND.GPTQ_TORCH."
        )
    reports: list[dict[str, Any]] = []
    for name in names:
        module = model.get_submodule(name)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        logical_weight = module.dequantize_weight().T.detach().to("cpu", dtype)
        new_module = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="cpu",
            dtype=dtype,
        )
        new_module.weight = torch.nn.Parameter(logical_weight, requires_grad=False)
        if module.bias is not None:
            new_module.bias = torch.nn.Parameter(
                module.bias.detach().to("cpu", dtype), requires_grad=False
            )
        setattr(parent, child_name, new_module)
        reports.append(
            {
                "name": name,
                "shape": list(new_module.weight.shape),
                "dtype": str(dtype),
                "source": "(q,s,z,g) from shared GPTQ checkpoint",
                "operation": "W_fake = s[g] * (q - z[g])",
            }
        )
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")
    return reports


def load_shared_gptq_fake_model(
    config: dict[str, Any],
    *,
    device: str = "cuda",
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    """Load the Real checkpoint with a pure Torch GPTQ decoder, then densify."""

    from gptqmodel import BACKEND, GPTQModel

    validate_shared_quant_config(config)
    checkpoint = config["models"]["real_gptq"]
    loaded = GPTQModel.load(
        model_id_or_path=checkpoint,
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    model = getattr(loaded, "model", loaded)
    dtype = _torch_dtype(config["quantization"]["fake"]["compute_dtype"])
    reports = dequantize_torch_quant_model_(model, dtype=dtype)
    model = model.to(device=device, dtype=dtype).eval()
    return model, reports
