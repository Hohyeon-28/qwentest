import pytest
import torch

from src.quant_utils import fingerprint_quant_tensor_dict


def quant_tuple():
    generator = torch.Generator().manual_seed(123)
    return {
        "model.layer.qweight": torch.randint(
            -(2**31), 2**31 - 1, (16, 32), dtype=torch.int32, generator=generator
        ),
        "model.layer.scales": torch.rand(
            1, 32, dtype=torch.float16, generator=generator
        ),
        "model.layer.qzeros": torch.randint(
            -(2**31), 2**31 - 1, (1, 4), dtype=torch.int32, generator=generator
        ),
        "model.layer.g_idx": torch.zeros(128, dtype=torch.int32),
    }


def test_identical_q_s_z_g_have_identical_fingerprint():
    tensors = quant_tuple()
    clone = {name: tensor.clone() for name, tensor in tensors.items()}
    fake = fingerprint_quant_tensor_dict(tensors, checkpoint="same")
    real = fingerprint_quant_tensor_dict(clone, checkpoint="same")
    assert fake["tuple_sha256"] == real["tuple_sha256"]
    assert fake["component_sha256"] == real["component_sha256"]


@pytest.mark.parametrize(
    "component", ("qweight", "scales", "qzeros", "g_idx")
)
def test_change_in_any_component_changes_tuple_fingerprint(component):
    tensors = quant_tuple()
    baseline = fingerprint_quant_tensor_dict(tensors)
    changed = {name: tensor.clone() for name, tensor in tensors.items()}
    key = next(name for name in changed if name.endswith(f".{component}"))
    changed[key].view(-1)[0] += 1
    candidate = fingerprint_quant_tensor_dict(changed)
    assert candidate["tuple_sha256"] != baseline["tuple_sha256"]


def test_missing_component_fails_closed():
    tensors = quant_tuple()
    tensors.pop("model.layer.g_idx")
    with pytest.raises(ValueError, match="incomplete"):
        fingerprint_quant_tensor_dict(tensors)
