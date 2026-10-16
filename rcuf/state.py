"""State operators in HF layout [batch, value_head, key_dim, value_dim]."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


def recurrent_states(cache) -> dict[int, torch.Tensor]:
    """Require the pinned Transformers 5.19 DynamicCache interface."""
    result = {}
    if not hasattr(cache, "layers"):
        raise TypeError("Unsupported cache. Install transformers==5.19.0; legacy caches are unsafe here.")
    for i, layer in enumerate(cache.layers):
        states = getattr(layer, "recurrent_states", None)
        if states is not None and states.get(0) is not None:
            state = states[0]
            if state.ndim != 4 or state.dtype != torch.float32:
                raise ValueError(
                    f"Layer {i}: expected a four-dimensional FP32 recurrent state, got {state.shape}/{state.dtype}"
                )
            result[i] = state
    if not result:
        raise ValueError("No recurrent states found. Qwen3-4B is a pure Transformer; use Qwen3.5-4B.")
    return result


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            setattr(value, key, _move(item, device))
    return value


def clone_cache(cache, device=None):
    result = copy.deepcopy(cache)
    return result if device is None else _move(result, device)


def capture_states(cache) -> dict[int, torch.Tensor]:
    return {i: state.detach().cpu().clone() for i, state in recurrent_states(cache).items()}


@dataclass
class Anchor:
    french: torch.Tensor
    english: torch.Tensor
    difference: torch.Tensor
    basis: torch.Tensor
    low_rank_difference: torch.Tensor
    retained_energy: torch.Tensor
    effective_rank: torch.Tensor


@torch.no_grad()
def make_anchor(french: torch.Tensor, english: torch.Tensor, rank: int) -> Anchor:
    if french.shape != english.shape or french.ndim != 4 or rank > french.shape[-2]:
        raise ValueError("Anchor shape/rank mismatch.")
    difference = french.float() - english.float()
    # Key dimension is the LEFT matrix axis in HF, so use left singular vectors.
    u, singular, _ = torch.linalg.svd(difference, full_matrices=False)
    basis = u[..., :rank]
    # Zero differences have arbitrary singular vectors: never restore their unrelated state components.
    active = singular[..., :rank] > (singular[..., :1] * 1e-6).clamp_min(1e-12)
    basis = basis * active.unsqueeze(-2)
    low_rank = basis @ (basis.transpose(-1, -2) @ difference)
    energy = singular[..., :rank].square().sum(-1) / singular.square().sum(-1).clamp_min(1e-30)
    return Anchor(french.float(), english.float(), difference, basis, low_rank, energy, active.sum(-1))


@torch.no_grad()
def edit_state(state: torch.Tensor, anchor: Anchor, method: str, strength: float) -> torch.Tensor:
    if method == "add_full":
        delta = strength * anchor.difference
    elif method == "add_lowrank":
        delta = strength * anchor.low_rank_difference
    elif method in {"replace", "wrong_anchor", "replace_norm_matched"}:
        target = anchor.english if method == "wrong_anchor" else anchor.french
        u = anchor.basis
        delta = strength * (u @ (u.transpose(-1, -2) @ (target - state)))
        if method == "replace_norm_matched":
            # Match the additive intervention's Frobenius norm in EACH layer,
            # using this unedited state. This soft replacement is not idempotent.
            wanted = abs(strength) * anchor.low_rank_difference.double().norm()
            native = delta.double().norm()
            if native == 0 and wanted > 0:
                raise ValueError("Cannot match a positive norm with a zero replacement direction.")
            delta = delta * (wanted / native.clamp_min(1e-30)).to(delta.dtype)
    else:
        raise ValueError(f"Unknown state method: {method}")
    return state + delta


@torch.no_grad()
def edit_cache(cache, anchors: dict[int, Anchor], method: str, strength: float) -> dict:
    states = recurrent_states(cache)
    if states.keys() != anchors.keys():
        raise ValueError("Cache and anchor layer sets differ.")
    squared_delta = squared_state = 0.0
    layer_audit = {}
    for i, state in states.items():
        updated = edit_state(state, anchors[i], method, strength)
        if not torch.isfinite(updated).all():
            raise ValueError(f"Non-finite edited state in layer {i}")
        actual_norm = (updated - state).double().norm().item()
        squared_delta += actual_norm**2
        squared_state += state.double().square().sum().item()
        if method == "replace_norm_matched":
            wanted = abs(strength) * anchors[i].low_rank_difference.double().norm().item()
            if abs(actual_norm - wanted) > 1e-6 + 1e-5 * wanted:
                raise ValueError(f"Layer {i}: norm matching failed: {actual_norm} vs {wanted}")
            layer_audit[str(i)] = {"actual_norm": actual_norm, "target_additive_norm": wanted}
        state.copy_(updated)
    return {
        "edit_frobenius_norm": squared_delta**0.5,
        "state_frobenius_norm": squared_state**0.5,
        "relative_edit_norm": (squared_delta / max(squared_state, 1e-30)) ** 0.5,
        "edited_layers": len(states),
        **({"norm_match_per_layer": layer_audit} if layer_audit else {}),
    }
