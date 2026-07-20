from __future__ import annotations

from typing import Any


MODAL_CANONICAL_OBJECTIVE = "valid_rgb_canonical_v1"
MODAL_CANONICAL_GAUSSIAN_CONTROL = "disabled"
MODAL_CANONICAL_TRAINABLE_FIELDS = (
    "means",
    "colors",
    "opacities",
    "scales",
    "quats",
)
MODAL_CANONICAL_TRAINABLE_NAMES = frozenset(
    f"fg.params.{field}" for field in MODAL_CANONICAL_TRAINABLE_FIELDS
)


def configure_canonical_only_trainability(model: Any) -> None:
    """Freeze a modal model except for its shared foreground canonical field."""
    if model.trajectory_type != "modal_activation" or not model.has_modal:
        raise ValueError("canonical-only optimization requires a modal model")
    model.requires_grad_(False)
    missing = [
        field
        for field in MODAL_CANONICAL_TRAINABLE_FIELDS
        if field not in model.fg.params
    ]
    if missing:
        raise ValueError(
            "Foreground canonical is missing required parameters: "
            + ", ".join(missing)
        )
    for field in MODAL_CANONICAL_TRAINABLE_FIELDS:
        model.fg.params[field].requires_grad_(True)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if trainable_names != MODAL_CANONICAL_TRAINABLE_NAMES:
        raise ValueError(
            "Canonical-only optimization received unexpected trainable parameters: "
            f"{sorted(trainable_names)}"
        )
