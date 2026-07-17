from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru

from flow3d.data.casual_dataset import CasualDataset
from flow3d.metrics import mSSIM
from flow3d.modal_utils import ModalFrameMap, build_modal_envelope_layout
from flow3d.scene_model import SceneModel


MODAL_SHAPE_PARAMETERIZATION = "anchor_delta_phi_v1"
MODAL_DELTA_PHI_STATE_KEY = "modal_refinement.params.delta_phi"
MODAL_ANCHOR_MASK_STATE_KEY = "modal_anchor_mask"


@dataclass
class ModalReconstructionConfig:
    work_dir: str
    out_dir: str
    ckpt_path: str | None = None
    fps: float = 30.0


@dataclass(frozen=True)
class ReconstructionFrame:
    dataset_index: int
    ts: int
    frame_name: str
    view_id: str
    view_index: int
    local_index: int
    time_sec: float


class _ViewMetricAccumulator:
    def __init__(self, device: torch.device):
        self.frame_count = 0
        self.valid_pixel_count = 0
        self.foreground_pixel_count = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.ssim = mSSIM().to(device)

    def update(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> dict[str, float | int | None]:
        if (
            rendered.shape != target.shape
            or rendered.ndim != 3
            or rendered.shape[-1] != 3
        ):
            raise ValueError(
                "rendered and target images must have matching shape (H, W, 3)"
            )
        if valid_mask.shape != rendered.shape[:2]:
            raise ValueError("valid mask shape does not match rendered image")
        if foreground_mask.shape != rendered.shape[:2]:
            raise ValueError("foreground mask shape does not match rendered image")
        if not bool(torch.isfinite(rendered).all()):
            raise ValueError("rendered image contains non-finite values")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("target image contains non-finite values")

        valid = valid_mask.bool()
        foreground = foreground_mask.bool()
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            raise ValueError("reconstruction frame has no valid pixels")
        if bool((foreground & ~valid).any()):
            raise ValueError("foreground mask contains pixels outside the valid mask")

        diff = rendered - target
        valid_channels = valid[..., None].to(diff.dtype)
        self.frame_count += 1
        self.valid_pixel_count += valid_count
        self.foreground_pixel_count += int(foreground.sum().item())
        absolute_error_sum = float(
            (diff.abs() * valid_channels).sum().detach().cpu().item()
        )
        squared_error_sum = float(
            (diff.square() * valid_channels).sum().detach().cpu().item()
        )
        self.absolute_error_sum += absolute_error_sum
        self.squared_error_sum += squared_error_sum
        self.ssim.update(
            rendered[None],
            target[None],
            valid[None].to(rendered.dtype),
        )
        channel_count = 3 * valid_count
        mean_squared_error = squared_error_sum / channel_count
        return {
            "valid_pixel_count": valid_count,
            "foreground_pixel_count": int(foreground.sum().item()),
            "rgb_l1": absolute_error_sum / channel_count,
            "psnr_db": (
                None
                if mean_squared_error == 0.0
                else -10.0 * math.log10(mean_squared_error)
            ),
            "ssim": float(self.ssim.similarity[-1].detach().cpu().item()),
        }

    def summary(self) -> dict[str, Any]:
        if self.frame_count == 0 or self.valid_pixel_count == 0:
            raise ValueError("cannot summarize an empty reconstruction view")
        channel_count = 3 * self.valid_pixel_count
        mean_squared_error = self.squared_error_sum / channel_count
        psnr_db = (
            None
            if mean_squared_error == 0.0
            else -10.0 * math.log10(mean_squared_error)
        )
        return {
            "frame_count": self.frame_count,
            "valid_pixel_count": self.valid_pixel_count,
            "foreground_pixel_count": self.foreground_pixel_count,
            "rgb_l1": self.absolute_error_sum / channel_count,
            "psnr_db": psnr_db,
            "ssim": float(self.ssim.compute().detach().cpu().item()),
        }


def _validate_modal_shape_checkpoint_contract(
    state_dict: Mapping[str, Any],
    init_metadata: Any,
) -> bool:
    marker = (
        init_metadata.get("modal_shape_parameterization")
        if isinstance(init_metadata, dict)
        else None
    )
    has_delta_phi = MODAL_DELTA_PHI_STATE_KEY in state_dict
    has_anchor_mask = MODAL_ANCHOR_MASK_STATE_KEY in state_dict
    has_any_shape_state = has_delta_phi or has_anchor_mask
    has_complete_shape_state = has_delta_phi and has_anchor_mask

    if marker is None and not has_any_shape_state:
        return False
    has_frozen_envelope = (
        isinstance(init_metadata, dict)
        and init_metadata.get("modal_envelope_frozen") is True
    )
    envelope_source = (
        init_metadata.get("modal_envelope_init_ckpt")
        if isinstance(init_metadata, dict)
        else None
    )
    if (
        marker == MODAL_SHAPE_PARAMETERIZATION
        and has_complete_shape_state
        and has_frozen_envelope
        and isinstance(envelope_source, str)
        and bool(envelope_source)
    ):
        return True
    raise ValueError(
        "Checkpoint has an incomplete or incompatible modal shape-refinement "
        "contract: expected no shape marker/state for Stage 2, or "
        f"modal_shape_parameterization={MODAL_SHAPE_PARAMETERIZATION!r} with "
        f"both {MODAL_DELTA_PHI_STATE_KEY!r} and "
        f"{MODAL_ANCHOR_MASK_STATE_KEY!r}, frozen envelope metadata, and "
        "an envelope source checkpoint"
    )


def _load_training_config(work_dir: Path) -> dict[str, Any]:
    cfg_path = work_dir / "cfg.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must contain a mapping: {cfg_path}")
    return payload


def _matching_config_value(
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    key: str,
) -> Any:
    top_value = train_cfg.get(key)
    data_value = data_cfg.get(key)
    top_missing = top_value is None or top_value == () or top_value == []
    data_missing = data_value is None or data_value == () or data_value == []
    if not top_missing and not data_missing:
        if isinstance(top_value, (list, tuple)) and isinstance(
            data_value, (list, tuple)
        ):
            values_match = tuple(top_value) == tuple(data_value)
        else:
            values_match = top_value == data_value
        if not values_match:
            raise ValueError(
                f"Training config has conflicting top-level and data.{key} values"
            )
    return top_value if data_missing else data_value


def _dataset_kwargs_from_training_config(
    train_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    if train_cfg.get("trajectory_type") != "modal_activation":
        raise ValueError(
            "Modal reconstruction requires trajectory_type='modal_activation'"
        )
    raw_data_cfg = train_cfg.get("data")
    if not isinstance(raw_data_cfg, dict):
        raise ValueError("Training config must contain a data mapping")
    data_cfg = dict(raw_data_cfg)
    if data_cfg.get("camera_type") != "vggt":
        raise ValueError("Modal reconstruction requires data.camera_type='vggt'")

    data_dir_value = data_cfg.get("data_dir")
    if not isinstance(data_dir_value, str) or not data_dir_value:
        raise ValueError("Training config data.data_dir must be a non-empty path")
    data_dir = Path(data_dir_value).expanduser()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    view_configs_value = _matching_config_value(
        train_cfg, data_cfg, "vggt_view_configs"
    )
    if not isinstance(view_configs_value, (list, tuple)) or not view_configs_value:
        raise ValueError("Modal reconstruction requires VGGT view configs")
    view_configs = tuple(
        str(Path(path).expanduser()) for path in view_configs_value
    )
    for view_config in view_configs:
        if not Path(view_config).is_file():
            raise FileNotFoundError(f"VGGT view config does not exist: {view_config}")

    frame_map_value = _matching_config_value(train_cfg, data_cfg, "modal_frame_map")
    if not isinstance(frame_map_value, str) or not frame_map_value:
        raise ValueError("Modal reconstruction requires modal_frame_map")
    frame_map_path = Path(frame_map_value).expanduser()
    if not frame_map_path.is_file():
        raise FileNotFoundError(f"Modal frame map does not exist: {frame_map_path}")

    data_cfg["data_dir"] = str(data_dir)
    data_cfg["vggt_view_configs"] = view_configs
    data_cfg["modal_frame_map"] = str(frame_map_path)
    data_cfg["load_depths"] = False
    data_cfg["load_tracks"] = False
    for key in ("modal_train_view_id", "modal_max_local_frames_per_view"):
        if key not in data_cfg and key in train_cfg:
            data_cfg[key] = train_cfg[key]
    return data_cfg, frame_map_path


def _load_training_dataset(
    train_cfg: Mapping[str, Any],
) -> tuple[CasualDataset, Path]:
    dataset_kwargs, frame_map_path = _dataset_kwargs_from_training_config(train_cfg)
    return CasualDataset(**dataset_kwargs), frame_map_path


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    use_2dgs: bool,
) -> SceneModel:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    if "modal.params.activations" in state_dict:
        raise ValueError(
            "Constant per-view harmonic activation checkpoints are not supported; "
            "reconstruct from a harmonic-envelope checkpoint"
        )
    init_metadata = checkpoint.get("init_metadata")
    if (
        "modal.params.envelope_knots" in state_dict
        and (
            not isinstance(init_metadata, dict)
            or init_metadata.get("modal_parameterization")
            != "per_view_harmonic_envelope_v1"
        )
    ):
        parameterization = (
            init_metadata.get("modal_parameterization")
            if isinstance(init_metadata, dict)
            else None
        )
        raise ValueError(
            "Checkpoint uses an incompatible modal parameterization "
            f"({parameterization!r}); expected "
            "'per_view_harmonic_envelope_v1'"
        )
    if "modal.params.envelope_knots" in state_dict:
        if not isinstance(init_metadata, dict):
            raise ValueError("Envelope checkpoint metadata must be a mapping")
        if init_metadata.get("modal_envelope_interpolation") != "linear_complex":
            raise ValueError(
                "Checkpoint must use linear_complex modal envelope interpolation"
            )
        metadata_interval = init_metadata.get("modal_envelope_knot_interval_sec")
        if (
            isinstance(metadata_interval, bool)
            or not isinstance(metadata_interval, (int, float))
            or not math.isfinite(float(metadata_interval))
            or float(metadata_interval) <= 0.0
        ):
            raise ValueError("Checkpoint metadata has an invalid envelope interval")
        state_interval = state_dict.get("modal_envelope_knot_interval_sec")
        if (
            not isinstance(state_interval, torch.Tensor)
            or state_interval.ndim != 0
            or not np.isclose(
                float(state_interval.item()),
                float(metadata_interval),
                rtol=1.0e-6,
                atol=1.0e-8,
            )
        ):
            raise ValueError(
                "Checkpoint envelope interval state does not match metadata"
            )
    _validate_modal_shape_checkpoint_contract(state_dict, init_metadata)
    try:
        model = SceneModel.init_from_state_dict(state_dict)
    except (AssertionError, KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint cannot restore a SceneModel: {checkpoint_path}"
        ) from exc
    model.use_2dgs = bool(use_2dgs)
    model = model.to(device)
    model.eval()
    if model.trajectory_type != "modal_activation" or model.modal is None:
        raise ValueError("Checkpoint does not contain a modal_activation model")
    return model


def _load_reconstruction_frames(
    frame_map_path: Path,
    frame_names: Sequence[str],
) -> tuple[list[str], dict[str, list[ReconstructionFrame]]]:
    with frame_map_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Modal frame map must contain a mapping: {frame_map_path}")
    version = payload.get("version")
    if version != 1:
        raise ValueError(f"Unsupported modal frame map version: {version!r}")
    view_ids = payload.get("views")
    view_fps_hz = payload.get("view_fps_hz")
    records = payload.get("frames")
    if (
        not isinstance(view_ids, list)
        or not view_ids
        or any(not isinstance(view_id, str) or not view_id for view_id in view_ids)
    ):
        raise ValueError(f"{frame_map_path} must contain non-empty string views")
    if len(set(view_ids)) != len(view_ids):
        raise ValueError(f"{frame_map_path} contains duplicate view ids")
    if any(
        view_id in {".", ".."} or "/" in view_id or "\\" in view_id
        for view_id in view_ids
    ):
        raise ValueError(f"{frame_map_path} contains a path-unsafe view id")
    if not isinstance(view_fps_hz, dict) or set(view_fps_hz) != set(view_ids):
        raise ValueError(
            f"{frame_map_path} view_fps_hz keys must exactly match views"
        )
    validated_view_fps_hz: dict[str, float] = {}
    for view_id in view_ids:
        value = view_fps_hz[view_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid FPS for view_id={view_id!r}")
        fps_hz = float(value)
        if not math.isfinite(fps_hz) or fps_hz <= 0.0:
            raise ValueError(f"Invalid FPS for view_id={view_id!r}")
        validated_view_fps_hz[view_id] = fps_hz
    if not isinstance(records, list) or not records:
        raise ValueError(f"{frame_map_path} must contain non-empty frames")
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("Training dataset contains duplicate frame names")

    by_name: dict[str, tuple[str, int, float]] = {}
    all_local_indices = {view_id: set() for view_id in view_ids}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Modal frame records must be mappings")
        frame_name = record.get("frame_name")
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError("Modal frame record is missing a non-empty frame_name")
        if frame_name in by_name:
            raise ValueError(f"Duplicate frame_name in modal frame map: {frame_name}")
        view_id_value = record.get("view_id")
        if (
            not isinstance(view_id_value, str)
            or view_id_value not in validated_view_fps_hz
        ):
            raise ValueError(
                f"Unknown view_id {view_id_value!r} for frame {frame_name}"
            )
        view_id = view_id_value
        local_index_value = record.get("local_index")
        if isinstance(local_index_value, bool) or not isinstance(
            local_index_value, int
        ):
            raise ValueError(f"Frame record {frame_name} has invalid local_index")
        local_index = int(local_index_value)
        if local_index < 0:
            raise ValueError(f"Frame record {frame_name} has negative local_index")
        if local_index in all_local_indices[view_id]:
            raise ValueError(
                f"Duplicate local_index={local_index} for view_id={view_id!r}"
            )
        time_sec_value = record.get("time_sec")
        if isinstance(time_sec_value, bool) or not isinstance(
            time_sec_value, (int, float)
        ):
            raise ValueError(f"Frame record {frame_name} has invalid time_sec")
        time_sec = float(time_sec_value)
        if not math.isfinite(time_sec) or time_sec < 0.0:
            raise ValueError(f"Frame record {frame_name} has invalid time_sec")
        expected_time_sec = local_index / validated_view_fps_hz[view_id]
        if abs(time_sec - expected_time_sec) > 1.0e-9:
            raise ValueError(
                f"Frame record {frame_name} time_sec is inconsistent with "
                "local_index/view_fps_hz"
            )
        all_local_indices[view_id].add(local_index)
        by_name[frame_name] = (view_id, local_index, time_sec)
    for view_id, local_indices in all_local_indices.items():
        if not local_indices:
            raise ValueError(
                f"Modal frame map view {view_id!r} has no frame records"
            )
        if local_indices != set(range(len(local_indices))):
            raise ValueError(
                f"Modal frame map view {view_id!r} local_index values must be "
                "contiguous from zero"
            )

    view_to_index = {view_id: index for index, view_id in enumerate(view_ids)}
    frames_by_view = {view_id: [] for view_id in view_ids}
    missing = []
    for dataset_index, frame_name in enumerate(frame_names):
        frame_record = by_name.get(frame_name)
        if frame_record is None:
            missing.append(frame_name)
            continue
        view_id, local_index, time_sec = frame_record
        frames_by_view[view_id].append(
            ReconstructionFrame(
                dataset_index=dataset_index,
                ts=dataset_index,
                frame_name=frame_name,
                view_id=view_id,
                view_index=view_to_index[view_id],
                local_index=local_index,
                time_sec=time_sec,
            )
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{frame_map_path} is missing {len(missing)} dataset frames, "
            f"first missing: {preview}"
        )
    for view_id, frames in frames_by_view.items():
        frames.sort(key=lambda frame: frame.local_index)
    if not any(frames_by_view.values()):
        raise ValueError("Modal frame map contains no frames selected by the dataset")
    return list(view_ids), frames_by_view


def _validate_model_alignment(
    model: SceneModel,
    dataset: CasualDataset,
    view_ids: Sequence[str],
    frames_by_view: Mapping[str, Sequence[ReconstructionFrame]],
) -> None:
    frame_count = len(dataset.frame_names)
    if dataset.num_frames != frame_count:
        raise ValueError("Dataset frame_names length does not match num_frames")
    if model.num_frames != frame_count:
        raise ValueError(
            f"Checkpoint has {model.num_frames} frames but dataset has {frame_count}"
        )
    if model.modal is None:
        raise ValueError("Checkpoint has no modal envelope")
    envelope_knots = model.modal.params["envelope_knots"]
    num_modes = int(model.modal_phi_real.shape[0])
    if envelope_knots.ndim != 3 or tuple(envelope_knots.shape[1:]) != (
        num_modes,
        2,
    ):
        raise ValueError(
            "Checkpoint harmonic envelope shape is inconsistent: expected "
            f"(N, {num_modes}, 2), got {tuple(envelope_knots.shape)}"
        )
    if not bool(torch.isfinite(envelope_knots).all()):
        raise ValueError("Checkpoint harmonic envelope contains non-finite values")
    if num_modes == 0:
        raise ValueError("Checkpoint contains no modal modes")
    frequencies_hz = model.modal_freqs_hz.detach().cpu()
    if frequencies_hz.shape != (num_modes,):
        raise ValueError("Checkpoint modal frequency shape is inconsistent")
    if not bool(torch.isfinite(frequencies_hz).all()) or bool(
        (frequencies_hz <= 0.0).any()
    ):
        raise ValueError("Checkpoint modal frequencies must be finite and positive")

    dataset_Ks = dataset.get_Ks().detach().cpu()
    dataset_w2cs = dataset.get_w2cs().detach().cpu()
    checkpoint_Ks = model.Ks.detach().cpu()
    checkpoint_w2cs = model.w2cs.detach().cpu()
    if dataset_Ks.shape != checkpoint_Ks.shape or not torch.allclose(
        dataset_Ks, checkpoint_Ks, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Dataset camera intrinsics do not match checkpoint")
    if dataset_w2cs.shape != checkpoint_w2cs.shape or not torch.allclose(
        dataset_w2cs, checkpoint_w2cs, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Dataset camera extrinsics do not match checkpoint")

    expected_view_indices = torch.full((frame_count,), -1, dtype=torch.long)
    expected_local_indices = torch.full((frame_count,), -1, dtype=torch.long)
    expected_times_sec = torch.full(
        (frame_count,), float("nan"), dtype=torch.float32
    )
    for view_id in view_ids:
        for frame in frames_by_view[view_id]:
            expected_view_indices[frame.ts] = frame.view_index
            expected_local_indices[frame.ts] = frame.local_index
            expected_times_sec[frame.ts] = frame.time_sec
    if bool((expected_view_indices < 0).any()) or bool(
        (expected_local_indices < 0).any()
    ):
        raise ValueError("Reconstruction frame groups do not cover every checkpoint frame")
    if not torch.equal(
        model.modal_frame_view_indices.detach().cpu(), expected_view_indices
    ):
        raise ValueError("Modal frame view indices do not match checkpoint")
    if not torch.equal(
        model.modal_frame_local_indices.detach().cpu(), expected_local_indices
    ):
        raise ValueError("Modal frame local indices do not match checkpoint")
    if not hasattr(model, "modal_frame_times_sec"):
        raise ValueError(
            "Checkpoint has no modal_frame_times_sec and is not a harmonic-envelope "
            "checkpoint"
        )
    actual_times_sec = model.modal_frame_times_sec.detach().cpu()
    if actual_times_sec.shape != expected_times_sec.shape:
        raise ValueError("Modal frame times shape does not match checkpoint")
    if not bool(torch.isfinite(actual_times_sec).all()):
        raise ValueError("Modal frame times contain non-finite values")
    if not torch.equal(
        actual_times_sec.to(dtype=torch.float32), expected_times_sec
    ):
        raise ValueError("Modal frame times do not match checkpoint")

    interval_tensor = model.modal_envelope_knot_interval_sec.detach().cpu()
    if interval_tensor.ndim != 0 or not bool(torch.isfinite(interval_tensor)) or float(
        interval_tensor.item()
    ) <= 0.0:
        raise ValueError("Checkpoint modal envelope knot interval is invalid")
    expected_layout = build_modal_envelope_layout(
        ModalFrameMap(
            view_ids=list(view_ids),
            frame_view_indices=expected_view_indices,
            frame_local_indices=expected_local_indices,
            frame_times_sec=expected_times_sec,
        ),
        float(interval_tensor.item()),
    )
    expected_envelope_state = {
        "modal_envelope_knot_offsets": expected_layout.knot_offsets,
        "modal_envelope_knot_times_sec": expected_layout.knot_times_sec,
        "modal_frame_envelope_left": expected_layout.frame_left_indices,
        "modal_frame_envelope_right": expected_layout.frame_right_indices,
        "modal_frame_envelope_lerp": expected_layout.frame_lerp_weights,
    }
    for name, expected in expected_envelope_state.items():
        actual = getattr(model, name).detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Checkpoint {name} shape/dtype is inconsistent")
        if not torch.equal(actual, expected.detach().cpu()):
            raise ValueError(f"Checkpoint {name} does not match the frame map")
    expected_knot_shape = (
        int(expected_layout.knot_times_sec.shape[0]),
        num_modes,
        2,
    )
    if tuple(envelope_knots.shape) != expected_knot_shape:
        raise ValueError(
            "Checkpoint harmonic envelope shape is inconsistent: expected "
            f"{expected_knot_shape}, got {tuple(envelope_knots.shape)}"
        )


def _magnitude_stats(magnitudes: np.ndarray) -> dict[str, float]:
    magnitudes = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if magnitudes.size == 0:
        raise ValueError("Envelope magnitude statistics require non-empty values")
    if not np.isfinite(magnitudes).all() or np.any(magnitudes < 0.0):
        raise ValueError("Envelope magnitudes must be finite and non-negative")
    return {
        "rms": float(np.sqrt(np.mean(np.square(magnitudes)))),
        "p50": float(np.percentile(magnitudes, 50)),
        "p90": float(np.percentile(magnitudes, 90)),
        "max": float(magnitudes.max()),
    }


def _harmonic_envelope_modes(
    frame_envelopes: torch.Tensor,
    frequencies_hz: torch.Tensor,
    knot_count: int,
) -> list[dict[str, float | int]]:
    if frame_envelopes.ndim != 3 or frame_envelopes.shape[-1] != 2:
        raise ValueError("Frame envelopes must have shape (T, K, 2)")
    if frame_envelopes.shape[0] <= 0:
        raise ValueError("Frame envelope summary requires at least one frame")
    if frequencies_hz.shape != (frame_envelopes.shape[1],):
        raise ValueError("Modal frequency shape does not match harmonic envelope")
    if knot_count <= 0:
        raise ValueError("Harmonic envelope mode summary requires positive knot count")
    envelopes = frame_envelopes.detach().cpu().numpy()
    frequencies = frequencies_hz.detach().cpu().numpy()
    if (
        not np.isfinite(envelopes).all()
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
    ):
        raise ValueError("Harmonic envelope contains invalid values")
    modes: list[dict[str, float | int]] = []
    for mode_slot, frequency_hz in enumerate(frequencies):
        magnitudes = np.linalg.norm(envelopes[:, mode_slot], axis=-1)
        stats = _magnitude_stats(magnitudes)
        modes.append(
            {
                "mode_slot": mode_slot,
                "frequency_hz": float(frequency_hz),
                "knot_count": knot_count,
                "magnitude_rms": stats["rms"],
                "magnitude_p50": stats["p50"],
                "magnitude_p90": stats["p90"],
                "magnitude_max": stats["max"],
            }
        )
    return modes


def _shape_refinement_metrics(
    model: SceneModel,
) -> list[dict[str, float | int]] | None:
    if not model.has_modal_refinement:
        return None
    if model.modal_refinement is None or model.modal_anchor_mask is None:
        raise ValueError("Stage 3A model has incomplete modal refinement state")

    delta_phi = model.modal_refinement.params["delta_phi"].detach()
    anchor_mask = model.modal_anchor_mask.detach()
    expected_shape = tuple(model.modal_phi_real.shape) + (2,)
    if tuple(delta_phi.shape) != expected_shape:
        raise ValueError(
            "Stage 3A delta_phi shape is inconsistent: expected "
            f"{expected_shape}, got {tuple(delta_phi.shape)}"
        )
    if anchor_mask.dtype != torch.bool or tuple(anchor_mask.shape) != tuple(
        model.modal_phi_real.shape[:2]
    ):
        raise ValueError("Stage 3A modal anchor mask is inconsistent")
    if not bool(torch.isfinite(delta_phi).all()):
        raise ValueError("Stage 3A delta_phi contains non-finite values")

    effective_real, effective_imag = model.get_effective_modal_phi()
    staged_real = model.modal_phi_real.detach()
    staged_imag = model.modal_phi_imag.detach()
    metrics: list[dict[str, float | int]] = []
    for mode_slot in range(delta_phi.shape[0]):
        mode_anchor_mask = anchor_mask[mode_slot]
        anchor_count = int(mode_anchor_mask.sum().item())
        if anchor_count == 0:
            raise ValueError(
                f"Stage 3A mode slot {mode_slot} has no fixed-anchor Gaussians"
            )

        mode_delta = delta_phi[mode_slot]
        delta_norm = torch.sqrt(mode_delta.square().sum(dim=(-1, -2)))
        staged_norm = torch.sqrt(
            staged_real[mode_slot].square().sum(dim=-1)
            + staged_imag[mode_slot].square().sum(dim=-1)
        )
        refined_norm = torch.sqrt(
            effective_real[mode_slot].square().sum(dim=-1)
            + effective_imag[mode_slot].square().sum(dim=-1)
        )
        anchor_delta_norm = delta_norm[mode_anchor_mask]
        anchor_staged_norm = staged_norm[mode_anchor_mask]
        anchor_refined_norm = refined_norm[mode_anchor_mask]
        delta_rms = torch.sqrt(anchor_delta_norm.square().mean())
        staged_anchor_rms = torch.sqrt(anchor_staged_norm.square().mean())
        refined_anchor_rms = torch.sqrt(anchor_refined_norm.square().mean())
        if not bool(torch.isfinite(staged_anchor_rms)) or float(
            staged_anchor_rms.item()
        ) <= 0.0:
            raise ValueError(
                f"Stage 3A mode slot {mode_slot} has invalid staged anchor RMS"
            )

        nonanchor_delta_norm = delta_norm[~mode_anchor_mask]
        nonanchor_delta_count = int((nonanchor_delta_norm != 0).sum().item())
        nonanchor_delta_max = (
            0.0
            if nonanchor_delta_norm.numel() == 0
            else float(nonanchor_delta_norm.max().cpu().item())
        )
        metrics.append(
            {
                "mode_slot": mode_slot,
                "frequency_hz": float(model.modal_freqs_hz[mode_slot].cpu().item()),
                "anchor_count": anchor_count,
                "delta_rms": float(delta_rms.cpu().item()),
                "delta_max": float(anchor_delta_norm.max().cpu().item()),
                "staged_anchor_rms": float(staged_anchor_rms.cpu().item()),
                "refined_anchor_rms": float(refined_anchor_rms.cpu().item()),
                "relative_delta_rms": float(
                    (delta_rms / staged_anchor_rms).cpu().item()
                ),
                "nonanchor_delta_max": nonanchor_delta_max,
                "nonanchor_delta_count": nonanchor_delta_count,
            }
        )
    return metrics


def _training_target(
    observed: torch.Tensor,
    valid_mask: torch.Tensor,
    foreground_mask: torch.Tensor,
    has_background_gaussians: bool,
) -> torch.Tensor:
    support = valid_mask if has_background_gaussians else foreground_mask
    return observed * support[..., None].to(observed.dtype) + (
        1.0 - support[..., None].to(observed.dtype)
    )


def _comparison_frame(
    observed: torch.Tensor,
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    if observed.shape != rendered.shape or observed.shape != target.shape:
        raise ValueError("Comparison images must have matching shapes")
    panels = torch.cat(
        [
            observed.clamp(0.0, 1.0),
            rendered.clamp(0.0, 1.0),
            (rendered - target).abs().clamp(0.0, 1.0),
        ],
        dim=1,
    )
    frame = (
        panels.mul(255.0)
        .round()
        .to(torch.uint8)
        .detach()
        .cpu()
        .numpy()
    )
    height, width = frame.shape[:2]
    padded_height = ((height + 15) // 16) * 16
    padded_width = ((width + 15) // 16) * 16
    if padded_height == height and padded_width == width:
        return frame
    return np.pad(
        frame,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="constant",
    )


def _combined_summary(
    accumulators: Sequence[_ViewMetricAccumulator],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frame_count = sum(accumulator.frame_count for accumulator in accumulators)
    valid_pixel_count = sum(
        accumulator.valid_pixel_count for accumulator in accumulators
    )
    foreground_pixel_count = sum(
        accumulator.foreground_pixel_count for accumulator in accumulators
    )
    if frame_count == 0 or valid_pixel_count == 0:
        raise ValueError("Cannot combine empty reconstruction metrics")
    absolute_error_sum = sum(
        accumulator.absolute_error_sum for accumulator in accumulators
    )
    squared_error_sum = sum(
        accumulator.squared_error_sum for accumulator in accumulators
    )
    mean_squared_error = squared_error_sum / (3 * valid_pixel_count)
    weighted_ssim = sum(
        float(summary["ssim"]) * int(summary["frame_count"])
        for summary in summaries
    ) / frame_count
    return {
        "frame_count": frame_count,
        "valid_pixel_count": valid_pixel_count,
        "foreground_pixel_count": foreground_pixel_count,
        "rgb_l1": absolute_error_sum / (3 * valid_pixel_count),
        "psnr_db": (
            None
            if mean_squared_error == 0.0
            else -10.0 * math.log10(mean_squared_error)
        ),
        "ssim": weighted_ssim,
    }


def _write_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def _write_envelope_knots(
    path: Path,
    model: SceneModel,
    view_ids: Sequence[str],
) -> None:
    if model.modal is None:
        raise ValueError("Cannot write envelope knots without modal parameters")
    knots = model.modal.params["envelope_knots"].detach().cpu().numpy()
    if knots.ndim != 3 or knots.shape[-1] != 2 or not np.isfinite(knots).all():
        raise ValueError("Checkpoint envelope knots are invalid")
    real = knots[..., 0]
    imaginary = knots[..., 1]
    magnitude = np.hypot(real, imaginary)
    phase = np.arctan2(imaginary, real)
    phase = np.where(magnitude <= 1.0e-12, np.nan, phase)
    np.savez_compressed(
        path,
        view_ids=np.asarray(view_ids),
        frequencies_hz=model.modal_freqs_hz.detach().cpu().numpy(),
        knot_offsets=model.modal_envelope_knot_offsets.detach().cpu().numpy(),
        knot_times_sec=(
            model.modal_envelope_knot_times_sec.detach().cpu().numpy()
        ),
        knot_real=real,
        knot_imag=imaginary,
        knot_magnitude=magnitude,
        knot_phase_rad=phase,
    )


def _write_temporal_metrics(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    if not records:
        raise ValueError("Cannot write empty temporal reconstruction metrics")
    ordered = sorted(records, key=lambda record: int(record["ts"]))
    expected_ts = list(range(len(ordered)))
    actual_ts = [int(record["ts"]) for record in ordered]
    if actual_ts != expected_ts:
        raise ValueError("Temporal reconstruction records do not cover global ts order")
    envelope_magnitude = np.stack(
        [np.asarray(record["envelope_magnitude"]) for record in ordered],
        axis=0,
    )
    if not np.isfinite(envelope_magnitude).all():
        raise ValueError("Temporal envelope magnitudes contain non-finite values")
    psnr_db = np.asarray(
        [
            np.inf if record["psnr_db"] is None else float(record["psnr_db"])
            for record in ordered
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        frame_name=np.asarray([str(record["frame_name"]) for record in ordered]),
        view_index=np.asarray(
            [int(record["view_index"]) for record in ordered], dtype=np.int64
        ),
        local_index=np.asarray(
            [int(record["local_index"]) for record in ordered], dtype=np.int64
        ),
        time_sec=np.asarray(
            [float(record["time_sec"]) for record in ordered], dtype=np.float64
        ),
        rgb_l1=np.asarray(
            [float(record["rgb_l1"]) for record in ordered], dtype=np.float64
        ),
        psnr_db=psnr_db,
        ssim=np.asarray(
            [float(record["ssim"]) for record in ordered], dtype=np.float64
        ),
        envelope_magnitude=envelope_magnitude,
    )


def _resolve_checkpoint_path(work_dir: Path, ckpt_path: str | None) -> Path:
    return (
        Path(ckpt_path).expanduser()
        if ckpt_path is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )


def run(cfg: ModalReconstructionConfig) -> None:
    if not math.isfinite(cfg.fps) or cfg.fps <= 0:
        raise ValueError("fps must be finite and positive")
    work_dir = Path(cfg.work_dir).expanduser()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work directory does not exist: {work_dir}")
    output_dir = Path(cfg.out_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    checkpoint_path = _resolve_checkpoint_path(work_dir, cfg.ckpt_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    train_cfg = _load_training_config(work_dir)
    dataset, frame_map_path = _load_training_dataset(train_cfg)
    view_ids, frames_by_view = _load_reconstruction_frames(
        frame_map_path,
        dataset.frame_names,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint_model(
        checkpoint_path,
        device,
        bool(train_cfg.get("use_2dgs", False)),
    )
    _validate_model_alignment(model, dataset, view_ids, frames_by_view)
    modal = model.modal
    if modal is None:
        raise ValueError("Checkpoint has no modal envelope")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        dataset_Ks = dataset.get_Ks()
        dataset_w2cs = dataset.get_w2cs()
        view_metrics: dict[str, dict[str, Any]] = {}
        accumulators = []
        summaries = []
        temporal_records: list[dict[str, Any]] = []
        active_envelope_magnitudes: list[np.ndarray] = []
        active_view_ids = [
            view_id for view_id in view_ids if frames_by_view[view_id]
        ]

        for view_id in active_view_ids:
            frames = frames_by_view[view_id]
            frame_ts = torch.as_tensor(
                [frame.ts for frame in frames],
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                envelope_real, envelope_imag = model.compute_modal_envelopes(frame_ts)
            frame_envelopes = torch.stack(
                [envelope_real, envelope_imag],
                dim=-1,
            )
            frame_envelope_magnitudes = torch.linalg.vector_norm(
                frame_envelopes,
                dim=-1,
            ).detach().cpu().numpy()
            active_envelope_magnitudes.append(frame_envelope_magnitudes)
            accumulator = _ViewMetricAccumulator(device)
            video_path = temporary_dir / f"{view_id}_comparison.mp4"
            writer = imageio.get_writer(str(video_path), fps=float(cfg.fps))
            try:
                for frame_position, frame in enumerate(frames):
                    observed = dataset.load_image(frame.dataset_index).to(device)
                    tri_mask = dataset.load_mask(frame.dataset_index).to(device)
                    if observed.ndim != 3 or observed.shape[-1] != 3:
                        raise ValueError(
                            f"Frame {frame.frame_name} image must have shape (H, W, 3)"
                        )
                    if tri_mask.shape != observed.shape[:2]:
                        raise ValueError(
                            f"Frame {frame.frame_name} mask shape does not match image"
                        )
                    valid_mask = tri_mask != 0
                    foreground_mask = tri_mask == 1
                    height, width = observed.shape[:2]
                    with torch.inference_mode():
                        render_outputs = model.render(
                            frame.ts,
                            dataset_w2cs[frame.dataset_index][None].to(device),
                            dataset_Ks[frame.dataset_index][None].to(device),
                            (width, height),
                            bg_color=1.0,
                        )
                    rendered_value = render_outputs.get("img")
                    if not isinstance(rendered_value, torch.Tensor):
                        raise ValueError(
                            "SceneModel.render did not return an img tensor"
                        )
                    if rendered_value.shape != (1, height, width, 3):
                        raise ValueError(
                            f"Rendered image has unexpected shape {tuple(rendered_value.shape)}"
                        )
                    rendered = rendered_value[0]
                    target = _training_target(
                        observed,
                        valid_mask,
                        foreground_mask,
                        model.has_bg,
                    )
                    frame_metrics = accumulator.update(
                        rendered,
                        target,
                        valid_mask,
                        foreground_mask,
                    )
                    temporal_records.append(
                        {
                            "ts": frame.ts,
                            "frame_name": frame.frame_name,
                            "view_index": frame.view_index,
                            "local_index": frame.local_index,
                            "time_sec": frame.time_sec,
                            "rgb_l1": frame_metrics["rgb_l1"],
                            "psnr_db": frame_metrics["psnr_db"],
                            "ssim": frame_metrics["ssim"],
                            "envelope_magnitude": (
                                frame_envelope_magnitudes[frame_position]
                            ),
                        }
                    )
                    writer.append_data(
                        _comparison_frame(observed, rendered, target)
                    )
            finally:
                writer.close()

            summary = accumulator.summary()
            view_index = frames[0].view_index
            summary["activation_magnitude"] = _magnitude_stats(
                frame_envelope_magnitudes,
            )
            knot_start = int(model.modal_envelope_knot_offsets[view_index].item())
            knot_end = int(
                model.modal_envelope_knot_offsets[view_index + 1].item()
            )
            summary["harmonic_envelope_modes"] = _harmonic_envelope_modes(
                frame_envelopes,
                model.modal_freqs_hz,
                knot_end - knot_start,
            )
            view_metrics[view_id] = summary
            accumulators.append(accumulator)
            summaries.append(summary)
            guru.info(
                f"Rendered {len(frames)} frames for {view_id} -> {video_path.name}"
            )

        overall = _combined_summary(accumulators, summaries)
        overall["activation_magnitude"] = _magnitude_stats(
            np.concatenate(
                [values.reshape(-1) for values in active_envelope_magnitudes]
            ),
        )
        metrics = {
            "version": 2,
            "work_dir": str(work_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "frame_map": str(frame_map_path.resolve()),
            "fps": float(cfg.fps),
            "view_order": view_ids,
            "declared_views": view_ids,
            "active_views": active_view_ids,
            "num_modes": int(model.modal_phi_real.shape[0]),
            "modal_parameterization": "per_view_harmonic_envelope_v1",
            "modal_envelope_knot_interval_sec": float(
                model.modal_envelope_knot_interval_sec.detach().cpu().item()
            ),
            "envelope_knots_path": "envelope_knots.npz",
            "temporal_metrics_path": "temporal_metrics.npz",
            "views": view_metrics,
            "overall": overall,
        }
        shape_refinement = _shape_refinement_metrics(model)
        if shape_refinement is not None:
            metrics["shape_refinement"] = shape_refinement
        _write_envelope_knots(
            temporary_dir / "envelope_knots.npz",
            model,
            view_ids,
        )
        _write_temporal_metrics(
            temporary_dir / "temporal_metrics.npz",
            temporal_records,
        )
        _write_metrics(temporary_dir / "metrics.json", metrics)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    guru.info(f"Saved modal reconstruction -> {output_dir}")


def main() -> None:
    run(tyro.cli(ModalReconstructionConfig))


if __name__ == "__main__":
    main()
