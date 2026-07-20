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
from flow3d.modal_canonical_optimization import (
    MODAL_CANONICAL_GAUSSIAN_CONTROL,
    MODAL_CANONICAL_OBJECTIVE,
    MODAL_CANONICAL_TRAINABLE_FIELDS,
)
from flow3d.modal_flow_coordinates import (
    MODAL_FLOW_COORDINATE_GAUGE,
    MODAL_FLOW_COORDINATE_PARAMETERIZATION,
    SUPPORTED_MODAL_FLOW_COORDINATE_SOLVERS,
)
from flow3d.modal_joint_optimization import (
    MODAL_JOINT_OBJECTIVE,
    MODAL_JOINT_PARAMETERIZATION,
    MODAL_PHI_OBJECTIVE,
    MODAL_PHI_PARAMETERIZATION,
)
from flow3d.scene_model import SceneModel


MODAL_PARAMETERIZATION = MODAL_FLOW_COORDINATE_PARAMETERIZATION
MODAL_COORDINATE_GAUGE = MODAL_FLOW_COORDINATE_GAUGE
SUPPORTED_MODAL_PARAMETERIZATIONS = {
    MODAL_PARAMETERIZATION,
    MODAL_JOINT_PARAMETERIZATION,
    MODAL_PHI_PARAMETERIZATION,
}


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
) -> tuple[SceneModel, dict[str, Any]]:
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
    legacy_modal_keys = (
        "modal.params.activations",
        "modal.params.envelope_knots",
        "modal_refinement.params.delta_phi",
        "modal_refinement_mask",
        "modal_refinement_role",
    )
    present_legacy_keys = [key for key in legacy_modal_keys if key in state_dict]
    if present_legacy_keys:
        raise ValueError(
            "Checkpoint uses the removed harmonic-envelope or shape-refinement "
            f"route ({present_legacy_keys}); rebuild it from the static checkpoint, "
            "modal manifest, and a per-frame flow-coordinate artifact"
        )
    init_metadata = checkpoint.get("init_metadata")
    if not isinstance(init_metadata, dict):
        raise ValueError("Flow-coordinate checkpoint init_metadata must be a mapping")
    parameterization = init_metadata.get("modal_parameterization")
    if parameterization not in SUPPORTED_MODAL_PARAMETERIZATIONS:
        raise ValueError(
            "Checkpoint uses an incompatible modal parameterization "
            f"({parameterization!r}); expected one of "
            f"{sorted(SUPPORTED_MODAL_PARAMETERIZATIONS)!r}"
        )
    if init_metadata.get("modal_coordinate_solver") not in SUPPORTED_MODAL_FLOW_COORDINATE_SOLVERS:
        raise ValueError("Checkpoint has an incompatible modal coordinate solver")
    if init_metadata.get("modal_coordinate_gauge") != MODAL_COORDINATE_GAUGE:
        raise ValueError("Checkpoint has an incompatible modal coordinate gauge")
    expected_phi_trainable = parameterization in {
        MODAL_JOINT_PARAMETERIZATION,
        MODAL_PHI_PARAMETERIZATION,
    }
    expected_coordinate_trainable = parameterization == MODAL_JOINT_PARAMETERIZATION
    expected_objective = {
        MODAL_JOINT_PARAMETERIZATION: MODAL_JOINT_OBJECTIVE,
        MODAL_PHI_PARAMETERIZATION: MODAL_PHI_OBJECTIVE,
    }.get(parameterization)
    if expected_objective is not None and init_metadata.get(
        "modal_training_objective"
    ) != expected_objective:
        raise ValueError("Checkpoint has an incompatible modal training objective")
    if init_metadata.get("modal_phi_trainable") is not expected_phi_trainable:
        raise ValueError("Checkpoint modal phi trainability metadata is inconsistent")
    if init_metadata.get("modal_coordinates_trainable") is not expected_coordinate_trainable:
        raise ValueError("Checkpoint coordinate trainability metadata is inconsistent")
    modal_optimization = init_metadata.get("modal_optimization")
    if modal_optimization == "canonical_only":
        if parameterization != MODAL_FLOW_COORDINATE_PARAMETERIZATION:
            raise ValueError(
                "Canonical-only checkpoint must keep the fixed coordinate "
                "parameterization"
            )
        if init_metadata.get("modal_training_objective") != MODAL_CANONICAL_OBJECTIVE:
            raise ValueError(
                "Canonical-only checkpoint has an incompatible training objective"
            )
        if init_metadata.get("canonical_gaussians_trainable") != "foreground_all":
            raise ValueError(
                "Canonical-only checkpoint has inconsistent Gaussian trainability"
            )
        if init_metadata.get("canonical_trainable_fields") != list(
            MODAL_CANONICAL_TRAINABLE_FIELDS
        ):
            raise ValueError(
                "Canonical-only checkpoint has inconsistent canonical fields"
            )
        if init_metadata.get("canonical_gaussian_control") != (
            MODAL_CANONICAL_GAUSSIAN_CONTROL
        ):
            raise ValueError(
                "Canonical-only checkpoint must disable Gaussian control"
            )
    coordinate_source = init_metadata.get("modal_coordinate_source")
    if not isinstance(coordinate_source, str) or not coordinate_source:
        raise ValueError("Checkpoint has no modal coordinate source artifact")
    ridge = init_metadata.get("modal_coordinate_ridge_relative")
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not math.isfinite(float(ridge))
        or float(ridge) <= 0.0
    ):
        raise ValueError("Checkpoint has an invalid modal coordinate ridge value")
    required_state = {
        "modal_coordinate_real",
        "modal_coordinate_imag",
        "modal_phi_real",
        "modal_phi_imag",
        "modal_freqs_hz",
        "modal_frame_view_indices",
        "modal_frame_local_indices",
        "modal_frame_times_sec",
    }
    missing_state = sorted(required_state - set(state_dict))
    if missing_state:
        raise ValueError(
            "Flow-coordinate checkpoint is missing required modal state: "
            f"{missing_state}"
        )
    try:
        model = SceneModel.init_from_state_dict(state_dict)
    except (AssertionError, KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint cannot restore a SceneModel: {checkpoint_path}"
        ) from exc
    model.use_2dgs = bool(use_2dgs)
    model = model.to(device)
    model.eval()
    if model.trajectory_type != "modal_activation" or not model.has_modal:
        raise ValueError("Checkpoint does not contain a modal_activation model")
    if model.has_modal_joint != (parameterization == MODAL_JOINT_PARAMETERIZATION):
        raise ValueError("Checkpoint modal parameterization does not match model state")
    if model.has_modal_phi_refinement != (
        parameterization == MODAL_PHI_PARAMETERIZATION
    ):
        raise ValueError("Checkpoint phi-only parameterization does not match model state")
    return model, init_metadata


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
    num_modes = int(model.modal_phi_real.shape[0])
    coordinate_real = model.modal_coordinate_real.detach().cpu()
    coordinate_imag = model.modal_coordinate_imag.detach().cpu()
    expected_coordinate_shape = (frame_count, num_modes)
    if tuple(coordinate_real.shape) != expected_coordinate_shape:
        raise ValueError(
            "Checkpoint modal_coordinate_real shape is inconsistent: expected "
            f"{expected_coordinate_shape}, got {tuple(coordinate_real.shape)}"
        )
    if tuple(coordinate_imag.shape) != expected_coordinate_shape:
        raise ValueError(
            "Checkpoint modal_coordinate_imag shape is inconsistent: expected "
            f"{expected_coordinate_shape}, got {tuple(coordinate_imag.shape)}"
        )
    if not bool(torch.isfinite(coordinate_real).all()) or not bool(
        torch.isfinite(coordinate_imag).all()
    ):
        raise ValueError("Checkpoint modal coordinates contain non-finite values")
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
        raise ValueError("Checkpoint has no modal_frame_times_sec")
    actual_times_sec = model.modal_frame_times_sec.detach().cpu()
    if actual_times_sec.shape != expected_times_sec.shape:
        raise ValueError("Modal frame times shape does not match checkpoint")
    if not bool(torch.isfinite(actual_times_sec).all()):
        raise ValueError("Modal frame times contain non-finite values")
    if not torch.equal(
        actual_times_sec.to(dtype=torch.float32), expected_times_sec
    ):
        raise ValueError("Modal frame times do not match checkpoint")


def _magnitude_stats(magnitudes: np.ndarray) -> dict[str, float]:
    magnitudes = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if magnitudes.size == 0:
        raise ValueError("Modal coordinate statistics require non-empty values")
    if not np.isfinite(magnitudes).all() or np.any(magnitudes < 0.0):
        raise ValueError("Modal coordinate magnitudes must be finite and non-negative")
    return {
        "rms": float(np.sqrt(np.mean(np.square(magnitudes)))),
        "mean": float(np.mean(magnitudes)),
        "p50": float(np.percentile(magnitudes, 50)),
        "p90": float(np.percentile(magnitudes, 90)),
        "max": float(magnitudes.max()),
    }


def _coordinate_frequency_metrics(
    coordinates: np.ndarray,
    times_sec: np.ndarray,
    assigned_frequency_hz: float,
) -> tuple[float, float]:
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    times_sec = np.asarray(times_sec, dtype=np.float64)
    if coordinates.ndim != 1 or times_sec.shape != coordinates.shape:
        raise ValueError("Coordinate frequency diagnostics require matching 1-D arrays")
    if coordinates.size < 2:
        return 0.0, 0.0
    time_steps = np.diff(times_sec)
    if not np.isfinite(time_steps).all() or np.any(time_steps <= 0.0):
        raise ValueError("Coordinate frame times must be finite and strictly increasing")
    time_step = float(time_steps[0])
    if not np.allclose(time_steps, time_step, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError("Coordinate frequency diagnostics require uniform frame times")

    centered = coordinates - coordinates.mean()
    spectrum = np.fft.fft(centered)
    frequencies_hz = np.fft.fftfreq(coordinates.size, d=time_step)
    power = np.square(np.abs(spectrum))
    nonzero = frequencies_hz != 0.0
    total_power = float(power[nonzero].sum())
    if not math.isfinite(total_power) or total_power <= 0.0:
        return 0.0, 0.0
    nonzero_indices = np.flatnonzero(nonzero)
    dominant_index = int(nonzero_indices[np.argmax(power[nonzero])])
    assigned_indices = np.unique(
        [
            int(np.argmin(np.abs(frequencies_hz - assigned_frequency_hz))),
            int(np.argmin(np.abs(frequencies_hz + assigned_frequency_hz))),
        ]
    )
    assigned_energy_ratio = float(
        np.clip(power[assigned_indices].sum() / total_power, 0.0, 1.0)
    )
    return float(frequencies_hz[dominant_index]), assigned_energy_ratio


def _coordinate_modes(
    frame_coordinates: torch.Tensor,
    frame_times_sec: torch.Tensor,
    frequencies_hz: torch.Tensor,
) -> list[dict[str, float | int]]:
    if frame_coordinates.ndim != 3 or frame_coordinates.shape[-1] != 2:
        raise ValueError("Frame coordinates must have shape (T, K, 2)")
    if frame_coordinates.shape[0] <= 0:
        raise ValueError("Frame coordinate summary requires at least one frame")
    if frequencies_hz.shape != (frame_coordinates.shape[1],):
        raise ValueError("Modal frequency shape does not match frame coordinates")
    if frame_times_sec.shape != (frame_coordinates.shape[0],):
        raise ValueError("Frame times shape does not match frame coordinates")
    coordinates = frame_coordinates.detach().cpu().numpy()
    times_sec = frame_times_sec.detach().cpu().numpy()
    frequencies = frequencies_hz.detach().cpu().numpy()
    if (
        not np.isfinite(coordinates).all()
        or not np.isfinite(times_sec).all()
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
    ):
        raise ValueError("Modal coordinate summary contains invalid values")
    modes: list[dict[str, float | int]] = []
    for mode_slot, frequency_hz in enumerate(frequencies):
        real = coordinates[:, mode_slot, 0]
        imaginary = coordinates[:, mode_slot, 1]
        complex_coordinates = real + 1j * imaginary
        magnitudes = np.abs(complex_coordinates)
        stats = _magnitude_stats(magnitudes)
        dominant_frequency_hz, assigned_energy_ratio = (
            _coordinate_frequency_metrics(
                complex_coordinates,
                times_sec,
                float(frequency_hz),
            )
        )
        modes.append(
            {
                "mode_slot": mode_slot,
                "frequency_hz": float(frequency_hz),
                "coordinate_real_mean": float(real.mean()),
                "coordinate_imaginary_mean": float(imaginary.mean()),
                "coordinate_magnitude_mean": stats["mean"],
                "coordinate_magnitude_rms": stats["rms"],
                "coordinate_magnitude_p50": stats["p50"],
                "coordinate_magnitude_p90": stats["p90"],
                "coordinate_magnitude_max": stats["max"],
                "dominant_signed_frequency_hz": dominant_frequency_hz,
                "assigned_frequency_energy_ratio": assigned_energy_ratio,
            }
        )
    return modes




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


def _joint_refinement_metrics(model: SceneModel) -> dict[str, Any] | None:
    if not model.has_trainable_modal_phi:
        return None
    with torch.inference_mode():
        coordinate_real, coordinate_imag = model.get_all_modal_coefficients()
        coordinate_delta = torch.sqrt(
            (coordinate_real - model.modal_coordinate_real).square()
            + (coordinate_imag - model.modal_coordinate_imag).square()
        )
        phi_real, phi_imag = model.get_effective_modal_phi()
        phi_delta = torch.sqrt(
            (phi_real - model.modal_phi_real).square()
            + (phi_imag - model.modal_phi_imag).square()
        )
        phi_initial = torch.sqrt(
            model.modal_phi_real.square() + model.modal_phi_imag.square()
        )
        mode_summaries = []
        for mode_slot in range(model.modal_phi_real.shape[0]):
            selected = model.modal_phi_trainable_mask[mode_slot]
            frozen = ~selected
            delta_values = phi_delta[mode_slot, selected]
            initial_values = phi_initial[mode_slot, selected]
            mode_summaries.append(
                {
                    "mode_slot": mode_slot,
                    "frequency_hz": float(model.modal_freqs_hz[mode_slot].item()),
                    "trainable_point_count": int(selected.sum().item()),
                    "delta_rms": float(torch.sqrt(delta_values.square().mean()).item()),
                    "delta_max": float(delta_values.amax().item()),
                    "relative_delta_rms": float(
                        torch.sqrt(delta_values.square().mean()).item()
                        / max(torch.sqrt(initial_values.square().mean()).item(), 1e-12)
                    ),
                    "frozen_delta_max": float(
                        phi_delta[mode_slot, frozen].amax().item()
                        if bool(frozen.any().item())
                        else 0.0
                    ),
                }
            )
    return {
        "coordinate_delta_rms": float(
            torch.sqrt(coordinate_delta.square().mean()).item()
        ),
        "coordinate_delta_max": float(coordinate_delta.amax().item()),
        "phi_modes": mode_summaries,
    }


def _change_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("Canonical change diagnostics require finite values")
    if bool((values < 0).any().item()):
        raise ValueError("Canonical change magnitudes must be non-negative")
    return {
        "rms": float(torch.sqrt(values.square().mean()).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.amax().item()),
    }


def _checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Checkpoint has no model state: {path}")
    state = payload["model"]
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Checkpoint model state must contain tensors: {path}")
    return state


def _canonical_change_metrics(
    checkpoint_path: Path,
    init_checkpoint_path: Path,
    init_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if init_metadata.get("modal_optimization") != "canonical_only":
        return None
    if not init_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Canonical-only diagnostics require the initialization checkpoint: "
            f"{init_checkpoint_path}"
        )
    initial = _checkpoint_model_state(init_checkpoint_path)
    current = (
        initial
        if checkpoint_path.resolve() == init_checkpoint_path.resolve()
        else _checkpoint_model_state(checkpoint_path)
    )
    if set(initial) != set(current):
        raise ValueError("Canonical-only init/last model state keys do not match")
    allowed = {
        f"fg.params.{field}" for field in MODAL_CANONICAL_TRAINABLE_FIELDS
    }
    changed_frozen = [
        key
        for key in initial
        if key not in allowed and not torch.equal(initial[key], current[key])
    ]
    if changed_frozen:
        raise ValueError(
            "Canonical-only checkpoint changed frozen model state: "
            + ", ".join(changed_frozen[:10])
        )
    for key in allowed:
        if key not in initial or initial[key].shape != current[key].shape:
            raise ValueError(
                f"Canonical-only Gaussian identity/shape changed for {key}"
            )

    means_delta = torch.linalg.vector_norm(
        current["fg.params.means"] - initial["fg.params.means"], dim=-1
    )
    initial_rgb = torch.sigmoid(initial["fg.params.colors"])
    current_rgb = torch.sigmoid(current["fg.params.colors"])
    rgb_delta = torch.linalg.vector_norm(current_rgb - initial_rgb, dim=-1)
    opacity_delta = torch.abs(
        torch.sigmoid(current["fg.params.opacities"])
        - torch.sigmoid(initial["fg.params.opacities"])
    )
    scale_ratio = torch.exp(
        current["fg.params.scales"] - initial["fg.params.scales"]
    )
    symmetric_scale_ratio = torch.maximum(scale_ratio, scale_ratio.reciprocal())
    initial_quats = initial["fg.params.quats"] / torch.linalg.vector_norm(
        initial["fg.params.quats"], dim=-1, keepdim=True
    ).clamp_min(1e-12)
    current_quats = current["fg.params.quats"] / torch.linalg.vector_norm(
        current["fg.params.quats"], dim=-1, keepdim=True
    ).clamp_min(1e-12)
    quat_dot = torch.abs((initial_quats * current_quats).sum(dim=-1)).clamp(0, 1)
    quaternion_angle = 2.0 * torch.acos(quat_dot)
    identical_quats = (
        current["fg.params.quats"] == initial["fg.params.quats"]
    ).all(dim=-1)
    quaternion_angle = torch.where(
        identical_quats,
        torch.zeros_like(quaternion_angle),
        quaternion_angle,
    )
    return {
        "reference_checkpoint": str(init_checkpoint_path.resolve()),
        "foreground_gaussian_count": int(initial["fg.params.means"].shape[0]),
        "frozen_state_unchanged": True,
        "mean_displacement": _change_stats(means_delta),
        "rgb_change": _change_stats(rgb_delta),
        "opacity_change": _change_stats(opacity_delta),
        "symmetric_scale_ratio": _change_stats(symmetric_scale_ratio),
        "scale_ratio_min": float(scale_ratio.amin().item()),
        "scale_ratio_max": float(scale_ratio.amax().item()),
        "quaternion_angle_rad": _change_stats(quaternion_angle),
    }


def _write_modal_coordinates(
    path: Path,
    model: SceneModel,
    view_ids: Sequence[str],
    frames: Sequence[ReconstructionFrame],
) -> None:
    ordered = sorted(frames, key=lambda frame: frame.ts)
    if [frame.ts for frame in ordered] != list(range(model.num_frames)):
        raise ValueError("Modal coordinate export frames do not cover global ts order")
    with torch.inference_mode():
        effective_real, effective_imaginary = model.get_all_modal_coefficients()
    real = effective_real.detach().cpu().numpy()
    imaginary = effective_imaginary.detach().cpu().numpy()
    initial_real = model.modal_coordinate_real.detach().cpu().numpy()
    initial_imaginary = model.modal_coordinate_imag.detach().cpu().numpy()
    if (
        real.shape != imaginary.shape
        or real.shape != (model.num_frames, model.modal_phi_real.shape[0])
        or not np.isfinite(real).all()
        or not np.isfinite(imaginary).all()
    ):
        raise ValueError("Checkpoint modal coordinates are invalid")
    magnitude = np.hypot(real, imaginary)
    phase = np.arctan2(imaginary, real)
    phase = np.where(magnitude <= 1.0e-12, np.nan, phase)
    np.savez_compressed(
        path,
        view_ids=np.asarray(view_ids),
        frequencies_hz=model.modal_freqs_hz.detach().cpu().numpy(),
        frame_names=np.asarray([frame.frame_name for frame in ordered]),
        frame_view_indices=np.asarray(
            [frame.view_index for frame in ordered], dtype=np.int64
        ),
        frame_local_indices=np.asarray(
            [frame.local_index for frame in ordered], dtype=np.int64
        ),
        frame_times_sec=np.asarray(
            [frame.time_sec for frame in ordered], dtype=np.float64
        ),
        coordinate_real=real,
        coordinate_imag=imaginary,
        initial_coordinate_real=initial_real,
        initial_coordinate_imag=initial_imaginary,
        delta_coordinate_real=real - initial_real,
        delta_coordinate_imag=imaginary - initial_imaginary,
        coordinate_magnitude=magnitude,
        coordinate_phase_rad=phase,
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
    coordinate_magnitude = np.stack(
        [np.asarray(record["coordinate_magnitude"]) for record in ordered],
        axis=0,
    )
    if not np.isfinite(coordinate_magnitude).all():
        raise ValueError("Temporal coordinate magnitudes contain non-finite values")
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
        coordinate_magnitude=coordinate_magnitude,
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
    model, init_metadata = _load_checkpoint_model(
        checkpoint_path,
        device,
        bool(train_cfg.get("use_2dgs", False)),
    )
    _validate_model_alignment(model, dataset, view_ids, frames_by_view)

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
        active_coordinate_magnitudes: list[np.ndarray] = []
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
                coordinate_real, coordinate_imag = (
                    model.compute_modal_coefficients(frame_ts)
                )
            frame_coordinates = torch.stack(
                [coordinate_real, coordinate_imag],
                dim=-1,
            )
            frame_coordinate_magnitudes = torch.linalg.vector_norm(
                frame_coordinates,
                dim=-1,
            ).detach().cpu().numpy()
            active_coordinate_magnitudes.append(frame_coordinate_magnitudes)
            frame_times_sec = torch.as_tensor(
                [frame.time_sec for frame in frames],
                dtype=torch.float64,
            )
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
                            "coordinate_magnitude": (
                                frame_coordinate_magnitudes[frame_position]
                            ),
                        }
                    )
                    writer.append_data(
                        _comparison_frame(observed, rendered, target)
                    )
            finally:
                writer.close()

            summary = accumulator.summary()
            summary["activation_magnitude"] = _magnitude_stats(
                frame_coordinate_magnitudes,
            )
            summary["modal_coordinate_modes"] = _coordinate_modes(
                frame_coordinates,
                frame_times_sec,
                model.modal_freqs_hz,
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
                [values.reshape(-1) for values in active_coordinate_magnitudes]
            ),
        )
        metrics = {
            "version": 3,
            "work_dir": str(work_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "frame_map": str(frame_map_path.resolve()),
            "fps": float(cfg.fps),
            "view_order": view_ids,
            "declared_views": view_ids,
            "active_views": active_view_ids,
            "num_modes": int(model.modal_phi_real.shape[0]),
            "modal_optimization": init_metadata.get("modal_optimization", "fixed"),
            "modal_training_objective": init_metadata.get(
                "modal_training_objective"
            ),
            "modal_parameterization": init_metadata["modal_parameterization"],
            "modal_coordinate_solver": init_metadata["modal_coordinate_solver"],
            "modal_coordinate_gauge": MODAL_COORDINATE_GAUGE,
            "modal_coordinate_source": str(
                init_metadata["modal_coordinate_source"]
            ),
            "modal_coordinate_ridge_relative": float(
                init_metadata["modal_coordinate_ridge_relative"]
            ),
            "activation_magnitude_semantics": "absolute_complex_coordinate",
            "modal_coordinates_path": "modal_coordinates.npz",
            "temporal_metrics_path": "temporal_metrics.npz",
            "views": view_metrics,
            "overall": overall,
        }
        if "modal_coordinate_physics" in init_metadata:
            metrics["modal_coordinate_physics"] = init_metadata[
                "modal_coordinate_physics"
            ]
            metrics["modal_coordinate_prephysics_source"] = init_metadata.get(
                "modal_coordinate_prephysics_source"
            )
        joint_refinement = _joint_refinement_metrics(model)
        if joint_refinement is not None:
            key = "joint_refinement" if model.has_modal_joint else "phi_refinement"
            metrics[key] = joint_refinement
        canonical_refinement = _canonical_change_metrics(
            checkpoint_path,
            work_dir / "checkpoints" / "init.ckpt",
            init_metadata,
        )
        if canonical_refinement is not None:
            metrics["canonical_refinement"] = canonical_refinement
        all_frames = [
            frame
            for view_id in view_ids
            for frame in frames_by_view[view_id]
        ]
        _write_modal_coordinates(
            temporary_dir / "modal_coordinates.npz",
            model,
            view_ids,
            all_frames,
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
