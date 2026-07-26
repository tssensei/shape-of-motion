import glob
from collections.abc import Callable

import imageio.v2 as iio
import numpy as np
from loguru import logger as guru
from segment_anything import SamPredictor, sam_model_registry
from tracker.base_tracker import BaseTracker


def init_sam_model(checkpoint_dir: str, sam_model_type: str, device) -> SamPredictor:
    checkpoints = glob.glob(f"{checkpoint_dir}/*{sam_model_type}*.pth")
    if len(checkpoints) == 0:
        raise ValueError(
            f"No checkpoints found for model type {sam_model_type} in {checkpoint_dir}"
        )
    checkpoints = sorted(checkpoints)
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoints[-1])
    sam.to(device=device)
    guru.info(f"loaded model checkpoint {checkpoints[-1]}")
    return SamPredictor(sam)


def init_tracker(checkpoint_dir, device) -> BaseTracker:
    checkpoints = glob.glob(f"{checkpoint_dir}/*XMem*.pth")
    if len(checkpoints) == 0:
        raise ValueError(f"No XMem checkpoints found in {checkpoint_dir}")
    checkpoints = sorted(checkpoints)
    return BaseTracker(checkpoints[-1], device)


def track_masks(
    tracker: BaseTracker,
    img_paths: list[str],
    cano_mask: np.ndarray,
    cano_t: int,
    save_mask: Callable[[int, np.ndarray], None],
) -> None:
    """
    :param img_paths: ordered paths to the RGB frames
    :param cano_mask: (H, W) index mask
    :param cano_t: canonical frame index
    :param save_mask: callback that saves one tracked index mask
    """
    T = len(img_paths)
    if T == 0:
        raise ValueError("Cannot track masks without input frames.")
    if cano_t < 0 or cano_t >= T:
        raise IndexError(f"Canonical frame index {cano_t} is outside [0, {T}).")
    cano_mask = cano_mask > 0.5

    def load_frame(t: int) -> np.ndarray:
        frame = iio.imread(img_paths[t])
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected an RGB image, got {frame.shape}: {img_paths[t]}")
        return frame[:, :, :3]

    # forward from canonical_id
    forward_total = T - cano_t
    guru.info(f"Tracking {forward_total} frames forward from frame {cano_t}.")
    try:
        for step, t in enumerate(range(int(cano_t), T), start=1):
            frame = load_frame(t)
            if t == cano_t:
                mask = tracker.track(frame, cano_mask)
            else:
                mask = tracker.track(frame)
            save_mask(t, mask)
            if step % 25 == 0 or step == forward_total:
                guru.info(f"Forward tracking: {step} / {forward_total} frames saved.")
    finally:
        tracker.clear_memory()

    # backward from canonical_id
    if cano_t > 0:
        guru.info(f"Tracking {cano_t} frames backward from frame {cano_t}.")
        try:
            tracker.track(load_frame(cano_t), cano_mask)
            for step, t in enumerate(range(int(cano_t) - 1, -1, -1), start=1):
                mask = tracker.track(load_frame(t))
                save_mask(t, mask)
                if step % 25 == 0 or step == cano_t:
                    guru.info(f"Backward tracking: {step} / {cano_t} frames saved.")
        finally:
            tracker.clear_memory()
