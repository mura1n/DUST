from datetime import datetime
from typing import Optional, Tuple

import torch
from pytorch3d.transforms import matrix_to_quaternion

from models.gaussians.basics import interpolate_quats, quat_to_rotmat

STANDARD_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def parse_timestamp_to_seconds(timestamp: str) -> float:
    value = timestamp.strip()
    if not value:
        raise ValueError("timestamp line is empty")

    for fmt in STANDARD_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue

    numeric_value = float(value)
    if numeric_value > 1e12:
        return numeric_value / 1e6
    return numeric_value


def load_timestamp_file(file_path: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    if end_timestep > len(lines):
        raise ValueError(
            f"Timestamp file {file_path} has {len(lines)} entries, but end_timestep={end_timestep}"
        )

    parsed = [parse_timestamp_to_seconds(line) for line in lines[start_timestep:end_timestep]]
    return torch.tensor(parsed, dtype=torch.float64)


def normalize_timestamp_sequence(
    timestamps: torch.Tensor,
    min_value: Optional[torch.Tensor] = None,
    max_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    timestamps = timestamps.to(dtype=torch.float64)
    if timestamps.numel() <= 1:
        return torch.zeros_like(timestamps)

    if min_value is None:
        min_value = torch.min(timestamps)
    if max_value is None:
        max_value = torch.max(timestamps)
    if float(max_value - min_value) <= 0.0:
        return torch.zeros_like(timestamps)
    return (timestamps - min_value) / (max_value - min_value)


def _poses_to_quaternions(reference_poses: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    num_frames, num_instances = valid_mask.shape
    quats = torch.zeros(num_frames, num_instances, 4, dtype=reference_poses.dtype, device=reference_poses.device)
    quats[..., 0] = 1.0
    flat_valid_mask = valid_mask.reshape(-1)
    if flat_valid_mask.any():
        rotations = reference_poses[..., :3, :3].reshape(-1, 3, 3)
        quats.reshape(-1, 4)[flat_valid_mask] = matrix_to_quaternion(rotations[flat_valid_mask])
    return quats


def _get_timestamp_brackets(
    frame_timestamps: torch.Tensor,
    fallback_frame: int,
    target_timestamp: float,
) -> Tuple[int, int, float]:
    timestamps = frame_timestamps.detach().cpu()
    target = float(target_timestamp)

    if target <= float(timestamps[0]):
        return 0, 0, 0.0
    if target >= float(timestamps[-1]):
        last_idx = len(timestamps) - 1
        return last_idx, last_idx, 0.0

    upper_idx = int(torch.searchsorted(timestamps, torch.tensor(target, dtype=timestamps.dtype)).item())
    if upper_idx < len(timestamps) and torch.isclose(
        timestamps[upper_idx],
        torch.tensor(target, dtype=timestamps.dtype),
    ):
        return upper_idx, upper_idx, 0.0
    lower_idx = max(0, upper_idx - 1)

    lower_time = float(timestamps[lower_idx])
    upper_time = float(timestamps[upper_idx])
    if upper_time <= lower_time:
        return fallback_frame, fallback_frame, 0.0

    ratio = (target - lower_time) / (upper_time - lower_time)
    return lower_idx, upper_idx, ratio


def interpolate_rigid_transforms(
    instances_quats: torch.Tensor,
    instances_trans: torch.Tensor,
    instances_fv: torch.Tensor,
    frame_timestamps: torch.Tensor,
    fallback_frame: int,
    target_timestamp: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_timestamp is None or frame_timestamps is None:
        return instances_quats[fallback_frame], instances_trans[fallback_frame]

    lower_idx, upper_idx, ratio = _get_timestamp_brackets(
        frame_timestamps=frame_timestamps,
        fallback_frame=fallback_frame,
        target_timestamp=target_timestamp,
    )
    if lower_idx == upper_idx:
        return instances_quats[lower_idx], instances_trans[lower_idx]

    lower_quats = instances_quats[lower_idx]
    upper_quats = instances_quats[upper_idx]
    lower_trans = instances_trans[lower_idx]
    upper_trans = instances_trans[upper_idx]
    current_quats = instances_quats[fallback_frame]
    current_trans = instances_trans[fallback_frame]

    valid_mask = instances_fv[lower_idx] & instances_fv[upper_idx]
    ratio_tensor = torch.tensor(ratio, dtype=instances_trans.dtype, device=instances_trans.device)
    interpolated_trans = lower_trans + (upper_trans - lower_trans) * ratio_tensor
    safe_lower_quats = torch.where(valid_mask[:, None], lower_quats, current_quats)
    safe_upper_quats = torch.where(valid_mask[:, None], upper_quats, current_quats)
    interpolated_quats = interpolate_quats(
        safe_lower_quats,
        safe_upper_quats,
        fraction=ratio,
    )
    quats = torch.where(valid_mask[:, None], interpolated_quats, current_quats)
    trans = torch.where(valid_mask[:, None], interpolated_trans, current_trans)
    return quats, trans


def build_group_translations_from_reference(
    reference_quats: torch.Tensor,
    reference_trans: torch.Tensor,
    instances_fv: torch.Tensor,
    reference_timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
) -> torch.Tensor:
    target_frames = []
    for frame_idx, target_timestamp in enumerate(target_timestamps):
        _, target_trans = interpolate_rigid_transforms(
            instances_quats=reference_quats,
            instances_trans=reference_trans,
            instances_fv=instances_fv,
            frame_timestamps=reference_timestamps,
            fallback_frame=frame_idx,
            target_timestamp=float(target_timestamp),
        )
        target_frames.append(target_trans)
    return torch.stack(target_frames, dim=0)


def build_group_residuals_from_reference(
    reference_quats: torch.Tensor,
    reference_trans: torch.Tensor,
    instances_fv: torch.Tensor,
    reference_timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
) -> torch.Tensor:
    target_trans = build_group_translations_from_reference(
        reference_quats=reference_quats,
        reference_trans=reference_trans,
        instances_fv=instances_fv,
        reference_timestamps=reference_timestamps,
        target_timestamps=target_timestamps,
    )
    return target_trans - reference_trans


def build_group_poses_from_reference(
    reference_poses: torch.Tensor,
    reference_frame_mask: torch.Tensor,
    reference_timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_quats = _poses_to_quaternions(reference_poses, reference_frame_mask)
    reference_trans = reference_poses[..., :3, 3]
    target_poses = []
    target_masks = []

    max_frame_idx = reference_poses.shape[0] - 1
    for frame_idx, target_timestamp in enumerate(target_timestamps):
        safe_fallback_frame = min(frame_idx, max_frame_idx)
        lower_idx, upper_idx, _ = _get_timestamp_brackets(
            frame_timestamps=reference_timestamps,
            fallback_frame=safe_fallback_frame,
            target_timestamp=float(target_timestamp),
        )
        valid_mask = reference_frame_mask[lower_idx] & reference_frame_mask[upper_idx]
        quats, trans = interpolate_rigid_transforms(
            instances_quats=reference_quats,
            instances_trans=reference_trans,
            instances_fv=reference_frame_mask,
            frame_timestamps=reference_timestamps,
            fallback_frame=safe_fallback_frame,
            target_timestamp=float(target_timestamp),
        )
        pose = torch.eye(4, dtype=reference_poses.dtype, device=reference_poses.device).view(1, 4, 4).repeat(trans.shape[0], 1, 1)
        pose[:, :3, :3] = quat_to_rotmat(quats)
        pose[:, :3, 3] = trans
        target_poses.append(pose)
        target_masks.append(valid_mask)

    return torch.stack(target_poses, dim=0), torch.stack(target_masks, dim=0)


def build_group_frame_mask_from_reference(
    reference_frame_mask: torch.Tensor,
    reference_timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
) -> torch.Tensor:
    target_masks = []
    max_frame_idx = reference_frame_mask.shape[0] - 1
    for frame_idx, target_timestamp in enumerate(target_timestamps):
        safe_fallback_frame = min(frame_idx, max_frame_idx)
        lower_idx, upper_idx, _ = _get_timestamp_brackets(
            frame_timestamps=reference_timestamps,
            fallback_frame=safe_fallback_frame,
            target_timestamp=float(target_timestamp),
        )
        if lower_idx == upper_idx:
            target_masks.append(reference_frame_mask[lower_idx])
        else:
            target_masks.append(reference_frame_mask[lower_idx] & reference_frame_mask[upper_idx])
    return torch.stack(target_masks, dim=0)


def select_effective_group_trans(
    instances_trans: torch.Tensor,
    residual_trans_by_group,
    pose_group: str,
) -> torch.Tensor:
    if residual_trans_by_group is None or pose_group not in residual_trans_by_group:
        return instances_trans
    return instances_trans + residual_trans_by_group[pose_group]


def compute_time_offset_velocity_loss(
    instances_trans: torch.Tensor,
    instances_fv: torch.Tensor,
    frame_timestamps: torch.Tensor,
    current_frame: int,
    current_camera_timestamp: float,
    smooth_range: int,
) -> Optional[torch.Tensor]:
    if smooth_range <= 0:
        return None

    losses = []
    current_trans = instances_trans[current_frame]
    current_timestamp = float(current_camera_timestamp)

    for frame_interval in range(1, smooth_range + 1):
        prev_frame = current_frame - frame_interval
        next_frame = current_frame + frame_interval
        if prev_frame < 0 or next_frame >= instances_trans.shape[0]:
            continue

        valid_mask = instances_fv[prev_frame] & instances_fv[current_frame] & instances_fv[next_frame]
        if valid_mask.sum() == 0:
            continue

        prev_timestamp = float(frame_timestamps[prev_frame])
        next_timestamp = float(frame_timestamps[next_frame])
        if next_timestamp <= prev_timestamp:
            continue

        interpolation_ratio = (current_timestamp - prev_timestamp) / (next_timestamp - prev_timestamp)
        prev_trans = instances_trans[prev_frame][valid_mask]
        next_trans = instances_trans[next_frame][valid_mask]
        target_trans = prev_trans + (next_trans - prev_trans) * interpolation_ratio
        losses.append(torch.norm(current_trans[valid_mask] - target_trans, dim=-1).mean())

    if not losses:
        return None
    return torch.stack(losses).mean()
