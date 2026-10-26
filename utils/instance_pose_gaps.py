import math
from typing import Iterator, Tuple

import torch


def _iter_internal_gaps(frame_mask: torch.Tensor, max_pose_gap: int) -> Iterator[Tuple[int, int, int]]:
    """Yield bounded internal gaps as `(start_frame, end_frame, gap_length)` triples."""
    valid_indices = torch.nonzero(frame_mask, as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        return

    for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
        gap_length = int(end_idx.item() - start_idx.item() - 1)
        if 0 < gap_length <= max_pose_gap:
            yield int(start_idx.item()), int(end_idx.item()), gap_length


def _nan_quaternion_like(quaternion: torch.Tensor) -> torch.Tensor:
    """Return a NaN quaternion with the same dtype/device as the input."""
    return torch.full_like(quaternion, float("nan"))


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion or return NaNs when the input is unusable."""
    norm = torch.linalg.norm(quaternion)
    if not torch.isfinite(norm) or norm <= 0:
        return _nan_quaternion_like(quaternion)
    normalized = quaternion / norm
    if not torch.isfinite(normalized).all():
        return _nan_quaternion_like(quaternion)
    return normalized


def matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a quaternion without reducing input precision."""
    matrix = rotation_matrix
    if not torch.isfinite(matrix).all():
        return torch.full((4,), float("nan"), dtype=matrix.dtype, device=matrix.device)

    m00 = matrix[0, 0]
    m01 = matrix[0, 1]
    m02 = matrix[0, 2]
    m10 = matrix[1, 0]
    m11 = matrix[1, 1]
    m12 = matrix[1, 2]
    m20 = matrix[2, 0]
    m21 = matrix[2, 1]
    m22 = matrix[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        scale = torch.sqrt(trace + matrix.new_tensor(1.0)) * matrix.new_tensor(2.0)
        quaternion = torch.stack(
            [
                0.25 * scale,
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
            ]
        )
    elif m00 > m11 and m00 > m22:
        scale = torch.sqrt(matrix.new_tensor(1.0) + m00 - m11 - m22) * matrix.new_tensor(2.0)
        quaternion = torch.stack(
            [
                (m21 - m12) / scale,
                0.25 * scale,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
            ]
        )
    elif m11 > m22:
        scale = torch.sqrt(matrix.new_tensor(1.0) + m11 - m00 - m22) * matrix.new_tensor(2.0)
        quaternion = torch.stack(
            [
                (m02 - m20) / scale,
                (m01 + m10) / scale,
                0.25 * scale,
                (m12 + m21) / scale,
            ]
        )
    else:
        scale = torch.sqrt(matrix.new_tensor(1.0) + m22 - m00 - m11) * matrix.new_tensor(2.0)
        quaternion = torch.stack(
            [
                (m10 - m01) / scale,
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                0.25 * scale,
            ]
        )

    return _normalize_quaternion(quaternion.to(dtype=rotation_matrix.dtype, device=rotation_matrix.device))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to a rotation matrix, returning NaNs for invalid input."""
    normalized = _normalize_quaternion(quaternion)
    if not torch.isfinite(normalized).all():
        return torch.full((3, 3), float("nan"), dtype=quaternion.dtype, device=quaternion.device)

    w, x, y, z = normalized
    two = normalized.new_tensor(2.0)
    return torch.stack(
        [
            torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)]),
            torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)]),
            torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)]),
        ]
    )


def slerp_short_path(q0: torch.Tensor, q1: torch.Tensor, ratio: float) -> torch.Tensor:
    """Slerp between quaternions along the shorter arc, or return NaNs on invalid input."""
    q0_normalized = _normalize_quaternion(q0)
    q1_normalized = _normalize_quaternion(q1)
    if not torch.isfinite(q0_normalized).all() or not torch.isfinite(q1_normalized).all():
        return _nan_quaternion_like(q0)

    dot = torch.dot(q0_normalized, q1_normalized)
    if dot < 0:
        q1_normalized = -q1_normalized
        dot = -dot

    dot = torch.clamp(dot, -1.0, 1.0)
    if dot > 1.0 - 1e-6:
        return _normalize_quaternion(q0_normalized + (q1_normalized - q0_normalized) * ratio)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    if torch.abs(sin_theta_0) < 1e-6:
        return _normalize_quaternion(q0_normalized + (q1_normalized - q0_normalized) * ratio)

    theta = theta_0 * ratio
    scale_0 = torch.sin(theta_0 - theta) / sin_theta_0
    scale_1 = torch.sin(theta) / sin_theta_0
    return _normalize_quaternion(scale_0 * q0_normalized + scale_1 * q1_normalized)


def _fallback_rotation(start_rotation: torch.Tensor, end_rotation: torch.Tensor, ratio: float) -> torch.Tensor:
    """Choose the nearer endpoint rotation; ties resolve to the earlier endpoint."""
    if ratio <= 0.5:
        return start_rotation.clone()
    return end_rotation.clone()


def interpolate_rotation_matrices(start_rotation: torch.Tensor, end_rotation: torch.Tensor, ratio: float) -> torch.Tensor:
    """Interpolate rotations by quaternion short-path slerp with deterministic endpoint fallback.

    Falling back to the nearer endpoint is intentional for this feature: a short-gap frame
    stays renderable even if one endpoint rotation is malformed or interpolation becomes non-finite.
    """
    start_quaternion = matrix_to_quaternion(start_rotation)
    end_quaternion = matrix_to_quaternion(end_rotation)
    if not torch.isfinite(start_quaternion).all() or not torch.isfinite(end_quaternion).all():
        return _fallback_rotation(start_rotation, end_rotation, ratio)

    interpolated_quaternion = slerp_short_path(start_quaternion, end_quaternion, ratio)
    normalized_interpolated = _normalize_quaternion(interpolated_quaternion)
    if not torch.isfinite(normalized_interpolated).all():
        return _fallback_rotation(start_rotation, end_rotation, ratio)

    interpolated_rotation = quaternion_to_matrix(normalized_interpolated).to(
        dtype=start_rotation.dtype,
        device=start_rotation.device,
    )
    if not torch.isfinite(interpolated_rotation).all():
        return _fallback_rotation(start_rotation, end_rotation, ratio)
    return interpolated_rotation


def fill_missing_instance_poses(
    instances_pose: torch.Tensor,
    frame_mask: torch.Tensor,
    enabled: bool,
    max_pose_gap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill only short internal pose gaps and return repaired/original/interpolated masks aligned."""
    repaired_pose = instances_pose.clone()
    repaired_mask = frame_mask.clone()
    original_mask = frame_mask.clone()
    interpolated_mask = torch.zeros_like(frame_mask, dtype=torch.bool)

    if not enabled or max_pose_gap <= 0:
        return repaired_pose, repaired_mask, original_mask, interpolated_mask

    for instance_idx in range(frame_mask.shape[1]):
        instance_mask = frame_mask[:, instance_idx]
        for start_frame, end_frame, gap_length in _iter_internal_gaps(instance_mask, max_pose_gap):
            start_translation = repaired_pose[start_frame, instance_idx, :3, 3]
            end_translation = repaired_pose[end_frame, instance_idx, :3, 3]
            start_rotation = repaired_pose[start_frame, instance_idx, :3, :3]
            end_rotation = repaired_pose[end_frame, instance_idx, :3, :3]

            for gap_offset in range(1, gap_length + 1):
                frame_idx = start_frame + gap_offset
                ratio = gap_offset / (gap_length + 1)
                repaired_pose[frame_idx, instance_idx] = repaired_pose[start_frame, instance_idx]
                repaired_pose[frame_idx, instance_idx, :3, :3] = interpolate_rotation_matrices(
                    start_rotation=start_rotation,
                    end_rotation=end_rotation,
                    ratio=ratio,
                )
                repaired_pose[frame_idx, instance_idx, :3, 3] = start_translation + (end_translation - start_translation) * ratio
                repaired_mask[frame_idx, instance_idx] = True
                interpolated_mask[frame_idx, instance_idx] = True

    return repaired_pose, repaired_mask, original_mask, interpolated_mask
