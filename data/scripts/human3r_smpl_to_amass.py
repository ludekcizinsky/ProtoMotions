"""
Convert Human3R SMPL parameter dumps into AMASS-style motion clips.

The converter expects the output directory produced by ``demo.py`` when run
with ``--save_smpl`` enabled. Specifically, the directory must contain a
``smpl/`` folder with per-frame ``*.npz`` files created by the modified
Human3R demo script. Each file stores the SMPL parameters of all tracked
humans for that frame.

For every unique ``smpl_id`` track, an AMASS-compatible ``.npz`` clip is
generated with the keys consumed by ``convert_amass_to_isaac.py``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import typer


def _axis_angle_to_quat(axis_angles: np.ndarray) -> np.ndarray:
    axis_angles = axis_angles.astype(np.float64)
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    small_mask = angles < 1e-8

    safe_axes = np.divide(
        axis_angles,
        angles,
        out=np.zeros_like(axis_angles),
        where=~small_mask,
    )

    sin_half = np.sin(half_angles)
    cos_half = np.cos(half_angles)

    quat = np.concatenate([cos_half, safe_axes * sin_half], axis=-1)
    if np.any(small_mask):
        quat[small_mask[..., 0]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat.astype(np.float32)


def _quat_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions.astype(np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)

    qw = quaternions[..., :1]
    q_xyz = quaternions[..., 1:]
    sin_half = np.linalg.norm(q_xyz, axis=-1, keepdims=True)

    angles = 2.0 * np.arctan2(sin_half, qw)
    small_mask = sin_half < 1e-8

    safe_axes = np.divide(
        q_xyz,
        sin_half,
        out=np.zeros_like(q_xyz),
        where=~small_mask,
    )

    axis_angles = safe_axes * angles
    if np.any(small_mask):
        axis_angles[small_mask[..., 0]] = 0.0
    return axis_angles.astype(np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.broadcast_to(q1, q2.shape)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def _sanitize_sequence_name(name: str) -> str:
    replacements = [
        (" ", "_"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
    ]
    for src, dst in replacements:
        name = name.replace(src, dst)
    return name


def _ensure_matrix(array: np.ndarray, fallback_dim: Optional[int] = None) -> np.ndarray:
    """
    Ensure arrays have shape (N, D). If the input is 1-D, treat it as a single
    row. If it is empty, use ``fallback_dim`` (default 0) to create an empty
    matrix.
    """
    array = np.asarray(array)
    if array.dtype == object:
        items = [np.asarray(item) for item in array.reshape(-1)]
        if not items:
            dim = fallback_dim or 0
            return np.zeros((0, dim), dtype=np.float32)
        array = np.stack(items, axis=0)
    if array.size == 0:
        dim = fallback_dim or 0
        return np.zeros((0, dim), dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, array.shape[0]).astype(np.float32)
    if array.ndim == 2:
        return array.astype(np.float32)
    raise ValueError(f"Expected array with <=2 dims, received shape {array.shape}.")


def _ensure_pose_array(pose: np.ndarray) -> np.ndarray:
    """
    Normalize pose arrays to shape (num_humans, num_joints, 3).
    """
    pose = np.asarray(pose)
    if pose.dtype == object:
        pose_list = [np.asarray(item) for item in pose.reshape(-1)]
        if not pose_list:
            return np.zeros((0, 0, 3), dtype=np.float32)
        pose = np.stack(pose_list, axis=0)

    if pose.size == 0:
        return pose.reshape(0, 0, 3).astype(np.float32)

    if pose.ndim == 1:
        if pose.size % 3 != 0:
            raise ValueError(f"Pose array of shape {pose.shape} cannot be grouped into axis-angle triples.")
        pose = pose.reshape(1, -1, 3)
    elif pose.ndim == 2:
        if pose.shape[-1] == 3:
            pose = pose.reshape(1, pose.shape[0], 3)
        elif pose.shape[0] % 3 == 0 and pose.shape[1] != 3:
            pose = pose.reshape(1, -1, 3)
        elif pose.shape[1] % 3 == 0:
            pose = pose.reshape(pose.shape[0], -1, 3)
        else:
            raise ValueError(f"Pose array of shape {pose.shape} cannot be grouped into axis-angle triples.")
    elif pose.ndim == 3 and pose.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Pose array with shape {pose.shape} is not supported.")

    return pose.astype(np.float32)


def _unwrap_optional(data: np.ndarray) -> Optional[np.ndarray]:
    """
    Handle optional entries stored via ``np.savez``.
    """
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if data.dtype == object:
            if data.size == 1:
                value = data.item()
                if value is None:
                    return None
                return _unwrap_optional(np.asarray(value))
            # Fall back to stacking object entries.
            return np.stack([_unwrap_optional(np.asarray(x)) for x in data], axis=0)
        return data
    return np.asarray(data)


@dataclass
class _TrackAccumulator:
    poses: List[np.ndarray] = field(default_factory=list)
    trans: List[np.ndarray] = field(default_factory=list)
    betas: List[np.ndarray] = field(default_factory=list)
    expressions: List[Optional[np.ndarray]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)

    def add(
        self,
        frame_idx: int,
        pose: np.ndarray,
        translation: np.ndarray,
        betas: np.ndarray,
        expression: Optional[np.ndarray],
    ) -> None:
        self.frame_indices.append(frame_idx)
        self.poses.append(pose.astype(np.float32))
        self.trans.append(translation.astype(np.float32))
        self.betas.append(betas.astype(np.float32))
        if expression is not None and expression.size > 0:
            self.expressions.append(expression.astype(np.float32))
        else:
            self.expressions.append(None)

    def finalize(self) -> Dict[str, np.ndarray]:
        if not self.poses:
            raise ValueError("Cannot finalize empty track.")

        # Stack frame-wise data.
        order = np.argsort(self.frame_indices)
        pose_seq = np.stack(self.poses, axis=0)[order].astype(np.float32)
        # apply -90 degree rotation about X-axis to align with AMASS conventions
        rot_x_neg90 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0, 0.0], dtype=np.float32)
        root_quat = _axis_angle_to_quat(pose_seq[:, 0, :])
        rotated_root = _quat_multiply(
            np.broadcast_to(rot_x_neg90, root_quat.shape), root_quat
        )
        pose_seq[:, 0, :] = _quat_to_axis_angle(rotated_root)
        trans_seq = np.stack(self.trans, axis=0)[order].astype(np.float32)
        trans_seq[:, 2] -= 0.2  # lower the sequence slightly to align with ground

        # Flatten poses to AMASS axis-angle (F, J*3).
        pose_seq_flat = pose_seq.reshape(pose_seq.shape[0], -1)

        # Average betas/expressions across frames to reduce jitter.
        betas_stack = np.stack(self.betas, axis=0)[order]
        betas_vec = betas_stack.mean(axis=0).astype(np.float32)

        expression_vec: Optional[np.ndarray] = None
        if any(expr is not None for expr in self.expressions):
            ordered_exprs = [self.expressions[idx] for idx in order if self.expressions[idx] is not None]
            if ordered_exprs:
                expression_vec = np.stack(ordered_exprs, axis=0).mean(axis=0).astype(np.float32)

        result = {
            "poses": pose_seq_flat,
            "trans": trans_seq,
            "betas": betas_vec,
        }
        if expression_vec is not None and expression_vec.size > 0:
            result["expression"] = expression_vec
        return result


def _load_frame_packets(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        packet = {key: data[key] for key in data.files}
    return packet


def _gather_unique_track_ids(packets: Iterable[Dict[str, np.ndarray]]) -> List[int]:
    ids: set[int] = set()
    for packet in packets:
        if "smpl_id" not in packet:
            continue
        smpl_ids = np.asarray(packet["smpl_id"]).astype(np.int64).ravel()
        ids.update(int(idx) for idx in smpl_ids if smpl_ids.size > 0)
    return sorted(ids)


def convert_human3r_to_amass(
    human3r_dir: Path,
    output_dir: Path,
    fps: float,
    gender: str,
    track_ids: Optional[List[int]] = None,
    sequence_name: Optional[str] = None,
) -> None:
    human3r_dir = human3r_dir.resolve()
    output_dir = output_dir.resolve()

    smpl_dir = human3r_dir / "smpl"
    if not smpl_dir.exists():
        raise FileNotFoundError(f"Expected SMPL dump directory not found: {smpl_dir}")

    frame_files = sorted(smpl_dir.glob("*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No per-frame SMPL files found in {smpl_dir}")

    packets = [_load_frame_packets(path) for path in frame_files]

    if track_ids is None or len(track_ids) == 0:
        track_ids = _gather_unique_track_ids(packets)
    else:
        track_ids = sorted({int(tid) for tid in track_ids})

    if not track_ids:
        raise ValueError("No tracks detected in SMPL dumps.")

    active_ids = set(track_ids)
    accumulators: Dict[int, _TrackAccumulator] = {tid: _TrackAccumulator() for tid in track_ids}

    for frame_idx, packet in enumerate(packets):
        smpl_ids = np.atleast_1d(
            np.asarray(packet.get("smpl_id", np.array([], dtype=np.int64))).astype(np.int64)
        )
        if smpl_ids.size == 0:
            continue

        poses = _ensure_pose_array(packet["rotvec"])
        trans = _ensure_matrix(np.asarray(packet["transl"]), fallback_dim=3)
        betas = _ensure_matrix(np.asarray(packet["shape"]), fallback_dim=10)

        expr = None
        if "expression" in packet:
            expr_unwrapped = _unwrap_optional(packet["expression"])
            if expr_unwrapped is not None and expr_unwrapped.size > 0:
                fallback_dim = expr_unwrapped.shape[-1] if expr_unwrapped.ndim else None
                expr = _ensure_matrix(np.asarray(expr_unwrapped), fallback_dim=fallback_dim)

        # Align lengths: handles cases with single human degeneracy.
        num_humans = smpl_ids.shape[0]
        if poses.shape[0] != num_humans:
            raise ValueError(
                f"Pose count {poses.shape[0]} does not match smpl_id count {num_humans} in frame {frame_files[frame_idx]}"
            )
        if trans.shape[0] != num_humans:
            raise ValueError(
                f"Translation count {trans.shape[0]} does not match smpl_id count {num_humans}."
            )
        if betas.shape[0] != num_humans:
            raise ValueError(
                f"Betas count {betas.shape[0]} does not match smpl_id count {num_humans}."
            )
        if expr is not None and expr.shape[0] != num_humans:
            raise ValueError(
                f"Expression count {expr.shape[0]} does not match smpl_id count {num_humans}."
            )

        for human_idx, smpl_track in enumerate(smpl_ids.tolist()):
            if smpl_track not in active_ids:
                continue
            accum = accumulators[smpl_track]
            accum.add(
                frame_idx=frame_idx,
                pose=poses[human_idx],
                translation=trans[human_idx],
                betas=betas[human_idx],
                expression=None if expr is None else expr[human_idx],
            )

    sequence_name = _sanitize_sequence_name(sequence_name or human3r_dir.name)
    sequence_dir = output_dir / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    gender = gender.lower()
    if gender not in {"neutral", "male", "female"}:
        raise ValueError("gender must be one of {'neutral', 'male', 'female'}.")

    for track_id, accumulator in accumulators.items():
        if not accumulator.poses:
            print(f"[WARN] Track {track_id} is empty. Skipping.")
            continue

        result = accumulator.finalize()

        poses_flat = result["poses"]
        target_joints = 55
        current_joints = poses_flat.shape[1] // 3
        if current_joints < target_joints:
            pad = np.zeros(
                (poses_flat.shape[0], (target_joints - current_joints) * 3),
                dtype=np.float32,
            )
            poses_flat = np.concatenate([poses_flat, pad], axis=1)
        elif current_joints > target_joints:
            poses_flat = poses_flat[:, : target_joints * 3]
        result["poses"] = poses_flat.astype(np.float32)

        result["gender"] = np.array(gender)
        result["mocap_framerate"] = np.array(fps, dtype=np.float32)

        out_file = sequence_dir / f"{sequence_name}_track{track_id:02d}.npz"
        np.savez(out_file, **result)
        print(f"[OK] Saved AMASS-style motion: {out_file}")


def main(
    human3r_dir: Path = typer.Argument(
        ...,
        help="Directory generated by Human3R demo (expects a 'smpl/' sub-directory).",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Destination directory for AMASS-style clips.",
    ),
    track_ids: Optional[List[int]] = typer.Option(
        None,
        "--track-id",
        "-t",
        help="Optional track ids to export. If omitted, convert all tracks found.",
    ),
    fps: float = typer.Option(30.0, help="Frame rate of the Human3R sequence."),
    gender: str = typer.Option("neutral", help="SMPL gender tag to store in the clip metadata."),
    sequence_name: Optional[str] = typer.Option(
        None,
        help="Optional sequence name for the exported clips. Defaults to the Human3R directory name.",
    ),
) -> None:
    convert_human3r_to_amass(
        human3r_dir=human3r_dir,
        output_dir=output_dir,
        fps=fps,
        gender=gender,
        track_ids=track_ids,
        sequence_name=sequence_name,
    )


if __name__ == "__main__":
    typer.run(main)
