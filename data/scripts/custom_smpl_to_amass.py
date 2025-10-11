"""
Utilities to convert custom SMPL parameter exports into AMASS-style motion clips.

The converter expects the preprocessing directory produced by the downstream
pipeline described in ``training.helpers.dataset.FullSceneDataset``. The
directory must contain::

    cameras_normalize.npz
    mean_shape.npy
    poses.npy
    normalize_trans.npy

Each SMPL track is converted into an AMASS-compatible ``.npz`` file with the
keys used by ``convert_amass_to_isaac.py``. The resulting clips can be referenced
from a YAML motion descriptor and packaged with ``package_motion_lib.py``.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer


# +90 degree rotation about the X-axis maps Y-up data into Z-up coordinates.
_ROT_X90_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
_ROT_X90_QUAT = np.array(
    [
        np.cos(np.pi / 4.0),
        np.sin(np.pi / 4.0),
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def _axis_angle_to_quat(axis_angles: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors (..., 3) to quaternions (..., 4)."""
    axis_angles = axis_angles.astype(np.float64)
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    small_mask = angles < 1e-8

    # Avoid division by zero for very small angles by falling back to the identity.
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
    """Convert quaternions (..., 4) back to axis-angle (..., 3)."""
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
    """Hamilton product of two quaternions, supporting broadcasting."""
    q1 = np.broadcast_to(q1, q2.shape)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def _apply_global_rotation_to_poses(pose_seq: np.ndarray) -> np.ndarray:
    """
    Apply the +90° X-axis rotation to the root (global) joint orientation only.
    Downstream joints are already expressed in the root frame, so they remain unchanged.
    """
    rotated = pose_seq.copy()
    root_axis_angle = rotated[:, :3]
    root_quat = _axis_angle_to_quat(root_axis_angle)
    rotated_root = _quat_multiply(_ROT_X90_QUAT.reshape(1, 4), root_quat)
    rotated[:, :3] = _quat_to_axis_angle(rotated_root)
    return rotated


def _load_scale(preprocess_dir: Path) -> float:
    """Extract the global scale used by the normalization step."""
    camera_dict = np.load(preprocess_dir / "cameras_normalize.npz")
    scale_key_candidates = [key for key in camera_dict.files if key.startswith("scale_mat_")]
    if not scale_key_candidates:
        raise KeyError(
            "Could not find keys that start with 'scale_mat_' in cameras_normalize.npz. "
            "This file is required to recover the global scale."
        )

    scale_mat = camera_dict[scale_key_candidates[0]]
    return 1.0 / float(scale_mat[0, 0])


def _sanitize_sequence_name(name: str) -> str:
    """Produce filesystem friendly sequence names."""
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


def convert_tracks_to_amass(
    preprocess_dir: Path,
    output_dir: Path,
    track_ids: Optional[List[int]],
    fps: float,
    gender: str,
    sequence_name: Optional[str],
) -> None:
    preprocess_dir = preprocess_dir.resolve()
    output_dir = output_dir.resolve()

    if not preprocess_dir.exists():
        raise FileNotFoundError(f"Preprocess directory not found: {preprocess_dir}")

    poses_path = preprocess_dir / "poses.npy"
    trans_path = preprocess_dir / "normalize_trans.npy"
    betas_path = preprocess_dir / "mean_shape.npy"

    for path in (poses_path, trans_path, betas_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected file is missing: {path}")

    poses = np.load(poses_path)
    translations = np.load(trans_path)
    betas = np.load(betas_path)

    if poses.ndim == 2:
        poses = poses[:, np.newaxis, :]
    if translations.ndim == 2:
        translations = translations[:, np.newaxis, :]
    if betas.ndim == 1:
        betas = betas[np.newaxis, :]

    num_frames, num_tracks, pose_dim = poses.shape
    if pose_dim not in {72, 24 * 3}:
        raise ValueError(
            f"Expected 72 axis-angle parameters per frame, got pose dimension {pose_dim}."
        )

    if translations.shape[:2] != (num_frames, num_tracks) or translations.shape[2] != 3:
        raise ValueError(
            "normalize_trans.npy is expected to have shape [num_frames, num_tracks, 3]."
        )

    if betas.shape[0] != num_tracks:
        raise ValueError(
            f"mean_shape.npy first dimension {betas.shape[0]} must match number of tracks {num_tracks}."
        )

    if track_ids is None or len(track_ids) == 0:
        track_ids = list(range(num_tracks))

    missing_ids = [tid for tid in track_ids if tid < 0 or tid >= num_tracks]
    if missing_ids:
        raise ValueError(
            f"Requested track ids {missing_ids} are out of bounds for {num_tracks} tracks."
        )

    scale = _load_scale(preprocess_dir)
    sequence_name = _sanitize_sequence_name(sequence_name or preprocess_dir.name)
    sequence_dir = output_dir / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    gender = gender.lower()
    if gender not in {"neutral", "male", "female"}:
        raise ValueError("gender must be one of {'neutral', 'male', 'female'}.")

    for track_id in track_ids:
        pose_seq = poses[:, track_id].astype(np.float32)
        pose_seq = _apply_global_rotation_to_poses(pose_seq)
        trans_seq = (translations[:, track_id] * scale).astype(np.float32)
        trans_seq = trans_seq @ _ROT_X90_MATRIX.T
        betas_vec = betas[track_id].astype(np.float32)

        out_file = sequence_dir / f"{sequence_name}_track{track_id:02d}.npz"
        np.savez(
            out_file,
            poses=pose_seq,
            trans=trans_seq,
            betas=betas_vec,
            gender=np.array(gender),
            mocap_framerate=np.array(fps, dtype=np.float32),
        )
        print(f"[OK] Saved AMASS-style motion: {out_file}")


def main(
    preprocess_dir: Path = typer.Argument(..., help="Directory with custom preprocessing outputs."),
    output_dir: Path = typer.Argument(..., help="Destination directory for AMASS-style clips."),
    track_ids: Optional[List[int]] = typer.Option(
        None,
        "--track-id",
        "-t",
        help="Track id(s) to export. If omitted, all tracks are converted.",
    ),
    fps: float = typer.Option(30.0, help="Frame rate of the source sequence."),
    gender: str = typer.Option("neutral", help="SMPL gender tag to store in the clip metadata."),
    sequence_name: Optional[str] = typer.Option(
        None,
        help="Optional name for the exported sequence. Defaults to the preprocessing directory name.",
    ),
) -> None:
    """
    Convert SMPL parameters produced by the custom pipeline into AMASS-style motion files.
    """

    convert_tracks_to_amass(
        preprocess_dir=preprocess_dir,
        output_dir=output_dir,
        track_ids=track_ids,
        fps=fps,
        gender=gender,
        sequence_name=sequence_name,
    )


if __name__ == "__main__":
    typer.run(main)
