"""Reusable exact forward visual-pose frontend from Script 48.


Inputs and conventions
----------------------
Dataset root contains ``left_rgb/*.png`` (BGR through OpenCV),
``depth/*.npy`` (float32 camera-z depth in metres), ``masks/*.png`` (any
positive pixel is foreground, normalized to uint8 255), and
``camera_intrinsics.json`` containing a 3x3 ``K``.  Ground truth in
``frame_poses.json`` is not read by :func:`run_forward`; evaluation is a
separate post-tracking operation.

Points are column-vector camera-frame coordinates.  Backprojection is
``[x,y,z] = [(u-cx)/fx*z, (v-cy)/fy*z, z]``.  A relative transform satisfies
``P_(k+1) ~= R_(k+1,k) P_k + t_(k+1,k)``.  Absolute body-to-camera attitude is
left propagated as ``R_BC[k+1] = R_(k+1,k) @ R_BC[k]`` and reorthonormalized.

Exact frontend parameters
-------------------------
SIFT: nfeatures=2500, contrastThreshold=0.01, edgeThreshold=10 (other OpenCV
defaults).  CLAHE: clipLimit=2.0, tileGridSize=(8,8).  Foreground mask erosion:
9x9 ones, one iteration.  Descriptor matching: BFMatcher L2, k=2, Lowe ratio
0.75.  2-D RANSAC: estimateAffinePartial2D, threshold=3 px, maxIters=2000,
confidence=0.99; minimum 12 ratio matches and 8 later matches.  LK forward:
21x21, 3 pyramid levels, COUNT|EPS, 30 iterations, epsilon=0.01, initial flow
from SIFT; backward LK uses the same settings; forward/backward error <1.5 px.
Both endpoints require raw-mask distance-transform distance >=8 px.  Nearest
depth sampling rounds pixels; valid z is finite and 0.01<z<100 m.

Quality weights multiply normalized descriptor-distance exp(-1.5*d),
ratio exp(-ratio), boundary (0.25+0.75*clip(distance/20,0,1)), and LK
exp(-0.5*(fb_error/1px)^2), floored at 1e-4.  3-D RANSAC threshold=0.012 m,
900 iterations, deterministic RNG seed 48, 3-point unweighted SVD hypotheses,
early exit at >88% inliers and mean residual <45% of threshold, up to four
inlier/refit rounds, weighted SVD refinement, then residual reweighting with
``exp(-0.5*(residual/max(median,1e-6))^2)``.  SVD estimates ``B=R A+t`` and
forces det(R)>0; every returned rotation is projected onto SO(3).
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_SOURCE = Path(__file__).with_name(
    "48_position_attitude_translating_asteroid_tracker.py"
)
_SPEC = importlib.util.spec_from_file_location("_script48_exact", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load exact Script 48 source: {_SOURCE}")
_S48 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_S48)

# Public exact primitives (same function objects as Script 48).
rotation_angle_deg = _S48.rotation_angle_deg
reorthonormalize = _S48.reorthonormalize
estimate_rigid_svd = _S48.estimate_rigid_svd
estimate_rigid_weighted_svd = _S48.estimate_rigid_weighted_svd
ransac_rigid_3d = _S48.ransac_rigid_3d
load_rgb = _S48.load_rgb
load_depth = _S48.load_depth
load_mask = _S48.load_mask
backproject = _S48.backproject
make_sift = _S48.make_sift
estimate_direction = _S48.estimate_direction

PARAMETERS = {
    name: getattr(_S48, name)
    for name in (
        "MAX_SIFT_FEATURES", "RATIO_TEST", "RANSAC_THRESH_M",
        "RANSAC_ITERS", "ERODE_KERNEL", "MIN_BOUNDARY_DIST_PX",
        "MIN_2D_MATCHES", "MIN_3D_MATCHES", "CLAHE_CLIP", "CLAHE_TILE",
        "MIN_DEPTH_M", "MAX_DEPTH_M", "MIN_QUALITY_WEIGHT", "RNG_SEED",
    )
}

@dataclass(frozen=True)
class ForwardResult:
    relative_rotations: np.ndarray
    relative_translations: np.ndarray
    absolute_rotations: np.ndarray
    pair_records: tuple[dict, ...]

def propagate_absolute(relative_rotations: np.ndarray, initial_rotation: np.ndarray) -> np.ndarray:
    """Script 48 left-multiplication convention, independent of ground truth."""
    absolute = [np.asarray(initial_rotation, dtype=np.float64).copy()]
    for relative in relative_rotations:
        absolute.append(reorthonormalize(relative @ absolute[-1]))
    return np.asarray(absolute)

def run_forward(dataset_root: Path, initial_rotation: Optional[np.ndarray] = None) -> ForwardResult:
    """Run the exact frontend.  This function never opens frame_poses.json."""
    root = Path(dataset_root)
    with open(root / "camera_intrinsics.json", "r", encoding="utf-8") as f:
        K = np.asarray(json.load(f)["K"], dtype=np.float64)
    rgb = sorted((root / "left_rgb").glob("*.png"))
    depth = sorted((root / "depth").glob("*.npy"))
    masks = sorted((root / "masks").glob("*.png"))
    n = min(len(rgb), len(depth), len(masks))
    if n < 2:
        raise RuntimeError(f"Need at least two synchronized frames under {root}")
    detector = make_sift()
    rotations, translations, records = [], [], []
    for k in range(n - 1):
        estimate = estimate_direction(
            load_rgb(rgb[k]), load_rgb(rgb[k + 1]),
            load_depth(depth[k]), load_depth(depth[k + 1]),
            load_mask(masks[k]), load_mask(masks[k + 1]), K, detector,
        )
        rotations.append(estimate["R"] if estimate["success"] else np.eye(3))
        translations.append(estimate["t"] if estimate["success"] else np.zeros(3))
        records.append({
            "pair_index": k, "success": bool(estimate["success"]),
            "n_inliers_3d": int(estimate["n_inliers_3d"]),
            "mean_residual_m": estimate["mean_residual_m"],
            "mean_lk_fb_err_px": estimate["mean_lk_fb_err_px"],
        })
    relative_R = np.asarray(rotations)
    relative_t = np.asarray(translations)
    R0 = np.eye(3) if initial_rotation is None else initial_rotation
    return ForwardResult(relative_R, relative_t, propagate_absolute(relative_R, R0), tuple(records))

def load_ground_truth_attitude(dataset_root: Path) -> np.ndarray:
    """Evaluation-only helper, intentionally separate from run_forward."""
    with open(Path(dataset_root) / "frame_poses.json", "r", encoding="utf-8") as f:
        frames = json.load(f)
    return np.asarray([x["object_rotation_body_to_camera"] for x in frames], dtype=np.float64)

def attitude_errors(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.asarray([rotation_angle_deg(a @ b.T) for a, b in zip(estimated, truth)])
