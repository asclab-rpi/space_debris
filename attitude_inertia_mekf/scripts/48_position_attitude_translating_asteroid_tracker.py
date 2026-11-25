"""
SCRIPT 48: Position + attitude tracker/evaluator for translating/tumbling asteroid
================================================================================

Dataset expected
----------------
    output/textured_translating_tumbling_asteroid_200_frame_sequence_v2_fixed/

This script is the moving-target version of Script 46. It keeps the traditional
SIFT + LK + depth + RANSAC + weighted SVD attitude pipeline and adds direct
frame-by-frame position evaluation for the translating asteroid.

Run
---
    cd C:/Blender_Proc
    python -m scripts.48_position_attitude_translating_asteroid_tracker

or:
    python scripts/48_position_attitude_translating_asteroid_tracker.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# Paths / configuration
# =============================================================================

DATA_ROOT = Path(r"output\textured_translating_tumbling_asteroid_200_frame_sequence_v2_fixed")
RGB_DIR = DATA_ROOT / "left_rgb"
DEPTH_DIR = DATA_ROOT / "depth"
MASK_DIR = DATA_ROOT / "masks"
POSES_JSON = DATA_ROOT / "frame_poses.json"
INTRINSICS_JSON = DATA_ROOT / "camera_intrinsics.json"
OUTPUT_DIR = DATA_ROOT / "position_attitude_script48"

# Tracking front-end parameters, kept close to Scripts 44-46.
MAX_SIFT_FEATURES = 2500
RATIO_TEST = 0.75
RANSAC_THRESH_M = 0.012
RANSAC_ITERS = 900
ERODE_KERNEL = 9
MIN_BOUNDARY_DIST_PX = 8.0
MIN_2D_MATCHES = 12
MIN_3D_MATCHES = 8
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 100.0
SAVE_DEBUG_EVERY = 50

QUALITY_WEIGHTED_AVERAGE = True
MIN_QUALITY_WEIGHT = 1e-6

# Position estimator settings.
ASTEROID_RADIUS_M = 0.75
MASK_CENTROID_MAX_PIXELS = 120000
SPHERE_FIT_MAX_PIXELS = 12000
SPHERE_FIT_MIN_PIXELS = 500
SPHERE_FIT_ROBUST_LOSS = "soft_l1"
SPHERE_FIT_MAX_NFEV = 35
RANGE_FRONT_PERCENTILE = 4.0
RNG_SEED = 48


# =============================================================================
# Utility / math helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotation_angle_deg(R: np.ndarray) -> float:
    val = (np.trace(R) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def reorthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def rot_log(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(reorthonormalize(R).astype(np.float64))
    return rvec.reshape(3)


def rot_exp(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return reorthonormalize(R)


def so3_interpolate(R_a: np.ndarray, R_b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    R_delta = R_b @ R_a.T
    return reorthonormalize(rot_exp(alpha * rot_log(R_delta)) @ R_a)


def estimate_rigid_svd(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate B ~= R A + t."""
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cb - R @ ca
    return reorthonormalize(R), t


def estimate_rigid_weighted_svd(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted estimate B ~= R A + t."""
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, MIN_QUALITY_WEIGHT)
    w = w / np.sum(w)

    ca = np.sum(A * w[:, None], axis=0)
    cb = np.sum(B * w[:, None], axis=0)
    AA = A - ca
    BB = B - cb
    H = (AA * w[:, None]).T @ BB

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cb - R @ ca
    return reorthonormalize(R), t


def ransac_rigid_3d(
    A: np.ndarray,
    B: np.ndarray,
    thresh: float,
    iters: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[float]]:
    """RANSAC + weighted SVD refinement for B ~= R A + t."""
    n = len(A)
    if n < 3:
        return None, None, np.zeros(n, dtype=bool), None

    rng = np.random.default_rng(RNG_SEED)
    all_idx = np.arange(n)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_mean = np.inf

    for _ in range(iters):
        sample = rng.choice(all_idx, size=3, replace=False)
        try:
            R, t = estimate_rigid_svd(A[sample], B[sample])
        except Exception:
            continue

        pred = (R @ A.T).T + t
        resid = np.linalg.norm(pred - B, axis=1)
        inliers = resid < thresh
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        mean_res = float(np.mean(resid[inliers]))
        if count > best_count or (count == best_count and mean_res < best_mean):
            best_count = count
            best_mean = mean_res
            best_inliers = inliers
        if count > 0.88 * n and mean_res < 0.45 * thresh:
            break

    if best_count < 3:
        return None, None, best_inliers, None

    inliers = best_inliers.copy()
    R = None
    t = None
    for _ in range(4):
        if np.count_nonzero(inliers) < 3:
            break
        try:
            if weights is not None:
                R, t = estimate_rigid_weighted_svd(A[inliers], B[inliers], weights[inliers])
            else:
                R, t = estimate_rigid_svd(A[inliers], B[inliers])
        except Exception:
            return None, None, inliers, None

        pred = (R @ A.T).T + t
        resid = np.linalg.norm(pred - B, axis=1)
        new_inliers = resid < thresh
        if np.array_equal(new_inliers, inliers):
            break
        if np.count_nonzero(new_inliers) >= 3:
            inliers = new_inliers

    if R is None or t is None or np.count_nonzero(inliers) < 3:
        return None, None, inliers, None

    if weights is not None:
        pred = (R @ A.T).T + t
        resid = np.linalg.norm(pred - B, axis=1)
        med = np.median(resid[inliers]) if np.any(inliers) else np.median(resid)
        residual_weights = np.exp(-0.5 * (resid / max(float(med), 1e-6)) ** 2)
        final_weights = weights * residual_weights
        try:
            R, t = estimate_rigid_weighted_svd(A[inliers], B[inliers], final_weights[inliers])
        except Exception:
            pass

    pred = (R @ A.T).T + t
    resid = np.linalg.norm(pred - B, axis=1)
    inliers = resid < thresh
    mean_res = float(np.mean(resid[inliers])) if np.any(inliers) else None
    return reorthonormalize(R), t, inliers, mean_res


# =============================================================================
# Image / feature helpers
# =============================================================================

def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return img


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    return ((mask > 0).astype(np.uint8) * 255)


def load_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def preprocess_gray(rgb_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(gray)


def erode_mask(mask: np.ndarray) -> np.ndarray:
    if ERODE_KERNEL <= 1:
        return mask.copy()
    kernel = np.ones((ERODE_KERNEL, ERODE_KERNEL), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)


def make_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is not available. Install/use OpenCV with SIFT enabled.")
    return cv2.SIFT_create(nfeatures=MAX_SIFT_FEATURES, contrastThreshold=0.01, edgeThreshold=10)


def match_descriptors(desc0, desc1):
    if desc0 is None or desc1 is None or len(desc0) == 0 or len(desc1) == 0:
        return [], np.zeros(0), np.zeros(0)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc0, desc1, k=2)
    matches = []
    ratios = []
    distances = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        ratio = float(m.distance / max(n.distance, 1e-12))
        if m.distance < RATIO_TEST * n.distance:
            matches.append(m)
            ratios.append(ratio)
            distances.append(float(m.distance))
    return matches, np.asarray(ratios, dtype=np.float64), np.asarray(distances, dtype=np.float64)


def estimate_2d_ransac_mask(pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    if len(pts0) < 4:
        return np.zeros(len(pts0), dtype=bool)
    try:
        _, inliers = cv2.estimateAffinePartial2D(
            pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
        )
    except Exception:
        return np.ones(len(pts0), dtype=bool)
    if inliers is None:
        return np.zeros(len(pts0), dtype=bool)
    return inliers.ravel().astype(bool)


def lk_refine_points(gray0: np.ndarray, gray1: np.ndarray, pts0: np.ndarray, pts1_init: np.ndarray):
    if len(pts0) == 0:
        return pts1_init, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    p0 = pts0.reshape(-1, 1, 2).astype(np.float32)
    p1_guess = pts1_init.reshape(-1, 1, 2).astype(np.float32)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, p1_guess, **lk_params)
    if p1 is None or st_fwd is None:
        return pts1_init, np.zeros(len(pts0), dtype=bool), np.full(len(pts0), np.inf)
    p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray0,
        p1,
        p0.copy(),
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    p1_flat = p1.reshape(-1, 2)
    if p0_back is None or st_back is None:
        return p1_flat, np.zeros(len(pts0), dtype=bool), np.full(len(pts0), np.inf)
    p0_back_flat = p0_back.reshape(-1, 2)
    fb_err = np.linalg.norm(p0_back_flat - pts0, axis=1)
    valid = (st_fwd.ravel() > 0) & (st_back.ravel() > 0) & np.isfinite(fb_err) & (fb_err < 1.5)
    return p1_flat.astype(np.float32), valid, fb_err.astype(np.float64)


def sample_depth_nearest(depth: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u = np.round(pts[:, 0]).astype(int)
    v = np.round(pts[:, 1]).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.full(len(pts), np.nan, dtype=np.float32)
    z[valid] = depth[v[valid], u[valid]]
    valid = valid & np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    return z, valid


def sample_boundary_distance(dist_img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = dist_img.shape
    u = np.round(pts[:, 0]).astype(int)
    v = np.round(pts[:, 1]).astype(int)
    out = np.zeros(len(pts), dtype=np.float32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out[valid] = dist_img[v[valid], u[valid]]
    return out


def backproject(K: np.ndarray, pts: np.ndarray, z: np.ndarray) -> np.ndarray:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (pts[:, 0] - cx) / fx * z
    y = (pts[:, 1] - cy) / fy * z
    return np.column_stack((x, y, z)).astype(np.float64)


def quality_weights(raw_dist: np.ndarray, ratio: np.ndarray, boundary_dist: np.ndarray, lk_fb_err: np.ndarray) -> np.ndarray:
    n = len(raw_dist)
    w = np.ones(n, dtype=np.float64)
    if n > 0 and np.max(raw_dist) > np.min(raw_dist):
        rd = (raw_dist - np.min(raw_dist)) / (np.max(raw_dist) - np.min(raw_dist) + 1e-12)
        w *= np.exp(-1.5 * rd)
    if n > 0:
        w *= np.exp(-1.0 * np.clip(ratio, 0.0, 1.0))
    bd = np.clip(boundary_dist / 20.0, 0.0, 1.0)
    w *= (0.25 + 0.75 * bd)
    if len(lk_fb_err) == n:
        w *= np.exp(-0.5 * (lk_fb_err / 1.0) ** 2)
    return np.maximum(w, 1e-4)


def measurement_quality(n_inliers: int, mean_residual_m: Optional[float], mean_lk_fb_err_px: Optional[float]) -> float:
    if n_inliers <= 0:
        return MIN_QUALITY_WEIGHT
    res = mean_residual_m if mean_residual_m is not None and np.isfinite(mean_residual_m) else RANSAC_THRESH_M
    lk = mean_lk_fb_err_px if mean_lk_fb_err_px is not None and np.isfinite(mean_lk_fb_err_px) else 1.5
    q = float(n_inliers) * math.exp(-res / max(RANSAC_THRESH_M, 1e-9)) * math.exp(-0.5 * lk)
    return max(q, MIN_QUALITY_WEIGHT)


def draw_debug_matches(path: Path, img0: np.ndarray, img1: np.ndarray, kp0, kp1, matches, selected_mask: np.ndarray):
    selected = [matches[i] for i in range(len(matches)) if i < len(selected_mask) and selected_mask[i]]
    if len(selected) == 0:
        selected = matches[:70]
    vis = cv2.drawMatches(img0, kp0, img1, kp1, selected[:70], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), vis)


# =============================================================================
# Position estimators
# =============================================================================

def mask_pixel_cloud(K: np.ndarray, depth: np.ndarray, mask: np.ndarray, rng: np.random.Generator, max_pixels: int):
    ys, xs = np.where((mask > 0) & np.isfinite(depth) & (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M))
    n = len(xs)
    if n == 0:
        return np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
    if n > max_pixels:
        idx = rng.choice(np.arange(n), size=max_pixels, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    pts2 = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    z = depth[ys, xs].astype(np.float64)
    P = backproject(K, pts2, z)
    return P, pts2, z


def estimate_position_methods(K: np.ndarray, depth: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Dict:
    P_dense, pts2_dense, z_dense = mask_pixel_cloud(K, depth, mask, rng, MASK_CENTROID_MAX_PIXELS)
    out = {
        "visible_centroid": np.full(3, np.nan),
        "range_radius_center": np.full(3, np.nan),
        "sphere_fit_center": np.full(3, np.nan),
        "n_mask_points": int(len(P_dense)),
        "sphere_fit_success": False,
        "sphere_fit_cost": None,
        "sphere_fit_n_points": 0,
        "front_depth_m": None,
        "median_depth_m": None,
    }
    if len(P_dense) == 0:
        return out

    out["visible_centroid"] = P_dense.mean(axis=0)
    uv_centroid = pts2_dense.mean(axis=0)
    front_depth = float(np.percentile(z_dense, RANGE_FRONT_PERCENTILE))
    med_depth = float(np.percentile(z_dense, 50.0))
    corrected_z = front_depth + ASTEROID_RADIUS_M
    out["front_depth_m"] = front_depth
    out["median_depth_m"] = med_depth
    out["range_radius_center"] = backproject(K, uv_centroid.reshape(1, 2), np.array([corrected_z]))[0]

    P_fit, _, _ = mask_pixel_cloud(K, depth, mask, rng, SPHERE_FIT_MAX_PIXELS)
    out["sphere_fit_n_points"] = int(len(P_fit))
    if len(P_fit) < SPHERE_FIT_MIN_PIXELS:
        return out

    c0 = out["range_radius_center"].copy()
    if not np.all(np.isfinite(c0)):
        c0 = out["visible_centroid"].copy()
        c0[2] = front_depth + ASTEROID_RADIUS_M

    if SCIPY_AVAILABLE:
        def residual(c):
            return np.linalg.norm(P_fit - c.reshape(1, 3), axis=1) - ASTEROID_RADIUS_M
        try:
            lower = c0 + np.array([-1.5, -1.5, -1.5])
            upper = c0 + np.array([1.5, 1.5, 1.5])
            res = least_squares(
                residual,
                c0,
                loss=SPHERE_FIT_ROBUST_LOSS,
                f_scale=0.03,
                bounds=(lower, upper),
                max_nfev=SPHERE_FIT_MAX_NFEV,
            )
            out["sphere_fit_center"] = res.x.astype(np.float64)
            out["sphere_fit_success"] = bool(res.success)
            out["sphere_fit_cost"] = float(res.cost)
        except Exception:
            out["sphere_fit_center"] = c0
    else:
        X = P_fit
        A = np.column_stack((2.0 * X, np.ones(len(X))))
        b = np.sum(X * X, axis=1) - ASTEROID_RADIUS_M ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            out["sphere_fit_center"] = sol[:3]
            out["sphere_fit_success"] = True
        except Exception:
            out["sphere_fit_center"] = c0
    return out


# =============================================================================
# One-direction feature/depth rigid transform estimate
# =============================================================================

def estimate_direction(img0, img1, depth0, depth1, mask0_raw, mask1_raw, K, sift, debug_path=None) -> Dict:
    result = {
        "success": False,
        "R": None,
        "t": None,
        "n_kp0": 0,
        "n_kp1": 0,
        "n_matches_raw": 0,
        "n_inliers_2d": 0,
        "n_lk_valid": 0,
        "n_boundary_valid": 0,
        "n_valid_3d": 0,
        "n_inliers_3d": 0,
        "mean_residual_m": None,
        "mean_lk_fb_err_px": None,
        "quality": MIN_QUALITY_WEIGHT,
        "inlier_centroid_frame0": np.full(3, np.nan),
        "inlier_centroid_frame1": np.full(3, np.nan),
    }
    mask0 = erode_mask(mask0_raw)
    mask1 = erode_mask(mask1_raw)
    dist0 = boundary_distance(mask0_raw)
    dist1 = boundary_distance(mask1_raw)
    gray0 = preprocess_gray(img0)
    gray1 = preprocess_gray(img1)

    kp0, desc0 = sift.detectAndCompute(gray0, mask0)
    kp1, desc1 = sift.detectAndCompute(gray1, mask1)
    kp0 = [] if kp0 is None else kp0
    kp1 = [] if kp1 is None else kp1
    result["n_kp0"] = len(kp0)
    result["n_kp1"] = len(kp1)

    matches, ratios, raw_dists = match_descriptors(desc0, desc1)
    result["n_matches_raw"] = len(matches)
    if len(matches) < MIN_2D_MATCHES:
        return result

    pts0 = np.array([kp0[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts1 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.float32)
    in2d = estimate_2d_ransac_mask(pts0, pts1)
    result["n_inliers_2d"] = int(np.count_nonzero(in2d))
    if result["n_inliers_2d"] < MIN_3D_MATCHES:
        return result

    pts0 = pts0[in2d]
    pts1 = pts1[in2d]
    matches_2d = [matches[i] for i in np.where(in2d)[0]]
    ratios = ratios[in2d]
    raw_dists = raw_dists[in2d]

    pts1_lk, lk_valid, lk_fb_err = lk_refine_points(gray0, gray1, pts0, pts1)
    result["n_lk_valid"] = int(np.count_nonzero(lk_valid))
    if result["n_lk_valid"] < MIN_3D_MATCHES:
        return result

    pts0 = pts0[lk_valid]
    pts1 = pts1_lk[lk_valid]
    lk_fb_err = lk_fb_err[lk_valid]
    matches_lk = [matches_2d[i] for i in np.where(lk_valid)[0]]
    ratios = ratios[lk_valid]
    raw_dists = raw_dists[lk_valid]

    bd0 = sample_boundary_distance(dist0, pts0)
    bd1 = sample_boundary_distance(dist1, pts1)
    bd = np.minimum(bd0, bd1)
    boundary_valid = bd >= MIN_BOUNDARY_DIST_PX
    result["n_boundary_valid"] = int(np.count_nonzero(boundary_valid))
    if result["n_boundary_valid"] < MIN_3D_MATCHES:
        return result

    pts0 = pts0[boundary_valid]
    pts1 = pts1[boundary_valid]
    bd = bd[boundary_valid]
    lk_fb_err = lk_fb_err[boundary_valid]
    matches_final = [matches_lk[i] for i in np.where(boundary_valid)[0]]
    ratios = ratios[boundary_valid]
    raw_dists = raw_dists[boundary_valid]

    z0, v0 = sample_depth_nearest(depth0, pts0)
    z1, v1 = sample_depth_nearest(depth1, pts1)
    valid3d = v0 & v1
    result["n_valid_3d"] = int(np.count_nonzero(valid3d))
    if result["n_valid_3d"] < MIN_3D_MATCHES:
        return result

    pts0_v = pts0[valid3d]
    pts1_v = pts1[valid3d]
    z0_v = z0[valid3d]
    z1_v = z1[valid3d]
    bd_v = bd[valid3d]
    lk_v = lk_fb_err[valid3d]
    ratio_v = ratios[valid3d]
    rawdist_v = raw_dists[valid3d]

    P0 = backproject(K, pts0_v, z0_v)
    P1 = backproject(K, pts1_v, z1_v)
    weights = quality_weights(rawdist_v, ratio_v, bd_v, lk_v)
    R, t, in3d, mean_res = ransac_rigid_3d(P0, P1, thresh=RANSAC_THRESH_M, iters=RANSAC_ITERS, weights=weights)
    if R is None or np.count_nonzero(in3d) < MIN_3D_MATCHES:
        return result

    result["success"] = True
    result["R"] = R
    result["t"] = t
    result["n_inliers_3d"] = int(np.count_nonzero(in3d))
    result["mean_residual_m"] = None if mean_res is None else float(mean_res)
    result["mean_lk_fb_err_px"] = float(np.mean(lk_v[in3d])) if np.any(in3d) else None
    result["quality"] = measurement_quality(result["n_inliers_3d"], result["mean_residual_m"], result["mean_lk_fb_err_px"])
    result["inlier_centroid_frame0"] = np.mean(P0[in3d], axis=0)
    result["inlier_centroid_frame1"] = np.mean(P1[in3d], axis=0)

    if debug_path is not None:
        selected = np.zeros(len(matches_final), dtype=bool)
        if len(selected) >= len(in3d):
            selected[:len(in3d)] = in3d
        draw_debug_matches(debug_path, img0, img1, kp0, kp1, matches_final, selected)
    return result


# =============================================================================
# Metrics / plotting
# =============================================================================

def position_metrics(name: str, est: np.ndarray, gt: np.ndarray) -> Dict:
    err = est - gt
    norms = np.linalg.norm(err, axis=1)
    finite = np.isfinite(norms)
    if not np.any(finite):
        return {"name": name, "mean_error_m": None, "median_error_m": None, "max_error_m": None, "final_error_m": None}
    e = err[finite]
    n = norms[finite]
    return {
        "name": name,
        "mean_error_m": float(np.mean(n)),
        "median_error_m": float(np.median(n)),
        "max_error_m": float(np.max(n)),
        "final_error_m": float(norms[np.where(finite)[0][-1]]),
        "rmse_m": float(np.sqrt(np.mean(n ** 2))),
        "mean_abs_component_error_m": np.mean(np.abs(e), axis=0).tolist(),
        "bias_components_m": np.mean(e, axis=0).tolist(),
    }


def save_position_component_plot(path: Path, title: str, frames: np.ndarray, est: np.ndarray, gt: np.ndarray):
    labels = ["x", "y", "z"]
    plt.figure(figsize=(12, 5))
    for j in range(3):
        plt.plot(frames, est[:, j], label=f"est {labels[j]}")
        plt.plot(frames, gt[:, j], linestyle="--", label=f"true {labels[j]}")
    plt.xlabel("Frame")
    plt.ylabel("Position [m]")
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 90)
    print("SCRIPT 48: position + attitude tracker/evaluator for translating asteroid")
    print("=" * 90)
    print(f"DATA_ROOT  = {DATA_ROOT}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"SCIPY_AVAILABLE = {SCIPY_AVAILABLE}")
    print("")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")
    if not POSES_JSON.exists():
        raise FileNotFoundError(f"Missing frame_poses.json: {POSES_JSON}")
    if not INTRINSICS_JSON.exists():
        raise FileNotFoundError(f"Missing camera_intrinsics.json: {INTRINSICS_JSON}")

    with open(POSES_JSON, "r", encoding="utf-8") as f:
        frame_meta = json.load(f)
    with open(INTRINSICS_JSON, "r", encoding="utf-8") as f:
        intr = json.load(f)
    K = np.array(intr["K"], dtype=np.float64)

    rgb_files = sorted(RGB_DIR.glob("*.png"))
    depth_files = sorted(DEPTH_DIR.glob("*.npy"))
    mask_files = sorted(MASK_DIR.glob("*.png"))
    n_frames = min(len(rgb_files), len(depth_files), len(mask_files), len(frame_meta))
    if n_frames < 2:
        raise RuntimeError("Need at least two frames.")

    rgb_files = rgb_files[:n_frames]
    depth_files = depth_files[:n_frames]
    mask_files = mask_files[:n_frames]
    frame_meta = frame_meta[:n_frames]

    R_gt_abs = [np.array(frame_meta[i]["object_rotation_body_to_camera"], dtype=np.float64) for i in range(n_frames)]
    R_gt_rel = [R_gt_abs[i + 1] @ R_gt_abs[i].T for i in range(n_frames - 1)]
    t_gt = np.array([np.array(frame_meta[i]["object_translation_object_in_camera"], dtype=np.float64) for i in range(n_frames)])

    print(f"Frames: {n_frames}")
    print(f"Pairs:  {n_frames - 1}")
    print(f"GT position range min: {t_gt.min(axis=0)}")
    print(f"GT position range max: {t_gt.max(axis=0)}")
    print("")

    rng = np.random.default_rng(RNG_SEED)
    sift = make_sift()

    # Position estimates.
    pos_visible = []
    pos_range_radius = []
    pos_sphere_fit = []
    pos_info = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        depth = load_depth(depth_files[i])
        mask = load_mask(mask_files[i])
        info = estimate_position_methods(K, depth, mask, rng)
        pos_visible.append(info["visible_centroid"])
        pos_range_radius.append(info["range_radius_center"])
        pos_sphere_fit.append(info["sphere_fit_center"])
        pos_info.append({
            "frame_index": i,
            "n_mask_points": info["n_mask_points"],
            "front_depth_m": info["front_depth_m"],
            "median_depth_m": info["median_depth_m"],
            "sphere_fit_success": info["sphere_fit_success"],
            "sphere_fit_cost": info["sphere_fit_cost"],
            "sphere_fit_n_points": info["sphere_fit_n_points"],
        })
    pos_visible = np.array(pos_visible)
    pos_range_radius = np.array(pos_range_radius)
    pos_sphere_fit = np.array(pos_sphere_fit)
    position_runtime_s = time.perf_counter() - t0

    # Pairwise attitude / relative transform estimates.
    R_forward, R_back_inv, R_bidir = [], [], []
    t_forward_rel, t_back_inv_rel, t_bidir_rel = [], [], []
    err_forward, err_back_inv, err_bidir = [], [], []
    forward_success, backward_success, bidir_success = [], [], []
    alpha_values, pair_quality_bidir = [], []
    per_pair = []
    t_tracking0 = time.perf_counter()

    for k in range(n_frames - 1):
        img0 = load_rgb(rgb_files[k])
        img1 = load_rgb(rgb_files[k + 1])
        depth0 = load_depth(depth_files[k])
        depth1 = load_depth(depth_files[k + 1])
        mask0 = load_mask(mask_files[k])
        mask1 = load_mask(mask_files[k + 1])
        debug_f = OUTPUT_DIR / f"debug_forward_pair_{k:04d}.png" if k % SAVE_DEBUG_EVERY == 0 else None
        debug_b = OUTPUT_DIR / f"debug_backward_pair_{k:04d}.png" if k % SAVE_DEBUG_EVERY == 0 else None

        fwd = estimate_direction(img0, img1, depth0, depth1, mask0, mask1, K, sift, debug_path=debug_f)
        bwd = estimate_direction(img1, img0, depth1, depth0, mask1, mask0, K, sift, debug_path=debug_b)

        R_f = fwd["R"] if fwd["success"] else None
        t_f = fwd["t"] if fwd["success"] else None
        if bwd["success"]:
            R_bi = reorthonormalize(bwd["R"].T)
            t_bi = -R_bi @ bwd["t"]
        else:
            R_bi = None
            t_bi = None

        if R_f is not None and R_bi is not None:
            qf = float(fwd["quality"])
            qb = float(bwd["quality"])
            alpha = qb / max(qf + qb, 1e-12) if QUALITY_WEIGHTED_AVERAGE else 0.5
            alpha = float(np.clip(alpha, 0.25, 0.75))
            R_use = so3_interpolate(R_f, R_bi, alpha)
            t_use = (1.0 - alpha) * t_f + alpha * t_bi
            used = "bidirectional"
            q_use = (1.0 - alpha) * qf + alpha * qb
        elif R_f is not None:
            alpha = 0.0
            R_use = R_f
            t_use = t_f
            used = "forward_only"
            q_use = float(fwd["quality"])
        elif R_bi is not None:
            alpha = 1.0
            R_use = R_bi
            t_use = t_bi
            used = "backward_only"
            q_use = float(bwd["quality"])
        else:
            alpha = 0.5
            R_use = np.eye(3)
            t_use = np.zeros(3)
            used = "identity_fallback"
            q_use = MIN_QUALITY_WEIGHT

        R_forward.append(R_f if R_f is not None else np.eye(3))
        R_back_inv.append(R_bi if R_bi is not None else np.eye(3))
        R_bidir.append(R_use)
        t_forward_rel.append(t_f if t_f is not None else np.zeros(3))
        t_back_inv_rel.append(t_bi if t_bi is not None else np.zeros(3))
        t_bidir_rel.append(t_use)

        R_gt = R_gt_rel[k]
        ef = rotation_angle_deg(R_f @ R_gt.T) if R_f is not None else np.nan
        eb = rotation_angle_deg(R_bi @ R_gt.T) if R_bi is not None else np.nan
        eu = rotation_angle_deg(R_use @ R_gt.T)
        err_forward.append(float(ef))
        err_back_inv.append(float(eb))
        err_bidir.append(float(eu))
        forward_success.append(bool(fwd["success"]))
        backward_success.append(bool(bwd["success"]))
        bidir_success.append(used != "identity_fallback")
        alpha_values.append(float(alpha))
        pair_quality_bidir.append(float(q_use))

        per_pair.append({
            "pair_index": k,
            "used": used,
            "alpha_toward_backward_inverted": float(alpha),
            "gt_pair_rotation_deg": float(rotation_angle_deg(R_gt)),
            "gt_position_delta_m": (t_gt[k + 1] - t_gt[k]).tolist(),
            "estimated_relative_translation_forward_m": (np.zeros(3) if t_f is None else t_f).tolist(),
            "estimated_relative_translation_backward_inverted_m": (np.zeros(3) if t_bi is None else t_bi).tolist(),
            "estimated_relative_translation_bidirectional_m": t_use.tolist(),
            "forward": {"success": bool(fwd["success"]), "pair_error_deg": None if not np.isfinite(ef) else float(ef), "n_inliers_3d": int(fwd["n_inliers_3d"]), "mean_residual_m": fwd["mean_residual_m"], "mean_lk_fb_err_px": fwd["mean_lk_fb_err_px"], "quality": float(fwd["quality"])},
            "backward_inverted": {"success": bool(bwd["success"]), "pair_error_deg": None if not np.isfinite(eb) else float(eb), "n_inliers_3d": int(bwd["n_inliers_3d"]), "mean_residual_m": bwd["mean_residual_m"], "mean_lk_fb_err_px": bwd["mean_lk_fb_err_px"], "quality": float(bwd["quality"])},
            "bidirectional": {"success": bool(used != "identity_fallback"), "pair_error_deg": float(eu), "quality": float(q_use)},
        })

        if k % 25 == 0 or k == n_frames - 2:
            print(f"Pair {k:03d}/{n_frames-2:03d} used={used:<16s} att_err_bidir={eu:7.4f} deg  inliers_f/b={fwd['n_inliers_3d']:4d}/{bwd['n_inliers_3d']:4d} alpha={alpha:5.2f}")

    tracking_runtime_s = time.perf_counter() - t_tracking0

    def integrate_abs(R_rel_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        R_abs = [R_gt_abs[0].copy()]
        err = [0.0]
        for i, R_rel in enumerate(R_rel_list):
            R_next = reorthonormalize(R_rel @ R_abs[-1])
            R_abs.append(R_next)
            err.append(rotation_angle_deg(R_next @ R_gt_abs[i + 1].T))
        return R_abs, err

    R_abs_fwd, abs_err_fwd = integrate_abs(R_forward)
    R_abs_bwd, abs_err_bwd = integrate_abs(R_back_inv)
    R_abs_bidir, abs_err_bidir = integrate_abs(R_bidir)

    metrics_position = {
        "visible_centroid": position_metrics("visible_centroid", pos_visible, t_gt),
        "range_radius_center": position_metrics("range_radius_center", pos_range_radius, t_gt),
        "sphere_fit_center": position_metrics("sphere_fit_center", pos_sphere_fit, t_gt),
    }
    metrics_attitude = {
        "forward": {"mean_pair_error_deg": float(np.nanmean(err_forward)), "median_pair_error_deg": float(np.nanmedian(err_forward)), "max_pair_error_deg": float(np.nanmax(err_forward)), "mean_abs_error_deg": float(np.mean(abs_err_fwd)), "median_abs_error_deg": float(np.median(abs_err_fwd)), "max_abs_error_deg": float(np.max(abs_err_fwd)), "final_abs_error_deg": float(abs_err_fwd[-1])},
        "backward_inverted": {"mean_pair_error_deg": float(np.nanmean(err_back_inv)), "median_pair_error_deg": float(np.nanmedian(err_back_inv)), "max_pair_error_deg": float(np.nanmax(err_back_inv)), "mean_abs_error_deg": float(np.mean(abs_err_bwd)), "median_abs_error_deg": float(np.median(abs_err_bwd)), "max_abs_error_deg": float(np.max(abs_err_bwd)), "final_abs_error_deg": float(abs_err_bwd[-1])},
        "bidirectional": {"mean_pair_error_deg": float(np.nanmean(err_bidir)), "median_pair_error_deg": float(np.nanmedian(err_bidir)), "max_pair_error_deg": float(np.nanmax(err_bidir)), "mean_abs_error_deg": float(np.mean(abs_err_bidir)), "median_abs_error_deg": float(np.median(abs_err_bidir)), "max_abs_error_deg": float(np.max(abs_err_bidir)), "final_abs_error_deg": float(abs_err_bidir[-1])},
    }

    # Save arrays.
    np.save(OUTPUT_DIR / "gt_position_cv.npy", t_gt)
    np.save(OUTPUT_DIR / "position_visible_centroid.npy", pos_visible)
    np.save(OUTPUT_DIR / "position_range_radius_center.npy", pos_range_radius)
    np.save(OUTPUT_DIR / "position_sphere_fit_center.npy", pos_sphere_fit)
    np.save(OUTPUT_DIR / "R_forward_relative.npy", np.array(R_forward))
    np.save(OUTPUT_DIR / "R_backward_inverted_relative.npy", np.array(R_back_inv))
    np.save(OUTPUT_DIR / "R_bidirectional_relative.npy", np.array(R_bidir))
    np.save(OUTPUT_DIR / "t_forward_relative.npy", np.array(t_forward_rel))
    np.save(OUTPUT_DIR / "t_backward_inverted_relative.npy", np.array(t_back_inv_rel))
    np.save(OUTPUT_DIR / "t_bidirectional_relative.npy", np.array(t_bidir_rel))
    np.save(OUTPUT_DIR / "R_absolute_forward.npy", np.array(R_abs_fwd))
    np.save(OUTPUT_DIR / "R_absolute_backward_inverted.npy", np.array(R_abs_bwd))
    np.save(OUTPUT_DIR / "R_absolute_bidirectional.npy", np.array(R_abs_bidir))
    np.save(OUTPUT_DIR / "pairwise_error_forward_deg.npy", np.array(err_forward))
    np.save(OUTPUT_DIR / "pairwise_error_backward_inverted_deg.npy", np.array(err_back_inv))
    np.save(OUTPUT_DIR / "pairwise_error_bidirectional_deg.npy", np.array(err_bidir))
    np.save(OUTPUT_DIR / "absolute_error_forward_deg.npy", np.array(abs_err_fwd))
    np.save(OUTPUT_DIR / "absolute_error_backward_inverted_deg.npy", np.array(abs_err_bwd))
    np.save(OUTPUT_DIR / "absolute_error_bidirectional_deg.npy", np.array(abs_err_bidir))
    np.save(OUTPUT_DIR / "alpha_bidirectional.npy", np.array(alpha_values))

    # Plots.
    frames = np.arange(n_frames)
    pairs = np.arange(n_frames - 1)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, abs_err_fwd, label="forward")
    plt.plot(frames, abs_err_bwd, label="backward inverted")
    plt.plot(frames, abs_err_bidir, label="bidirectional")
    plt.xlabel("Frame")
    plt.ylabel("Absolute attitude error [deg]")
    plt.title("Script 48 cumulative attitude error comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_attitude_error_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(pairs, err_forward, label="forward")
    plt.plot(pairs, err_back_inv, label="backward inverted")
    plt.plot(pairs, err_bidir, label="bidirectional")
    plt.xlabel("Frame pair index")
    plt.ylabel("Pairwise attitude error [deg]")
    plt.title("Script 48 pairwise attitude error comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pairwise_attitude_error_comparison.png", dpi=160)
    plt.close()

    pos_err_visible = np.linalg.norm(pos_visible - t_gt, axis=1)
    pos_err_range = np.linalg.norm(pos_range_radius - t_gt, axis=1)
    pos_err_sphere = np.linalg.norm(pos_sphere_fit - t_gt, axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(frames, pos_err_visible, label="visible depth centroid")
    plt.plot(frames, pos_err_range, label="front depth + radius")
    plt.plot(frames, pos_err_sphere, label="known-radius sphere fit")
    plt.xlabel("Frame")
    plt.ylabel("Position error norm [m]")
    plt.title("Script 48 position error norms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_error_norms.png", dpi=160)
    plt.close()

    save_position_component_plot(OUTPUT_DIR / "position_components_visible_centroid.png", "Script 48 position components: visible depth centroid", frames, pos_visible, t_gt)
    save_position_component_plot(OUTPUT_DIR / "position_components_range_radius.png", "Script 48 position components: front depth + radius", frames, pos_range_radius, t_gt)
    save_position_component_plot(OUTPUT_DIR / "position_components_sphere_fit.png", "Script 48 position components: known-radius sphere fit", frames, pos_sphere_fit, t_gt)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, t_gt[:, 2], label="true range z")
    plt.plot(frames, pos_visible[:, 2], label="visible centroid z")
    plt.plot(frames, pos_range_radius[:, 2], label="front+radius z")
    plt.plot(frames, pos_sphere_fit[:, 2], label="sphere fit z")
    plt.xlabel("Frame")
    plt.ylabel("Range z [m]")
    plt.title("Script 48 range estimate comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "range_z_comparison.png", dpi=160)
    plt.close()

    f_inliers = [p["forward"]["n_inliers_3d"] for p in per_pair]
    b_inliers = [p["backward_inverted"]["n_inliers_3d"] for p in per_pair]
    plt.figure(figsize=(12, 5))
    plt.plot(pairs, f_inliers, label="forward 3D inliers")
    plt.plot(pairs, b_inliers, label="backward 3D inliers")
    plt.xlabel("Frame pair index")
    plt.ylabel("Inlier count")
    plt.title("Script 48 SIFT-LK 3D RANSAC inlier counts")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "match_inlier_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(pairs, alpha_values, label="alpha toward backward-inverted")
    plt.xlabel("Frame pair index")
    plt.ylabel("SO(3) interpolation alpha")
    plt.title("Script 48 bidirectional averaging weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bidirectional_weight_alpha.png", dpi=160)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(t_gt[:, 0], t_gt[:, 1], t_gt[:, 2], label="true center")
    ax.plot(pos_visible[:, 0], pos_visible[:, 1], pos_visible[:, 2], label="visible centroid")
    ax.plot(pos_range_radius[:, 0], pos_range_radius[:, 1], pos_range_radius[:, 2], label="front+radius")
    ax.plot(pos_sphere_fit[:, 0], pos_sphere_fit[:, 1], pos_sphere_fit[:, 2], label="sphere fit")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Script 48 3D position trajectories")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_trajectory_3d.png", dpi=160)
    plt.close()

    summary = {
        "script": "48_position_attitude_translating_asteroid_tracker.py",
        "data_root": str(DATA_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "n_frames": n_frames,
        "n_pairs": n_frames - 1,
        "configuration": {
            "detector": "SIFT",
            "lk_refine": True,
            "weighted_svd": True,
            "bidirectional_average": True,
            "quality_weighted_bidirectional_average": QUALITY_WEIGHTED_AVERAGE,
            "ratio_test": RATIO_TEST,
            "ransac_thresh_m": RANSAC_THRESH_M,
            "erode_kernel": ERODE_KERNEL,
            "min_boundary_dist_px": MIN_BOUNDARY_DIST_PX,
            "asteroid_radius_m_for_position_correction": ASTEROID_RADIUS_M,
            "scipy_available_for_sphere_fit": SCIPY_AVAILABLE,
            "position_methods": ["visible depth centroid", "front depth percentile + known radius", "known-radius sphere fit"],
        },
        "gt_position_range_m": {"min": t_gt.min(axis=0).tolist(), "max": t_gt.max(axis=0).tolist()},
        "success": {"forward_success": int(np.count_nonzero(forward_success)), "backward_success": int(np.count_nonzero(backward_success)), "bidirectional_nonfallback": int(np.count_nonzero(bidir_success))},
        "runtime": {"position_total_s": float(position_runtime_s), "position_mean_per_frame_ms": float(position_runtime_s / n_frames * 1000.0), "tracking_total_s": float(tracking_runtime_s), "tracking_mean_per_pair_ms": float(tracking_runtime_s / max(n_frames - 1, 1) * 1000.0)},
        "attitude_metrics": metrics_attitude,
        "position_metrics": metrics_position,
        "mean_inliers": {"forward": float(np.mean(f_inliers)), "backward": float(np.mean(b_inliers))},
        "alpha": {"mean": float(np.mean(alpha_values)), "median": float(np.median(alpha_values)), "min": float(np.min(alpha_values)), "max": float(np.max(alpha_values))},
        "position_per_frame": pos_info,
        "per_pair": per_pair,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print("=" * 90)
    print("SCRIPT 48 COMPLETE")
    print("=" * 90)
    print(f"Results: {OUTPUT_DIR}")
    print("")
    print("Attitude final absolute error:")
    print(f"  forward:             {abs_err_fwd[-1]:.6f} deg")
    print(f"  backward inverted:   {abs_err_bwd[-1]:.6f} deg")
    print(f"  bidirectional avg:   {abs_err_bidir[-1]:.6f} deg")
    print("")
    print("Position mean error:")
    print(f"  visible centroid:    {metrics_position['visible_centroid']['mean_error_m']:.6f} m")
    print(f"  front + radius:      {metrics_position['range_radius_center']['mean_error_m']:.6f} m")
    print(f"  sphere fit:          {metrics_position['sphere_fit_center']['mean_error_m']:.6f} m")
    print("")
    print("Position final error:")
    print(f"  visible centroid:    {metrics_position['visible_centroid']['final_error_m']:.6f} m")
    print(f"  front + radius:      {metrics_position['range_radius_center']['final_error_m']:.6f} m")
    print(f"  sphere fit:          {metrics_position['sphere_fit_center']['final_error_m']:.6f} m")
    print("=" * 90)


if __name__ == "__main__":
    main()
