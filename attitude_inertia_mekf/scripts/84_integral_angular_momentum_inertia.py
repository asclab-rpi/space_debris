"""Derivative-free torque-free inertia identification from a frozen attitude track.

Estimator stage reads only a frozen estimated attitude trajectory and timing.
Ground truth is loaded later, solely for evaluation and an oracle identifiability
diagnostic.  Absolute inertia scale is not observable; all matrices use trace=1.

Model:
    H_I = R_IB(t) I_B omega_B(t) = constant

Rather than differentiating angular velocity in Euler's equation, this script
fits a constant inertial angular-momentum vector across the full trajectory.
Physical principal moments are parameterized as
    J = [b+c, a+c, a+b], a,b,c > 0,
which guarantees positive definiteness and the rigid-body triangle inequalities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


DEFAULT_ROOT = Path(__file__).resolve().parent / "results" / "script81_compound_satellite_debris"


def reorthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def relative_body_rates(R: np.ndarray, dt: float, window: int = 21, polyorder: int = 3):
    """Return midpoint R_IB and body omega without differentiating omega.

    Assumes R[k] maps body coordinates at time k into a common inertial/reference
    frame.  Log(R[k].T R[k+1])/dt is therefore body-resolved angular velocity.
    A Savitzky-Golay filter reduces visual increment noise.
    """
    R = np.asarray([reorthonormalize(x) for x in R], float)
    dR = np.asarray([R[k].T @ R[k + 1] for k in range(len(R) - 1)])
    omega_raw = Rotation.from_matrix(dR).as_rotvec() / dt
    n = len(omega_raw)
    win = min(int(window), n if n % 2 else n - 1)
    if win >= 5:
        win = max(win, polyorder + 2 + ((polyorder + 2) % 2 == 0))
        omega = savgol_filter(omega_raw, win, min(polyorder, win - 2), axis=0, mode="interp")
    else:
        omega = omega_raw.copy()
    half = Rotation.from_rotvec(Rotation.from_matrix(dR).as_rotvec() * 0.5).as_matrix()
    R_mid = np.asarray([R[k] @ half[k] for k in range(n)])
    return R_mid, omega, omega_raw


def inertia_from_parameters(x: np.ndarray) -> np.ndarray:
    """Map 5 unconstrained values to a trace-one physically realizable inertia.

    c is fixed to one because a common scaling of a,b,c disappears when trace
    is normalized.  Removing that gauge freedom makes conditioning meaningful.
    """
    Q = Rotation.from_rotvec(x[:3]).as_matrix()
    a, b = np.exp(np.clip(x[3:5], -12.0, 12.0)); c = 1.0
    moments = np.array([b + c, a + c, a + b], float)
    moments /= moments.sum()
    return Q @ np.diag(moments) @ Q.T


def residual_vector(x, R_mid, omega, scale):
    I = inertia_from_parameters(x)
    H = x[5:8]
    predicted = np.einsum("nij,nj->ni", R_mid, omega @ I.T)
    return ((predicted - H) / scale).ravel()


def initial_guesses(R_mid, omega, rng, starts):
    speed = np.median(np.linalg.norm(omega, axis=1))
    H0 = np.mean(np.einsum("nij,nj->ni", R_mid, omega / 3.0), axis=0)
    yield np.r_[np.zeros(3), np.zeros(2), H0]
    for _ in range(starts - 1):
        yield np.r_[rng.normal(0, 1.2, 3), rng.normal(0, 0.8, 2), H0 + rng.normal(0, max(speed, 1e-5) * 0.05, 3)]


def fit_inertia(R: np.ndarray, dt: float, window: int = 21, starts: int = 32, seed: int = 7):
    R_mid, omega, omega_raw = relative_body_rates(R, dt, window)
    # Robust scale in angular-momentum units; trace(I)=1 fixes the arbitrary scale.
    scale = max(np.median(np.linalg.norm(omega - np.median(omega, axis=0), axis=1)), 1e-5)
    rng = np.random.default_rng(seed)
    best = None
    for x0 in initial_guesses(R_mid, omega, rng, starts):
        fit = least_squares(
            residual_vector, x0, args=(R_mid, omega, scale), loss="soft_l1",
            f_scale=1.0, max_nfev=2500, xtol=1e-11, ftol=1e-11, gtol=1e-11,
        )
        score = np.mean(np.square(residual_vector(fit.x, R_mid, omega, scale)))
        if best is None or score < best[0]:
            best = (score, fit)
    score, fit = best
    I = inertia_from_parameters(fit.x)
    H_samples = np.einsum("nij,nj->ni", R_mid, omega @ I.T)
    H = fit.x[5:8]
    centered = H_samples - H
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    rel_rms = rms / max(float(np.linalg.norm(H)), 1e-15)
    s = np.linalg.svd(fit.jac, compute_uv=False)
    jac_condition = float(s[0] / max(s[-1], 1e-15))
    return {
        "I": I, "H": H, "omega": omega, "omega_raw": omega_raw,
        "R_mid": R_mid, "H_samples": H_samples, "relative_H_rms": rel_rms,
        "jacobian_singular_values": s, "jacobian_condition": jac_condition,
        "cost": float(fit.cost), "optimality": float(fit.optimality),
        "success": bool(fit.success), "message": str(fit.message),
    }


def bootstrap(R, dt, base, window, samples=100, block=12, seed=19):
    """Moving-block bootstrap of angular-momentum samples; returns inertia draws."""
    Rm, w, _ = relative_body_rates(R, dt, window)
    n = len(w); rng = np.random.default_rng(seed); draws = []
    for b in range(samples):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            idx.extend(range(start, min(start + block, n)))
        idx = np.asarray(idx[:n])
        # Fit resampled conservation equations. Order is irrelevant to this global invariant.
        rb = fit_inertia_from_samples(Rm[idx], w[idx], base, seed + b)
        draws.append(rb)
    return np.asarray(draws)


def fit_inertia_from_samples(Rm, w, base, seed):
    scale = max(np.median(np.linalg.norm(w - np.median(w, axis=0), axis=1)), 1e-5)
    vals, vecs = np.linalg.eigh(base["I"])
    # Principal-moment triangle inverse: a=(J2+J3-J1)/2, etc.
    a = max((vals[1] + vals[2] - vals[0]) / 2, 1e-8)
    b = max((vals[0] + vals[2] - vals[1]) / 2, 1e-8)
    c = max((vals[0] + vals[1] - vals[2]) / 2, 1e-8)
    x0 = np.r_[Rotation.from_matrix(vecs).as_rotvec(), np.log([a/c, b/c]), base["H"]]
    fit = least_squares(residual_vector, x0, args=(Rm, w, scale), loss="soft_l1", max_nfev=1000)
    return inertia_from_parameters(fit.x)


def normalized_error(I, truth):
    T = np.asarray(truth, float); T /= np.trace(T)
    return float(np.linalg.norm(I - T) / np.linalg.norm(T))


def load_ground_truth_rotations(root: Path, n: int):
    rows = [json.loads((root / "ground_truth" / f"pose_{k:06d}.json").read_text()) for k in range(n)]
    R = np.asarray([x["rotation_body_to_camera"] for x in rows], float)
    # Express all attitudes relative to frame zero, matching the frozen estimator anchor.
    return np.asarray([x @ R[0].T for x in R])


def save_plots(out, result, draws, Itrue=None):
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(result["omega"]))
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    ax[0].plot(t, np.degrees(result["omega_raw"]), alpha=.25)
    ax[0].plot(t, np.degrees(result["omega"]), linewidth=1.7)
    ax[0].set(title="Body angular velocity used by derivative-free identifier", ylabel="deg/s")
    ax[0].grid(alpha=.25)
    h = result["H_samples"]
    ax[1].plot(t, h); ax[1].axhline(result["H"][0], color="C0", ls="--")
    ax[1].axhline(result["H"][1], color="C1", ls="--"); ax[1].axhline(result["H"][2], color="C2", ls="--")
    ax[1].set(title="Inertial angular-momentum constancy", xlabel="transition", ylabel="normalized H")
    ax[1].grid(alpha=.25); fig.savefig(out / "angular_momentum_fit.png", dpi=170); plt.close(fig)

    mats = [("Estimate", result["I"])]
    if Itrue is not None: mats.append(("Truth (evaluation only)", Itrue / np.trace(Itrue)))
    fig, axes = plt.subplots(1, len(mats), figsize=(5.5 * len(mats), 4.6), squeeze=False, constrained_layout=True)
    lim = max(np.max(np.abs(m)) for _, m in mats)
    for a, (name, M) in zip(axes[0], mats):
        im = a.imshow(M, cmap="coolwarm", vmin=-lim, vmax=lim)
        for i in range(3):
            for j in range(3): a.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center")
        a.set(title=name, xticks=range(3), yticks=range(3))
    fig.colorbar(im, ax=axes.ravel().tolist(), label="trace-normalized inertia")
    fig.savefig(out / "inertia_matrix_comparison.png", dpi=170); plt.close(fig)

    moments = np.linalg.eigvalsh(draws)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([moments[:, i] for i in range(3)], tick_labels=["J1", "J2", "J3"])
    ax.scatter(range(1, 4), np.linalg.eigvalsh(result["I"]), color="C3", label="full-data estimate")
    if Itrue is not None:
        ax.scatter(range(1, 4), np.linalg.eigvalsh(Itrue / np.trace(Itrue)), color="black", marker="x", s=70, label="truth")
    ax.set(title="Moving-block bootstrap: normalized principal moments", ylabel="moment / trace(I)")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.savefig(out / "bootstrap_principal_moments.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--trajectory", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=100)
    args = ap.parse_args()
    root = args.root.resolve()
    traj = args.trajectory or root / "estimated_dynamics" / "R_est_frozen.npy"
    out = args.output or root / "inertia_integral_fit_script84"
    meta = json.loads((root / "dataset_summary.json").read_text())
    dt = 1.0 / float(meta["fps"])

    # ---------------- Estimator: no truth reads above this freeze marker. ----------------
    R_est = np.load(traj)
    result = fit_inertia(R_est, dt, args.window, args.starts)
    draws = bootstrap(R_est, dt, result, args.window, args.bootstrap) if args.bootstrap else np.empty((0, 3, 3))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "normalized_inertia_estimate.npy", result["I"])
    np.save(out / "bootstrap_inertia_draws.npy", draws)
    estimator_summary = {
        "method": "derivative-free constant inertial angular momentum",
        "ground_truth_used_by_estimator": False,
        "trajectory": str(traj), "frames": int(len(R_est)), "dt_s": dt,
        "savgol_window": args.window, "multistarts": args.starts,
        "normalized_inertia_trace_one": result["I"].tolist(),
        "principal_moments_trace_one": np.linalg.eigvalsh(result["I"]).tolist(),
        "constant_inertial_H": result["H"].tolist(),
        "relative_angular_momentum_rms": result["relative_H_rms"],
        "jacobian_condition_number": result["jacobian_condition"],
        "bootstrap_samples": int(len(draws)),
        "bootstrap_matrix_std": np.std(draws, axis=0).tolist() if len(draws) else None,
    }
    (out / "ESTIMATE_FROZEN_NO_GROUND_TRUTH").write_text(json.dumps(estimator_summary, indent=2))

    # ---------------- Evaluation only: truth may now be read. ----------------
    truth_props = json.loads((root / "inertial_properties.json").read_text())
    Itrue = np.asarray(truth_props["inertia_about_com_body_kg_m2"], float)
    Rgt = load_ground_truth_rotations(root, len(R_est))
    oracle = fit_inertia(Rgt, dt, args.window, max(12, args.starts // 2))
    report = {
        **estimator_summary,
        "evaluation_only": {
            "true_normalized_inertia": (Itrue / np.trace(Itrue)).tolist(),
            "estimated_normalized_relative_frobenius_error": normalized_error(result["I"], Itrue),
            "oracle_truth_pose_fit_normalized_inertia": oracle["I"].tolist(),
            "oracle_truth_pose_fit_relative_error": normalized_error(oracle["I"], Itrue),
            "oracle_relative_angular_momentum_rms": oracle["relative_H_rms"],
        },
        "interpretation": "Absolute inertia scale is unobservable. A large bootstrap spread or large Jacobian condition number means the motion does not identify even normalized inertia reliably.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    save_plots(out, result, draws, Itrue)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
