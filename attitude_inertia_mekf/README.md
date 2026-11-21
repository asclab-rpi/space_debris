# RGB-D Attitude and Inertia Estimation for a Tumbling Target

Estimates the attitude, angular velocity and normalized inertia of a torque-free tumbling target from simulated RGB-D camera images, then refines them with a multiplicative extended Kalman filter (MEKF). The included configuration uses an observer-eclipse dataset in which the target passes from sunlight through penumbra and umbra and back.

Author: Tevin McLaren, Advanced Space Concepts Laboratory, Rensselaer Polytechnic Institute

## Pipeline

1. **Visual frontend (script 85 notebook):** SIFT features with Lucas–Kanade refinement are tracked between consecutive frames on the target mask. The features are back-projected with depth into 3D, and RANSAC plus a weighted SVD gives the relative rotation of each frame pair. Chaining the pairs gives the attitude trajectory.
2. **Angular velocity:** body rates come from consecutive attitudes, smoothed with a Savitzky–Golay filter.
3. **Initial inertia estimate:** a derivative-free fit over the first `T_INIT` seconds finds the trace-normalized inertia that keeps the inertial angular momentum constant (torque-free motion). Only normalized inertia (ratios) is observable from attitude alone.
4. **MEKF:** the filter fuses the frame-pair angular-velocity measurements, starting at `T_INIT` from the initial estimate. It works in the principal frame of the initial inertia estimate, so it estimates attitude, angular velocity and two principal inertia ratios (I2/I1, I3/I1). The principal axes stay fixed at the initial estimate. A pair is rejected when its RANSAC inlier count falls below a threshold.

Ground truth is loaded only in the notebook's evaluation section, after all estimator outputs are frozen. The estimator never uses it.

## Folder layout

Everything for this pipeline lives in `attitude_inertia_mekf/`. Run all commands from that folder.

```
attitude_inertia_mekf/
├── 85_end_to_end_rgbd_attitude_inertia.ipynb   frontend, initial inertia estimate, evaluation, MEKF inputs
├── mekf_rgbd_principal.py                      MEKF with 2 principal inertia ratios
├── scripts/                                    modules imported by the notebook
└── requirements.txt
```

## Setup

Python 3.11.

```
pip install -r requirements.txt
```

The package is `numpy-quaternion`, imported as `quaternion`. Running `pip install quaternion` installs an unrelated package.

## Data

The rendered dataset is not stored in this repository.

- **Access:** the dataset is too large for git. Contact the author for a copy.
- **Expected structure** of the `eclipse_full` folder:

```
eclipse_full/
├── rgb/            rgb_*.png
├── depth/          depth_*.npy   (camera-axis Z depth, metres)
├── masks/          mask_*.png
├── ground_truth/   pose_*.json   (evaluation only)
├── K.npy           camera intrinsics
└── inertial_properties.json      (evaluation only)
```

The code looks for data in two places, set by environment variables:

| Variable | Contents | Default if unset |
|---|---|---|
| `RGBD_DATA_ROOT` | the folder that contains `eclipse_full` | `attitude_inertia_mekf/data` |
| `RGBD_STRIDED_ROOT` | where strided copies of the dataset are written | `<RGBD_DATA_ROOT>/strided` |

Either put the dataset in `attitude_inertia_mekf/data/eclipse_full`, or set the variables. On Windows:

```
setx RGBD_DATA_ROOT "D:\path\to\datasets"
setx RGBD_STRIDED_ROOT "D:\path\to\strided"
```

Restart Jupyter or your IDE after `setx`.

## Running

1. **Notebook:** open `85_end_to_end_rgbd_attitude_inertia.ipynb` from the `attitude_inertia_mekf` folder and run all cells. Settings are in the first code cell:
   - `FRAME_STRIDE`: keep every Nth frame. The default 5 gives 2 FPS from the 10 FPS renders.
   - `T_INIT`: seconds of data used for the initial inertia estimate.

   Results are written to `<strided dataset>/notebook85_end_to_end/`.
2. **MEKF:** run `python mekf_rgbd_principal.py`. It reads the notebook outputs, including the saved principal moments and axes, starts filtering at `T_INIT`, and plots:
   - attitude, angular velocity and the principal inertia ratios against truth, with uncertainty bounds
   - rejected measurements
   - eclipse periods

## Notes

- **Notebook outputs:** clear them before committing (Kernel → Restart & Clear Output), or install `nbstripout` and run `nbstripout --install` once in the repo.
- **Ground truth for the MEKF:** the notebook's evaluation section writes full-rate (10 Hz) truth arrays (`q_gt_cam0_wxyz.npy`, `omega_cam0_rad_s.npy`) for plotting MEKF errors. The filter itself never reads truth.
- **Rejected measurements:** frame pairs with fewer than `MIN_INLIERS` RANSAC inliers are rejected. The filter propagates through them without an update.
- **Eclipse marking:** it uses the 95th-percentile brightness of the target in each frame, so it tracks illumination rather than how much of the target is visible.
