#%% -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:33:38 2026

@author: Tevin

MEKF takes in vision-based angular velocity measurements from frontend script 85

notations
    d prefix: error state value
    hat: estimated value
    m: measured value
    t: ground truth value
    n: noisy value

frame convention
    C: camera frame (never moves, serves as inertial frame)
    B: body-fixed axes based on geometry of satellite model
    B': body-fixed axes that line up with C at t=0 (B' = R0*B)
    P: principal axes of the INITIAL inertia estimate (fixed in the body, P = B' rotated by V)

PRINCIPAL-AXIS VERSION: the filter body frame is P, so the inertia is diagonal and only
the two principal ratios are estimated. The principal axes themselves are fixed from the
initial estimate and are NOT refined by the filter.

GLobal state: x = [q (P -> C), omega (in P), p]
parameters: p = [p1 = I2/I1, 
                p2 = I3/I1]
                where I1 = 1 (smallest principal moment)

error state: dx = [alpha, domega, dp]
    (attitude error (rotation vector) denoted "alpha")
                     

Measurement : z = measured omega in camera frame from camera observations
predicted state: x_hat
predicted measurement: z_hat = predicted omega in camera frame
"""

import numpy as np
import matplotlib.pyplot as plt

import quaternion
from pathlib import Path
from scipy.spatial.transform import Rotation


#%% 
# initialize

import os
from pathlib import Path
DATA_ROOT    = Path(os.environ.get("RGBD_DATA_ROOT", Path(__file__).resolve().parent / "data"))
STRIDED_ROOT = Path(os.environ.get("RGBD_STRIDED_ROOT", DATA_ROOT / "strided"))

DATASET_observer_eclipse_full5 = STRIDED_ROOT / "eclipse_full_s0_e999_x5"


# ground truth and measurement files 
stride = 5
DATASET = DATASET_observer_eclipse_full5

OUTPUT = DATASET / "notebook85_end_to_end"
GT_DIR = DATASET / "ground_truth"

I_true = np.load(OUTPUT/"I_true.npy")
q_meas = np.load(OUTPUT/"q_est_frozen_wxyz.npy")

I_hat = np.load(OUTPUT/"normalized_inertia_estimate_init.npy")
q_true = np.load(OUTPUT/"q_gt_cam0_wxyz.npy")          # wxyz
omega_cam_true = np.load(OUTPUT/"omega_cam0_rad_s.npy")
omega_cam_meas = np.load(OUTPUT/"omega_cam_pair_rad_s.npy")
pair_valid = np.load(OUTPUT/"pair_valid.npy")


# principal frame P from the initial inertia estimate (computed and saved by script 85, cell 10)
J_hat = np.load(OUTPUT/"principal_moments_init.npy")   # ascending principal moments, trace = 1
V = np.load(OUTPUT/"principal_axes_init.npy")          # columns = principal axes in B', proper rotation
q_BP = quaternion.from_rotation_matrix(V)              # rotates P coordinates into B'


n_truth_frames = 1000
rate_truth = 10 # Hz, from script 85
dt = 1 / rate_truth
tstop = n_truth_frames / rate_truth 
t = np.arange(0, tstop, dt)
N = t.size

# global state
dim_rot = 7 # 4 q + 3 omega components
dim_p = 2 # 2 principal inertia ratios
dim_x = dim_rot + dim_p # 9

# error state
dim_rot_err = 6 # 3 attitude error + 3 omega error
dim_x_err = dim_rot_err + dim_p # 8

# measurements/observations
dim_z = 4 # 4 q components
dim_z_err = 3 # 3 attitude error components


# set up sampling rate ratios
rate_measure = int(rate_truth/stride) # Hz
rate_ratio = stride


q_meas_sample = q_meas
M = q_meas.shape[0]



#%% 

# define functions


def rotate_to_frame(q_array, omega_array):
    """"
    multiplies q_array as a Rotation matrix by omega_array, rotating omega into new frame
    must be np.arrays of size 4 and 3
    """

    return Rmat(q_array) @ omega_array

# measurement function
def h(x_interval):
    """
    inputs current estimates of q (B' -> C) and omega (B') across interval (of most recent omega values up to last measurement update)
    converts q to Rotation matrix
    multiplies by omega to rotate omega into C frame
    returns average omega across interval
    """
    v = np.array([rotate_to_frame(x[0:4], x[4:7]) for x in x_interval])          # inertial-frame omega at each step
    return (0.5*(v[0] + v[-1]) + v[1:-1].sum(0)) / rate_ratio          # trapezoid average

def H_jac(x_hat):
    """
    Jacobian of h(x_interval) wrt error state
    Returns H shape 3 x 11
    """

    Rq = Rmat(x_hat[:4])
    H = np.zeros((dim_z_err, dim_x_err))
    H[:, 0:3] = -Rq @ skew_symmetric(x_hat[4:7])                      
    H[:, 3:6] = Rq                                                    
    return H



def F_jac(x_hat_old, eps=1e-6):
    """
    outputs shape (dim_error+dim_p x dim_error+dim_p) = 8 x 8
    uses BODY-FRAME Euler equations, thus quaternion error equations used in predict/update must also be in body frame
    """
    q = x_hat_old[:4]
    omega = x_hat_old[4:7]
    p = x_hat_old[7:]
    
    F = np.eye(dim_x_err)
    
    # attitude error block
    F[0:3, 0:3] = np.eye(3) + -1*skew_symmetric(omega) * dt         # d(alpha + alpha_dot) wrt alpha = I3 - omega_skew
    F[0:3, 3:6] = np.eye(3) * dt                                    # d(alpha + alpha_dot) wrt omega = I3

    
    # omega block: finite-difference the nonlinear omega_dot
    base = calc_omega_dot(omega, p)
    
    # d(omega_dot)/d(omega)  -> 3x3
    J_omega = np.zeros((3, 3))
    for i in range(3):
        d_omega = omega.copy()
        d_omega[i] += eps
        J_omega[:, i] = (calc_omega_dot(d_omega, p) - base) / eps
    
    F[3:6, 3:6] += dt * J_omega   # add to existing identity, don't overwrite
    
    # d(omega_dot)/dp  -> 3x2
    J_p = np.zeros((3, dim_p))
    for i in range(dim_p):
        d_p = p.copy()
        d_p[i] += eps
        J_p[:, i] = (calc_omega_dot(omega, d_p) - base) / eps
    
    F[3:6, 6:] = dt * J_p
    
    return F

def f(x):
    """
    Nonlinear state propagation, one dt step
    inertia parameters held constant (affected only by Q, not f)
    """
    q = x[:4]
    omega = x[4:7]    
    p = x[7:]
    

    # quaternion kinematics
    q_dot = 0.5 * W_body(q).T @ omega # eq from [6]
    q_new = q + dt * q_dot
    q_new /= np.linalg.norm(q_new)

    
    # solve general Euler rotation equation: I*omega_dot + omega x (I*omega) = M
    omega_dot = calc_omega_dot(omega, p)
    omega_new = omega + dt * omega_dot



    x_new = np.empty_like(x)
    x_new[:4]  = q_new
    x_new[4:7] = omega_new
    x_new[7:]  = x[7:]     # parameters unchanged in propagation
    return x_new




def calc_omega_dot(omega, p):
    """
    solve general body frame Euler rotation equation: I*omega_dot + omega x (I*omega) = M
    """
    
    p1, p2 = p
    I = np.diag([1.0, p1, p2])       # principal frame: diagonal inertia
    
    M = np.zeros(3)
    b = M - np.cross(omega, I @ omega)
    return np.linalg.solve(I, b)

def skew_symmetric(vector_3d):
    v = vector_3d
    return np.array([
                     [0, -v[2], v[1],],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]
                     ])

def W_body(q):
    """
    quaternion rate matrix in body coordinates

    """
    q0, q1, q2, q3 = q
    
    return np.array([
        [-q1, q0, q3, -q2 ],
        [-q2, -q3, q0, q1],
        [-q3, q2, -q1, q0]
        ])


def inertia_tensor_to_parameters(I):
    """
    converts 3x3 inertia tensor into the 2 principal ratios [I2/I1, I3/I1]
    (principal moments sorted ascending, so I1 is the smallest)
    """
    J = np.linalg.eigvalsh(I)
    return J[1:] / J[0]

 
def inertia_from_p(p):
    """
    converts parameters back to 3x3 intertia tensor
    """
    p1, p2 = p
    return np.diag([1.0, p1, p2])

def is_pd(I_mat, min_eig=0.1, max_cond=1e3):
    """
    physical-validity helper (used by the PD guard in the update step)

    True if I_mat is a physically valid AND well-conditioned inertia tensor
    (eigenvalues > 0 with a real margin, not just nominally positive)."""
    try:
        ev = np.linalg.eigvalsh(I_mat)
    except np.linalg.LinAlgError:
        return False
    return ev.min() > min_eig and ev.max() / ev.min() < max_cond

def Rmat(q):
    return quaternion.as_rotation_matrix(np.quaternion(*q))





#%% 
# noise matrices

# process noise matrix Q

sigma_Q_q = 1e-3
sigma_Q_omega = 1e-3
sigma_Q_p = 0.02


Q = np.eye(dim_x_err)
Q[:3,:3] *= sigma_Q_q**2
Q[3:6, 3:6] *= sigma_Q_omega**2
Q[6:,6:] *= sigma_Q_p**2



# measurement noise matrix R
sigma_R = 0.0005

omega_cam_true_sample = np.array([omega_cam_true[stride*j : stride*(j+1)+1].mean(0) for j in range(len(omega_cam_meas))])
R_omega = np.diag(np.var((omega_cam_meas - omega_cam_true_sample)[pair_valid], axis=0))

R = R_omega*5
#R = np.eye(3) * sigma_R
# convert to attitude error for filter measurement matrix
#R = np.eye(dim_z_err) * (2 * sigma_R) ** 2
#R = np.diag(var_85)


# increasing sigma_Q/sigma_R: more responsive but sensitive to measurement noise
# decreasing sigma_Q/sigma_R: more smooth

#%% 

# filter start time: end of the T_INIT window used by script 85 for the initial inertia estimate
N_INIT = int(np.load(OUTPUT/"N_INIT.npy"))   # frames in the init window (T_INIT*FPS + 1)
j0 = N_INIT - 1                              # last init frame, at t = T_INIT
k0 = j0 * stride                             # truth-rate step at t = T_INIT
jp = max(j for j in range(j0) if pair_valid[j])   # latest valid omega pair before the start




# initial global state x
x0 = np.zeros(dim_x)
q0_CP = np.quaternion(*q_meas[j0]) * q_BP                 # P -> C = (B' -> C)(P -> B')
x0[:4] = quaternion.as_float_array(q0_CP)
x0[4:7] = Rmat(x0[:4]).T @ omega_cam_meas[jp]             # first measured rate, expressed in P

# inertia parameters initialization
p_true = inertia_tensor_to_parameters(I_true)

# initial inertia ratio guess
p_guess = J_hat[1:] / J_hat[0]          # [I2/I1, I3/I1] from the saved principal moments
x0[7:] = p_guess/1.5 # extra offset for testing




#%%
# initial covariance

# initial attitude vector uncertainty
sigma_P0_q = 1e-3

# omega0 is left at 0 (unknown) while the true |omega| ~ 0.4 rad/s, so P0 must admit ~0.5 rad/s of uncertainty (std)
sigma_P0_omega = 0.5

# P0 must reflect real initial guess error
sigma_P0_p_products = 2 # off-diagonal products of inertia, typically smaller in magnitude than moments
sigma_P0_p_moments = 30 # diagonal moments of inertia

P0 = np.eye(dim_x_err)
P0[:3,:3] *= sigma_P0_q**2
P0[3:6, 3:6] *= sigma_P0_omega**2
P0[6:8,6:8] *= sigma_P0_p_moments**2     # both parameters are principal-moment ratios


"""draws = np.load(OUTPUT/"inertia_bootstrap_init.npy")
p_draws = np.array([inertia_tensor_to_parameters(D) for D in draws])
P0[6:11, 6:11] = np.diag(np.var(p_draws, axis=0))   # optionally inflate x2–3"""



P = np.full((N, dim_x_err, dim_x_err), np.nan)   # NaN before the filter starts
P[k0] = P0




#%% 
# initialize arrays

# for prediction arrays, use index k, N steps

# global state
x_hat = np.full((N, dim_x), np.nan) # NaN before the filter starts
x_hat[k0] = x0

# error state
dx_hat = np.empty((M, dim_x_err))

F = np.empty((N, dim_x_err, dim_x_err))


# measurement arrays, use index k_m, M steps

H = np.empty((M, dim_z_err, dim_x_err)) 
S = np.empty((M, dim_z_err, dim_z_err)) 
K = np.empty((M, dim_x_err, dim_z_err))


# load array of valid measurements from script 85
valid_meas = np.load(OUTPUT/"meas_valid.npy")


pd_guard_on = False

rejected_k = []
#%% 
# EKF predict/update loop

N_INIT = np.load(OUTPUT/"N_INIT.npy")
start_index = N_INIT*stride

for k in range(k0 + 1, N):
    
    # Step 1: predict

    F[k] = F_jac(x_hat[k-1])
    
    # predicted state estimate (a priori, before incorporating measurment)
    x_hat[k] = f(x_hat[k-1])    

    # predicted covariance of state estimate
    P[k] = F[k] @ P[k-1] @ F[k].T + Q
    
    # Step 2: update
    
    if k % rate_ratio == 0 and k//rate_ratio < M: # only runs when new measurement "arrives" based on rate_measure; inequality makes sure k_m doesn't exceed M
    
        k_m = k // rate_ratio # index used for the measurement arrays
        j = k_m - 1 # index for omega pairs (k_m-1 and k_m)

        if not pair_valid[j]: # skips update step if measurement is not valid
            rejected_k.append(k)
            continue  
        
        
        
        # innovation/measurement residual (error between actual sensor measurements and predicted sensor measurements)
                # the pair measures the AVERAGE rate over the interval -> compare with the predicted average

        interval_start = k-rate_ratio
        interval_end = k+1
        x_hat_interval = x_hat[interval_start: interval_end]

        w_avg = h(x_hat_interval)
        y = omega_cam_meas[j] - w_avg


        q_hat = np.quaternion(*x_hat[k, 0:4])

        

        # measurement matrix
        H[k_m] = H_jac(x_hat[k])
        
        # innovation/residual covariance
        S[k_m] = H[k_m] @ P[k] @ H[k_m].T + R

        
        
        # Kalman gain
        K[k_m] = P[k] @ H[k_m].T @ np.linalg.inv(S[k_m])

        
        # updated state estimate (a posteriori, after measurement):

        # compute correction
        dx_hat[k_m] = K[k_m] @ y      # estimated attitude error [0:3], omega error [3:6], inertia parameter error [6:8]

        
        if pd_guard_on:
            # PD guard (insurance): shrink the parameter correction until the
            # resulting inertia tensor is physically valid and well-conditioned 
            p_old, dp_full = x_hat[k, 7:], dx_hat[k_m, 6:]
            if not is_pd(inertia_from_p(p_old + dp_full)):
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if is_pd(inertia_from_p(p_old + mid * dp_full)):
                        lo = mid
                    else:
                        hi = mid
                dx_hat[k_m, 6:] = lo * dp_full
        

        alpha_hat = dx_hat[k_m, 0:3]
        
        # inject additive correction into global state
        x_hat[k, 4:] += dx_hat[k_m, 3:]

        # map estimated attitude error to quaternion
        dq_hat = np.quaternion(1.0, *(alpha_hat/2)).normalized()

        # update attitude MULTIPLICATIVELY
        q_hat = q_hat * dq_hat 
        x_hat[k,0:4] = np.array([q_hat.w, q_hat.x, q_hat.y, q_hat.z])

        # normalize q
        x_hat[k,:4] /= np.linalg.norm(x_hat[k,:4])
        
        # updated covariance of global state estimate (using Joseph form)
        term1 = (np.eye(dim_x_err) - K[k_m]@H[k_m])
        term2 = K[k_m] @ R @ K[k_m].T
        P[k] = term1 @ P[k] @ term1.T + term2

        

        

# filter states are in P; convert to B' so they compare directly with the B' ground truth
q_BP_inv = q_BP.inverse()
q_estimates = np.array([quaternion.as_float_array(np.quaternion(*x[:4]) * q_BP_inv) for x in x_hat])  # B' -> C
omega_body_estimates = x_hat[:, 4:7] @ V.T          # omega_B' = V @ omega_P
p_estimates = x_hat[:, 7:]

# covariance blocks rotated from P into B' (attitude error and omega are both body-frame vectors)
P_att_B = V @ P[:, 0:3, 0:3] @ V.T
P_w_B   = V @ P[:, 3:6, 3:6] @ V.T

#%% 
# attitude and inertia error

angle_errors = np.empty(N)
q_errors = np.array(q_estimates-q_true)

attitude_errors = dx_hat[:, 0:3]
attitude_errors_mag = np.linalg.norm(attitude_errors, axis=1)


# ang velocity error (body frame)
omega_body_true = np.empty_like(omega_cam_true)
for i in range(omega_cam_true.shape[0]):
    q = 1/np.quaternion(*q_true[i])
    q_cam_to_body = np.array([q.w,q.x,q.y,q.z])
    omega_body_true[i] = np.array([rotate_to_frame(q_cam_to_body, omega_cam_true[i])])

omega_body_errors = omega_body_estimates - omega_body_true
omega_body_errors_mag = np.linalg.norm(omega_body_errors, axis=1)

# quaternion error, formula from [19]
for k in range(N):
    dq = 1/np.quaternion(*q_true[k]) * np.quaternion(*q_estimates[k]) # body frame convention
    angle_errors[k] = np.rad2deg(2*np.arccos(dq.w)) # converts q scalar to degrees



# error of each q component
RMSE_q = np.empty(4)
print()
for i in range(4):
    RMSE_q[i] = np.sqrt(np.sum(q_errors[k0:,i]**2)/N)
    print(f'RMSE q{i}: {RMSE_q[i]:.4f}')
    RMSE_deg  = np.sqrt(np.nanmean(angle_errors[k0:]**2))

#print(f'Max angle error: {np.nanmax(angle_errors):.2f} deg')

# final inertia parameter error
p_errors = (p_true - p_estimates[-1])/p_true

print(f'\nRMSE δq: {RMSE_deg:.2f} deg')
print(f'Max angle error: {np.max(angle_errors):.2f} deg')
print(f'Final angle error: {angle_errors[-1]:.2f} deg')
#print(f"\nFinal inertia parameter error: {p_errors}")

def J_from_ratios(r):  # r = [I2/I1, I3/I1]
    return np.diag([1.0, r[0], r[1]])

err_pct = 100 * np.linalg.norm(J_from_ratios(p_estimates[-1]) - J_from_ratios(p_true)) / np.linalg.norm(J_from_ratios(p_true))

print(f"\nFinal normalized inertia parameter error: {err_pct:.5f} %")
print(f"Ground truth Interia ratios: {p_true}")
print(f" Estimated Inertia ratios: {p_estimates[-1]}")


#%%  
# plot 1: q components and angle error
fig1, axes1 = plt.subplots(5, 1, figsize=(12, 6), sharex=True)

color_err = "r"
 
# q plots
q0_true = q_true[:,0]
q1_true = q_true[:,1]
q2_true = q_true[:,2]
q3_true = q_true[:,3]
 
labels = ["$q_0$", "$q_1$", "$q_2$", "$q_3$"]
true_arrays = [q0_true, q1_true, q2_true, q3_true]
 
# measurement sample times (sample k_m arrives at truth step k = k_m*rate_ratio)
t_meas = np.arange(M) * rate_ratio * dt
meas_mask = t_meas < tstop # only plot samples inside the simulated time span

# compute q uncertainty bounds from P matrix
sigma_q = np.empty((N, 4))
for k in range(N):
    G = 0.5 * W_body(q_estimates[k]).T                    # 4x3
    sigma_q[k] = np.sqrt(np.diag(G @ P_att_B[k] @ G.T))


for idx, (ax, lbl, truth) in enumerate(zip(axes1[:4], labels, true_arrays)):
    ax.plot(t, truth,"k--",  lw=1, label="$q_t$")
    ax.plot(t, q_estimates[:, idx], "b-", lw=1,   label=r"$\hat{q}$")
    ax.set_ylabel(lbl)

    # uncertainty bounds sigma comes from P diagonal
    q_lower = q_estimates[:, idx] - 3*sigma_q[:, idx]
    q_upper = q_estimates[:, idx] + 3*sigma_q[:, idx]

    ax.fill_between(t, q_lower, q_upper, color="b", alpha=0.15, label=r"$\pm3\sigma$")


    ax.grid(True)

axes1[0].legend()

# attitude error plot
axes1[4].plot(t, angle_errors, color = color_err, lw=1)
axes1[4].set_ylabel(r"$\delta q$ (deg)")
axes1[4].grid(True)

# show when weak measurements are rejected
rejected_k = np.array(rejected_k, dtype=int)
t_rej = t[rejected_k]
axes1[4].plot(t_rej, angle_errors[rejected_k], "kx", ms=5, label = "rejected measurement if inlier count < 50")
axes1[4].legend()


# eclipse marking from script 85 brightness (frame j is at t = j*stride*dt)
bright_rel = np.load(OUTPUT/"frame_brightness_rel.npy")
t_frame = np.arange(bright_rel.size) * stride * dt
PENUMBRA_THR, UMBRA_THR = 0.7, 0.3        # tune from the 85 brightness plot

def _spans(flag):
    """(t_start, t_end) of contiguous True runs in a per-frame flag"""
    e = np.diff(np.r_[0, flag.astype(int), 0])
    return [(t_frame[s], t_frame[min(f, flag.size-1)]) for s, f in zip(np.where(e == 1)[0], np.where(e == -1)[0])]

def mark_eclipse(axes, show_rejected=False):
    pen = _spans((bright_rel < PENUMBRA_THR) & (bright_rel >= UMBRA_THR))
    umb = _spans(bright_rel < UMBRA_THR)

    color = "k"
    pen_alpha = 0.1
    umb_alpha = 0.2

    for ax in axes:
        for a, b in pen: ax.axvspan(a, b, color=color, alpha=pen_alpha, lw=0, zorder=0)
        for a, b in umb: ax.axvspan(a, b, color=color,    alpha=umb_alpha, lw=0, zorder=0)
        if show_rejected:
            for tr in t_rej: ax.axvline(tr, color="r", alpha=0.35, lw=1, zorder=0)
    if pen: axes[0].axvspan(np.nan, np.nan, color=color, alpha=pen_alpha, label="penumbra")
    if umb: axes[0].axvspan(np.nan, np.nan, color=color,    alpha=umb_alpha, label="umbra")
    if show_rejected and len(t_rej):
        axes[0].plot([], [], color="r", lw=1, label=f"rejected ({len(t_rej)})")
    axes[0].legend()


mark_eclipse(axes1)


fig1.supxlabel("Time (s)")
 
fig1.suptitle(f"MEKF Attitude (unit quaternion)", fontsize=13)
plt.tight_layout()
plt.show()





#%%
# plot 2: ang velocity
fig2, axes2 = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
 
color = "g"
omega1_true = omega_body_true[:,0]
omega2_true = omega_body_true[:,1]
omega3_true = omega_body_true[:,2]

 
labels = [r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"]
true_arrays = [omega1_true, omega2_true, omega3_true]
 

for idx, (ax, lbl, truth) in enumerate(zip(axes2[:3], labels, true_arrays)):
    ax.plot(t, truth,"k--",  lw=1, label=r"$\omega_t$")
    ax.plot(t, omega_body_estimates[:, idx], color=color, linestyle= "-", lw=1,   label=r"$\hat{\omega}$")
    ax.set_ylabel(lbl)

    # uncertainty bounds sigma comes from P diagonal
    sigma_w = np.sqrt(P_w_B[:, idx, idx])
    
    w_lower = omega_body_estimates[:, idx] - sigma_w
    w_upper = omega_body_estimates[:, idx] + sigma_w

    ax.fill_between(t, w_lower, w_upper, color=color, alpha=0.15, label=r"$\pm1\sigma$")


    ax.grid(True)

axes2[0].legend()



# omega error plot
axes2[-1].plot(t, omega_body_errors_mag, color = color_err, lw=1)
axes2[-1].grid(True)

axes2[-1].plot(t_rej, omega_body_errors_mag[rejected_k], "kx", ms=5, label = "rejected measurement if inlier count < 50")
axes2[-1].set_ylabel(r"$|\delta\omega|$")
axes2[-1].legend()

# mark eclipse
mark_eclipse(axes2)



fig2.supxlabel("Time (s)")
fig2.supylabel(r"$\omega$ (rad/s)")

fig2.suptitle(f"MEKF Angular Velocity, Measurement rate = {rate_measure} Hz", fontsize=13)
plt.tight_layout()
plt.show()


#%% 
# plot 3: inertia parameter estimates
colors = ["c", "m", "g", "orange", "r"]  # one color per parameter, extend if needed

fig3, axes3 = plt.subplots(dim_p, 1, figsize=(12, 6), sharex=True)
inertia_labels = [r"$I_2 / I_1$", r"$I_3 / I_1$"]

for i in range(dim_p):
    color = colors[i % len(colors)]

    # true value + estimate
    axes3[i].axhline(p_true[i], color=color, linestyle="--", lw=1, label="truth")
    axes3[i].plot(t, p_estimates[:, i], color=color, lw=1, label="estimate")

    # uncertainty bounds sigma comes from P diagonal
    sigma_p = np.sqrt(P[:, dim_rot_err + i, dim_rot_err + i])
    p_lower = p_estimates[:, i] - sigma_p
    p_upper = p_estimates[:, i] + sigma_p

    axes3[i].fill_between(t, p_lower, p_upper, color=color, alpha=0.15, label=r"$\pm1\sigma$")

    axes3[i].set_ylabel(inertia_labels[i])
    axes3[i].legend(loc="right")
    axes3[i].grid(True)


# mark eclipse
mark_eclipse(axes3)

plt.xlim(0, tstop)
fig3.supxlabel("Time (s)")
fig3.suptitle("MEKF Principal Inertia Ratios", fontsize=13)
plt.tight_layout()
plt.show()

# %%

