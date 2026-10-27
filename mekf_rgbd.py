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


GLobal state: x = [q, omega, p]
parameters: p = [p1 = Iyy/Ixx, 
                p2 = Izz/Ixx, 
                p3 = Ixy/Ixx,
                p4 = Ixz/Ixx,
                p5 = Iyz/Ixx,]
                where Ixx = 1

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
DATASET_ideal = Path(r"C:\Users\Tevin\OneDrive - Rensselaer Polytechnic Institute\URP\scripts\rgbd_inertia_mekf_pipeline\script81_lean")

# entering eclipse at frame 800, skips every x frames --> 10/x = fps
DATASET_penumbra_ingress5 = Path(r"C:\Blender_Proc\strided\script102_block8_penumbra_ingress_s0_e999_x5")
DATASET_penumbra_ingress10 = Path(r"C:\Blender_Proc\strided\script102_block8_penumbra_ingress_s0_e999_x10")
DATASET_penumbra_ingress2 = Path(r"C:\Blender_Proc\strided\script102_block8_penumbra_ingress_s0_e999_x2")

# choose ratio of predict steps to measure/update steps
stride = 5
if stride == 5:
    DATASET = DATASET_penumbra_ingress5
if stride == 10:
    DATASET = DATASET_penumbra_ingress10
if stride == 2:
    DATASET = DATASET_penumbra_ingress2
    

OUTPUT = DATASET / "notebook85_end_to_end"
GT_DIR = DATASET / "ground_truth"


# ground truth files are in B frame (body-fixed, axes not aligned with camera at t0), convert to B' (body-fixed, aligned with camera at t0)
I_true = np.load(GT_DIR/"I_true.npy")
R0 = np.load(GT_DIR/"R0_body_to_cam.npy")
I_true = R0 @ I_true @ R0.T
q_meas = np.load(OUTPUT/"q_est_frozen_wxyz.npy")


if DATASET == DATASET_ideal:
    I_hat = np.load(OUTPUT/"normalized_inertia_estimate.npy")
    q_true = np.load(OUTPUT/"q_gt_wxyz.npy").T
else:
    I_hat = I_true
    q_true = np.load(GT_DIR/"q_gt_cam0_wxyz.npy") # wxyz
    omega_cam_true = np.load(GT_DIR/"omega_cam0_rad_s.npy")
    omega_cam_meas = np.load(OUTPUT/"omega_cam_pair_rad_s.npy")
    pair_valid  = np.load(OUTPUT/"pair_valid.npy")


n_truth_frames = 1000
rate_truth = 10 # Hz, from script 85
dt = 1 / rate_truth
tstop = n_truth_frames / rate_truth 
t = np.arange(0, tstop, dt)
N = t.size

# global state
dim_rot = 7 # 4 q + 3 omega components
dim_p = 5 # 5 inertia parameters
dim_x = dim_rot + dim_p # 12

# error state
dim_rot_err = 6 # 3 attitude error + 3 omega error
dim_x_err = dim_rot_err + dim_p # 11

# measurements/observations
dim_z = 4 # 4 q components
dim_z_err = 3 # 3 attitude error components


# set up sampling rate ratios
rate_measure = int(rate_truth/stride) # Hz
rate_ratio = stride


"""q_meas_sample = q_meas[::rate_ratio] # creates sample array of q_meas, simulates lower measurement rates available in spacecraft
M = q_meas_sample.shape[0]"""

q_meas_sample = q_meas
M = q_meas.shape[0]


"""
error = q_meas - q_true
error_mag = np.linalg.norm(error)

plt.plot(t,error[:,1])
plt.show()
"""

#%% 

# define functions

# measurement function
def h(x_interval):
    """
    inputs current estimates of q (B' -> C) and omega (B') across interval (of most recent omega values up to last measurement update)
    converts q to Rotation matrix
    multiplies by omega to rotate into C frame
    returns average omega across interval
    """
    v = np.array([Rmat(x[:4]) @ x[4:7] for x in x_interval])          # inertial-frame omega at each step
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
    outputs shape (dim_error+dim_p x dim_error+dim_p) = 11 x 11
    uses BODY-FRAME Euler equations, thus quaternion error equations used in predict/update must also be in body frame
    """
    q = x_hat_old[:4]
    omega = x_hat_old[4:7]
    p = x_hat_old[7:12]
    
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
    
    # d(omega_dot)/dp  -> 3x5
    J_p = np.zeros((3, 5))
    for i in range(5):
        d_p = p.copy()
        d_p[i] += eps
        J_p[:, i] = (calc_omega_dot(omega, d_p) - base) / eps
    
    F[3:6, 6:11] = dt * J_p
    
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
    
    p1, p2, p3, p4, p5 = p
    I = np.array([
        [1, -p3, -p4],
        [-p3, p1, -p5],
        [-p4, -p5, p2]
        ])
    
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
    converts 3x3 inertia tensor np array into 1D array with 5 inertia ratios
    """
    Ixx, Iyy, Izz = np.diag(I)
    Ixy, Ixz, Iyz = -I[0,1], -I[0,2], -I[1,2]

    return np.array([
        Iyy/Ixx,
        Izz/Ixx,
        Ixy/Ixx,
        Ixz/Ixx,
        Iyz/Ixx
        ])

 
def inertia_from_p(p):
    """
    converts parameters back to 3x3 intertia tensor
    """
    p1, p2, p3, p4, p5 = p
    return np.array([[1, -p3, -p4], 
                     [-p3, p1, -p5], 
                     [-p4, -p5, p2]])

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
sigma_Q_p = 1e-4


Q = np.eye(dim_x_err)
Q[:3,:3] *= sigma_Q_q**2
Q[3:6, 3:6] *= sigma_Q_omega**2
Q[6:,6:] *= sigma_Q_p**2



# measurement noise matrix R
sigma_R = 0.07

omega_cam_true_sample = np.array([omega_cam_true[stride*j : stride*(j+1)+1].mean(0) for j in range(len(omega_cam_meas))])
R_omega = np.diag(np.var((omega_cam_meas - omega_cam_true_sample)[pair_valid], axis=0))

R = R_omega

# convert to attitude error for filter measurement matrix
#R = np.eye(dim_z_err) * (2 * sigma_R) ** 2
#R = np.diag(var_85)


# increasing sigma_Q/sigma_R: more responsive but sensitive to measurement noise
# decreasing sigma_Q/sigma_R: more smooth

#%% 
# initial global state x
x0 = np.zeros(dim_x)
x0[:4] = q_meas[0]
x0[4:7] = omega_cam_meas[0]

# inertia parameters initialization
p_true = inertia_tensor_to_parameters(I_true)

# initial inertia ratio guess
#p_guess = p_true + 0.05
p_guess = inertia_tensor_to_parameters(I_hat) + 0.1
x0[7:] = p_guess


"""p_hat = inertia_tensor_to_parameters(I_hat)
p_true = inertia_tensor_to_parameters(I_true)

I_error =  p_hat - p_true
print(p_hat, p_true, I_error)"""

#%%
# initial covariance

# initial attitude vector uncertainty
sigma_P0_q = 1e-3

# omega0 is left at 0 (unknown) while the true |omega| ~ 0.4 rad/s, so P0 must admit ~0.5 rad/s of uncertainty (std)
sigma_P0_omega = 0.5

# P0 must reflect real initial guess error
sigma_P0_p = 0.5

P0 = np.eye(dim_x_err)
P0[:3,:3] *= sigma_P0_q**2
P0[3:6, 3:6] *= sigma_P0_omega**2
P0[6:,6:] *= sigma_P0_p**2

P = np.empty((N, dim_x_err, dim_x_err))
P[0] = P0



#%% 
# initialize arrays

# for prediction arrays, use index k, N steps

# global state
x_hat = np.empty((N, dim_x))
x_hat[0] = x0

# error state
dx_hat = np.empty((M, dim_x_err))

F = np.empty((N, dim_x_err, dim_x_err))


# measurement arrays, use index k_m, M steps
z = q_meas_sample

H = np.empty((M, dim_z_err, dim_x_err)) 
S = np.empty((M, dim_z_err, dim_z_err)) 
K = np.empty((M, dim_x_err, dim_z_err))


# load array of valid measurements from script 85
valid_meas = np.load(OUTPUT/"meas_valid.npy")




#%% 
# EKF predict/update loop

for k in range(1,N):
    
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
        dx_hat[k_m] = K[k_m] @ y      # estimated attitude error [0:3], omega error [3:6], inertia parameter error [6:11]

        
        
        # PD guard (insurance): shrink the parameter correction until the
        # resulting inertia tensor is physically valid and well-conditioned 
        p_old, dp_full = x_hat[k, 7:12], dx_hat[k_m, 6:11]
        if not is_pd(inertia_from_p(p_old + dp_full)):
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if is_pd(inertia_from_p(p_old + mid * dp_full)):
                    lo = mid
                else:
                    hi = mid
            dx_hat[k_m, 6:11] = lo * dp_full
        

        alpha_hat = dx_hat[k_m, 0:3]
        
        # inject additive correction into global state
        x_hat[k, 4:12] += dx_hat[k_m, 3:11]

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


        

q_estimates = x_hat[:, :4]
p_estimates = x_hat[:, 7:]

#%% 
# angle and inertia error

angle_errors = np.empty(N)
q_errors = np.array(q_estimates-q_true)


# quaternion error, formula from [19]
for k in range(N):
    e = 1/np.quaternion(*q_true[k]) * np.quaternion(*q_estimates[k]) # body frame convention
    angle_errors[k] = np.rad2deg(2*np.arccos(np.clip(abs(e.w), 0, 1.0))) # converts q scalar to degrees

# error of each q component
RMSE_q = np.empty(4)
print()
for i in range(4):
    RMSE_q[i] = np.sqrt(np.sum(q_errors[:,i]**2)/N)
    print(f'RMSE q{i}: {RMSE_q[i]:.4f}')

RMSE_deg = np.sqrt(np.sum(angle_errors**2)/N)

# final inertia parameter error
p_errors = (p_true - p_estimates[-1])/p_true

print(f'\nRMSE δq: {RMSE_deg:.2f} deg')
print(f'Max angle error: {np.max(angle_errors):.2f} deg')
print(f'Final angle error: {angle_errors[-1]:.2f} deg')
print(f"\nFinal inertia parameter error: {p_errors}")

def J_from_ratios(r):  # r = [Jyy, Jzz, Jxy, Jxz, Jyz]
    Jyy, Jzz, Jxy, Jxz, Jyz = r
    return np.array([[1, Jxy, Jxz], [Jxy, Jyy, Jyz], [Jxz, Jyz, Jzz]])

err_pct = 100 * np.linalg.norm(J_from_ratios(p_estimates[-1]) - J_from_ratios(p_true)) / np.linalg.norm(J_from_ratios(p_true))

print(f"Final normalized inertia parameter error: {err_pct:.5f} %")

#%% 
# plotting
 
# plot 1: q components and angle error
fig1, axes1 = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
 
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
 
for idx, (ax, lbl, truth) in enumerate(zip(axes1[:4], labels, true_arrays)):
    ax.plot(t, truth,"k--",  lw=1, label="$q_t$")
    ax.plot(t, q_estimates[:, idx], "b-", lw=1,   label=r"$\hat{q}$ (10 Hz)")
    ax.plot(t_meas[meas_mask], q_meas_sample[meas_mask, idx], "x", color="tab:orange",
            ms=6, mfc="none", label=f"z ({rate_measure} Hz)")
    ax.set_ylabel(lbl)
    ax.grid(True)
 
axes1[0].legend()
 
# error plot
axes1[4].plot(t, angle_errors, "r-", lw=1)
axes1[4].set_ylabel("Angle error (deg)")
axes1[4].grid(True)
 
axes1[-1].set_xlabel("Time (s)")
 
fig1.suptitle(f"Quaternion-based Attitude MEKF, stride = {stride}", fontsize=13)
plt.tight_layout()
plt.show()

# plot 2: inertia parameter estimates
colors = ["c", "m", "g", "orange", "r"]  # one color per parameter, extend if needed

fig2, axes2 = plt.subplots(dim_p, 1, figsize=(12, 10), sharex=True)

for i in range(dim_p):
    color = colors[i % len(colors)]

    # true value + estimate
    axes2[i].axhline(p_true[i], color=color, linestyle="--", lw=1, label="truth")
    axes2[i].plot(t, p_estimates[:, i], color=color, lw=1, label="guess")

    # uncertainty bounds sigma comes from P diagonal
    sigma_p = np.sqrt(P[:, dim_rot_err + i, dim_rot_err + i])
    p_lower = p_estimates[:, i] - sigma_p
    p_upper = p_estimates[:, i] + sigma_p

    axes2[i].fill_between(t, p_lower, p_upper, color=color, alpha=0.15, label=r"$\pm1\sigma$")

    axes2[i].set_ylabel(f"$p_{i+1}$")
    axes2[i].legend()
    axes2[i].grid(True)



axes2[-1].set_xlabel("Time (s)")

fig2.suptitle("Inertia Ratio Parameter Estimation", fontsize=13)
plt.tight_layout()
plt.show()

# %%

