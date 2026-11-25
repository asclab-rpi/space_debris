"""Notebook helpers for Script 48/74/76 attitude and inertia comparison.

No function in this module reads ground truth. Evaluation remains in the
notebook after all three estimator trajectories have been frozen.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

pf=importlib.import_module("pose_frontend_script48"); s48=pf._S48

def valid_depth(d): return np.isfinite(d)&(d>0)&(d<1e6)

def roi_pair(im0,im1,d0,d1,m0,m1,K,scale=3.,padding=24):
    u=(m0>0)|(m1>0); y,x=np.where(u)
    x0=max(0,int(x.min())-padding);x1=min(d0.shape[1],int(x.max())+padding+1)
    y0=max(0,int(y.min())-padding);y1=min(d0.shape[0],int(y.max())+padding+1)
    size=(max(2,int((x1-x0)*scale)),max(2,int((y1-y0)*scale)))
    ri0=cv2.resize(im0[y0:y1,x0:x1],size,interpolation=cv2.INTER_CUBIC)
    ri1=cv2.resize(im1[y0:y1,x0:x1],size,interpolation=cv2.INTER_CUBIC)
    rd0=cv2.resize(d0[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)
    rd1=cv2.resize(d1[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)
    rm0=cv2.resize(m0[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)
    rm1=cv2.resize(m1[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)
    Kr=K.copy();Kr[0,0]*=scale;Kr[1,1]*=scale
    Kr[0,2]=(K[0,2]-x0)*scale;Kr[1,2]=(K[1,2]-y0)*scale
    return ri0,ri1,rd0,rd1,rm0,rm1,Kr,(x0,y0,x1,y1),scale

def estimate_with_points(img0,img1,d0,d1,m0,m1,K,sift):
    """Exact Script-48 estimate_direction path plus accepted pixel arrays."""
    out={"success":False,"R":None,"t":None,"n_inliers_3d":0,"mean_residual_m":None,
         "mean_lk_fb_err_px":None,"points0":np.empty((0,2)),"points1":np.empty((0,2)),
         "residuals_m":np.empty(0)}
    mask0=s48.erode_mask(m0);mask1=s48.erode_mask(m1)
    dist0=s48.boundary_distance(m0);dist1=s48.boundary_distance(m1)
    g0=s48.preprocess_gray(img0);g1=s48.preprocess_gray(img1)
    kp0,de0=sift.detectAndCompute(g0,mask0);kp1,de1=sift.detectAndCompute(g1,mask1)
    kp0=[] if kp0 is None else kp0;kp1=[] if kp1 is None else kp1
    matches,ratios,raw=s48.match_descriptors(de0,de1)
    if len(matches)<s48.MIN_2D_MATCHES:return out
    p0=np.array([kp0[m.queryIdx].pt for m in matches],np.float32);p1=np.array([kp1[m.trainIdx].pt for m in matches],np.float32)
    keep=s48.estimate_2d_ransac_mask(p0,p1)
    if keep.sum()<s48.MIN_3D_MATCHES:return out
    p0,p1,ratios,raw=p0[keep],p1[keep],ratios[keep],raw[keep]
    p1lk,keep,fb=s48.lk_refine_points(g0,g1,p0,p1)
    if keep.sum()<s48.MIN_3D_MATCHES:return out
    p0,p1,ratios,raw,fb=p0[keep],p1lk[keep],ratios[keep],raw[keep],fb[keep]
    bd=np.minimum(s48.sample_boundary_distance(dist0,p0),s48.sample_boundary_distance(dist1,p1));keep=bd>=s48.MIN_BOUNDARY_DIST_PX
    if keep.sum()<s48.MIN_3D_MATCHES:return out
    p0,p1,ratios,raw,fb,bd=p0[keep],p1[keep],ratios[keep],raw[keep],fb[keep],bd[keep]
    z0,v0=s48.sample_depth_nearest(d0,p0);z1,v1=s48.sample_depth_nearest(d1,p1);keep=v0&v1
    if keep.sum()<s48.MIN_3D_MATCHES:return out
    p0,p1,z0,z1,ratios,raw,fb,bd=p0[keep],p1[keep],z0[keep],z1[keep],ratios[keep],raw[keep],fb[keep],bd[keep]
    P0=s48.backproject(K,p0,z0);P1=s48.backproject(K,p1,z1);weights=s48.quality_weights(raw,ratios,bd,fb)
    R,t,in3d,res=s48.ransac_rigid_3d(P0,P1,thresh=s48.RANSAC_THRESH_M,iters=s48.RANSAC_ITERS,weights=weights)
    if R is None or in3d.sum()<s48.MIN_3D_MATCHES:return out
    residuals=np.linalg.norm((R@P0.T).T+t-P1,axis=1)
    out.update(success=True,R=pf.reorthonormalize(R),t=t,n_inliers_3d=int(in3d.sum()),
               mean_residual_m=float(res),mean_lk_fb_err_px=float(np.mean(fb[in3d])),
               points0=p0[in3d],points1=p1[in3d],residuals_m=residuals[in3d])
    return out

def load_inputs(root:Path):
    rgb=sorted((root/'rgb').glob('rgb_*.png'));dep=sorted((root/'depth').glob('depth_*.npy'));masks=sorted((root/'masks').glob('mask_*.png'))
    n=min(len(rgb),len(dep),len(masks));return rgb[:n],dep[:n],masks[:n],np.load(root/'K.npy')

def run_adjacent(root:Path,mode='script48',scale=3.,padding=24):
    rgb,dep,masks,K=load_inputs(root);det=pf.make_sift();Rabs=np.eye(3);Rs=[Rabs.copy()];rel=[];records=[];points=[]
    for k in range(len(rgb)-1):
        im0,im1=pf.load_rgb(rgb[k]),pf.load_rgb(rgb[k+1]);d0,d1=pf.load_depth(dep[k]),pf.load_depth(dep[k+1])
        m0=cv2.imread(str(masks[k]),0);m1=cv2.imread(str(masks[k+1]),0);bounds=(0,0,d0.shape[1],d0.shape[0]);used_scale=1.
        args=(im0,im1,d0,d1,m0,m1,K)
        if mode=='script74_roi':
            *args,bounds,used_scale=roi_pair(*args,scale=scale,padding=padding)
        e=estimate_with_points(*args,det);R=e['R'] if e['success'] else np.eye(3);Rabs=pf.reorthonormalize(R@Rabs)
        rel.append(R);Rs.append(Rabs.copy());p0=e['points0'].copy();p1=e['points1'].copy()
        if mode=='script74_roi' and len(p0):
            p0[:,0]=p0[:,0]/used_scale+bounds[0];p0[:,1]=p0[:,1]/used_scale+bounds[1]
            p1[:,0]=p1[:,0]/used_scale+bounds[0];p1[:,1]=p1[:,1]/used_scale+bounds[1]
        points.append((p0,p1,e['residuals_m']))
        records.append(dict(frame=k+1,success=e['success'],inliers=e['n_inliers_3d'],rms_m=e['mean_residual_m'],lk_fb_px=e['mean_lk_fb_err_px']))
        print(f"{mode} {k:03d}->{k+1:03d}: {e['n_inliers_3d']} inliers",flush=True)
    return dict(name=mode,R=np.asarray(Rs),relative_R=np.asarray(rel),records=records,inlier_points=points)

def _unpack(x,n):return np.concatenate((np.eye(3)[None],Rotation.from_rotvec(x.reshape(n-1,3)).as_matrix()))
def _residuals(x,n,adj,keyframes,sa,sk,ss):
    R=_unpack(x,n);blocks=[]
    for i,M in enumerate(adj):blocks.append(Rotation.from_matrix(M@(R[i+1]@R[i].T).T).as_rotvec()/sa)
    for a,b,M,q in keyframes:blocks.append(np.sqrt(q)*Rotation.from_matrix(M@(R[b]@R[a].T).T).as_rotvec()/sk)
    for i in range(1,n-1):
        d0=R[i]@R[i-1].T;d1=R[i+1]@R[i].T;blocks.append(Rotation.from_matrix(d1@d0.T).as_rotvec()/ss)
    return np.concatenate(blocks)
def _sparsity(n,kfs):
    rows=3*((n-1)+len(kfs)+(n-2));S=lil_matrix((rows,3*(n-1)),dtype=int);r=0
    def mark(frame):
        if frame>0:S[r:r+3,3*(frame-1):3*frame]=1
    for i in range(n-1):mark(i);mark(i+1);r+=3
    for a,b,_,_ in kfs:mark(a);mark(b);r+=3
    for i in range(1,n-1):mark(i-1);mark(i);mark(i+1);r+=3
    return S.tocsr()

def run_pose_graph(root:Path,roi_result,spacing=10,scale=3.,padding=24,adj_sigma_deg=.08,key_sigma_deg=.20,smooth_sigma_deg=.04,max_nfev=100):
    rgb,dep,masks,K=load_inputs(root);det=pf.make_sift();adj=roi_result['relative_R'];R0=roi_result['R'];kfs=[];rows=[]
    for source in range(0,len(rgb)-spacing,spacing):
        target=source+spacing;im0,im1=pf.load_rgb(rgb[source]),pf.load_rgb(rgb[target]);d0,d1=pf.load_depth(dep[source]),pf.load_depth(dep[target]);m0=cv2.imread(str(masks[source]),0);m1=cv2.imread(str(masks[target]),0)
        args=roi_pair(im0,im1,d0,d1,m0,m1,K,scale,padding);e=estimate_with_points(*args[:7],det);accepted=False;dis=np.nan;q=0.
        if e['success']:
            comp=np.eye(3)
            for k in range(source,target):comp=adj[k]@comp
            dis=float(np.degrees(Rotation.from_matrix(e['R']@comp.T).magnitude()));q=float(np.clip((e['n_inliers_3d']/40)*np.exp(-.5*(e['mean_residual_m']/.002)**2),.20,1.))
            accepted=bool(e['n_inliers_3d']>=18 and e['mean_residual_m']<=.002 and e['mean_lk_fb_err_px']<=.8 and dis<=1.)
            if accepted:kfs.append((source,target,e['R'],q))
        rows.append(dict(source=source,target=target,inliers=e['n_inliers_3d'],disagreement_deg=dis,accepted=accepted))
        print(f"keyframe {source:03d}->{target:03d}: {'accepted' if accepted else 'rejected'}")
    x0=Rotation.from_matrix(R0[1:]).as_rotvec().ravel();fun=lambda x:_residuals(x,len(R0),adj,kfs,np.deg2rad(adj_sigma_deg),np.deg2rad(key_sigma_deg),np.deg2rad(smooth_sigma_deg))
    fit=least_squares(fun,x0,jac_sparsity=_sparsity(len(R0),kfs),loss='soft_l1',f_scale=1.,max_nfev=max_nfev)
    return dict(name='script76_pose_graph',R=_unpack(fit.x,len(R0)),relative_R=adj,records=roi_result['records'],inlier_points=roi_result['inlier_points'],keyframes=rows,optimizer=fit)

def continuous_quat(R):
    q=Rotation.from_matrix(R).as_quat()
    for k in range(1,len(q)):
        if np.dot(q[k],q[k-1])<0:q[k]*=-1
    return q

def estimate_dynamics(R,dt):
    dR=np.asarray([R[k].T@R[k+1] for k in range(len(R)-1)]);raw=Rotation.from_matrix(dR).as_rotvec()/dt
    win=min(11,len(raw) if len(raw)%2 else len(raw)-1);omega=savgol_filter(raw,win,min(2,win-1),axis=0,mode='interp') if win>=3 else raw
    alpha=np.gradient(omega,dt,axis=0);B=[]
    for j in range(6):
        E=np.zeros((3,3))
        if j<3:E[j,j]=1
        else:a,b=((0,1),(0,2),(1,2))[j-3];E[a,b]=E[b,a]=1
        B.append(E)
    A=np.vstack([np.column_stack([E@a+np.cross(w,E@w) for E in B]) for w,a in zip(omega,alpha)])
    _,s,Vt=np.linalg.svd(A,full_matrices=False);th=Vt[-1];I=sum(c*E for c,E in zip(th,B));I*=1 if np.trace(I)>0 else -1;I/=np.trace(I)
    vals,vec=np.linalg.eigh(I);Ispd=vec@np.diag(np.maximum(vals,1e-6))@vec.T;Ispd/=np.trace(Ispd)
    return dict(omega=omega,alpha=alpha,I=Ispd,singular_values=s,nullspace_ratio=float(s[-2]/max(s[-1],1e-15)))

def attitude_error_deg(Rest,Rgt):return np.degrees(Rotation.from_matrix(np.asarray([a@b.T for a,b in zip(Rest,Rgt)])).magnitude())

def predict_torque_free(R0,w0,I,steps,dt,substeps=5):
    R,w=R0.copy(),w0.copy();Rs=[R.copy()];ws=[w.copy()];h=dt/substeps
    for _ in range(steps):
        for __ in range(substeps):
            wd=np.linalg.solve(I,-np.cross(w,I@w));wm=w+.5*h*wd
            R=pf.reorthonormalize(R@Rotation.from_rotvec(wm*h).as_matrix());w=w+h*wd
        Rs.append(R.copy());ws.append(w.copy())
    return np.asarray(Rs),np.asarray(ws)
