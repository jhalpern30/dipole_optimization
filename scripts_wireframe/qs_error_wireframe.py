#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume, SurfaceRZFourier
from simsopt._core import load
from simsopt.mhd.vmec import Vmec 
from create_surface import *
from simsopt.field.wireframefield import WireframeField, enclosed_current
from simsopt.geo.wireframe import ToroidalWireframe, windowpane_wireframe

eq_name = 'wout_NAS_n2n4_AR6.2.03'  # this needs to be set based on the run
out_dir = 'Bt1.0_BdFalse_ntf4_np8_nt6_VVa_0.27'
script_dir = os.getcwd()
eq_dir = os.path.join(script_dir, 'equilibria')
eq_name_full = os.path.join(eq_dir, eq_name + '.nc')
full_dir = os.path.join(script_dir, '../outputs/nInboard_scan_10_31_24', eq_name, out_dir)

dof_scale = 0.15
R0 = 1
surf_s = 1

nphi = 64
ntheta = 64
nfp = 2
nmax = 20
mmax = 20

vmec = Vmec(eq_name_full)
iota_ax = vmec.iota_axis()
iota_edge = vmec.iota_edge()
print("iota axis, iota edge from VMEC")
print(iota_ax)
print(iota_edge)

phis = np.linspace(0,1/nfp,nphi,endpoint=False)
thetas = np.linspace(0,1,nphi,endpoint=False)

surf_nfp1 = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, quadpoints_phi=phis,quadpoints_theta=thetas)
surf_prev = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=2,stellsym=True,
                                quadpoints_theta=surf_nfp1.quadpoints_theta,
                                quadpoints_phi=surf_nfp1.quadpoints_phi)
surf_prev.least_squares_fit(surf_nfp1.gamma())
surf_prev.set_dofs(dof_scale*surf_prev.get_dofs())
surf_prev.set_rc(0,0,R0)

vol_target = surf_prev.volume()

import booz_xform as bx

mboz = 48 # number of poloidal harmonics for Boozer transformation
nboz = 48 # number of toroidal harmonics for Boozer transformation

b = bx.Booz_xform()
b.read_wout(eq_name_full)
b.mboz = mboz
b.nboz = nboz
b.run()
b.write_boozmn(os.path.join(full_dir,'boozmn_'+eq_name+'.nc'))

bmnc = b.bmnc_b
xm = b.xm_b
xn = b.xn_b
booz_non_qs_rms = np.sqrt(np.sum(np.abs(bmnc[xn!=0])**2)/np.sum(np.abs(bmnc)**2))
booz_non_qs_max = np.max(np.abs(bmnc[xn!=0]))/np.max(np.abs(bmnc))
surf_booz = b.s_b
iota = b.iota[-1]

print("\n booz xform results")
print(' RMS QA metric: ',booz_non_qs_rms)
print(' Max QA metric: ',booz_non_qs_max)
print(' Axis iota: ', b.iota[0])
print(' Mean iota: ',np.mean(b.iota))
print(' Edge iota: ', b.iota[-1])


# load the saved data from the previous run
surf_wf = load(os.path.join(full_dir, 'surf_wf.json'))
bs_tf = load(os.path.join(full_dir, 'TF_biot_savart_opt.json'))
wf_currents = np.load(os.path.join(full_dir, 'WF_currents.npy'))
loaded_dict = np.load(os.path.join(full_dir, 'WF_data.npz'))
wf_dict = {key: loaded_dict[key].item() for key in loaded_dict}
win_nPhi = wf_dict["win_nPhi"]
win_nTheta = wf_dict["win_nTheta"]
win_size = wf_dict["win_size"]
win_gap = wf_dict["win_gap"]

# Initialize an idential wireframe as the run and set the currents equal to the final currents
wf = windowpane_wireframe(surf_wf, win_nPhi, win_nTheta, win_size, win_size, \
                          win_gap, win_gap)
wf.currents = wf_currents
# get the field from the wireframe
bs_wf = WireframeField(wf)

# get the total field
bs = bs_tf + bs_wf

coils = bs_tf.coils
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = -2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

xm = [] 
xn = []

# m = 0 modes 
for n in range(0,nmax+1):
    xn.append(float(n*nfp))
    xm.append(float(0))

for m in range(1,mmax+1):
    for n in range(-nmax,nmax+1):
        xn.append(float(n*nfp))
        xm.append(float(m))

xm = np.array(xm)
xn = np.array(xn)

mpols = [6, 8, 10]
non_qs_rms = []
non_qs_max = []
iotas = []

print("Starting Boozer Surface BFGS Solves")
for mpol in mpols:
    print(mpol)
    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=mpol, stellsym=surf_prev.stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.least_squares_fit(surf_prev.gamma())
    
    vol = Volume(s)
    constraint_weight = 1
    boozer_surface = BoozerSurface(bs, s, vol, vol_target, constraint_weight)
    # second derivatives not implemented in wireframe yet (as of 10/3/2024), so only run L-BFGS, not Newton
    # Newton would be run if you use run_code instance
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(constraint_weight=constraint_weight, iota=iota, G=G0, \
            tol=1e-8, maxiter=1500, verbose=True, limited_memory=False, weight_inv_modB=True)
    iotas.append(res['iota'])
    print('iota: ', iotas[-1])

    x = s.gamma() 
    x = x.reshape((x.size//3, 3)).copy()
    bs.set_points(x)
    B = bs.B()
    modB = np.sqrt(np.sum(B*B,axis=-1)).reshape(np.shape(s.gamma()[:,:,0]))
    
    theta, phi = np.meshgrid(s.quadpoints_theta, s.quadpoints_phi)		
    
    phi *= 2*np.pi 
    theta *= 2*np.pi 
    
    bmnc = np.zeros_like(xm)
    for im in range(len(xm)):
    	angle = xm[im] * theta  - xn[im] * phi
    	bmnc[im] = np.sum(modB*np.cos(angle))/np.sum(np.cos(angle)**2)

    modB_ift = np.zeros_like(modB)
    for im in range(len(xm)):
    	angle = xm[im] * theta  - xn[im] * phi
    	modB_ift += bmnc[im] * np.cos(angle)

    non_qs_rms.append(np.sqrt(np.sum(np.abs(bmnc[xn!=0])**2)/np.sum(np.abs(bmnc)**2)))
    non_qs_max.append(np.max(np.abs(bmnc[xn!=0]))/np.max(np.abs(bmnc)))

    print('non_qs_rms: ',non_qs_rms[-1])
    print('non_qs_max: ',non_qs_max[-1])
    
    s_prev = s

lines = [f"Iota Values: \n"
         f"From VMEC: iota_edge       = {iota_edge:.5f}, iota_axis = {iota_ax:.5f} \n", 
         f"From booz xform: iota_edge = {b.iota[-1]:.5f}, iota_axis = {b.iota[0]:.5f}, iota_mean = {np.mean(b.iota):.5f} \n",
         f"\n"
         f"QS Error Values \n",
         f"From booz xform: rms QS metric = {booz_non_qs_rms:.5f}, max QS metric = {booz_non_qs_max:.5f}\n", \
         f"From Boozer Surface: \n", 
         f"mpols = {mpols} \n", 
         f"rms QS metrics = {non_qs_rms} \n", 
         f"max QS metrics = {non_qs_max} \n"]

with open(os.path.join(full_dir, "qs_error.txt"), "w") as file1:
    file1.writelines(lines)

