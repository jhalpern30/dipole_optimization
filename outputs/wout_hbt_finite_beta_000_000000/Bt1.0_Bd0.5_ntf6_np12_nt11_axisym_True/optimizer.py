#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import os
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.wireframe import ToroidalWireframe, windowpane_wireframe
from simsopt.geo import HBTCylFourier, create_equally_spaced_cylindrical_curves, ToroidalFlux, curves_to_vtk
from simsopt.field.coil import apply_symmetries_to_curves, apply_symmetries_to_currents, coils_via_symmetries, Current, Coil, ScaledCurrent
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.wireframefield import WireframeField, enclosed_current
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.solve.wireframe_optimization import optimize_wireframe
from simsopt.geo import CurveCurveDistance, CurveSurfaceDistance
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pylab as plt
from simsopt._core import load
from simsopt.mhd import VirtualCasing, Vmec, virtual_casing

########################################################################
########################## Input Parameters ############################
########################################################################

# On-axis magnetic field (Tesla)
field_on_axis = 1.0

# TF coils parameters
n_tf = 6
num_fixed = 1
full_TF_scan = False
TF_r_values = [0.5, 0.55]
vc_flag = True

# Windowpane parameters
axisymmetric = True            # use axisymmetric vessel for dipoles
max_dipole_field = 0.5            # [T], set by engineering constraints
win_nPhi = 12            # Number of windows per half period, toroidal dimension
win_nTheta = 11          # Number of windows, poloidal dimension
win_size = 8              # Number of grid cells/window, both dimensions
win_gap = 2                # Number of grid cells between adjacent windows

# Define a toroidal surface on which the wireframe is to be constructed
VV_a = 0.25
VV_R0 = 1.0

# plasma surface parameters
plas_nPhi = 64
plas_nTheta = 64
surf_s = 1.0
surf_dof_scale = 0.15

# for poster figs
dpi = 100
titlefontsize = 18
axisfontsize = 16
legendfontsize = 14
ticklabelfontsize = 14
cbarfontsize = 18
###############################

# Import plasma boundary - this script is HBT but could be used for any equilibria you don't
# want to include TF degrees of freedom on
# Note - you might have to adjust this section depending on what your equilibrium file looks like
# This is current set up for HBT's equilibria
eq_dir = '/Users/jakehalpern/Projects/C-REX/Cleaned_up_for_Github/equilibria'
eq_name = 'wout_hbt_finite_beta_000_000000'
eq_name_full = os.path.join(eq_dir, eq_name + '.nc')

surf_plas = SurfaceRZFourier.from_wout(eq_name_full, surf_s, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)
surf_plas.set_rc(0,0,VV_R0)

# Geometric properties of the target plasma boundary
n = surf_plas.normal()
absn = np.linalg.norm(n, axis=2)
unitn = n * (1./absn)[:,:,None]
sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
surf_area = sqrt_area**2

major_radius = surf_plas.major_radius()
minor_radius = surf_plas.minor_radius()
aspect_ratio = surf_plas.aspect_ratio()
volume = surf_plas.volume()
wf_plas_offset = VV_a - surf_plas.minor_radius() # for consistency between non/axisymmetry, set offset = average VV to plasma distance

theta = surf_plas.quadpoints_theta
phi = surf_plas.quadpoints_phi

print('------ Plasma Surface Parameters ------')
print(f' nfp = {surf_plas.nfp}')
print(f' major radius = {major_radius}')
print(f' minor radius = {minor_radius}')
print(f' aspect ratio = {aspect_ratio}')
print(f' volume= {volume}')
print(f' avg plasma-vessel distance = {wf_plas_offset}')
print('\n')

# Approximation for the dipole dimensions based on toroidal/poloidal circumference of vessel, only exact for axisym = True
# We later calculate the B field assuming a square dipole peak field - requires ~ square dipoles on inboard side 
# and dipoles not too large a fraction of the circumference to be valid
theta_circum = 100 / 2.54 * VV_a * 2 * 3.14
phi_circum   = 100 / 2.54 * (VV_R0 - VV_a) * 2 * 3.14
min_theta = theta_circum / win_nTheta * win_size / (win_size + win_gap)
min_phi   = phi_circum / (win_nPhi * 2 * surf_plas.nfp) * win_size / (win_size + win_gap)
min_dim = 2.54 / 100 * np.min([min_phi, min_theta])
opt_nPhi_nTheta_ratio = (VV_R0 - VV_a) / VV_a / 2 / surf_plas.nfp
print('------ Dipole Sizes ------')
print(f"Minimum dipole dimensions (theta, phi): ({2.54*min_theta:.2f}, {2.54*min_phi:2f}) [cm]")
print(f"Minimum dipole dimension (either direction): {min_dim:.2f} [m]")
print(f"Optimum nPhi to nTheta ratio to have ~ square dipoles on inboard side = {opt_nPhi_nTheta_ratio}")
print('\n')

if axisymmetric: 
    # axisymmetric surface
    quad_theta = np.linspace(0, 1, plas_nTheta)
    quad_phi = np.linspace(0, 1 / 2 / surf_plas.nfp, plas_nPhi) # only a half period of the vessel
    surf_wf = SurfaceRZFourier(quadpoints_phi=quad_phi, quadpoints_theta=quad_theta, nfp = surf_plas.nfp)
    surf_wf.set_rc(0,0,VV_R0)
    surf_wf.set_rc(1,0,VV_a)
    surf_wf.set_zs(1,0,VV_a)
    # use this for poincare plots post-processing
    quad_phi_full = np.linspace(0, 1, plas_nPhi) # only a half period of the vessel
    surf_wf_full = SurfaceRZFourier(quadpoints_phi=quad_phi_full, quadpoints_theta=quad_theta, nfp = surf_plas.nfp)
    surf_wf_full.set_rc(0,0,VV_R0)
    surf_wf_full.set_rc(1,0,VV_a)
    surf_wf_full.set_zs(1,0,VV_a)
else:
    # non-axisymmetric surface (copy the plasma surface and extend via normal)
    # again, this will need to be changed depending on the input surface type
    # non-axisymmetric surface
    # WARNING: This isn't up to date, not guaranteed to work
    surf_wf = SurfaceRZFourier.from_wout(input_QA, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta, stellsym = True)
    surf_wf.extend_via_normal(wf_plas_offset)

# Create the wireframe
wf = windowpane_wireframe(surf_wf, win_nPhi, win_nTheta, win_size, win_size, \
                          win_gap, win_gap)
wf_dict = {"win_nPhi":win_nPhi, "win_nTheta":win_nTheta, "win_size":win_size, "win_gap": win_gap}

## plot cross sections
# only need one phi_slice for HBT but keeping generic if you want to do
# dipole only (keep TFs fixed) 
plt.figure(figsize=(8,5))
phi_array = np.arange(0, 1.01, 0.2)
for phi_slice in phi_array:
    cs = surf_plas.cross_section(phi_slice*np.pi)
    magaxis_r = np.mean(cs, axis=0)[0]**2 + np.mean(cs, axis=0)[1]**2
    magaxis_z = np.mean(cs, axis=0)[2]
    r = np.sqrt(cs[:,0]**2 + cs[:,1]**2)
    z = np.mean(cs, axis=0)[2]
    cs2 = surf_wf.cross_section(phi_slice*np.pi)
    r2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2)
    plt.plot(r, cs[:,2], label=fr'$\phi$ = {phi_slice:.2f}π')
    plt.plot(r2, cs2[:,2], 'k')
    plt.plot(np.mean(r), np.mean(z), 'kx')

plt.xlabel('R', fontsize=axisfontsize, fontweight='bold')
plt.ylabel('Z', fontsize=axisfontsize, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=legendfontsize)
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.savefig('x_section.png', dpi = dpi)
plt.clf()  

########################################################################
########################### TF Optimization ############################
########################################################################

if vc_flag:
    # Resolution for the virtual casing calculation:
    vc_src_nphi = 80
    # (For the virtual casing src_ resolution, only nphi needs to be
    # specified; the theta resolution is computed automatically to
    # minimize anisotropy of the grid.)
    
    # Once the virtual casing calculation has been run once, the results
    # can be used for many coil optimizations. Therefore here we check to
    # see if the virtual casing output file alreadys exists. If so, load
    # the results, otherwise run the virtual casing calculation and save
    # the results.
    eq_name_full_for_vc = os.path.join(eq_dir, eq_name + f'nPhi_{plas_nPhi}_nTheta_{plas_nTheta}' + '.nc')
    head, tail = os.path.split(eq_name_full_for_vc)
    vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
    vcase = virtual_casing.VirtualCasing()
    
    print('virtual casing data file:', vc_filename)
    if os.path.isfile(vc_filename):
        print('Loading saved virtual casing result')
        vc = VirtualCasing.load(vc_filename)
    else:
        # Virtual casing must not have been run yet.
        print('Running the virtual casing calculation')
        vc = VirtualCasing.from_vmec(eq_name_full, src_nphi=vc_src_nphi, trgt_nphi=plas_nPhi, trgt_ntheta=plas_nTheta)
        vc.save(vc_filename)

TF_radius = TF_r_values[0]
if len(TF_r_values) > 1:
    print(f"\n WARNING: Only one TF radius accepted for dipole only optimization. Using the first entry: TF_radius = {TF_radius}m \n")
    
print(f'Minimum Dipole Dimension to TF Radius Ratio: {min_dim/TF_radius:3f}')

# Generate the optimum TF coils
base_tf_curves_cylindrical = create_equally_spaced_cylindrical_curves(n_tf, surf_plas.nfp, stellsym=True, R0=VV_R0, minor_r=TF_radius,numquadpoints=32)
tf_curves_cylindrical = apply_symmetries_to_curves(base_curves=base_tf_curves_cylindrical, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
mu0 = 4.0 * np.pi * 1e-7
poloidal_current = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0
scale_factor = -poloidal_current/(2*n_tf*surf_plas.nfp)
base_tf_currents = [Current(1) for c in base_tf_curves_cylindrical]
tf_currents = apply_symmetries_to_currents(base_currents=base_tf_currents, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves_cylindrical, base_tf_currents)]
tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(tf_curves_cylindrical, tf_currents)]
mf_tf = BiotSavart(tf_coils)

Jf = SquaredFlux(surf_plas, mf_tf,definition='local')
JF = Jf
JF_preTF = JF.J()

def plot_relBfinal_norm(mf_tf, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, label):
    mf_tf.set_points(surf_plas.gamma().reshape((-1, 3)))
    Bfinal = mf_tf.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rBnorm = np.max(np.abs(relBfinal_norm))
    print(f'    Surface-averaged |Bn|/|B| = {mean_abs_relBfinal_norm:.8e}')
    
    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50, cmap='coolwarm', vmin=-max_rBnorm, vmax=max_rBnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=axisfontsize, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=cbarfontsize, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=ticklabelfontsize)
    ax.set_title(f'{label} Surface-averaged \n |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=titlefontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'relBn_{label.replace(" ", "")}.png', dpi=dpi)
    plt.clf()
    return relBfinal_norm, mean_abs_relBfinal_norm, np.max(relBfinal_norm)

# get pre TF Bnormal
relBfinal_norm, TF_mean_abs_relBfinal_norm, TF_relBfinal_norm_max = plot_relBfinal_norm(
    mf_tf, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Pre TF'
)

# initial mod B plot
mf_tf.set_points(surf_plas.gamma().reshape((-1, 3)))
Bfinal = mf_tf.B().reshape(n.shape)
Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
print(f'Initial Surface-averaged |B| = {mean_abs_modBfinal:.3f}')
fig, ax = plt.subplots()
contour = ax.contourf(phi, theta, np.squeeze(modBfinal).T, levels=25, cmap='viridis')
ax.set_xlabel(r'$\phi/2\pi$', fontsize=axisfontsize, fontweight='bold')
ax.set_ylabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
cbar = fig.colorbar(contour, ax=ax)
cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=cbarfontsize, fontweight='bold')
cbar.ax.tick_params(axis='y', which='major', labelsize=ticklabelfontsize)
ax.set_title(f'Initial Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=titlefontsize, fontweight='bold')
plt.tight_layout()
plt.savefig('modB_initial.png', dpi=dpi)
plt.clf()

# plot coil ripple parameter delta
delta = (np.max(modBfinal, axis=0) - np.min(modBfinal, axis=0)) / (np.max(modBfinal, axis=0) + np.min(modBfinal, axis=0))
theta = surf_plas.quadpoints_theta
# Plot delta as a function of theta
plt.figure(figsize=(10, 6))
plt.plot(theta, delta, marker='o')
plt.xlabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
plt.ylabel(r'$\delta \approx \left.\frac{B_{\mathrm{max}} - B_{\mathrm{min}}}{B_{\mathrm{max}} + B_{\mathrm{min}}}\right|_\phi$', fontsize=axisfontsize, fontweight='bold')
plt.title(r'Initial LCFS-averaged $\delta$: {:.2e}'.format(np.mean(delta)), fontsize=titlefontsize, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.grid(True)
plt.savefig('delta_initial.png')
plt.clf()  

########################################################################
######################### Dipole Optimization ##########################
########################################################################

print('----- Beginning WF Optimization -----')
regularization_lambda = 10e-14 # prioritize low field error only
params = {'reg_lambda': regularization_lambda, 'assume_no_crossings': True}
if vc_flag: 
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf, bn_plas_curr=vc.B_external_normal.reshape(plas_nPhi*plas_nTheta, 1), verbose=False)
else:
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf, verbose=False)
    
max_dipole_field = np.sqrt(2) * mu0 * np.abs(wf.currents.max()) / np.pi / min_dim

########################################################################
########################### Post-Processing ############################
########################################################################

# Determine the total magnetic field by adding the wireframe and TF coil fields
mf_tot = mf_tf + res['wframe_field']
# Surface integral of the squared normal flux through the plasma boundary
Bnormal_mf = SquaredFlux(surf_plas, mf_tot, target=vc.B_external_normal).J()
print('    Squared flux integral from field calc = %.8e' % (Bnormal_mf))

# get final Bnormal
relBfinal_norm, final_mean_abs_relBfinal_norm, final_relBfinal_norm_max = plot_relBfinal_norm(
    mf_tot, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Final'
)

# plot final mod B
mf_tot.set_points(surf_plas.gamma().reshape((-1, 3)))
Bfinal = mf_tot.B().reshape(n.shape)
Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
print(f'    Surface-averaged |B| = {mean_abs_modBfinal:.3f}')
fig, ax = plt.subplots()
contour = ax.contourf(phi, theta, np.squeeze(modBfinal).T, levels=25, cmap='viridis')
ax.set_xlabel(r'$\phi/2\pi$', fontsize=axisfontsize, fontweight='bold')
ax.set_ylabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
cbar = fig.colorbar(contour, ax=ax)
cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=cbarfontsize, fontweight='bold')
cbar.ax.tick_params(axis='y', which='major', labelsize=ticklabelfontsize)
ax.set_title(f'Final Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=titlefontsize, fontweight='bold')
plt.tight_layout()
plt.savefig('modB_final.png', dpi=dpi)
plt.clf()

# plot coil ripple parameter delta
delta = (np.max(modBfinal, axis=0) - np.min(modBfinal, axis=0)) / (np.max(modBfinal, axis=0) + np.min(modBfinal, axis=0))
plt.figure(figsize=(10, 6))
plt.plot(theta, delta, marker='o')
plt.xlabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
plt.ylabel(r'$\delta \approx \left.\frac{B_{\mathrm{max}} - B_{\mathrm{min}}}{B_{\mathrm{max}} + B_{\mathrm{min}}}\right|_\phi$', fontsize=axisfontsize, fontweight='bold')
plt.title(r'Final LCFS-averaged $\delta$: {:.2e}'.format(np.mean(delta)), fontsize=titlefontsize, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.grid(True)
plt.savefig('delta_final.png')
plt.clf()  

# Plot of the current in each wireframe segment
wf.make_plot_2d(quantity='currents', extent='half period', linewidths=0.8)
plt.xlabel('Dipole Toroidal Index', fontsize=axisfontsize, fontweight='bold')
plt.ylabel('Dipole Poloidal Index', fontsize=axisfontsize, fontweight='bold')
plt.title(f'Max current = {np.max(wf.currents)/1e3:.4f} kA', fontsize=titlefontsize, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.savefig('currents.png', dpi=dpi)
plt.clf()

from mayavi import mlab
mlab.options.offscreen = True
# 3D visualization of the wireframe, TF coils, and plasma boundary
mlab.figure(size=(1200,900), bgcolor=(1.0,1.0,1.0))
wf.make_plot_3d(engine='mayavi', to_show='active', tube_radius=0.005)
surf_plas_full = SurfaceRZFourier.from_wout(eq_name_full, surf_s, range='full torus', nphi=2*wf.nfp*plas_nPhi, ntheta=plas_nTheta)
surf_plas_full.set_dofs(surf_dof_scale*surf_plas_full.get_dofs())
surf_plas_full.set_rc(0,0,VV_R0)
surf_plas_full.plot(engine='mayavi', close=True, wireframe=False, \
                    show=False, color=(0.75, 0.75, 0.75))
if n_tf > 0:
    for coil in tf_coils:
        coil.curve.plot(close=True, show=False, engine='mayavi')
mlab.view(distance=surf_plas.get_rc(0,0)*6)
mlab.savefig('plot3d.png', size=(1400, 1000))

########################################################################
############################## Save Data ###############################
########################################################################

# convert the currents for a half period into the whole torus
currents = wf.currents
currents_full = np.zeros((2*surf_wf.nfp*wf.nSegments))
for i in range(2*surf_wf.nfp):
    ind0 = i*wf.nSegments
    ind1 = (i+1)*wf.nSegments
    currents_full[ind0:ind1] = wf.currents[:]

# see sample post-processing script (Poincare plots) for how to regenerate wireframe
curves = [c.curve for c in base_tf_coils]
surf_plas.to_vtk('surf_plas', extra_data={"B_N/B": relBfinal_norm})
surf_wf.to_vtk('vacuumvessel')
mf_tf.save('TF_biot_savart_opt.json');
wf.to_vtk('wf_grid')
np.save('WF_currents', currents_full)
surf_wf_full.save('surf_wf.json')
np.savez('WF_data', **wf_dict)
curves_to_vtk(curves, 'TF_coils', close=True)

# Write the optimization output file
# Can always add more to this as we determine what is relevant information
lines = [f"Equilibrium file location: {eq_name_full} with s = {surf_s} and dof_scale = {surf_dof_scale} \n", 
         f"Equilibrium parameters: major radius = {major_radius:.3f}, minor radius = {minor_radius:.3f}, volume = {volume:.3f}, aspect ratio = {aspect_ratio:.3f} \n",
         f"On axis magnetic field = {field_on_axis} T \n",
         f"Maximum allowed dipole field = {max_dipole_field} [T] \n", \
         f"TF parameters: ntf = {n_tf}, num_fixed = {num_fixed}, radius = {TF_radius:.3f} [m]\n", 
         f"Dipole parameters: nPhi = {win_nPhi}, nTheta = {win_nTheta}, min dimensions (phi, theta) = ({min_theta:.2f},{min_phi:.2f}) [in] \n",
         f"Vessel parameters: Axisymmetric? {axisymmetric}, major radius = {VV_R0}, minor radius = {VV_a} , average plasma-vessel distance = {wf_plas_offset:.4f} [m]\n",
         f"\n", 
         f"Optimization results \n",
         f"Pre TF Surface-averaged/Max |Bn|/|B| = {TF_mean_abs_relBfinal_norm:.3e}/{TF_relBfinal_norm_max:.3e} \n",
         f"Final Surface-averaged |B| = {mean_abs_modBfinal:.3e} [T]\n",
         f"Final Surface-averaged |B| = {mean_abs_modBfinal:.3f} + \n",
         f"Maximum WF Current = {np.abs(wf.currents[wf.unconstrained_segments()]).max():.3e}\n",          f"Maximum TF Current = {scale_factor:.3e} [A] \n",
         f"Maximum determined dipole field = {max_dipole_field:.3f} [T]"]
 
with open("parameters.txt", "w") as file1:
    file1.writelines(lines)
