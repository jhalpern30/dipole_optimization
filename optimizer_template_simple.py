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
from create_surface import *

# for general B dot n / B plots
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


########################################################################
########################## Input Parameters ############################
########################################################################
# Windowpane parameters
axisymmetric = AXISYM_VAL            # use axisymmetric vessel for dipoles
max_dipole_field = BD_VAL            # [T], set by engineering constraints
win_nPhi = DIP_NPHI_VAL              # Number of windows per half period, toroidal dimension
win_nTheta = DIP_NTHETA_VAL          # Number of windows, poloidal dimension
win_size = DIP_SIZE_VAL              # Number of grid cells/window, both dimensions
win_gap = DIP_GAP_VAL                # Number of grid cells between adjacent windows
uneven_grid = UNEVEN_GRID_VAL        # if true, dipoles will be varying sizes
nInboard = NINBOARD_VAL              # number of smaller dipoles on inboard side
nOutboard = NOUTBOARD_VAL            # number of large dipoles on outboard side
theta_inboard_start = THETA_START_VAL # starting point for smaller dipoles, normalized such that 0 -> 1 = 0 -> 2pi
theta_inboard_end = THETA_END_VAL     # ending point for smaller dipoles

# Define a toroidal surface on which the wireframe is to be constructed
VV_a = DIP_MINOR_RAD_VAL
VV_R0 = DIP_MAJOR_RAD_VAL

# plasma surface parameters
plas_nPhi = SURF_NPHI_VAL
plas_nTheta = SURF_NTHETA_VAL
surf_s = SURF_S_VAL
surf_dof_scale = SURF_DOF_SCALE_VAL

# On-axis magnetic field (Tesla)
field_on_axis = BT_VAL

# TF coils parameters
n_tf = NTF_VAL
num_fixed = NUM_FIXED_VAL
fixed_geo = FIXED_GEO_VAL
TF_radius = TF_RADIUS_VAL

# for poster figs
dpi = DPI_VAL
titlefontsize = TITLE_SIZE_VAL
axisfontsize = AXIS_SIZE_VAL
legendfontsize = LEGEND_SIZE_VAL
ticklabelfontsize = TICK_SIZE_VAL
cbarfontsize = CBAR_SIZE_VAL
###############################

# Import plasma boundary and generate a winding surface for the wireframe
# Note - you might have to adjust this section depending on what your equilibrium file looks like
# This is current set up for one of the Henneberg NAS equilibria Elizabeth passed on
eq_dir = EQ_DIR
eq_name = EQ_NAME_VAL
eq_name_full = os.path.join(eq_dir, eq_name + '.nc')
surf_plas = create_surface(eq_name_full,'half period', plas_nPhi, plas_nTheta, surf_s, surf_dof_scale, VV_R0)

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
if uneven_grid: # adjust for only inboard dipoles
    min_theta = theta_circum * (theta_inboard_end - theta_inboard_start) / nInboard * win_size / (win_size + win_gap)
else:
    min_theta = theta_circum / win_nTheta * win_size / (win_size + win_gap)
min_phi   = phi_circum / (win_nPhi * 2 * surf_plas.nfp) * win_size / (win_size + win_gap)
min_dim = 2.54 / 100 * np.min([min_phi, min_theta])
opt_nPhi_nTheta_ratio = (VV_R0 - VV_a) / VV_a / 2 / surf_plas.nfp
print('------ Dipole Sizes ------')
print(f"Circumferences in poloidal/toroidal direction: ({theta_circum:.2f}, {phi_circum:.2f}) [in]")
print(f"Minimum dipole dimensions (theta, phi): ({min_theta:.2f}, {min_phi:2f}) [in]")
print(f"Minimum dipole dimensions (theta, phi): ({2.54*min_theta:.2f}, {2.54*min_phi:2f}) [cm]")
print(f"Minimum dipole dimension (either direction): {min_dim:.2f} [m]")
print(f"Optimum nPhi to nTheta ratio to have ~ square dipoles on inboard side = {opt_nPhi_nTheta_ratio}")
print('\n')

segs_per_win_plus_gap = win_size + win_gap
wf_nTheta = win_nTheta * segs_per_win_plus_gap
wf_nPhi = win_nPhi * segs_per_win_plus_gap
# generate nonuniform quadpoints in theta if using different dipole sizes
if uneven_grid:
    # Generate uneven spacing for the inboard side (smaller spacing)
    total_inboard_points = nInboard * segs_per_win_plus_gap
    inboard_grid = np.linspace(theta_inboard_start, theta_inboard_end, total_inboard_points)
    # Generate uneven spacing for the outboard side (larger spacing)
    total_outboard_points = nOutboard * segs_per_win_plus_gap
    # Outboard region split into two parts (0 to theta_inboard_start and theta_inboard_end to 2pi)
    outboard_grid_left = np.linspace(0, theta_inboard_start, total_outboard_points // 2, endpoint=False)
    outboard_grid_right = np.flip(np.linspace(1, theta_inboard_end, total_outboard_points // 2, endpoint=False))
    # Combine the inboard and outboard regions
    quad_theta = np.concatenate([outboard_grid_left, inboard_grid, outboard_grid_right])
    print(quad_theta)
else:
    quad_theta = np.linspace(0, 1, wf_nTheta)

if axisymmetric: 
    # axisymmetric surface
    quad_phi = np.linspace(0, 1 / 2 / surf_plas.nfp, wf_nPhi) # only a half period of the vessel
    surf_wf = create_axisym_wf_surface(quad_phi, quad_theta, surf_plas.nfp, VV_R0, VV_a)
    # use this for poincare plots post-processing
    quad_phi_full = np.linspace(0, 1, plas_nPhi)
    surf_wf_full = create_axisym_wf_surface(quad_phi_full, quad_theta, surf_plas.nfp, VV_R0, VV_a)
else:
    # non-axisymmetric surface (copy the plasma surface and extend via normal)
    # again, this will need to be changed depending on the input surface type
    # WARNING: This might not be up to date, not guaranteed to work
    surf_wf = create_surface(eq_name_full,'half period', plas_nPhi, plas_nTheta, surf_s, surf_dof_scale, VV_R0)
    surf_wf.extend_via_normal(wf_plas_offset)    

# Create the wireframe
wf = windowpane_wireframe(surf_wf, win_nPhi, win_nTheta, win_size, win_size, \
                          win_gap, win_gap, uneven_grid=uneven_grid)
wf_dict = {"win_nPhi":win_nPhi, "win_nTheta":win_nTheta, "win_size":win_size, "win_gap": win_gap}
if uneven_grid:
    uneven_params = {'nInboard':nInboard, 'nOutboard':nOutboard, 'theta_inboard_start':theta_inboard_start, 'theta_inboard_end':theta_inboard_end}

## plot cross sections
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

# JMH 2024-09-12 
# Could eventually add option to choose these once we know engineering constraints better
# Right now these seem ok, aka the weight is high enough that the threshold is maintained

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 10000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.05
CS_WEIGHT = 10000

tf_opt_params = ["Coil-Coil", CC_THRESHOLD, CC_WEIGHT, "Coil-Vessel", CS_THRESHOLD, CS_WEIGHT]

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    return J, grad

# Generate the TF coils
base_tf_curves_cylindrical = create_equally_spaced_cylindrical_curves(n_tf, surf_plas.nfp, stellsym=True, R0=VV_R0, minor_r=TF_radius,numquadpoints=32)
tf_curves_cylindrical = apply_symmetries_to_curves(base_curves=base_tf_curves_cylindrical, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
mu0 = 4.0 * np.pi * 1e-7
# toroidal solenoid approximation - B_T = mu0 * I * N / L = mu0 * I * N / L
# therefore, TF current = B_T * 2 * pi * R0 / mu0 / (2 * nfp * n_tf), n_tf = # per half field period!
poloidal_current = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0
scale_factor = -poloidal_current/(2*n_tf*surf_plas.nfp)
base_tf_currents = [Current(1) for c in base_tf_curves_cylindrical]
tf_currents = apply_symmetries_to_currents(base_currents=base_tf_currents, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves_cylindrical, base_tf_currents)]
tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(tf_curves_cylindrical, tf_currents)]
mf_tf = BiotSavart(tf_coils)

# Fix usually 1 TF coil current to prevent optimizer just sending B fields to 0
# Trying something new - fix all three TF currents during shift/tilt optimization, then only fix 1 during combined optimization
#for i in range(num_fixed):
#    base_tf_coils[i].current.fix_all()
for c in base_tf_coils:
    c.curve.fix_all()
    c.current.fix_all()
    if not fixed_geo:
        c.curve.unfix('R0')
        c.curve.unfix('r_rotation')

Jccdist = CurveCurveDistance(tf_curves_cylindrical, CC_THRESHOLD, num_basecurves=n_tf)
Jcsdist = CurveSurfaceDistance(base_tf_curves_cylindrical, surf_wf, CS_THRESHOLD)

Jf = SquaredFlux(surf_plas, mf_tf,definition='local')
JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
JF_preTF = JF.J()

# get pre TF Bnormal
relBfinal_norm, pre_TF_mean_abs_relBfinal_norm, pre_TF_relBfinal_norm_max = plot_relBfinal_norm(
    mf_tf, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Pre TF'
)

# optimize TF coils
dofs = JF.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'maxcor': 300}, tol=1e-15)
JF_postTF = JF.J()

print('\n----- TF Optimization Results ----- ')
print(f"DOF Names for TF Optimization: {JF.dof_names}")
print(f"DOF Values for TF Optimization: {JF.x}")
print(f'Squared Flux Percentage Decrease After TF Optimization = {(JF_preTF - JF_postTF) / JF_preTF * 100:.2f}%')

# get post TF Bnormal
relBfinal_norm, post_TF_mean_abs_relBfinal_norm, post_TF_relBfinal_norm_max = plot_relBfinal_norm(
    mf_tf, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Post TF'
)

########################################################################
######################### Dipole Optimization ##########################
########################################################################

# this is with a scalar lambda value - faster, but not as accurate when hitting field limits on dipoles
def fun_wf(regularization_lambda):
    params = {'reg_lambda': 10**regularization_lambda, \
          'assume_no_crossings': True}  # MUST be true for a windowpane solution
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, \
                             ext_field=mf_tf, verbose=False)
    wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max()
    print(f'     Maximum WF Current = {wf_max:.3f}, Maximum Allowed Current = {max_I:.3f},  Lambda  = {regularization_lambda:.3e}')
    return np.abs(wf_max - max_I)

print('\n----- Beginning WF Optimization -----')
start_time = time.time()

if max_dipole_field: 
    # determine the approximate allowable max current in the dipoles, assuming a square dipole
    # B_0 = sqrt(2) / pi * mu0 * I / a
    max_I = max_dipole_field * np.pi * min_dim / np.sqrt(2) / mu0
    # find the lambda that gives a max current closest to the max dipole field
    res = minimize_scalar(fun_wf, bounds=(-12, -6), method='bounded', tol=1e0)
    # rerun with the optimal lambda
    regularization_lambda = res.x
    params = {'reg_lambda': 10**regularization_lambda, 'assume_no_crossings': True}  # MUST be true for a windowpane solution
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf)
else: # if not restricting dipole current, just optimize with low lambda
    params = {'reg_lambda': 10**-14, 'assume_no_crossings': True}  # MUST be true for a windowpane solution
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf, verbose=True)

wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max()
print(f'Maximum WF Current = {wf_max:.1f} [A], Maximum WF Field = {np.sqrt(2) * mu0 * wf_max / np.pi / min_dim:.3f} [T]\n')

########################################################################
######################## Combined Optimization #########################
########################################################################

# Trying something new - fix all three TF currents during shift/tilt optimization, then only fix 1 during combined optimization
for i in range(num_fixed, n_tf):
    base_tf_coils[i].current.unfix_all()

# Now iterate between TFs and dipoles to refine optimization
JF_tol = 0.03 # minimum percentage decrease in TF/WF optimization in order to stop
# get initial squared flux difference btw TF only and TF and then WF only as starting point
JF_i = JF_postTF
# now add in dipole contribution to B field
Jf = SquaredFlux(surf_plas, mf_tf + res['wframe_field'], definition='local')
JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
print('\nIterating Between TF and WF Optimization')
while (JF_i - JF.J()) / JF_i > JF_tol: # keep looping through TF/WF optimization, making sure currents are less than max_I as needed
    dofs = JF.x
    JF_i = JF.J()
    res_tf = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'maxcor': 300}, tol=1e-15)
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf, verbose=False)
    wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max()
    Jf = SquaredFlux(surf_plas, mf_tf + res['wframe_field'], definition='local')
    JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
    print(f'\n Post Iteration Squared Flux = {JF.J():.4e}, % Difference = {(JF_i - JF.J()) / JF_i * 100:.3f}%, Target % Difference = {JF_tol * 100}% \n')

end_time = time.time()  # End the timer
total_time = end_time - start_time  # Calculate the total time
JF_final = JF.J()
print('----- Dipole Optimization Results ----- ')
print(f'Total WF/TF Optimization Time: {total_time:.2f} seconds')
print(f"DOF Names for TF Optimization: {JF.dof_names}")
print(f"Final DOF Values for TF Optimization: {JF.x}")
print(f'Squared Flux Percentage Between TF Only and Combined Optimization = {(JF_postTF - JF_final) / JF_postTF * 100:.2f}%')
print(f'Squared Flux Percentage Between Initial State and Final Optimization = {(JF_preTF - JF_final) / JF_preTF * 100:.2f}%')
print(f'Maximum WF Current = {wf_max:.1f} [A], Maximum WF Field = {np.sqrt(2) * mu0 * wf_max / np.pi / min_dim:.3f} [T]\n')

########################################################################
########################### Post-Processing ############################
########################################################################

# Determine the total magnetic field by adding the wireframe and TF coil fields
mf_tot = mf_tf + res['wframe_field']
# Surface integral of the squared normal flux through the plasma boundary
Bnormal_mf = SquaredFlux(surf_plas, mf_tot).J()
print('    Squared flux integral from field calc = %.8e' % (Bnormal_mf))
    
# Consistency check: use an Amperian loop to verify the total poloidal current
amploop = CurveXYZFourier(100, 1)
amploop.set('xc(1)', surf_plas.get_rc(0,0) + surf_plas.get_rc(0,1))
amploop.set('ys(1)', surf_plas.get_rc(0,0) - surf_plas.get_rc(0,1))
amploop_curr = enclosed_current(amploop, mf_tot, 1000)
print('    Enclosed poloidal current: %.3e' % (amploop_curr))
print('                    (expected: %.3e)' % (-2*surf_plas.nfp*np.sum([c.current.get_value() for c in base_tf_coils])))
    
# Consistency check: use an Amperian loop to check the total toroidal current
amploop_tor = CurveXYZFourier(100, 1)
amploop_tor.set('xc(0)', wf.surface.get_rc(0,0))
amploop_tor.set('xc(1)', 2*wf.surface.get_rc(1,0))
amploop_tor.set('zs(1)', 2*wf.surface.get_zs(1,0))
amploop_tor_curr = enclosed_current(amploop_tor, mf_tot, 1000)
print('    Enclosed toroidal current: %.3e' % (amploop_tor_curr))

# get final Bnormal
relBfinal_norm, final_mean_abs_relBfinal_norm, final_relBfinal_norm_max = plot_relBfinal_norm(
    mf_tot, surf_plas, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Final'
)

# plot final mod B
mf_tf.set_points(surf_plas.gamma().reshape((-1, 3)))
Bfinal = mf_tf.B().reshape(n.shape)
Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
print(f'    Surface-averaged |B| = {mean_abs_modBfinal:.3f}')

fig, ax = plt.subplots()
contour = ax.contourf(phi, theta, np.squeeze(modBfinal).T, levels=50, cmap='viridis')
ax.set_xlabel(r'$\phi/2\pi$', fontsize=axisfontsize, fontweight='bold')
ax.set_ylabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
cbar = fig.colorbar(contour, ax=ax)
cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=cbarfontsize, fontweight='bold')
cbar.ax.tick_params(axis='y', which='major', labelsize=ticklabelfontsize)
ax.set_title(f'Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=titlefontsize, fontweight='bold')
plt.tight_layout()
plt.savefig('modB_final.png', dpi=dpi)
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
surf_plas_full = create_surface(eq_name_full,'full torus', 2*wf.nfp*plas_nPhi, plas_nTheta, surf_s, surf_dof_scale, VV_R0)
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

for c in base_tf_coils:
    c.current.fix_all()

with open("TF_shifts_tilts.txt", "w") as file:
    file.write("R0 in meters, rotations in radians \n")
    for i in range(len(JF.x)):
        file.write(f"{JF.dof_names[i]} = {JF.x[i]:.2f} \n")

# Write the optimization output file
# Can always add more to this as we determine what is relevant information
lines = [f"Equilibrium file location: {eq_name_full} with s = {surf_s} and dof_scale = {surf_dof_scale} \n", 
         f"Equilibrium parameters: major radius = {major_radius:.3f}, minor radius = {minor_radius:.3f}, volume = {volume:.3f}, aspect ratio = {aspect_ratio:.3f} \n",
         f"On axis magnetic field = {field_on_axis} T \n",
         f"Maximum allowed dipole field = {max_dipole_field} [T] \n", \
         f"TF parameters: ntf = {n_tf}, num_fixed = {num_fixed}, radius = {TF_radius:.3f} [m], opt params = {tf_opt_params} \n", 
         f"Dipole parameters: nPhi = {win_nPhi}, nTheta = {win_nTheta}, min dimensions (phi, theta) = ({min_phi:.2f},{min_theta:.2f}) [in] \n",
         f"Uneven grid? {uneven_grid}. If True, uneven grid params: {uneven_params} \n"
         f"Vessel parameters: Axisymmetric? {axisymmetric}, major radius = {VV_R0}, minor radius = {VV_a} , average plasma-vessel distance = {wf_plas_offset:.4f} [m]\n",
         f"\n", 
         f"Optimization results \n",
         f"Pre TF Surface-averaged/Max |Bn|/|B| = {pre_TF_mean_abs_relBfinal_norm:.3e}/{pre_TF_relBfinal_norm_max:.3e} \n", 
         f"Post TF Surface-averaged/Max |Bn|/|B| = {post_TF_mean_abs_relBfinal_norm:.3e}/{post_TF_relBfinal_norm_max:.3e} \n",
         f"Final Surface-averaged/Max |Bn|/|B| = {final_mean_abs_relBfinal_norm:.3e}/{final_relBfinal_norm_max:.3e} \n",
         f'Squared Flux Percentage Decrease After TF Optimization = {(JF_preTF - JF_postTF) / JF_preTF * 100:.2f}% \n',
         f'Squared Flux Percentage Between TF Only and Combined Optimization = {(JF_postTF - JF_final) / JF_postTF * 100:.2f}% \n',
         f'Squared Flux Percentage Between Initial State and Final Optimization = {(JF_preTF - JF_final) / JF_preTF * 100:.2f}% \n',
         f"Final Surface-averaged |B| = {mean_abs_modBfinal:.3f} \n",
         f"Maximum WF Current = {wf_max:.3e}, Maximum Dipole Field = {np.sqrt(2) * mu0 * wf_max / np.pi / min_dim} [T]\n"]
 
with open("parameters.txt", "w") as file1:
    file1.writelines(lines)
