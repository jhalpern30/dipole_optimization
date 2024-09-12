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

################################
####### Input Parameters #######
################################
# Windowpane parameters
axisymmetric = True            # use axisymmetric vessel for dipoles
max_dipole_field = 0.5            # [T], set by engineering constraints
win_nPhi = 6              # Number of windows per half period, toroidal dimension
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

# On-axis magnetic field (Tesla)
field_on_axis = 1.0

# TF coils parameters
n_tf = 3
num_fixed = 1
full_TF_scan = False
TF_r_values = [0.5, 0.55]

# for poster figs
dpi = 100
titlefontsize = 18
axisfontsize = 16
legendfontsize = 14
ticklabelfontsize = 14
cbarfontsize = 18
###############################

# Import plasma boundary and generate a winding surface for the wireframe
# Note - you might have to adjust this section depending on what your equilibrium file looks like
# This is current set up for one of the Henneberg NAS equilibria Elizabeth passed on
test_dir = '/Users/jakehalpern/Projects/C-REX/Cleaned_up_for_Github/equilibria'
eq_name = 'wout_NAS_n2n4_AR6.2.03'
input_QA = os.path.join(test_dir, eq_name + '.nc')

surf_nfp1 = SurfaceRZFourier.from_wout(input_QA, surf_s, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)

surf_plas = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=2,stellsym=True,
                                quadpoints_theta=surf_nfp1.quadpoints_theta,
                                quadpoints_phi=surf_nfp1.quadpoints_phi)
surf_plas.least_squares_fit(surf_nfp1.gamma())
# surf_plas = SurfaceRZFourier.from_wout(input_QA, surf_s, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)
surf_plas.set_dofs(surf_dof_scale*surf_plas.get_dofs())
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
print(f"Circumferences in poloidal/toroidal direction: ({theta_circum:.2f}, {phi_circum:.2f}) [in]")
print(f"Minimum dipole dimensions (theta, phi): ({min_theta:.2f}, {min_phi:2f}) [in]")
print(f"Minimum dipole dimensions (theta, phi): ({2.54*min_theta:.2f}, {2.54*min_phi:2f}) [cm]")
print(f"Minimum dipole dimension (either direction): {min_dim:.2f} [m]")
print(f"Optimum nPhi to nTheta ratio to have ~ square dipoles on inboard side = {opt_nPhi_nTheta_ratio}")
print('\n')

# Plot surface curvatures
surface_curvatures = surf_plas.surface_curvatures()
nphi,ntheta,_ = surf_plas.gamma().shape
Phi, Theta = np.meshgrid(phi, theta)
mean_curvature = np.transpose(surface_curvatures[:, :, 0].reshape((nphi,ntheta)))
gaussian_curvature = np.transpose(surface_curvatures[:, :, 1].reshape((nphi,ntheta)))
kappa_1 = np.transpose(surface_curvatures[:, :, 2].reshape((nphi,ntheta)))
kappa_2 = np.transpose(surface_curvatures[:, :, 3].reshape((nphi,ntheta)))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ['Mean Curvature', 'Gaussian Curvature', 'Kappa_1', 'Kappa_2']
data = [mean_curvature, gaussian_curvature, kappa_1, kappa_2]
for ax, title, datum in zip(axes.flat, titles, data):
    cax = ax.pcolormesh(Phi, Theta, datum, cmap='seismic', shading='auto')
    ax.set_title(title)
    ax.set_xlabel('Phi')
    ax.set_ylabel('Theta')
    fig.colorbar(cax, ax=ax)
fig.suptitle('Surface Curvature Terms', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('surface_curvatures.png', dpi=dpi)
plt.clf()

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
    surf_wf.set_rc(0,0,VV_R0)
    surf_wf.set_rc(1,0,VV_a)
    surf_wf.set_zs(1,0,VV_a)
else:
    # non-axisymmetric surface (copy the plasma surface and extend via normal)
    # again, this will need to be changed depending on the input surface type
    surf_wf = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=2,stellsym=True,
                                    quadpoints_theta=surf_nfp1.quadpoints_theta,
                                    quadpoints_phi=surf_nfp1.quadpoints_phi)
    surf_wf.least_squares_fit(surf_nfp1.gamma())
    surf_wf.set_dofs(surf_plas.get_dofs())
    surf_wf.set_rc(0,0,VV_R0)
    #surf_wf = SurfaceRZFourier.from_wout(input_QA, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)
    surf_wf.extend_via_normal(wf_plas_offset)    

# Create the wireframe
wf = windowpane_wireframe(surf_wf, win_nPhi, win_nTheta, win_size, win_size, \
                          win_gap, win_gap)
wf_dict = {"win_nPhi":win_nPhi, "win_nTheta":win_nTheta, "win_size":win_size, "win_gap": win_gap}

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

print('----- Optimizing TF coil radius for minimum normal field -----')
# Define the range of TF coil radius values
J_values = []
dof_names = []
dof_values = []
start_time = time.time()

for r in TF_r_values:
    # Generate some TF coils.
    base_tf_curves_cylindrical = create_equally_spaced_cylindrical_curves(n_tf, surf_plas.nfp, stellsym=True, R0=VV_R0, minor_r= r,numquadpoints=32)
    tf_curves_cylindrical = apply_symmetries_to_curves(base_curves=base_tf_curves_cylindrical, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
    
    # Determine required poloidal current
    mu0 = 4.0 * np.pi * 1e-7
    poloidal_current = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0
    scale_factor = -poloidal_current/(2*n_tf*surf_plas.nfp)
    base_tf_currents = [Current(1) for c in base_tf_curves_cylindrical]
    tf_currents = apply_symmetries_to_currents(base_currents=base_tf_currents, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
    base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves_cylindrical, base_tf_currents)]
    tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(tf_curves_cylindrical, tf_currents)]
    mf_tf = BiotSavart(tf_coils)
    
    # Fixing the tf coils geometry and current in the first num_fixed coils
    for i in range(num_fixed):
        base_tf_coils[i].current.fix_all()
    for c in base_tf_coils:
        c.curve.fix_all()
        c.curve.unfix('R0')
        c.curve.unfix('r_rotation')
    
    Jccdist = CurveCurveDistance(tf_curves_cylindrical, CC_THRESHOLD, num_basecurves=n_tf)
    Jcsdist = CurveSurfaceDistance(base_tf_curves_cylindrical, surf_wf, CS_THRESHOLD)
    
    Jf = SquaredFlux(surf_plas, mf_tf,definition='local')
    JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'maxcor': 300}, tol=1e-15)
    J_value = res.fun
    J_values.append(J_value)
    dof_values.append(JF.x.copy())
    if not dof_names:
        dof_names = JF.dof_names
    print(f'     TF radius = {r:.3f} m, J={J_value:.6e}')
    if not full_TF_scan and len(J_values) > 2:
        if abs(J_values[-1] - J_values[-2]) / J_values[-2] < 0.05:
            index = np.where(TF_r_values == r)[0][0]
            TF_r_values = TF_r_values[:index+1]
            break

end_time = time.time()  # End the timer
total_time = end_time - start_time  # Calculate the total time
print(f'TF optimization time: {total_time:.2f} seconds')

# Analysis to find the optimal TF coil radius where the change in J is less than 5% cause J decreases monotonically
for i in range(1, len(J_values)):
    if abs(J_values[i] - J_values[i-1]) / J_values[i-1] < 0.05:
        TF_R1 = TF_r_values[i]
        print(f'Optimal value of TF coil radius where J changes less than 5% is {TF_R1:.2f} m with J={J_values[i]:.6e}')
        J_opt = J_values[i]
        break
else:
    # If no break occurred, set TF_R1 to the last value
    TF_R1 = TF_r_values[-1]
    J_opt = J_values[-1]
    print(f'Optimal value of TF coil radius is {TF_R1:.2f} m with J={J_values[-1]:.6e}')

print(f'Minimum Dipole Dimension to TF Radius Ratio: {min_dim/TF_R1:3f}')

dof_types = {}
for idx, name in enumerate(dof_names):
    dof_type = name.split(':')[1]
    if dof_type not in dof_types:
        dof_types[dof_type] = []
    dof_types[dof_type].append(idx)

# Plotting J and DOF values versus TF coil radius
num_plots = len(dof_types) + 1
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
fig.suptitle(f'Optimization of TF radius with VV radius = {VV_a:.3f} [m] \n Optimum: J = {J_opt:.6e} at r = {TF_R1:.3f} [m]', fontsize=titlefontsize, fontweight='bold')
plt.subplots_adjust(top=0.92)
# Plot J versus TF coil radius
axes[0].plot(TF_r_values, np.log10(J_values), marker='o')
axes[0].set_ylabel('log10(J)', fontsize=18, fontweight='bold')
axes[0].grid(True)
axes[0].tick_params(axis='both', which='major', labelsize=ticklabelfontsize)

# Plot each DOF type versus TF coil radius
for i, (dof_type, indices) in enumerate(dof_types.items(), start=1):
    for index in indices:
        tf_number = dof_names[index].split(':')[0].split('HBTCylFourier')[-1]
        if 'Current' in tf_number:
            tf_number = dof_names[index].split(':')[0].split('Current')[-1]
        axes[i].plot(TF_r_values, [values[index] for values in dof_values], marker='o', label=f'TF {tf_number}')
    if dof_type == 'x0':
        dof_type = 'Scaled Current'
    axes[i].set_ylabel(dof_type, fontsize=18, fontweight='bold')
    axes[i].legend(fontsize=legendfontsize)
    axes[i].grid(True)
    axes[i].tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
axes[-1].set_xlabel('TF radius [m]', fontsize=18, fontweight='bold')
plt.tight_layout
plt.savefig('TF_optimization_current_R0_r_rotation.png', dpi=dpi)
plt.clf()

# Generate the optimum TF coils
base_tf_curves_cylindrical = create_equally_spaced_cylindrical_curves(n_tf, surf_plas.nfp, stellsym=True, R0=VV_R0, minor_r=TF_R1,numquadpoints=32)
tf_curves_cylindrical = apply_symmetries_to_curves(base_curves=base_tf_curves_cylindrical, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
mu0 = 4.0 * np.pi * 1e-7
poloidal_current = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0
scale_factor = -poloidal_current/(2*n_tf*surf_plas.nfp)
base_tf_currents = [Current(1) for c in base_tf_curves_cylindrical]
tf_currents = apply_symmetries_to_currents(base_currents=base_tf_currents, nfp=surf_plas.nfp, stellsym=surf_plas.stellsym)
base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves_cylindrical, base_tf_currents)]
tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(tf_curves_cylindrical, tf_currents)]
mf_tf = BiotSavart(tf_coils)

# Fix usually 1 TF coil current to prevent optimizer just sending B fields to 0
for i in range(num_fixed):
    base_tf_coils[i].current.fix_all()
for c in base_tf_coils:
    c.curve.fix_all()
    c.curve.unfix('R0')
    c.curve.unfix('r_rotation')

Jccdist = CurveCurveDistance(tf_curves_cylindrical, CC_THRESHOLD, num_basecurves=n_tf)
Jcsdist = CurveSurfaceDistance(base_tf_curves_cylindrical, surf_wf, CS_THRESHOLD)

Jf = SquaredFlux(surf_plas, mf_tf,definition='local')
JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
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

def optimize_wireframe_wrapper(Lambda):
    params = {'reg_lambda': Lambda, 'assume_no_crossings': True}
    res = optimize_wireframe(wf, 'rcls', params, surf_plas=surf_plas, ext_field=mf_tf, verbose=False)
    return res

def update_Lambda(Lambda):
    # hyperparameter that avoids scale going exactly to 1 when wf_max -> max_I
    # larger -> faster convergence but oscillations can occur
    # initial tests indicate about 0.01 to 0.1 works fairly well
    scale_fac = 0.05
    # Increase lambda for currents above the threshold
    for i in range(len(wf.currents)):
        if wf_max > max_I:
            Lambda[i] *= np.abs(wf.currents[i]) / max_I + scale_fac 
    # If all currents are below the tolerance (i.e. too large a jump occurred), scale up entire array
    if wf_max < max_I * (1 - cur_tol):
        Lambda *= wf_max / max_I - scale_fac
    return Lambda

# various tolerances used in optimization
cur_tol = 0.01  # final max current will be +- cur_tol % from max allowed current, reduces iterations below
JF_tol = 0.005 # minimum percentage decrease in TF/WF optimization in order to stop

# determine the approximate allowable max current in the dipoles, assuming a square dipole
# B_0 = sqrt(2) / pi * mu0 * I / a
max_I = max_dipole_field * np.pi * min_dim / np.sqrt(2) / mu0

print('\n----- Beginning WF Optimization -----')
start_time = time.time()

# Optimize the wireframe with very low current restriction and scalar lambda to get the max current in the ideal configuration
lambda_init = 10e-14 # prioritize low field error only
res = optimize_wireframe_wrapper(lambda_init)
wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max().copy()
print(f'Optimal current for minimum Bnormal: {wf_max:.2f} A')
field_flag = True

if wf_max < max_I:
    max_B = np.sqrt(2) * mu0 * optimal_I / np.pi / min_dim
    field_flag = False # makes sure it doesn't enter regularization matrix loops in TF/WF iterations
    Lambda = lambda_init
    optimal_I = wf_max
    print(f"Maximum WF current in minimum Bnormal configuration doesn't reach the maximum allowed dipole fields = {max_dipole_field} A")
    print(f"Maximum B field in dipole = {max_B:.3f} T")
    print("Not entering any matrix lambda optimization loops")
else: # only need to do the loop if allowed current is less than optimal current
    print(f"Maximum current with no current limitation, {wf_max:.2f} A, is above the maximum allowed current, {max_I:.2f} A")
    print('Optimizing regularization matrix entries to bring maximum current below maximum allowed current')
    lambda_init = 10e-12 # make everything prioritize Bnormal relatively well
    Lambda = np.ones(wf.nSegments) * lambda_init
    res = optimize_wireframe_wrapper(Lambda)
    wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max().copy()
    while wf_max > max_I or wf_max < max_I * (1 - cur_tol):
        print(f'     Maximum WF Current = {wf_max:.3f}, Maximum Allowed Current = {max_I:.3f}, Maximum Lambda Value = {Lambda[wf.unconstrained_segments()].max():.3e}')
        # Update regularization matrix and reoptimize
        Lambda = update_Lambda(Lambda)
        res = optimize_wireframe_wrapper(Lambda)
        wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max().copy()

print('Exiting WF only optimization')
print(f'Maximum WF Current = {wf_max:.2f} A, Maximum Allowed Current = {max_I:.2f}')

########################################################################
######################## Combined Optimization #########################
########################################################################

# Now iterate between TFs and dipoles to refine optimization
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
    res = optimize_wireframe_wrapper(Lambda)
    wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max().copy()
    while field_flag and (wf_max > max_I * (1 + cur_tol) or wf_max < max_I * (1 - cur_tol)): 
        print(f'     Maximum WF Current = {wf_max:.3f}, Maximum Allowed Current = {max_I:.3f}, Maximum Lambda Value = {Lambda[wf.unconstrained_segments()].max():.3e}')
        Lambda = update_Lambda(Lambda)
        res = optimize_wireframe_wrapper(Lambda)
        wf_max = np.abs(wf.currents[wf.unconstrained_segments()]).max().copy()
    Jf = SquaredFlux(surf_plas, mf_tf + res['wframe_field'], definition='local')
    JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
    print(f'\n Post Iteration Squared Flux = {JF.J():4e}, % Difference = {(JF_i - JF.J()) / JF_i * 100:3f}%, Target % Difference = {JF_tol * 100}% \n')

end_time = time.time()  # End the timer
total_time = end_time - start_time  # Calculate the total time
JF_final = JF.J()
print(JF_preTF, JF_postTF, JF_final)
print('----- Dipole Optimization Results ----- ')
print(f'Total WF/TF Optimization Time: {total_time:.2f} seconds')
print(f'Squared Flux Percentage Between TF Only and Combined Optimization = {(JF_postTF - JF_final) / JF_postTF * 100:.2f}%')
print(f'Squared Flux Percentage Between Initial State and Final Optimization = {(JF_preTF - JF_final) / JF_preTF * 100:.2f}%')
print(f'Maximum WF Current = {wf_max:.1f}, Max/Min Allowed Current With {cur_tol * 100}% Tolerance = {max_I*(1+cur_tol):.1f}/{max_I*(1-cur_tol):.1f} \n')

########################################################################
########################### Post-Processing ############################
########################################################################

if field_flag:
    # Plot the lambda matrix array entries and the corresponding currents
    # (This can be a check to see if the initial value is ok, want a spread of lambdas)
    # Identify non-zero elements in wf.currents
    non_zero_indices = np.nonzero(wf.currents)[0]
    filtered_lambda = Lambda[non_zero_indices]
    filtered_currents = wf.currents[non_zero_indices]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(non_zero_indices, np.log10(filtered_lambda), 'o')
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('log10(lambda)')
    ax2.plot(non_zero_indices, np.log10(np.abs(filtered_currents)), 'o', color='orange')
    ax2.axhline(y=np.log10(max_I), color='r', linestyle='--', label=f'log max current = {np.log10(max_I):1f}')
    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('log10(|I|)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('lambda_and_currents.png', dpi=dpi)
    plt.clf()

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
print(f'   Surface-averaged |B| = {mean_abs_modBfinal:.3f}')

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
surf_wf.save('surf_wf.json')
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
lines = [f"Equilibrium file location: {input_QA} with s = {surf_s} and dof_scale = {surf_dof_scale} \n", 
         f"Equilibrium parameters: major radius = {major_radius:.3f}, minor radius = {minor_radius:.3f}, volume = {volume:.3f}, aspect ratio = {aspect_ratio:.3f} \n",
         f"On axis magnetic field = {field_on_axis} T \n",
         f"Maximum allowed dipole field = {max_dipole_field} [T] \n", \
         f"TF parameters: ntf = {n_tf}, num_fixed = {num_fixed}, radius = {TF_R1:.3f} [m], opt params = {tf_opt_params} \n", 
         f"Dipole parameters: nPhi = {win_nPhi}, nTheta = {win_nTheta}, min dimensions (phi, theta) = ({min_theta:.2f},{min_phi:.2f}) [in] \n",
         f"Vessel parameters: Axisymmetric? {axisymmetric}, major radius = {VV_R0}, minor radius = {VV_a} , average plasma-vessel distance = {wf_plas_offset:.4f} [m]\n",
         f"\n", 
         f"Optimization results \n",
         f"Pre TF Surface-averaged/Max |Bn|/|B| = {pre_TF_mean_abs_relBfinal_norm:.3e}/{pre_TF_relBfinal_norm_max:.3e} \n", 
         f"Post TF Surface-averaged/Max |Bn|/|B| = {post_TF_mean_abs_relBfinal_norm:.3e}/{post_TF_relBfinal_norm_max:.3e} \n",
         f"Final Surface-averaged/Max |Bn|/|B| = {final_mean_abs_relBfinal_norm:.3e}/{final_relBfinal_norm_max:.3e} \n",
         f'Squared Flux Percentage Decrease After TF Optimization = {(JF_preTF - JF_postTF) / JF_preTF * 100:.2f}% \n',
         f'Squared Flux Percentage Between TF Only and Combined Optimization = {(JF_postTF - JF_final) / JF_postTF * 100:.2f}% \n',
         f'Squared Flux Percentage Between Initial State and Final Optimization = {(JF_preTF - JF_final) / JF_preTF * 100:.2f}% \n',
         f"Final Surface-averaged |B| = {mean_abs_modBfinal:.3f} + \n",
         f"Maximum WF Current = {np.abs(wf.currents[wf.unconstrained_segments()]).max():.3e}, Maximum Allowed Current With {cur_tol * 100}% Tolerance = {max_I*(1+cur_tol):.3e} \n"]
 
with open("parameters.txt", "w") as file1:
    file1.writelines(lines)
