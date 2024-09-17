#!/usr/bin/env python
# coding: utf-8

import numpy as np
from simsopt._core.optimizable import load
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.wireframefield import WireframeField, enclosed_current
from simsopt.geo.wireframe import ToroidalWireframe, windowpane_wireframe
import time
import os
from create_surface import *

# these need to be set based on the directory where the outputs you want to process are
eq_name = 'wout_NAS_n2n4_AR6.2.03'  # this needs to be set based on the run
out_dir = 'Bt1.0_Bd0.5_ntf3_np6_nt11_axisym_True'

# load the equilibrium surface
script_dir = os.path.dirname(os.path.abspath(__file__))
eq_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equilibria')
eq_name_full = os.path.join(eq_dir, eq_name + '.nc')
print(f'\nGenerating Poincare Plot for Equilibrium {eq_name_full} with Outputs in {out_dir}')
plas_nPhi = 128
plas_nTheta = 64
dof_scale = 0.15
R0 = 1
surf_s = 1
# creating the plasma surface will be hardcoded for now
surf_plas = create_surface(eq_name_full=eq_name_full, surf_range='full torus', plas_nPhi=plas_nPhi, plas_nTheta=plas_nTheta, surf_s=surf_s, dof_scale=dof_scale, R0=R0)
full_dir = os.path.join(script_dir, 'outputs', eq_name, out_dir)

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
bs = bs_wf + bs_tf

from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data

nfieldlines = 100 # Number of field lines for integration 
tmax_fl = 10000 # Maximum toroidal angle for integration
tol = 1e-8 # Tolerance for field line integration
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                   # on a grid for the magnetic field evaluation
nr = 25 # Number of radial points for interpolation
nphi = 12 # Number of toroidal angle points for interpolation
nz = 12 # Number of vertical points for interpolation
degree = 4 # Degree for interpolation

# Extend surface, since we want to look at field lines beyond it
surf_extended = create_surface(eq_name_full=eq_name_full, surf_range='full torus', plas_nPhi=plas_nPhi, plas_nTheta=plas_nTheta, surf_s=surf_s, dof_scale=dof_scale, R0=R0)
surf_extended.extend_via_normal(0.05)

# Use extended surface to determine initial conditions
gamma = surf_extended.gamma()
R = np.sqrt(gamma[:,:,0]**2 + gamma[:,:,1]**2)
Z = gamma[:,:,2]

nfp = surf_plas.nfp

Rmin = np.min(R)
Rmax = np.max(R)
Zmax = np.max(Z)

# The parameter h sets the grid size for the classifier, 
# and p is the order. These parameters are not too critical
# to the Poincare calculation. 
print("Creating Extended Surface Classifier") 
sc_fieldline = SurfaceClassifier(surf_extended, h=0.02, p=2)

def trace_fieldlines(bfield):
    # Set up initial conditions 
    R0 = np.linspace(Rmin, Rmax, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(full_dir, 'poincare_fieldline.png'), dpi=300,surf=surf_plas,mark_lost=False)
    return fieldlines_phi_hits

rrange = (Rmin, Rmax, nr)
phirange = (0, 2*np.pi/nfp, nphi)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, nz)

start_time = time.time()
if interpolate:
    print("Interpolating Field")
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    bsh.set_points(surf_plas.gamma().reshape((-1, 3)))
    bs.set_points(surf_plas.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)))
else:
    bsh = bs
end_time = time.time()
print(f"Interpolation Time = {end_time - start_time} s")

print("Tracing Fieldlines")
start_time = time.time()
hits = trace_fieldlines(bsh)
end_time = time.time()
print(f"Poincare Plot Time = {end_time - start_time} s")
