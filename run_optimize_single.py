"""
File: run_optimize_scan.py
Author: Jake Halpern
Last Edit Date: 02/2025
Description: This script sends off a single optimization run of windowpane coil currents on 
             an axisymmetric surface for a specified plasma equilibrium
"""
import os
import shutil
import numpy as np
from optimize import optimize

script_dir = os.path.dirname(os.path.abspath(__file__))
###### set this to wherever you'd like all the outputs from this script to go
run_dir = os.path.join(script_dir, '../outputs/20250204_CurvePlanarFourier_test')

### Simulation parameters ###
# Dipole parameters
fil_distance = 0.05 # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05 # distance between dipole panels between half field periods of the device [m]
dipole_radius = 0.035 # radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
# Vacuum Vessel
VV_a = 0.2500402157723215                    # minor radius of vacuum vessel (horizontal)
VV_b = 0.3382203712428585                    # minor radius of vacuum vessel (vertical)
VV_R0 = 0.9908912828999443                    # major radius of vacuum vessel
# Plasma Surface
plas_nPhi = 128; plas_nTheta=64    # plasma surface quad points
surf_s = 1                  # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1      # used to scale the dofs of the surface
eq_name = 'eq_opt_circle_constraint_6_resolution_3_01_19'  # name of the wout file from vmec
eq_dir = os.path.join(script_dir, 'equilibria') # equilibria should be in this folder
# TF coils parameters (radius current set as 1.6 * VV_b)
n_tf = 10                       # number of TF coils per half field period
num_fixed = n_tf                  # number of TF coil currents to fix during combined optimization
field_on_axis = 1.0            # on-axis magnetic field (Tesla)
TF_R0 = VV_R0
TF_a = 0.40006434523571444
TF_b = 0.5411525939885736
# Optimization parameters
definition = "local"         # definition of squared flux, either local, normalized, or quadratic flux
precomputed = True           # if true, will use precomputed Biot-Savart with scaled currents during optimization
MAXITER = 2500               # Number of iterations to perform:
CURRENT_THRESHOLD = 5E5      # Current penality threshold and weight
CURRENT_WEIGHT = 1E-12       # make sure weight is appropriate for the current threshold
verbose=True
# Figure parameters
dpi = 100; titlefontsize = 18; axisfontsize = 16; legendfontsize = 14; ticklabelfontsize = 14; cbarfontsize = 18

extra = ""
# only add these parameters to output file if they are unusual runs, can add more as needed
if VV_a != 0.25:
    extra = extra + f"_VVa_{VV_a}"
if VV_R0 != 1.0:
    extra = extra + f"_VV_R0_{VV_R0}"
if VV_a != VV_b:
    extra = extra + f"_ellipticalVV"
if dpi != 100:
    extra = extra + f"_for_poster"

# Name the file the next optimization number
num = str(input("Run number for directory naming: "))
if num == "":
    if not os.path.exists(os.path.join(run_dir, f'{eq_name}')) or not os.listdir(os.path.join(run_dir, f'{eq_name}')):
        num = 1
    else:
        num = max(int(f.split('_')[0]) for f in os.listdir(os.path.join(run_dir, f'{eq_name}')) if f.split('_')[0].isdigit()) + 1
else:
    num = int(num)
extra = extra + str(input("Anything extra to add to pathname? If none, just press enter: "))

run_dir = os.path.join(run_dir, f'{eq_name}/{num:02}_ntf{n_tf}_diprad_{dipole_radius}{extra}/')

###### Change directory
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

# Copy all helpful files to document the optimization here
shutil.copy(os.path.join(script_dir, "optimize.py"), os.path.join(run_dir, "optimize.py"), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'helper_functions.py'), os.path.join(run_dir, 'helper_functions.py'), follow_symlinks=True)

optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            plas_nPhi=plas_nPhi, plas_nTheta=plas_nTheta, surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a, TF_b=TF_b,   # TF parameters
            definition=definition, precomputed=precomputed, MAXITER=MAXITER, CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            dpi=dpi, titlefontsize=titlefontsize, axisfontsize=axisfontsize, legendfontsize=legendfontsize, ticklabelfontsize=ticklabelfontsize, cbarfontsize=cbarfontsize,
            output_dir=run_dir, verbose=verbose)