import os
import shutil
import numpy as np

# Run this script as python run_optimizer.py 

##### set this to wherever this script is
script_dir = os.path.dirname(os.path.abspath(__file__))

###### set this to wherever you'd like all the outputs from this script to go
run_dir = os.path.join(script_dir, '../outputs/20250109_filaments_test')

# this is the name of the script you're going to replace the parameters for and run
script_template_name = 'wp_coils_w_BS_speedup.py' ### use this for HBT calcs (no TF dofs)
#script_template_name = 'optimizer_template.py' ### Python script name in current folder.
script_name = 'optimizer.py' ### what you want the script saved as in outputs folder

### Simulation parameters ###
# Dipole parameters
fil_distance = 0.05 # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05 # distance between dipole panels between half field periods of the device [m]
dipole_radius = 0.05 # radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
# Vacuum Vessel
VV_a = 0.3                    # minor radius of vacuum vessel (horizontal)
VV_b = 0.3                    # minor radius of vacuum vessel (vertical)
VV_R0 = 1.0                  # major radius of vacuum vessel
# Plasma Surface
plas_nPhi = 64                 # toroidal points on plasma surface
plas_nTheta = 64               # poloidal points on plasma surface
surf_s = 1.0                   # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1      # used to scale the dofs of the surface
eq_name = 'wout_nfp2ginsburg_000_003186'  # name of the wout file from vmec
#eq_name = 'wout_hbt_finite_beta_000_000000'   # use this for HBT calcs
eq_dir = os.path.join(script_dir, 'equilibria') # equilibria should be in this folder
# TF coils parameters (radius current set as 1.6 * VV_b)
n_tf = 5                       # number of TF coils per half field period
num_fixed = 1                  # number of TF coil currents to fix during combined optimization
field_on_axis = 1.0            # on-axis magnetic field (Tesla)
# Optimization parameters
local = True                 # use local definition of quadratic flux
normalized = False           # use normalized definition of quadratic flux
MAXITER = 1000               # Number of iterations to perform:
CURRENT_THRESHOLD = 1E7      # Current penality threshold and weight
CURRENT_WEIGHT = 1E-5
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

# put all necessary files to run the optimization here
shutil.copy(os.path.join(script_dir, script_template_name), os.path.join(run_dir, script_name), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'create_surface.py'), os.path.join(run_dir, 'create_surface.py'), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'helper_functions.py'), os.path.join(run_dir, 'helper_functions.py'), follow_symlinks=True)

##### Replace parameters in python file
# Read the template file
with open(script_name, 'r') as file:
    filedata = file.read()

# Replace the target strings
filedata = filedata.replace('FIL_DIST', str(fil_distance))
filedata = filedata.replace('HALF_PER_DIST', str(half_per_distance))
filedata = filedata.replace('DIP_RAD', str(dipole_radius))
filedata = filedata.replace('VVA_VAL', str(VV_a))
filedata = filedata.replace('VVB_VAL', str(VV_b))
filedata = filedata.replace('VVR0_VAL', str(VV_R0))
filedata = filedata.replace('SURF_NPHI_VAL', str(plas_nPhi))
filedata = filedata.replace('SURF_NTHETA_VAL', str(plas_nTheta))
filedata = filedata.replace('SURF_S_VAL', str(surf_s))
filedata = filedata.replace('SURF_DOF_SCALE_VAL', str(surf_dof_scale))
filedata = filedata.replace('EQ_DIR', f"'{eq_dir}'")
filedata = filedata.replace('EQ_NAME_VAL', f"'{eq_name}'")
filedata = filedata.replace('NTF_VAL', str(n_tf))
filedata = filedata.replace('NUM_FIXED_VAL', str(num_fixed))
filedata = filedata.replace('BT_VAL', str(field_on_axis))
filedata = filedata.replace('LOCAL_VAL', str(local))
filedata = filedata.replace('NORMALIZED_VAL', str(normalized))
filedata = filedata.replace('MAX_ITER_VAL', str(MAXITER))
filedata = filedata.replace('CUR_THRESH_VAL', str(CURRENT_THRESHOLD))
filedata = filedata.replace('CUR_WEIGHT_VAL', str(CURRENT_WEIGHT))
filedata = filedata.replace('DPI_VAL', str(dpi))
filedata = filedata.replace('TITLE_SIZE_VAL', str(titlefontsize))
filedata = filedata.replace('AXIS_SIZE_VAL', str(axisfontsize))
filedata = filedata.replace('LEGEND_SIZE_VAL', str(legendfontsize))
filedata = filedata.replace('TICK_SIZE_VAL', str(ticklabelfontsize))
filedata = filedata.replace('CBAR_SIZE_VAL', str(cbarfontsize))

# Write the file out again
with open(script_name, 'w') as file:
    file.write(filedata)

# Execute the modified script locally
os.system(f'python -u -W "ignore" {script_name} | tee script_output.txt')
