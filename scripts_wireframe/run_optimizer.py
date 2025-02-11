import os
import shutil
import numpy as np

# Run this script as python run_optimizer.py 

##### set this to wherever this script is
script_dir = os.path.dirname(os.path.abspath(__file__))

###### set this to wherever you'd like all the outputs from this script to go
run_dir = os.path.join(script_dir, 'outputs')

# this is the name of the script you're going to replace the parameters for and run
script_template_name = 'optimizer_template_updated.py' ### use this for HBT calcs (no TF dofs)
#script_template_name = 'optimizer_template.py' ### Python script name in current folder.
script_name = 'optimizer.py' ### what you want the script saved as in outputs folder

### Simulation parameters ###
# TF coils parameters
n_tf = 5                       # number of TF coils per half field period
num_fixed = 1                  # number of TF coil currents to fix
full_TF_scan = True           # keep this at False unless you have a reason to see the whole TF scan
vc_flag = False                # Use finite beta when running HBT_optimizer_template
field_on_axis = 1.0            # B field at magnetic axis in Tesla
fixed_geo = False               # fix the TF geometry
TF_radii = [0.9, 1, 1.1]         # range of TF coil radii to optimize over, put in increasing order
    
# Windowpane parameters
axisymmetric = True            # use axisymmetric vessel for dipoles
max_dipole_field = 1.0         # [T], set by engineering constraints
win_nPhi = 7                    # Number of windows per half period, toroidal dimension
win_nTheta = 15               # Number of windows, poloidal dimension
win_size = 8                   # Number of grid cells/window, both dimensions
win_gap = 2                    # Number of grid cells between adjacent windows

# Vacuum Vessel
VV_a = 0.75                   # minor radius of vacuum vessel
VV_R0 = 1.90                   # major radius of vacuum vessel

# Plasma Surface
plas_nPhi = 64                 # toroidal points on plasma surface
plas_nTheta = 64               # poloidal points on plasma surface
surf_s = 1.0                   # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 0.50          # used to scale the dofs of the surface
eq_name = 'wout_NAS_n2_AR4.03'  # name of the wout file from vmec
#eq_name = 'wout_hbt_finite_beta_000_000000'   # use this for HBT calcs
eq_dir = os.path.join(script_dir, 'equilibria') # equilibria should be in this folder

# Change figure specs (i.e. increase dpi, font sizes for posters)
dpi = 100; titlefontsize = 18; axisfontsize = 16; legendfontsize = 14; ticklabelfontsize = 14; cbarfontsize = 18

extra = str(input("Anything extra to add to pathname? If none, just press enter: "))
# only add these parameters to output file if they are unusual runs, can add more as needed
if VV_a != 0.25:
    extra = extra + f"_VVa_{VV_a}"
if VV_R0 != 1.0:
    extra = extra + f"VV_R0_{VV_R0}"
if fixed_geo != False:
    extra = extra + f"_fixedTFs"
if axisymmetric != True:
    extra = extra + f"_nonaxisym"
if dpi != 100:
    extra = extra + f"_for_poster"
    
run_dir = os.path.join(run_dir, f'{eq_name}/Bt{field_on_axis}_Bd{max_dipole_field}_ntf{n_tf}_np{win_nPhi}_nt{win_nTheta}{extra}/')

###### Change directory
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

shutil.copy(os.path.join(script_dir, script_template_name), os.path.join(run_dir, script_name), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'create_surface.py'), os.path.join(run_dir, 'create_surface.py'), follow_symlinks=True)


##### Replace parameters in python file
# Read the template file
with open(script_name, 'r') as file:
    filedata = file.read()

# Replace the target strings
filedata = filedata.replace('AXISYM_VAL', str(axisymmetric))
filedata = filedata.replace('BD_VAL', str(max_dipole_field))
filedata = filedata.replace('DIP_NPHI_VAL', str(win_nPhi))
filedata = filedata.replace('DIP_NTHETA_VAL', str(win_nTheta))
filedata = filedata.replace('DIP_SIZE_VAL', str(win_size))
filedata = filedata.replace('DIP_GAP_VAL', str(win_gap))
filedata = filedata.replace('DIP_MINOR_RAD_VAL', str(VV_a))
filedata = filedata.replace('DIP_MAJOR_RAD_VAL', str(VV_R0))
filedata = filedata.replace('SURF_NPHI_VAL', str(plas_nPhi))
filedata = filedata.replace('SURF_NTHETA_VAL', str(plas_nTheta))
filedata = filedata.replace('SURF_S_VAL', str(surf_s))
filedata = filedata.replace('SURF_DOF_SCALE_VAL', str(surf_dof_scale))
filedata = filedata.replace('EQ_NAME_VAL', f"'{eq_name}'")
filedata = filedata.replace('EQ_DIR', f"'{eq_dir}'")
filedata = filedata.replace('NTF_VAL', str(n_tf))
filedata = filedata.replace('NUM_FIXED_VAL', str(num_fixed))
filedata = filedata.replace('FIXED_GEO_VAL', str(fixed_geo))
filedata = filedata.replace('FULL_TF_SCAN_VAL', str(full_TF_scan))
filedata = filedata.replace('TF_RADII', str(TF_radii))
filedata = filedata.replace('VC_FLAG', str(vc_flag))
filedata = filedata.replace('BT_VAL', str(field_on_axis))
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
os.system(f'python -W "ignore" {script_name}')
