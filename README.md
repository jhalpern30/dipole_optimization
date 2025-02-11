## About this repository
This repository is set up with the essential scripts necessary for optimizing windowpane (dipole) coils. 
The majority of optimization scripts are directly related to the coil optimization for Columbia's
on campus hybrid experiment, (i.e. axisymmetric shaping coil sets with no TF geometry optimization); 
however, I included an example as used in the windowpane study for Antoine Baillod's port optimization
paper - see windowpanes_on_generalized_surf.py. 

## How to run the code
These scripts can all be run from the master simsopt branch, as they make use of functionality built-in
of the CurvePlanarFourier objects. The main focus of these scripts is how to use the geometric dofs, i.e.
the coil center and rotation quaternion, to orient planar coils on a given winding surface and then optimize their currents.

If you have any questions or find any bugs, please reach out to me at jmh2363@columbia.edu