NOTE! This section of the repo is deprecated and for safekeeping for the future if necessary. These are script related to the wireframe approach written by Ken Hammond, but that I shifted away from due to various reasons, mainly flexibility is choosing the coil locations on the surface. This folder features example scripts of how to do this type of optimization:
- run_optimizer_..._.py sends off runs
- optimizer.py is the bulk optimization script in different iterations
- poincare/qs_error_wireframe.py do the poincare plot/qs error analysis on the wireframe outputs
I just copied these scripts into this folder, so the script paths are likely broken, but the contents can be useful for future reference. (JMH 02/10/2025)
---------------------------------------------------------------------------

This repository is set up for those wanting to do coil/equilibrium optimization for the Columbia REconfigurable eXperiment (C-REX) project

## Setting up the correct simsopt installation
You'll need to setup the correct simsopt branch, located here: https://github.com/jhalpern30/simsopt/tree/wireframe
Note that this is just Ken Hammond's wireframe branch but with a few modifications, mainly the specific TF coil class used for C-REX
You can then follow the instructions in the Paul group google drive, /Software installation/SIMSOPT (Orbit_resonance) Installation manual for OSX, replacing the line
git clone -b orbit_resonance https://github.com/hiddenSymmetries/simsopt.git
with
git clone -b wireframe https://github.com/jhalpern30/simsopt.git

## Using the optimization scripts
After following all the steps to install and run the correct branch of simsopt above, you can clone this repository
Most of the files in this repository should be commented well and self-explanatory. 
Those wanting to run stellarator coil optimization should use run_optimizer.py; set your input parameters here and then run the file. This updates the main script, optimizer_template, and runs it, producing the optimization outputs in the /outputs folder. 

## Equilibria
Current equilibria are located in the /equilibria folder. You can run with different types located in here. If you are primarily doing stage-I optimization, you can push new files here as well for testing/documentation. 