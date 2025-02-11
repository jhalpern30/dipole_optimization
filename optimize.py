from simsopt.geo import create_equally_spaced_curves, curves_to_vtk, SurfaceRZFourier
from simsopt.field import Current, ScaledCurrent, Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.objectives import SquaredFlux
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from helper_functions import *

def optimize(fil_distance, half_per_distance, dipole_radius, # dipole parameters
             VV_a, VV_b, VV_R0,  # vessel parameters
             plas_nPhi, plas_nTheta, surf_s, surf_dof_scale, eq_dir, eq_name,  # equilibrium parameters
             ntf, num_fixed, field_on_axis, TF_R0, TF_a, TF_b,   # TF parameters
             definition, precomputed, MAXITER, CURRENT_THRESHOLD, CURRENT_WEIGHT, 
             dpi, titlefontsize, axisfontsize, legendfontsize, ticklabelfontsize, cbarfontsize,
             output_dir, verbose=False):
    
    # Create the plasma surface
    eq_name_full = os.path.join(eq_dir, eq_name + '.nc')
    surf = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)
    surf.set_dofs(surf_dof_scale*surf.get_dofs())
    # Create a surface representing the vacuum vessel that dipoles will be placed on
    VV = SurfaceRZFourier(nfp=surf.nfp)
    VV.set_rc(0,0,VV_R0); VV.set_rc(1,0,VV_a); VV.set_zs(1,0,VV_b)
    plot_cross_section(surf, VV, output_dir, axisfontsize, legendfontsize, ticklabelfontsize, dpi)

    # Initialize TF Coils
    # order is the number of fourier moments to include - we don't need a large value
    # 2 should be fine for elliptical
    numquadpoints=64 # number of points per coil - this should be fine for both types of coils
    base_tf_curves = create_equally_spaced_curves(
        ncurves=ntf,
        nfp=surf.nfp,
        stellsym=surf.stellsym,
        R0=TF_R0,
        R1=TF_a, # HBT is 0.4m/0.25m = 1.6, so keep this for now? 
        order=2, 
        numquadpoints=numquadpoints)
    # add this for elliptical TF coils - keep same ellipticity as VV
    for c in base_tf_curves:
        c.set("zs(1)", -TF_b) # see create_equally_spaced_curves doc for minus sign info
    # We define the currents with 1A, so that our dof is order unity. We
    # then scale the current by a scale_factor of 1E6.
    base_tf_currents = [Current(1) for c in base_tf_curves]
    # toroidal solenoid approximation - B_T = mu0 * I * N / L = mu0 * I * N / L
    # therefore, TF current = B_T * 2 * pi * R0 / mu0 / (2 * nfp * n_tf), n_tf = # per half field period!
    mu0 = 4.0 * np.pi * 1e-7
    scale_factor = 2.0*np.pi*surf.get_rc(0,0)*field_on_axis/mu0/(2*ntf*surf.nfp)
    base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves, base_tf_currents)]
    # Initialize dipoles 
    base_wp_curves, base_wp_currents = generate_windowpane_array(winding_surface=VV, inboard_radius=dipole_radius, wp_fil_spacing=fil_distance, 
                                                                 half_per_spacing=half_per_distance, wp_n=4, numquadpoints=numquadpoints, order=12, verbose=verbose)
    nwptot = len(base_wp_curves * 2 * surf.nfp)
    wp_scale_factor = 1E2 # Initialize coils at 100kA (reasonable guess for 1T field)
    base_wp_coils = [Coil(curve, ScaledCurrent(current, wp_scale_factor)) for curve, current in zip(base_wp_curves, base_wp_currents)]

    # Optimize currents
    bs = optimize_windowpane_currents(base_wp_coils=base_wp_coils, base_tf_coils=base_tf_coils, surf_plasma=surf, 
                                      definition=definition, precomputed=precomputed,
                                      current_threshold=CURRENT_THRESHOLD, current_weight=CURRENT_WEIGHT, 
                                      maxiter=MAXITER, num_fixed=num_fixed, verbose=verbose)
    
    # Post-processing
    # get final Bnormal
    relBfinal_norm, final_mean_abs_relBfinal_norm, final_relBfinal_norm_max = plot_relBfinal_norm_modB(
        bs, surf, output_dir, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Final')
    Jf = SquaredFlux(surf, bs, definition=definition)
    # plots currents on surface
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(base_wp_coils, VV)
    plot_coil_currents_on_theta_phi_grid(wp_currents_phis_thetas, output_dir, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi)
    
    # Prep coil data
    tf_curves = apply_symmetries_to_curves(base_tf_curves, surf.nfp, surf.stellsym)
    wp_curves = apply_symmetries_to_curves(base_wp_curves, surf.nfp, surf.stellsym)
    base_tf_currents_scaled = [c.current for c in base_tf_coils]
    tf_currents = apply_symmetries_to_currents(base_tf_currents_scaled, surf.nfp, surf.stellsym)
    tf_currents = [c.get_value() for c in tf_currents] # just get the amplitudes
    base_wp_currents_scaled = [c.current for c in base_wp_coils]
    wp_currents = apply_symmetries_to_currents(base_wp_currents_scaled, surf.nfp, surf.stellsym)
    wp_currents = [c.get_value() for c in wp_currents] 
    # Save various files
    VV.to_vtk(os.path.join(output_dir, 'vacuum_vessel'))
    curves_to_vtk(tf_curves, os.path.join(output_dir, 'tf_coils'), close=True, extra_data={'currents':np.array([current for current in tf_currents for _ in range(numquadpoints+1)])})
    curves_to_vtk(wp_curves, os.path.join(output_dir, 'wp_coils'), close=True, extra_data={'currents':np.array([current for current in wp_currents for _ in range(numquadpoints+1)])})
    bs.save(os.path.join(output_dir, 'bs_opt.json'));
    # Save the BdotN on the full torus surface
    surf_full = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range='full torus', nphi=2*surf.nfp*plas_nPhi, ntheta=plas_nTheta)
    bs.set_points(surf_full.gamma().reshape(-1,3))
    Bdotn = np.sum(bs.B().reshape(surf_full.unitnormal().shape) * surf_full.unitnormal(), axis=2)
    modB = bs.AbsB().reshape((2*surf.nfp*plas_nPhi,plas_nTheta))
    BdotN_norm = Bdotn / modB
    surf_full.to_vtk(os.path.join(output_dir, 'surf_full'), extra_data={"B_N": BdotN_norm[:, :, None]})
    bs.set_points(surf.gamma().reshape(-1,3)) # have to set points back for Jf.J in results section

    results = {
            # input parameters
            "filament_distance": fil_distance,             
            "half_period_distance": half_per_distance, 
            "inboard_radius": dipole_radius,
            "VV_a": VV_a, 
            "VV_b": VV_b, 
            "VV_R0": VV_R0,
            "plas_nPhi": plas_nPhi, 
            "plas_nTheta": plas_nTheta, 
            "surf_s": surf_s, 
            "surf_dof_scale": surf_dof_scale, 
            "eq_dir": eq_dir, 
            "eq_name": eq_name,  
            "num_tf_coils": ntf, 
            "num_fixed_current_tfs": num_fixed, 
            "field_on_axis": field_on_axis,   
            "squared_flux_def": definition, 
            "max_iterations": MAXITER, 
            "current_threshold": CURRENT_THRESHOLD, 
            "current_weight": CURRENT_WEIGHT,             
            # derived quantities
            "surf_nfp": surf.nfp,
            "surf_major_radius": surf.major_radius(),
            "surf_minor_radius": surf.minor_radius(),
            "surf_aspect_ratio": surf.aspect_ratio(),
            "surf_volume": surf.volume(),
            "initial_tf_current": scale_factor,
            "initial_wp_current": wp_scale_factor,
            "num_wps": nwptot, 
            "TF_a": VV_a*1.6, # need to modify this if we make this an independent variable
            "TF_b": VV_b*1.6,
            # optimization results
            "max_tf_current": np.max(np.abs(np.array(tf_currents))),
            "min_tf_current": np.min(np.abs(np.array(tf_currents))),
            "max_wp_current": np.max(np.abs(np.array(wp_currents))),
            "min_wp_current": np.min(np.abs(np.array(wp_currents))),
            "final_squared_flux": Jf.J(), 
            "final_mean_abs_relBfinal_norm": final_mean_abs_relBfinal_norm,
            "final_relBfinal_norm_max": final_relBfinal_norm_max, 
            "peak_wp_field": np.max(np.abs(np.array(wp_currents))) * mu0 / 2 / dipole_radius,
            "MA_meters": get_total_amp_meters(base_tf_coils, base_wp_coils, VV) / 1e6
        }

    with open(os.path.join(output_dir, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)