# imports
import numpy as np
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
import fabio
import lmfit
import argparse
import time

from tools.comparison import detectormaker_fitting, voxelgridmaker_fitting, slabmaker_fitting, rebin_qmap_and_mask, plot_linecuts
from tools.comparison import mirror_qmap_positive_qxy_only, trim_sim_data, evaluate_fit, plot_fit, shift_peak, optimize_scale_offset
from tools.utilities import  parse_config_file, str_to_bool, save_config_to_txt, create_shared_array


# Define the model function for scaling and offsetting sim_comp_map
def fullfit_model(params, fixed_slab_params, fixed_voxelgrid_params, fixed_detectormaker_params, fixed_exp_params):
        x_size = params['x_size']
        y_size = params['y_size']
        z_size = params['z_size']

        input_filepath, a, b, c, alpha, beta, gamma = fixed_slab_params
        r_voxel_size, q_voxel_size, max_q, energy, fill_bkg, smooth = fixed_voxelgrid_params
        num_pixels, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path = fixed_detectormaker_params
        rebin_map, rebin_mask, exp_qxy, exp_qz, pad_width, pad_range = fixed_exp_params

        #generate simulated GIWAXS
        coords_slab, elements_slab = slabmaker_fitting(input_filepath, x_size, y_size, z_size, a, b, c, alpha, beta, gamma)
        iq, qx, qy, qz = voxelgridmaker_fitting(coords_slab, elements_slab, r_voxel_size, q_voxel_size, max_q, energy, num_cpus=None, fill_bkg=fill_bkg, smooth=smooth)
        det_sum, det_h, det_v = detectormaker_fitting(iq, qx, qy, qz, num_pixels, max_q, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path, mirror=True)

        #trim simulated data to bounds of experimental data
        sim_int_trim, det_h_trim, det_v_trim = trim_sim_data(det_sum, det_h, det_v, exp_qxy, exp_qz) 

        sim_int_trim_pad = shift_peak(sim_int_trim.copy(), det_h_trim, det_v_trim, pad_width, pad_range)

        # Optimize scale and offset. Numerical approach to linear problem is very fast, no iterations
        scale, offset = optimize_scale_offset(sim_int_trim_pad, rebin_map, rebin_mask)
           
        # Apply scale and offset to sim_comp_map
        scaled_map = scale * sim_int_trim_pad + offset
        
        # Calculate residuals only where mask is 0
        residuals = (rebin_map - scaled_map) * (rebin_mask == 0)
        return residuals.ravel()  # Flatten to a 1D array for lmfit

def main(config):
    #Slabmaker variables
    input_filepath = config.get('input_filepath') 
    x_size_init = float(config.get('x_size_init', 150))
    y_size_init = float(config.get('y_size_init', 150))
    z_size_init = float(config.get('z_size_init', 150))
    fit_x = str_to_bool(config.get('fit_x', 'False'))
    fit_y = str_to_bool(config.get('fit_y', 'False'))
    fit_z = str_to_bool(config.get('fit_z', 'False'))
    a = float(config.get('a'))
    b = float(config.get('b'))
    c = float(config.get('c'))
    alpha = float(config.get('alpha'))
    beta = float(config.get('beta'))
    gamma = float(config.get('gamma'))

    # voxelgridmaker variables
    r_voxel_size = float(config.get('r_voxel_size', 0.3))
    q_voxel_size =  float(config.get('q_voxel_size', 0.04))
    max_q = float(config.get('max_q', 2.5))
    energy = float(config.get('energy', 10000))
    fill_bkg = str_to_bool(config.get('fill_bkg', 'False'))
    smooth = int(config.get('smooth', 0))

    #detectormaker variables
    num_pixels = int(config.get('num_pixels', max_q/q_voxel_size))
    angle_init_val1 = float(config.get('angle_init_val1', 0))
    angle_init_val2 = float(config.get('angle_init_val2', 0))
    angle_init_val3 = float(config.get('angle_init_val3', 0))
    angle_init_ax1 = config.get('angle_init_ax1', 'None')
    angle_init_ax2 = config.get('angle_init_ax2', 'None')
    angle_init_ax3 = config.get('angle_init_ax3', 'None')
    psi_start = float(config.get('psi_start'))
    psi_end = float(config.get('psi_end'))
    psi_num = int(config.get('psi_num'))
    psi_weights_path = config.get('psi_weights_path', None)
    phi_start = float(config.get('phi_start'))
    phi_end = float(config.get('phi_end'))
    phi_num = int(config.get('phi_num'))
    phi_weights_path = config.get('phi_weights_path', None)
    theta_start = float(config.get('theta_start'))
    theta_end = float(config.get('theta_end'))
    theta_num = int(config.get('theta_num'))
    theta_weights_path = config.get('theta_weights_path', None)
    mirror = str_to_bool(config.get('mirror', 'False'))
    angle_init_vals = (angle_init_val1, angle_init_val2, angle_init_val3)
    angle_init_axs = (angle_init_ax1, angle_init_ax2, angle_init_ax3)
    psis = np.linspace(psi_start, psi_end, psi_num)
    phis =np.linspace(phi_start, phi_end, phi_num)
    thetas = np.linspace(theta_start, theta_end, theta_num)

    #experimental comparison
    mask_path = config.get('mask_path')
    img_path = config.get('img_path')
    qxy_path = config.get('qxy_path')
    qz_path = config.get('qz_path')

    pad_width = float(config.get('pad_width'))
    pad_range_min = float(config.get('pad_range_min'))
    pad_range_max = float(config.get('pad_range_max'))
    pad_range = (pad_range_min, pad_range_max)

    #save directory
    save_folder = config.get('save_folder')
    os.makedirs(save_folder, exist_ok=True)
    num_evals = int(config.get('num_evals'))


    #sample calculation 
    coords_slab, elements_slab = slabmaker_fitting(input_filepath, x_size_init, y_size_init, z_size_init, a, b, c, alpha, beta, gamma)
    iq, qx, qy, qz = voxelgridmaker_fitting(coords_slab, elements_slab, r_voxel_size, q_voxel_size, max_q, energy, num_cpus=None, fill_bkg=fill_bkg, smooth=smooth)
    det_sum, det_h, det_v = detectormaker_fitting(iq, qx, qy, qz, num_pixels, max_q, angle_init_vals, 
                                                  angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path, mirror=mirror)
    
    #load experimental data, fitting mask
    exp_img = fabio.open(img_path).data
    exp_qxy = np.loadtxt(qxy_path)
    exp_qz = np.loadtxt(qz_path)
    mask = np.load(mask_path)
    #fix detector gaps, only take positive qxy
    exp_img, exp_qxy, exp_qz = mirror_qmap_positive_qxy_only(exp_img, exp_qxy, exp_qz)
    #trim simulated data to bounds of experimental data
    sim_int_trim, det_h_trim, det_v_trim = trim_sim_data(det_sum, det_h, det_v, exp_qxy, exp_qz)
    #rebin experimental data to simulated data resolution
    rebin_map, rebin_mask = rebin_qmap_and_mask(det_h_trim, det_v_trim, exp_img, exp_qxy, exp_qz, mask)
    # mask rebinned experimental data
    rebin_map[rebin_mask==1]=0

    # Initialize variable parameters
    params = lmfit.Parameters()
    params.add('x_size', value=x_size_init, min=25, max=x_size_init*3, vary=fit_x) 
    params.add('y_size', value=y_size_init, min=25, max=y_size_init*3, vary=fit_y)  
    params.add('z_size', value=z_size_init, min=25, max=z_size_init*3, vary=fit_z)

    fixed_slab_params = (input_filepath, a, b, c, alpha, beta, gamma)
    fixed_voxelgrid_params = (r_voxel_size, q_voxel_size, max_q, energy, fill_bkg, smooth)
    fixed_detectormaker_params = (num_pixels, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path)
    fixed_exp_params = (rebin_map, rebin_mask, exp_qxy, exp_qz, pad_width, pad_range)

    # Perform the fit using lmfit's minimize function
    result = lmfit.minimize(fullfit_model, params, args=(fixed_slab_params, fixed_voxelgrid_params, fixed_detectormaker_params, fixed_exp_params), method='nelder', max_nfev=num_evals)

    # Access the optimized scale and offset
    best_x_size = result.params['x_size'].value
    best_y_size = result.params['y_size'].value
    best_z_size = result.params['z_size'].value

    best_params = (best_x_size, best_y_size, best_z_size)

    rebin_map, sim_comp_map, diff_map = evaluate_fit(best_params,  fixed_slab_params, fixed_voxelgrid_params, fixed_detectormaker_params, fixed_exp_params)

    savepath = f'{save_folder}/fit_result.png'
    red_chi2 = result.redchi
    suptitle = f'Reduced Chi2 = {red_chi2:0.0f}'
    plot_fit(savepath, rebin_map, sim_comp_map, diff_map, det_h_trim, det_v_trim, max_q, suptitle)

    savepath = f'{save_folder}/fit_result_linecut.png'
    plot_linecuts(savepath, rebin_map, sim_comp_map, det_h_trim, det_v_trim, suptitle)

    savepath = f'{save_folder}/fit_result.txt'
    # Save the fit report to a .txt file
    with open(savepath, 'w') as file:
        file.write(lmfit.fit_report(result))

    save_config_to_txt(config, f'{save_folder}/fit_config.txt')

if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser(description="Process a configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    config_path = args.config
    config = parse_config_file(config_path)
    main(config)
    end = time.time()
    runtime = end-start
    print(f'\nTotal Time: {str(runtime)}')



