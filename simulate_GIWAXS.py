# imports
import numpy as np
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
import fabio
import argparse
import time
import glob

from tools.comparison import detectormaker_fitting, voxelgridmaker_fitting, slabmaker_fitting, rebin_qmap_and_mask, plot_linecuts
from tools.comparison import mirror_qmap_positive_qxy_only, trim_sim_data, plot_fit, shift_peak, optimize_scale_offset
from tools.utilities import  parse_config_file, str_to_bool, save_config_to_txt, create_shared_array, load_pdb_cell_params, write_xyz


def main(config):
    #Slabmaker variables
    input_folder = config.get('input_folder', None)
    input_path = config.get('input_filepath', None)
    filetype = config.get('filetype', None)
    x_size = float(config.get('x_size', 0))
    y_size = float(config.get('y_size', 0))
    z_size = float(config.get('z_size', 0))
    a = float(config.get('a', 0))
    b = float(config.get('b', 0))
    c = float(config.get('c', 0))
    alpha = float(config.get('alpha', 0))
    beta = float(config.get('beta', 0))
    gamma = float(config.get('gamma', 0))

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
    mask_path = config.get('mask_path', None)
    img_path = config.get('img_path', None)
    qxy_path = config.get('qxy_path', None)
    qz_path = config.get('qz_path', None)
    fit_scale_offset = str_to_bool(config.get('fit_scale_offset', 'False'))

    pad_width = float(config.get('pad_width', 0))
    pad_range_min = float(config.get('pad_range_min', 0))
    pad_range_max = float(config.get('pad_range_max', 0))
    pad_range = (pad_range_min, pad_range_max)

    #save directory
    save_folder = config.get('save_folder')
    os.makedirs(save_folder, exist_ok=True)

    if input_folder:
        if not filetype:
            print(filetype)
            raise Exception('filetype must be specified')
        input_paths = glob.glob(f'{input_folder}/*{filetype}')
        gen_names = [os.path.splitext(os.path.basename(path))[0] for path in input_paths]
    elif input_path:
        input_paths = [input_path]
        gen_names = [os.path.splitext(os.path.basename(path))[0] for path in input_paths]
    else:
        raise Exception('Either input_folder or input_path must be specified')

    for i, path in enumerate(input_paths):
        # Load file based on extension
        if path.lower().endswith('.xyz'):
            a2, b2, c2, alpha2, beta2, gamma2 = a, b, c, alpha, beta, gamma
        elif path.lower().endswith('.pdb'):
            a2, b2, c2, alpha2, beta2, gamma2 = load_pdb_cell_params(path)
        else:
            raise Exception('Files must be a .pdb or .xyz file')
        
        if x_size == 0 or y_size == 0 or z_size == 0:
            x_size = a
            y_size = b
            z_size = c
            print('at least one slab size was not defined. Defaulting x,y,z slab dimensions to a, b, c')

        if any(val == 0 for val in [a2, b2, c2, alpha2, beta2, gamma2]):
            raise Exception('at least one unit cell parameter a, b, c, alpha, beta, gamma not defined')

        #sample calculation 
        coords_slab, elements_slab = slabmaker_fitting(path, x_size, y_size, z_size, a2, b2, c2, alpha2, beta2, gamma2)
        iq, qx, qy, qz = voxelgridmaker_fitting(coords_slab, elements_slab, r_voxel_size, q_voxel_size, max_q, energy, num_cpus=None, fill_bkg=fill_bkg, smooth=smooth)
        det_sum, det_h, det_v = detectormaker_fitting(iq, qx, qy, qz, num_pixels, max_q, angle_init_vals, 
                                                    angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path, mirror=mirror)
        gen_name = gen_names[i]
        save_subfolder = f'{save_folder}/{gen_name}'
        os.makedirs(save_subfolder, exist_ok=True)
        #load experimental data, fitting mask
        if img_path:
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
            sim_int_trim[rebin_mask==1]=0

            #pad to make peak positions match
            if pad_width >0:
                sim_int_trim_pad = shift_peak(sim_int_trim.copy(), det_h_trim, det_v_trim, pad_width, pad_range)
            else:
                sim_int_trim_pad = sim_int_trim.copy()

            #fit scale and offset
            if fit_scale_offset:
                scale, offset = optimize_scale_offset(sim_int_trim_pad, rebin_map, rebin_mask)
                # Apply scale and offset to sim_comp_map
                scaled_map = scale * sim_int_trim_pad + offset

                sim_comp_map = scaled_map.copy()
                sim_comp_map[rebin_mask==1] = 0
            else:
                sim_comp_map = sim_int_trim_pad.copy()


            diff_map = np.where(rebin_map > 1e-10, sim_comp_map - rebin_map, 0)

            np.save(f'{save_subfolder}/det_h.npy', det_h_trim)
            np.save(f'{save_subfolder}/det_v.npy', det_v_trim)
            np.save(f'{save_subfolder}/rebin_map.npy', rebin_map)
            np.save(f'{save_subfolder}/sim_comp_map.npy', sim_comp_map)

            suptitle=''

            savepath = f'{save_subfolder}/fit_result.png'
            plot_fit(savepath, rebin_map, sim_comp_map, diff_map, det_h_trim, det_v_trim, max_q, suptitle)

            savepath = f'{save_subfolder}/fit_result_linecut.png'
            plot_linecuts(savepath, rebin_map, sim_comp_map, det_h_trim, det_v_trim, suptitle)

        else:
            if mirror:
                keep_h_idxs = np.where(det_h>=0)[0]
                keep_v_idxs = np.where(det_v>=0)[0]
                det_h = det_h[keep_h_idxs]
                det_v = det_v[keep_v_idxs]
                det_sum = det_sum[np.ix_(keep_v_idxs, keep_h_idxs)]
            np.save(f'{save_subfolder}/det_h.npy', det_h)
            np.save(f'{save_subfolder}/det_v.npy', det_v)
            np.save(f'{save_subfolder}/det_sum.npy', det_sum)

            det_sum[det_sum<=0] = 1e-6
            det_sum[det_sum!=det_sum] = 1e-6
            fig,ax=plt.subplots(figsize=(10,5))
            cax = ax.imshow(det_sum,
                    norm=matplotlib.colors.Normalize(vmin=np.percentile(det_sum, 1), vmax=np.percentile(det_sum, 99.5)),
                    #    norm=matplotlib.colors.LogNorm(vmin=np.percentile(sim_det_ints, 60), vmax=np.percentile(sim_det_ints, 99.95)),
                    extent=(np.min(det_v),np.max(det_v),np.min(det_h),np.max(det_h)),
                    cmap='turbo',
                    origin = 'lower')
            cbar = fig.colorbar(cax, ax=ax, shrink=0.82, aspect=20, pad=0.02)
            ax.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=16)
            ax.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=16)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
            plt.tight_layout()
            plt.savefig(f'{save_subfolder}/det_sum_lin.png', dpi=300)
            plt.close('all')

            fig,ax=plt.subplots(figsize=(10,5))
            cax = ax.imshow(det_sum,
                    norm=matplotlib.colors.LogNorm(vmin=np.percentile(det_sum, 50), vmax=np.percentile(det_sum, 99.95)),
                    extent=(np.min(det_v),np.max(det_v),np.min(det_h),np.max(det_h)),
                    cmap='turbo',
                    origin = 'lower')
            cbar = fig.colorbar(cax, ax=ax, shrink=0.82, aspect=20, pad=0.02)
            ax.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=16)
            ax.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=16)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
            plt.tight_layout()
            plt.savefig(f'{save_subfolder}/det_sum_log.png', dpi=300)

            
        save_config_to_txt(config, f'{save_subfolder}/config.txt')
        # write_xyz(f'{save_subfolder}/slab_x{x_size}_y{y_size}_z{z_size}.xyz', coords_slab, elements_slab)

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



