import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from multiprocessing import Pool
import os
import argparse
import time

from tools.utilities import  parse_config_file, str_to_bool, save_config_to_txt
from tools.detector import make_detector, rotate_about_normal, rotate_about_horizontal, rotate_about_vertical
from tools.detector import mirror_vertical_horizontal, generate_detector_ints

def main(config):
    # Input Parameters 
    iq_output_folder = config.get('iq_output_folder')
    gen_name = config.get('gen_name')
    max_q = float(config.get('max_q', 2.5))
    num_pixels = int(config.get('num_pixels', 500))
    angle_init_val1 = float(config.get('angle_init_val1', 0))
    angle_init_val2 = float(config.get('angle_init_val2', 0))
    angle_init_val3 = float(config.get('angle_init_val3', 0))
    angle_init_ax1 = config.get('angle_init_ax1', 'None')
    angle_init_ax2 = config.get('angle_init_ax2', 'None')
    angle_init_ax3 = config.get('angle_init_ax3', 'None')
    psi_start = float(config.get('psi_start'))
    psi_end = float(config.get('psi_end'))
    psi_num = int(config.get('psi_num'))
    phi_start = float(config.get('phi_start'))
    phi_end = float(config.get('phi_end'))
    phi_num = int(config.get('phi_num'))
    theta_start = float(config.get('theta_start'))
    theta_end = float(config.get('theta_end'))
    theta_num = int(config.get('theta_num'))
    mirror = str_to_bool(config.get('mirror', False))
    cleanup = str_to_bool(config.get('cleanup', False))
    num_cpus = int(config.get('num_cpus', os.cpu_count()))
    scratch_folder = config.get('scratch_folder', False)


    # dirr = os.getcwd()
    # save_path = f'{dirr}/{gen_name}_output_files/'
    save_path = iq_output_folder
    if not os.path.exists(save_path):
        raise Exception(f'Path does not exist: {save_path}')
    
    # load up 3D voxel grids from voxelgridmaker
    iq = np.load(f'{save_path}/{gen_name}_iq.npy')
    qx = np.load(f'{save_path}/{gen_name}_qx.npy')
    qy = np.load(f'{save_path}/{gen_name}_qy.npy')
    qz = np.load(f'{save_path}/{gen_name}_qz.npy')

    # make save paths
    det_sum_path = f'{save_path}/{gen_name}_det_sum/'
    i = 0
    while os.path.exists(det_sum_path):
        i += 1
        det_sum_path = f'{save_path}/{gen_name}_det_sum{i}/'
    os.mkdir(det_sum_path)

    if not scratch_folder:
        scratch_folder = save_path
    if i > 0:
        det_save_path = f'{scratch_folder}/{gen_name}_det_imgs{i}/'
    else:
        det_save_path = f'{scratch_folder}/{gen_name}_det_imgs/'

    # shouldnt already exist but just in case
    if not os.path.exists(det_save_path):
        os.mkdir(det_save_path)

    det_pixels = (num_pixels, num_pixels) # horizontal, vertical
    det_qs = (max_q, max_q) # horizontal, vertical

    # Set up detector
    det_x, det_y, det_z, det_h, det_v = make_detector(det_qs[0], det_pixels[0], det_qs[1], det_pixels[1])
    if angle_init_ax1=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val1)
    if angle_init_ax1=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val1)
    if angle_init_ax1=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val1)
        
    if angle_init_ax2=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val2)
    if angle_init_ax2=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val2)
    if angle_init_ax2=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val2)
        
    if angle_init_ax3=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val3)
    if angle_init_ax3=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val3)
    if angle_init_ax3=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val3)
        
    np.save(f'{det_sum_path}/{gen_name}_det_h.npy', det_h)
    np.save(f'{det_sum_path}/{gen_name}_det_v.npy', det_v)

    # Set up rotations to capture disorder in your film. psi=tilting, phi=fiber texture
    # Only need 1/4 of your total rotation space since symmetry allows us to mirror quadrants
    psis = np.linspace(psi_start, psi_end, num=int(psi_num)) # rotation in degrees of detector about detector normal axis
    phis = np.linspace(phi_start, phi_end, num=int(phi_num)) # rotation in degrees of detector about detector vertical axis
    thetas = np.linspace(theta_start, theta_end, num=int(theta_num)) # rotation in degrees of detector about detector horizontal axis

    args_list = [(iq, qx, qy, qz, det_h, det_v, det_x, det_y, det_z, psi, phi, theta, det_save_path) for psi in psis for phi in phis for theta in thetas]
    with Pool(processes=num_cpus) as pool:
        filenames = pool.map(generate_detector_ints, args_list)

    det_files = filenames
    for i, det_file in enumerate(det_files):
        det_img = np.load(det_file)
        if i == 0:
            det_sum = det_img
        else:
            det_sum += det_img

    # Fold detector sum image to capture full orientational space
    if mirror:
        det_sum = mirror_vertical_horizontal(det_sum)
    det_sum[det_sum != det_sum] = 1e-6
    det_sum[det_sum <= 0] = 1e-6
    np.save(f'{det_sum_path}/{gen_name}_det_sum.npy', det_sum)

    if cleanup:
        for filepath in filenames:
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error deleting file {filepath}: {e}")
        try:
            os.rmdir(det_save_path)
        except OSError as e:
            print(f"Error deleting directory {det_save_path}: {e}")

    fig, ax1 = subplots()
    cax = ax1.imshow(det_sum,
               norm=matplotlib.colors.LogNorm(vmin=np.percentile(det_sum, 10), vmax=np.percentile(det_sum, 99.9)),
               extent=(np.min(det_h), np.max(det_h), np.min(det_v), np.max(det_v)),
               cmap='turbo',
               origin='lower')
    ax1.set_xlabel('q horizontal (1/Å)')
    ax1.set_ylabel('q vertical (1/Å)')
    ax1.set_ylim(bottom=0)
    cbar = fig.colorbar(cax, ax=ax1)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{gen_name}_det_sum_log.png', dpi=300)

    fig, ax1 = subplots()
    cax = ax1.imshow(det_sum,
               norm=matplotlib.colors.Normalize(vmin=np.percentile(det_sum, 10), vmax=np.percentile(det_sum, 99.9)),
               extent=(np.min(det_h), np.max(det_h), np.min(det_v), np.max(det_v)),
               cmap='turbo',
               origin='lower')
    ax1.set_xlabel('q horizontal (1/Å)')
    ax1.set_ylabel('q vertical (1/Å)')
    ax1.set_ylim(bottom=0)
    cbar = fig.colorbar(cax, ax=ax1)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{gen_name}_det_sum_lin.png', dpi=300)

    save_config_to_txt(config, f'{save_path}/{gen_name}_config.txt')

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