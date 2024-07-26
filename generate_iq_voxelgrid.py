import numpy as np
from numpy.fft import fftn, fftshift
import glob
from multiprocessing import Pool
import os
import argparse

from ptable_dict import ptable, atomic_masses
from utilities import write_xyz, load_xyz, rotation_matrix, gaussian_kernel, parse_config_file
from meshgrids import generate_density_grid, convert_grid_qspace, downselect_meshgrid, multiply_ft_gaussian

def main(config):
    # Input Parameters
    xyz_path = config.get('xyz_path')
    gen_name = config.get('gen_name')
    buffer_val = float(config.get('buffer_val', 0.5))
    voxel_size = float(config.get('voxel_size', 0.3))
    min_ax_size = int(config.get('min_ax_size', 512))
    sigma = float(config.get('sigma', 0.2))
    max_q = float(config.get('max_q', 2.5))
    output_dir = config.get('output_dir', os.getcwd())
    
    dens_grid, x_axis, y_axis, z_axis = generate_density_grid(xyz_path, buffer_val, voxel_size, min_ax_size=min_ax_size)

    iq, qx, qy, qz = convert_grid_qspace(dens_grid, x_axis, y_axis, z_axis)

    # Free up memory
    del dens_grid
    del x_axis
    del y_axis
    del z_axis

    # Optional downselect iq meshgrid based on max q desired
    iq_small, qx_small, qy_small, qz_small = downselect_meshgrid(iq, qx, qy, qz, max_q)

    # Optional free up memory
    del iq
    del qx
    del qy
    del qz

    # Reassign variables
    iq = iq_small
    qx = qx_small
    qy = qy_small
    qz = qz_small

    # Apply real-space Gaussian smearing
    iq = multiply_ft_gaussian(iq, qx, qy, qz, sigma)

    # Save
    save_path = f'{output_dir}/{gen_name}_output_files/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    np.save(f'{save_path}{gen_name}_iq.npy', iq)
    np.save(f'{save_path}{gen_name}_qx.npy', qx)
    np.save(f'{save_path}{gen_name}_qy.npy', qy)
    np.save(f'{save_path}{gen_name}_qz.npy', qz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    config_path = args.config
    config = parse_config_file(config_path)
    main(config)