import numpy as np
from numpy.fft import fftn, fftshift
import glob
from multiprocessing import Pool
import os
import argparse

from tools.ptable_dict import ptable, atomic_masses
from tools.utilities import write_xyz, load_xyz, rotation_matrix, gaussian_kernel, parse_config_file
from tools.voxelgrids import generate_density_grid, convert_grid_qspace, downselect_meshgrid, multiply_ft_gaussian, add_f0_q_3d

def main(config):
    # Input Parameters
    xyz_folder = config.get('folder', None)
    xyz_path = config.get('xyz_path', None)
    gen_name = config.get('gen_name')
    voxel_size = float(config.get('voxel_size', 0.3))
    min_ax_size = int(config.get('min_ax_size', 512))
    f0_element = (config.get('f0_element', 'C'))
    max_q = float(config.get('max_q', 2.5))
    output_dir = config.get('output_dir', os.getcwd())

    if xyz_folder:
        xyz_paths = glob.glob(f'{xyz_folder}*.xyz')
    else:
        xyz_paths = [xyz_path]
        
    for i, xyz_path in enumerate(xyz_paths):
        dens_grid, x_axis, y_axis, z_axis = generate_density_grid(xyz_path, voxel_size, min_ax_size=min_ax_size)
    
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

        if i == 0:
            iq_sum = iq_small
        else:
            iq_sum += iq_small
    
    # Reassign variables
    iq = iq_sum
    qx = qx_small
    qy = qy_small
    qz = qz_small

    # using f0 scaling instead, this may be useful for some though
    # Apply real-space Gaussian smearing
    # iq = multiply_ft_gaussian(iq, qx, qy, qz, sigma)

    # apply (f0(q)/z)**2 scaling to scatting intensity values
    iq = add_f0_q_3d(iq, qx, qy, qz, f0_element)


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