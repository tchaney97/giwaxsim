import numpy as np
import glob
import os
import argparse

from tools.utilities import parse_config_file, most_common_element
from tools.voxelgrids import downselect_voxelgrid, add_f0_q_3d, generate_voxel_grid_low_mem

def main(config):
    # Input Parameters
    input_folder = config.get('input_folder', None)
    input_filepath = config.get('input_filepath', None)
    filetype = config.get('filetype', 'xyz')
    gen_name = config.get('gen_name')
    r_voxel_size = float(config.get('r_voxel_size', 0.3))
    q_voxel_size = float(config.get('q_voxel_size', 0.01))
    aff_num_qs = int(config.get('aff_num_qs', 1))
    energy = float(config.get('energy', 1))
    max_q = float(config.get('max_q', 2.5))
    output_dir = config.get('output_dir', os.getcwd())
    num_cpus = int(config.get('num_cpus', os.cpu_count()))
    scratch_dir = config.get('scratch_dir', os.getcwd())

    if input_folder:
        input_paths = glob.glob(f'{input_folder}*{filetype}')
    elif input_filepath:
        input_paths = [input_filepath]
    else:
        raise Exception('Either input_folder or input_path must be specified')
        
    for i, input_path in enumerate(input_paths):
        iq, qx, qy, qz = generate_voxel_grid_low_mem(input_path,
                                                    r_voxel_size, 
                                                    q_voxel_size, 
                                                    max_q, 
                                                    aff_num_qs, 
                                                    energy, 
                                                    gen_name, 
                                                    scratch_dir=scratch_dir, 
                                                    num_cpus=num_cpus)
    
        # Optional downselect iq meshgrid based on max q desired
        iq_small, qx_small, qy_small, qz_small = downselect_voxelgrid(iq, qx, qy, qz, max_q)
    
        # Optional free up memory
        del iq
        del qx
        del qy
        del qz

        if i == 0:
            iq_sum = iq_small
        else:
            iq_sum += iq_small
    
    iq_sum /= len(input_paths)
    
    # Reassign variables
    iq = iq_sum
    qx = qx_small
    qy = qy_small
    qz = qz_small

    # apply (f0(q)/z)**2 scaling to scatting intensity values
    if aff_num_qs == 1:
        f0_element = most_common_element(input_paths[0])
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