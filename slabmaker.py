import numpy as np
import os
import argparse

from tools.ptable_dict import ptable, atomic_masses
from tools.utilities import write_xyz, load_xyz, load_pdb, calc_real_space_abc, parse_config_file

def main(config):
    # Inputs
    input_filepath = config.get('input_filepath')
    output_folder = config.get('output_folder', os.path.dirname(input_filepath))
    gen_name = config.get('gen_name')
    x_size = float(config.get('x_size'))
    y_size = float(config.get('y_size'))
    z_size = float(config.get('z_size'))
    a = float(config.get('a'))
    b = float(config.get('b'))
    c = float(config.get('c'))
    alpha = float(config.get('alpha'))
    beta = float(config.get('beta'))
    gamma = float(config.get('gamma'))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    if input_filepath[-3:] == 'xyz':
        coords, elements = load_xyz(input_filepath)
    elif input_filepath[-3:] == 'pdb':
        coords, elements = load_pdb(input_filepath)
    else:
        raise Exception('files must be a .pdb or .xyz file')
    
    a_vect, b_vect, c_vect = calc_real_space_abc(a, b, c, alpha, beta, gamma)
    num_x = int(np.ceil(2*x_size/a_vect[0]))
    num_y = int(np.ceil(2*y_size/b_vect[1]))
    num_z = int(np.ceil(2*z_size/c_vect[2]))
    
    coords_original = coords
    elements_original = elements
    for i in range(num_x):
        coords_new = coords_original+[a_vect[0]*(i+1), 0, 0]
        if i == 0:
            coords_append = coords_new
            elements_append = elements_original
        else:
            coords_append = np.concatenate((coords_append, coords_new), axis=0)
            elements_append = np.concatenate((elements_append, elements), axis=0)  
        if i==num_x-1:
            coords = np.concatenate((coords_original, coords_append), axis=0)
            elements = np.concatenate((elements_original, elements_append), axis=0)
    
    coords_original = coords
    elements_original = elements
    for i in range(num_y):
        coords_new = coords_original+[b_vect[0]*(i+1), b_vect[1]*(i+1), 0]
        if i ==0:
            coords_append = coords_new
            elements_append = elements_original
        else:
            coords_append = np.concatenate((coords_append, coords_new), axis=0)
            elements_append = np.concatenate((elements_append, elements), axis=0)
        if i==num_y-1:
            coords = np.concatenate((coords_original, coords_append), axis=0)
            elements = np.concatenate((elements_original, elements_append), axis=0)
    
    coords_original = coords
    elements_original = elements
    for i in range(num_z):
        coords_new = coords_original + [c_vect[0]*(i+1),  c_vect[1]*(i+1), c_vect[2]*(i+1)]
        if i ==0:
            coords_append = coords_new
            elements_append = elements_original
        else:
            coords_append = np.concatenate((coords_append, coords_new), axis=0)
            elements_append = np.concatenate((elements_append, elements), axis=0)
        if i==num_z-1:
            coords = np.concatenate((coords_original, coords_append), axis=0)
            elements = np.concatenate((elements_original, elements_append), axis=0)
    
    x_max = np.max(coords[:,0])-np.min(coords[:,0])
    y_max = np.max(coords[:,1])-np.min(coords[:,1])
    z_max = np.max(coords[:,2])-np.min(coords[:,2])
    
    assert x_max>x_size
    assert y_max>y_size
    assert z_max>z_size
    
    x_buffer = (x_max-x_size)/2
    y_buffer = (y_max-y_size)/2
    z_buffer = (z_max-z_size)/2
    
    x_lower = x_buffer
    x_upper = x_max-x_buffer
    y_lower = y_buffer
    y_upper = y_max-y_buffer
    z_lower = z_buffer
    z_upper = z_max-z_buffer
    
    # Shift coords array to origin 
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    coords[:,2] -= np.min(coords[:,2])
    
    # Use NumPy masking to filter the coordinates
    mask = (
        (coords[:,0] >= x_lower) & (coords[:,0] <= x_upper) &
        (coords[:,1] >= y_lower) & (coords[:,1] <= y_upper) &
        (coords[:,2] >= z_lower) & (coords[:,2] <= z_upper)
    )
    
    coords_new = coords[mask]
    elements_new = elements[mask]
    
    save_path = f'{output_folder}/{gen_name}_rect_cut{int(x_size)}x{int(y_size)}x{int(z_size)}.xyz'
    write_xyz(save_path, coords_new, elements_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    config_path = args.config
    config = parse_config_file(config_path)
    main(config)