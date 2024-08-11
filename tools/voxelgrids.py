import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.pyplot import subplots
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift
from multiprocessing import Pool, shared_memory
import os
from scipy.signal import tukey

from tools.utilities import load_xyz, load_pdb, fft_gaussian, rotate_coords_z, get_element_f0_dict, get_element_f1_f2_dict
from tools.ptable_dict import ptable, aff_dict

def downselect_voxelgrid(grid, x_axis, y_axis, z_axis, max_val):
    """
    selects a cube portion of meshgrid centered about origin containing all voxels within
    max_val in each cartesian direction

    Parameters:
    - grid: A 3D numpy array to be downselected with indices (y,x,z)
    - x_axis: 1D array of x coordinate values
    - y_axis: 1D array of y coordinate values
    - z_axis: 1D array of z coordinate values
    - max_val: Maximum value for the selection range in each Cartesian direction


    Returns:
    - small_grid: a 3D numpy array with indices (y,x,z)
    - x_axis2: 1D array of x coordinate values
    - y_axis2: 1D array of y coordinate values
    - z_axis2: 1D array of z coordinate values
    """
    #add one voxel buffer to ensure max_val is contained in selection
    voxel_size = np.abs(x_axis[1]-x_axis[0])
    max_val += voxel_size

    x_idxs = np.where(np.abs(x_axis)<max_val)[0]
    y_idxs = np.where(np.abs(y_axis)<max_val)[0]
    z_idxs = np.where(np.abs(z_axis)<max_val)[0]

    small_grid = grid[y_idxs[0]:y_idxs[-1]+1, x_idxs[0]:x_idxs[-1]+1, z_idxs[0]:z_idxs[-1]+1]
    x_axis2 = x_axis[x_idxs[0]:x_idxs[-1]+1]
    y_axis2 = y_axis[y_idxs[0]:y_idxs[-1]+1]
    z_axis2 = z_axis[z_idxs[0]:z_idxs[-1]+1]

    return small_grid, x_axis2, y_axis2, z_axis2

def fill_bkg_dens3D(density_grid, coords, r_voxel_size):
    #apply bkg electron density
    # Define bounds in voxel coordinates
    x_bound = np.max(coords[:,0])
    y_bound = np.max(coords[:,1])
    z_bound = np.max(coords[:,2])
    x_bound_vox = int(x_bound / r_voxel_size)
    y_bound_vox = int(y_bound / r_voxel_size)
    z_bound_vox = int(z_bound / r_voxel_size)

    # Calculate average electron density within the bounds
    within_bounds = density_grid[:y_bound_vox, :x_bound_vox, :z_bound_vox]
    average_density = np.mean(within_bounds)

    # Fill voxels outside of bounds with average electron density value
    density_grid[y_bound_vox:, :, :] = average_density
    density_grid[:, x_bound_vox:, :] = average_density
    density_grid[:, :, z_bound_vox:] = average_density

    return density_grid

def generate_voxel_grid_high_mem(input_path, r_voxel_size, q_voxel_size, max_q, aff_num_qs, energy, gen_name, output_dir=None, bkg_edens=True):
    """
    Generates a 3D voxelized electron density grid from .xyz file.
    A average electron density is optionally applied outside of the smallest
    bounding cube for the coordinates in xyz path

    Parameters:
    - input_path: string, path to xyz or pdb file of molecule, NP, etc
    - voxel_size: real-space dimension for voxel side length
    - min_ax_size: minimum axis size, axis sizes are set to 2^n for fft efficiency
    - bkg_edens: boolean if you would like bkg_edens applied (helps reduce kiessig fringes)


    Returns:
    - density_grid: 3D meshgrid of electron density values
    - x_axis: 1D array of x coordinate values
    - y_axis: 1D array of y coordinate values
    - z_axis: 1D array of z coordinate values
    """

    # Extracting the atomic symbols and positions from the xyz file
    if input_path[-3:] == 'xyz':
        coords, elements = load_xyz(input_path)
    elif input_path[-3:] == 'pdb':
        coords, elements = load_pdb(input_path)
    else:
        raise Exception('files must be a .pdb or .xyz file')
    
    #check
    max_q_diag = np.sqrt(2)*max_q
    if max_q_diag > 2*np.pi/r_voxel_size:
        raise Exception('Max_q is non-physical for given voxel size')
    
    #calculate number of pixels with size r_voxel_size needed along real-space axis to acheive the desired q_voxel_size
    grid_size = int(np.ceil(2*np.pi/(q_voxel_size * r_voxel_size)))
    #make sure grid size is not too small
    x_bound = np.max(coords[:,0])-np.min(coords[:,0])
    y_bound = np.max(coords[:,1])-np.min(coords[:,1])
    z_bound = np.max(coords[:,2])-np.min(coords[:,2])
    min_bound = np.min([x_bound, y_bound, z_bound])
    if grid_size*r_voxel_size < min_bound:
        raise Exception('Calculated real-space bounds smaller than simulation. Please lower delta_q value')
    
    # Shift coords array to origin
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    coords[:,2] -= np.min(coords[:,2])

    #create axes
    x_axis = np.linspace(0, grid_size*r_voxel_size, grid_size)
    y_axis = np.linspace(0, grid_size*r_voxel_size, grid_size)
    z_axis = np.linspace(0, grid_size*r_voxel_size, grid_size)

    #good to have this parallelizable
    if aff_num_qs == 1:
        # Create an empty grid
        density_grid = np.zeros((grid_size, grid_size, grid_size), dtype=complex)
        #use f=f1+jf2
        f1_f2_dict = get_element_f1_f2_dict(energy, elements)
        f_values = np.array([f1_f2_dict[element] for element in elements])
        z_values = np.array([ptable[element] for element in elements])
        f_values += z_values
        #convert symbols to array of z_values used as f here
        # f_values = np.array([ptable[element] for element in elements])

        # Populate the grid
        grid_coords = (coords // r_voxel_size).astype(int)
        np.add.at(density_grid, (grid_coords[:,1], grid_coords[:,0], grid_coords[:,2]), f_values)

        #apply bkg electron density
        if bkg_edens:
            density_grid = fill_bkg_dens3D(density_grid, coords, r_voxel_size)

        master_iq_3D, qx_shifted, qy_shifted, qz_shifted  = convert_grid_qspace(density_grid, x_axis, y_axis, z_axis)
        del density_grid
        master_iq_3D, qx_shifted, qy_shifted, qz_shifted = downselect_voxelgrid(master_iq_3D, qx_shifted, qy_shifted, qz_shifted, max_q)


    #not very memory or cpu efficient implimentation. Fix later
    elif aff_num_qs > 1:
        f1_f2_dict = get_element_f1_f2_dict(energy, elements)
        f1_f2_values = np.array([f1_f2_dict[element] for element in elements])
        #build iq voxelgrid over many q values for proper f0(q)
        for i, aff_q_num in range(int(aff_num_qs)):
            # Create an empty grid
            density_grid = np.zeros((grid_size, grid_size, grid_size), dtype=complex)
            #calculate q_values for f0 evaluation
            step = (max_q_diag/(int(aff_num_qs)))
            q_val = 0.5*step+aff_q_num*step
            upper_q = q_val+step/2
            lower_q = q_val-step/2
            f0_dict = get_element_f0_dict(q_val, elements)
            f0_values = np.array([f0_dict[element] for element in elements])

            #for xraydb chantler tables it appears f1=f' and not f1=f'+f0 as usual
            f_values = f0_values + f1_f2_values

            grid_coords = (coords // r_voxel_size).astype(int)
            np.add.at(density_grid, (grid_coords[:,1], grid_coords[:,0], grid_coords[:,2]), f_values)
            #apply bkg electron density
            if bkg_edens:
                density_grid = fill_bkg_dens3D(density_grid, coords, r_voxel_size)
            
            iq_3D, qx_shifted, qy_shifted, qz_shifted = convert_grid_qspace(density_grid, x_axis, y_axis, z_axis)
            del density_grid
            iq_3D, qx_shifted, qy_shifted, qz_shifted = downselect_voxelgrid(iq_3D, qx_shifted, qy_shifted, qz_shifted, max_q)
            if i==0:
                qx_mesh, qy_mesh, qz_mesh = np.meshgrid(qx_shifted, qy_shifted, qz_shifted)
                qr_mesh = np.sqrt(qx_mesh**2 + qy_mesh**2 + qz_mesh**2)
                del qx_mesh, qy_mesh, qz_mesh
                master_iq_3D = np.zeros_like(iq_3D)
            iq_mask = (qr_mesh <= upper_q) & (qr_mesh > lower_q)
            master_iq_3D += np.where(iq_mask, iq_3D, 0)

        # Save
    if output_dir:
        save_path = f'{output_dir}/{gen_name}_output_files/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(f'{save_path}{gen_name}_iq.npy', master_iq_3D)
        np.save(f'{save_path}{gen_name}_qx.npy', qx_shifted)
        np.save(f'{save_path}{gen_name}_qy.npy', qy_shifted)
        np.save(f'{save_path}{gen_name}_qz.npy', qz_shifted)
    else:
        return master_iq_3D, qx_shifted, qy_shifted, qz_shifted

def convert_grid_qspace(density_grid, x_axis, y_axis, z_axis):
    """
    Generates a 3D voxelized scattering intensity grid from input electron density grid.
    Scattering is given as norm**2 of fft and new qmesh axes

    Parameters:
    - density_grid: 3D meshgrid of electron density values
    - x_axis: 1D array of x coordinate values
    - y_axis: 1D array of y coordinate values
    - z_axis: 1D array of z coordinate values

    Returns:
    - iq: 3D meshgrid of scattering complex values
    - qx_axis: 1D array of qx coordinate values
    - qy_axis: 1D array of qy coordinate values
    - qz_axis: 1D array of qz coordinate values
    """

    voxel_size = x_axis[1]-x_axis[0]
    grid_size_x = len(x_axis)
    grid_size_y = len(y_axis)
    grid_size_z = len(z_axis)

    # Calculate 3D q-values
    qx = np.fft.fftfreq(grid_size_x, d=voxel_size) * 2 * np.pi
    qy = np.fft.fftfreq(grid_size_y, d=voxel_size) * 2 * np.pi
    qz = np.fft.fftfreq(grid_size_z, d=voxel_size) * 2 * np.pi
    qx_shifted = fftshift(qx)
    qy_shifted = fftshift(qy)
    qz_shifted = fftshift(qz)

    # Compute the Fourier transform of the density grid
    ft_density = fftn(density_grid)
    ft_density_shifted = fftshift(ft_density)  # Shift the zero-frequency component to the center of the spectrum

    # Magnitude squared of the Fourier transform for scattering intensity I(q)
    iq = np.abs(ft_density_shifted)**2

    return iq, qx_shifted, qy_shifted, qz_shifted

def rotate_project_fft_coords(args):
        coords, f_values, phi, grid_size, r_voxel_size, temp_folder, tukey_val = args
        #rotate coords about phi
        coords_rot = rotate_coords_z(coords, phi)

        # Shift coords array to origin
        coords_rot[:,0] -= np.min(coords_rot[:,0])
        coords_rot[:,1] -= np.min(coords_rot[:,1])
        coords_rot[:,2] -= np.min(coords_rot[:,2])

        #project coords 2D
        # Convert y, z coordinates to pixel indices
        y_indices = (coords_rot[:, 1] // r_voxel_size).astype(int)
        z_indices = (coords_rot[:, 2] // r_voxel_size).astype(int)

        # Create a mask for valid indices (within the detector bounds)
        valid_mask = (y_indices >= 0) & (y_indices < grid_size) & (z_indices >= 0) & (z_indices < grid_size)
        y_indices = y_indices[valid_mask]
        z_indices = z_indices[valid_mask]
        valid_fs = f_values[valid_mask]

        # Initialize the detector grid
        detector_grid = np.zeros((grid_size, grid_size), dtype=complex)

        # Accumulate intensities in the corresponding pixels
        np.add.at(detector_grid, (z_indices, y_indices), valid_fs)

        if tukey:
            # Determine the region of interest (ROI) where data is located
            y_max = np.max(y_indices)
            z_max = np.max(z_indices)
            
            # Create a Tukey window specific to the ROI size
            # You can adjust this tukey_val between 0 (rectangular) and 1 (Hanning)
            tukey_y = tukey(y_max + 1, alpha=tukey_val)
            tukey_z = tukey(z_max + 1, alpha=tukey_val)
            roi_window = np.outer(tukey_y, tukey_z)
            
            # Apply the Tukey window to the corresponding region of the detector grid
            detector_grid[:y_max + 1, :z_max + 1] *= roi_window

        # Calculate axis q-values
        q_h = np.fft.fftfreq(grid_size, d=r_voxel_size) * 2 * np.pi
        q_v = np.fft.fftfreq(grid_size, d=r_voxel_size) * 2 * np.pi
        q_h_shifted = fftshift(q_h)
        q_v_shifted = fftshift(q_v)

        # Compute the Fourier transform of the density grid
        ft_density = fftn(detector_grid)
        ft_density_shifted = fftshift(ft_density)  # Shift the zero-frequency component to the center of the spectrum

        # Magnitude squared of the Fourier transform for scattering intensity I(q)
        iq_2d = np.abs(ft_density_shifted)**2

        #some way to assign each pixel qx,qy,qz with trig
        #rotating the coordinates by phi is equivalent to detector rotation by -phi
        right_qy = np.max(q_h_shifted) * np.cos(np.deg2rad(-phi))
        left_qy = -right_qy
        right_qx = -np.max(q_h_shifted) * np.sin(np.deg2rad(-phi))
        left_qx = -right_qx
        det_h_qy = np.linspace(left_qy, right_qy, num=len(q_h_shifted))
        det_h_qx = np.linspace(left_qx, right_qx, num=len(q_h_shifted))
        det_v_qz = q_v_shifted


        phi_100x = int(phi*100)
        filepath = f'{temp_folder}/tempfile_{phi_100x}'
        np.save(f'{filepath}_iq.npy', iq_2d)
        np.save(f'{filepath}_h_qx.npy', det_h_qx)
        np.save(f'{filepath}_h_qy.npy', det_h_qy)
        np.save(f'{filepath}_v_qz.npy', det_v_qz)

        return filepath

def create_shared_array(shape, name):
    # Create a shared memory array
    d_size = np.prod(shape) * np.dtype(np.float64).itemsize
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    return shm

def process_file(filepath, q_num, qx, qy, qz, voxel_grid_shm_name, voxel_grid_count_shm_name):
    # Access the shared memory arrays using their names
    voxel_grid_shm = shared_memory.SharedMemory(name=voxel_grid_shm_name)
    voxel_grid_count_shm = shared_memory.SharedMemory(name=voxel_grid_count_shm_name)

    # Create numpy arrays from the shared memory buffers for voxel grid and voxel count
    voxel_grid = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_shm.buf)
    voxel_grid_count = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_count_shm.buf)

    # load up files for each detector
    iq_2D = np.load(f'{filepath}_iq.npy')
    det_h_qx = np.load(f'{filepath}_h_qx.npy')
    det_h_qy = np.load(f'{filepath}_h_qy.npy')
    det_v_qz = np.load(f'{filepath}_v_qz.npy')

    # mask out values that do not fall within bounds
    det_h_mask = (det_h_qx <= np.max(qx)) & (det_h_qx >= np.min(qx)) & (det_h_qy <= np.max(qy)) & (det_h_qy >= np.min(qy))
    det_v_mask = (det_v_qz <= np.max(qz)) & (det_v_qz >= np.min(qz))

    # slice 1D axis value arrays
    det_h_qx = det_h_qx[det_h_mask]
    det_h_qy = det_h_qy[det_h_mask]
    det_v_qz = det_v_qz[det_v_mask]

    # slice 2D array based on horizontal and vertical detector masks
    iq_2D = iq_2D[det_v_mask, :][:, det_h_mask]

    qx_mesh, qz_mesh = np.meshgrid(det_h_qx, det_v_qz)
    qy_mesh, qz_mesh = np.meshgrid(det_h_qy, det_v_qz)

    qx_vals = np.ravel(qx_mesh)
    qy_vals = np.ravel(qy_mesh)
    qz_vals = np.ravel(qz_mesh)
    iq_vals = np.ravel(iq_2D)
    counter_vals = np.ones_like(iq_vals)

    # Convert y, z coordinates to pixel indices
    actual_q_voxel = np.diff(qz)[0] #voxels are cubes
    qx_indices = ((qx_vals-np.min(qx)) // actual_q_voxel).astype(int)
    qy_indices = ((qy_vals-np.min(qy)) // actual_q_voxel).astype(int)
    qz_indices = ((qz_vals-np.min(qz)) // actual_q_voxel).astype(int)

    # Accumulate intensities in the corresponding pixels
    np.add.at(voxel_grid, (qy_indices, qx_indices, qz_indices), iq_vals)
    np.add.at(voxel_grid_count, (qy_indices, qx_indices, qz_indices), counter_vals)
    

def frames_to_iq_parallel(filepaths, q_num, qx, qy, qz):
    # Create shared arrays for voxel grid and voxel count with the specified dimensions
    voxel_grid_shm = create_shared_array((q_num, q_num, q_num), 'voxel_grid_shared')
    voxel_grid_count_shm = create_shared_array((q_num, q_num, q_num), 'voxel_grid_count_shared')

    try:
        # Use ThreadPoolExecutor to process each file in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, filepath, q_num, qx, qy, qz, 'voxel_grid_shared', 'voxel_grid_count_shared') for filepath in filepaths]
            for future in as_completed(futures):
                future.result()

        # Create numpy arrays from the shared memory buffers
        voxel_grid = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_shm.buf)
        voxel_grid_count = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_count_shm.buf)

        # Final IQ calculation
        iq_3D = np.divide(voxel_grid, voxel_grid_count, out=np.zeros_like(voxel_grid), where=voxel_grid_count != 0)
    finally:
        # Ensure that shared memory is properly closed and unlinked
        voxel_grid_shm.close()
        voxel_grid_shm.unlink()
        voxel_grid_count_shm.close()
        voxel_grid_count_shm.unlink()

    return iq_3D

def generate_voxel_grid_low_mem(input_path, r_voxel_size, q_voxel_size, max_q, aff_num_qs, energy, gen_name, output_dir=None, scratch_folder=None, num_cpus=None, tukey_val=0):
    """
    Low memory method for generating a 3D voxelized scattering intensity reciprocal space grid from .xyz file.
    A average electron density is optionally applied outside of the smallest
    bounding cube for the coordinates in xyz path

    Parameters:
    - input_path: string, path to xyz or pdb file of molecule, NP, etc
    - r_voxel_size: real-space dimension for electron density voxel side length
    - q_voxel_size: scattering vector dimesnions for q-space voxel size (q_resolution)
    - max_q: maximum q-value desired along both axese of detector
    - aff_num_qs: (int) number of q-values at which the iq voxel (complexity scales O(N))
    - energy: x-ray energy in electron volts. Not needed unless aff_num_qs>1
    - output_dir: string, path to output directory, default "none" for no saving
    - scratch_dir: string, path to scratch directory for saving temporary files, default cwd
    - num_cpus: (int) number of cpus, default value "None" uses os.cpu_count()


    Returns:
    None if output file is specified, else:
    - iq_grid: 3D voxelgrid of scattering intensity values
    - qx_axis: 1D array of qx coordinate values
    - qy_axis: 1D array of qy coordinate values
    - qz_axis: 1D array of qz coordinate values
    """

    if num_cpus:
        num_cpus = int(num_cpus)
    else:
        num_cpus = os.cpu_count()

    #check
    max_q_diag = np.sqrt(2)*max_q
    if max_q_diag > 2*np.pi/r_voxel_size:
        raise Exception('Max_q is non-physical for given voxel size')

    # Extracting the atomic symbols and positions from the xyz file
    if input_path[-3:] == 'xyz':
        coords, elements = load_xyz(input_path)
    elif input_path[-3:] == 'pdb':
        coords, elements = load_pdb(input_path)
    else:
        raise Exception('files must be a .pdb or .xyz file')

    #calculate number of pixels with size r_voxel_size needed along real-space axis to acheive the desired q_voxel_size
    grid_size = int(np.ceil(2*np.pi/(q_voxel_size * r_voxel_size)))
    #make sure grid size is not too small
    x_bound = np.max(coords[:,0])-np.min(coords[:,0])
    y_bound = np.max(coords[:,1])-np.min(coords[:,1])
    z_bound = np.max(coords[:,2])-np.min(coords[:,2])
    min_bound = np.min([x_bound, y_bound, z_bound])
    if grid_size*r_voxel_size < min_bound:
        raise Exception('Calculated real-space bounds smaller than simulation. Please lower delta_q value')

    #create empty voxel grid
    max_q_diag = max_q_diag + max_q_diag%q_voxel_size
    q_num = ((2*max_q_diag/q_voxel_size)+1).astype(int)
    qx = qy = qz = np.linspace(-max_q_diag, max_q_diag, q_num)

    #some calculation to see how many phi angles we need to do
    # Calculate the number of angles needed
    delta_phi_rad = np.arctan(q_voxel_size/max_q_diag)
    phi_num = np.ceil(2*np.pi/delta_phi_rad).astype(int)
    last_phi = 180-(180/phi_num)
    phis = np.linspace(0,last_phi, num=phi_num)

    #save in some scratch folder to combine into voxelgrid later
    if scratch_folder:
        folder = scratch_folder
    else:
        folder = os.getcwd()
    temp_folder = f'{folder}/tempfiles'
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    #good to have this parallelizable
    if aff_num_qs == 1:
        #use f=f1+jf2
        f1_f2_dict = get_element_f1_f2_dict(energy, elements)
        f_values = np.array([f1_f2_dict[element] for element in elements], dtype=complex)
        # xraydb chantler lookup defines f1=f' and f2=f" contrary to convention
        z_values = np.array([ptable[element] for element in elements])
        f_values += z_values
        #parallel processing to generate frames used to construct iq_3D
        args = [(coords, f_values, phi, grid_size, r_voxel_size, temp_folder, tukey_val) for phi in phis]
        with Pool(processes=num_cpus) as pool:
            filepaths = pool.map(rotate_project_fft_coords, args)
        master_iq_3D = frames_to_iq_parallel(filepaths, q_num, qx, qy, qz)
        # Cleanup temporary files
        for filepath in filepaths:
            try:
                os.remove(f'{filepath}_iq.npy')
                os.remove(f'{filepath}_h_qx.npy')
                os.remove(f'{filepath}_h_qy.npy')
                os.remove(f'{filepath}_v_qz.npy')
            except OSError as e:
                print(f"Error deleting file {filepath}: {e}")

    #not very memory or cpu efficient implimentation. Fix later
    elif aff_num_qs > 1:
        qx_mesh, qy_mesh, qz_mesh = np.meshgrid(qx, qy, qz)
        qr_mesh = np.sqrt(qx_mesh**2 + qy_mesh**2 + qz_mesh**2)
        del qx_mesh, qy_mesh, qz_mesh
        master_iq_3D = np.zeros((q_num, q_num, q_num))
        f1_f2_dict = get_element_f1_f2_dict(energy, elements)
        f1_f2_values = np.array([f1_f2_dict[element] for element in elements])
        # z_values = np.array([ptable[element] for element in elements])
        #build iq voxelgrid over many q values for proper f0(q)
        for aff_q_num in range(int(aff_num_qs)):
            #calculate q_values for f0 evaluation
            step = (max_q_diag/(int(aff_num_qs)))
            q_val = 0.5*step+aff_q_num*step
            upper_q = q_val+step/2
            lower_q = q_val-step/2
            f0_dict = get_element_f0_dict(q_val, elements)
            f0_values = np.array([f0_dict[element] for element in elements])

            #for xraydb chantler tables it appears f1=f' and not f1=f'+f0 as usual
            f_values = f0_values + f1_f2_values

            #parallel processing of frames rotating around phi
            args = [(coords, f_values, phi, grid_size, r_voxel_size, temp_folder, tukey_val) for phi in phis]
            with Pool(processes=num_cpus) as pool:
                filepaths = pool.map(rotate_project_fft_coords, args)
            #generate iq voxelgrid, mask out q's based on valid f0 range, add to master
            iq_3D = frames_to_iq_parallel(filepaths, q_num, qx, qy, qz)
            iq_mask = (qr_mesh <= upper_q) & (qr_mesh > lower_q)
            master_iq_3D += np.where(iq_mask, iq_3D, 0)

            #cleanup memory
            del iq_3D, iq_mask

            # Cleanup temporary files
            for filepath in filepaths:
                try:
                    os.remove(f'{filepath}_iq.npy')
                    os.remove(f'{filepath}_h_qx.npy')
                    os.remove(f'{filepath}_h_qy.npy')
                    os.remove(f'{filepath}_v_qz.npy')
                except OSError as e:
                    print(f"Error deleting file {filepath}: {e}")
    else:
        raise Exception('Invalid aff_num_qs value. Must be non-negative integer')

    # Save
    if output_dir:
        save_path = f'{output_dir}/{gen_name}_output_files/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(f'{save_path}{gen_name}_iq.npy', master_iq_3D)
        np.save(f'{save_path}{gen_name}_qx.npy', qx)
        np.save(f'{save_path}{gen_name}_qy.npy', qy)
        np.save(f'{save_path}{gen_name}_qz.npy', qz)
    else:
        return master_iq_3D, qx, qy, qz

def multiply_ft_gaussian(grid, x_axis, y_axis, z_axis, sigma):
    """
    Multiplies a 3D meshgrid by a fourier transformed gaussian.
    This is used as an efficient way to apply real-space guassian
    distribution to atoms (i.e. debye waller factor)

    Parameters:
    - grid (np.ndarray): 3D numpy array representing q-space meshgrid
    - x_axis (np.ndarrar): 1D numpy array representing qx axis of meshgrid
    - y_axis (np.ndarrar): 1D numpy array representing qy axis of meshgrid
    - z_axis (np.ndarrar): 1D numpy array representing qz axis of meshgrid
    - sigma (float): sigma value for real-space guassian (debye-waller)
    """
    g_fft = fft_gaussian(x_axis, y_axis, z_axis, sigma)
    grid_smeared = grid * g_fft**2

    return grid_smeared


def plot_3D_grid(density_grid, x_axis, y_axis, z_axis, cmap, threshold_pct=98, num_levels=10, log=True):
    """
    Plots a 3D scatter plot of an electron density grid with color mapping and opacity levels.

    Parameters:
    density_grid (np.ndarray): A 3D numpy array representing the electron density grid.
    cmap (str): The name of the colormap to use for coloring the density levels.
    threshold_pct (float, optional): The percentile threshold to determine which density values to plot.
                                     Only values above this percentile will be included. Default is 98.
    num_levels (int, optional): The number of opacity levels to use in the plot. Default is 10.

    Returns:
    None: Displays a 3D scatter plot of the electron density grid.
    """

    y, x, z = np.where(density_grid>np.percentile(density_grid, threshold_pct))
    values = density_grid[y, x, z]
    if log:
        values = np.log(values)
    max_values = np.max(values)
    min_values = np.min(values)
    # Get the absolute coordinates
    x_abs = x_axis[x]
    y_abs = y_axis[y]
    z_abs = z_axis[z]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the number of levels of opacity
    opacities = np.linspace(0.3,0.01,num_levels)

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.3, 1, num_levels))

    for i in range(num_levels):
        # Calculate the opacity for the current level
        opacity = opacities[i]
        color = colors[i]

        mask_low = 100*i/num_levels
        mask_high = 100*(i+1)/num_levels
        # Determine the data points that fall into the current opacity level
        mask = (values > np.percentile(values, mask_low)) & (values <= np.percentile(values, mask_high))

        # Scatter plot for the current subset of data
        ax.scatter(x_abs[mask],
                   y_abs[mask],
                   z_abs[mask],
                   color=color,  # Use the single color for all points
                   alpha=opacity,
                   edgecolor='none')

    # Set labels and titles
    ax.set_xlabel('X (mesh units)')
    ax.set_ylabel('Y (mesh units)')
    ax.set_zlabel('Z (mesh units)')
    ax.set_title('3D Scatter Plot of Electron Density')

    # Setting equal aspect ratio
    max_range = np.array([x_abs.max()-x_abs.min(), y_abs.max()-y_abs.min(), z_abs.max()-z_abs.min()]).max() / 2.0
    mid_x = (x_abs.max()+x_abs.min()) * 0.5
    mid_y = (y_abs.max()+y_abs.min()) * 0.5
    mid_z = (z_abs.max()+z_abs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.patch.set_facecolor('black')  # Set the outer background color
    ax.set_facecolor('black')  # Set the background color of the plot
    # Change the color of the ticks and labels to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Change grid and pane colors
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(color='white', linestyle='--', linewidth=0.5)
    plt.show()

def add_f0_q_3d(iq, qx_axis, qy_axis, qz_axis, element):
    """
    Adds normalized f0(q)^2 dependence onto a reciprocal space scattering intensity map
    parameters:
    - iq: 3D voxel grid of scattering complex values
    - qx_axis: 1D array of qx coordinate values
    - qy_axis: 1D array of qy coordinate values
    - qz_axis: 1D array of qz coordinate values
    - element: (str) elemental symbol
    """
    iq_new = np.copy(iq)
    Z = ptable[element]
    aff = aff_dict[element]

    qx, qy, qz = np.meshgrid(qx_axis, qy_axis, qz_axis)
    q_grid_squared = qx**2 + qy**2 + qz**2

    del qx
    del qy
    del qz

    fq_norm = ((
                aff[0]*np.exp(-aff[1]*(q_grid_squared)/(16*np.pi**2))+
                aff[2]*np.exp(-aff[3]*(q_grid_squared)/(16*np.pi**2))+
                aff[4]*np.exp(-aff[5]*(q_grid_squared)/(16*np.pi**2))+
                aff[6]*np.exp(-aff[7]*(q_grid_squared)/(16*np.pi**2))+
                aff[8])/Z)**2
    iq_new *=fq_norm

    return iq_new
