import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift

from utilities import load_xyz, fft_gaussian
from ptable_dict import ptable, aff_dict

def generate_density_grid(xyz_path, voxel_size, min_ax_size=256, bkg_edens=True):
    """
    Generates a 3D voxelized electron density grid from .xyz file. 
    A average electron density is optionally applied outside of the smallest
    bounding cube for the coordinates in xyz path
    
    Parameters:
    - xyz_path: string, path to xyz file of molecule, NP, etc
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
    coords, symbols = load_xyz(xyz_path)

    # Shift coords array to origin
    coords = np.array(coords)
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    coords[:,2] -= np.min(coords[:,2])

    # axis grids
    grid_size_x = int(np.ceil((np.max(coords[:,0]))/voxel_size))
    grid_size_y = int(np.ceil((np.max(coords[:,1]))/voxel_size))
    grid_size_z = int(np.ceil((np.max(coords[:,2]))/voxel_size))

    #calcuate number of voxel grid points, pad to nearest 2^n
    grid_vox_x = 1 << (grid_size_x - 1).bit_length()
    grid_vox_y = 1 << (grid_size_y - 1).bit_length()
    grid_vox_z = 1 << (grid_size_z - 1).bit_length()
    if grid_vox_x < min_ax_size:
        grid_vox_x = min_ax_size
    if grid_vox_y < min_ax_size:
        grid_vox_y = min_ax_size
    if grid_vox_z < min_ax_size:
        grid_vox_z = min_ax_size

    #create axes
    x_axis = np.linspace(0, grid_vox_x*voxel_size, grid_vox_x)
    y_axis = np.linspace(0, grid_vox_y*voxel_size, grid_vox_y)
    z_axis = np.linspace(0, grid_vox_z*voxel_size, grid_vox_z)

    # Create an empty grid
    density_grid = np.zeros((grid_vox_y, grid_vox_x, grid_vox_z))


    # Populate the grid
    for coord, symbol in zip(coords, symbols):
        grid_coord = (coord / voxel_size).astype(int)
        density_grid[grid_coord[1], grid_coord[0], grid_coord[2]] += (ptable[symbol])

    #apply bkg electron density
    if bkg_edens:
        # Define bounds in voxel coordinates
        x_bound = np.max(coords[:,0])
        y_bound = np.max(coords[:,1])
        z_bound = np.max(coords[:,2])
        x_bound_vox = int(x_bound / voxel_size)
        y_bound_vox = int(y_bound / voxel_size)
        z_bound_vox = int(z_bound / voxel_size)
        
        # Calculate average electron density within the bounds
        within_bounds = density_grid[:y_bound_vox, :x_bound_vox, :z_bound_vox]
        average_density = np.mean(within_bounds)
        print(average_density)
    
        # Fill voxels outside of bounds with average electron density value
        density_grid[y_bound_vox:, :, :] = average_density
        density_grid[:, x_bound_vox:, :] = average_density
        density_grid[:, :, z_bound_vox:] = average_density
        

    return density_grid, x_axis, y_axis, z_axis


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

def downselect_meshgrid(grid, x_axis, y_axis, z_axis, max_val):
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
    
    for i, qy in enumerate(qy_axis):
        for j, qx in enumerate(qx_axis):
            for k, qz in enumerate(qz_axis):
                q = np.sqrt(qx**2 + qy**2 + qz**2)
                # table 6.1.1.4 from https://it.iucr.org/Cb/ch6o1v0001/ 
                fq = (
                    aff[0]*np.exp(-aff[1]*(q**2)/(16*np.pi**2))+
                    aff[2]*np.exp(-aff[3]*(q**2)/(16*np.pi**2))+
                    aff[4]*np.exp(-aff[5]*(q**2)/(16*np.pi**2))+
                    aff[6]*np.exp(-aff[7]*(q**2)/(16*np.pi**2))+
                    aff[8])
                
                fq_norm = (fq/Z)**2
                iq_new[i,j,k]*=fq_norm
            
    return iq_new