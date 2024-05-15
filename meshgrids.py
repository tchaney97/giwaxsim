import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from scipy.signal import convolve
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift

from utilities import load_xyz, gaussian_kernel
from ptable_dict import ptable

def generate_density_grid(xyz_path, sigma, voxel_size):
    """
    Generates a 3D voxelized electron density grid from .xyz file. electron density is 
    smeared using gaussian convolution with width sigma. Smearing is skipped if sigma=0 
    
    Parameters:
    - xyz_path: string, path to xyz file of molecule, NP, etc
    - sigma: peak width where FWHM=2 sqrt(2ln(2))sigma. Set to 0 for no smearing

    Returns:
    - density_grid: 3D meshgrid of electron density values
    - x_mesh: 3D meshgrid of x coordinate values 
    - y_mesh: 3D meshgrid of y coordinate values 
    - z_mesh: 3D meshgrid of z coordinate values 
    """
    # Extracting the atomic symbols and positions from the xyz file
    coords, symbols = load_xyz(xyz_path)

    # Shift coords array to origin (buffer ensures room for Gaussian smearing)
    coords = np.array(coords)
    buffer = 3 * sigma # same size as guassian kernel (made later)
    coords[:,0] -= np.min(coords[:,0])-buffer
    coords[:,1] -= np.min(coords[:,1])-buffer
    coords[:,2] -= np.min(coords[:,2])-buffer

    # axis grids
    grid_size_x = int(np.ceil((np.max(coords[:,0])+buffer)/voxel_size))
    grid_size_y = int(np.ceil((np.max(coords[:,1])+buffer)/voxel_size))
    grid_size_z = int(np.ceil((np.max(coords[:,2])+buffer)/voxel_size))
    x_axis = np.linspace(0, grid_size_x*voxel_size, grid_size_x)
    y_axis = np.linspace(0, grid_size_y*voxel_size, grid_size_y)
    z_axis = np.linspace(0, grid_size_z*voxel_size, grid_size_z)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_axis, y_axis, z_axis)

    # Create an empty grid
    density_grid = np.zeros_like(x_mesh)


    # Populate the grid
    for coord, symbol in zip(coords, symbols):
        grid_coord = (coord / voxel_size).astype(int)
        density_grid[grid_coord[1], grid_coord[0], grid_coord[2]] += (ptable[symbol])*voxel_size**3
    
    # Create a Gaussian kernel
    if sigma:
        sigma_voxel = sigma/voxel_size
        kernel_size = 3 * sigma_voxel + 1  # Ensure the kernel size covers enough of the Gaussian
        gaussian_kernel_3d = gaussian_kernel(kernel_size, sigma)
        # convolve gaussian with 
        density_grid = convolve(density_grid, gaussian_kernel_3d, mode='same')

    return density_grid, x_mesh, y_mesh, z_mesh

def convert_grid_qspace(density_grid, x_mesh, y_mesh, z_mesh):
    """
    Generates a 3D voxelized scattering intensity grid from input electron density grid.
    Scattering is given as norm**2 of fft and new qmesh axes
    
    Parameters:
    - density_grid: 3D meshgrid of electron density values
    - x_mesh: 3D meshgrid of x coordinate values 
    - y_mesh: 3D meshgrid of y coordinate values 
    - z_mesh: 3D meshgrid of z coordinate values 

    Returns:
    - iq: 3D meshgrid of scattering intensity values
    - qx_mesh: 3D meshgrid of qx coordinate values 
    - qy_mesh: 3D meshgrid of qy coordinate values 
    - qz_mesh: 3D meshgrid of qz coordinate values 
    """

    # cartesian indexing
    x_vals = x_mesh[0,:,0] # x = columns (2nd axis)
    y_vals = y_mesh[:,0,0] # y = rows (1st axis)
    z_vals = z_mesh[0,0,:] # z = depth (3rd axis)
    
    voxel_size = x_vals[1]-x_vals[0]
    grid_size_x = len(x_vals)
    grid_size_y = len(y_vals)
    grid_size_z = len(z_vals)

    # Calculate 3D q-values
    qx = np.fft.fftfreq(grid_size_x, d=voxel_size) * 2 * np.pi
    qy = np.fft.fftfreq(grid_size_y, d=voxel_size) * 2 * np.pi
    qz = np.fft.fftfreq(grid_size_z, d=voxel_size) * 2 * np.pi
    qx_shifted = fftshift(qx)
    qy_shifted = fftshift(qy)
    qz_shifted = fftshift(qz)
    qx_mesh, qy_mesh, qz_mesh = np.meshgrid(qx_shifted, qy_shifted, qz_shifted)

    # Compute the Fourier transform of the density grid
    ft_density = fftn(density_grid)
    ft_density_shifted = fftshift(ft_density)  # Shift the zero-frequency component to the center of the spectrum
    
    # Magnitude squared of the Fourier transform for scattering intensity I(q)
    iq = np.abs(ft_density_shifted)**2

    return iq, qx_mesh, qy_mesh, qz_mesh

def plot_3D_grid(density_grid, x_mesh, y_mesh, z_mesh, cmap, threshold_pct=98, num_levels=10):
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
    normalized_values = values / np.max(values)

    # Get the absolute coordinates
    x_abs = x_mesh[y, x, z]
    y_abs = y_mesh[y, x, z]
    z_abs = z_mesh[y, x, z]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the number of levels of opacity
    opacities = np.linspace(0.01,0.2,num_levels)

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.3, 1, num_levels))
    
    for i in range(num_levels):
        # Calculate the opacity for the current level
        opacity = opacities[i]
        color = colors[i]
        # Determine the data points that fall into the current opacity level
        mask = (normalized_values > (i / num_levels)) & (normalized_values <= ((i + 1) / num_levels))
        
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