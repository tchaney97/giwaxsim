import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from scipy.signal import convolve
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift
from tqdm.auto import tqdm
import pathlib
from utilities import load_xyz, gaussian_kernel
from ptable_dict import ptable
import xrft

def generate_density_grid(xyz_path, sigma, voxel_size, min_ax_size=256):
    """
    Generates a 3D voxelized electron density grid from .xyz file. electron density is 
    smeared using gaussian convolution with width sigma. Smearing is skipped if sigma=0 
    
    Parameters:
    - xyz_path: string, path to xyz file of molecule, NP, etc
    - sigma: peak width where FWHM=2 sqrt(2ln(2))sigma. Set to 0 for no smearing

    Returns:
    - density_grid: 3D meshgrid of electron density values
    - x_axis: 1D array of x coordinate values 
    - y_axis: 1D array of y coordinate values 
    - z_axis: 1D array of z coordinate values 
    """
    # Extracting the atomic symbols and positions from the xyz file
    coords, symbols = load_xyz(xyz_path)

    # Shift coords array to origin (buffer ensures room for Gaussian smearing)
    buffer = 3 * sigma # same size as guassian kernel (made later)
    coords[:,0] -= np.min(coords[:,0])-buffer
    coords[:,1] -= np.min(coords[:,1])-buffer
    coords[:,2] -= np.min(coords[:,2])-buffer

    # axis grids
    grid_size_x = int(np.ceil((np.max(coords[:,0])+buffer)/voxel_size))
    grid_size_y = int(np.ceil((np.max(coords[:,1])+buffer)/voxel_size))
    grid_size_z = int(np.ceil((np.max(coords[:,2])+buffer)/voxel_size))

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
    
    # Create a Gaussian kernel
    if sigma:
        sigma_voxel = sigma/voxel_size
        kernel_size = 6 * sigma_voxel + 1  # Ensure the kernel size covers enough of the Gaussian
        gaussian_kernel_3d = gaussian_kernel(kernel_size, sigma_voxel)
        # convolve gaussian with 
        density_grid = convolve(density_grid, gaussian_kernel_3d, mode='same')

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
    - iq: 3D meshgrid of scattering intensity values
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

    return fig, ax

    # plt.show()

# Dask functions below:
def generate_electron_grid_npys_fixed(xyz_path, 
                                voxel_size,
                                segments,
                                npySavePath,
                                min_ax_size = 256):
    """
    Generates a 3D voxelized electron density grid from .xyz file. Does not apply
    any smearing, electron values will be contained only inside assigned voxels.
    Targeted for large arrays where memory would be an issue to load the whole 
    array, so segments along the x direction are saved as npy files and then
    reloaded as a dask object (already persisted).
    
    Parameters:
    - xyz_path (str or pathlib.Path): path to xyz file of molecule, NP, etc
    - voxel_size (float): size of voxels in angstroms
    - segments (int): number of segments to make along x direction, this will 
                      also be used for the chunk sizes in the dask array
    - npySavePath (str or pathlib.Path): path to save grid segment npy files
    - min_ax_size (int): minimum number of voxels to use for grid, if not 
                         exceeded based on coordinate values & voxel size

    Returns:
    - x_axis: 1D numpy array of x coordinate values 
    - y_axis: 1D numpy array of y coordinate values 
    - z_axis: 1D numpy array of z coordinate values 
    - grid_vox_x: number of voxels along x direction
    - grid_vox_y: number of voxels along y direction
    - grid_vox_z: number of voxels along z direction

    Generates:
    - npy files in specified npySavePath folder
    """

    #set pathlib path
    xyz_path = pathlib.Path(xyz_path)
    npySavePath = pathlib.Path(npySavePath)
    
    # Extracting the atomic symbols and positions from the xyz file
    coords, symbols = load_xyz(xyz_path)

    # Shift coords array to origin (buffer ensures room for Gaussian smearing)
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    coords[:,2] -= np.min(coords[:,2])

    # axis grids
    grid_size_x = int(np.ceil(np.max(coords[:,0])/voxel_size))
    grid_size_y = int(np.ceil(np.max(coords[:,1])/voxel_size))
    grid_size_z = int(np.ceil(np.max(coords[:,2])/voxel_size))

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

    # Populate grid segments:
    # Sort coordinates & symbols by the x coordinate value (by entering as the last column in np.lexsort)
    ind = np.lexsort((coords[:,2], coords[:,1], coords[:,0]))  # return sorted indices values (first value)
    symbols = symbols[ind]
    coords = coords[ind]

    # Loop over each grid segment to populate
    for segment_num in tqdm(range(segments), desc='Populating & saving grid segments'):
        # Create empty grid segment
        grid_vox_x_segment = int(grid_vox_x/segments)
        density_grid_segment = np.zeros((grid_vox_y, grid_vox_x_segment, grid_vox_z))
        
        # Slice x_axis for segment to identify x maximum and minimum
        x_axis_slice = x_axis[ segment_num * grid_vox_x_segment:
                              (segment_num+1) * grid_vox_x_segment ]
        x_min = x_axis_slice[0]
        x_max = x_axis_slice[-1]
        
        # Downselect pre-sorted coords & symbols segment to apply to loop
        segment_coords_mask = (coords[:,0] > x_min) & (coords[:,0] < x_max)
        segment_coords = coords[segment_coords_mask]
        segment_symbols = symbols[segment_coords_mask]

        # Populate the grid 
        for coord, symbol in zip(segment_coords, segment_symbols):
            grid_coord = (coord / voxel_size).astype(int)
            density_grid_segment[ grid_coord[1], 
                                 (grid_coord[0]-(grid_vox_x_segment*segment_num)), 
                                  grid_coord[2]] += (ptable[symbol])
            
        npy_savename = f'grid_segment_along-x_num-{segment_num}_shape-{grid_vox_y}-{grid_vox_x_segment}-{grid_vox_z}.npy'
        np.save(npySavePath.joinpath(npy_savename), density_grid_segment)

    return x_axis, y_axis, z_axis, grid_vox_x, grid_vox_y, grid_vox_z

def xrft_fft(DA, num_chunks):
    fft_yz = xrft.fft(DA, dim=['y','z'], shift=True)  # take dft in y & z direction
    fft_yz_rechunked = fft_yz.chunk({'freq_y':int(len(DA.y))/num_chunks,'x':int(len(DA.x))})  # rechunk along y direction 
    fft_all = xrft.fft(fft_yz_rechunked, dim=['x'], shift=True)  # take dft in x direction
    return fft_all