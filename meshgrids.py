import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from scipy.signal import convolve
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift
from tqdm.auto import tqdm

from utilities import load_xyz, gaussian_kernel
from ptable_dict import ptable

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
                                sigma,
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
    - npySavePath (pathlib.Path): path to save grid segment npy files
    - sigma (float): used to shift coords to origin for gaussian smearing later
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

def get_allowed_vox_breaks(sigma, voxel_size, coords, axis):
    """Find all voxels along axis where there is enough space between point smeared with gaussian"""
    diff_threshold = voxel_size + (3 * sigma)  # Set diff threshold by voxel size + (3*sigma) 
    diff_axis = np.diff(coords, axis=axis)[:, 0]  # Get difference values array
    diff_axis = np.append(diff_axis, 0)  # Add extra zero at the end to make same shape as coords to use as mask
    # Find indices with difference value from previous greater than diff threshold 
    coord_idx_cuts_lo = np.nonzero(diff_axis>diff_threshold)[0]     # Indices on lower side of difference
    coord_idx_cuts_hi = np.nonzero(diff_axis>diff_threshold)[0] + 1 # Indices on higher side of difference  
    # Get corresponding coordinat values
    taken_coords_lo = coords[:,axis][coord_idx_cuts_lo]  
    taken_coords_hi = coords[:,axis][coord_idx_cuts_hi]  

    # We want to find the halfway point between the taken coords: 
    empty_coords = (coords[:,0][coord_idx_cuts_lo] + coords[:,0][coord_idx_cuts_hi]) / 2 # take average between lo & hi values elementwise
    allowed_voxel_breaks_in_material = (empty_coords / voxel_size).astype(int)   
    
    return allowed_voxel_breaks_in_material

def get_flexible_voxel_breaks(segment_vox_size, min_ax_size, allowed_vox_breaks, x_axis):
    """Based on allowed voxel points, define voxel slices along specified axis"""
    # Target voxel breakpoints (not including 0 & end)
    vox_break_targs = np.arange(segment_vox_size, min_ax_size, segment_vox_size)  

    # Generate actual allowed break points
    vox_breaks = np.array([0, min_ax_size])  # Add 0 & end first
    for vox_break_targ in vox_break_targs:
        if vox_break_targ > allowed_vox_breaks[-1]: 
            # Target is fine if it is beyond material in voxel space
            vox_breaks = np.append(vox_breaks, vox_break_targ)
        else:
            # Otherwise, look for closest value in allowed voxel breaks
            vox_break_diffs = np.abs(allowed_vox_breaks - vox_break_targ) 
            # Pull out closest values, these will likely shift the target values
            vox_break_shifted = allowed_vox_breaks[vox_break_diffs==vox_break_diffs.min()][0]
            vox_breaks = np.append(vox_breaks, vox_break_shifted)
    vox_breaks.sort()  # Put 0 & end in order

    # Reshape breakpoints into tuples of min/max
    vox_axis_mins = vox_breaks[:-1]
    vox_axis_maxs = vox_breaks[1:]
    vox_axis_slices = [(vox_axis_min,vox_axis_max) for vox_axis_min,vox_axis_max in zip(vox_axis_mins,vox_axis_maxs)]
    x_mins = x_axis[vox_axis_mins]
    x_maxs = x_axis[vox_axis_maxs[:-1]]
    x_maxs = np.append(x_maxs, x_axis[-1])

    return vox_axis_slices, x_mins, x_maxs

def flexible_density_grid_npys_along_x(xyz_path,
                                       npySavePath, 
                                       voxel_size,
                                       sigma,
                                       segment_vox_size,
                                       min_ax_size = 256):
    """
    Generates a 3D voxelized electron density grid from .xyz file. Targeted for 
    large arrays where memory would be an issue to load the whole array, so 
    segments along the x direction are saved as npy files and to be reloaded as 
    a dask object later (already persisted). With a target 'segment_vox_size', 
    the actual segment voxels will be adjusted to split between atoms.
    See related functions to segment along y or z dimensions. 
    
    Parameters:
    - xyz_path (str or pathlib.Path): path to xyz file of molecule, NP, etc
    - npySavePath (pathlib.Path): path to save grid segment npy files
    - voxel_size (float): size of voxels in angstroms
    - sigma (float or None): for applying gaussian convolution (and buffer 
                             voxels). No convolution if sigma==None.
    - segment_vox_size (int): target number of voxels per numpy segment, these
                              will be adjusted based on the material input 
                              and sigma value to segment between atoms
                              This should be a reasonable value for how much 
                              RAM your system has. 

                              e.g.: ( 512,  128,  512) floats =  256 MB RAM
                                    ( 512,  512,  512) floats =  1   GB RAM
                                    (1024,  128, 1024) floats =  1   GB RAM
                                    (1024, 1024, 1024) floats =  8   GB RAM
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
    # Extracting the atomic symbols and positions from the xyz file
    coords, symbols = load_xyz(xyz_path)

    # Shift coords array to origin (buffer ensures room for Gaussian smearing)
    buffer = 3 * sigma # same size as guassian kernel (made later)
    coords[:,0] -= np.min(coords[:,0])-buffer
    coords[:,1] -= np.min(coords[:,1])-buffer
    coords[:,2] -= np.min(coords[:,2])-buffer

    # Axis grids
    grid_size_x = int(np.ceil((np.max(coords[:,0])+buffer)/voxel_size))
    grid_size_y = int(np.ceil((np.max(coords[:,1])+buffer)/voxel_size))
    grid_size_z = int(np.ceil((np.max(coords[:,2])+buffer)/voxel_size))

    # Calcuate number of voxel grid points, pad to nearest 2^n
    grid_vox_x = 1 << (grid_size_x - 1).bit_length()
    grid_vox_y = 1 << (grid_size_y - 1).bit_length()
    grid_vox_z = 1 << (grid_size_z - 1).bit_length()
    if grid_vox_x < min_ax_size:
        grid_vox_x = min_ax_size
    if grid_vox_y < min_ax_size:
        grid_vox_y = min_ax_size
    if grid_vox_z < min_ax_size:
        grid_vox_z = min_ax_size

    # Create axes
    x_axis = np.linspace(0, grid_vox_x*voxel_size, grid_vox_x)
    y_axis = np.linspace(0, grid_vox_y*voxel_size, grid_vox_y)
    z_axis = np.linspace(0, grid_vox_z*voxel_size, grid_vox_z)

    # Sort coordinates & symbols by the specifed axis coordinate value (by entering as the last column in np.lexsort)
    axis = 0  # x values
    ind = np.lexsort((coords[:,2], coords[:,1], coords[:,0]))
    symbols = symbols[ind]
    coords = coords[ind]

    # Find all voxels along axis where there is enough space between point smeared with gaussian
    allowed_vox_breaks = get_allowed_vox_breaks(sigma, voxel_size, coords, axis)

    # Based on allowed voxel points above, define voxel slices along specified axis
    vox_axis_slices, x_mins, x_maxs = get_flexible_voxel_breaks(
                    segment_vox_size, min_ax_size, allowed_vox_breaks, x_axis)

    # Loop over each vox segment to populate
    grid_vox_segments = np.array([])  # Running total of each segment size
    for i, vox_axis_slice in enumerate(tqdm(vox_axis_slices, desc='Populating, smearing, & saving segments')):
        # Get segment size
        grid_vox_axis_segment = vox_axis_slice[1] - vox_axis_slice[0]
        # Add segment size to running total
        grid_vox_segments = np.append(grid_vox_segments, grid_vox_axis_segment)
        
        # Create empty grid segment
        density_grid_segment = np.zeros((grid_vox_y, grid_vox_axis_segment, grid_vox_z))
        
        # Find x min & max (real space units)
        x_min = x_mins[i]
        x_max = x_maxs[i]

        # Downselect pre-sorted coords & symbols segment to apply to loop
        segment_coords_mask = (coords[:,axis] >= x_min) & (coords[:,axis] < x_max)
        segment_coords = coords[segment_coords_mask]
        segment_symbols = symbols[segment_coords_mask]

        # Populate the grid 
        for coord, symbol in zip(segment_coords, segment_symbols):
            grid_coord = np.round((coord / voxel_size),0).astype('int')
            density_grid_segment[ grid_coord[1], 
                                 (grid_coord[0]-(int(grid_vox_segments.sum()))),  # shift the grid coord to be relative to segment not global 
                                  grid_coord[2] ] += (ptable[symbol])  
            
        # Create a Gaussian kernel
        if sigma:
            sigma_voxel = sigma/voxel_size
            kernel_size = 6 * sigma_voxel + 1  # Ensure the kernel size covers enough of the Gaussian
            gaussian_kernel_3d = gaussian_kernel(kernel_size, sigma_voxel)
            # convolve gaussian with 
            density_grid_segment = convolve(density_grid_segment, gaussian_kernel_3d, mode='same')
            
        npy_savename = f'grid_segment_along-x_sigma-{sigma}_num-{i}_shape-{grid_vox_y}-{grid_vox_axis_segment}-{grid_vox_z}.npy'
        np.save(npySavePath.joinpath(npy_savename), density_grid_segment)   

    return x_axis, y_axis, z_axis, grid_vox_x, grid_vox_y, grid_vox_z

# # Future functions for npys along y, along z:
# def flexible_density_grid_npys_along_yz():
#     # Sort coordinates & symbols by the specifed axis coordinate value (by entering as the last column in np.lexsort)
#     # Extract int for axis to chunk
#     # if axis_to_chunk == 'x':
#     axis = 0
#     ind = np.lexsort((coords[:,2], coords[:,1], coords[:,0]))
#     # elif axis_to_chunk == 'y':
#     #     axis = 1
#     #     ind = np.lexsort((coords[:,2], coords[:,0], coords[:,1]))
#     # elif axis_to_chunk == 'z':
#     #     axis = 2
#     #     ind = np.lexsort((coords[:,0], coords[:,1], coords[:,2]))    
#     elif axis_to_chunk=='y':
#         # density_grid_segment = np.zeros((grid_vox_axis_segment, grid_vox_x, grid_vox_z))
#         print('not implemented yet')
#     elif axis_to_chunk=='z':
#         # density_grid_segment = np.zeros((grid_vox_y, grid_vox_x, grid_vox_axis_segment))
#         print('not implemented yet')

# # Doesn't yet work... the below commands must be run in notebook
# def load_array_from_npy_stack(npy_paths):
#     arrs = []
#     for npy_path in npy_paths:
#         arr = np.load(npy_path)
#         arrs.append(arr)

#     return np.concatenate(arrs, axis=1)   

# def load_npy_files_to_dask(dask, npySavePath, grid_vox_x, grid_vox_y, grid_vox_z):
#     """
#     Load npy files back in as a dask array:
    
#     Inputs:
#     - dask: the full dask module, probably a better way to do this...
#     - npySavePath: pathlib.Path to the npy files
#     """
#     npy_paths = sorted(npySavePath.glob('*'))
#     density_grid = dask.delayed(load_array_from_npy_stack)(npy_paths)
#     density_grid = dask.array.from_delayed(density_grid, shape=(grid_vox_y, grid_vox_x, grid_vox_z), dtype=float)
#     density_grid = density_grid.rechunk((grid_vox_y, int(grid_vox_x/8), grid_vox_z))

#     return density_grid.persist()    
