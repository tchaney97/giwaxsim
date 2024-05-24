import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fabio
import numpy as np
from scipy.interpolate import griddata

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def match_qmap_index(qxy_val, qz_val, qxy, qz):
    """
    Finds the closest array indices in a qmap defined by axes qxy and qz for 
    given qxy and qz values.

    This function identifies the indices of the qxy and qz arrays that are 
    closest to a specified qxy_val and qz_val, respectively. It's useful for 
    mapping from continuous q-space to the discrete indices used in a 2D qmap 
    array representing GIWAXS data or similar.

    Parameters:
    - qxy_val (float): The qxy value for which the closest index is desired.
    - qz_val (float): The qz value for which the closest index is desired.
    - qxy (np.ndarray): 1D array of qxy vals defining the horizontal axis qmap.
    - qz (np.ndarray): 1D array of qz vals defining the vertical axis of qmap.

    Returns:
    - tuple: qz_idx, qxy_idx
    """
    qxy_idx = (np.abs(qxy - qxy_val)).argmin()
    qz_idx = (np.abs(qz - qz_val)).argmin()
    
    return qz_idx, qxy_idx

def normalize_qmap(qmap1, qmap2):
    """
    Scales the intensity of qmap1 so that the sum is equal to sum of qmap2
    """
    qmap1_max = np.nansum(qmap1)
    qmap2_max = np.nansum(qmap2)
    qmap_out = qmap1 * (qmap2_max/qmap1_max)
    
    return qmap_out

def mask_forbidden_pixels(image, qxy, qz, alpha_i_deg, energy):
    """
    Masks out pixels in a simulated GIWAXS image for inaccessible
    (qxy, qz) values due to the "missing wedge" in experiment geometry.
    
    Calculates minimum qxy values for each qz based on X-ray wavelength 
    (from energy) and incident angle (alpha_i). Pixels below this minimum 
    qxy are masked as np.nan, indicating inaccessible regions in reciprocal space.

    Parameters:
    - image (np.ndarray): 2D GIWAXS image array to be masked.
    - qxy (np.ndarray): 1D array of qxy values (horizontal axis of 'image').
    - qz (np.ndarray): 1D array of qz values (vertical axis of 'image').
    - alpha_i_deg (float): Incident X-ray angle in degrees.
    - energy (float): X-ray energy in electronvolts (eV).

    Returns:
    - np.ndarray: Modified copy of 'image' with inaccessible regions masked.

    Note:
    - X-ray wavelength (wl) calculated as wl = 12400 / energy (Ångströms).
    - Assumes 'qxy' and 'qz' arrays represent linearly spaced values 
      corresponding to 'image' axes.
    - Original 'image' array is not altered; a modified copy is returned.
    """
    #define
    wl = 12400/energy #Å
    alpha_i = np.deg2rad(alpha_i_deg)
    masked_image = np.copy(image)
    
    #calculate qxy minimums. Theta is angle between k_i and k_f
    qxy_mins = np.zeros_like(qz)
    for i, qz_val in enumerate(qz):
        theta = np.arcsin(qz_val*wl/(2*np.pi) - np.sin(alpha_i)) + alpha_i
        qxy_mins[i] = np.abs((2*np.pi/wl)*(np.cos(theta-alpha_i)-(np.cos(alpha_i))))
    
    #make mask based on qxy minimums
    mask=np.full((len(qz),len(qxy)), False)
    for z_idx, qz_val in enumerate(qz):
        mask[z_idx,np.abs(qxy)<qxy_mins[z_idx]]=True

    masked_image[mask]=np.nan
    
    return masked_image

def rebin_and_combine_qmaps(qmap1, qxy1, qz1, qmap2, qxy2, qz2):
    """
    Rebins qmap2 to match the dimensions of qmap1, then combines them into
    a single qmap with qmap1 on the left and qmap2 on the right, split down qxy=0.

    Parameters:
    - qmap1 (np.ndarray): The first qmap array.
    - qxy1 (np.ndarray): qxy axis values for the first qmap.
    - qz1 (np.ndarray): qz axis values for the first qmap.
    - qmap2 (np.ndarray): The second qmap array to be rebinned and combined.
    - qxy2 (np.ndarray): qxy axis values for the second qmap.
    - qz2 (np.ndarray): qz axis values for the second qmap.

    Returns:
    - np.ndarray: The combined qmap with qmap1 on the left and rebinned qmap2 on the right.
    """
    # Create meshgrid for qmap1 and qmap2
    grid1_x, grid1_z = np.meshgrid(qxy1, qz1)
    grid2_x, grid2_z = np.meshgrid(qxy2, qz2)

       # Flatten the meshgrids and qmap2 for interpolation
    points2 = np.vstack((grid2_x.ravel(), grid2_z.ravel())).T
    values2 = qmap2.ravel()

    # Check for matching lengths (this step is for debugging and can be removed later)
    assert points2.shape[0] == values2.shape[0], "The number of points and values must match."

    # Interpolate qmap2 onto the grid defined by qmap1
    qmap2_rebinned = griddata(points2, values2, (grid1_x, grid1_z), method='linear')
    qmap2_rebinned_norm = qmap2_rebinned*(np.nanpercentile(qmap1, 15)/np.nanpercentile(qmap2_rebinned, 75))
    # qmap2_rebinned_norm = normalize_qmap(qmap2_rebinned, qmap1)

    # Combine the qmaps
    # Find the index of qxy=0 in qxy1
    zero_idx = np.abs(qxy1).argmin()
    # Use the left half of qmap1 and right half of rebinned qmap2
    combined_qmap = np.copy(qmap1)
    combined_qmap[:, zero_idx:] = qmap2_rebinned_norm[:, zero_idx:]

    return combined_qmap

def mirror_qmap_positive_qxy_only(qmap, qxy, qz):
    """
    Modifies a qmap to discard pixels with qxy < 0, then creates a mirrored
    qmap for the remaining qmap with only positive qxy values, by reflecting
    the data across qxy=0.
    
    Parameters:
    - qmap (np.ndarray): The qmap array with dimensions corresponding to qz by qxy.
    - qxy (np.ndarray): The 1D array of qxy values, which may include negative values.
    - qz (np.ndarray): The 1D array of qz values.

    Returns:
    - np.ndarray: The mirrored qmap including only originally positive qxy values and their mirror.
    - np.ndarray: The new qxy array including the mirrored positive values.
    - np.ndarray: The qz array (unchanged).
    """
    # Filter out negative qxy values and their corresponding data in the qmap
    positive_qxy_indices = qxy >= 0
    qxy_positive = qxy[positive_qxy_indices]
    qmap_positive = qmap[:, positive_qxy_indices]
    
    # Mirror the positive qxy values
    qxy_mirrored = np.concatenate((-qxy_positive[::-1], qxy_positive))
    
    # Duplicate and mirror the positive qmap data across the qxy=0 axis
    qmap_mirrored = np.concatenate((qmap_positive[:, ::-1], qmap_positive), axis=1)
    
    return qmap_mirrored, qxy_mirrored, qz