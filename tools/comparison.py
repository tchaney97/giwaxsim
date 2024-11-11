import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import griddata
from tools.ptable_dict import ptable, aff_dict

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

def normalize_qmap_position(qmap1, qmap2, qxy_axis, qz_axis, qxy_pos, qz_pos):
    """
    Scales the intensity of qmap1 at selected position to equal qmap2
    qmaps must share same qxy and qz axes
    """

    qxy_idx = np.argmin(np.abs(qxy_axis-qxy_pos))
    qz_idx = np.argmin(np.abs(qz_axis-qz_pos))
    qmap1_int = qmap1[qz_idx, qxy_idx]
    qmap2_int = qmap2[qz_idx, qxy_idx]
    qmap_out = qmap1 * (qmap2_int/qmap1_int)

    print(f'qxy_idx={qxy_idx}, qz_idx={qz_idx}, qmap1_int={qmap1_int}, qmap2_int={qmap2_int}')
    
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

def rebin_and_combine_qmaps(qmap1, qxy1, qz1, qmap2, qxy2, qz2, pos=0):
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
    - pos (tuple or 0): add tuple of qxy and qz coordinates from qmap1 to normalize to

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
   
    if pos:
        qmap2_rebinned_norm = normalize_qmap_position(qmap2_rebinned, qmap1, qxy1, qz1, pos[0], pos[1])
    else:
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

def add_f0_q_dependence(det_img, det_h, det_v, element):
    """
    
    parameters:
    - det_img (np.ndarray): 2D GIWAXS image array to be masked.
    - det_h (np.ndarray): 1D array of qxy values in 1/Å units (horizontal axis of det_img).
    - det_v (np.ndarray): 1D array of qz values in 1/Å units (vertical axis of det_img).
    - element: (str) elemental symbol, some ionic species accepted ex: Li or Li1+
    """
    det_img_new = det_img
    for i, qz in enumerate(det_v):
        for j, qxy in enumerate(det_h):
            q = np.sqrt(qxy**2 + qz**2)
            aff = aff_dict[element]
            # table 6.1.1.4 from https://it.iucr.org/Cb/ch6o1v0001/ 
            fq=aff[0]*np.exp(-aff[1]*(q**2)/(16*np.pi**2))
            fq+=aff[2]*np.exp(-aff[3]*(q**2)/(16*np.pi**2))
            fq+=aff[4]*np.exp(-aff[5]*(q**2)/(16*np.pi**2))
            fq+=aff[6]*np.exp(-aff[7]*(q**2)/(16*np.pi**2))
            fq+=aff[8]

            fq_norm = fq/ptable[element]
            det_img_new[i,j]*=fq_norm
            
    return det_img_new

def rebin_qmap_and_mask(qxy1, qz1, qmap2, qxy2, qz2, qmap2_mask):
    """
    Rebins qmap2 to match the dimensions of qmap1, then combines them into
    a single qmap with qmap1 on the left and qmap2 on the right, split down qxy=0.

    Parameters:
    - qxy1 (np.ndarray): qxy axis values for the first qmap.
    - qz1 (np.ndarray): qz axis values for the first qmap.
    - qmap2 (np.ndarray): The second qmap array to be rebinned and combined.
    - qxy2 (np.ndarray): qxy axis values for the second qmap.
    - qz2 (np.ndarray): qz axis values for the second qmap.
    - qmap2_mask (np.ndarray): array of size(qmap2) where values of 1 are masked, 0 unmasked

    Returns:
    - qmap2_rebinned, qmap2_mask_rebinned 
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

    # rebin mask such that any rebinned pixel that contains a masked pixel is masked
    # Interpolate the mask as well, setting any interpolated pixel with a masked value to 1
    mask_values = qmap2_mask.ravel()
    qmap2_mask_rebinned = griddata(points2, mask_values, (grid1_x, grid1_z), method='linear')

    #change threshold to adjust how mask is rebinned. >0 will catch all pixels that contain masked region
    qmap2_mask_rebinned = np.where(qmap2_mask_rebinned > 0, 1, 0)  # Threshold to identify masked regions


    return qmap2_rebinned, qmap2_mask_rebinned

def scale_offset_mask_qmap(qmap, qmap_mask, scale=1, offset=0):
    """
    Applies a mask, scaling, and offset to a qmap.

    Parameters:
    - qmap (np.ndarray): The qmap array to be adjusted.
    - qmap_mask (np.ndarray): Mask array where values of 1 are masked, 0 unmasked.
    - scale (float): Scaling factor to apply to qmap values.
    - offset (float): Offset to add to qmap values after scaling.

    Returns:
    - qmap_adjusted (np.ndarray): The qmap after masking, scaling, and offset.
    """

    # Apply scale and offset to the masked qmap
    qmap_adjusted = qmap_masked * scale + offset

    # Apply the mask: set masked regions to NaN to exclude them from scaling and offset
    qmap_masked = np.where(qmap_mask == 1, 0, qmap)

    return qmap_adjusted
