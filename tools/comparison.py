import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import griddata
from tools.ptable_dict import ptable, aff_dict
from scipy.ndimage import map_coordinates
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from tools.utilities import load_xyz, load_pdb, calc_real_space_abc, get_element_f1_f2_dict, create_shared_array
from tools.voxelgrids import rotate_project_fft_coords, downselect_voxelgrid, add_f0_q_3d
from tools.detector import make_detector, rotate_about_normal, rotate_about_horizontal, rotate_about_vertical, generate_detector_ints, mirror_vertical_horizontal
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

def create_wedge_mask(image, qxy, qz, alpha_i_deg, energy):
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
    - np.ndarray: mask (0=unmasked, 1=masked).

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
    mask=np.full((len(qz),len(qxy)), 0)
    for z_idx, qz_val in enumerate(qz):
        mask[z_idx,np.abs(qxy)<qxy_mins[z_idx]]=1
    
    return mask

def trim_sim_data(sim_det_ints, det_h, det_v, exp_qxy, exp_qz):
    """
    Trims simulated detector intensity data and the corresponding horizontal (det_h)
    and vertical (det_v) detector axes to match the experimental q-space range.

    Parameters:
    - sim_det_ints (np.ndarray): 2D array of simulated detector intensities.
    - det_h (np.ndarray): 1D array of horizontal axis values for the detector.
    - det_v (np.ndarray): 1D array of vertical axis values for the detector.
    - exp_qxy (np.ndarray): 1D array of experimental qxy values (horizontal range).
    - exp_qz (np.ndarray): 1D array of experimental qz values (vertical range).

    Returns:
    - np.ndarray: Trimmed intensity data to match the experimental range.
    - np.ndarray: Trimmed horizontal axis values.
    - np.ndarray: Trimmed vertical axis values.
    """
    # Define bounds based on experimental data
    qxy_min, qxy_max = np.min(exp_qxy), np.max(exp_qxy)
    qz_min, qz_max = np.min(exp_qz), np.max(exp_qz)

    # Create masks to select only the overlapping range in q-space
    h_mask = (det_h >= qxy_min) & (det_h <= qxy_max)
    v_mask = (det_v >= qz_min) & (det_v <= qz_max)
    
    # Apply masks to trim data and axes
    det_h_trim = det_h[h_mask]
    det_v_trim = det_v[v_mask]
    sim_int_trim = sim_det_ints[v_mask, :][:, h_mask]  # Apply row and column masks sequentially
    
    return sim_int_trim, det_h_trim, det_v_trim

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

def mirror_qmap_positive_qxy_only_old(qmap, qxy, qz):
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

def mirror_qmap_positive_qxy_only(qmap, qxy, qz):
    """
    Modifies a qmap to discard pixels with qxy < 0, then creates a mirrored
    qmap for the remaining qmap with only positive qxy values, by reflecting
    the data across qxy=0. Any pixel with a value less than one on the positive
    side is replaced by the corresponding pixel on the negative side. Finally,
    adds the qxy = 0 column to the end of the mirrored qmap.

    Parameters:
    - qmap (np.ndarray): The qmap array with dimensions corresponding to qz by qxy.
    - qxy (np.ndarray): The 1D array of qxy values, which may include negative values.
    - qz (np.ndarray): The 1D array of qz values.

    Returns:
    - np.ndarray: The mirrored qmap including only originally positive qxy values and their mirror,
                  with the qxy=0 column appended at the end.
    - np.ndarray: The new qxy array including the mirrored positive values and qxy=0 at the end.
    - np.ndarray: The qz array (unchanged).
    """

    # Filter out qxy < 0 values to include only qxy >= 0
    non_negative_qxy_indices = qxy > 0
    qxy_positive = qxy[non_negative_qxy_indices]
    qmap_positive = qmap[:, non_negative_qxy_indices]
    
    # Identify the negative qxy values and mirror them for replacement
    negative_qxy_indices = qxy < 0
    qmap_negative_mirror = qmap[:, negative_qxy_indices][:, ::-1]

    # Pad qmap_negative_mirror with zeros on the right to match qmap_positive's shape if necessary
    if qmap_negative_mirror.shape[1] < qmap_positive.shape[1]:
        qmap_negative_mirror = np.pad(
            qmap_negative_mirror, 
            ((0, 0), (0, qmap_positive.shape[1] - qmap_negative_mirror.shape[1])), 
            mode='constant', 
            constant_values=0
        )
    
    # Identify low-value pixels in the positive side (<1)
    low_value_mask = qmap_positive*1.1 < qmap_negative_mirror

    # # Replace low-value pixels with values from the negative side
    qmap_positive[low_value_mask] = qmap_negative_mirror[low_value_mask]

    # # Append the qxy = 0 column at the beginning of qmap_positive
    if np.any(qxy==0):
        qxy_zero_index = np.where(qxy == 0)  # Find index of qxy = 0
        qmap_zero_column = qmap[:, qxy_zero_index].reshape(-1, 1)  # Extract column as 2D
        qmap_positive = np.concatenate((qmap_zero_column, qmap_positive), axis=1)
        qxy_positive = np.concatenate(([0], qxy_positive))  # Append qxy=0 at the end
        
    return qmap_positive, qxy_positive, qz


def mirror_qmap_fix_detgap(qmap, qxy, qz):
    """
    Modifies a qmap to discard pixels with qxy < 0, then creates a mirrored
    qmap for the remaining qmap with only positive qxy values, by reflecting
    the data across qxy=0. Any pixel with a value less than one on the positive
    side is replaced by the corresponding pixel on the negative side. Finally,
    adds the qxy = 0 column to the end of the mirrored qmap.

    Parameters:
    - qmap (np.ndarray): The qmap array with dimensions corresponding to qz by qxy.
    - qxy (np.ndarray): The 1D array of qxy values, which may include negative values.
    - qz (np.ndarray): The 1D array of qz values.

    Returns:
    - np.ndarray: The mirrored qmap including only originally positive qxy values and their mirror,
                  with the qxy=0 column appended at the end.
    - np.ndarray: The new qxy array including the mirrored positive values and qxy=0 at the end.
    - np.ndarray: The qz array (unchanged).
    """

    # Filter out qxy < 0 values to include only qxy >= 0
    non_negative_qxy_indices = qxy > 0
    qxy_positive = qxy[non_negative_qxy_indices]
    qmap_positive = qmap[:, non_negative_qxy_indices]
    
    # Identify the negative qxy values and mirror them for replacement
    negative_qxy_indices = qxy < 0
    qmap_negative_mirror = qmap[:, negative_qxy_indices][:, ::-1]

    # Pad qmap_negative_mirror with zeros on the right to match qmap_positive's shape if necessary
    if qmap_negative_mirror.shape[1] < qmap_positive.shape[1]:
        qmap_negative_mirror = np.pad(
            qmap_negative_mirror, 
            ((0, 0), (0, qmap_positive.shape[1] - qmap_negative_mirror.shape[1])), 
            mode='constant', 
            constant_values=0
        )
    
    # Identify low-value pixels in the positive side (<1)
    low_value_mask = qmap_positive*1.1 < qmap_negative_mirror

    # # Replace low-value pixels with values from the negative side
    qmap_positive[low_value_mask] = qmap_negative_mirror[low_value_mask]

    # # Mirror the positive qxy values
    qxy_mirrored = np.concatenate((-qxy_positive[::-1], qxy_positive))
    qmap_mirrored = np.concatenate((qmap_positive[:, ::-1], qmap_positive), axis=1)

    # # Append the qxy = 0 column at the end
    if np.any(qxy==0):
        qxy_zero_index = np.where(qxy == 0)  # Find index of qxy = 0
        qmap_zero_column = qmap[:, qxy_zero_index].reshape(-1, 1)  # Extract column as 2D
        qmap_mirrored = np.concatenate((qmap_mirrored, qmap_zero_column), axis=1)
        qxy_mirrored = np.append(qxy_mirrored, 0)  # Append qxy=0 at the end
        
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
    qmap_adjusted = qmap * scale + offset

    # Apply the mask: set masked regions to NaN to exclude them from scaling and offset
    qmap_masked = np.where(qmap_mask == 1, 0, qmap_adjusted)

    return qmap_masked

def linear_polar(img, o=None, r=None, output=None, order=1, cont=0):
    """
    Converts a Cartesian image to polar coordinates with x-axis as r and y-axis as theta.
    credit yxdragon: https://forum.image.sc/t/polar-transform-and-inverse-transform/40547
    """
    # Define origin if not provided
    if o is None:
        o = np.array(img.shape[:2]) / 2 - 0.5
    else:
        o = np.array(o)
    # Define radius if not provided
    if r is None:
        r = np.sqrt((np.array(img.shape[:2]) ** 2).sum()) / 2
    # Define output shape: (angle, radius) -> (theta, r)
    if output is None:
        shp = int(round(r * 2 * np.pi)), int(round(r))
        output = np.zeros(shp, dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)

    out_h, out_w = output.shape
    rs = np.linspace(0, r, out_w)
    ts = np.linspace(0, 2 * np.pi, out_h)

    # Calculate Cartesian coordinates from polar coordinates
    xs = rs * np.cos(ts[:, None]) + o[1]
    ys = rs * np.sin(ts[:, None]) + o[0]
    
    # Map coordinates and output polar image
    map_coordinates(img, (ys, xs), order=order, output=output, cval=cont)
    return output

def polar_linear(img, o=None, r=None, output=None, order=1, cont=0):
    """
    Converts a polar image with x-axis as r and y-axis as theta back to Cartesian coordinates.
    credit yxdragon: https://forum.image.sc/t/polar-transform-and-inverse-transform/40547
    """
    # Define radius if not provided
    if r is None:
        r = img.shape[1]
    # Define output shape
    if output is None:
        output_shape = (int(r * 2), int(r * 2))
        output = np.zeros(output_shape, dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    else:
        output_shape = output.shape
    # Default origin if not provided
    if o is None:
        o = np.array(output.shape) / 2 - 0.5
    else:
        o = np.array(o)

    out_h, out_w = output.shape

    # Create a meshgrid for Cartesian output coordinates
    ys, xs = np.mgrid[:out_h, :out_w] - o[:, None, None]
    
    # Convert Cartesian to polar coordinates (r, theta)
    rs = np.sqrt(ys ** 2 + xs ** 2)
    ts = np.arctan2(ys, xs)
    ts[ts < 0] += 2 * np.pi  # Map theta into [0, 2*pi] range

    # Scale r and theta to the polar image's coordinate space
    rs = (rs / r) * (img.shape[1] - 1)
    ts = (ts / (2 * np.pi)) * (img.shape[0] - 1)

    # Map coordinates and output Cartesian image
    map_coordinates(img, (ts, rs), order=order, output=output, cval=cont)
    return output

def add_pad(polar_image, r_axis, pad_width, pad_range):
    pad_range_min, pad_range_max = pad_range
    widths = np.diff(r_axis)
    widths_trim = widths[widths>0]
    pixel_width = np.mean(widths_trim)
    pad_width_pixels = int(np.round(pad_width/pixel_width))
    if pad_width_pixels == 0:
        return polar_image
    pad_start_idx = int(np.argmin(np.abs(r_axis-pad_range_min)))
    pad_end_idx = int(np.argmin(np.abs(r_axis-pad_range_max)))
    pad_range_pixels = pad_end_idx - pad_start_idx
    spacing = int((pad_range_pixels-1)/pad_width_pixels - 1)
    r_axis_copy = r_axis.copy()
    assert spacing >=0, 'pad_range is too small for desired pad_width'
    for i in range(pad_width_pixels):
        pad_idx = pad_start_idx + 2*i + spacing*i
        pad_values = polar_image[:,pad_idx]
        # insert pad values to the right of pad_idx in polar_img
        polar_image = np.insert(polar_image, pad_idx + 1, pad_values, axis=1)
        r_axis_copy = np.insert(r_axis_copy, pad_idx + 1, r_axis_copy[pad_idx])
    # trim end of polar image the same number of columns that were padded
    polar_image = polar_image[:, 0:-pad_width_pixels]

    return polar_image

def shift_peak(image, det_h, det_v, pad_width_qspace, pad_range_qspace):
    #create grid of each axis value in cartesian coordinates
    xx, yy = np.meshgrid(det_h, det_v)

    #convert grid of cartesian values to polar values
    r, chi = np.hypot(xx, yy), np.arctan2(yy, xx)

    row,col=image.shape
    radius = float((np.sqrt(row**2+col**2)))
    # Find the index in det_v_trim and det_h_trim closest to zero
    center_row = int(np.argmin(np.abs(det_v)))
    center_col = int(np.argmin(np.abs(det_h)))

    # polar_image = warp_polar(sim_comp_map, center=(center_row, center_col), radius=max_radius, scaling='linear')
    polar_image = linear_polar(image, o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    polar_r = linear_polar(r, o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    r_axis = polar_r[0,:]

    polar_image_copy = polar_image.copy()

    #correct π-π peak position for fitting
    polar_image_pad = add_pad(polar_image_copy, r_axis, pad_width_qspace, pad_range_qspace)
    polar_image_pad[polar_image==0]=0

    reconstructed_image = polar_linear(polar_image_pad, o=(center_row, center_col), r=None, output=image.shape)

    return reconstructed_image

def slabmaker_fitting(input_filepath, x_size, y_size, z_size, a, b, c, alpha, beta, gamma):
    
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

    return coords_new, elements_new

def voxelgridmaker_fitting(coords, elements, r_voxel_size, q_voxel_size, max_q, energy, num_cpus=None, fill_bkg=False, smooth=0):
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
    if q_num%2 == 0:
        q_num+=1
    qx = qy = qz = np.linspace(-max_q_diag, max_q_diag, q_num)

    #some calculation to see how many phi angles we need to do
    # Calculate the number of angles needed
    delta_phi_rad = np.arctan(q_voxel_size/max_q_diag)
    phi_num = np.ceil(2*np.pi/delta_phi_rad).astype(int)
    last_phi = 180-(180/phi_num)
    phis = np.linspace(0,last_phi, num=phi_num)


    #calculate f values
    f1_f2_dict = get_element_f1_f2_dict(energy, elements)
    f_values = np.array([f1_f2_dict[element] for element in elements], dtype=complex)
    # xraydb chantler lookup defines f1=f' and f2=f" contrary to convention
    z_values = np.array([ptable[element] for element in elements])
    f_values += z_values

    #calculate avg voxel f
    sum_f_values = np.sum(f_values)
    volume = x_bound * y_bound * z_bound
    avg_voxel_f = (sum_f_values/volume) * r_voxel_size**3

    # Create shared arrays for voxel grid and voxel count with the specified dimensions
    voxel_grid_shm = create_shared_array((q_num, q_num, q_num), 'voxel_grid_shared')
    voxel_grid_count_shm = create_shared_array((q_num, q_num, q_num), 'voxel_grid_count_shared')
    args = [(coords, f_values, phi, grid_size, r_voxel_size, avg_voxel_f, 
             x_bound, y_bound, z_bound, fill_bkg, smooth, qx, qy, qz, 'voxel_grid_shared', 'voxel_grid_count_shared') for phi in phis]

     # Multiprocessing (parallel) slower
    ###
    # with Pool(processes=num_cpus) as pool:
    #     pool.map(rotate_project_fft_coords, args)
    ###

    #multithreading (concurrent) this is faster in most cases
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rotate_project_fft_coords, arg) for arg in args]
        for future in as_completed(futures):
            future.result()

    # Create numpy arrays from the shared memory buffers
    voxel_grid = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_shm.buf)
    voxel_grid_count = np.ndarray((q_num, q_num, q_num), dtype=np.float64, buffer=voxel_grid_count_shm.buf)

    # Final IQ calculation
    master_iq_3D = np.divide(voxel_grid, voxel_grid_count, out=np.zeros_like(voxel_grid), where=voxel_grid_count != 0)

    voxel_grid_shm.close()
    voxel_grid_shm.unlink()
    voxel_grid_count_shm.close()
    voxel_grid_count_shm.unlink()
        
    # Optional downselect iq meshgrid based on max q desired
    iq_small, qx_small, qy_small, qz_small = downselect_voxelgrid(master_iq_3D, qx, qy, qz, max_q)

    # Optional free up memory
    del master_iq_3D
    del qx
    del qy
    del qz
    
    f0_element = 'C'
    iq_small = add_f0_q_3d(iq_small, qx_small, qy_small, qz_small, f0_element)
    
    return iq_small, qx_small, qy_small, qz_small

def detectormaker_fitting(iq, qx, qy, qz, num_pixels, max_q, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path, mirror=True):

    angle_init_val1, angle_init_val2, angle_init_val3 = angle_init_vals
    angle_init_ax1, angle_init_ax2, angle_init_ax3 = angle_init_axs    
    det_pixels = (num_pixels, num_pixels) # horizontal, vertical
    det_qs = (max_q, max_q) # horizontal, vertical

    # Set up detector
    det_x, det_y, det_z, det_h, det_v = make_detector(det_qs[0], det_pixels[0], det_qs[1], det_pixels[1])
    if angle_init_ax1=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val1)
    if angle_init_ax1=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val1)
    if angle_init_ax1=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val1)
        
    if angle_init_ax2=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val2)
    if angle_init_ax2=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val2)
    if angle_init_ax2=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val2)
        
    if angle_init_ax3=='psi':
            det_x, det_y, det_z = rotate_about_normal(det_x, det_y, det_z, angle_init_val3)
    if angle_init_ax3=='phi':
        det_x, det_y, det_z = rotate_about_vertical(det_x, det_y, det_z, angle_init_val3)
    if angle_init_ax3=='theta':
        det_x, det_y, det_z = rotate_about_horizontal(det_x, det_y, det_z, angle_init_val3)

    # Load weights or use default values
    psi_weights = np.load(psi_weights_path) if psi_weights_path else np.ones_like(psis)/len(psis)
    phi_weights = np.load(phi_weights_path) if phi_weights_path else np.ones_like(phis)/len(phis)
    theta_weights = np.load(theta_weights_path) if theta_weights_path else np.ones_like(thetas)/len(thetas)

    # Ensure weights match angle arrays
    assert len(psis) == len(psi_weights), 'psi weights length must equal psi_num'
    assert len(phis) == len(phi_weights), 'phi weights length must equal phi_num'
    assert len(thetas) == len(theta_weights), 'theta weights length must equal theta_num'

    assert np.abs(1-np.sum(psi_weights))<0.01, 'psi weights must sum to 1'
    assert np.abs(1-np.sum(phi_weights))<0.01, 'phi weights must sum to 1'
    assert np.abs(1-np.sum(theta_weights))<0.01, 'theta weights must sum to 1'

    # Create args list with angle-weight pairs
    det_ints_shm = create_shared_array(det_pixels, 'det_ints_shared')
    args_list = [
        (iq, qx, qy, qz, det_x, det_y, det_z, psi, psi_weight, phi, phi_weight, theta, theta_weight, 'det_ints_shared')
        for psi, psi_weight in zip(psis, psi_weights)
        for phi, phi_weight in zip(phis, phi_weights)
        for theta, theta_weight in zip(thetas, theta_weights)
    ]

    try:
        # Threading is faster but keeping the parallel processing code here in case
        # with Pool(processes=num_cpus) as pool:
        # pool.map(generate_detector_ints, args_list)
        # det_sum = np.ndarray(det_pixels, dtype=np.float64, buffer=det_ints_shm.buf)
        
        # Use ThreadPoolExecutor to process each file in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_detector_ints, arg) for arg in args_list]
            for future in as_completed(futures):
                future.result()

        det_sum = np.ndarray(det_pixels, dtype=np.float64, buffer=det_ints_shm.buf)
        # Fold detector sum image to capture full orientational space
        if mirror:
            det_sum = mirror_vertical_horizontal(det_sum)
        det_sum[det_sum != det_sum] = 1e-6
        det_sum[det_sum <= 0] = 1e-6
    
    finally:
        # Ensure that shared memory is properly closed and unlinked
        det_ints_shm.close()
        det_ints_shm.unlink()

    #arbitrary re-scaling for easier fitting
    det_sum *=1e-6
        
    return det_sum, det_h, det_v

#define helper function to optimize scale and offset. Numerical approach does not need starting params
def optimize_scale_offset(sim_map, rebin_map, rebin_mask):
    # Flatten arrays for least squares
    sim_flat = sim_map[rebin_mask == 0].ravel()
    rebin_flat = rebin_map[rebin_mask == 0].ravel()

    # Solve for scale and offset: minimize |scale * sim_flat + offset - rebin_flat|^2
    A = np.vstack([sim_flat, np.ones_like(sim_flat)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, rebin_flat, rcond=None)
    scale, offset = coeffs
    return scale, offset

def evaluate_fit(best_params,  fixed_slab_params, fixed_voxelgrid_params, fixed_detectormaker_params, fixed_exp_params):
    x_size, y_size, z_size = best_params
    input_filepath, a, b, c, alpha, beta, gamma = fixed_slab_params
    r_voxel_size, q_voxel_size, max_q, energy, fill_bkg, smooth = fixed_voxelgrid_params
    num_pixels, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path = fixed_detectormaker_params
    rebin_map, rebin_mask, exp_qxy, exp_qz, pad_width, pad_range = fixed_exp_params

    #generate simulated GIWAXS
    coords_slab, elements_slab = slabmaker_fitting(input_filepath, x_size, y_size, z_size, a, b, c, alpha, beta, gamma)
    iq, qx, qy, qz = voxelgridmaker_fitting(coords_slab, elements_slab, r_voxel_size, q_voxel_size, max_q, energy, num_cpus=None, fill_bkg=fill_bkg, smooth=smooth)
    det_sum, det_h, det_v = detectormaker_fitting(iq, qx, qy, qz, num_pixels, max_q, angle_init_vals, angle_init_axs, psis, psi_weights_path, phis, phi_weights_path, thetas, theta_weights_path, mirror=True)

    #trim simulated data to bounds of experimental data
    sim_int_trim, det_h_trim, det_v_trim = trim_sim_data(det_sum, det_h, det_v, exp_qxy, exp_qz)

    sim_int_trim_pad = shift_peak(sim_int_trim.copy(), det_h_trim, det_v_trim, pad_width, pad_range)

    # Optimize scale and offset
    scale, offset = optimize_scale_offset(sim_int_trim_pad, rebin_map, rebin_mask)

    # Apply scale and offset to sim_comp_map
    scaled_map = scale * sim_int_trim_pad + offset

    sim_comp_map = scaled_map.copy()
    sim_comp_map[rebin_mask==1] = 0

    diff_map = np.where(rebin_map > 1e-10, sim_comp_map - rebin_map, 0)

    return rebin_map, sim_comp_map, diff_map

def plot_fit(savepath, rebin_map, sim_comp_map, diff_map, det_h_trim, det_v_trim, max_q, suptitle):
    #plot results
    fontsize = 12
    cbar_shrink = 0.45
    cbar_pad = 0.05
    cbar_aspect=25
    fig,(ax2, ax1, ax3)=plt.subplots(1,3, figsize=(12,6))
    cax = ax1.imshow(rebin_map,
            norm=matplotlib.colors.Normalize(vmin=np.percentile(rebin_map, 1), vmax=np.percentile(rebin_map, 99.9)),
            extent=(np.min(det_h_trim),np.max(det_h_trim),np.min(det_v_trim),np.max(det_v_trim)),
            cmap='turbo',
            origin = 'lower')
    cbar1 = fig.colorbar(cax, ax=ax1, shrink=cbar_shrink, aspect=cbar_aspect, pad=cbar_pad)
    ax1.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax1.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.set_xlim(0,max_q)
    ax1.set_ylim(0,max_q)
    ax1.set_title('Experimental GIWAXS', fontsize=fontsize)

    cax = ax2.imshow(sim_comp_map,
            norm=matplotlib.colors.Normalize(vmin=np.percentile(rebin_map, 1), vmax=np.percentile(rebin_map, 99.9)),
            extent=(np.min(det_h_trim),np.max(det_h_trim),np.min(det_v_trim),np.max(det_v_trim)),
            cmap='turbo',
            origin = 'lower')
    cbar2 = fig.colorbar(cax, ax=ax2, shrink=cbar_shrink, aspect=cbar_aspect, pad=cbar_pad)
    ax2.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax2.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.set_xlim(0,max_q)
    ax2.set_ylim(0,max_q)
    ax2.set_title('Simulated GIWAXS', fontsize=fontsize)

    cmap_bound = np.max([np.abs(np.percentile(diff_map, 0.1)), np.abs(np.percentile(diff_map, 99.9))])
    cax = ax3.imshow(diff_map,
            norm=matplotlib.colors.Normalize(vmin=cmap_bound*-1, vmax=cmap_bound),
            #    norm=matplotlib.colors.LogNorm(vmin=np.percentile(qmap_compare, 30), vmax=np.percentile(qmap_compare, 99.95)),
            extent=(np.min(det_h_trim),np.max(det_h_trim),np.min(det_v_trim),np.max(det_v_trim)),
            cmap='bwr',
            origin = 'lower')
    cbar3 = fig.colorbar(cax, ax=ax3, shrink=cbar_shrink, aspect=cbar_aspect, pad=cbar_pad)
    cbar3.set_label('(Percent)', fontsize=fontsize)  # Add label to the colorbar
    ax3.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax3.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.set_xlim(0,max_q)
    ax3.set_ylim(0,max_q)
    ax3.set_title('Sim. - Exp.', fontsize=fontsize)


    plt.suptitle(suptitle)

    plt.tight_layout()

    plt.savefig(savepath, dpi=300)

def make_linecut(polar_image, chi_axis, r_axis, lower_chi, upper_chi):
    img_cut = polar_image[np.logical_and(chi_axis>lower_chi, chi_axis<upper_chi),:]
    img_cut[img_cut<0.1] = np.nan
    cut_ints = np.nanmean(img_cut, axis=0)
    cut_qs = r_axis.copy()
    assert len(cut_ints) == len(cut_qs), f'intensity array shape {np.shape(cut_ints)} does not match q array shape {np.shape(cut_qs)}'
    mask = np.ones(len(cut_qs), dtype=bool)
    mask[cut_qs==0] = False
    mask[cut_ints!=cut_ints] = False
    cut_ints = cut_ints[mask]
    cut_qs = cut_qs[mask]

    return cut_qs, cut_ints

def plot_linecuts(savepath, rebin_map, sim_comp_map, det_h_trim, det_v_trim, suptitle):

    #create grid of each axis value in cartesian coordinates
    xx, yy = np.meshgrid(det_h_trim, det_v_trim)

    #convert grid of cartesian values to polar values
    r, chi = np.hypot(xx, yy), np.arctan2(yy, xx)

    row,col=sim_comp_map.shape
    radius = float((np.sqrt(row**2+col**2)))
    # Find the index in det_v_trim and det_h_trim closest to zero
    center_row = int(np.argmin(np.abs(det_v_trim)))
    center_col = int(np.argmin(np.abs(det_h_trim)))

    #sim_image
    polar_image = linear_polar(sim_comp_map, o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    print(np.shape(polar_image))
    polar_image_exp = linear_polar(rebin_map, o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    polar_r = linear_polar(r, o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    polar_chi = linear_polar(chi,  o=(center_row, center_col), r=radius, output=None, order=1, cont=0)
    polar_chi_nans = polar_chi.copy()
    polar_chi_nans[polar_chi==0] = np.nan  
    chi_axis = np.nanmean(polar_chi_nans, axis=1)
    chi_axis *=180/np.pi
    print(np.shape(chi_axis))
    print(np.nanmax(chi_axis))
    polar_r_nans = polar_r.copy()
    polar_r_nans[polar_r==0] = np.nan
    r_axis = np.nanmean(polar_r_nans, axis=0)
    print(np.shape(r_axis))
    print(np.nanmax(r_axis))


    ip_sim_qs, ip_sim_ints = make_linecut(polar_image, chi_axis, r_axis, 0, 30)
    oop_sim_qs, oop_sim_ints = make_linecut(polar_image, chi_axis, r_axis, 60, 90)
    full_sim_qs, full_sim_ints = make_linecut(polar_image, chi_axis, r_axis, 0, 90)

    ip_exp_qs, ip_exp_ints = make_linecut(polar_image_exp, chi_axis, r_axis, 0, 30)
    oop_exp_qs, oop_exp_ints = make_linecut(polar_image_exp, chi_axis, r_axis, 60, 90)
    full_exp_qs, full_exp_ints = make_linecut(polar_image_exp, chi_axis, r_axis, 0, 90)



    #plot results
    fontsize = 12
    fig,(ax1, ax2, ax3)=plt.subplots(1,3, figsize=(12,4))

    ax1.plot(ip_exp_qs, ip_exp_ints, label = 'Experiment')
    ax1.plot(ip_sim_qs, ip_sim_ints, label = 'Simulation')
    ax1.set_xlabel('$\mathregular{q_{r}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax1.set_ylabel('Intensity (a.u.)',fontsize=fontsize)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.set_title('IP comparions', fontsize=fontsize)
    ax1.legend(prop={'size':fontsize})

    ax2.plot(oop_exp_qs, oop_exp_ints, label = 'Experiment')
    ax2.plot(oop_sim_qs, oop_sim_ints, label = 'Simulation')
    ax2.set_xlabel('$\mathregular{q_{r}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax2.set_ylabel('Intensity (a.u.)',fontsize=fontsize)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.set_title('OOP comparions', fontsize=fontsize)
    ax2.legend(prop={'size':fontsize})

    ax3.plot(full_exp_qs, full_exp_ints, label = 'Experiment')
    ax3.plot(full_sim_qs, full_sim_ints, label = 'Simulation')
    ax3.set_xlabel('$\mathregular{q_{r}}$ ($\AA^{-1}$)',fontsize=fontsize)
    ax3.set_ylabel('Intensity (a.u.)',fontsize=fontsize)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.set_title('Full comparions', fontsize=fontsize)
    ax3.legend(prop={'size':fontsize})


    plt.suptitle(suptitle)

    plt.tight_layout()

    plt.savefig(savepath, dpi=300)
