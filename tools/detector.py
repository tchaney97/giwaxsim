import numpy as np
from tools.utilities import rotation_matrix

def make_detector(h_max, num_pixels_h, v_max, num_pixels_v):
    """
    Generates a 2D detector that lies on the qz plane centered about the origin.
    Horizontal dimensions are -h_max, h_max in q and likewise for vertical.

    Parameters:
    - h_max: absolute maximum q-value for the horizontal detector dimension
    - num_pixels_h: number of pixels along horizontal detector dimension
    - v_max: absolute maximum q-value for the vertical detector dimension
    - num_pixels_v: number of pixels along vertical detector dimension

    Returns: 
    - det_x_grid: 2D meshgrid of detector pixel coordinates in q
    - det_y_grid: 2D meshgrid of detector pixel coordinates in q
    - det_z_grid: 2D meshgrid of detector pixel coordinates in q
    - h_axis_vals: 1D numpy array describing q values along horizontal axis
    - v_axis_vals: 1D numpy array describing q values along vertical axis
    """
    
    h_axis_vals = np.linspace(-h_max,h_max, num_pixels_h)
    v_axis_vals = np.linspace(-v_max,v_max, num_pixels_v)
    det_y_grid, det_z_grid = np.meshgrid(h_axis_vals, v_axis_vals)
    det_x_grid = np.zeros_like(det_y_grid)

    return det_x_grid, det_y_grid, det_z_grid, h_axis_vals, v_axis_vals



def rotate_about_normal(det_x_grid, det_y_grid, det_z_grid, psi):
    """
    Rotate the detector about the axis normal to the detector by psi degrees.
    
    Parameters:
    det_x_grid: 2D array of x-coordinates of the detector
    det_y_grid: 2D array of y-coordinates of the detector
    det_z_grid: 2D array of z-coordinates of the detector
    psi: Rotation angle in degrees about the normal axis
    
    Returns:
    rot_x_grid: 2D array of x-coordinates after rotation
    rot_y_grid: 2D array of y-coordinates after rotation
    rot_z_grid: 2D array of z-coordinates after rotation
    """
    # Convert angle to radians
    psi_rad = np.radians(psi)
    
    # Flatten the grids for easy manipulation
    x_flat = det_x_grid.flatten()
    y_flat = det_y_grid.flatten()
    z_flat = det_z_grid.flatten()
    
    # Calculate the normal axis to the detector
    # Use cross product of two vectors on the plane of the detector
    point1 = np.array([det_x_grid[0, 0], det_y_grid[0, 0], det_z_grid[0, 0]])
    point2 = np.array([det_x_grid[0, -1], det_y_grid[0, -1], det_z_grid[0, -1]])
    point3 = np.array([det_x_grid[-1, 0], det_y_grid[-1, 0], det_z_grid[-1, 0]])
    
    vec1 = point2 - point1
    vec2 = point3 - point1
    normal_axis = np.cross(vec1, vec2)
    normal_axis /= np.linalg.norm(normal_axis)  # Normalize to get unit vector
    
    # Build the rotation matrix for the normal axis
    R_normal = rotation_matrix(normal_axis, psi_rad)
    
    # Apply this rotation matrix to our coordinates
    rotated_coords = R_normal @ np.vstack([x_flat, y_flat, z_flat])
    
    # Reshape the final rotated coordinates back to the original shape
    rot_x_grid = rotated_coords[0].reshape(det_x_grid.shape)
    rot_y_grid = rotated_coords[1].reshape(det_y_grid.shape)
    rot_z_grid = rotated_coords[2].reshape(det_z_grid.shape)
    
    return rot_x_grid, rot_y_grid, rot_z_grid

def rotate_about_horizontal(det_x_grid, det_y_grid, det_z_grid, theta):
    """
    Rotate the detector about the horizontal axis of the detector by theta degrees.
    
    Parameters:
    det_x_grid: 2D array of x-coordinates of the detector
    det_y_grid: 2D array of y-coordinates of the detector
    det_z_grid: 2D array of z-coordinates of the detector
    theta: Rotation angle in degrees about the horizontal axis
    
    Returns:
    rot_x_grid: 2D array of x-coordinates after rotation
    rot_y_grid: 2D array of y-coordinates after rotation
    rot_z_grid: 2D array of z-coordinates after rotation
    """
    # Convert angle to radians
    theta_rad = np.radians(theta)
    
    # Flatten the grids for easy manipulation
    x_flat = det_x_grid.flatten()
    y_flat = det_y_grid.flatten()
    z_flat = det_z_grid.flatten()
    
   # Calculate the new rotation axis (horizontal axis)
    point1 = np.array([det_x_grid[0, 0], det_y_grid[0, 0], det_z_grid[0, 0]])
    point2 = np.array([det_x_grid[0, -1], det_y_grid[0, -1], det_z_grid[0, -1]])
    h_axis = point2 - point1
    h_axis /= np.linalg.norm(h_axis)  # Normalize to get unit vector
    
    # Build the rotation matrix for the normal axis
    R_h = rotation_matrix(h_axis, theta_rad)
    
    # Apply this rotation matrix to our coordinates
    rotated_coords = R_h @ np.vstack([x_flat, y_flat, z_flat])
    
    # Reshape the final rotated coordinates back to the original shape
    rot_x_grid = rotated_coords[0].reshape(det_x_grid.shape)
    rot_y_grid = rotated_coords[1].reshape(det_y_grid.shape)
    rot_z_grid = rotated_coords[2].reshape(det_z_grid.shape)
    
    return rot_x_grid, rot_y_grid, rot_z_grid

def rotate_about_vertical(det_x_grid, det_y_grid, det_z_grid, phi):
    """
    Rotate the detector about the vertical axis of the detector by phi degrees.
    
    Parameters:
    det_x_grid: 2D array of x-coordinates of the detector
    det_y_grid: 2D array of y-coordinates of the detector
    det_z_grid: 2D array of z-coordinates of the detector
    phi: Rotation angle in degrees about the vertical axis
    
    Returns:
    rot_x_grid: 2D array of x-coordinates after rotation
    rot_y_grid: 2D array of y-coordinates after rotation
    rot_z_grid: 2D array of z-coordinates after rotation
    """
    # Convert angle to radians
    phi_rad = np.radians(phi)
    
    # Flatten the grids for easy manipulation
    x_flat = det_x_grid.flatten()
    y_flat = det_y_grid.flatten()
    z_flat = det_z_grid.flatten()
    
   # Calculate the new rotation axis (horizontal axis)
    point1 = np.array([det_x_grid[0, 0], det_y_grid[0, 0], det_z_grid[0, 0]])
    point2 = np.array([det_x_grid[-1, 0], det_y_grid[-1, 0], det_z_grid[-1, 0]])
    v_axis = point2 - point1
    v_axis /= np.linalg.norm(v_axis)  # Normalize to get unit vector
    
    # Build the rotation matrix for the normal axis
    R_v = rotation_matrix(v_axis, phi_rad)
    
    # Apply this rotation matrix to our coordinates
    rotated_coords = R_v @ np.vstack([x_flat, y_flat, z_flat])
    
    # Reshape the final rotated coordinates back to the original shape
    rot_x_grid = rotated_coords[0].reshape(det_x_grid.shape)
    rot_y_grid = rotated_coords[1].reshape(det_y_grid.shape)
    rot_z_grid = rotated_coords[2].reshape(det_z_grid.shape)
    
    return rot_x_grid, rot_y_grid, rot_z_grid

def intersect_detector(int_voxels, qx, qy, qz, det_x_grid, det_y_grid, det_z_grid, h_axis_vals, v_axis_vals):
    """
#     This function calculates the intersection of intensity values from a 3D voxel grid by a detector
#     plane defined by three 2D meshgrids. The function returns a 2D array representing the integrated
#     values on the detector plane, as well as 1D arrays for the x and y coordinates of the detector.

#     Parameters:
#     - int_voxels (numpy.ndarray): A 3D array of intensity values in the voxel grid.
#     - qx (numpy.ndarray): A 3D array representing the x-coordinates in the voxel grid.
#     - qy (numpy.ndarray): A 3D array representing the y-coordinates in the voxel grid.
#     - qz (numpy.ndarray): A 3D array representing the z-coordinates in the voxel grid.
#     - det_x_grid: 2D array of x-coordinates of the detector
#     - det_y_grid: 2D array of y-coordinates of the detector
#     - det_z_grid: 2D array of z-coordinates of the detector
#     - h_axis_vals: 1D numpy array describing q values along horizontal axis
#     - v_axis_vals: 1D numpy array describing q values along vertical axis

#     Returns:
#     - det_ints (numpy.ndarray): A 2D array representing the integrated intensity values on the detector.
#     """
    det_ints = np.zeros_like(det_x_grid)
    for row in range(len(v_axis_vals)):
        for col in range(len(h_axis_vals)):
            x_idx = np.argmin(np.abs(qx-det_x_grid[row,col]))
            y_idx = np.argmin(np.abs(qy-det_y_grid[row,col]))
            z_idx = np.argmin(np.abs(qz-det_z_grid[row,col]))
            det_ints[row, col] = int_voxels[y_idx, x_idx, z_idx]

    return det_ints

def intersect_detector(int_voxels, qx, qy, qz, det_x_grid, det_y_grid, det_z_grid):
    """
    This function calculates the intersection of intensity values from a 3D voxel grid by a detector
    plane defined by three 2D meshgrids. The function returns a 2D array representing the integrated
    values on the detector plane, as well as 1D arrays for the x and y coordinates of the detector.

    Parameters:
    - int_voxels (numpy.ndarray): A 3D array of intensity values in the voxel grid.
    - qx (numpy.ndarray): A 3D array representing the x-coordinates in the voxel grid.
    - qy (numpy.ndarray): A 3D array representing the y-coordinates in the voxel grid.
    - qz (numpy.ndarray): A 3D array representing the z-coordinates in the voxel grid.
    - det_x_grid: 2D array of x-coordinates of the detector
    - det_y_grid: 2D array of y-coordinates of the detector
    - det_z_grid: 2D array of z-coordinates of the detector

    Returns:
    - det_ints (numpy.ndarray): A 2D array representing the integrated intensity values on the detector.
    """

    actual_q_voxel = np.diff(qz)[0] #voxels are cubes
    det_x_vals = det_x_grid.flatten()
    det_y_vals = det_y_grid.flatten()
    det_z_vals = det_z_grid.flatten()
    det_x_indices = ((det_x_vals-np.min(qx)) // actual_q_voxel).astype(int)
    det_y_indices = ((det_y_vals-np.min(qy)) // actual_q_voxel).astype(int)
    det_z_indices = ((det_z_vals-np.min(qz)) // actual_q_voxel).astype(int)

    # Ensure indices are within bounds
    det_x_indices = np.clip(det_x_indices, 0, int_voxels.shape[1] - 1)
    det_y_indices = np.clip(det_y_indices, 0, int_voxels.shape[0] - 1)
    det_z_indices = np.clip(det_z_indices, 0, int_voxels.shape[2] - 1)

    # Pull intensity values from the voxel grid using the calculated indices
    det_ints_flat = int_voxels[det_y_indices, det_x_indices, det_z_indices]

    # Reshape the flat 1D array back into the 2D shape of the detector grid
    det_ints = det_ints_flat.reshape(det_x_grid.shape)

    return det_ints

def rotate_psi_phi_theta(det_x, det_y, det_z, psi, phi, theta):
    
    det_x2, det_y2, det_z2 = det_x, det_y, det_z
    # psi = 0 #rotation in degrees of detector about detector normal axis
    det_x2, det_y2, det_z2 = rotate_about_normal(det_x2, det_y2, det_z2, psi)
    # phi = 0 #rotation in degrees of detector about detector vertical axis
    det_x2, det_y2, det_z2 = rotate_about_vertical(det_x2, det_y2, det_z2, phi)
    # theta = 0 #rotation in degrees of detector about detector horizontal axis
    det_x2, det_y2, det_z2 = rotate_about_horizontal(det_x2, det_y2, det_z2, theta)
    
    return det_x2, det_y2, det_z2
    
def mirror_vertical_horizontal(qmap):
    """
    Mirrors the values of a qmap array about the vertical and horizontal axis.
    origin of qmap must be in the center!
    
    Parameters:
    - qmap (np.ndarray): The qmap array

    Returns:
    - np.ndarray: The mirrored qmap
    """
    qmap_lr = np.fliplr(qmap)
    qmap_ud = np.flipud(qmap)
    qmap_ud_lr = np.fliplr(qmap_ud)
    qmap_sum = qmap + qmap_lr + qmap_ud + qmap_ud_lr
    
    # Handle the central row if the number of rows is odd
    if qmap.shape[0] % 2 != 0:
        new_row = qmap[qmap.shape[0]//2,:] + qmap[qmap.shape[0]//2,:][::-1]
        qmap_sum[qmap.shape[0] // 2, :] = new_row*2
    
    # Handle the central column if the number of columns is odd
    if qmap.shape[1] % 2 != 0:
        new_col = qmap[:,qmap.shape[1]//2] + qmap[:,qmap.shape[1]//2][::-1]
        qmap_sum[:, qmap.shape[1] // 2] = new_col*2
    
    if qmap.shape[1] % 2 != 0 and qmap.shape[0] % 2 != 0:
        qmap_sum[qmap.shape[0]//2, qmap.shape[1]//2]=qmap[qmap.shape[0]//2, qmap.shape[1]//2]*4

    return qmap_sum


def generate_detector_ints(args):
    """
    function to generate and save detector intensity.
    Used to map onto many workers with multiprocess(ing) Pool

    input:
    - args: iq,qx,qy,qz,det_h,det_v,det_x,det_y,det_z,psi,phi,theta,save_path

    output:
    - filename: string
    """
    iq, qx, qy, qz, det_x, det_y, det_z, psi, psi_weight, phi, phi_weight, theta, theta_weight, det_save_path = args
    det_x2, det_y2, det_z2 = rotate_psi_phi_theta(det_x, det_y, det_z, psi, phi, theta)
    det_int = intersect_detector(iq, qx, qy, qz, det_x2, det_y2, det_z2)
    det_int *= psi_weight*phi_weight*theta_weight
    # det_int = intersect_detector(iq, qx, qy, qz, det_x2, det_y2, det_z2, det_h, det_v)
    filename = f'{det_save_path}/det_ints_psi{psi*100:.0f}_phi{phi*100:.0f}_theta{theta*100:.0f}.npy'
    np.save(filename, det_int)
    return filename

