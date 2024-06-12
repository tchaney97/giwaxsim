import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from scipy.signal import convolve
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftn, fftshift
import glob
from multiprocessing import Pool
import os

from ptable_dict import ptable, atomic_masses
from utilities import write_xyz, load_xyz, rotation_matrix, gaussian_kernel
from meshgrids import generate_density_grid, convert_grid_qspace, plot_3D_grid, downselect_meshgrid, multiply_ft_gaussian
from detector import make_detector, rotate_about_normal, rotate_about_horizontal, rotate_about_vertical
from detector import intersect_detector, rotate_psi_phi_theta, mirror_vertical_horizontal, generate_detector_ints

dirr = '/projects/thch7683/giwaxs_sim_n2200/'
xyz_path = f'{dirr}xyz_files/rectangular_N2200_test2.xyz'
buffer_val = 0.5
voxel_size = 0.3
dens_grid, x_axis, y_axis, z_axis = generate_density_grid(xyz_path, buffer_val, voxel_size, min_ax_size=1500)

iq, qx, qy, qz = convert_grid_qspace(dens_grid, x_axis, y_axis, z_axis)

#free up memory
del dens_grid

# optional downselect iq meshgrid based on max q desired
max_q = 2.5
iq_small, qx_small, qy_small, qz_small = downselect_meshgrid(iq, qx, qy, qz, max_q)

#optional free up memory
del iq

#reassign variables
iq = iq_small
qx = qx_small
qy = qy_small
qz = qz_small

#apply debye waller real-space gaussian smearing
sigma = 0.2
iq = multiply_ft_gaussian(iq, qx, qy, qz, sigma)

np.save(f'{dirr}n2200_sim1_iq.npy', iq)
np.save(f'{dirr}n2200_sim1_qx.npy', qx)
np.save(f'{dirr}n2200_sim1_qy.npy', qy)
np.save(f'{dirr}n2200_sim1_qz.npy', qz)

save_path = f'{dirr}det_output_files/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

det_save_path = f'{save_path}face_on/'
if not os.path.exists(det_save_path):
    os.mkdir(det_save_path)

det_pixels = (500,500) #horizontal, vertical
det_qs = (2.5,2.5) #horizontal, vertical 

#set up detector
det_x, det_y, det_z, det_h, det_v = make_detector(det_qs[0], det_pixels[0], det_qs[1], det_pixels[1])
np.save(f'{save_path}det_h.npy', det_h)
np.save(f'{save_path}det_v.npy', det_v)

#set up rotations to capture disorder in your film. psi=tilting, phi=fiber texture
#only need 1/4 of your total rotation space since symmetry allows us to mirror quadrants
psis = np.linspace(90,80,num=11) #rotation in degrees of detector about detector normal axis
phis = np.linspace(0,180,num=180)[:-1] #rotation in degrees of detector about detector vertical axis
theta = 0 #rotation in degrees of detector about detector horizontal axis

args = [(iq, qx, qy, qz, det_h, det_v, det_x, det_y, det_z, psi, phi, theta, det_save_path) for psi in psis for phi in phis]
with Pool(processes=64) as pool:
    filenames = pool.map(generate_detector_ints, args)

det_files = filenames
for i, det_file in enumerate(det_files):
    det_img = np.load(det_file)
    if i == 0:
        det_sum = det_img
    else:
        det_sum += det_img
#fold detector sum image to capture full disorder space
det_sum = mirror_vertical_horizontal(det_sum)
np.save(f'{save_path}det_sum.npy', det_sum)

fig, ax1 = subplots()
cax = ax1.imshow(det_sum,
           norm=matplotlib.colors.LogNorm(vmin=np.percentile(det_sum, 10), vmax=np.percentile(det_sum, 99.9)),
           extent=(np.min(det_h),np.max(det_h),np.min(det_v),np.max(det_v)),
           cmap='turbo',
           origin = 'lower')
ax1.set_xlabel('q horizontal (1/Å)')
ax1.set_ylabel('q vertical (1/Å)')
ax1.set_ylim(bottom=0)
cbar = fig.colorbar(cax, ax=ax1)
plt.tight_layout()
plt.savefig(f'{save_path}det_sum.png', dpi=300)