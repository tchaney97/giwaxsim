# GIWAXSim
This repository contains scripts for generating forward simulations of X-ray scattering including GIWAXS (Grazing Incidence Wide-Angle X-ray Scattering) data. The simulations are created using structure `.xyz` or `.pdb` files and produce 3D voxel grids of scattering intensity values, which can then be used to generate 2D detector images at various geometries.

If you find this code useful for your research please consider citing it: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13508430.svg)](https://doi.org/10.5281/zenodo.13508430)

![gif of detector intersection of reciprocal space](sample_images/sidebyside5.gif)

## Requirements:
- Most simulations can be ran on personal computer, but depending on simulation size and resolution high performance computers may be needed
- Python >= 3.8
- numpy
- matplotlib
- fabio
- scipy
- xraydb

## Usage:
Forward simulations are created through primarily through: `simulate_GIWAXS.py`. This script is intended to be run in the command line with a single argument pointing to the configuration file (ex: `python simulate_GIWAXS.py --config /path/to/config_file.txt`). Details of this script and the configuration file format are described below. Note that whenever a directory is input as a string please do **not** include the trailing `/`:

### simulate_GIWAXS.py: 
This script takes a `.pdb` (preferred) or `.xyz` structure file and generates a sample "detector image" in units of Å<sup>-1</sup>. This detector image represents the X-ray scattering that would be observed for the given structure with defined size sampled over the specified structure orientations. With example config values this runs in <5min on M2 macbook air. 

The script runs through the following steps:
1. Projecting material coordinates onto the y-z plane.
2. Assigning each material coordinate a complex atomic scattering factor "f".
3. Optionally applying edge smoothing the 2D grid of "f" values to prevent termination ripples.
4. Taking the norm squared of 2D FFT on the "f" values, converting axes to q-space in Å<sup>-1</sup>, re-centering, and saving.
5. Rotating the material coordinates by some calculated delta phi and repeating steps 1-4 until 180 degrees rotation is achieved
6. Taking each detector slice and binning it into an evenly spaced 3D I(qx,qy,qz) voxel grid based on rotation. Bins are averaged at the end
8. Cropping reciprocal space voxel grid to relevant q-values.
9. Initializing a new detector plane size, resolution, and orientation.
10. Intersecting detector pixels with scattering intensity voxel grid.
11. Storing detector intensities at that orientation.
12. Rotating the detector and repeating step 3 for all orientations.
13. Summing final detector image of all orientations.

Optionally comparing to experimental data by:
14. Fitting a scale factor and flat background intensity for simulated data
15. Plotting overlayed linecuts between simulated and experimental data

#### Configuration file parameters:
- An example configuration file is in `/config_templates/voxelgridmaker_highmem_config.txt`\
\
<ins>Arguments to define the structure for scattering simulation</ins>
- `input_filepath`=(string) optionally a path to a `.xyz` or `.pdb` file for I(q) voxelgrid from single file
- `input_folder`=(string) optionally a path to a folder of `.pdb` files for average I(q) from many files
- `filetype`=(string) must be identified if specifying input_folder (ex: .pdb).
- `x_size`=(positive float) desired slab size along x-axis in Å.
- `y_size`=(positive float) desired slab size along y-axis in Å.
- `z_size`=(positive float) desired slab size along z-axis in Å.
- `a`=(positive float) input file box side length in Å, only needed for `.xyz` files.
- `b`=(positive float) input file box side length in Å, only needed for `.xyz` files.
- `c`=(positive float) input file box side length in Å, only needed for `.xyz` files.
- `alpha`=(positive float) input file interior angle in degrees, only needed for `.xyz` files.
- `beta`=(positive float) input file interior angle in degrees, only needed for `.xyz` files.
- `gamma`=(positive float) input file interior angle in degrees, only needed for `.xyz` files.\
\
<ins>Arguments to define the creation of reciprocal space voxel grid</ins>
- `r_voxel_size`=(positive float) side length dimension of square real-space voxels in Å.
- `q_voxel_size`=(positive float) side length dimension of square reciprocal-space voxels in Å<sup>-1</sup>.
- `max_q`=(positive float) determines the q-value to which the iq voxel grid is cropped.
- `energy`=(positive float) X-ray energy in eV for simulation of f' and f" scattering factors
- `fill_bkg`=(boolean) flag to fill the padded real-space (needed to reach desired q_voxel_size) with average electron density. (simulate aggregate suspended in electron density matched matrix)
- `smooth`=(positve float) sigma value used to create smooth transition from slab electron density to padded electron density. 0 for no smoothing\
\
<ins>Arguments to define the sampling of reciprocal space voxel grid with a "detector" plane</ins>
- `num_pixels`=(positive integer) number of pixels along each detector axis.
- `angle_init_val1`=(float) 1st initializing detector rotation in degrees about `angle_init_ax1`.
- `angle_init_val2`=(float) 2nd initializing detector rotation in degrees about `angle_init_ax2`.
- `angle_init_val3`=(float) 3rd initializing detector rotation in degrees about `angle_init_ax3`.
- `angle_init_ax1`=(string) rotation axis for 1st initializing rotation; set to none for no rotation.
- `angle_init_ax2`=(string) rotation axis for 2nd initializing rotation; set to none for no rotation.
- `angle_init_ax3`=(string) rotation axis for 3rd initializing rotation; set to none for no rotation.
- `psi_start`=(float) starting value in degrees for psi.
- `psi_end`=(float) ending value in degrees for psi.
- `psi_num`=(positive integer) number of linearly spaced psi steps.
- `psi_weights_path`=(string) optional path to `.npy` file that holds 1D list of weights for each psi
- `phi_start`=(float) starting value in degrees for phi.
- `phi_end`=(float) ending value in degrees for phi.
- `phi_num`=(positive integer) number of linearly spaced phi steps.
- `phi_weights_path`=(string) optional path to `.npy` file that holds 1D list of weights for each phi
- `theta_start`=(float) starting value in degrees for theta.
- `theta_end`=(float) ending value in degrees for theta.
- `theta_num`=(positive integer) number of linearly spaced theta steps.
- `theta_weights_path`=(string) optional path to `.npy` file that holds 1D list of weights for each theta
- `mirror`=(boolean) a flag to mirror final detector image about vertical and horizontal axes.
- `save_folder`=(string) optional path to output directory; if not defined, `os.get_cwd()` is used.\
\
<ins>Optional experimental comparison arguments:</ins>
- `mask_path` = (string) optional path to mask file (`.npy`) same size as experimental detector image. pixels==1 are masked
- `img_path` = (string) optional path to experimental q-converted dector image file (`.tif`)
- `qxy_path` = (string) optional path to qxy axis values in Å<sup>-1</sup> for image file (`.txt`)
- `qz_path` = (string) optional path to qz axis values in Å<sup>-1</sup> for image file (`.txt`)
- `fit_scale_offset` = (boolean) optional flag to optimize the scale and constant background of simulated data to experimental data

#### Tips:
- `r_voxel_size` and `q_voxel_size` determine your q-uncertainty and q-resolution respectively. Choosing a small `r_voxel_size` and small `q_voxel_size` requires very large arrays that will utilize more memory and slow the simulation. Reasonable values for PC use are in example config files
- `aff_num_qs` can determine how accurate your f0(q) values are. Appreciable differences in polymer scattering patterns have been found between using 1 and 5. Note that computation time increases linearly with `aff_num_qs` so it is recommended not to exceed 10.
- Rotation axes are defined as psi, phi, and theta for rotation about detector normal, vertical, and horizontal axes, respectively.
- The detector begins with the vertical axis pointing along positive qz, the horizontal axis along positive qy, and the normal axis along positive qx.
- Use “init” rotations to set up your detector such that psi and phi will capture the disorder you desire. Phi is usually used for fiber texture and psi for orientational disorder.
- Visualization tools are available as jupyter notebooks in `./test_notebooks` to better understand these manipulations.
- For in-plane isotropy (common with spun coat films), only ¼ of the total rotation space needs to be probed as the GIWAXS detector plane is mirrored about the horizontal and vertical axis after summing.
- For example, if you are trying to match an experimental sample with spun coat texture and ±15° tilting about the polymer backbone axis, then you may define `psi_start`, `psi_end`, `psi_num` = (0, 15, 16) and `phi_start`, `phi_end`, `phi_num` = (0, 179, 180).
- weights paths for rotations should sum to 1 and be same length as `psi/phi/theta_num`. It can be made from a pole figure (carefully consider if any correction such as sin(Chi) should be used!). If no path is specified even weighting is applied to all angles

## Legacy: 

### slabmaker.py:
This script takes a `.xyz` periodic unit cell and propagates it to a desired orthorhombic slab size.

Configuration file parameters:\
An example configuration file is in `/config_templates/slabmaker_config.txt`
- `input_filepath`=(string) path to `.xyz` or `.pdb` file containing periodic cell
- `output_filepath`=(string) directory where you would like `.xyz` slab saved (optional).
- `gen_name`=(string) same `gen_name` used in `voxelgridmaker.py`.
- `x_size`=(float) size in Å of slab along x-axis.
- `y_size`=(float) size in Å of slab along y-axis.
- `z_size`=(float) size in Å of slab along z-axis.
- `a`=(float) cell side length in Å.
- `b`=(float) cell side length in Å.
- `c`=(float) cell side length in Å.
- `alpha`=(float) cell interior angle in degrees.
- `beta`=(float) cell interior angle in degrees.
- `gamma`=(float) cell interior angle in degrees.


### voxelgridmaker.py: 
This script takes a `.xyz` or `.pdb` structure file and converts it into a 3D voxel grid of scattering intensity values with axes in units of Å<sup>-1</sup>. With example config values this runs in <5min on M2 macbook air. The script runs through the following steps:
1. Projecting material coordinates onto the y-z plane.
2. Assigning each material coordinate a complex atomic scattering factor "f".
3. Optionally windowing the 2D grid of "f" values to prevent termination ripples.
4. Taking the norm squared of 2D FFT on the "f" values, converting axes to q-space in Å<sup>-1</sup>, re-centering, and saving.
5. Rotating the material coordinates by some calculated delta phi and repeating steps 1-4 until 180 degrees rotation
6. Taking each detector slice and binning it into an evenly spaced 3D I(qx,qy,qz) voxel grid based on rotation. Bins are averaged at the end
8. Cropping reciprocal space voxel grid to relevant q-values and saving them for later use.

Configuration file parameters:\
An example configuration file is in `/config_templates/voxelgridmaker_highmem_config.txt`
- `input_filepath`=(string) optionally a path to a `.xyz` file for I(q) voxelgrid from single file
- `input_folder`=(string) optionally a path to a folder of `.xyz` or `.pdb` files for average I(q) from many files
- `gen_name`=(string) a short sample name used to create directories and output files.
- `r_voxel_size`=(positive float) side length dimension of square real-space voxels in Å.
- `q_voxel_size`=(positive float) side length dimension of square reciprocal-space voxels in Å<sup>-1</sup>.
- `aff_num_qs`=(positive integer) number of q bins to evaluate atomic scattering factor f0(q).
- `energy`=(positive float) X-ray energy in eV for simulation of f' and f" scattering factors
- `max_q`=(positive float) determines the q-value to which the iq voxel grid is cropped.
- `output_dir`=(string) optional path to output directory; if not defined, `os.get_cwd()` is used.
- `num_cpus`=(positive integer) number of cpu cores to utilize for multiprocessing
- `fill_bkg`=(boolean) flag to fill the padded real-space (needed to reach desired q_voxel_size) with average electron density
- `smooth`=(positve float) sigma value used to create smooth transition from slab electron density to padded electron density. 0 for no smoothing
- `fix_dc_offset`=(boolean) flag to check for and fix any bright streak along q-axes that result from "dc offset"
- `scratch_folder`=(string) path to a scratch directory for storing temporary orientation frames deleted during cleanup. Do not include trailing `/`. Default os.getcwd()

Tips:
- `r_voxel_size` and `q_voxel_size` determine your q-uncertainty and q-resolution respectively. Choosing a small `r_voxel_size` and small `q_voxel_size` requires very large arrays that will utilize more memory and slow the simulation. Reasonable values for PC use are in example config files
- `aff_num_qs` can determine how accurate your f0(q) values are. Appreciable differences in polymer scattering patterns have been found between using 1 and 5. Note that computation time increases linearly with `aff_num_qs` so it is recommended not to exceed 10.
- if using tukey windowing the slabs described by the `.xyz` file should be orthorhombic (slabmaker.py can do this for you) 

### detectormaker.py:
This script loads the iq reciprocal space voxel grid and associated axes generated by `voxelgridmaker.py` and uses them to populate scattering intensity on a 2D detector plane at various geometries. These geometries are summed to produce a final “det_sum” as the simulated GIWAXS. With example config values this runs in ~30s on M2 macbook air. The steps are:
1. Initializing detector plane size, resolution, and orientation.
2. Intersecting detector pixels with scattering intensity voxels.
3. Saving detector intensities at that orientation.
4. Rotating the detector and repeating step 3 for all orientations.
5. Summing final detector image of all orientations.

Configuration file parameters:\
An example configuration file is in `/config_templates/detectormaker_config.txt`
- `iq_output_folder`=(string) output from `voxelgridmaker.py` (form `./name_output_files`).
- `gen_name`=(string) same `gen_name` used in `voxelgridmaker.py`.
- `max_q`=(positive float) maximum q-value on detector must be ≤ max_q used to make iq file.
- `num_pixels`=(positive integer) number of pixels along each detector axis.
- `angle_init_val1`=(float) 1st initializing detector rotation in degrees about `angle_init_ax1`.
- `angle_init_val2`=(float) 2nd initializing detector rotation in degrees about `angle_init_ax2`.
- `angle_init_val3`=(float) 3rd initializing detector rotation in degrees about `angle_init_ax3`.
- `angle_init_ax1`=(string) rotation axis for 1st initializing rotation; set to none for no rotation.
- `angle_init_ax2`=(string) rotation axis for 2nd initializing rotation; set to none for no rotation.
- `angle_init_ax3`=(string) rotation axis for 3rd initializing rotation; set to none for no rotation.
- `psi_start`=(float) starting value in degrees for psi.
- `psi_end`=(float) ending value in degrees for psi.
- `psi_num`=(positive integer) number of linearly spaced psi steps.
- `phi_start`=(float) starting value in degrees for phi.
- `phi_end`=(float) ending value in degrees for phi.
- `phi_num`=(positive integer) number of linearly spaced phi steps.
- `theta_start`=(float) starting value in degrees for theta.
- `theta_end`=(float) ending value in degrees for theta.
- `theta_num`=(positive integer) number of linearly spaced theta steps.
- `mirror`=(boolean) a flag to mirror final detector image about vertical and horizontal axes. Omit flag for False (writing `mirror=False` is still interpreted as True)
- `cleanup`=(boolean) a flag to automatically delete single orientation frames after averaging (can range 1-100s of gb). Omit flag for False (writing `cleanup=False` is still interpreted as True)
- `num_cpus`=(positive integer) number of cpu cores to utilize for multiprocessing
- `scratch_folder`=(string) path to a scratch directory for storing temporary orientation frames deleted during cleanup. Do not include trailing `/`. Default os.getcwd()

Tips:
- Rotation axes are defined as psi, phi, and theta for rotation about detector normal, vertical, and horizontal axes, respectively.
- The detector begins with the vertical axis pointing along positive qz, the horizontal axis along positive qy, and the normal axis along positive qx.
- Use “init” rotations to set up your detector such that psi and phi will capture the disorder you desire. Phi is usually used for fiber texture and psi for orientational disorder.
- Visualization tools are available as jupyter notebooks in `./test_notebooks` to better understand these manipulations.
- For fiber texture, only ¼ of the total rotation space needs to be probed as the GIWAXS detector plane is mirrored about the horizontal and vertical axis after summing.
- For example, if you are trying to match an experimental sample with fiber texture and ±15° tilting about the backbone axis, then you may define `psi_start`, `psi_end`, `psi_num` = (0, 15, 16) and `phi_start`, `phi_end`, `phi_num` = (0, 179, 180).
- If you do not want mirroring you will manually have to comment out code, better solution will be added soon



## To do:
- Add capability for polarization effects 
  - Time=low, complexity=low
- Add memory requirement estimator tool
  - Time=medium, complexity=low
- Convert plotting functions from notebook to script
  - Time=medium, complexity=low
- Check for and remove duplicated atomic positions in slabmaker
  - Time=medium, complexity=medium
- Optional GUI
  - Time=high, complexity=medium
- Convert to classes
  - Time=high, complexity=low
- Supress termination ripples in voxelgridmaker.py
  - Time=medium, complexity=high
- progress bars!
  - Time=low, complexity=low
