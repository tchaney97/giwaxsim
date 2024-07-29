# GIWAXSim Instruction Manual
This repository contains scripts for generating forward simulations of GIWAXS (Grazing Incidence Wide-Angle X-ray Scattering) data. The simulations are created using molecular `.xyz` files and produce 3D voxel grids of scattering intensity values, which can then be used to generate 2D detector images at various geometries.

![gif of detector intersection of reciprocal space](example_giwaxs/phi_sidebyside.gif)

## Requirements:
- Python 3
- numpy
- matplotlib
- fabio
- scipy

## Usage:
Forward simulations are created through two different scripts: `voxelgridmaker.py` and `detectormaker.py`. These scripts are intended to be run in the command line with a single argument pointing to the configuration file (ex: `python voxelgridmaker.py --config /path/to/config_file.txt`). Details of these scripts and their configuration file formats are described below:

### voxelgridmaker.py:
This script takes a `.xyz` file and converts it into a 3D voxel grid of scattering intensity values with axes in units of Å<sup>-1</sup> through the following steps:
1. Mapping the `.xyz` file onto an electron density voxel grid.
2. Taking the FFT of the electron density voxel grid.
3. Taking amplitude of values, recentering axes, converting to q-units, and applying a general atomic form factor.
4. Cropping reciprocal space voxel grid to relevant q-values and saving them for later use.

Configuration file parameters:\
An example configuration file is in `/config_templates/voxelgridmaker.txt`
- `xyz_path` – (string) path to a `.xyz` file you would like to generate I vs q voxel grid for.
- `gen_name` – (string) a short sample name used to create directories and output files.
- `voxel_size` – (positive float) side length dimension of square real-space voxels in Å.
- `min_ax_size` – (positive integer) minimum number of voxels along each axis.
- `f0_element` – (string) elemental symbol for z-normalized f0(q) scaling.
- `max_q` – (positive float) determines the q-value to which the iq voxel grid is cropped.
- `output_dir` – (string) path to output directory; if not defined, `os.get_cwd()` is used.

Tips:
- It is advantageous to choose an axis length that is larger than the slab described by the `.xyz` file. This “padding” can lower the bin size of the FFT, resulting in better q-resolution.
- For computation speed, `min_ax_length` should ideally be a power of 2. Avoid primes.
- Carefully choose the real-space voxel dimension since it will carry over as q-uncertainty.
- The slabs described by the `.xyz` file should be orthorhombic to allow for the code to properly pad with an average electron density to best prevent termination ripples in reciprocal space.

### detectormaker.py:
This script loads the iq reciprocal space voxel grid and associated axes generated by `voxelgridmaker.py` and uses them to populate scattering intensity on a 2D detector plane at various geometries. These geometries are summed to produce a final “det_sum” as the simulated GIWAXS. The steps are:
1. Initializing detector plane size, resolution, and orientation.
2. Intersecting detector pixels with scattering intensity voxels.
3. Saving detector intensities at that orientation.
4. Rotating the detector and repeating step 3 for all orientations.
5. Summing final detector image of all orientations.

Configuration file parameters:\
An example configuration file is in `/config_templates/detectormaker.txt`
- `iq_output_folder` – (string) output from `generate_iq_voxelgrid.py` (form `name_output_files/`).
- `gen_name` – (string) same `gen_name` used in `voxelgridmaker.py`.
- `max_q` – (positive float) maximum q-value on detector must be ≤ max_q used to make iq file.
- `num_pixels` – (positive integer) number of pixels along each detector axis.
- `angle_init_val1` – (float) 1st initializing detector rotation in degrees about `angle_init_ax1`.
- `angle_init_val2` – (float) 2nd initializing detector rotation in degrees about `angle_init_ax2`.
- `angle_init_val3` – (float) 3rd initializing detector rotation in degrees about `angle_init_ax3`.
- `angle_init_ax1` – (string) rotation axis for 1st initializing rotation; set to none for no rotation.
- `angle_init_ax2` – (string) rotation axis for 2nd initializing rotation; set to none for no rotation.
- `angle_init_ax3` – (string) rotation axis for 3rd initializing rotation; set to none for no rotation.
- `psi_start` – (float) starting value in degrees for psi.
- `psi_end` – (float) ending value in degrees for psi.
- `psi_num` – (positive integer) number of linearly spaced psi steps.
- `phi_start` – (float) starting value in degrees for phi.
- `phi_end` – (float) ending value in degrees for phi.
- `phi_num` – (positive integer) number of linearly spaced phi steps.

Tips:
- Rotation axes are defined as psi, phi, and theta for rotation about detector normal, vertical, and horizontal axes, respectively.
- The detector begins with the vertical axis pointing along positive qz, the horizontal axis along positive qy, and the normal axis along positive qx.
- Use “init” rotations to set up your detector such that psi and phi will capture the disorder you desire. Phi is usually used for fiber texture and psi for orientational disorder.
- Visualization tools are available as jupyter notebooks in `./test_notebooks` to better understand these manipulations.
- Only ¼ of the total rotation space needs to be probed as the GIWAXS detector plane is mirrored about the horizontal and vertical axis after summing.
- For example, if you are trying to match an experimental sample with fiber texture and ±15° tilting about the backbone axis, then you may define `psi_start`, `psi_end`, `psi_num` = (0, 15, 16) and `phi_start`, `phi_end`, `phi_num` = (0, 179, 180).

## Other tools:

### slabmaker.py:
This script takes a `.xyz` periodic unit cell and propagates it to a desired orthorhombic slab size.

Configuration file parameters:\
An example configuration file is in `/config_templates/slabmaker.txt`
- `input_filepath` – (string) path to `.xyz` cell (ex: `./test_xyz_files/graphite_UnitCell.xyz`).
- `output_filepath` – (string) directory where you would like `.xyz` slab saved (optional).
- `gen_name` – (string) same `gen_name` used in `voxelgridmaker.py`.
- `x_size` – (float) size in Å of slab along x-axis.
- `y_size` – (float) size in Å of slab along y-axis.
- `z_size` – (float) size in Å of slab along z-axis.
- `a` – (float) cell side length in Å.
- `b` – (float) cell side length in Å.
- `c` – (float) cell side length in Å.
- `alpha` – (float) cell interior angle in degrees.
- `beta` – (float) cell interior angle in degrees.
- `gamma` – (float) cell interior angle in degrees.

### plotandcompare.py:
Script in progress. Current tools are contained in jupyter notebooks in the `test_notebooks` folder.

### estimateresources.py:
Script in progress. Current tools are contained in jupyter notebooks in the `test_notebooks` folder.
