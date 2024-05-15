# giwaxs_forward_sim
### this tool is intented to simulate GIWAXS given a simulated or hypothesized structure. The input is a .xyz file of atomic symbols and positions (cartesian 3D). First a voxel grid of electron density is constructed based on these atomic positions, number of electrons per atom, and a guassian smearing of the electron density with user defined sigma (species dependent smearing in the future?). 

### With the voxelized electron density map the tool then constructs a 3d fourier transfrom of the electron density and norm^2 this to give the equivalent of x-ray scattered intensity (note that this code does *not* include q-dependent atomic form factor, f0 nor energy dependent f' and f".)

### Finally a detector plane can be constructed and rotated. The intersecting values of scattered intensity with this detector can then be returned and plotted as an equivalent to what is obtained in GIWAXS experiment

