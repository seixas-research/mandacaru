//! Setup of the periodic PAW problem: the compensation shapes, the lattice sums
//! of the compact charges and the Gaussian ions, and the density-independent
//! fields. Atomic units; the inputs are plain structs, not datasets.
pub mod basis_gradient;
pub mod dense;
pub mod electrostatics;
pub mod fields;
pub mod gradients;
pub mod images;
pub mod ions;
pub mod provider;
pub mod pulay;
pub mod remainder;
pub mod shapes;
pub mod shells;
pub mod spheres;
pub mod spin_orbit;
pub mod spinor_tau;
pub mod strain_grid;
pub mod strain_provider;
pub mod strain_pulay;
pub mod strain_samples;
pub mod strain_spheres;
pub mod strain_sums;
pub mod symmetry;
