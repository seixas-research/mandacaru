//! Cached quantum problems. All energies are Hartree; qubit zero is the MSB.
pub mod algebra;
pub mod evolution;
pub mod optimizer;
pub mod properties;
pub mod solver;
pub mod sparse;
pub use algebra::{Basis, Encoding, Operator};
pub use num_complex::Complex64 as C64;
pub use solver::{CachedProblem, SolveOptions, SolveResult};
