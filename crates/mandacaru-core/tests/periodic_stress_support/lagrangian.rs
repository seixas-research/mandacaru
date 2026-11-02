//! The frozen ENERGY as one function of the strain: `F(eps) = E[X(eps); eps]` with the Lowdin
//! carry `X = S(eps)^{-1/2} y` (`y = S(0)^{1/2} C0` and the occupations `f` fixed), every piece
//! rebuilt at the strained cell from the setup (`oracle`'s builders) and the density and the
//! channel moments re-formed from `D(eps) = X f X^+`. This is the plan's rung-3 oracle: it
//! differences the whole nonlinear functional, so it sees a term that is missing from the
//! chain-rule decomposition, which the piece-by-piece oracle cannot. `-TS` is constant at fixed
//! `f` and is not in it: `F(0)` is `scf::state_energy`, not the free energy (the entropy term of
//! the SCF is 4e-10 Ha on the LiH state). One k-point.
#![allow(dead_code)]

use super::{matrix3, oracle, xc, Eps, Frame, World};
use crate::{
    setup_pulay_support::{form, trace},
    strain_samples_support::Carried,
};
use mandacaru_core::periodic::setup::{
    basis_gradient::{short_range_cross, ShortRange},
    pulay::{kinetic_form, projection, Matrix, Mesh},
    shells::{DenseRows, Sampling},
};
use nalgebra::linalg::SymmetricEigen;
use num_complex::Complex64;

/// `S^{power}` of a Hermitian positive matrix.
fn power(s: &Matrix, power: f64) -> Matrix {
    let eigen = SymmetricEigen::new(s.clone());
    let scaled = Matrix::from_fn(s.nrows(), s.ncols(), |i, j| {
        eigen.eigenvectors[(i, j)] * Complex64::new(eigen.eigenvalues[j].powf(power), 0.)
    });
    scaled * eigen.eigenvectors.adjoint()
}

/// The overlap `S(eps)` at the strained cell, the Bloch sums and the projections on the way.
struct Cell {
    psi: Vec<Complex64>,
    projections: Matrix,
    overlap: Matrix,
}

fn cell(w: &World, mesh: &Mesh, carried: &Carried) -> Cell {
    let (ctx, data) = (w.ctx(), &w.data);
    let (n, m) = (mesh.len(), data.crystal.m);
    let mut psi = vec![Complex64::new(0., 0.); m * n];
    carried
        .shells
        .values(
            ctx,
            &Sampling {
                lattice: &carried.lattice,
                region: mesh.region(),
                kpoints: &[carried.k],
            },
            mesh.points(),
            &mut DenseRows::new(&mut psi, m, n),
        )
        .unwrap();
    let projections = projection(
        ctx,
        &carried.shells,
        &carried.lattice,
        carried.k,
        &carried.projectors,
    )
    .unwrap();
    let overlap = form(&psi, &psi, n, mesh.dv, None)
        + &projections * &data.augmentation * projections.adjoint();
    Cell {
        psi,
        projections,
        overlap,
    }
}

/// `y = S(0)^{1/2} C0` for the coefficients of `w.bands[0]`.
pub fn carried_coefficients(w: &World) -> Matrix {
    let zero = [[0.; 3]; 3];
    let frame = Frame::new(w, &zero);
    let mesh = Mesh::new(w.ctx(), &frame.grid).unwrap();
    let carried = Carried::new(
        &w.data.lattice,
        &w.data.shells,
        &w.data.projectors,
        &matrix3(&zero),
        w.kpoints[0],
    );
    let reference = cell(w, &mesh, &carried);
    power(&reference.overlap, 0.5) * &w.bands[0].coefficients
}

/// `F(eps)`: the band energy, the electrostatics (`list` is the pinned dense list), the xc energy
/// and the constants, for the density the carried orbitals make at the strained cell. `w.rho`
/// and `w.q` are replaced by the strained cell's own for the duration of the call.
pub fn energy(w: &mut World, eps: &Eps, y: &Matrix, list: &[[i64; 3]]) -> f64 {
    assert_eq!(w.kpoints.len(), 1, "one k-point");
    let (ctx, m) = (w.ctx().clone(), w.data.crystal.m);
    let frame = Frame::new(w, eps);
    let mesh = Mesh::new(&ctx, &frame.grid).unwrap();
    let n = mesh.len();
    let carried = Carried::new(
        &w.data.lattice,
        &w.data.shells,
        &w.data.projectors,
        &matrix3(eps),
        w.kpoints[0],
    );
    let strained = cell(w, &mesh, &carried);
    let x = power(&strained.overlap, -0.5) * y;
    let bands = &w.bands[0];
    let scaled = Matrix::from_fn(m, x.ncols(), |i, j| {
        x[(i, j)] * Complex64::new(bands.occupations[j], 0.)
    });
    let density = &scaled * x.adjoint();
    let weight = 2.0 * w.weights[0];
    // The density at the nodes and the channel moments of the carried state.
    let rho: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = Complex64::new(0., 0.);
            for mu in 0..m {
                for nu in 0..m {
                    s += strained.psi[mu * n + i].conj()
                        * density[(nu, mu)]
                        * strained.psi[nu * n + i];
                }
            }
            weight * s.re
        })
        .collect();
    let r = strained.projections.adjoint() * &density * &strained.projections;
    let q: Vec<Complex64> = w
        .data
        .blocks
        .iter()
        .map(|b| {
            let mut t = Complex64::new(0., 0.);
            for i in 0..b.nrows() {
                for j in 0..b.ncols() {
                    t += b[(i, j)] * r[(j, i)];
                }
            }
            t * weight
        })
        .collect();
    // The band energy: kinetic + short range + projections against D_ion.
    let kinetic = kinetic_form(
        &ctx,
        &frame.grid,
        &mesh,
        carried.k,
        (&strained.psi, m),
        (&strained.psi, m),
    )
    .unwrap();
    let mut short = Matrix::zeros(m, m);
    for (a, atom) in frame.atoms.iter().enumerate() {
        let sphere = ShortRange {
            center: atom.center,
            potential: &w.data.potentials[a],
            sigma: w.data.sigma,
        };
        let block = short_range_cross(
            &ctx,
            &carried.lattice,
            &[carried.k],
            &sphere,
            &carried.shells,
            None,
            &carried.shells,
        )
        .unwrap();
        short += Matrix::from_row_slice(m, m, block.as_slice());
    }
    let fixed = kinetic
        + short
        + &strained.projections * &w.data.ion_operator * strained.projections.adjoint();
    let band = weight * trace(&density, &fixed);
    // The electrostatics and the xc energy of the carried density.
    let (old_rho, old_q) = (
        std::mem::replace(&mut w.rho, rho),
        std::mem::replace(&mut w.q, q),
    );
    let electrostatic = oracle::hartree_energy(w, [eps; 4])
        + oracle::dense_energy(w, eps, list)
        + oracle::linear_energy(w, eps)
        + oracle::constants_energy(w, eps);
    let core = w.core(&frame.grid, &frame.centers());
    let xc_energy = xc::evaluate(
        &ctx,
        w.grid(),
        &xc::wave(&ctx, &frame.grid),
        frame.grid.index.dv(),
        w.functional,
        w.data.crystal.relativistic,
        &w.rho,
        &core,
    )
    .energy;
    w.rho = old_rho;
    w.q = old_q;
    band + electrostatic + xc_energy + w.data.crystal.constant
}
