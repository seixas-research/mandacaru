//! The explicit derivative of the one-body energy of a PAW-LCAO supercell at fixed densities
//! (Bloch-VQE stage 6a, `BLOCH_VQE_PLAN.md` 4.6): the crystal's own force engines contracted with
//! the supercell's densities folded to the mesh.
//!
//! The supercell's one-body energy at a raw-AO density `P` and energy-weighted density `W`,
//! `tr(P h) - tr(W S)`, is `sum_k w_k Re tr[D(k) h(k) - W(k) S(k)]` on the Gamma-centered mesh
//! (`w_k = 1/N`) with the folded densities `D(k) = U [sum_{R,R'} e^{-ik.(R-R')} P_{R,R'}] U^H`
//! in the crystal's (complex-harmonic) rows, `W(k)` the same way: the caller folds them
//! (`molecular::periodic::paw`). `h(k)` is the crystal's bare operator plus the Hartree-Fock
//! one-center operator, `fixed(k) + V_ion + C (sum_a m_a B_a + Delta) C^H`, `m_a = conj(u_ion) +
//! linear` (the fields of zero electron density). Moving a primitive atom moves every image with
//! it, which is what a crystal's derivative is, so the result is `dE_super / dtau` of the
//! collective displacement, per primitive atom.
//!
//! The terms, the electron density entering only linearly:
//! - the matrix terms of [`contract_matrices`] (P1-P5): the overlap against `W`, the kinetic
//!   operator, the ions' local potential and the short-range spheres against `D`, the projections
//!   with `D_ion + sum_a m_a B_a + Delta` against `D` and the augmentation against `W`;
//! - `ion`: `<K rho | d ion>` (`rho` the smooth density of `D`) and `N d e_ion`;
//! - `dense`: `Re (q^+ d u_ion + q d linear)` (`q` the channel moments of `D`);
//! - `constants`: `N d c_ion`.
//!
//! Not here: the electron-electron terms (the hats against `rho`, `q^+ dU q / 2`, the pair
//! densities of the two-body integrals), which are the two-body derivative.
use super::{
    super::{
        gradient::{pulay::contract_matrices, Breakdown, Derivatives, Matrix},
        invalid,
        setup::provider::{Fields, SetupDerivatives},
        solver, Crystal,
    },
    setup::Setup,
};
use crate::{math::fft::Fft3, math::storage::StoredMatrix, quantum::C64, Error, Result};

/// `dE/dtau` per primitive atom of `sum_k w_k Re tr[D(k) h(k) - W(k) S(k)] + N (e_ion + c_ion)`
/// on the crystal `c` of the supercell's mesh (one point per mesh point, weights `1/N`, one scalar
/// spin channel): `densities[k]` and `weighted[k]` are `D(k)` and `W(k)` in the crystal's rows,
/// `onsite` the Hartree-Fock one-center operator (`p x p`), `cells` the `N` of the supercell. The
/// terms of the [`Breakdown`] that belong to the electron-electron energy are zero.
pub fn one_body_gradient(
    c: &Crystal,
    densities: &[Matrix],
    weighted: &[Matrix],
    onsite: &Matrix,
    cells: usize,
) -> Result<Breakdown> {
    if c.spinor || c.spins != 1 || c.hubbard.is_some() || c.hybrid.is_some() {
        return Err(Error::Unsupported(
            "the PAW-LCAO supercell's force takes a scalar crystal with one spin channel, \
             without DFT+U or exact exchange"
                .into(),
        ));
    }
    let nk = c.points.len();
    let (m, p) = (c.m, c.p);
    if densities.len() != nk
        || weighted.len() != nk
        || densities
            .iter()
            .chain(weighted)
            .any(|d| d.shape() != (m, m))
        || onsite.shape() != (p, p)
        || cells != nk
    {
        return Err(invalid(
            "the supercell's force takes one m x m density and energy-weighted density per \
             mesh point of the crystal, and a p x p one-center operator",
        ));
    }
    let ctx = &c.context;
    let setup = Setup::new(c)?;
    let cores = setup.cores()?;
    let mut layout = setup.layout(c);
    // The densities carry both spins; the bare operator has no exchange-correlation term, so the
    // matrix terms are the four of a semilocal functional.
    layout.degeneracy = 1.;
    layout.request.functional = "lda".into();
    let provider = SetupDerivatives::new(ctx, setup.inputs(c, &cores), layout, setup.operators())?;
    let stored = densities
        .iter()
        .map(|d| StoredMatrix::new(ctx, d.clone(), false))
        .collect::<Result<Vec<_>>>()?;
    let n = c.grid.index.points();
    let (rho, q) = solver::density(c, &stored)?;
    let mut fft = Fft3::new(ctx, c.grid.index.shape())?;
    let mut weights = solver::reciprocal(c, &mut fft, &rho.as_slice()[..n])?;
    let volume = n as f64 * c.grid.index.dv();
    for (w, kernel) in weights.as_mut_slice().iter_mut().zip(c.kernel.as_slice()) {
        *w = w.conj() * (kernel / volume);
    }
    let (potential, moments) = solver::bare_fields(c)?;
    let fields = Fields::new(
        vec![potential.as_slice().to_vec()],
        vec![0.; n],
        weights.as_slice().to_vec(),
        moments.as_slice().to_vec(),
    );
    let extra = [onsite.clone()];
    let matrix = contract_matrices(&provider, &fields, Some(&extra), |_, k| {
        (densities[k].clone(), weighted[k].clone())
    })?;
    let atoms = provider.layout().atoms;
    let scale = cells as f64;
    let q = q.as_slice();
    let mut breakdown = Breakdown {
        overlap: matrix.overlap,
        kinetic: matrix.kinetic,
        local: matrix.local,
        short_range: matrix.short_range,
        projection_rows: matrix.rows,
        projection_columns: matrix.columns,
        hats: vec![[0.; 3]; atoms],
        ion: vec![[0.; 3]; atoms],
        dense: vec![[0.; 3]; atoms],
        constants: vec![[0.; 3]; atoms],
        core: vec![[0.; 3]; atoms],
        kinetic_density: vec![[0.; 3]; atoms],
        core_kinetic: vec![[0.; 3]; atoms],
        spin_orbit_rows: vec![[0.; 3]; atoms],
        spin_orbit_columns: vec![[0.; 3]; atoms],
        exchange_phase: vec![[0.; 3]; atoms],
        exchange_functions: vec![[0.; 3]; atoms],
        exchange_rows: vec![[0.; 3]; atoms],
        exchange_columns: vec![[0.; 3]; atoms],
    };
    for atom in 0..atoms {
        for axis in 0..3 {
            breakdown.ion[atom][axis] = provider.ion_overlap(&fields, atom, axis)?
                + scale * provider.ion_energy(atom, axis)?;
            let dense = provider.dense(atom, axis)?;
            breakdown.dense[atom][axis] = q
                .iter()
                .enumerate()
                .map(|(a, qa)| qa.conj() * dense.uion[a] + qa * dense.linear[a])
                .sum::<C64>()
                .re;
            breakdown.constants[atom][axis] = scale * provider.constant(atom, axis)?;
        }
    }
    Ok(breakdown)
}
