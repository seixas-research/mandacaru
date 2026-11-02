//! The spin-orbit term between the spinor Bloch sums.
//!
//! The molecule's operator ([`crate::models::spin_orbit::pair`]) with the projections of the
//! Bloch sums on the home cell's spin-orbit projectors in place of the molecular ones, the same
//! unfolding as the scalar nonlocal term `C(k) D C(k)^dagger`:
//!
//! ```text
//! h^SO_{mu s, nu t}(k) = sum_{a,l} sum_{ij, m m'} C^SO_{mu,lmi}(k) D^{SO,a,l}_{ij}
//!                        <l m s| L.S |l m' t> C^SO_{nu,lm'j}(k)^*
//! ```
//!
//! The matrix is `(2M, 2M)` over `(spin, function)`, alpha block first; the crystal's projectors
//! carry complex harmonics.
use super::shells::Shells;
use super::spheres::{projections, Projector};
use crate::{
    models::spin_orbit::{pair, Channel, Projectors},
    product, Buffer, Context, Error, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::collections::BTreeMap;

/// The radial coupling `D^{SO}` of one `(atom, l)`: `radial x radial`, row-major.
#[derive(Clone, Debug)]
pub struct Coupling {
    pub radial: usize,
    pub values: Vec<f64>,
}

/// The channels of the term: for each `(atom, l)` with a coupling and `l > 0`, in order of first
/// appearance, the projector positions ordered `m = -l..l` outer and radial index inner. A
/// channel that lacks any `(m, index)` of its rectangle, or a duplicated label, is refused.
pub fn channels(
    ctx: &Context,
    projectors: &[Projector],
    couplings: &BTreeMap<(usize, usize), Coupling>,
) -> Result<Vec<Channel>> {
    let mut order: Vec<(usize, usize)> = Vec::new();
    let mut labels: BTreeMap<(usize, usize), BTreeMap<(i32, usize), usize>> = BTreeMap::new();
    for (position, p) in projectors.iter().enumerate() {
        let group = labels.entry((p.atom, p.l)).or_insert_with(|| {
            order.push((p.atom, p.l));
            BTreeMap::new()
        });
        if group.insert((p.m, p.index), position).is_some() {
            return Err(Error::Invalid(
                "duplicate spin-orbit projector label".into(),
            ));
        }
    }
    let mut out = Vec::new();
    for key in order {
        let l = key.1;
        let Some(coupling) = couplings.get(&key) else {
            continue;
        };
        if l == 0 {
            continue;
        }
        let positions = &labels[&key];
        let mut radial: Vec<usize> = positions.keys().map(|(_, i)| *i).collect();
        radial.sort_unstable();
        radial.dedup();
        let expected: Vec<(i32, usize)> = (-(l as i32)..=l as i32)
            .flat_map(|m| radial.iter().map(move |&i| (m, i)))
            .collect();
        if expected.len() != positions.len() || expected.iter().any(|e| !positions.contains_key(e))
        {
            return Err(Error::Invalid(
                "spin-orbit channel needs every angular and radial projector".into(),
            ));
        }
        if coupling.radial != radial.len()
            || coupling.values.len() != product(&[coupling.radial, coupling.radial])?
        {
            return Err(Error::Invalid(
                "spin-orbit block dimensions must match its radial projectors".into(),
            ));
        }
        let mut indices = ctx.allocate::<usize>(expected.len())?;
        for (to, e) in indices.as_mut_slice().iter_mut().zip(&expected) {
            *to = positions[e];
        }
        let mut matrix = ctx.allocate::<Complex64>(coupling.values.len())?;
        for (to, v) in matrix.as_mut_slice().iter_mut().zip(&coupling.values) {
            *to = Complex64::new(*v, 0.);
        }
        out.push(Channel {
            l,
            real: false,
            indices,
            radial: radial.len(),
            coupling: matrix,
        });
    }
    Ok(out)
}

/// `(k, 2M, 2M)` row-major: `h^{SO}(k)` at the Cartesian `kpoints`, Hermitian. Zero (and still
/// that shape) when `couplings` is empty, a crystal without the term.
pub fn matrices(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    projectors: &[Projector],
    couplings: &BTreeMap<(usize, usize), Coupling>,
) -> Result<Buffer<Complex64>> {
    let m = shells.functions();
    let mut out = ctx.allocate::<Complex64>(product(&[kpoints.len(), 2 * m, 2 * m])?)?;
    if couplings.is_empty() {
        return Ok(out);
    }
    let found = channels(ctx, projectors, couplings)?;
    let c = projections(ctx, shells, lattice, kpoints, projectors)?;
    let count = projectors.len();
    for k in 0..kpoints.len() {
        let block = &c.as_slice()[k * m * count..(k + 1) * m * count];
        let h = pair(
            ctx,
            Projectors {
                left: block,
                right: block,
                rows: m,
                columns: m,
                count,
            },
            &found,
            true,
        )?;
        out.as_mut_slice()[k * 4 * m * m..(k + 1) * 4 * m * m].copy_from_slice(h.as_slice());
    }
    Ok(out)
}
