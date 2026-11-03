//! Oracles of the analytic PAW-LCAO gradient, built only on public accessors.
//!
//! * [`fixed_density_derivative`] is the retired finite-difference route kept as
//!   a test: the symmetric Richardson difference of the geometry-dependent
//!   energy expression at fixed orthonormal-frame densities (`at_geometry`, so
//!   a displaced Lowdin matrix carries the Pulay term).
//! * [`term_derivatives`] differentiates each energy term at fixed
//!   atomic-orbital density instead, from the matrices and fields of the
//!   displaced integrals, so that every term of the analytic gradient has its
//!   own independent central-difference reference.
#![allow(dead_code)]

pub mod synthetic;

use mandacaru_core::{
    math::{fft::Poisson, storage::Precision},
    models::{functional, xc},
    molecular::{MeanField, MolecularIntegrals},
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

pub type Matrix = DMatrix<C64>;

/// The geometry of `base` with one coordinate moved by `delta` Bohr.
pub fn moved(base: &MolecularIntegrals, atom: usize, axis: usize, delta: f64) -> Geometry {
    let mut positions = base.geometry.positions().to_vec();
    positions[atom][axis] += delta;
    Geometry::new(
        base.geometry.numbers().to_vec(),
        positions,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap()
}
/// Richardson difference (steps `h` and `h / 2`) of `value`, per Bohr.
pub fn richardson(h: f64, mut value: impl FnMut(f64) -> f64) -> (f64, f64) {
    let coarse = (value(h) - value(-h)) / (2. * h);
    let fine = (value(0.5 * h) - value(-0.5 * h)) / h;
    ((4. * fine - coarse) / 3., (fine - coarse).abs() / 3.)
}

fn electronic(rebuilt: &MolecularIntegrals, state: &MeanField, d: &[Matrix]) -> f64 {
    match state.functional.as_deref() {
        Some(name) => rebuilt.kohn_sham_energy(name, d).unwrap(),
        None => rebuilt.hartree_fock_energy(&state.method, d).unwrap(),
    }
}
/// The retired route: the derivative of the energy expression at fixed
/// orthonormal-frame densities, with the Coulomb pivots of the reference.
pub fn fixed_density_derivative(
    base: &MolecularIntegrals,
    state: &MeanField,
    atom: usize,
    axis: usize,
) -> (f64, f64) {
    richardson(1e-3, |delta| {
        let geometry = moved(base, atom, axis, delta);
        let rebuilt = if state.functional.is_some() {
            base.at_geometry(geometry).unwrap()
        } else {
            base.at_geometry_with_fixed_pivots(geometry).unwrap()
        };
        electronic(&rebuilt, state, &state.densities) + rebuilt.nuclear
    })
}

/// The atomic-orbital density of a converged state and its occupied orbitals.
pub struct Ao {
    /// The density of each spin channel in the atomic-orbital basis.
    pub density: Vec<DMatrix<f64>>,
    /// Occupation of a spin channel: 2 for a restricted state, else 1.
    pub weight: f64,
    /// `(spin channel, coefficients)` of the occupied orbitals.
    pub orbitals: Vec<(usize, DVector<f64>)>,
    /// The energy-weighted density `sum_o f eps a a^T`.
    pub w: DMatrix<f64>,
}
pub fn ao_state(base: &MolecularIntegrals, state: &MeanField) -> Ao {
    let m = base.m();
    let weight = if state.method == "rhf" { 2. } else { 1. };
    let mut density = Vec::new();
    let mut orbitals = Vec::new();
    for (spin, d) in state.densities.iter().enumerate() {
        let ortho = d.map(|v| v.re);
        density.push(&base.lowdin * &ortho * &base.lowdin);
        let count = if state.method == "rhf" {
            state.particles[0]
        } else {
            state.particles[spin]
        };
        let eigen = ortho.clone().symmetric_eigen();
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
        for &k in order.iter().take(count) {
            orbitals.push((spin, &base.lowdin * eigen.eigenvectors.column(k)));
        }
    }
    Ao {
        density,
        weight,
        orbitals,
        w: base.energy_weighted_density(state).unwrap(),
    }
}
fn rows(x: &MolecularIntegrals) -> Vec<f64> {
    let (m, n) = (x.m(), x.grid.index.points());
    let mut out = vec![0.; m * n];
    for a in 0..m {
        x.samples
            .widen_range(a * n, &mut out[a * n..(a + 1) * n])
            .unwrap();
    }
    out
}
fn trace(d: &DMatrix<f64>, a: &DMatrix<f64>) -> f64 {
    d.component_mul(&a.transpose()).sum()
}
fn matrix(buffer: &[f64], m: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(m, m, buffer)
}

/// Every geometry-dependent energy term at the integrals `x` for the fixed
/// atomic-orbital density `ao` (names as the analytic gradient's terms), and
/// their sum plus the model constant: the energy expression itself.
pub fn term_energies(
    ctx: &Context,
    x: &MolecularIntegrals,
    state: &MeanField,
    ao: &Ao,
) -> BTreeMap<&'static str, f64> {
    let (m, n) = (x.m(), x.grid.index.points());
    let dv = x.grid.index.dv();
    let corrections = x.corrections.as_ref().unwrap();
    let hf = state.functional.is_none();
    let f = ao.weight;
    let half_overlap = x.lowdin.clone().try_inverse().unwrap();
    let mut terms = BTreeMap::new();
    let kinetic = matrix(x.kinetic.as_slice(), m);
    let potential = matrix(x.potential.as_slice(), m);
    let overlap = matrix(x.overlap.as_slice(), m);
    let projected = &corrections.potential;
    let augmentation = &corrections.overlap;
    let sum = |g: &dyn Fn(&DMatrix<f64>) -> f64| ao.density.iter().map(g).sum::<f64>() * f;
    terms.insert("kinetic", sum(&|d| trace(d, &kinetic)));
    terms.insert("external", sum(&|d| trace(d, &(&potential - projected))));
    let mut one_center = sum(&|d| trace(d, projected));
    if let (false, Some(dft)) = (hf, corrections.dft.as_ref()) {
        let one = &half_overlap * &dft.one * &half_overlap;
        one_center += sum(&|d| trace(d, &one));
    }
    terms.insert("projector_potential", one_center);
    terms.insert("pulay_grid", -trace(&ao.w, &(&overlap - augmentation)));
    terms.insert("pulay_projector", -trace(&ao.w, augmentation));
    terms.insert("nuclear", x.nuclear);
    // Grid densities of the fixed atomic-orbital density on the displaced basis.
    let chi = rows(x);
    let density = |d: &DMatrix<f64>| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let v = DVector::from_fn(m, |a, _| chi[a * n + k]);
                (&v.transpose() * d * &v)[(0, 0)].max(0.)
            })
            .collect()
    };
    let spins: Vec<Vec<f64>> = (0..2)
        .map(|s| density(&ao.density[if ao.density.len() == 1 { 0 } else { s }]))
        .collect();
    let total: Vec<C64> = (0..n)
        .map(|k| C64::new(spins[0][k] + spins[1][k], 0.))
        .collect();
    let mut poisson = Poisson::isolated(ctx, &x.grid).unwrap();
    let mut potential_h = vec![C64::new(0., 0.); n];
    poisson.solve_into(&total, &mut potential_h).unwrap();
    terms.insert(
        "hartree",
        0.5 * total
            .iter()
            .zip(&potential_h)
            .map(|(r, v)| r.re * v.re)
            .sum::<f64>()
            * dv,
    );
    if let (Some(name), Some(dft)) = (state.functional.as_deref(), corrections.dft.as_ref()) {
        let mut fields = vec![0.; 2 * n];
        for s in 0..2 {
            for k in 0..n {
                fields[s * n + k] = spins[s][k] + 0.5 * dft.core.as_slice()[k];
            }
        }
        let definition = functional(name).unwrap();
        let relativistic =
            dft.relativistic_exchange && matches!(definition.semilocal, "lda" | "pbe");
        // A meta-GGA sees the orbitals' own kinetic energy density and density
        // gradient (product rule), the partial core's by the grid.
        let meta = if definition.requires_tau {
            let mut slopes = vec![0.; 3 * m * n];
            for a in 0..m {
                for axis in 0..3 {
                    xc::derivative(
                        &x.grid,
                        &chi[a * n..(a + 1) * n],
                        axis,
                        &mut slopes[(axis * m + a) * n..(axis * m + a + 1) * n],
                    )
                    .unwrap();
                }
            }
            let mut tau = vec![0.; 2 * n];
            let mut gradient = vec![0.; 6 * n];
            for sp in 0..2 {
                let d = &ao.density[if ao.density.len() == 1 { 0 } else { sp }];
                for k in 0..n {
                    let mut t = 0.;
                    let mut g = [0.; 3];
                    for a in 0..m {
                        for b in 0..m {
                            let dab = d[(a, b)];
                            for axis in 0..3 {
                                t += 0.5
                                    * dab
                                    * slopes[(axis * m + a) * n + k]
                                    * slopes[(axis * m + b) * n + k];
                                g[axis] +=
                                    2. * dab * chi[a * n + k] * slopes[(axis * m + b) * n + k];
                            }
                        }
                    }
                    tau[sp * n + k] = t.max(0.) + 0.5 * dft.core_tau.as_slice()[k];
                    for axis in 0..3 {
                        gradient[(sp * 3 + axis) * n + k] = g[axis];
                    }
                }
            }
            let mut slope = vec![0.; n];
            for axis in 0..3 {
                xc::derivative(&x.grid, dft.core.as_slice(), axis, &mut slope).unwrap();
                for sp in 0..2 {
                    for k in 0..n {
                        gradient[(sp * 3 + axis) * n + k] += 0.5 * slope[k];
                    }
                }
            }
            Some((tau, gradient))
        } else {
            None
        };
        let xc = xc::evaluate_fields_relativistic(
            ctx,
            &x.grid,
            &fields,
            meta.as_ref().map(|(t, _)| t.as_slice()),
            meta.as_ref().map(|(_, g)| g.as_slice()),
            name,
            relativistic,
        )
        .unwrap();
        terms.insert("xc", xc.energy);
    }
    // The pair factors of the compensation charges (and, for Hartree-Fock, the
    // one-center tensor) in the atomic-orbital basis: `f = S^{1/2} f~ S^{1/2}`.
    let pairs = if hf {
        &*corrections.pairs
    } else {
        &corrections.dft.as_ref().unwrap().pairs
    };
    let ao_factors: Vec<DMatrix<f64>> = pairs
        .factors
        .iter()
        .map(|f| &half_overlap * f * &half_overlap)
        .collect();
    let mut coulomb = 0.;
    let mut exchange = 0.;
    for (factor, &weight) in ao_factors.iter().zip(&pairs.weights) {
        let moment: f64 = ao.density.iter().map(|d| trace(d, factor)).sum::<f64>() * f;
        coulomb += 0.5 * weight * moment * moment;
        if hf {
            exchange -= 0.5
                * f
                * weight
                * ao.density
                    .iter()
                    .map(|d| trace(&(d * factor), &(d * factor)))
                    .sum::<f64>();
        }
    }
    terms.insert("pairs_coulomb", coulomb);
    // Exact exchange of the grid pair densities of the fixed orbitals under a
    // Poisson kernel: `sum_ij (ij|ij)` over the ordered pairs of one spin.
    let mut orbitals: Vec<(usize, Vec<f64>)> = Vec::new();
    for (spin, a) in &ao.orbitals {
        let mut psi = vec![0.; n];
        for b in 0..m {
            for k in 0..n {
                psi[k] += a[b] * chi[b * n + k];
            }
        }
        orbitals.push((*spin, psi));
    }
    let pair_sum = |solver: &mut Poisson| -> f64 {
        let mut source = vec![C64::new(0., 0.); n];
        let mut solved = vec![C64::new(0., 0.); n];
        let mut energy = 0.;
        for i in 0..orbitals.len() {
            for j in 0..=i {
                if orbitals[i].0 != orbitals[j].0 {
                    continue;
                }
                for k in 0..n {
                    source[k] = C64::new(orbitals[i].1[k] * orbitals[j].1[k], 0.);
                }
                solver.solve_into(&source, &mut solved).unwrap();
                let pair: f64 = source
                    .iter()
                    .zip(&solved)
                    .map(|(r, v)| r.re * v.re)
                    .sum::<f64>()
                    * dv;
                energy += if i == j { pair } else { 2. * pair };
            }
        }
        energy
    };
    if hf {
        terms.insert("pairs_exchange", exchange);
        terms.insert("exchange", -0.5 * f * pair_sum(&mut poisson));
    }
    // A screened hybrid: `-fraction` times the exchange form under `erfc(omega r)/r`
    // (grid pairs, compensation charges and the screened one-center tensor), the
    // frozen short-range semilocal exchange of the spheres and its polarization.
    if let Some((omega, fraction)) = state
        .functional
        .as_deref()
        .and_then(|name| functional(name).unwrap().screened_exchange)
    {
        let screened = x
            .paw_setup
            .as_ref()
            .unwrap()
            .screened_exchange(x, omega)
            .unwrap();
        let mut screened_pairs = 0.;
        for (factor, &weight) in screened.pairs.factors.iter().zip(&screened.pairs.weights) {
            let factor = &half_overlap * factor * &half_overlap;
            screened_pairs -= 0.5
                * f
                * weight
                * ao.density
                    .iter()
                    .map(|d| trace(&(d * &factor), &(d * &factor)))
                    .sum::<f64>();
        }
        terms.insert("hybrid_pairs", fraction * screened_pairs);
        let one_body = &half_overlap * &screened.one_body * &half_overlap;
        terms.insert("hybrid_one_body", -fraction * sum(&|d| trace(d, &one_body)));
        let mut solver = Poisson::screened(ctx, &x.grid, omega).unwrap();
        terms.insert("hybrid_grid", -0.5 * fraction * f * pair_sum(&mut solver));
        if ao.density.len() == 2 {
            let ortho: Vec<DMatrix<f64>> = ao
                .density
                .iter()
                .map(|d| &half_overlap * d * &half_overlap)
                .collect();
            let (energy, _) = screened
                .spin
                .polarization(ctx, &ortho[0], &ortho[1])
                .unwrap();
            terms.insert("hybrid_polarization", -fraction * energy);
        }
    }
    terms
}
/// The central difference of every term of [`term_energies`] along one
/// coordinate (Richardson, steps `h`, `h/2`), per Bohr, with the Richardson
/// estimate of its own step error.
pub fn term_derivatives(
    ctx: &Context,
    base: &MolecularIntegrals,
    state: &MeanField,
    ao: &Ao,
    atom: usize,
    axis: usize,
) -> BTreeMap<&'static str, (f64, f64)> {
    let h = 1e-3;
    let mut cache: Vec<(f64, BTreeMap<&'static str, f64>)> = Vec::new();
    for delta in [h, -h, 0.5 * h, -0.5 * h] {
        let x = base.at_geometry(moved(base, atom, axis, delta)).unwrap();
        cache.push((delta, term_energies(ctx, &x, state, ao)));
    }
    let pick = |delta: f64| &cache.iter().find(|(d, _)| *d == delta).unwrap().1;
    let mut out = BTreeMap::new();
    for key in pick(h).keys() {
        let coarse = (pick(h)[key] - pick(-h)[key]) / (2. * h);
        let fine = (pick(0.5 * h)[key] - pick(-0.5 * h)[key]) / h;
        out.insert(
            *key,
            ((4. * fine - coarse) / 3., (fine - coarse).abs() / 3.),
        );
    }
    out
}
/// The analytic gradient summed by term (`name` before the first `/`).
pub fn by_term(
    gradient: &mandacaru_core::molecular::forces::Gradient,
) -> BTreeMap<String, Vec<f64>> {
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (name, values) in &gradient.terms {
        let term = name.split('/').next().unwrap().to_string();
        let entry = out
            .entry(term)
            .or_insert_with(|| vec![0.; values.as_slice().len()]);
        for (e, v) in entry.iter_mut().zip(values.as_slice()) {
            *e += v;
        }
    }
    out
}
pub fn precision_of(x: &MolecularIntegrals) -> Precision {
    x.precision()
}

/// A term's gate is `GATE` times its measured floor.
pub const GATE: f64 = 100.;
/// The measured floor of one term's oracle (Ha/Bohr): the largest |analytic -
/// oracle| over the synthetic and dataset cases (`paw_forces_terms`,
/// `paw_forces_dataset`), rounded up. The oracle's own Richardson step error is
/// printed beside each comparison; for the real datasets it is 1e-8..1e-6, so
/// the floor is the agreement, not the step.
pub fn floor(term: &str) -> f64 {
    match term {
        "hybrid_grid" | "hybrid_one_body" | "hybrid_polarization" => 1e-9,
        "hybrid_pairs" => 5e-9,
        "kinetic" | "external" | "hartree" | "xc" => 1e-9,
        "exchange" | "pulay_grid" | "pulay_projector" => 1e-10,
        "nuclear" | "projector_potential" => 1e-10,
        "pairs_exchange" => 1e-9,
        "pairs_coulomb" => 5e-9,
        _ => panic!("no measured floor for {term}"),
    }
}
/// The smallest miss, in gates of `term`, of the wrong variants of its analytic
/// value on one coordinate: each part of the term removed, then sign-flipped.
/// A part under 100 gates cannot be told from the oracle's noise by removing it
/// and is not tested; the number tested and skipped is returned with the bite.
pub fn bite(
    parts: &[(&str, &[f64])],
    term: &str,
    index: usize,
    analytic: f64,
    reference: f64,
) -> (f64, usize, usize) {
    let gate = GATE * floor(term);
    let (mut bite, mut tested, mut skipped) = (f64::INFINITY, 0, 0);
    for (name, values) in parts
        .iter()
        .filter(|(n, _)| n.split('/').next() == Some(term))
    {
        let part = values[index];
        if part.abs() < GATE * gate {
            skipped += 1;
            let _ = name;
            continue;
        }
        tested += 1;
        for wrong in [analytic - part, analytic - 2. * part] {
            bite = bite.min((wrong - reference).abs() / gate);
        }
    }
    (bite, tested, skipped)
}
