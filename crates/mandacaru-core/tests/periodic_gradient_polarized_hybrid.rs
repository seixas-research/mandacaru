//! F1, HSE06: the force of a COLLINEAR SPIN-POLARIZED (`spins = 2`) hybrid crystal on a real
//! converged SCF, against the reconverged free-energy finite difference of the same discretized
//! energy (two step families of `freeze_periodic_gradients.py`, 14.4 item 1): the H2 triplet in a
//! box at Gamma, every coordinate of both atoms. (The LiH triplet cannot serve the hybrid: its
//! unrestricted hybrid SCF reaches a second, higher state at some geometries; see the freeze script.) Whole k mesh, no Python. Needs
//! `MANDACARU_PAW_PATH`.
//!
//! What spin adds to the exact exchange, and what is pinned here:
//! * the exchange acts WITHIN a spin: `K_s[D_s]`, no cross-spin term; an unrestricted run has no
//!   restricted factor 1/2 and 2 (`spin_scale = 1`, `mult = 1`);
//! * the spheres' spin-scaling remainder `P = (F[2 n_up] + F[2 n_dn]) / 2 - F[n]` is nonzero for a
//!   sphere with a magnetization (both H spheres here), so the one-center operator of each spin has a
//!   part that depends on the OTHER spin's projected density; its response `Z` to the projections
//!   must carry it. The test measures the remainder's own force contribution (the response of
//!   the model with the remainder minus the model without it, chained to the atoms) and requires
//!   it to be resolved by the gate, and the force without it to miss the oracle.
mod periodic_gradient_crystals_support;

use mandacaru_core::{
    periodic::{
        gradient::{contract, freeze, gradient, scf, Derivatives},
        hybrid_gradient::{explicit, projection_forces},
        screened_augmentation::{Augmentation, OneCenter, OneCenterTerms},
        screened_exchange::Screening,
    },
    quantum::C64,
    Context, Result,
};
use nalgebra::DMatrix;
use periodic_gradient_crystals_support::{gates::*, *};

/// The converged moment (Bohr magnetons) of the triplet.
const MOMENT: f64 = 2.0;
/// Bound on the SCF energy and the hybrid tie (Hartree).
const ENERGY_GATE: f64 = 1e-9;
/// The re-diagonalized bands against the SCF's (the exchange residual the SCF stops at).
const BAND_GATE: f64 = 1e-9;

type Matrix = DMatrix<C64>;

/// A one-center model without the spin-scaling remainder. The collinear model is a sum over spins
/// of single-spin terms (`E_K(Q_up) + E_K(Q_dn)`, linear `V` and constants included) plus the
/// remainder, and the remainder vanishes for equal spin densities. So the model evaluated at
/// `(Q_s, Q_s)` is twice the single-spin term of `Q_s`, and averaging the two evaluations
/// reproduces the model without the remainder: the energy `(E(Q_up, Q_up) + E(Q_dn, Q_dn)) / 2`
/// and, for spin `s`, the operator of `E(Q_s, Q_s)` for that spin.
struct WithoutRemainder<'a>(&'a dyn OneCenter);

impl OneCenter for WithoutRemainder<'_> {
    fn collinear(&self, ctx: &Context, projected: &[Matrix]) -> Result<OneCenterTerms> {
        if projected.len() != 2 {
            return self.0.collinear(ctx, projected);
        }
        let mut energy = 0.;
        let mut operators = Vec::new();
        for (s, q) in projected.iter().enumerate() {
            let same = self.0.collinear(ctx, &[q.clone(), q.clone()])?;
            energy += 0.5 * same.energy;
            operators.push(same.operators[s].clone());
        }
        Ok(OneCenterTerms { energy, operators })
    }

    fn spinor(&self, ctx: &Context, projected: &Matrix) -> Result<OneCenterTerms> {
        self.0.spinor(ctx, projected)
    }
}

/// One case: the SCF, the energy tie, the engine, the comparison with the frozen oracle.
fn check(case: &'static str) {
    let start = std::time::Instant::now();
    let data = Data::load(case);
    assert_eq!(data.crystal.spins, 2, "{case}: a spin-polarized snapshot");
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < ENERGY_GATE,
        "{case}: the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    assert!(
        (result.magnetization - MOMENT).abs() < 0.05,
        "{case}: moment {}",
        result.magnetization
    );
    let tie = scf::hybrid_state_energy(
        &data.crystal,
        &result.matrices,
        data.options.exxdiv_treatment,
    )
    .unwrap()
        - result.energy;
    eprintln!("{case}: hybrid_state_energy - SCF energy = {tie:e}");
    assert!(tie.abs() < 1e-12, "{case}: energy tie {tie:e}");

    let state = data.state(&result);
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let frozen = freeze(&provider, &state, None).unwrap();
    // The re-diagonalized hybrid bands are the SCF's, spin by spin, for every band with weight in
    // the density. The EMPTY bands of a fully polarized run (the spin-down channel of a triplet has
    // none occupied) are not compared: their eigenvalues are those of an operator acting on a
    // vanishing density (the spin-scaling remainder of an empty channel is a large positive shift,
    // 17 Ha in the SCF's own last iteration, 80 Ha from the output density) and weigh nothing in
    // `D` or `W`; they are reported.
    let nk = data.crystal.points.len();
    let mut band_gap = 0f64;
    for (index, eps) in result.eigenvalues.iter().enumerate() {
        let again = &frozen.state.bands[index / nk][index % nk].eigenvalues;
        eprintln!(
            "{case}: spin {} k {}: occupations {:?}, SCF eigenvalues {eps:?}, re-diagonalized {again:?}",
            index / nk,
            index % nk,
            result.occupations[index]
        );
        for ((a, b), f) in eps.iter().zip(again).zip(&result.occupations[index]) {
            if *f > 1e-10 {
                band_gap = band_gap.max((a - b).abs());
            }
        }
    }
    eprintln!("{case}: occupied re-diagonalized eigenvalues - SCF eigenvalues: {band_gap:e}");
    assert!(band_gap < BAND_GATE, "{case}: bands {band_gap:e}");

    let breakdown = contract(&provider, &frozen).unwrap();
    eprintln!(
        "{case}: moment {:.6}, SCF {:.1} s ({} iterations), engine {:.1} s",
        result.magnetization,
        start.elapsed().as_secs_f64(),
        result.history.len(),
        start.elapsed().as_secs_f64()
    );
    let other = second_family(case);
    let worst = compare(case, &data, Some(&other), &breakdown.forces());
    eprintln!("{case}: worst miss {worst:.3} gates");
    for (name, term) in [
        ("exchange_phase", &breakdown.exchange_phase),
        ("exchange_functions", &breakdown.exchange_functions),
        ("exchange_rows", &breakdown.exchange_rows),
        ("exchange_columns", &breakdown.exchange_columns),
    ] {
        eprintln!("{case}: {name} {term:?}");
    }
    // Mesh-free terms stay exactly translation invariant; the projection chains of the engine and
    // of the exchange move a row and a column together.
    for (name, term) in [
        ("dense", &breakdown.dense),
        ("constants", &breakdown.constants),
        ("short_range", &breakdown.short_range),
    ] {
        assert!(net(term) < 1e-12, "{case}: {name} sum rule {:e}", net(term));
    }
    let chain = |rows: &[[f64; 3]], columns: &[[f64; 3]]| -> Vec<[f64; 3]> {
        rows.iter()
            .zip(columns)
            .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
            .collect()
    };
    assert!(
        net(&chain(
            &breakdown.projection_rows,
            &breakdown.projection_columns
        )) < 1e-12,
        "{case}: projection chain"
    );
    assert!(
        net(&chain(
            &breakdown.exchange_rows,
            &breakdown.exchange_columns
        )) < 1e-12,
        "{case}: exchange projection chain"
    );
}

#[test]
fn h2_hse06_force_is_the_reconverged_finite_difference() {
    check("h2_hse06");
}

/// The spin-scaling remainder is active and resolved: the response of the model with the
/// remainder minus the model without it, chained to the atoms, is the force the remainder adds.
/// It is far above the gate on at least one coordinate, and the force WITHOUT it (the total
/// force minus that contribution) misses the oracle by more than 100 gates.
#[test]
fn the_spin_scaling_remainder_is_resolved_by_the_gate() {
    let case = "h2_hse06";
    let data = Data::load(case);
    let result = data.solve();
    let state = data.state(&result);
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let right = gradient(&provider, &state, None).unwrap();
    let other = second_family(case);
    let honest = bite_families(&data, Some(&other), &right.forces());
    assert!(honest < 1., "the right force misses by {honest} gates");

    let crystal = &data.crystal;
    let hybrid = crystal.hybrid.as_ref().expect("a hybrid crystal");
    let without = WithoutRemainder(hybrid.one_center());
    let plain = Augmentation::new(crystal, data.sites(), &data.shapes)
        .unwrap()
        .with_one_center(&without);
    let full = crystal.hybrid_augmentation().unwrap();
    let exxdiv = data.options.exxdiv_treatment;
    let screening = Screening::of(crystal).unwrap();
    let a = explicit(crystal, &result.matrices, screening, exxdiv, &full, None).unwrap();
    let b = explicit(crystal, &result.matrices, screening, exxdiv, &plain, None).unwrap();
    eprintln!(
        "one-center exchange energy: with the remainder {:.12e}, without {:.12e}",
        a.one_center_energy, b.one_center_energy
    );
    assert!(
        (a.one_center_energy - b.one_center_energy).abs() > 1e-6,
        "the remainder is not active on this crystal"
    );
    let nk = crystal.points.len();
    let response: Vec<Matrix> = a.sphere.iter().zip(&b.sphere).map(|(x, y)| x - y).collect();
    let gradients: Vec<_> = (0..nk)
        .map(|k| provider.projection_gradient(k).unwrap())
        .collect();
    let layout = data.layout();
    let chain = projection_forces(
        &response,
        &gradients,
        &layout.row_owner,
        &layout.projector_owner,
        layout.atoms,
    )
    .unwrap()
    .total();
    // The remainder's share of the gradient, and the force without it.
    let without_forces: Vec<[f64; 3]> = right
        .forces()
        .iter()
        .zip(&chain)
        .map(|(f, r)| [f[0] + r[0], f[1] + r[1], f[2] + r[2]])
        .collect();
    let resolved = in_gates(&data, Some(&other), &chain);
    let bite = bite_families(&data, Some(&other), &without_forces);
    eprintln!("remainder force {chain:?}: largest {resolved:.1} gates; force without it misses by {bite:.1} gates");
    assert!(resolved > 100., "the remainder is {resolved:.1} gates");
    assert!(
        bite > 100.,
        "the force without the remainder bites {bite:.1}"
    );
}
