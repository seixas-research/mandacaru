//! MG-F2b: the molecular PAW spin-orbit term (`models::spin_orbit` with real harmonics).
//!
//! The molecule's projectors carry the real harmonics of `math::angular::cartesian`; the L.S
//! operator of `models::spin_orbit::angular(l, real = true)` must be written in exactly that basis.
//! A wrong sign or `m` label would leave the L.S spectrum (and an isolated atom's splitting)
//! unchanged and still couple orbital and spin directions wrongly in a molecule, so the operator is
//! checked entry by entry against `L = -i r x grad` applied to those harmonics on an angular rule.
use mandacaru_core::{
    math::angular::{cartesian, cartesian_gradient},
    models::spin_orbit::angular,
    paw::pair_tables::angular_rule,
    Context,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

#[test]
fn the_real_harmonic_ls_operator_is_l_dot_s_of_the_molecular_harmonics() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let rule = angular_rule(24, 48).unwrap();
    for l in 1..=3usize {
        let width = 2 * l + 1;
        let ms: Vec<i32> = (-(l as i32)..=l as i32).collect();
        // Orthonormal on the sphere.
        let overlap = DMatrix::<f64>::from_fn(width, width, |a, b| {
            rule.iter().map(|(u, w)| w * cartesian(l, ms[a], *u) * cartesian(l, ms[b], *u)).sum()
        });
        assert!((overlap - DMatrix::identity(width, width)).norm() < 1e-11, "l = {l}: overlap");
        // (L_k)_ab = -i sum_w w S_a (u x grad S_b)_k.
        let generator = |k: usize| {
            DMatrix::<C64>::from_fn(width, width, |a, b| {
                let value: f64 = rule
                    .iter()
                    .map(|(u, w)| {
                        let g = cartesian_gradient(l, ms[b], *u);
                        let cross = [u[1] * g[2] - u[2] * g[1], u[2] * g[0] - u[0] * g[2], u[0] * g[1] - u[1] * g[0]];
                        w * cartesian(l, ms[a], *u) * cross[k]
                    })
                    .sum();
                C64::new(0., -value)
            })
        };
        let (lx, ly, lz) = (generator(0), generator(1), generator(2));
        let pauli = [
            [[C64::new(0., 0.), C64::new(1., 0.)], [C64::new(1., 0.), C64::new(0., 0.)]],
            [[C64::new(0., 0.), C64::new(0., -1.)], [C64::new(0., 1.), C64::new(0., 0.)]],
            [[C64::new(1., 0.), C64::new(0., 0.)], [C64::new(0., 0.), C64::new(-1., 0.)]],
        ];
        let size = 2 * width;
        let ls = DMatrix::<C64>::from_fn(size, size, |i, j| {
            let (s, a, t, b) = (i / width, i % width, j / width, j % width);
            0.5 * (lx[(a, b)] * pauli[0][s][t] + ly[(a, b)] * pauli[1][s][t] + lz[(a, b)] * pauli[2][s][t])
        });
        let packed = angular(&ctx, l, true).unwrap();
        let model = DMatrix::<C64>::from_row_slice(size, size, packed.as_slice());
        let gap = (&model - &ls).norm();
        eprintln!("l = {l}: |angular(real) - L.S of the molecular harmonics| = {gap:.2e}");
        assert!(gap < 1e-10, "l = {l}: {gap:e}");
    }
}

mod paw_molecular_support;

/// A gold atom with one s, p and d shell: on one atom h^SO cannot mix l, and with one radial
/// function per l its spectrum is exactly the j = l +- 1/2 pattern of L.S in each shell: s 0 (x2),
/// p a/2 (x4) and -a (x2), d b (x6) and -3b/2 (x4). This tests the harmonics, the channel layout,
/// `pair` and the orthonormal transform together, on the shipped `pbe-fr` dataset.
#[test]
fn a_gold_atom_has_the_j_split_spin_orbit_spectrum() {
    use mandacaru_core::{molecular::spinor_dft::spin_orbit_term, paw::molecular::Arrays};
    use paw_molecular_support::{analytic_shells, dataset, molecule_with};
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
        let file = root.join("pbe-fr").join("Au.parquet");
        let _check = dataset::read(&ctx, "pbe-fr", "Au");
        let species = vec![serde_json::json!({ "file": file.to_string_lossy() })];
        let integrals = molecule_with(
            &ctx,
            &[79],
            &[[0.07, -0.04, 0.05]],
            12.,
            0.5,
            (species, vec![0], Arrays::new()),
            "raw",
            &analytic_shells(&[0, 1, 2]),
            &[("spin_orbit", serde_json::json!(true)), ("allow_heavy_dft", serde_json::json!(true))],
        );
        let h = spin_orbit_term(&integrals).unwrap().expect("gold carries a spin-orbit term");
        let hermiticity = (&h - h.adjoint()).norm();
        assert!(hermiticity < 1e-12, "h^SO is not Hermitian: {hermiticity:e}");
        let mut eigen: Vec<f64> = nalgebra::SymmetricEigen::new(h.clone()).eigenvalues.iter().copied().collect();
        eigen.sort_by(f64::total_cmp);
        // Levels within 1e-3 (relative) are one j level: the cubic grid's overlap and Loewdin
        // transform make a d shell slightly anisotropic (a t2g/eg-like 1e-4 split of j = 5/2).
        let mut groups: Vec<(f64, usize)> = Vec::new();
        for v in eigen {
            match groups.last_mut() {
                Some((value, count)) if (v - *value).abs() < 1e-3 * value.abs().max(1e-6) => {
                    *value = (*value * *count as f64 + v) / (*count + 1) as f64;
                    *count += 1;
                }
                _ => groups.push((v, 1)),
            }
        }
        eprintln!("gold h^SO spectrum (value, multiplicity): {groups:?}");
        let data = dataset::read(&ctx, "pbe-fr", "Au");
        let coupled: Vec<usize> = data.spin_orbit.keys().map(|&l| l as usize).filter(|&l| (1..=2).contains(&l)).collect();
        eprintln!("gold: spin-orbit channels of the dataset within the basis (l): {coupled:?}");
        assert!(coupled.contains(&2), "the gold dataset couples its d shell");
        assert_eq!(groups.iter().map(|(_, c)| c).sum::<usize>(), 18);
        // Each coupled shell: j = l + 1/2 at c l / 2 (x 2l + 2) and j = l - 1/2 at -c (l + 1) / 2
        // (x 2l): ratio -(l + 1) / l. An uncoupled shell stays at zero.
        let mut zeros = 2; // the s shell
        for l in 1..=2usize {
            let (up, down) = (2 * l + 2, 2 * l);
            if !coupled.contains(&l) {
                zeros += up + down;
                continue;
            }
            let high = groups.iter().find(|(v, c)| *c == up && *v > 0.).map(|(v, _)| *v);
            let high = high.unwrap_or_else(|| panic!("l = {l}: no j = l + 1/2 level of multiplicity {up}"));
            let want = -(l as f64 + 1.) / l as f64 * high;
            let low = groups.iter().find(|(v, c)| *c == down && (v - want).abs() < 1e-3 * want.abs());
            assert!(low.is_some(), "l = {l}: no j = l - 1/2 level at {want:e} (x{down})");
            eprintln!("gold: l = {l} coupling {:+.6e} Ha, ratio {:+.6}", 2. * high / l as f64, low.unwrap().0 / high);
        }
        let zero_count: usize = groups.iter().filter(|(v, _)| v.abs() < 1e-10).map(|(_, c)| c).sum();
        assert_eq!(zero_count, zeros, "the uncoupled levels");
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Two exponents per l (s, p, d): 18 functions, enough spinors for gold's semicore valence.
fn double_shells() -> Vec<(usize, Vec<f64>, Vec<f64>)> {
    let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
    let mut out = Vec::new();
    for l in 0..=2usize {
        for exponent in [1.1, 0.45] {
            let values = radii
                .iter()
                .map(|&r| {
                    let cut = if r >= 7. { 0. } else { (1. - (r / 7.).powi(2)).powi(3) };
                    r.powi(l as i32) * (-exponent * r).exp() * cut
                })
                .collect();
            out.push((l, radii.clone(), values));
        }
    }
    out
}

/// A gold atom built for spinor Kohn-Sham, on the two-exponent basis.
fn gold(ctx: &Context) -> mandacaru_core::molecular::MolecularIntegrals {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::molecule_with;
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let file = root.join("pbe-fr").join("Au.parquet");
    molecule_with(
        ctx,
        &[79],
        &[[0.07, -0.04, 0.05]],
        12.,
        0.5,
        (vec![serde_json::json!({ "file": file.to_string_lossy() })], vec![0], Arrays::new()),
        "raw",
        &double_shells(),
        &[("spin_orbit", serde_json::json!(true)), ("allow_heavy_dft", serde_json::json!(true))],
    )
}

/// F2b: the spinor SCF of gold with its spin-orbit term. It converges; the term lowers the energy
/// (`Re tr(D h^SO) < 0`); the d levels are j-split (not the degenerate scalar d shell); the Fock
/// with the term is still the derivative of the energy; and RKS refuses the molecule by name.
#[test]
#[ignore = "slow: 21 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_atom_with_spin_orbit"]
fn a_gold_atom_with_spin_orbit_converges_split_and_variational() {
    use mandacaru_core::molecular::{spinor_dft::{spin_orbit_term, spinor_response}, ScfOptions};
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let integrals = gold(&ctx);
        let electrons = dataset::read(&ctx, "pbe-fr", "Au").valence_charge.round() as usize;
        let refusal = integrals.kohn_sham("pbe", [electrons.div_ceil(2), electrons / 2], true, &ScfOptions::default());
        assert!(refusal.as_ref().err().is_some_and(|e| e.to_string().contains("spinor Kohn-Sham")), "RKS/UKS must refuse");
        let options = ScfOptions { tolerance: 1e-10, max_iterations: 400, ..Default::default() };
        let state = integrals.spinor_kohn_sham("pbe", electrons, &options).unwrap();
        assert!(state.converged, "the spinor SCF of gold did not converge");
        let h = spin_orbit_term(&integrals).unwrap().unwrap();
        let d = &state.densities[0];
        let lowering = (d * &h).trace().re;
        eprintln!("gold PBE+SOC: {electrons} electrons, E = {:+.10e} Ha, Re tr(D h^SO) = {lowering:+.6e} Ha, {} iterations",
            state.electronic_energy, state.iterations.len());
        assert!(lowering < -1e-5, "the spin-orbit term does not lower the energy: {lowering:e}");
        let levels = &state.energies[0];
        eprintln!("gold levels (Ha): {:?}", levels.iter().map(|e| format!("{e:+.5}")).collect::<Vec<_>>());
        // The d shell's j = 5/2 and j = 3/2 levels: ten consecutive levels split in two groups.
        let mut best = (0usize, f64::INFINITY);
        for start in 0..=levels.len().saturating_sub(10) {
            let spread = levels[start + 9] - levels[start];
            if spread < best.1 {
                best = (start, spread);
            }
        }
        let shell: Vec<f64> = levels[best.0..best.0 + 10].to_vec();
        let gaps: Vec<f64> = shell.windows(2).map(|w| w[1] - w[0]).collect();
        let (split, at) = gaps.iter().copied().enumerate().map(|(i, g)| (g, i)).fold((0., 0), |b, x| if x.0 > b.0 { x } else { b });
        eprintln!("gold d levels split by {split:+.4e} Ha after level {}", at + 1);
        assert!(split > 1e-3, "the d shell is not j-split: {split:e}");
        assert!(at + 1 == 4 || at + 1 == 6, "the split is not 4/6 (j = 3/2, 5/2): {}", at + 1);
        // dE = Re tr(F dD) with the term, on a direction mixing spin and space.
        let m = integrals.m();
        let direction = {
            let raw = DMatrix::<C64>::from_fn(2 * m, 2 * m, |i, j| C64::new(((i * 7 + j * 3) % 11) as f64 / 11. - 0.5, ((i * 5 + j * 13) % 7) as f64 / 7. - 0.5));
            (&raw + raw.adjoint()) * C64::new(0.05, 0.)
        };
        let at_d = spinor_response(&integrals, d, "pbe", Some(&h)).unwrap();
        let energy = |e: f64| spinor_response(&integrals, &(d + &direction * C64::new(e, 0.)), "pbe", Some(&h)).unwrap().energy;
        let step = 1e-5;
        let fd = (energy(step) - energy(-step)) / (2. * step);
        let analytic = (&at_d.fock * &direction).trace().re;
        eprintln!("gold: dE/de {fd:+.10e}, Re tr(F dD) {analytic:+.10e}, diff {:.2e}", fd - analytic);
        assert!((fd - analytic).abs() < 1e-7 * (1. + analytic.abs()), "{fd} vs {analytic}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Au-H with one s/p/d shell on each atom, gold's spin-orbit term on (H carries none, but its basis
/// functions move through gold's spin-orbit sphere).
fn gold_hydride(ctx: &Context, positions: &[[f64; 3]; 2]) -> mandacaru_core::molecular::MolecularIntegrals {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::{analytic_shells, molecule_with};
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let file = |e: &str| root.join("pbe-fr").join(format!("{e}.parquet")).to_string_lossy().into_owned();
    molecule_with(
        ctx,
        &[79, 1],
        positions,
        14.,
        0.5,
        (vec![serde_json::json!({ "file": file("Au") }), serde_json::json!({ "file": file("H") })], vec![0, 1], Arrays::new()),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &[("spin_orbit", serde_json::json!(true)), ("allow_heavy_dft", serde_json::json!(true))],
    )
}

/// F2d increment 3a: the explicit spin-orbit gradient `Re tr(D dh^SO/dR)` at a fixed AO-frame
/// spinor density is the central difference of `Re tr(D h^SO(R))`, with the molecule (its basis
/// centers and projector spheres) rebuilt at each displaced geometry: both halves of the ownership
/// identity, the projectors' atom (Au) and the functions' atoms (Au and H); and a geometry continued
/// with `at_geometry` carries the fresh build's term.
#[test]
#[ignore = "slow: 16 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spin_orbit_gradient"]
fn the_spin_orbit_gradient_is_the_derivative_of_the_spin_orbit_energy() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    {
        let base_positions = [[0.07, -0.04, 0.05], [0.31, 0.22, 2.95]];
        let base = gold_hydride(&ctx, &base_positions);
        let setup = base.paw_setup.as_ref().unwrap();
        let ao = base.lowdin.nrows();
        let d = {
            let raw = DMatrix::<C64>::from_fn(2 * ao, 2 * ao, |i, j| {
                C64::new(((i * 7 + j * 3) % 11) as f64 / 11. - 0.5, ((i * 5 + j * 13) % 7) as f64 / 7. - 0.5)
            });
            (&raw + raw.adjoint()) * C64::new(0.5, 0.)
        };
        let analytic = setup.spin_orbit_gradient(&ctx, base.geometry.positions(), &d).unwrap();
        let energy = |atom: usize, axis: usize, step: f64| {
            let mut positions = base_positions;
            positions[atom][axis] += step;
            let moved = gold_hydride(&ctx, &positions);
            let h = moved.paw_setup.as_ref().unwrap().spin_orbit(&ctx, moved.geometry.positions()).unwrap().unwrap();
            (&d * h).trace().re
        };
        let step = 1e-3;
        let mut largest = 0f64;
        for atom in 0..2 {
            for axis in [0, 2] {
                let fd = (energy(atom, axis, step) - energy(atom, axis, -step)) / (2. * step);
                let a = analytic[atom][axis];
                largest = largest.max(a.abs());
                eprintln!("atom {atom} axis {axis}: analytic {a:+.10e}, fd {fd:+.10e}, diff {:.2e}", a - fd);
                assert!((a - fd).abs() < 1e-6 * (1. + a.abs()), "atom {atom} axis {axis}: {a} vs {fd}");
            }
        }
        assert!(largest > 1e-4, "the spin-orbit gradient is too small to be tested: {largest:e}");
        // A continued geometry carries the same term as a fresh build there: the shared setup
        // places the functions at the new positions (it once kept the first geometry's centers).
        let mut positions = base_positions;
        positions[1][2] += 0.05;
        let fresh = gold_hydride(&ctx, &positions);
        let geometry = mandacaru_core::Geometry::new(base.geometry.numbers().to_vec(), positions.to_vec(),
            *base.geometry.cell(), base.geometry.pbc(), mandacaru_core::LengthUnit::Bohr).unwrap();
        let continued = base.at_geometry(geometry).unwrap();
        let term = |m: &mandacaru_core::molecular::MolecularIntegrals| m.paw_setup.as_ref().unwrap().spin_orbit(&ctx, m.geometry.positions()).unwrap().unwrap();
        let gap = (&term(&continued) - &term(&fresh)).norm();
        assert!(gap < 1e-12, "a continued geometry's spin-orbit term differs from a fresh build's: {gap:e}");
        // Translation invariance: the atom sum of the explicit term vanishes.
        for axis in 0..3 {
            let sum: f64 = analytic.iter().map(|g| g[axis]).sum();
            assert!(sum.abs() < 1e-10 * (1. + largest), "axis {axis}: atom sum {sum:e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Au-Be on the two-exponent basis, spin-orbit on.
fn gold_beryllium(ctx: &Context, positions: &[[f64; 3]; 2]) -> mandacaru_core::molecular::MolecularIntegrals {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::molecule_with;
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let file = |e: &str| root.join("pbe-fr").join(format!("{e}.parquet")).to_string_lossy().into_owned();
    molecule_with(
        ctx,
        &[79, 4],
        positions,
        14.,
        0.5,
        (vec![serde_json::json!({ "file": file("Au") }), serde_json::json!({ "file": file("Be") })], vec![0, 1], Arrays::new()),
        "raw",
        &double_shells(),
        &[("spin_orbit", serde_json::json!(true)), ("allow_heavy_dft", serde_json::json!(true))],
    )
}

/// F2d, the deciding gate: the analytic spinor LDA force with spin-orbit coupling of an odd-electron
/// heavy dimer (Au-Be) against the Richardson difference of the converged state's energy at
/// displaced geometries (the fixed-density oracle, as `paw_forces_support::fixed_density_derivative`). Odd, so
/// the state carries a magnetization (`max_a>=1 |Re D_a|` is asserted); a dimer, so the force is
/// interatomic, not an egg-box ripple.
#[test]
#[ignore = "slow: 96 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_lda"]
fn a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("lda");
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with PBE.
#[test]
#[ignore = "slow: 440 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_pbe"]
fn a_gold_dimer_pbe_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("pbe");
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with PBE+rVV10.
#[test]
#[ignore = "slow: 871 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_rvv10"]
fn a_gold_dimer_rvv10_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("pbe+rvv10");
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with r2SCAN.
#[test]
#[ignore = "fails: the spinor r2SCAN SCF does not converge on Au-Be (residual floor ~1e-4 after 600 iterations per start, 2321 s, measured 2026-11-09; the local-frame meta-GGA is sharp near magnetization nodes and no SCF driver or frame softening converges it, see PAW_MIGRATION_LOG 2026-11-09 Q8 follow-up and Q9); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_r2scan"]
fn a_gold_dimer_r2scan_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("r2scan");
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with HSE06
/// (F2c): the screened exchange of the spinors, spin-flip pairs and the spheres' vector
/// polarization included.
#[test]
#[ignore = "slow: 1059 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_hse06"]
fn a_gold_dimer_hse06_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("hse06");
}

fn gold_dimer_force_gate(functional: &str) {
    use mandacaru_core::{models::noncollinear::component, molecular::ScfOptions};
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    {
        let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        let electrons = (dataset::read(&ctx, "pbe-fr", "Au").valence_charge + dataset::read(&ctx, "pbe-fr", "Be").valence_charge).round() as usize;
        assert!(electrons % 2 == 1, "Au-Be must have an odd electron count: {electrons}");
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 600, ..Default::default() };
        let base = gold_beryllium(&ctx, &positions);
        let state = base.spinor_kohn_sham(functional, electrons, &options).unwrap();
        assert!(state.converged, "the base spinor SCF did not converge");
        let m = base.m();
        let magnetization = (1..4).map(|a| component(&state.densities[0], m, a).map(|v| v.re).norm()).fold(0f64, f64::max);
        let gradient = base.spinor_gradient(&state).unwrap();
        let so = gradient.terms.iter().find(|(n, _)| *n == "spin_orbit/projections").map(|(_, b)| b.as_slice()[5].abs()).unwrap_or(0.);
        // The fixed-density difference: the base state's own energy at displaced geometries, the
        // full derivative of a stationary state. A reconverged difference is not the oracle for a
        // heavy open shell: displaced SCFs can land on another branch.
        let d = state.densities[0].clone();
        let energy = |dz: f64| {
            use mandacaru_core::molecular::spinor_dft::{spin_orbit_term, spinor_response};
            let mut moved = positions;
            moved[1][2] += dz;
            // Continue the base integrals: a fresh build is another discretization, whose
            // difference is not this derivative (a 0.17 Ha/Bohr "miss" measured 2026-11-09).
            let geometry = mandacaru_core::Geometry::new(base.geometry.numbers().to_vec(), moved.to_vec(),
                *base.geometry.cell(), base.geometry.pbc(), mandacaru_core::LengthUnit::Bohr).unwrap();
            let integrals = base.at_geometry(geometry).unwrap();
            let h = spin_orbit_term(&integrals).unwrap();
            spinor_response(&integrals, &d, functional, h.as_ref()).unwrap().energy + integrals.nuclear
        };
        let h = 1e-3;
        let coarse = (energy(h) - energy(-h)) / (2. * h);
        let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
        let oracle = (4. * fine - coarse) / 3.;
        let analytic = gradient.total.as_slice()[5]; // Be, z
        let to_ev = 27.211386245988 / 0.529177210903;
        eprintln!(
            "Au-Be SOC {functional} ({electrons} e): dE/dz analytic {analytic:+.10e}, FD {oracle:+.10e} (central {coarse:+.10e} / {fine:+.10e}), miss {:.2e} eV/A; max |Re D_a| {magnetization:.3e}; SO term {so:.3e}",
            (analytic - oracle).abs() * to_ev
        );
        assert!(magnetization > 1e-3, "the state carries no magnetization: {magnetization:e}");
        assert!(so > 1e-6, "the spin-orbit force term is not exercised: {so:e}");
        assert!((analytic - oracle).abs() * to_ev < 2e-5, "{analytic:e} vs {oracle:e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}


/// The spinor deriver on a 60-electron heavy core: the converged UKS LDA state of Au-Be
/// (`pbe-sr`, no spin-orbit term) embedded as `diag(D_up, D_dn)` is a stationary spinor state, and
/// its spinor gradient is the UKS gradient, which `paw_forces_dataset::multi_shell_core_terms`
/// anchors term by term. LDA: a GGA's local frame need not keep an embedded collinear state
/// stationary where the polarization changes sign.
#[test]
#[ignore = "slow: 10 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored an_embedded_collinear"]
fn an_embedded_collinear_heavy_state_has_the_unrestricted_gradient() {
    use mandacaru_core::{molecular::{spinor_dft::spinor_response, ScfOptions}, paw::molecular::Arrays};
    use paw_molecular_support::{dataset, molecule_with};
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    {
        let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap());
        let file = |e: &str| root.join("pbe-sr").join(format!("{e}.parquet")).to_string_lossy().into_owned();
        let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        let base = molecule_with(&ctx, &[79, 4], &positions, 14., 0.5,
            (vec![serde_json::json!({ "file": file("Au") }), serde_json::json!({ "file": file("Be") })], vec![0, 1], Arrays::new()),
            "raw", &double_shells(), &[("allow_heavy_dft", serde_json::json!(true))]);
        let electrons = (dataset::read(&ctx, "pbe-sr", "Au").valence_charge + dataset::read(&ctx, "pbe-sr", "Be").valence_charge).round() as usize;
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 600, ..Default::default() };
        let uks = base.kohn_sham("lda", [electrons.div_ceil(2), electrons / 2], true, &options).unwrap();
        assert!(uks.converged);
        let m = base.m();
        let mut d = DMatrix::<C64>::zeros(2 * m, 2 * m);
        d.view_mut((0, 0), (m, m)).copy_from(&uks.densities[0]);
        d.view_mut((m, m), (m, m)).copy_from(&uks.densities[1]);
        let embedded = spinor_response(&base, &d, "lda", None).unwrap();
        assert!((embedded.energy - uks.electronic_energy).abs() < 1e-10, "{} vs {}", embedded.energy, uks.electronic_energy);
        let spinor = base.spinor_gradient_of(&d, "lda", embedded.energy).unwrap();
        let collinear = base.gradient(&uks).unwrap();
        let gap = (0..6).map(|k| (spinor.total.as_slice()[k] - collinear.total.as_slice()[k]).abs()).fold(0f64, f64::max);
        let largest = collinear.total.as_slice().iter().fold(0f64, |a, v| a.max(v.abs()));
        eprintln!("Au-Be pbe-sr LDA ({electrons} e): spinor - UKS gradient {gap:.3e} (largest {largest:.3e})");
        assert!(largest > 1e-2 && gap < 1e-10, "spinor {:?} vs UKS {:?}", spinor.total.as_slice(), collinear.total.as_slice());
    }
    assert_eq!(ctx.memory_used(), 0);
}
