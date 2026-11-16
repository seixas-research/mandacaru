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
            &[("spin_orbit", serde_json::json!(true))],
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
        &[("spin_orbit", serde_json::json!(true))],
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
    gold_hydride_of(ctx, "pbe-fr", positions)
}

/// [`gold_hydride`] on the library set `set`.
fn gold_hydride_of(ctx: &Context, set: &str, positions: &[[f64; 3]; 2]) -> mandacaru_core::molecular::MolecularIntegrals {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::{analytic_shells, molecule_with};
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let file = |e: &str| root.join(set).join(format!("{e}.parquet")).to_string_lossy().into_owned();
    molecule_with(
        ctx,
        &[79, 1],
        positions,
        14.,
        0.5,
        (vec![serde_json::json!({ "file": file("Au") }), serde_json::json!({ "file": file("H") })], vec![0, 1], Arrays::new()),
        "raw",
        &analytic_shells(&[0, 1, 2]),
        &[("spin_orbit", serde_json::json!(true))],
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
        &[("spin_orbit", serde_json::json!(true))],
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
    gold_dimer_force_gate("lda", 5e-7);
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with PBE.
#[test]
#[ignore = "slow: 440 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_pbe"]
fn a_gold_dimer_pbe_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("pbe", 2e-5);
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with PBE+rVV10.
#[test]
#[ignore = "slow: 871 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_rvv10"]
fn a_gold_dimer_rvv10_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("pbe+rvv10", 2e-5);
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with r2SCAN.
#[test]
#[ignore = "slow: a spinor r2SCAN SCF of a heavy dimer and six fixed-density energies (MG-F8: the frame of zeta and eta converges it, the m_hat frame did not); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_r2scan"]
fn a_gold_dimer_r2scan_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("r2scan", 5e-7);
}

/// [`a_gold_dimer_lda_force_with_spin_orbit_is_the_fixed_density_finite_difference`] with HSE06
/// (F2c): the screened exchange of the spinors, spin-flip pairs and the spheres' vector
/// polarization included.
#[test]
#[ignore = "slow: 1059 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored a_gold_dimer_hse06"]
fn a_gold_dimer_hse06_force_with_spin_orbit_is_the_fixed_density_finite_difference() {
    gold_dimer_force_gate("hse06", 2e-5);
}

/// The step sweep of the fixed-density oracle of [`gold_dimer_force_gate`] (MG-F8 follow-up b):
/// the central difference of the base state's energy at continued geometries, at steps from
/// 4e-3 to 1.25e-4 Bohr, against the analytic force. A smooth energy gives a miss falling as
/// `h^2`; a kink of the discrete energy (a table or a cutoff the atoms move through) gives a miss
/// falling as `h` or scattering at the energy's own rounding. Printed, LDA and r2SCAN.
#[test]
#[ignore = "slow: one spinor SCF of a heavy dimer and twelve fixed-density energies per functional; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_fixed_density_oracle_of_the_gold_dimer --nocapture"]
fn the_fixed_density_oracle_of_the_gold_dimer_is_swept_in_its_step() {
    for functional in ["lda", "r2scan"] {
        gold_dimer_oracle_sweep(functional);
    }
}

fn gold_dimer_oracle_sweep(functional: &str) {
    use mandacaru_core::molecular::ScfOptions;
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
    let electrons = (dataset::read(&ctx, "pbe-fr", "Au").valence_charge
        + dataset::read(&ctx, "pbe-fr", "Be").valence_charge)
        .round() as usize;
    let options = ScfOptions { tolerance: 1e-11, max_iterations: 600, ..Default::default() };
    let base = gold_beryllium(&ctx, &positions);
    let state = base.spinor_kohn_sham(functional, electrons, &options).unwrap();
    assert!(state.converged);
    let analytic = base.spinor_gradient(&state).unwrap().total.as_slice()[5];
    let d = state.densities[0].clone();
    let energy = |dz: f64, fixed: bool| {
        use mandacaru_core::molecular::spinor_dft::{spin_orbit_term, spinor_response};
        let mut moved = positions;
        moved[1][2] += dz;
        let geometry = mandacaru_core::Geometry::new(base.geometry.numbers().to_vec(), moved.to_vec(),
            *base.geometry.cell(), base.geometry.pbc(), mandacaru_core::LengthUnit::Bohr).unwrap();
        let integrals = if fixed {
            base.at_geometry_with_fixed_pivots(geometry).unwrap()
        } else {
            base.at_geometry(geometry).unwrap()
        };
        let h = spin_orbit_term(&integrals).unwrap();
        spinor_response(&integrals, &d, functional, h.as_ref()).unwrap().energy + integrals.nuclear
    };
    let to_ev = 27.211386245988 / 0.529177210903;
    for fixed in [false, true] {
        for h in [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4] {
            let central = (energy(h, fixed) - energy(-h, fixed)) / (2. * h);
            eprintln!(
                "  {functional} {} pivots, step {h:.3e}: central {central:+.12e}, miss {:+.3e} eV/A",
                if fixed { "fixed" } else { "fresh" },
                (central - analytic) * to_ev
            );
        }
    }
    eprintln!("  {functional}: analytic {analytic:+.12e}");
}

/// `tolerance` (eV/A): the ordered 5e-7 for LDA and r2SCAN, met under the two-level oracle
/// (2.0e-7, 2.1e-7, measured 2026-11-15); 2e-5 for PBE, PBE+rVV10 and HSE06, whose gates have not
/// been rerun with this oracle.
fn gold_dimer_force_gate(functional: &str, tolerance: f64) {
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
        // The oracle (MG-F8 follow-up b, the step sweep of
        // `the_fixed_density_oracle_of_the_gold_dimer_is_swept_in_its_step`): the central
        // differences fall as h^2 from 4e-3 to 1e-3 Bohr and scatter below 5e-4 (the energy's own
        // rounding, about 1e-11 Ha), so the oracle takes the larger steps and removes h^2 and h^4
        // by two Richardson levels; its resolution is the spread of that limit over two step
        // ladders (8e-3 to 2e-3 and 4e-3 to 1e-3).
        let central: Vec<f64> = [8e-3, 4e-3, 2e-3, 1e-3]
            .iter()
            .map(|&h| (energy(h) - energy(-h)) / (2. * h))
            .collect();
        let first: Vec<f64> = central.windows(2).map(|w| (4. * w[1] - w[0]) / 3.).collect();
        let second: Vec<f64> = first.windows(2).map(|w| (16. * w[1] - w[0]) / 15.).collect();
        let (oracle, resolution) = (second[1], (second[1] - second[0]).abs());
        let analytic = gradient.total.as_slice()[5]; // Be, z
        let to_ev = 27.211386245988 / 0.529177210903;
        eprintln!(
            "Au-Be SOC {functional} ({electrons} e): dE/dz analytic {analytic:+.12e}, oracle {oracle:+.12e} \
             (central {central:.12?}; first level {first:.12?}; second level {second:.12?}), miss {:.2e} eV/A, \
             oracle resolution {:.2e} eV/A; max |Re D_a| {magnetization:.3e}; SO term {so:.3e}",
            (analytic - oracle).abs() * to_ev,
            resolution * to_ev
        );
        assert!(magnetization > 1e-3, "the state carries no magnetization: {magnetization:e}");
        assert!(so > 1e-6, "the spin-orbit force term is not exercised: {so:e}");
        // Measured 2026-11-15: the resolution is 1.6e-7 (LDA) and 1.5e-7 eV/A (r2SCAN), above the
        // 1e-7 sought: after the h^2 term the first-level remainders halve with the step (an h|h|
        // term of the energy at the base geometry), so the ladder does not converge as a
        // polynomial in h^2; printed, not asserted.
        assert!((analytic - oracle).abs() * to_ev < tolerance, "{analytic:e} vs {oracle:e}");
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
            "raw", &double_shells(), &[]);
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

/// `exp(t A)` of an anti-Hermitian `A`, through the eigenvectors of the Hermitian `-i A`.
fn unitary(a: &DMatrix<C64>, t: f64) -> DMatrix<C64> {
    let h = a * C64::new(0., -1.);
    let h = (&h + h.adjoint()) * C64::new(0.5, 0.);
    let (values, vectors) = mandacaru_core::math::linalg::eigen(h).unwrap();
    let phases = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        values.len(),
        values.iter().map(|v| C64::from_polar(1., t * v)),
    ));
    &vectors * phases * vectors.adjoint()
}

/// MG-F8, the collinear limit of the spinor meta-GGA frame: the UKS r2SCAN state of Au-Be
/// (`pbe-sr`, no spin-orbit term; its spin polarization changes sign near Be) embedded as
/// `diag(D_up, D_dn)` has the UKS energy to 1e-10 Ha, and its Fock blocks are the UKS
/// derivative: `dE/de` along a collinear and a spin-flip direction equals `Re tr(F dD)`.
#[test]
#[ignore = "slow: one UKS r2SCAN SCF of a heavy dimer; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_r2scan_frame"]
fn the_spinor_r2scan_frame_has_the_unrestricted_limit() {
    use mandacaru_core::{molecular::{spinor_dft::spinor_response, ScfOptions}, paw::molecular::Arrays};
    use paw_molecular_support::{dataset, molecule_with};
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    {
        let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap());
        let file = |e: &str| root.join("pbe-sr").join(format!("{e}.parquet")).to_string_lossy().into_owned();
        let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        let base = molecule_with(&ctx, &[79, 4], &positions, 14., 0.5,
            (vec![serde_json::json!({ "file": file("Au") }), serde_json::json!({ "file": file("Be") })], vec![0, 1], Arrays::new()),
            "raw", &double_shells(), &[]);
        let electrons = (dataset::read(&ctx, "pbe-sr", "Au").valence_charge + dataset::read(&ctx, "pbe-sr", "Be").valence_charge).round() as usize;
        let options = ScfOptions { tolerance: 1e-9, max_iterations: 400, ..Default::default() };
        let uks = base.kohn_sham("r2scan", [electrons.div_ceil(2), electrons / 2], true, &options).unwrap();
        let m = base.m();
        let mut d = DMatrix::<C64>::zeros(2 * m, 2 * m);
        d.view_mut((0, 0), (m, m)).copy_from(&uks.densities[0]);
        d.view_mut((m, m), (m, m)).copy_from(&uks.densities[1]);
        let embedded = spinor_response(&base, &d, "r2scan", None).unwrap();
        let collinear = base.kohn_sham_energy("r2scan", &uks.densities).unwrap();
        let gap = (embedded.energy - collinear).abs();
        // The polarization changes sign: the frame flips there.
        let spin = (&uks.densities[0] - &uks.densities[1]).map(|v| v.re);
        eprintln!("Au-Be pbe-sr r2SCAN ({electrons} e, UKS converged {}): embedded spinor - UKS energy {gap:.3e} Ha", uks.converged);
        assert!(gap < 1e-10, "{} vs {collinear}", embedded.energy);
        // Directions: a collinear one (the spin density) and a spin flip (the up-down block).
        let mut flip = DMatrix::<C64>::zeros(2 * m, 2 * m);
        let mut along = DMatrix::<C64>::zeros(2 * m, 2 * m);
        for i in 0..m {
            for j in 0..m {
                let x = C64::new(((i * 7 + j * 3) % 11) as f64 / 11. - 0.5, ((i + 2 * j) % 5) as f64 / 5. - 0.4);
                flip[(i, m + j)] = x;
                flip[(m + j, i)] = x.conj();
                along[(i, j)] = C64::new(spin[(i, j)], 0.);
                along[(m + i, m + j)] = C64::new(-spin[(i, j)], 0.);
            }
        }
        for (label, direction) in [("collinear", along), ("spin flip", flip)] {
            let energy = |e: f64| spinor_response(&base, &(&d + &direction * C64::new(e, 0.)), "r2scan", None).unwrap().energy;
            let analytic = (&embedded.fock * &direction).trace().re;
            let mut miss = f64::MAX;
            for h in [1e-4, 2e-5] {
                let slope = (8. * (energy(h) - energy(-h)) - (energy(2. * h) - energy(-2. * h))) / (12. * h);
                miss = (slope - analytic).abs();
                eprintln!("  {label}, step {h:.0e}: dE/de {slope:+.10e}, Re tr(F dD) {analytic:+.10e}, miss {miss:.2e}");
            }
            assert!(miss < 1e-7 * analytic.abs().max(1.), "{label}: miss {miss:e} of {analytic:e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// MG-F8, the potential of the spinor meta-GGA frame is the derivative of its energy on a
/// noncollinear state: Au-Be r2SCAN with spin-orbit coupling a few iterations from a canted start
/// (the magnetization turns in space), `dE/de` along random Hermitian directions equals
/// `Re tr(F dD)` to 1e-7 relative.
#[test]
#[ignore = "slow: a few spinor r2SCAN iterations of a heavy dimer; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_r2scan_potential"]
fn the_spinor_r2scan_potential_is_the_derivative_of_its_energy() {
    use mandacaru_core::molecular::{spinor_dft::{spin_orbit_term, spinor_response}, ScfOptions};
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    {
        let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        let base = gold_beryllium(&ctx, &positions);
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 4, ..Default::default() };
        let state = base
            .spinor_kohn_sham_noncollinear("r2scan", 0, &[[0., 1.2, 1.6], [0.8, 0., -0.6]], None, &options)
            .unwrap();
        let d = state.densities[0].clone();
        let m = base.m();
        let transverse = (1..3).map(|a| mandacaru_core::models::noncollinear::component(&d, m, a).map(|v| v.re).norm()).fold(0f64, f64::max);
        assert!(transverse > 1e-3, "the state is not noncollinear: {transverse:e}");
        let h_so = spin_orbit_term(&base).unwrap();
        let at = spinor_response(&base, &d, "r2scan", h_so.as_ref()).unwrap();
        for seed in 0..3usize {
            let mut direction = DMatrix::<C64>::zeros(2 * m, 2 * m);
            for i in 0..2 * m {
                for j in 0..=i {
                    let x = C64::new(((i * 13 + j * 7 + seed * 5) % 17) as f64 / 17. - 0.5, if i == j { 0. } else { ((i * 3 + j + seed) % 7) as f64 / 7. - 0.5 });
                    direction[(i, j)] = x;
                    direction[(j, i)] = x.conj();
                }
            }
            let h = 2e-5;
            let energy = |e: f64| spinor_response(&base, &(&d + &direction * C64::new(e, 0.)), "r2scan", h_so.as_ref()).unwrap().energy;
            let slope = (8. * (energy(h) - energy(-h)) - (energy(2. * h) - energy(-2. * h))) / (12. * h);
            let analytic = (&at.fock * &direction).trace().re;
            let miss = (slope - analytic).abs();
            eprintln!("direction {seed}: dE/de {slope:+.10e}, Re tr(F dD) {analytic:+.10e}, miss {miss:.2e} (transverse |D_a| {transverse:.2e})");
            assert!(miss < 1e-7 * analytic.abs().max(1.), "direction {seed}: {slope:e} vs {analytic:e}");
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// MG-F8, the deciding measurement (`PAW_MIGRATION_LOG.md` 2026-11-09 Q8 follow-up): the Au-Be
/// r2SCAN state with spin-orbit coupling after 100 SCF iterations, scanned along its
/// steepest-descent rotation `C exp(t K)`, `K_ai = -F_ai / |F_ov|`. The `m_hat` frame had a feature
/// about 1e-3 wide there (curvature 56 at h = 1e-4 against 4 at 1e-2); the frame of `zeta` and
/// `eta` must be quadratic over the decade, as PBE was: the curvature within 20 % from h = 1e-2
/// to 1e-4 (measured 3.5908 to 3.5909) and the slope of the smallest step the analytic one.
#[test]
#[ignore = "slow: 100 spinor r2SCAN iterations of a heavy dimer and a line scan, minutes; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_r2scan_energy_is_quadratic"]
fn the_spinor_r2scan_energy_is_quadratic_along_the_steepest_descent_rotation() {
    use mandacaru_core::molecular::{spinor_dft::{spin_orbit_term, spinor_response}, ScfOptions};
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(2), 1536 << 20).unwrap();
    {
        let positions = [[0.07, -0.04, 0.05], [0.12, 0.09, 4.35]];
        let electrons = (dataset::read(&ctx, "pbe-fr", "Au").valence_charge + dataset::read(&ctx, "pbe-fr", "Be").valence_charge).round() as usize;
        let base = gold_beryllium(&ctx, &positions);
        let options = ScfOptions { tolerance: 1e-11, max_iterations: 100, ..Default::default() };
        let state = base.spinor_kohn_sham("r2scan", electrons, &options).unwrap();
        let h_so = spin_orbit_term(&base).unwrap();
        let c = state.coefficients[0].clone();
        let f = state.occupations[0].clone();
        let density = |c: &DMatrix<C64>| {
            let occupied = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(f.len(), f.iter().map(|x| C64::new(*x, 0.))));
            c * occupied * c.adjoint()
        };
        let d0 = density(&c);
        let rebuilt = (&d0 - &state.densities[0]).norm();
        assert!(rebuilt < 1e-8 * d0.norm(), "the orbitals do not give the state's density: {rebuilt:e}");
        let at = spinor_response(&base, &d0, "r2scan", h_so.as_ref()).unwrap();
        let fmo = c.adjoint() * &at.fock * &c;
        let n = f.len();
        let mut k = DMatrix::<C64>::zeros(n, n);
        for a in 0..n {
            for i in 0..n {
                if f[i] > 0.5 && f[a] < 0.5 {
                    k[(a, i)] = -fmo[(a, i)];
                    k[(i, a)] = fmo[(a, i)].conj();
                }
            }
        }
        let norm = k.norm();
        k /= C64::new(norm, 0.);
        let energy = |t: f64| spinor_response(&base, &density(&(&c * unitary(&k, t))), "r2scan", h_so.as_ref()).unwrap().energy;
        let e0 = at.energy;
        let commutator = &k * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(n, f.iter().map(|x| C64::new(*x, 0.))));
        let analytic = (&fmo * (&commutator + commutator.adjoint())).trace().re;
        let mut curvatures = Vec::new();
        let mut last_slope = 0.;
        for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4] {
            let (plus, minus) = (energy(h), energy(-h));
            let slope = (plus - minus) / (2. * h);
            let curvature = (plus + minus - 2. * e0) / (h * h);
            eprintln!("h {h:.0e}: slope {slope:+.6e} (analytic {analytic:+.6e}), curvature {curvature:.4}");
            curvatures.push(curvature);
            last_slope = slope;
        }
        eprintln!("state after {} iterations, converged {}", state.iterations.len(), state.converged);
        let (low, high) = curvatures.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(high < 1.2 * low, "the curvature runs {low:.3} to {high:.3}: the frame is sharp");
        // The new frame converges this SCF (97 iterations measured 2026-11-15, where the m_hat
        // frame stalled at 1e-4 after 600), so the scan may start at the minimum: the slope is
        // then zero to the step's rounding (7e-10 at h = 1e-4).
        assert!((last_slope - analytic).abs() < 1e-4 * analytic.abs() + 1e-8, "{last_slope:e} vs {analytic:e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// Mercury (5s 5p 5d 6s valence, 20 electrons) of the library set `set` on the two-exponent
/// basis; a fully relativistic set carries its spin-orbit term.
fn mercury(ctx: &Context, set: &str) -> mandacaru_core::molecular::MolecularIntegrals {
    use mandacaru_core::paw::molecular::Arrays;
    use paw_molecular_support::molecule_with;
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let file = root.join(set).join("Hg.parquet");
    molecule_with(
        ctx,
        &[80],
        &[[0.07, -0.04, 0.05]],
        12.,
        0.5,
        (vec![serde_json::json!({ "file": file.to_string_lossy() })], vec![0], Arrays::new()),
        "raw",
        &double_shells(),
        &[("spin_orbit", serde_json::json!(set.ends_with("-fr")))],
    )
}

fn spinor_request(n: usize, approximation: mandacaru_core::molecular::excitations::Approximation) -> mandacaru_core::molecular::excitations::Request {
    let mut r = mandacaru_core::molecular::excitations::Request::new(n);
    r.approximation = approximation;
    r.tolerance = 1e-8;
    r.max_iterations = 400;
    r
}

fn tight() -> mandacaru_core::molecular::ScfOptions {
    mandacaru_core::molecular::ScfOptions { tolerance: 1e-12, max_iterations: 400, ..Default::default() }
}

/// T8's spin-orbit gate (`dev_plan/TDDFT_PLAN.md`, section 12): the spinor linear response of the
/// closed-shell mercury atom, through `excitations`, checked by identities of the engine's own
/// numbers.
/// - On the scalar-relativistic set (no spin-orbit term) the spinor roots are the collinear
///   singlets and triplets (x3).
/// - With the term (`lda-fr`, `pbe-fr`) the spinors come in Kramers pairs and the 6s -> 6p roots
///   are the 3P0, 3P1, 3P2 and 1P1 multiplets (1, 3, 5, 3), bright only for J = 1.
/// - The static polarizability `2 sum |mu_z|^2 / omega` of every Casida state is the finite-field
///   curvature `-d^2 E / dF^2` of the spinor SCF under `F z` in the one-body operator (a test-only
///   perturbation).
#[test]
#[ignore = "slow: 214 s, 226 MB (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_response_of_mercury --nocapture"]
fn the_spinor_response_of_mercury_has_its_multiplets_and_its_polarizability() {
    use mandacaru_core::molecular::{
        excitations::{excitations, Approximation, Request, Spin},
        multipoles::position_integrals,
    };
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        let options = tight();
        // The collinear limit on a real heavy dataset: no spin-orbit term.
        let plain = mercury(&ctx, "lda-sr");
        let restricted = plain.kohn_sham("lda", [10, 10], false, &options).unwrap();
        let spinor = plain.spinor_kohn_sham("lda", 20, &options).unwrap();
        assert!(restricted.converged && spinor.converged);
        for approximation in [Approximation::TammDancoff, Approximation::Casida] {
            let mut collinear = vec![];
            for (spin, copies) in [(Spin::Singlet, 1), (Spin::Triplet, 3)] {
                let mut r: Request = spinor_request(6, approximation);
                r.spin = Some(spin);
                for omega in excitations(&plain, &restricted, &r).unwrap().energies {
                    collinear.extend(std::iter::repeat_n(omega, copies));
                }
            }
            collinear.sort_by(f64::total_cmp);
            let found = excitations(&plain, &spinor, &spinor_request(12, approximation)).unwrap();
            let worst = (0..12).map(|j| (found.energies[j] - collinear[j]).abs()).fold(0f64, f64::max);
            eprintln!("Hg lda-sr through the spinor path, {}: spinor {:?}\n  collinear {:?}\n  max |d omega| = {worst:.3e} Ha",
                approximation.name(), &found.energies[..12], &collinear[..12]);
            if worst > 1e-8 {
                failures.push(format!("collinear limit {}: {worst:.3e}", approximation.name()));
            }
        }
        for (set, functional) in [("lda-fr", "lda"), ("pbe-fr", "pbe")] {
            let mut integrals = mercury(&ctx, set);
            let state = integrals.spinor_kohn_sham(functional, 20, &options).unwrap();
            assert!(state.converged);
            let m = integrals.m();
            let dimension = 2 * 20 * (2 * m - 20);
            let casida = excitations(&integrals, &state, &spinor_request(dimension / 2, Approximation::Casida)).unwrap();
            let spinors = casida.spinor.as_ref().expect("a spinor reference");
            let levels = &casida.channels[0].orbital_energies;
            let kramers = (0..levels.len() / 2).map(|k| (levels[2 * k + 1] - levels[2 * k]).abs()).fold(0f64, f64::max);
            let pairs: Vec<String> = (0..levels.len() / 2).map(|k| format!("{:+.6}", levels[2 * k])).collect();
            eprintln!("Hg {set}: spinor levels (Kramers pairs, Ha) {pairs:?}\n  largest pair splitting {kramers:.2e} Ha");
            if kramers > 1e-8 {
                failures.push(format!("{set}: Kramers pairs split by {kramers:.2e}"));
            }
            eprintln!("Hg {set} Casida: {} states of {dimension} real roots, largest residual {:.2e}",
                casida.energies.len(), casida.residuals.iter().copied().fold(0f64, f64::max));
            if casida.energies.len() != dimension / 2 {
                failures.push(format!("{set}: {} states, not {}", casida.energies.len(), dimension / 2));
            }
            // Multiplets of the lowest 40 states, grouped within 1e-4 Ha.
            let mut groups: Vec<(f64, f64, usize, f64)> = vec![]; // (first, last, count, sum f)
            for (j, &omega) in casida.energies.iter().take(40).enumerate() {
                let f = casida.oscillator_strengths[j];
                match groups.last_mut() {
                    Some(g) if omega - g.1 < 1e-4 => {
                        g.1 = omega;
                        g.2 += 1;
                        g.3 += f;
                    }
                    _ => groups.push((omega, omega, 1, f)),
                }
            }
            for (k, (first, last, count, f)) in groups.iter().enumerate() {
                eprintln!("  group {k}: omega {first:.8} Ha (x{count}, spread {:.2e}), sum f {f:.6e}", last - first);
            }
            // The lowest manifold, 6s -> 6p: 3P0, 3P1, 3P2 and 1P1, only the J = 1 levels bright
            // (3P1 by spin-orbit mixing). Above it the cubic grid splits J >= 2 into its cubic
            // irreducible parts (2 + 3, 1 + 3 + 3), so only the 6s -> 6p levels are asserted.
            let counts: Vec<usize> = groups.iter().take(4).map(|g| g.2).collect();
            let bright = groups[3].3;
            let ratio = (groups[2].0 - groups[1].0) / (groups[1].0 - groups[0].0);
            eprintln!("  6s -> 6p: multiplicities {counts:?}, f(3P1) / f(1P1) = {:.3e}, interval ratio (3P2 - 3P1) / (3P1 - 3P0) = {ratio:.3} (2 in the LS limit)", groups[1].3 / bright);
            if counts != [1, 3, 5, 3] {
                failures.push(format!("{set}: the 6s -> 6p multiplicities are {counts:?}"));
            }
            if groups[0].3.max(groups[2].3) > 1e-12 * bright || !(groups[1].3 > 1e-3 * bright) {
                failures.push(format!("{set}: 6s -> 6p strengths {:?}", groups.iter().take(4).map(|g| g.3).collect::<Vec<_>>()));
            }
            // The same manifold at the default tolerance (1e-5), as a user asks for it: every
            // root's partner is found and the states are those of the full solve.
            let tda = excitations(&integrals, &state, &Request::new(12)).unwrap();
            eprintln!("  default tolerance, TDA: {} states for 12 asked", tda.energies.len());
            if tda.energies.len() != 12 {
                failures.push(format!("{set}: default tolerance TDA gave {} states", tda.energies.len()));
            }
            let default = excitations(&integrals, &state, &{ let mut r = Request::new(12); r.approximation = Approximation::Casida; r }).unwrap();
            let shift = (0..12).map(|j| (default.energies[j] - casida.energies[j]).abs()).fold(0f64, f64::max);
            let strengths = (0..12).map(|j| (default.oscillator_strengths[j] - casida.oscillator_strengths[j]).abs()).fold(0f64, f64::max);
            eprintln!("  default tolerance, Casida: {} states, largest |d omega| {shift:.2e} Ha and |d f| {strengths:.2e} against the full solve", default.energies.len());
            if default.energies.len() != 12 || shift > 1e-6 || strengths > 1e-4 * bright {
                failures.push(format!("{set}: default tolerance: {} states, |d omega| {shift:.2e}, |d f| {strengths:.2e}", default.energies.len()));
            }
            let axis = 2;
            let alpha: f64 = casida.energies.iter().zip(&spinors.transition_dipoles)
                .map(|(w, mu)| 2. * mu[axis].norm_sqr() / w).sum();
            // The finite-field curvature along z of the spinor SCF.
            let z = position_integrals(&integrals, [0.; 3]).unwrap()[axis].clone();
            let base: Vec<f64> = integrals.one.as_slice().to_vec();
            let mut energy = |field: f64| {
                for a in 0..m {
                    for b in 0..m {
                        integrals.one.as_mut_slice()[a * m + b] = base[a * m + b] + field * z[(a, b)];
                    }
                }
                let s = integrals.spinor_kohn_sham(functional, 20, &options).unwrap();
                assert!(s.converged, "the field {field} SCF did not converge");
                s.electronic_energy
            };
            let e0 = energy(0.);
            let mut curvature = |f: f64| -(energy(f) + energy(-f) - 2. * e0) / (f * f);
            let (coarse, fine) = (curvature(4e-3), curvature(2e-3));
            let finite_field = (4. * fine - coarse) / 3.;
            let miss = (finite_field - alpha).abs() / alpha.abs();
            eprintln!("Hg {set} alpha_zz: sum over states {alpha:.10} Bohr^3, finite field {finite_field:.10} (coarse {coarse:.10}, fine {fine:.10}), relative miss {miss:.2e}");
            if miss > 1e-5 {
                failures.push(format!("{set}: polarizability {alpha} against {finite_field}: {miss:.2e}"));
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// T8: the first-order Fock `G(k)` of the spinor response is the Richardson difference of the
/// spinor Fock builder along `dD(k)`, for complex `k` (whose real and imaginary parts carry both
/// `A + B` and `A - B`), on a Kramers-unpolarized reference with spin-orbit (Hg, `lda-fr`: the
/// isotropic kernel) and on magnetized ones (the gold atom's odd electron, `pbe-fr` LDA and PBE:
/// the transverse kernel).
#[test]
#[ignore = "slow: 58 s (measured 2026-11-15); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_first_order_fock --nocapture"]
fn the_spinor_first_order_fock_is_the_derivative_of_the_spinor_fock() {
    use mandacaru_core::molecular::{
        excitations::spinor::Spinor,
        spinor_dft::{spin_orbit_term, spinor_response},
    };
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        let gold_electrons = paw_molecular_support::dataset::read(&ctx, "pbe-fr", "Au").valence_charge.round() as usize;
        let cases = [("Hg lda-fr", "lda"), ("Au pbe-fr", "lda"), ("Au pbe-fr", "pbe")];
        for (label, functional) in cases {
            let (integrals, electrons) = if label.starts_with("Hg") { (mercury(&ctx, "lda-fr"), 20) } else { (gold(&ctx), gold_electrons) };
            let state = integrals.spinor_kohn_sham(functional, electrons, &tight()).unwrap();
            assert!(state.converged);
            let d = &state.densities[0];
            let m = integrals.m();
            let moment = (0..m).map(|p| (1..4).map(|a| mandacaru_core::models::noncollinear::component(d, m, a)[(p, p)].re.abs()).fold(0f64, f64::max)).fold(0f64, f64::max);
            let operator = Spinor::new(&integrals, &state).unwrap();
            if label.starts_with("Au") {
                // A magnetized reference with spin-orbit coupling has its excitation energies but
                // no gradient (a Kramers doublet's moment direction is nearly free: the Au-H
                // cation's references are saddles of it). The seam skips the multiplet check,
                // which an atom's roots would trip first.
                use mandacaru_core::molecular::excitations::{excitations, gradient::seams, Request};
                let found = excitations(&integrals, &state, &Request::new(2)).unwrap();
                let refusal = seams::spinor_state(&integrals, &state, &found, 0);
                assert!(refusal.as_ref().err().is_some_and(|e| e.to_string().contains("magnetized")), "{label}: a magnetized spinor reference with spin-orbit coupling must refuse its gradient by name");
            }
            let o = operator.occupied();
            let v = 2 * m - o;
            let h = spin_orbit_term(&integrals).unwrap();
            let fock = |e: f64, change: &DMatrix<C64>| spinor_response(&integrals, &(d + change * C64::new(e, 0.)), functional, h.as_ref()).unwrap().fock;
            for seed in [1usize, 2] {
                let k = DMatrix::<C64>::from_fn(o, v, |i, a| {
                    C64::new((((i * 7 + a * 3 + seed) % 11) as f64 / 11. - 0.5) * 0.1, (((i * 5 + a * 13 + 2 * seed) % 7) as f64 / 7. - 0.5) * 0.1)
                });
                let change = operator.change(&k);
                let analytic = operator.first_order(&k).unwrap();
                let derivative = |step: f64| (fock(step, &change) - fock(-step, &change)) / C64::new(2. * step, 0.);
                let (coarse, fine) = (derivative(2e-4), derivative(1e-4));
                let richardson = (fine * C64::new(4., 0.) - coarse) / C64::new(3., 0.);
                let (co, cv) = (operator.orbitals().columns(0, o), operator.orbitals().columns(o, v));
                let projected = (cv.adjoint() * richardson * co).transpose();
                let miss = (&projected - &analytic).norm() / analytic.norm();
                eprintln!("{label} {functional} seed {seed}: |G| = {:.4e}, relative miss {miss:.2e}, largest |m_pp| {moment:.2e}", analytic.norm());
                if miss > 1e-6 {
                    failures.push(format!("{label} {functional} seed {seed}: {miss:.2e}"));
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// One gold hydride gate (`gold_hydride_of`, gold's spin-orbit term on): for each
/// `(set, functional, approximation, components)` the gradient of the lowest state whose nearest
/// root is at least 1e-3 Ha away against Richardson differences of reconverged `E_0 + omega` on
/// the base grid, with the Lagrangian's identities at the base geometry: the closed shell is
/// Kramers-unpolarized, so the time-odd share of `P` must vanish and every displaced reference
/// stays unpolarized. A component marked `nested` takes the kink-aware protocol: nested central differences at 2e-5,
/// 1e-5 and 5e-6 Bohr, and the mean of the first pair that agrees to 5e-7.
fn gold_hydride_gate(ctx: &Context, cases: &[(&str, &str, mandacaru_core::molecular::excitations::Approximation, &[(usize, usize, bool)])], failures: &mut Vec<String>) {
    use mandacaru_core::molecular::excitations::{excitations, excited_state, gradient::seams};
    use paw_molecular_support::dataset;
    {
        let base_positions = [[0.07, -0.04, 0.05], [0.31, 0.22, 2.95]];
        for &(set, functional, approximation, components) in cases {
            let electrons = (dataset::read(ctx, set, "Au").valence_charge + dataset::read(ctx, set, "H").valence_charge).round() as usize;
            let options = mandacaru_core::molecular::ScfOptions { tolerance: 1e-12, max_iterations: 300, ..Default::default() };
            let request = spinor_request(10, approximation);
            let base = gold_hydride_of(ctx, set, &base_positions);
            // A displaced geometry keeps the base's grid (`at_geometry`): a fresh build would
            // place a new grid around the moved atoms and take the grid's own force out of
            // the difference.
            let solve = |positions: &[[f64; 3]; 2]| {
                let geometry = mandacaru_core::Geometry::new(base.geometry.numbers().to_vec(), positions.to_vec(),
                    *base.geometry.cell(), base.geometry.pbc(), mandacaru_core::LengthUnit::Bohr).unwrap();
                let integrals = base.at_geometry(geometry).unwrap();
                let state = integrals.spinor_kohn_sham(functional, electrons, &options).unwrap();
                assert!(state.converged, "{set}: the spinor SCF did not converge at {positions:?}");
                (integrals, state)
            };
            let state = base.spinor_kohn_sham(functional, electrons, &options).unwrap();
            assert!(state.converged, "{set}: the spinor SCF did not converge");
            let m = base.m();
            let d = &state.densities[0];
            let moment = (0..m).map(|p| (1..4).map(|a| mandacaru_core::models::noncollinear::component(d, m, a)[(p, p)].re.abs()).fold(0f64, f64::max)).fold(0f64, f64::max);
            let found = excitations(&base, &state, &request).unwrap();
            let separation = |energies: &[f64], j: usize| (0..energies.len()).filter(|&k| k != j).map(|k| (energies[k] - energies[j]).abs()).fold(f64::INFINITY, f64::min);
            println!("Au-H {set} {functional} {}: {electrons} electrons, largest |m_pp| {moment:.1e}, states {:?}", approximation.name(),
                found.energies.iter().map(|w| format!("{w:.6}")).collect::<Vec<_>>());
            let root = (0..found.energies.len() - 1).find(|&j| separation(&found.energies, j) > 1e-3).expect("a nondegenerate state");
            let excited = excited_state(&base, &state, &request, root).unwrap();
            let (_, _, diagnostics) = seams::spinor_state(&base, &state, &found, root).unwrap();
            println!("  root {root}: omega {:.10} rebuilt {:.10}, separation {:.2e}, asymmetry {:.1e}, odd {:.1e}, ground seam {:.1e}, Z residual {:.1e} in {}",
                excited.excitation, diagnostics.omega, excited.separation, diagnostics.asymmetry, diagnostics.odd, diagnostics.ground, excited.z_residual, excited.z_iterations);
            if (diagnostics.omega - excited.excitation).abs() > 1e-9 || diagnostics.asymmetry > 1e-7 || diagnostics.ground > 1e-9 || diagnostics.odd > 1e-8 {
                failures.push(format!("{set}: identities {diagnostics:?}"));
            }
            let energy = |atom: usize, axis: usize, step: f64| {
                let mut positions = base_positions;
                positions[atom][axis] += step;
                let (moved, state) = solve(&positions);
                let found = excitations(&moved, &state, &request).unwrap();
                let d = &state.densities[0];
                let moment = (0..m).map(|p| (1..4).map(|a| mandacaru_core::models::noncollinear::component(d, m, a)[(p, p)].re.abs()).fold(0f64, f64::max)).fold(0f64, f64::max);
                println!("    ({atom}, {axis}) {step:+.0e}: omega {:.10}, separation {:.2e}, largest |m_pp| {moment:.1e}", found.energies[root], separation(&found.energies, root));
                assert!(moment < 1e-9, "a displaced reference is magnetized: {moment:e}");
                state.free_energy() + moved.nuclear + found.energies[root]
            };
            let mut worst = 0f64;
            for &(atom, axis, nested) in components {
                let central = |h: f64| (energy(atom, axis, h) - energy(atom, axis, -h)) / (2. * h);
                let analytic = excited.gradient.total.as_slice()[3 * atom + axis];
                let reference = if nested {
                    let steps: Vec<f64> = [2e-5, 1e-5, 5e-6].iter().map(|&h| central(h)).collect();
                    println!("  ({atom}, {axis}): central differences at 2e-5, 1e-5, 5e-6 {steps:?}");
                    match steps.windows(2).find(|w| (w[0] - w[1]).abs() < 5e-7) {
                        Some(w) => 0.5 * (w[0] + w[1]),
                        None => {
                            failures.push(format!("{set} ({atom}, {axis}): no pair of nested differences agrees to 5e-7"));
                            steps[2]
                        }
                    }
                } else {
                    let (coarse, fine) = (central(1e-3), central(5e-4));
                    (4. * fine - coarse) / 3.
                };
                println!("  ({atom}, {axis}): analytic {analytic:+.10} FD {reference:+.10} diff {:.2e}", analytic - reference);
                worst = worst.max((analytic - reference).abs());
            }
            for (name, values) in &excited.parts {
                println!("  {name}: {:?}", values.iter().map(|v| format!("{v:+.4e}")).collect::<Vec<_>>());
            }
            println!("Au-H {set} {functional} {}: max |dE| {worst:.3e} Ha/Bohr", approximation.name());
            if worst > 2e-7 {
                failures.push(format!("{set} {functional}: {worst:.3e} Ha/Bohr"));
            }
        }
    }
}

/// T8: the analytic gradient of an excited state of a spinor reference with spin-orbit coupling
/// (`excited_state`, the Kramers-unpolarized closed shell Au-H, 20 valence electrons, gold's term
/// on) against Richardson central differences of independently converged `E_0 + omega` on the
/// same fixed grid: LDA (`lda-fr`) Tamm-Dancoff and PBE (`pbe-fr`) Casida, on the lowest state
/// whose nearest root is at least 1e-3 Ha away (an Omega = 0 state: the |Omega| >= 1 pairs are
/// degenerate up to the grid's splitting), with the Lagrangian's identities at the base geometry.
/// LDA's omega along gold's z is not monotone in the step between 2e-3 and 2.5e-4 Bohr (the same
/// to 1e-10 at SCF 1e-13 and solver 1e-11), so that component takes the kink-aware protocol of
/// the HSE06 gate: nested central differences at 2e-5, 1e-5 and 5e-6 Bohr, and the mean of the
/// first pair that agrees to 5e-7, the floor that energies converged to 1e-12 leave at these
/// steps (measured 2026-11-16: the three differ by 2.6e-7 and 1.9e-7; `dev_plan/DECISIONS.md`,
/// 2026-11-16, T8 spinor forces). The time-odd
/// share of the relaxed density must vanish, and every displaced reference stays unpolarized.
#[test]
#[ignore = "slow: 1562 s, 212 MB (measured 2026-11-16); run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_excited_state_gradient_of_gold_hydride --nocapture"]
fn the_spinor_excited_state_gradient_of_gold_hydride_follows_its_energy() {
    use mandacaru_core::molecular::excitations::Approximation;
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        // (atom, axis, nested): `nested` takes the kink-aware protocol.
        let lda: &[(usize, usize, bool)] = &[(0, 2, true), (1, 2, false), (1, 0, false)];
        let pbe: &[(usize, usize, bool)] = &[(0, 2, false), (1, 2, false), (1, 0, false)];
        gold_hydride_gate(&ctx, &[("lda-fr", "lda", Approximation::TammDancoff, lda), ("pbe-fr", "pbe", Approximation::Casida, pbe)], &mut failures);
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// T8: the two approximations the closed-shell Au-H gate does not take, LDA (`lda-fr`) with
/// Casida and PBE (`pbe-fr`) with Tamm-Dancoff, against reconverged energies
/// ([`gold_hydride_gate`]); before, they rested on the collinear limit alone.
#[test]
#[ignore = "slow; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_spinor_excited_state_gradients_of_gold_hydride_in_the_other_approximations --nocapture"]
fn the_spinor_excited_state_gradients_of_gold_hydride_in_the_other_approximations_follow_their_energies() {
    use mandacaru_core::molecular::excitations::Approximation;
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        let components: &[(usize, usize, bool)] = &[(0, 2, false), (1, 2, false), (1, 0, false)];
        gold_hydride_gate(&ctx, &[("lda-fr", "lda", Approximation::Casida, components), ("pbe-fr", "pbe", Approximation::TammDancoff, components)], &mut failures);
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// T8: a degenerate spinor pair (an |Omega| = 1 pair of the closed-shell Au-H, `pbe-fr`, Casida)
/// has no gradient of one member, but its state-averaged energy has one. The mean of the two
/// members' gradients is invariant under a unitary turn within the pair, `excited_state` on
/// either root returns it, and it follows Richardson differences of the reconverged mean
/// `E_0 + (omega_a + omega_b) / 2` on the base grid.
#[test]
#[ignore = "slow; run: cargo test --profile release-test -p mandacaru-core --test molecular_spin_orbit -- --ignored the_state_averaged_gradient_of_a_degenerate_spinor_pair --nocapture"]
fn the_state_averaged_gradient_of_a_degenerate_spinor_pair_follows_its_energy() {
    use mandacaru_core::molecular::excitations::{excitations, excited_state, gradient::seams, Approximation};
    use paw_molecular_support::dataset;
    let ctx = Context::new(Some(3), 1 << 30).unwrap();
    let mut failures: Vec<String> = vec![];
    {
        let (set, functional) = ("pbe-fr", "pbe");
        let positions = [[0.07, -0.04, 0.05], [0.31, 0.22, 2.95]];
        let electrons = (dataset::read(&ctx, set, "Au").valence_charge + dataset::read(&ctx, set, "H").valence_charge).round() as usize;
        let options = mandacaru_core::molecular::ScfOptions { tolerance: 1e-12, max_iterations: 300, ..Default::default() };
        let request = spinor_request(10, Approximation::Casida);
        let base = gold_hydride_of(&ctx, set, &positions);
        let state = base.spinor_kohn_sham(functional, electrons, &options).unwrap();
        assert!(state.converged);
        let mut found = excitations(&base, &state, &request).unwrap();
        let w = found.energies.clone();
        let pair = (0..w.len() - 1).find(|&j| (w[j + 1] - w[j]).abs() < 1e-6).expect("a degenerate pair");
        println!("Au-H {set} casida: states {:?}; pair ({pair}, {}) split by {:.2e} Ha", w.iter().map(|x| format!("{x:.10}")).collect::<Vec<_>>(), pair + 1, w[pair + 1] - w[pair]);
        let gradient = |found: &mandacaru_core::molecular::excitations::Excitations, j: usize| seams::spinor_state(&base, &state, found, j).unwrap().0;
        let (a, b) = (gradient(&found, pair), gradient(&found, pair + 1));
        let mean: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
        let spread = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f64, f64::max);
        // A unitary turn within the pair, `(a, b) -> (c a + s e^{i phi} b, -s e^{-i phi} a + c b)`,
        // under which a state scales as `z (X, Y) = (z X, conj(z) Y)` (`p = X + Y`, `q = X - Y`).
        {
            let spinors = found.spinor.as_mut().unwrap();
            let (c, s, phase) = (0.6f64, 0.8f64, C64::from_polar(1., 0.7));
            let half = C64::new(0.5, 0.);
            let parts = |j: usize| ((&spinors.plus[j] + &spinors.minus[j]) * half, (&spinors.plus[j] - &spinors.minus[j]) * half);
            let ((xa, ya), (xb, yb)) = (parts(pair), parts(pair + 1));
            let combine = |za: C64, zb: C64| (&xa * za + &xb * zb, &ya * za.conj() + &yb * zb.conj());
            let (first, second) = (combine(C64::new(c, 0.), phase * s), combine(-phase.conj() * s, C64::new(c, 0.)));
            for (j, (x, y)) in [(pair, first), (pair + 1, second)] {
                spinors.plus[j] = &x + &y;
                spinors.minus[j] = &x - &y;
            }
        }
        let (a2, b2) = (gradient(&found, pair), gradient(&found, pair + 1));
        let turned = a2.iter().zip(&b2).zip(&mean).map(|((x, y), m)| (0.5 * (x + y) - m).abs()).fold(0f64, f64::max);
        let averaged = excited_state(&base, &state, &request, pair).unwrap();
        let routed = averaged.gradient.total.as_slice().iter().zip(&mean).map(|(x, m)| (x - m).abs()).fold(0f64, f64::max);
        println!("  members differ by {spread:.3e} Ha/Bohr; the mean moves by {turned:.2e} under a turn within the pair; excited_state's gradient is the mean to {routed:.2e}; its omega {:.10}", averaged.excitation);
        if turned > 1e-9 || routed > 1e-12 || (averaged.excitation - 0.5 * (w[pair] + w[pair + 1])).abs() > 1e-14 {
            failures.push(format!("invariance {turned:.2e}, routing {routed:.2e}"));
        }
        let solve = |positions: &[[f64; 3]; 2]| {
            let geometry = mandacaru_core::Geometry::new(base.geometry.numbers().to_vec(), positions.to_vec(), *base.geometry.cell(), base.geometry.pbc(), mandacaru_core::LengthUnit::Bohr).unwrap();
            let integrals = base.at_geometry(geometry).unwrap();
            let state = integrals.spinor_kohn_sham(functional, electrons, &options).unwrap();
            assert!(state.converged);
            let found = excitations(&integrals, &state, &request).unwrap();
            let w = &found.energies;
            println!("    pair {:.10} {:.10} (split {:.2e})", w[pair], w[pair + 1], w[pair + 1] - w[pair]);
            state.free_energy() + integrals.nuclear + 0.5 * (w[pair] + w[pair + 1])
        };
        let mut worst = 0f64;
        for (atom, axis) in [(0, 2), (1, 2), (1, 0)] {
            let central = |h: f64| {
                let (mut up, mut down) = (positions, positions);
                up[atom][axis] += h;
                down[atom][axis] -= h;
                (solve(&up) - solve(&down)) / (2. * h)
            };
            let (coarse, fine) = (central(1e-3), central(5e-4));
            let reference = (4. * fine - coarse) / 3.;
            let analytic = mean[3 * atom + axis];
            println!("  ({atom}, {axis}): state-averaged {analytic:+.10} FD {reference:+.10} diff {:.2e}; members {:+.10} {:+.10}", analytic - reference, a[3 * atom + axis], b[3 * atom + axis]);
            worst = worst.max((analytic - reference).abs());
        }
        println!("Au-H {set} casida pair: max |dE| {worst:.3e} Ha/Bohr");
        if worst > 2e-7 {
            failures.push(format!("{worst:.3e} Ha/Bohr"));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert!(failures.is_empty(), "{failures:#?}");
}
