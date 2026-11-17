//! The core-valence exchange of PAW Hartree-Fock (`paw::core_exchange`, stage 4-0c):
//!
//! - C2: the angular factor `(q_c / 2) (l L l_c; 0 0 0)^2` of the closed form equals an
//!   independent product quadrature over the sphere (Gauss-Legendre in `cos theta`, uniform in
//!   `phi`) of the explicit `m_c` sum of real Gaunt integrals, for `l, l_c <= 3`, at every `m`;
//!   and the term reduces, for a closed `s` core, to the one-multipole form the 1s-only path used.
//! - C0: for every Li-Ne dataset of `lda-sr/` and `pbe-sr/` (a single closed 1s core), the term
//!   built from the core orbital solved again on the dataset's own atom equals the term built
//!   from the orbital the stored density gives (`sqrt(2 pi n_c)`), to twice the density's
//!   reproduction error.
//!
//! - Q9 (the wiring): `FrozenCore` builds the core of a record's stored states or refuses it by
//!   name; the molecule's and the supercell's one-center operator (`hartree_fock_onsite_channels`,
//!   one owner) is the closed-1s term for Li-Ne and the multi-shell term for Si and Cu, in the raw
//!   projector basis (`V^-1 X V^-T`); a record without core states or with an open subshell is
//!   refused naming its element; the molecular Hartree-Fock of SiH4 and Cu2 converges.
//!
//! Needs `MANDACARU_PAW_PATH` (read only).
mod paw_molecular_support;
mod paw_supercell_support;
mod setup_shells_support;

use mandacaru_core::math::angular::real;
use mandacaru_core::molecular::ScfOptions;
use mandacaru_core::paw::{
    core_exchange::{core_valence_exchange, CoreShell, FrozenCore, StoredShell},
    molecular::hartree_fock_onsite_channels,
    onecenter_radial::three_j_squared,
    pair_tables::coulomb,
    record::PawDataset,
};
use mandacaru_core::Context;
use nalgebra::DMatrix;
use paw_molecular_support::dataset;
use std::f64::consts::PI;
use std::path::PathBuf;

/// Gauss-Legendre nodes and weights on [-1, 1] (Newton on the Legendre recurrence).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.; n];
    let mut w = vec![0.; n];
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1., z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (z * p1 - p0) / (z * z - 1.);
            let step = p1 / dp;
            z -= step;
            if step.abs() < 1e-16 {
                let (mut p0, mut p1) = (1., z);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let dp = n as f64 * (z * p1 - p0) / (z * z - 1.);
                x[i] = z;
                w[i] = 2. / ((1. - z * z) * dp * dp);
                break;
            }
        }
    }
    (x, w)
}

#[test]
fn the_angular_factor_is_the_m_sum_of_gaunt_integrals_by_quadrature() {
    let (nodes, weights) = gauss_legendre(24);
    let phis = 48;
    // Integral over the sphere of Y_a Y_b Y_c (real harmonics), by the product rule.
    let integral = |a: (usize, i32), b: (usize, i32), c: (usize, i32)| -> f64 {
        let mut sum = 0.;
        for (x, w) in nodes.iter().zip(&weights) {
            let theta = x.acos();
            for k in 0..phis {
                let phi = 2. * PI * k as f64 / phis as f64;
                sum += w
                    * (2. * PI / phis as f64)
                    * real(a.0, a.1, theta, phi)
                    * real(b.0, b.1, theta, phi)
                    * real(c.0, c.1, theta, phi);
            }
        }
        sum
    };
    for l in 0..=3usize {
        for lc in 0..=3usize {
            for big_l in l.abs_diff(lc)..=l + lc {
                let closed = (2 * lc + 1) as f64 * three_j_squared(l, big_l, lc).unwrap();
                for m in -(l as i32)..=(l as i32) {
                    // A full core shell, summed over m_c and over the multipole's M.
                    let mut quadrature = 0.;
                    for mc in -(lc as i32)..=(lc as i32) {
                        for big_m in -(big_l as i32)..=(big_l as i32) {
                            let g = integral((l, m), (lc, mc), (big_l, big_m));
                            quadrature += 4. * PI / (2 * big_l + 1) as f64 * g * g;
                        }
                    }
                    assert!(
                        (quadrature - closed).abs() < 1e-12,
                        "l={l} lc={lc} L={big_l} m={m}: quadrature {quadrature} closed {closed}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_closed_s_core_reduces_to_the_single_multipole_term() {
    let n = 400;
    let r: Vec<f64> = (1..=n).map(|i| 0.01 * i as f64).collect();
    let core: Vec<f64> = r.iter().map(|x| 2. * (-2. * x).exp()).collect();
    let waves: Vec<Vec<f64>> = (0..2)
        .map(|k| {
            r.iter()
                .map(|x| x.powi(2) * (-(1. + k as f64) * x).exp())
                .collect()
        })
        .collect();
    for l in 0..=3usize {
        let got = core_valence_exchange(
            &r,
            l,
            &waves,
            &[CoreShell {
                l: 0,
                occupation: 2.,
                radial: &core,
            }],
            n,
        )
        .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let left: Vec<f64> = (0..n).map(|k| waves[i][k] * core[k]).collect();
                let right: Vec<f64> = (0..n).map(|k| waves[j][k] * core[k]).collect();
                let want = -0.5 * (coulomb(&r, &left, &right, l) + coulomb(&r, &right, &left, l))
                    / (2 * l + 1) as f64;
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-14 * want.abs().max(1e-300),
                    "l={l} ({i}, {j}): {} {want}",
                    got[(i, j)]
                );
            }
        }
    }
}

/// The first `extent` points of `r` that hold the core: where the trapezoid charge of `n_c`
/// reaches `1 - 1e-14` of its total, plus two (the 1s-only path's rule).
fn extent(r: &[f64], core: &[f64]) -> usize {
    let cumulative = |upto: usize| -> f64 {
        (1..upto)
            .map(|i| {
                0.5 * (r[i] - r[i - 1])
                    * 4.
                    * PI
                    * (r[i - 1].powi(2) * core[i - 1] + r[i].powi(2) * core[i])
            })
            .sum()
    };
    let total = cumulative(r.len());
    let mut integrated = 0.;
    for i in 1..r.len() {
        integrated += 0.5
            * (r[i] - r[i - 1])
            * 4.
            * PI
            * (r[i - 1].powi(2) * core[i - 1] + r[i].powi(2) * core[i]);
        if integrated >= (1. - 1e-14) * total {
            return (i + 2).min(r.len());
        }
    }
    r.len()
}

/// The stored states of `dataset` as the wiring reads them.
fn stored(dataset: &PawDataset) -> Vec<StoredShell<'_>> {
    dataset
        .core_states
        .as_ref()
        .expect("a version-3 record")
        .iter()
        .map(|s| StoredShell {
            n: s.n,
            l: s.l,
            occupation: s.occupation,
            orbital: s.orbital.as_slice(),
        })
        .collect()
}

/// `V^-1 X V^-T`, the operator of a channel in the raw projector basis from its bare `X`.
fn raw_basis(vanderbilt: &[f64], x: &DMatrix<f64>) -> DMatrix<f64> {
    let n = x.nrows();
    let inverse = DMatrix::from_row_slice(n, n, vanderbilt)
        .try_inverse()
        .unwrap();
    &inverse * x * inverse.transpose()
}

fn largest(m: &DMatrix<f64>) -> f64 {
    m.iter().fold(0f64, |a, v| a.max(v.abs()))
}

/// A hydrogenic-like core of three closed subshells on a grid: `1s`, `2s` and `2p` as smooth
/// radial functions (not eigenstates), with the density they sum.
fn synthetic_core() -> (Vec<f64>, Vec<f64>, Vec<(u32, u32, f64, Vec<f64>)>) {
    let r: Vec<f64> = (1..=600).map(|i| 0.01 * i as f64).collect();
    let u = |l: u32, k: f64| -> Vec<f64> {
        r.iter()
            .map(|x| x.powi(l as i32 + 1) * (-k * x).exp())
            .collect()
    };
    let norm = |u: &mut Vec<f64>| {
        let s: f64 = u
            .windows(2)
            .zip(r.windows(2))
            .map(|(v, x)| 0.5 * (x[1] - x[0]) * (v[0].powi(2) + v[1].powi(2)))
            .sum();
        u.iter_mut().for_each(|v| *v /= s.sqrt());
    };
    let mut states = vec![
        (1, 0, 2., u(0, 6.)),
        (2, 0, 2., u(0, 2.)),
        (2, 1, 6., u(1, 2.5)),
    ];
    states.iter_mut().for_each(|s| norm(&mut s.3));
    let density = (0..r.len())
        .map(|k| {
            states.iter().map(|s| s.2 * s.3[k].powi(2)).sum::<f64>() / (4. * PI * r[k].powi(2))
        })
        .collect();
    (r, density, states)
}

fn shells_of(states: &[(u32, u32, f64, Vec<f64>)]) -> Vec<StoredShell<'_>> {
    states
        .iter()
        .map(|s| StoredShell {
            n: s.0,
            l: s.1,
            occupation: s.2,
            orbital: &s.3,
        })
        .collect()
}

#[test]
fn a_core_the_states_cannot_describe_is_refused_by_name() {
    let (r, density, states) = synthetic_core();
    let electrons = 10.;
    assert!(FrozenCore::from_states("Xx", &r, &density, electrons, &shells_of(&states)).is_ok());
    let reason = |states: &[(u32, u32, f64, Vec<f64>)], electrons: f64, density: &[f64]| {
        FrozenCore::from_states("Xx", &r, density, electrons, &shells_of(states))
            .err()
            .unwrap()
    };
    // No states, an open subshell, a different charge, a different density: each names the element
    // and what is wrong.
    assert!(reason(&[], electrons, &density).contains("Xx: the dataset lists no core states"));
    let mut open = states.clone();
    open[2].2 = 4.;
    let text = reason(&open, electrons, &density);
    assert!(
        text.contains("Xx") && text.contains("partially filled"),
        "{text}"
    );
    let text = reason(&states, 9., &density);
    assert!(text.contains("Xx") && text.contains("hold"), "{text}");
    let mut bent = density.clone();
    bent[100] *= 1.01;
    let text = reason(&states, electrons, &bent);
    assert!(
        text.contains("Xx") && text.contains("stored core density"),
        "{text}"
    );
    let mut short = states.clone();
    short[0].3.pop();
    assert!(reason(&short, electrons, &density).contains("points on a grid"));
}

#[test]
fn the_exchange_of_a_frozen_core_is_the_sum_of_its_subshells() {
    let (r, density, states) = synthetic_core();
    let core = FrozenCore::from_states("Xx", &r, &density, 10., &shells_of(&states)).unwrap();
    let waves: Vec<Vec<f64>> = (0..2)
        .map(|k| {
            r.iter()
                .map(|x| x.powi(2) * (-(0.8 + k as f64) * x).exp())
                .collect()
        })
        .collect();
    for l in 0..=3usize {
        let total = core.exchange(&r, l, &waves).unwrap();
        let mut sum = DMatrix::zeros(2, 2);
        for s in &states {
            // The subshell alone, on its own density.
            let rho: Vec<f64> = (0..r.len())
                .map(|k| s.2 * s.3[k].powi(2) / (4. * PI * r[k].powi(2)))
                .collect();
            let alone = [StoredShell {
                n: s.0,
                l: s.1,
                occupation: s.2,
                orbital: &s.3,
            }];
            sum += FrozenCore::from_states("Xx", &r, &rho, s.2, &alone)
                .unwrap()
                .exchange(&r, l, &waves)
                .unwrap();
        }
        // Each subshell is cut where its own charge is complete to 1e-14 (its extent), the sum
        // where the whole core's is: the terms differ by the tail beyond the smaller extents, which
        // is first order in the core amplitude there (about the square root of the 1e-14 charge
        // left out, so a few 1e-9 of the term: 5.6e-9 measured for l = 0).
        let gap = largest(&(&total - &sum)) / largest(&total);
        println!("synthetic core, l = {l}: sum of subshells against the whole core {gap:.2e}");
        assert!(gap <= 1e-7, "l = {l}: {gap:e}");
        // The valence's exchange with a core is attractive: negative semidefinite.
        let eigenvalues = total.symmetric_eigen().eigenvalues;
        assert!(
            eigenvalues.iter().all(|v| *v <= 1e-14),
            "l = {l}: {eigenvalues}"
        );
    }
}

#[test]
fn the_operator_of_li_to_ne_is_the_closed_1s_term_in_the_raw_basis() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let mut worst = 0f64;
    for set in ["lda-sr", "pbe-sr"] {
        for symbol in ["Li", "Be", "B", "C", "N", "O", "F", "Ne"] {
            let data = dataset::read(&ctx, set, symbol);
            let r = data.r.as_slice();
            let size = extent(r, data.core_density.as_slice());
            let from_density: Vec<f64> = data
                .core_density
                .as_slice()
                .iter()
                .map(|n| (2. * PI * n).sqrt())
                .collect();
            let operator = hartree_fock_onsite_channels(&ctx, &data).unwrap();
            for (channel, found) in data.channels.values().zip(&operator) {
                let waves: Vec<Vec<f64>> = channel
                    .ae_waves
                    .iter()
                    .map(|w| w.as_slice().to_vec())
                    .collect();
                let shell = CoreShell {
                    l: 0,
                    occupation: 2.,
                    radial: &from_density,
                };
                let old = core_valence_exchange(r, found.l, &waves, &[shell], size).unwrap();
                let old = raw_basis(channel.vanderbilt.data(), &old);
                let deviation = largest(&(&found.core_exchange - &old)) / largest(&old);
                worst = worst.max(deviation);
                assert!(
                    deviation < 1e-8,
                    "{set}/{symbol} l = {}: {deviation:e}",
                    found.l
                );
            }
        }
    }
    println!("Li-Ne: the operator against the closed-1s route of the stored density, worst {worst:.3e} relative");
}

#[test]
fn silicon_and_copper_carry_their_multi_shell_core_exchange() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    for (set, symbol, electrons) in [
        ("lda-sr", "Si", 10.),
        ("lda-sr", "Cu", 10.),
        ("pbe-sr", "Si", 10.),
    ] {
        let data = dataset::read(&ctx, set, symbol);
        let r = data.r.as_slice();
        let shells = stored(&data);
        assert!(shells.len() > 1, "{symbol} has a multi-shell core");
        let core =
            FrozenCore::from_states(symbol, r, data.core_density.as_slice(), electrons, &shells)
                .unwrap();
        let operator = hartree_fock_onsite_channels(&ctx, &data).unwrap();
        for (channel, found) in data.channels.values().zip(&operator) {
            let waves: Vec<Vec<f64>> = channel
                .ae_waves
                .iter()
                .map(|w| w.as_slice().to_vec())
                .collect();
            let bare = core.exchange(r, found.l, &waves).unwrap();
            // The wiring is the bare term in the raw projector basis.
            let want = raw_basis(channel.vanderbilt.data(), &bare);
            assert!(largest(&(&found.core_exchange - &want)) <= 1e-12 * largest(&want));
            // Attractive and symmetric (the bare term, in the basis of the waves).
            assert!(largest(&(&bare - bare.transpose())) <= 1e-12 * largest(&bare));
            let eigenvalues = bare.clone().symmetric_eigen().eigenvalues;
            assert!(
                eigenvalues.iter().all(|v| *v <= 1e-12),
                "{symbol} l = {}: {eigenvalues}",
                found.l
            );
            // Shell by shell, the multipoles of `(l L l_c)`: the term of one subshell with the
            // `q_c / 2` weight equals the sum of its multipoles, `-(q_c/2) (l L l_c)^2 R^L`.
            let (rows, columns) = (bare.nrows(), bare.ncols());
            let mut by_multipole = DMatrix::zeros(rows, columns);
            let size = extent(r, data.core_density.as_slice());
            for s in &shells {
                let radial: Vec<f64> = s.orbital.iter().zip(r).map(|(u, x)| u / x).collect();
                let pairs: Vec<Vec<f64>> = waves
                    .iter()
                    .map(|w| (0..size).map(|k| w[k] * radial[k]).collect())
                    .collect();
                let (l, lc) = (found.l, s.l as usize);
                let mut line = format!(
                    "{symbol} ({}{}) q = {} channel l = {l}:",
                    s.n,
                    ["s", "p", "d", "f"][lc],
                    s.occupation
                );
                for big_l in l.abs_diff(lc)..=l + lc {
                    let angular = three_j_squared(l, big_l, lc).unwrap();
                    if angular == 0. {
                        continue;
                    }
                    let mut part = DMatrix::zeros(bare.nrows(), bare.ncols());
                    for i in 0..part.nrows() {
                        for j in 0..part.ncols() {
                            part[(i, j)] = -0.5
                                * s.occupation
                                * angular
                                * 0.5
                                * (coulomb(&r[..size], &pairs[i], &pairs[j], big_l)
                                    + coulomb(&r[..size], &pairs[j], &pairs[i], big_l));
                        }
                    }
                    line += &format!("  L = {big_l}: X[0,0] = {:.6e}", part[(0, 0)]);
                    by_multipole += part;
                }
                println!("{line}");
            }
            let gap = largest(&(&bare - &by_multipole)) / largest(&bare);
            println!(
                "{symbol} l = {}: multipole sum against the operator {gap:.2e}",
                found.l
            );
            assert!(gap <= 1e-9, "{symbol} l = {}: {gap:e}", found.l);
            println!(
                "{set}/{symbol} l = {}: X[0,0] = {:.6e} Ha, largest |X| = {:.6e} Ha",
                found.l,
                bare[(0, 0)],
                largest(&bare)
            );
        }
    }
}

#[test]
fn a_record_without_core_states_or_with_an_open_subshell_is_refused_by_name() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let mut data = dataset::read(&ctx, "lda-sr", "Si");
    assert!(hartree_fock_onsite_channels(&ctx, &data).is_ok());
    data.core_states.as_mut().unwrap()[2].occupation = 5.;
    let text = hartree_fock_onsite_channels(&ctx, &data)
        .err()
        .unwrap()
        .to_string();
    assert!(
        text.contains("Si") && text.contains("partially filled"),
        "{text}"
    );
    data.core_states = None;
    let text = hartree_fock_onsite_channels(&ctx, &data)
        .err()
        .unwrap()
        .to_string();
    assert!(
        text.contains("Si") && text.contains("no core states"),
        "{text}"
    );
    // A core-free record needs no states.
    let hydrogen = dataset::read(&ctx, "lda-sr", "H");
    assert!(hartree_fock_onsite_channels(&ctx, &hydrogen).is_ok());
}

/// RHF (Hartree) of `integrals`: the total energy with the nuclear repulsion, and the iterations.
fn rhf_energy(integrals: &mandacaru_core::molecular::MolecularIntegrals) -> (f64, usize) {
    let n = integrals.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-10,
        max_iterations: 300,
        ..Default::default()
    };
    let state = integrals
        .mean_field("rhf", [n / 2, n / 2], &options)
        .unwrap();
    assert!(
        state.converged,
        "RHF did not converge in {} iterations",
        state.iterations.len()
    );
    (
        state.electronic_energy + integrals.nuclear,
        state.iterations.len(),
    )
}

/// A molecule in a cubic box on the real atomic basis, for the library's `lda-sr` datasets.
fn in_a_box(
    ctx: &Context,
    atoms: &[(&'static str, [f64; 3])],
    side_angstrom: f64,
    spacing_angstrom: f64,
) -> mandacaru_core::periodic::setup::PeriodicSetup {
    let bohr = 1. / 0.529177210903;
    let nodes = (side_angstrom / spacing_angstrom).round() as usize;
    paw_supercell_support::molecule_in_a_box(
        ctx,
        atoms,
        nodes as f64 * spacing_angstrom * bohr,
        nodes,
    )
}

#[test]
fn the_molecular_hartree_fock_of_silane_and_dicopper_converges() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let d = 2.8 / 3f64.sqrt();
    let silane = vec![
        ("Si", [0., 0., 0.]),
        ("H", [d, d, d]),
        ("H", [d, -d, -d]),
        ("H", [-d, d, -d]),
        ("H", [-d, -d, d]),
    ];
    let setup = in_a_box(&ctx, &silane, 7., 0.25);
    let (energy, iterations) =
        rhf_energy(&paw_supercell_support::molecular_integrals(&ctx, &setup));
    println!("SiH4 RHF {energy:.8} Ha ({iterations} iterations)");
    assert!(energy.is_finite());
    let copper = vec![("Cu", [-2.3, 0., 0.]), ("Cu", [2.3, 0., 0.])];
    let setup = in_a_box(&ctx, &copper, 8., 0.25);
    let (energy, iterations) =
        rhf_energy(&paw_supercell_support::molecular_integrals(&ctx, &setup));
    println!("Cu2 RHF {energy:.8} Ha ({iterations} iterations)");
    assert!(energy.is_finite());
}

#[test]
fn the_silicon_supercell_builds_its_one_center_operator_and_runs_hartree_fock() {
    use mandacaru_core::molecular::{periodic::paw::HamiltonianKind, MolecularIntegrals};
    use std::sync::Arc;
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    for mesh in [[1, 1, 1], [2, 1, 1]] {
        let setup = paw_supercell_support::cells_of(
            &ctx,
            "lda-sr",
            false,
            &paw_supercell_support::silicon_cell(),
            1,
            [0.; 3],
        );
        let started = std::time::Instant::now();
        let integrals =
            MolecularIntegrals::paw_supercell(&ctx, Arc::new(setup), mesh, None).unwrap();
        assert_eq!(integrals.kind(), HamiltonianKind::PawSupercell);
        let built = started.elapsed().as_secs_f64();
        let (energy, iterations) = rhf_energy(&integrals);
        let cells: usize = mesh.iter().product();
        println!(
            "Si {mesh:?}: {} functions, {} electrons, RHF {:.8} Ha per cell ({iterations} iterations), build {built:.1} s, total {:.1} s",
            integrals.m(),
            integrals.neutral_electrons(),
            energy / cells as f64,
            started.elapsed().as_secs_f64()
        );
    }
}

/// Every cored file of the library, its core states against `FrozenCore::from_states`: the files
/// Hartree-Fock refuses, by element and reason (3.5 s for the 360 cored files of the four sets) (an open subshell, a charge or density mismatch).
#[test]
fn the_library_cores_are_all_accepted_or_refused_by_name() {
    let root = PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let (mut checked, mut refused) = (0usize, Vec::new());
    for set in ["lda-sr", "pbe-sr", "lda-fr", "pbe-fr"] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(set))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
            .collect();
        files.sort();
        for path in files {
            let data = PawDataset::read(&ctx, &path).unwrap();
            let electrons = data.atomic_number as f64 - data.valence_charge;
            if electrons <= 1e-6 {
                continue;
            }
            checked += 1;
            let result = match data.core_states.as_ref() {
                None => Err(format!("{}: no core states", data.symbol)),
                Some(_) => FrozenCore::from_states(
                    &data.symbol,
                    data.r.as_slice(),
                    data.core_density.as_slice(),
                    electrons,
                    &stored(&data),
                )
                .map(|_| ()),
            };
            if let Err(reason) = result {
                refused.push(format!("{set}/{reason}"));
            }
        }
    }
    println!("{checked} cored files checked, {} refused", refused.len());
    for line in &refused {
        println!("REFUSED {line}");
    }
    assert!(refused.is_empty());
}
