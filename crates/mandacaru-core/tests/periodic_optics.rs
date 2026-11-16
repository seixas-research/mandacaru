//! The crystal momentum matrix and the diamagnetic band sum (`periodic::optics::momentum`,
//! TDDFT_PLAN.md T6b0): projector-basis invariance, Hermiticity, the `-k` closure of the mesh, the
//! band sums of small insulators, the first moment of the independent-particle `epsilon_2` (R10,
//! test side), and a molecule in a box against the molecular engine (R12, velocity part). Needs
//! `MANDACARU_PAW_PATH`.
mod berry_support;
mod paw_molecular_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::{
    molecular::{
        excitations::{excitations, Approximation, Coupling, Request, Spin},
        ScfOptions,
    },
    periodic::{
        bloch_mesh::MeshIndex,
        optics::{
            momentum::{check_inversion_closed, diamagnetic_sum, one_center_momentum, transitions},
            realtime::{self, Formulation},
        },
    },
};
use nalgebra::{Matrix3, Vector3};
use paw_molecular_support::{dataset, molecule_with};

/// The band sums of `cell` solved in `basis`, along each of `directions`, with the electron
/// count and the largest anti-Hermitian residual of `P`.
fn sums(
    cell: &Cell,
    basis: &str,
    mesh: [usize; 3],
    directions: &[[f64; 3]],
) -> Vec<(f64, f64, f64)> {
    let solved = solve_in(cell, basis);
    let sampler = solved.sampler();
    let index = MeshIndex::new(mesh, [0.; 3]).unwrap();
    directions
        .iter()
        .map(|d| {
            let out = diamagnetic_sum(&sampler, &index, *d).unwrap();
            (out.total, out.electrons, out.residual)
        })
        .collect()
}

#[test]
fn the_one_center_momentum_is_hermitian_in_both_projector_bases() {
    for basis in ["raw", "dual"] {
        let solved = solve_in(&h2_chain("lda-sr", "lda"), basis);
        let blocks = one_center_momentum(&solved.ctx, &solved.setup).unwrap();
        let scale = blocks.iter().map(|b| b.camax()).fold(0f64, f64::max);
        assert!(scale > 1e-6, "{basis}: the one-center momentum vanishes");
        for b in &blocks {
            let defect = (b - b.adjoint()).camax();
            assert!(defect < 1e-10 * scale, "{basis}: not Hermitian, {defect:e}");
        }
    }
}

#[test]
fn the_one_center_momentum_of_lithium_hydride_is_hermitian() {
    // The rock-salt LiH SZ momentum failed its bound (1.2e-2 at grid 16^3): every block of Delta p,
    // lithium's included, must be Hermitian in both projector bases.
    for basis in ["raw", "dual"] {
        let solved = solve_in(&lih("lda-sr"), basis);
        let blocks = one_center_momentum(&solved.ctx, &solved.setup).unwrap();
        let scale = blocks.iter().map(|b| b.camax()).fold(0f64, f64::max);
        let defect = blocks.iter().map(|b| (b - b.adjoint()).camax()).fold(0f64, f64::max);
        eprintln!("LiH {basis}: Delta p scale {scale:.3e}, anti-Hermitian {defect:.3e}");
        assert!(defect < 1e-10 * scale, "{basis}: not Hermitian, {defect:e}");
    }
}

#[test]
fn the_diamagnetic_sum_does_not_depend_on_the_projector_basis() {
    // Raw and dual projectors are two coordinates of one PAW transform: the band sum (a physical
    // observable of the converged bands) must agree, which checks the move of Delta p to the raw
    // basis. H2 along x at Gamma; LiH on a 2 x 1 x 1 mesh along its bond.
    for (cell, mesh, label) in [
        (h2_chain("lda-sr", "lda"), [1, 1, 1], "H2"),
        (lih("lda-sr"), [2, 1, 1], "LiH"),
    ] {
        let directions = [[1., 0., 0.], [0., 1., 0.]];
        let raw = sums(&cell, "raw", mesh, &directions);
        let dual = sums(&cell, "dual", mesh, &directions);
        for ((direction, (raw, electrons, residual)), (dual, _, _)) in
            directions.iter().zip(raw).zip(dual)
        {
            eprintln!(
                "{label} along {direction:?}: N_dia raw {raw:.8}, dual {dual:.8}, N {electrons}, \
                 anti-Hermitian residual {residual:.2e}"
            );
            // A minimal s basis has no transition across the H2 axis: the sum is zero there.
            assert!(
                raw >= 0. && (raw - dual).abs() <= 1e-6 * raw.abs().max(1e-12),
                "{label}: {raw} vs {dual}"
            );
        }
    }
}

#[test]
fn a_mesh_not_closed_under_inversion_is_refused() {
    assert!(check_inversion_closed(&MeshIndex::new([2, 2, 1], [0.; 3]).unwrap()).is_ok());
    assert!(check_inversion_closed(&MeshIndex::new([3, 1, 1], [0.; 3]).unwrap()).is_ok());
    let shifted = MeshIndex::new([2, 1, 1], [0.1, 0., 0.]).unwrap();
    let message = check_inversion_closed(&shifted).unwrap_err().to_string();
    assert!(message.contains("-k"), "{message}");
}

/// H2 (1.4 Bohr along x) centered in a cubic box of `length` Bohr on a grid of `spacing` Bohr.
fn h2_in_box(length: f64, spacing: f64) -> Cell {
    let n = (length / spacing).round() as usize;
    let half = length / 2.;
    Cell {
        library: "lda-sr",
        symbols: vec!["H", "H"],
        lattice: Matrix3::from_diagonal(&Vector3::new(length, length, length)),
        grid: [n, n, n],
        fractional: vec![
            [(half - 0.7) / length, 0.5, 0.5],
            [(half + 0.7) / length, 0.5, 0.5],
        ],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [1, 1, 1],
        hubbard: None,
        width: 1e-3,
    }
}

/// Transitions grouped by gap (within `GROUP` Hartree): the mean gap and the summed weight of
/// each group, ascending in gap.
fn grouped(mut lines: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    const GROUP: f64 = 1e-4;
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64, usize)> = vec![];
    for (gap, weight) in lines {
        match out.last_mut() {
            Some(last) if gap - last.0 / last.2 as f64 <= GROUP => {
                last.0 += gap;
                last.1 += weight;
                last.2 += 1;
            }
            _ => out.push((gap, weight, 1)),
        }
    }
    out.into_iter().map(|(g, w, n)| (g / n as f64, w)).collect()
}

/// R12 (velocity part): the crystal engine at Gamma against the molecular engine on the same
/// molecule, radial tables, dataset and projector basis (H2 DZP, LDA, independent particles).
/// Box sequence measured 2026-11-15 (grid 0.4 Bohr): the largest gap difference is 8.5e-3 Ha at
/// 10.4 Bohr, 2.0e-4 at 12.8 and 3.3e-5 at 16; the bright sigma -> sigma* group's velocity weight
/// differs by 2.5e-3, 1.5e-4 and 1.6e-4 (relative), the along-bond `N_dia` by 4.5e-3, 1.0e-4 and
/// 1.9e-5. The weak line at 2.88 Ha plateaus at 4.6e-3 to 4.9e-3 from 12.8 Bohr on: not a box
/// effect, and not attributed (the two engines' grids differ: the molecular one is not periodic
/// and has one more point per axis); recorded here and not widened. The k-derivative (Peierls)
/// velocity of the Bloch sums is 1e-9 against a largest |p_n| of 1.17 / Bohr: it carries none of
/// the intra-molecular velocity P holds. The gate runs the converged 16 Bohr box.
#[test]
fn a_molecule_in_a_box_has_the_molecular_velocity_strengths() {
    {
        let length = 16.;
        let spacing = 0.4;
        let cell = h2_in_box(length, spacing);
        let solved = solve_sized(&cell, "raw", "DZP");
        let sampler = solved.sampler();
        let gamma = MeshIndex::new([1, 1, 1], [0.; 3]).unwrap();
        let axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let along: Vec<_> = axes
            .iter()
            .map(|e| transitions(&sampler, &gamma, *e).unwrap())
            .collect();
        // 2 |P_cv|^2 summed over the axes: the singlet |p_n|^2 of the same transition.
        let crystal: Vec<(f64, f64)> = (0..along[0].gaps.len())
            .map(|t| {
                let weight: f64 = along.iter().map(|a| a.degeneracy * a.weights[t]).sum();
                (along[0].gaps[t], weight)
            })
            .collect();
        let crystal_dia = along[0].diamagnetic(1).total;

        // The molecular engine on the crystal's own radial tables (atom 0; H2 is homonuclear).
        let description = solved.setup.description.as_ref().unwrap();
        let mut seen = vec![];
        let mut shells = vec![];
        for (f, &atom) in description.function_atom.iter().enumerate() {
            let t = description.function_table[f];
            if atom == 0 && !seen.contains(&t) {
                seen.push(t);
                let table = &description.tables[t];
                assert!(table.confinement.is_none());
                shells.push((table.l, table.r.clone(), table.values.clone()));
            }
        }
        let ctx = &solved.ctx;
        let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
        let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "lda-sr", "H"));
        let positions: Vec<[f64; 3]> = cell
            .fractional
            .iter()
            .map(|f| [f[0] * length, f[1] * length, f[2] * length])
            .collect();
        let integrals = molecule_with(
            ctx,
            &[1, 1],
            &positions,
            length,
            spacing,
            (vec![hydrogen], vec![0, 0], arrays),
            "raw",
            &shells,
            &[],
        );
        let state = integrals
            .kohn_sham(
                "lda",
                [1, 1],
                false,
                &ScfOptions {
                    tolerance: 1e-11,
                    max_iterations: 400,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(state.converged);
        let m = state.energies[0].len();
        let mut request = Request::new(m - 1);
        request.approximation = Approximation::Casida;
        request.spin = Some(Spin::Singlet);
        request.coupling = Coupling::Independent;
        request.tolerance = 1e-10;
        let roots = excitations(&integrals, &state, &request).unwrap();
        let molecule: Vec<(f64, f64)> = roots
            .energies
            .iter()
            .zip(&roots.transition_velocities)
            .map(|(w, p)| (*w, p.iter().map(|v| v * v).sum()))
            .collect();
        let molecule_dia: f64 = roots
            .energies
            .iter()
            .zip(&roots.transition_velocities)
            .map(|(w, p)| 2. * p[0] * p[0] / w)
            .sum();

        let (a, b) = (grouped(crystal), grouped(molecule));
        eprintln!(
            "box {length}: {} crystal groups, {} molecular groups",
            a.len(),
            b.len()
        );
        assert_eq!(
            a.len(),
            b.len(),
            "the two engines group the transitions differently"
        );
        let brightest = b.iter().fold(0f64, |t, g| t.max(g.1));
        for (x, y) in a.iter().zip(&b) {
            eprintln!(
                "  gap {:.6} / {:.6} Ha, weight {:.6e} / {:.6e}",
                x.0, y.0, x.1, y.1
            );
            assert!((x.0 - y.0).abs() < 1e-4, "gap {} vs {}", x.0, y.0);
            let relative = (x.1 - y.1).abs() / y.1.max(1e-300);
            if y.1 > 0.1 * brightest {
                assert!(relative < 1e-3, "bright line at {}: {relative:e}", y.0);
            } else if y.1 > 1e-3 * brightest {
                assert!(relative < 1e-2, "weak line at {}: {relative:e}", y.0);
            } else {
                assert!(
                    x.1 < 1e-12 * brightest,
                    "a dark molecular line is bright: {}",
                    x.1
                );
            }
        }
        eprintln!("  N_dia along the bond: crystal {crystal_dia:.6}, molecule {molecule_dia:.6}");
        assert!((crystal_dia - molecule_dia).abs() < 1e-4 * molecule_dia);

        // The k-derivative (Peierls) velocity of the Bloch sums: for a molecule whose images do
        // not overlap, S(k) and the one-body matrix do not depend on k, so that velocity misses
        // what P carries (the intra-atomic and intra-molecular part). Reported, not gated.
        let step = 1e-4;
        let points = solved
            .setup
            .matrices_at(ctx, &[[step, 0., 0.], [-step, 0., 0.]], &[0.5, 0.5])
            .unwrap();
        let dk = 2. * step * 2. * std::f64::consts::PI / length;
        let overlap = (&points[0].overlap - &points[1].overlap).camax() / dk;
        let fixed = (&points[0].fixed - &points[1].fixed).camax() / dk;
        let momentum = a.iter().fold(0f64, |t, g| t.max(g.1)).sqrt();
        eprintln!(
            "  Peierls velocity: max |dS/dk| {overlap:.3e}, max |d fixed/dk| {fixed:.3e} Bohr; \
             largest |p_n| {momentum:.3e} / Bohr"
        );
    }
}

#[test]
fn the_first_moment_of_the_independent_epsilon2_is_the_band_sum() {
    // R10, test side: a self-consistency check of the weights `independent_epsilon2` and the
    // band sum share (the same P, the same gaps), `int omega epsilon_2 d omega = 2 pi^2 N_dia /
    // Omega`, with lines narrow against the smallest gap. The sign and the prefactor against an
    // independent route are pinned by the real-time run (R10, T6b).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    for direction in [[1., 0., 0.], [0., 1., 0.]] {
        let lines = transitions(&sampler, &mesh, direction).unwrap();
        let dia = lines.diamagnetic(sampler.channels()).total;
        let smallest = lines.gaps.iter().fold(f64::INFINITY, |a, g| a.min(*g));
        let largest = lines.gaps.iter().fold(0f64, |a, g| a.max(*g));
        let width = 0.002 * smallest;
        let step = width / 8.;
        let count = ((largest + 12. * width) / step).ceil() as usize;
        let frequencies: Vec<f64> = (1..=count).map(|i| i as f64 * step).collect();
        let epsilon2 = lines
            .independent_epsilon2(&solved.ctx, &frequencies, width)
            .unwrap();
        assert!(
            epsilon2.iter().all(|v| *v >= 0.),
            "epsilon_2 is not positive"
        );
        let moment: f64 = frequencies
            .iter()
            .zip(&epsilon2)
            .map(|(w, e)| w * e * step)
            .sum();
        let expected = 2. * std::f64::consts::PI.powi(2) * dia / lines.volume;
        eprintln!(
            "LiH along {direction:?}: first moment {moment:.10e}, 2 pi^2 N_dia / Omega \
             {expected:.10e} (N_dia {dia:.8})"
        );
        assert!(
            expected > 0. && (moment - expected).abs() < 1e-4 * expected,
            "{moment} vs {expected}"
        );
    }
}

/// Silicon (fcc, 10.26 Bohr, two atoms) on 20^3 points.
fn silicon() -> Cell {
    let a = 10.26 / 2.;
    Cell {
        library: "lda-sr",
        symbols: vec!["Si", "Si"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid: [20, 20, 20],
        fractional: vec![[0.; 3], [0.25; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [4, 4, 4],
        hubbard: None,
        width: 1e-3,
    }
}

/// Lithium fluoride (rocksalt, 4.03 Angstrom, primitive cell) on 24^3 points.
fn lithium_fluoride() -> Cell {
    let a = 4.03 / 0.529177210903 / 2.;
    Cell {
        library: "lda-sr",
        symbols: vec!["Li", "F"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid: [24, 24, 24],
        fractional: vec![[0.; 3], [0.5; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [4, 4, 4],
        hubbard: None,
        width: 1e-3,
    }
}

#[test]
#[ignore = "R9a, slow: Si and LiF at three PAW-LCAO sizes and two meshes (Si 349 s, LiF 307 s, 1.3 GB)"]
fn the_production_band_sum_against_the_converged_one() {
    // R9a (Q20 revised, Q30a): N_dia along x at the production basis (DZP) against the largest
    // size the generator offers (QZ2P) of the same dataset, on the SCF's 4^3 whole mesh and on a
    // 6^3 refinement (the states off the SCF mesh come from the converged potential). Q30: the
    // gate is decided only if QZ2P moved less than 1 % from the next size down (QZP); otherwise
    // the verdict is "undecided" and F1 stands. Both N and N_dia are reported, never compared.
    // Measured 2026-11-15 (v3 library): (QZ2P against QZP, DZP against QZ2P on 4^3 and 6^3).
    // Si is decided and fires Q20 (DECISIONS.md: F2 owed, F1 kept until F1 against F2 is
    // measured); LiF is undecided (its largest size still moves 2.8 %). The gate pins them.
    let recorded = [
        ("Si", 0.00449, -0.06103, -0.06263),
        ("LiF", 0.02841, -0.12630, -0.13263),
    ];
    let only = std::env::var("R9A_ONLY").ok();
    for (cell, label) in [(silicon(), "Si"), (lithium_fluoride(), "LiF")] {
        if only.as_deref().is_some_and(|o| o != label) {
            continue;
        }
        let mut table = vec![];
        for size in ["DZP", "QZP", "QZ2P"] {
            let solved = solve_on(
                &cell,
                "raw",
                size,
                mandacaru_core::Context::new(Some(2), 3 << 29).unwrap(),
            );
            let gamma = &solved.crystal.points[0];
            let (overlap, _) = mandacaru_core::math::linalg::eigen(gamma.overlap.clone()).unwrap();
            let sampler = solved.sampler();
            let mut sums = vec![];
            for mesh in [[4, 4, 4], [6, 6, 6]] {
                if size == "QZP" && mesh[0] == 6 {
                    continue;
                }
                let index = MeshIndex::new(mesh, [0.; 3]).unwrap();
                let out = diamagnetic_sum(&sampler, &index, [1., 0., 0.]).unwrap();
                eprintln!(
                    "{label} {size} mesh {mesh:?}: N_dia {:.6}, N {}, residual {:.2e}, m {}, \
                     smallest overlap eigenvalue {:.3e}",
                    out.total, out.electrons, out.residual, solved.crystal.m, overlap[0]
                );
                sums.push(out.total);
            }
            table.push((size, sums));
        }
        let (dzp, qzp, qz2p) = (&table[0].1, &table[1].1[0], &table[2].1);
        let ladder = (qz2p[0] - qzp).abs() / qz2p[0];
        let deficit: Vec<f64> = dzp.iter().zip(qz2p).map(|(p, c)| (p - c) / c).collect();
        eprintln!(
            "{label}: QZ2P against QZP {:.3} %; DZP against QZ2P {:.3} % (4^3), {:.3} % (6^3)",
            100. * ladder,
            100. * deficit[0],
            100. * deficit[1]
        );
        let verdict = if ladder >= 0.01 {
            "undecided (the largest size moved more than 1 %)"
        } else if deficit.iter().any(|d| d.abs() > 0.05) {
            "Q20 fires: the production basis misses the converged N_dia by more than 5 %"
        } else {
            "F1 passes"
        };
        eprintln!("{label}: {verdict}");
        let (_, l, d4, d6) = recorded.iter().find(|r| r.0 == label).unwrap();
        assert!(
            (ladder - l).abs() < 2e-4
                && (deficit[0] - d4).abs() < 2e-4
                && (deficit[1] - d6).abs() < 2e-4,
            "{label}: the band sums moved from the recorded R9a values"
        );
    }
}

/// The F2 transfer weights of `solved` at the fractional `k`: per occupied-empty transition of
/// channel 0, its gap and `|<u_c,k+A|u_v,k>|^2 / A^2` (the symmetric pair of `+-A` along the
/// Cartesian `axis`), the first-order population a Bloch sum re-formed at `k + A` gives (Q20's
/// F2), which is `|x_cv|^2` if the periodic-part overlap carries the intra-atomic position.
fn transfer_weights(solved: &Solved, k: [f64; 3], axis: usize, strength: f64) -> Vec<(f64, f64)> {
    use mandacaru_core::periodic::wannier::sampling::Frame;
    let sampler = solved.sampler();
    let (crystal, state, setup) = (&*solved.crystal, &*solved.state, &*solved.setup);
    let ctx = &solved.ctx;
    let bands: Vec<usize> = (0..crystal.bands()).collect();
    let frame = |fractional: [f64; 3]| {
        let points = setup.matrices_at(ctx, &[fractional], &[1.]).unwrap();
        Frame::from_point(crystal, state, &points[0], 0, &bands).unwrap()
    };
    let at = frame(k);
    let occupied = state.occupations[0].iter().filter(|f| **f > 1.).count();
    let mut weights = vec![0f64; occupied * (bands.len() - occupied)];
    for sign in [1., -1.] {
        let a: [f64; 3] = std::array::from_fn(|i| if i == axis { sign * strength } else { 0. });
        let shift = sampler.reciprocal.try_inverse().unwrap() * Vector3::from(a);
        let moved = frame(std::array::from_fn(|i| k[i] + shift[i]));
        // <u_m,k|u_n,k+A>: the right frame sits at the left one's k plus the displacement.
        let overlap = sampler.overlap(&at, &moved, a).unwrap();
        for v in 0..occupied {
            for c in occupied..bands.len() {
                weights[v * (bands.len() - occupied) + c - occupied] +=
                    overlap[(v, c)].norm_sqr() / (2. * strength * strength);
            }
        }
    }
    let levels = at.energies.as_slice();
    (0..occupied)
        .flat_map(|v| (occupied..bands.len()).map(move |c| (v, c)))
        .map(|(v, c)| {
            (
                levels[c] - levels[v],
                weights[v * (bands.len() - occupied) + c - occupied],
            )
        })
        .collect()
}

#[test]
fn a_bloch_sum_at_k_plus_a_carries_the_length_form_of_the_intra_atomic_position() {
    // F2's first gate (Q20, DECISIONS.md 2026-11-15): on the H2 box of R12 the transfer weights of
    // Bloch sums re-formed at k + A are compared with the molecular engine's length-form
    // |x_cv|^2 (the independent-particle transition dipoles, |mu_n|^2 / 2 per singlet), and their
    // velocity counterparts 2 gap |x|^2 against F1's band-sum weights. Measured and printed.
    let (length, spacing) = (16., 0.4);
    let cell = h2_in_box(length, spacing);
    let solved = solve_sized(&cell, "raw", "DZP");
    let along_bond = grouped(transfer_weights(&solved, [0.; 3], 0, 1e-3));
    // The molecular engine's length weights along the bond, on the same tables.
    let description = solved.setup.description.as_ref().unwrap();
    let mut seen = vec![];
    let mut shells = vec![];
    for (f, &atom) in description.function_atom.iter().enumerate() {
        let t = description.function_table[f];
        if atom == 0 && !seen.contains(&t) {
            seen.push(t);
            let table = &description.tables[t];
            shells.push((table.l, table.r.clone(), table.values.clone()));
        }
    }
    let ctx = &solved.ctx;
    let mut arrays = mandacaru_core::paw::molecular::Arrays::new();
    let hydrogen = dataset::species(ctx, &mut arrays, &dataset::read(ctx, "lda-sr", "H"));
    let positions: Vec<[f64; 3]> = cell
        .fractional
        .iter()
        .map(|f| [f[0] * length, f[1] * length, f[2] * length])
        .collect();
    let integrals = molecule_with(
        ctx,
        &[1, 1],
        &positions,
        length,
        spacing,
        (vec![hydrogen], vec![0, 0], arrays),
        "raw",
        &shells,
        &[],
    );
    let state = integrals
        .kohn_sham(
            "lda",
            [1, 1],
            false,
            &ScfOptions {
                tolerance: 1e-11,
                max_iterations: 400,
                ..Default::default()
            },
        )
        .unwrap();
    let m = state.energies[0].len();
    let mut request = Request::new(m - 1);
    request.approximation = Approximation::Casida;
    request.spin = Some(Spin::Singlet);
    request.coupling = Coupling::Independent;
    request.tolerance = 1e-10;
    let roots = excitations(&integrals, &state, &request).unwrap();
    let molecule = grouped(
        roots
            .energies
            .iter()
            .zip(&roots.transition_dipoles)
            .map(|(w, mu)| (*w, 0.5 * mu[0] * mu[0]))
            .collect(),
    );
    let brightest = molecule.iter().fold(0f64, |t, g| t.max(g.1));
    for (x, y) in along_bond.iter().zip(&molecule) {
        eprintln!(
            "F2 H2 box: gap {:.6} / {:.6} Ha, |x|^2 {:.6e} / molecular {:.6e}",
            x.0, y.0, x.1, y.1
        );
        if y.1 > 1e-3 * brightest {
            assert!(
                (x.1 - y.1).abs() < 2e-2 * y.1,
                "line at {}: {} vs {}",
                y.0,
                x.1,
                y.1
            );
        }
    }
    // The propagation's own transfer (`realtime::prepare` with the moved basis): the transferred
    // occupied band's weight on each empty eigenvector of its H_o at k + A, over A^2.
    let strength = 1e-3;
    let mut moved = realtime::Request::new([1., 0., 0.], strength, 0.2, 1);
    moved.formulation = Formulation::Moved;
    moved.kernel = realtime::Kernel::Independent;
    let prepared = realtime::prepare(&solved.sampler(), &moved).unwrap();
    let (levels, vectors) =
        mandacaru_core::math::linalg::eigen(prepared.ground[0].clone()).unwrap();
    let orbitals = &prepared.channels[0].orbitals;
    let o = orbitals.ncols();
    let populations: Vec<(f64, f64)> = (o..levels.len())
        .map(|c| {
            let weight: f64 = (0..o)
                .map(|v| (vectors.column(c).adjoint() * orbitals.column(v))[(0, 0)].norm_sqr())
                .sum();
            (levels[c] - levels[o - 1], weight / (strength * strength))
        })
        .collect();
    for (x, y) in grouped(populations).iter().zip(&molecule) {
        eprintln!(
            "F2 transfer of the propagation: gap {:.6} Ha, |x|^2 {:.6e} / molecular {:.6e}",
            x.0, x.1, y.1
        );
        if y.1 > 1e-3 * brightest {
            assert!(
                (x.1 - y.1).abs() < 2e-2 * y.1,
                "transfer at {}: {} vs {}",
                y.0,
                x.1,
                y.1
            );
        }
    }
    // LiH: the F2 (length-form) and F1 (velocity-form) band sums at Gamma and k = 1/2 along a_1.
    let lih_solved = solve(&lih("lda-sr"));
    let mut length_sum = 0.;
    for k in [[0., 0., 0.], [0.5, 0., 0.]] {
        length_sum += transfer_weights(&lih_solved, k, 0, 1e-3)
            .iter()
            .map(|(gap, x2)| 2. * 2. * gap * x2 / 2.)
            .sum::<f64>();
    }
    let velocity = diamagnetic_sum(
        &lih_solved.sampler(),
        &MeshIndex::new([2, 1, 1], [0.; 3]).unwrap(),
        [1., 0., 0.],
    )
    .unwrap();
    eprintln!(
        "F2 LiH 2x1x1 along x: length-form sum 2 g w gap |x|^2 = {length_sum:.6}, F1 band sum \
         N_dia = {:.6}, N = {}",
        velocity.total, velocity.electrons
    );
}

#[test]
#[ignore = "F1/F2 evidence, slow: the length-form band sum of Si at DZP and QZ2P on 4^3"]
fn the_length_form_band_sum_of_silicon_against_the_converged_basis() {
    // Q20's decision after the trigger: F2's first-order populations are the length form
    // (|x_cv|^2, the H2-box gate above). Its band sum N_len = g sum_k w 2 gap |x|^2 is measured at
    // the production and the converged basis, the same comparison R9a makes for F1's N_dia.
    let mesh = MeshIndex::new([4, 4, 4], [0.; 3]).unwrap();
    for size in ["DZP", "QZ2P"] {
        let solved = solve_on(
            &silicon(),
            "raw",
            size,
            mandacaru_core::Context::new(Some(2), 3 << 29).unwrap(),
        );
        let weight = 1. / mesh.points() as f64;
        let sum: f64 = (0..mesh.points())
            .map(|i| {
                transfer_weights(&solved, mesh.point(i), 0, 1e-3)
                    .iter()
                    .map(|(gap, x2)| 2. * weight * 2. * gap * x2)
                    .sum::<f64>()
            })
            .sum();
        eprintln!("Si {size} 4^3: length-form N_len {sum:.6}");
    }
}
