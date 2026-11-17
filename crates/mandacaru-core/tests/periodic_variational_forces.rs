//! Stage 6a of the Bloch methods (`BLOCH_VQE_PLAN.md` 6.6): the analytic force of a variational
//! state on the PAW-LCAO supercell, oracle first.
//!
//! The oracle of gate G1 is the central difference, under a collective displacement of one
//! primitive atom (every image of it moves with it, the only motion a supercell of the crystal
//! has), of the energy at fixed raw-AO densities: `E(P; R) = tr(P h(R)) + 1/2 P P (ab|cd)(R) -
//! 1/4 P P (ac|bd)(R) + E_nuc(R)` with the Hamiltonian of the displaced crystal setup. Its
//! closure: the relaxed RHF force equals it less the overlap term,
//! `dE_RHF/dR = dE/dR|_P - tr(W dS/dR)`, `W = P F P / 2` the energy-weighted density (raw AOs).
//! Needs `MANDACARU_PAW_PATH`.
mod paw_supercell_support;
mod setup_shells_support;
mod variational_forces_support;

use mandacaru_core::{
    molecular::{
        periodic::paw::{
            metric_gradient, one_body_gradient, two_body_energies, two_body_gradient, PawSupercell,
        },
        ScfOptions,
    },
    periodic::{
        setup::PeriodicSetup, solver::bare_hamiltonian,
        wannier::downfold::basis::hartree_fock_onsite,
    },
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use paw_supercell_support::{
    cells_of, closed_shell_energy, displaced, hydrogen_chain_cell, lithium_hydride_cell,
    nitrogen_chain_cell, raw_atomic_orbitals,
};

/// The setup with its primitive atom `atom` (and every image) moved by `shift` (Bohr).
fn moved(ctx: &Context, setup: &PeriodicSetup, atom: usize, shift: [f64; 3]) -> PeriodicSetup {
    let mut fractional = setup.fractional().unwrap();
    let step = setup.lattice.try_inverse().unwrap() * nalgebra::Vector3::from(shift);
    for a in 0..3 {
        fractional[atom][a] += step[a];
    }
    setup.with_positions(ctx, &fractional).unwrap()
}

/// A symmetric supercell matrix that is not positive (a relaxed density need not be).
fn symmetric(size: usize, seed: f64, diagonal: f64) -> DMatrix<f64> {
    DMatrix::from_fn(size, size, |a, b| {
        let (i, j) = (a.min(b) as f64, a.max(b) as f64);
        0.3 * (seed * (7. * i + 3. * j + 1.)).sin() + if a == b { diagonal } else { 0. }
    })
}

/// Stage 6a sub-packages (i) and (ii): the one-body explicit derivative of the PAW-LCAO
/// supercell at a fixed raw-AO density `P` and energy-weighted density `W` (both symmetric, not
/// positive) is the crystal's force engines at the mesh points (`one_body_gradient`), on s and p
/// functions (the N2 chain: the complex harmonics `U` count). The
/// convention first: `sum_k (1/N) Re tr[D(k) h(k)]` with the mesh crystal's own bare operator and
/// one-center operator equals `tr(P h)` of the folded supercell, and the same for `S`, on a
/// (3,1,1) mesh (points off the time-reversal-invariant ones, so the phase and the side of `U`
/// count). Then, per term group, against central differences of the folded supercell's own
/// matrices under a collective displacement (Richardson of 4, 2, 1 mBohr): `tr(P h) + E_ions`
/// against the `D` pass, `tr(W S)` against the `W` pass less the ions' pass (zero densities), the
/// ions' `E_ions` against that pass.
fn one_body_sweep(
    cases: Vec<(&str, paw_supercell_support::Primitive, [usize; 3])>,
    components: Option<&[(usize, usize)]>,
    steps: &[f64],
    tolerance: f64,
) {
    let ctx = Context::new(Some(4), 1 << 30).unwrap();
    for (name, cell, mesh) in cases {
        let setup = cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]);
        let base = PawSupercell::new(&ctx, &setup, mesh).unwrap();
        let count = base.n_cells();
        let size = base.overlap.nrows();
        let rotation = (&base.harmonics - DMatrix::<C64>::identity(base.m, base.m)).norm();
        println!(
            "{name}: {} functions per cell, |U - 1| = {rotation:.3}",
            base.m
        );
        if name == "N2 chain" {
            assert!(
                rotation > 0.5,
                "the p fixture has no complex harmonics to rotate"
            );
        }
        let p = symmetric(size, 0.37, 0.5);
        let w = symmetric(size, 0.61, -0.2);
        let zero = DMatrix::<f64>::zeros(size, size);
        // The convention, against the mesh crystal's own operators.
        let crystal = setup
            .with_kpoints(&ctx, &base.kpoints, &vec![1. / count as f64; count])
            .unwrap()
            .build(&ctx)
            .unwrap();
        let onsite = hartree_fock_onsite(&crystal).unwrap();
        let (folded_p, folded_w) = (
            base.fold_density(&p).unwrap(),
            base.fold_density(&w).unwrap(),
        );
        let (mut one, mut overlap) = (0f64, 0f64);
        for (k, point) in crystal.points.iter().enumerate() {
            let h = bare_hamiltonian(&crystal, point).unwrap()
                + &point.projections * &onsite * point.projections.adjoint();
            let trace = |a: &DMatrix<C64>, b: &DMatrix<C64>| (a * b).trace().re / count as f64;
            one += trace(&folded_p[k], &h);
            overlap += trace(&folded_w[k], &point.overlap);
        }
        let (one_gap, overlap_gap) = (
            (one - p.dot(&base.one_body)).abs(),
            (overlap - w.dot(&base.overlap)).abs(),
        );
        println!("{name} {mesh:?}: convention tr(P h) {:.12} gap {one_gap:.1e}, tr(W S) {:.12} gap {overlap_gap:.1e}", p.dot(&base.one_body), w.dot(&base.overlap));
        assert!(
            one_gap < 1e-11 && overlap_gap < 1e-11,
            "{name}: {one_gap:e} {overlap_gap:e}"
        );
        // The derivative, term group by term group.
        let one_body = one_body_gradient(&ctx, &setup, mesh, &p, &zero).unwrap();
        let weighted = one_body_gradient(&ctx, &setup, mesh, &zero, &w).unwrap();
        // The ions' terms do not depend on the densities: the pass at zero densities is them alone.
        let ions = one_body_gradient(&ctx, &setup, mesh, &zero, &zero)
            .unwrap()
            .gradient();
        let (dh, ds) = (one_body.gradient(), weighted.gradient());
        let mut worst = 0f64;
        let atoms = setup.centers.len();
        let every: Vec<(usize, usize)> = (0..atoms)
            .flat_map(|a| (0..3).map(move |x| (a, x)))
            .collect();
        for &(atom, axis) in components.unwrap_or(&every) {
            {
                let (mut e1, mut es, mut ei) = (Vec::new(), Vec::new(), Vec::new());
                for &step in steps {
                    let mut sides = Vec::new();
                    for sign in [1., -1.] {
                        let mut shift = [0.; 3];
                        shift[axis] = sign * step;
                        let cells =
                            PawSupercell::new(&ctx, &moved(&ctx, &setup, atom, shift), mesh)
                                .unwrap();
                        sides.push((
                            p.dot(&cells.one_body) + cells.ions,
                            w.dot(&cells.overlap),
                            cells.ions,
                        ));
                    }
                    e1.push((sides[0].0 - sides[1].0) / (2. * step));
                    es.push((sides[0].1 - sides[1].1) / (2. * step));
                    ei.push((sides[0].2 - sides[1].2) / (2. * step));
                }
                // Richardson levels (the step error is even in h): the last of the deepest level.
                let richardson = |v: &[f64]| {
                    let mut level = v.to_vec();
                    let mut factor = 4.;
                    while level.len() > 1 {
                        level = level
                            .windows(2)
                            .map(|w| (factor * w[1] - w[0]) / (factor - 1.))
                            .collect();
                        factor *= 4.;
                    }
                    level[0]
                };
                let (fd1, fds, fdi) = (richardson(&e1), richardson(&es), richardson(&ei));
                let overlap_term = ds[atom][axis] - ions[atom][axis];
                let (miss1, misss, missi) = (
                    (dh[atom][axis] - fd1).abs(),
                    (overlap_term + fds).abs(),
                    (ions[atom][axis] - fdi).abs(),
                );
                println!(
                    "{name} atom {atom} axis {axis}: d[tr(P h) + E_ions] analytic {:+.12} oracle {fd1:+.12} miss {miss1:.1e} (first two central differences differ {:.1e}); -d tr(W S) analytic {overlap_term:+.12} oracle {:+.12} miss {misss:.1e}; d E_ions analytic {:+.12} oracle {fdi:+.12} miss {missi:.1e}",
                    dh[atom][axis], (e1[0] - e1[1]).abs(), -fds, ions[atom][axis]
                );
                worst = worst.max(miss1).max(misss).max(missi);
            }
        }
        for (term, values) in one_body.terms() {
            let largest = values.iter().flatten().fold(0f64, |a, v| a.max(v.abs()));
            if largest > 0. {
                println!("{name} term {term}: largest {largest:.3e}");
            }
        }
        println!("{name}: worst miss {worst:.2e} Ha/Bohr");
        assert!(worst < tolerance, "{name}: {worst:e}");
    }
}

/// The fast case of [`one_body_sweep`]: the convention on the three cells, one component of the
/// H2 chain (3,1,1) at 4, 2, 1 mBohr.
#[test]
fn the_one_body_derivative_of_a_paw_supercell_is_the_crystal_engines_at_the_mesh() {
    one_body_sweep(
        vec![("H2 chain", hydrogen_chain_cell(), [3, 1, 1])],
        Some(&[(0, 0)]),
        &[4e-3, 2e-3, 1e-3],
        1e-9,
    );
    one_body_sweep(
        vec![
            ("LiH", lithium_hydride_cell(), [2, 1, 1]),
            ("N2 chain", nitrogen_chain_cell(), [3, 1, 1]),
        ],
        Some(&[]),
        &[4e-3, 2e-3, 1e-3],
        1e-9,
    );
}

/// The full sweep of [`one_body_sweep`]: every atom and axis of the H2 chain (3,1,1), LiH (2,1,1)
/// and the N2 chain (3,1,1), central differences at 8, 4, 2, 1 mBohr and three Richardson levels.
#[test]
#[ignore = "slow: about 3 min (measured 2026-11-15)"]
fn the_one_body_derivative_of_a_paw_supercell_is_the_crystal_engines_on_every_component() {
    one_body_sweep(
        vec![
            ("H2 chain", hydrogen_chain_cell(), [3, 1, 1]),
            ("LiH", lithium_hydride_cell(), [2, 1, 1]),
            ("N2 chain", nitrogen_chain_cell(), [3, 1, 1]),
        ],
        None,
        &[8e-3, 4e-3, 2e-3, 1e-3],
        1e-9,
    );
}

/// G1's oracle on the H2 chain (2, 1, 1): the fixed-density central differences at 8, 4, 2 and
/// 1 mBohr, their Richardson levels (the step error is `h^2`), and the closure against the
/// relaxed RHF central difference through the overlap term.
#[test]
fn the_fixed_density_oracle_of_a_paw_supercell_converges_and_closes_the_rhf_force() {
    let ctx = Context::new(Some(4), 1 << 30).unwrap();
    let mesh = [2, 1, 1];
    let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
    let base = displaced(&ctx, &setup, mesh, 0, [0.; 3]);
    let n = base.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let rhf = base.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    let raw = &base.lowdin * rhf.coefficients[0].map(|v| v.re);
    let occupied = raw.columns(0, n / 2);
    let p: DMatrix<f64> = &occupied * occupied.transpose() * 2.;
    let (_, h, eri) = raw_atomic_orbitals(&base);
    let m = h.nrows();
    let fock = DMatrix::from_fn(m, m, |a, b| {
        h[(a, b)]
            + (0..m * m)
                .map(|cd| {
                    let (c, d) = (cd / m, cd % m);
                    p[(c, d)]
                        * (eri[(a * m + b) * m * m + cd]
                            - 0.5 * eri[(a * m + c) * m * m + b * m + d])
                })
                .sum::<f64>()
    });
    let w = &p * fock * &p * 0.5;
    let steps = [8e-3, 4e-3, 2e-3, 1e-3];
    let mut net = [0f64; 3];
    for (atom, axis) in [(0, 0), (0, 1), (1, 0)] {
        let (mut fixed, mut relaxed, mut overlap) = (Vec::new(), Vec::new(), Vec::new());
        for &step in &steps {
            let mut sides = Vec::new();
            for sign in [1., -1.] {
                let mut shift = [0.; 3];
                shift[axis] = sign * step;
                let moved = displaced(&ctx, &setup, mesh, atom, shift);
                let (s, h, eri) = raw_atomic_orbitals(&moved);
                let state = moved.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
                assert!(state.converged);
                sides.push((
                    closed_shell_energy(&h, &eri, moved.nuclear, &p),
                    state.electronic_energy + moved.nuclear,
                    w.dot(&s),
                ));
            }
            let central =
                |f: fn(&(f64, f64, f64)) -> f64| (f(&sides[0]) - f(&sides[1])) / (2. * step);
            fixed.push(central(|v| v.0));
            relaxed.push(central(|v| v.1));
            overlap.push(central(|v| v.2));
        }
        let richardson =
            |v: &[f64]| -> Vec<f64> { v.windows(2).map(|w| (4. * w[1] - w[0]) / 3.).collect() };
        let (fixed1, relaxed1, overlap1) = (
            richardson(&fixed),
            richardson(&relaxed),
            richardson(&overlap),
        );
        let fixed2 = richardson(&fixed1);
        let last = fixed1.len() - 1;
        let resolution = (fixed1[last] - fixed1[last - 1]).abs();
        let closure = (relaxed1[last] - (fixed1[last] - overlap1[last])).abs();
        println!(
            "atom {atom} axis {axis}: fixed density {fixed:.10?}, Richardson {fixed1:.10?}, second {fixed2:.10?}; relaxed RHF {relaxed1:.10?}; overlap term {overlap1:.10?}; resolution {resolution:.2e}, closure {closure:.2e} Ha/Bohr"
        );
        if axis == 0 {
            net[0] += fixed1[last];
        }
        assert!(resolution < 1e-8, "atom {atom} axis {axis}: {resolution:e}");
        assert!(closure < 1e-7, "atom {atom} axis {axis}: {closure:e}");
    }
    println!(
        "net fixed-density derivative along x (both atoms): {:.3e} Ha/Bohr",
        net[0]
    );
}

/// Richardson levels of central differences at halving steps (the step error is even in h): the
/// last value of the deepest level.
fn richardson(values: &[f64]) -> f64 {
    let mut level = values.to_vec();
    let mut factor = 4.;
    while level.len() > 1 {
        level = level
            .windows(2)
            .map(|w| (factor * w[1] - w[0]) / (factor - 1.))
            .collect();
        factor *= 4.;
    }
    level[0]
}

/// Stage 6a (iii), first slice: the derivative functions of the PAW-LCAO supercell (the Bloch
/// slopes of the moving atom's functions, the projections' `P_X G_C - G_C Q_X`) as extra modes of
/// the pair engine carry the overlap's own derivative: `M[d a, b] + M[a, d b]` of the pair
/// densities' metric against central differences of the folded overlap under a collective
/// displacement, element by element, on s functions (the H2 chain (3,1,1)) and p functions (the
/// N2 chain (2,1,1), complex harmonics); the `G = 0` charge block (the compensation charges'
/// monopoles through `B_c`) against the same; the sampled Bloch sums against the crystal's.
#[test]
fn the_derivative_functions_of_a_paw_supercell_carry_the_overlap_derivative() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let steps = [4e-3, 2e-3, 1e-3];
    for (name, cell, mesh) in [
        ("H2 chain", hydrogen_chain_cell(), [3, 1, 1]),
        ("N2 chain", nitrogen_chain_cell(), [2, 1, 1]),
    ] {
        let setup = cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]);
        let base = PawSupercell::new(&ctx, &setup, mesh).unwrap();
        let size = base.overlap.nrows();
        let frame = DMatrix::<f64>::identity(size, size);
        let mut worst = (0f64, 0f64, 0f64);
        for atom in 0..setup.centers.len() {
            let (metric, charge, samples) =
                metric_gradient(&ctx, &setup, mesh, &frame, atom).unwrap();
            worst.2 = worst.2.max(samples);
            for axis in 0..3 {
                let mut differences = Vec::new();
                for &step in &steps {
                    let mut shift = [0.; 3];
                    shift[axis] = step;
                    let plus = PawSupercell::new(&ctx, &moved(&ctx, &setup, atom, shift), mesh)
                        .unwrap()
                        .overlap;
                    shift[axis] = -step;
                    let minus = PawSupercell::new(&ctx, &moved(&ctx, &setup, atom, shift), mesh)
                        .unwrap()
                        .overlap;
                    differences.push((plus - minus) / (2. * step));
                }
                let oracle = DMatrix::from_fn(size, size, |a, b| {
                    richardson(&differences.iter().map(|d| d[(a, b)]).collect::<Vec<_>>())
                });
                let (mut miss, mut charge_miss, mut imaginary) = (0f64, 0f64, 0f64);
                for a in 0..size {
                    for b in 0..size {
                        miss = miss.max((metric[axis][(a, b)].re - oracle[(a, b)]).abs());
                        charge_miss =
                            charge_miss.max((charge[axis][(a, b)].re - oracle[(a, b)]).abs());
                        imaginary = imaginary.max(metric[axis][(a, b)].im.abs());
                    }
                }
                println!(
                    "{name} {mesh:?} atom {atom} axis {axis}: |dS| largest {:.3e}, metric miss {miss:.2e}, charge miss {charge_miss:.2e}, imaginary {imaginary:.1e} (first two differences {:.1e} apart)",
                    oracle.amax(),
                    (&differences[0] - &differences[1]).amax()
                );
                worst.0 = worst.0.max(miss);
                worst.1 = worst.1.max(charge_miss);
            }
        }
        println!(
            "{name}: worst metric miss {:.2e}, charge miss {:.2e}, sampled Bloch sums gap {:.1e}",
            worst.0, worst.1, worst.2
        );
        assert!(
            worst.2 < 1e-12,
            "{name}: the sampled Bloch sums are not the crystal's: {:e}",
            worst.2
        );
        assert!(worst.0 < 1e-9, "{name}: {:e}", worst.0);
    }
}

/// The Coulomb (J) and exchange (K) parts of the closed-shell two-body energy of the raw-AO
/// density `P = 2 C C^T` (`C` the occupied orbitals, raw AOs) as Coulomb forms over `C`:
/// J `(1, 2 I)`, K `(-2, e_i e_i^T)` and `(-2, (e_i e_j^T + e_j e_i^T) / sqrt 2)`.
fn closed_shell_forms(r: usize) -> (Vec<(f64, DMatrix<f64>)>, Vec<(f64, DMatrix<f64>)>) {
    let coulomb = vec![(1., DMatrix::<f64>::identity(r, r) * 2.)];
    let mut exchange = Vec::new();
    for i in 0..r {
        for j in 0..=i {
            let mut x = DMatrix::<f64>::zeros(r, r);
            if i == j {
                x[(i, i)] = 1.;
            } else {
                x[(i, j)] = std::f64::consts::FRAC_1_SQRT_2;
                x[(j, i)] = std::f64::consts::FRAC_1_SQRT_2;
            }
            exchange.push((-2., x));
        }
    }
    (coulomb, exchange)
}

/// `(E_J, E_K)` of the raw-AO density `p` on the raw-AO integrals `eri` (chemists' order).
fn coulomb_exchange(eri: &[f64], p: &DMatrix<f64>) -> (f64, f64) {
    let m = p.nrows();
    let (mut j, mut k) = (0., 0.);
    for ab in 0..m * m {
        for cd in 0..m * m {
            let (a, b, c, d) = (ab / m, ab % m, cd / m, cd % m);
            j += 0.5 * eri[ab * m * m + cd] * p[(a, b)] * p[(c, d)];
            k -= 0.25 * eri[ab * m * m + cd] * p[(a, c)] * p[(b, d)];
        }
    }
    (j, k)
}

/// The RHF state of the supercell of `mesh` on `setup`: its Hamiltonian, the occupied orbitals
/// in raw AOs, the raw-AO density `P = 2 C C^T`, the raw-AO one-body operator and integrals.
#[allow(clippy::type_complexity)]
fn rhf_state(
    ctx: &Context,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    cholesky: Option<f64>,
) -> (
    mandacaru_core::molecular::MolecularIntegrals,
    DMatrix<f64>,
    DMatrix<f64>,
    DMatrix<f64>,
    Vec<f64>,
) {
    let copy = setup
        .with_positions(ctx, &setup.fractional().unwrap())
        .unwrap();
    let base = mandacaru_core::molecular::MolecularIntegrals::paw_supercell(
        ctx,
        std::sync::Arc::new(copy),
        mesh,
        cholesky,
    )
    .unwrap();
    let n = base.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let rhf = base.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    assert!(rhf.converged);
    let raw = &base.lowdin * rhf.coefficients[0].map(|v| v.re);
    let occupied: DMatrix<f64> = raw.columns(0, n / 2).into_owned();
    let p: DMatrix<f64> = &occupied * occupied.transpose() * 2.;
    let (_, h, eri) = raw_atomic_orbitals(&base);
    (base, occupied, p, h, eri)
}

/// Stage 6a (iii-a, iii-b) on one supercell: at fixed raw-AO coefficients (the RHF occupied
/// orbitals), J and K apart, the forms' energies equal the supercell Hamiltonian's own two-body
/// energy (its Cholesky-factored integrals in raw AOs), and the derivative (the functions' and the
/// shapes' motion) equals the Richardson limit of central differences of the forms' energies under
/// a collective displacement (`steps`, Bohr) on `components`. Returns the worst miss, the energy
/// identity's gap and the J and K totals.
#[allow(clippy::type_complexity)]
fn two_body_sweep(
    ctx: &Context,
    name: &str,
    setup: &PeriodicSetup,
    mesh: [usize; 3],
    occupied: &DMatrix<f64>,
    p: &DMatrix<f64>,
    eri: Option<&[f64]>,
    components: &[(usize, usize)],
    steps: &[f64],
) -> (f64, f64, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let r = occupied.ncols();
    let (coulomb, exchange) = closed_shell_forms(r);
    let all: Vec<(f64, DMatrix<f64>)> = coulomb.iter().chain(&exchange).cloned().collect();
    let split = |energies: Vec<f64>| (energies[0], energies[1..].iter().sum::<f64>());
    let (ej, ek) = eri.map_or((f64::NAN, f64::NAN), |eri| coulomb_exchange(eri, p));
    let (fj, fk) = split(two_body_energies(ctx, setup, mesh, occupied, &all).unwrap());
    let identity = if eri.is_some() {
        (ej - fj).abs().max((ek - fk).abs())
    } else {
        0.
    };
    println!("{name} {mesh:?}: {r} occupied; E_J {ej:.12} forms {fj:.12} gap {:.1e}; E_K {ek:.12} forms {fk:.12} gap {:.1e}", (ej - fj).abs(), (ek - fk).abs());
    let gj = two_body_gradient(ctx, setup, mesh, occupied, &coulomb).unwrap();
    let gk = two_body_gradient(ctx, setup, mesh, occupied, &exchange).unwrap();
    println!(
        "{name}: sampled Bloch sums gap {:.1e}, imaginary parts J {:.1e} K {:.1e}",
        gj.samples.max(gk.samples),
        gj.imaginary,
        gk.imaginary
    );
    let (tj, tk) = (gj.total(), gk.total());
    let mut worst = 0f64;
    for &(atom, axis) in components {
        let (mut dj, mut dk) = (Vec::new(), Vec::new());
        for &step in steps {
            let mut sides = Vec::new();
            for sign in [1., -1.] {
                let mut shift = [0.; 3];
                shift[axis] = sign * step;
                sides.push(split(
                    two_body_energies(ctx, &moved(ctx, setup, atom, shift), mesh, occupied, &all)
                        .unwrap(),
                ));
            }
            dj.push((sides[0].0 - sides[1].0) / (2. * step));
            dk.push((sides[0].1 - sides[1].1) / (2. * step));
        }
        let (oj, ok) = (richardson(&dj), richardson(&dk));
        let (mj, mk) = ((tj[atom][axis] - oj).abs(), (tk[atom][axis] - ok).abs());
        // The oracle's own resolution: the two ladders one level shorter (the largest and the
        // smallest steps dropped in turn) differ by about the deepest level's error.
        let spread = |d: &[f64]| (richardson(&d[..d.len() - 1]) - richardson(&d[1..])).abs();
        println!(
            "{name} atom {atom} axis {axis}: J analytic {:+.12} (functions {:+.12}, shapes {:+.12}) oracle {oj:+.12} miss {mj:.2e} spread {:.2e}; K analytic {:+.12} (functions {:+.12}, shapes {:+.12}) oracle {ok:+.12} miss {mk:.2e} spread {:.2e}",
            tj[atom][axis], gj.functions[atom][axis], gj.shapes[atom][axis], spread(&dj), tk[atom][axis], gk.functions[atom][axis], gk.shapes[atom][axis], spread(&dk)
        );
        worst = worst.max(mj).max(mk);
    }
    println!("{name}: worst two-body miss {worst:.2e} Ha/Bohr; forms' energy against the Hamiltonian's {identity:.1e} Ha");
    (worst, identity, tj, tk)
}

/// Stage 6a (iii) on s functions and G1: the two-body sweep of the H2 chain (2,1,1) on every atom
/// and axis, then the one-body and two-body layers together against the fixed-density oracle of
/// the supercell Hamiltonian, and the relaxed RHF force (the overlap term of `W = P F P / 2`
/// subtracted) against the displaced SCF.
#[test]
fn the_two_body_derivative_of_a_paw_supercell_closes_g1() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let mesh = [2, 1, 1];
    let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
    let (base, occupied, p, h, eri) = rhf_state(&ctx, &setup, mesh, Some(1e-12));
    let n = base.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let every: Vec<(usize, usize)> = (0..2).flat_map(|a| (0..3).map(move |x| (a, x))).collect();
    let (worst, identity, tj, tk) = two_body_sweep(
        &ctx,
        "H2 chain",
        &setup,
        mesh,
        &occupied,
        &p,
        Some(&eri),
        &every,
        &[4e-3, 2e-3, 1e-3],
    );
    assert!(identity < 1e-9, "{identity:e}");
    // G1: the one-body layer at (P, W = 0) and the two-body layer against the fixed-density
    // oracle of the supercell Hamiltonian (its Cholesky-factored integrals, Richardson of 8, 4,
    // 2, 1 mBohr), and the relaxed RHF force through the overlap term.
    let m = h.nrows();
    let fock = DMatrix::from_fn(m, m, |a, b| {
        h[(a, b)]
            + (0..m * m)
                .map(|cd| {
                    let (c, d) = (cd / m, cd % m);
                    p[(c, d)]
                        * (eri[(a * m + b) * m * m + cd]
                            - 0.5 * eri[(a * m + c) * m * m + b * m + d])
                })
                .sum::<f64>()
    });
    let w = &p * fock * &p * 0.5;
    let zero = DMatrix::<f64>::zeros(m, m);
    let one = one_body_gradient(&ctx, &setup, mesh, &p, &zero)
        .unwrap()
        .gradient();
    let overlap = one_body_gradient(&ctx, &setup, mesh, &zero, &w)
        .unwrap()
        .gradient();
    let ions = one_body_gradient(&ctx, &setup, mesh, &zero, &zero)
        .unwrap()
        .gradient();
    let mut g1 = 0f64;
    let mut net = 0f64;
    for (atom, axis) in [(0, 0), (0, 1), (1, 0)] {
        let (mut fixed, mut relaxed) = (Vec::new(), Vec::new());
        for &step in &[8e-3, 4e-3, 2e-3, 1e-3] {
            let mut sides = Vec::new();
            for sign in [1., -1.] {
                let mut shift = [0.; 3];
                shift[axis] = sign * step;
                let moved = displaced(&ctx, &setup, mesh, atom, shift);
                let (_, h, eri) = raw_atomic_orbitals(&moved);
                let state = moved.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
                assert!(state.converged);
                sides.push((
                    closed_shell_energy(&h, &eri, moved.nuclear, &p),
                    state.electronic_energy + moved.nuclear,
                ));
            }
            fixed.push((sides[0].0 - sides[1].0) / (2. * step));
            relaxed.push((sides[0].1 - sides[1].1) / (2. * step));
        }
        let (fixed, relaxed) = (richardson(&fixed), richardson(&relaxed));
        let analytic = one[atom][axis] + tj[atom][axis] + tk[atom][axis];
        let force = analytic + (overlap[atom][axis] - ions[atom][axis]);
        let (miss, relaxed_miss) = ((analytic - fixed).abs(), (force - relaxed).abs());
        println!("G1 atom {atom} axis {axis}: fixed density analytic {analytic:+.12} oracle {fixed:+.12} miss {miss:.2e}; relaxed RHF analytic {force:+.12} oracle {relaxed:+.12} miss {relaxed_miss:.2e} Ha/Bohr");
        g1 = g1.max(miss);
        if axis == 0 {
            net += force;
        }
    }
    println!("G1 worst miss {g1:.2e} Ha/Bohr; analytic net RHF force along x {net:.2e}");
    assert!(worst < 1e-9, "two-body: {worst:e}");
    assert!(g1 < 1e-9, "G1: {g1:e}");
}

/// Stage 6a (iii) on p functions: the two-body sweep of the N2 chain (3,1,1) (8 functions per
/// cell: the complex harmonics of the projectors and the `L > 0` compensation shapes count), at
/// fixed raw-AO coefficients, every atom and axis. The frame is the first 15 raw AOs (the
/// occupied count), not RHF orbitals: the derivative at fixed coefficients needs no SCF (its
/// oracle is the forms' energy itself), so the energy identity against the Hamiltonian is not
/// checked here (H2 carries it). Gate 1e-8 Ha/Bohr, the oracle's resolution on this cell
/// (MG-REST item 7, 2026-11-15): on the 8, 4, 2, 1 mBohr ladder the misses are 7.1e-10 to 5.4e-9
/// (J) and 1.3e-10 to 9.6e-10 (K) on derivatives up to 15.4 Ha/Bohr, while the two ladders one
/// level shorter differ by 4.5e-10 to 3.7e-8 (J) and 1.2e-10 to 7.4e-9 (K); the 4, 2, 1 ladder
/// alone missed 5.7e-9.
#[test]
#[ignore = "slow: about 6 min (331 s measured 2026-11-15: 48 displaced two-body energies of the N2 (3,1,1) chain); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_variational_forces -- --ignored p_functions --nocapture"]
fn the_two_body_derivative_of_a_paw_supercell_holds_on_p_functions() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let mesh = [3, 1, 1];
    let setup = cells_of(&ctx, "lda-sr", false, &nitrogen_chain_cell(), 1, [0.; 3]);
    let size = 3 * PawSupercell::new(&ctx, &setup, mesh).unwrap().m;
    let occupied: DMatrix<f64> = DMatrix::<f64>::identity(size, size)
        .columns(0, 15)
        .into_owned();
    let p: DMatrix<f64> = &occupied * occupied.transpose() * 2.;
    let every: Vec<(usize, usize)> = (0..2).flat_map(|a| (0..3).map(move |x| (a, x))).collect();
    let (worst, identity, _, _) = two_body_sweep(
        &ctx,
        "N2 chain",
        &setup,
        mesh,
        &occupied,
        &p,
        None,
        &every,
        &[8e-3, 4e-3, 2e-3, 1e-3],
    );
    assert!(worst < 1e-8 && identity == 0., "{worst:e}");
}

/// `sum gamma h + 1/2 sum Gamma <pq|rs> + E_nn` of the raw-AO orbitals `a` on `integrals`
/// (spin-summed RDMs over the columns of `a`).
fn energy_at_fixed_orbitals(
    integrals: &mandacaru_core::molecular::MolecularIntegrals,
    a: &DMatrix<f64>,
    one: &[f64],
    two: &[f64],
) -> f64 {
    use mandacaru_core::molecular::transform;
    let ctx = &integrals.context;
    let m = integrals.m();
    let t = a.ncols();
    let c = (integrals.lowdin.clone().try_inverse().unwrap() * a).map(|v| C64::new(v, 0.));
    let h = DMatrix::from_row_slice(m, m, integrals.one.as_slice()).map(|v| C64::new(v, 0.));
    let h = c.adjoint() * h * &c;
    let eri = integrals.eri().unwrap();
    let g = transform::four(ctx, &eri, [&c, &c, &c, &c]).unwrap();
    let mut energy = integrals.nuclear;
    for p in 0..t {
        for q in 0..t {
            energy += one[p * t + q] * h[(p, q)].re;
        }
    }
    for (d, v) in two.iter().zip(g.as_slice()) {
        energy += 0.5 * d * v.re;
    }
    energy
}

/// G2 through the seam (`MolecularIntegrals::fixed_density_gradient` of a PAW-LCAO supercell):
/// the ADAPT-VQE state of the H2 chain (2,1,1) (8 qubits, full space) at fixed raw-AO orbitals,
/// its spin-summed RDMs as `FixedDensities` (the Coulomb forms of `two_body_forms`), against the
/// central differences of the energy at fixed orbitals on the collectively displaced supercell
/// with fixed Cholesky pivots (Richardson of 4, 2, 1 mBohr). The seam returns each image's share
/// (`1/N` of its primitive atom's derivative), so the images summed are the collective derivative.
#[test]
fn the_fixed_density_gradient_of_an_adapt_state_on_a_paw_supercell_is_the_derivative_at_fixed_orbitals(
) {
    use mandacaru_core::{
        molecular::variational_gradient::{two_body_forms, FixedDensities},
        quantum::{properties, SolveOptions},
        Geometry, LengthUnit,
    };
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let mesh = [2, 1, 1];
    let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
    let base = displaced(&ctx, &setup, mesh, 0, [0.; 3]);
    let n = base.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let state = base.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    let model = state.model(&base).unwrap();
    let problem = model
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap();
    let solved = problem
        .solve(
            &SolveOptions {
                adapt: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let rdm = properties::spatial_rdms(
        &ctx,
        &problem.encoding,
        problem.register("test").unwrap(),
        solved.state.complex128().unwrap(),
    )
    .unwrap();
    let t = rdm.m;
    let one: Vec<f64> = rdm.one.as_slice().iter().map(|v| v.re).collect();
    let two: Vec<f64> = rdm.two.as_slice().iter().map(|v| v.re).collect();
    let a = &base.lowdin * state.model_orbitals.map(|v| v.re);
    assert_eq!(a.ncols(), t);
    let m = base.m();
    let densities = FixedDensities {
        one: &a * DMatrix::from_row_slice(t, t, &one) * a.transpose(),
        weighted: DMatrix::zeros(m, m),
        frame: a.clone(),
        two: two_body_forms(&two, t).unwrap(),
        cluster_frame: None,
    };
    let at_base = energy_at_fixed_orbitals(&base, &a, &one, &two);
    println!("ADAPT on the H2 chain (2,1,1): energy {:.12}, at fixed orbitals {at_base:.12} Ha; {} forms", solved.energy, densities.two.len());
    assert!((at_base - solved.energy).abs() < 1e-9);
    let gradient = base.fixed_density_gradient(&densities).unwrap();
    for (term, values) in &gradient.terms {
        println!(
            "term {term}: {:?}",
            values
                .as_slice()
                .iter()
                .map(|v| format!("{v:+.6e}"))
                .collect::<Vec<_>>()
        );
    }
    let atoms = setup.centers.len();
    let cells = 2;
    let mut worst = 0f64;
    for atom in 0..atoms {
        for axis in 0..3 {
            let analytic: f64 = (0..cells)
                .map(|cell| gradient.total.as_slice()[(cell * atoms + atom) * 3 + axis])
                .sum();
            let mut differences = Vec::new();
            for &step in &[4e-3, 2e-3, 1e-3] {
                let mut sides = Vec::new();
                for sign in [1., -1.] {
                    let mut positions = base.geometry.positions().to_vec();
                    for cell in 0..cells {
                        positions[cell * atoms + atom][axis] += sign * step;
                    }
                    let geometry = Geometry::new(
                        base.geometry.numbers().to_vec(),
                        positions,
                        *base.geometry.cell(),
                        base.geometry.pbc(),
                        LengthUnit::Bohr,
                    )
                    .unwrap();
                    let moved = base.at_geometry_with_fixed_pivots(geometry).unwrap();
                    sides.push(energy_at_fixed_orbitals(&moved, &a, &one, &two));
                }
                differences.push((sides[0] - sides[1]) / (2. * step));
            }
            let oracle = richardson(&differences);
            let miss = (analytic - oracle).abs();
            println!("G2 atom {atom} axis {axis}: analytic {analytic:+.12} oracle {oracle:+.12} miss {miss:.2e} Ha/Bohr");
            worst = worst.max(miss);
        }
    }
    println!("G2 worst miss {worst:.2e} Ha/Bohr");
    assert!(worst < 1e-8, "{worst:e}");
}

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;

/// G3 on one supercell: the full analytic force of an ADAPT-VQE state (its parameters moved off
/// the optimum by `shift`, so the Z-vector correction carries weight) on the PAW-LCAO supercell
/// (`response::gradient`, the derivative of the energy per cell per primitive atom) against the
/// finite-difference oracle along the collective image directions (displaced SCF continued from the
/// reference, the model's selection redone, frames aligned, fixed Cholesky pivots, Richardson of
/// 1 and 0.5 mBohr). Returns the worst miss in eV/A and the analytic net force (Ha/Bohr).
fn g3_case(
    name: &str,
    cell: paw_supercell_support::Primitive,
    mesh: [usize; 3],
    kind: &'static str,
    spin: i64,
    active: Option<&str>,
    shift: f64,
) -> (f64, f64) {
    use mandacaru_core::{
        molecular::response,
        quantum::{lagrangian::Ansatz, SolveOptions},
    };
    use variational_forces_support::oracle;
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let setup = cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]);
    let base = displaced(&ctx, &setup, mesh, 0, [0.; 3]);
    let n = base.neutral_electrons() as i64;
    let particles = [((n + spin) / 2) as usize, ((n - spin) / 2) as usize];
    let state = base
        .mean_field(
            kind,
            particles,
            &ScfOptions {
                tolerance: 1e-12,
                ..ScfOptions::default()
            },
        )
        .unwrap();
    println!(
        "{name} {mesh:?}: {n} electrons, {} orbitals, levels {:.5?}",
        base.m(),
        state
            .energies
            .iter()
            .map(|e| e[..e.len().min(8)].to_vec())
            .collect::<Vec<_>>()
    );
    assert!(state.converged);
    let model = match active {
        Some(spec) => std::sync::Arc::new(state.active_model(&base, spec).unwrap()),
        None => state.model(&base).unwrap(),
    };
    let problem = model
        .problem(&ctx, "jordan_wigner", true, false, true)
        .unwrap();
    let solved = problem
        .solve(
            &SolveOptions {
                adapt: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let theta: Vec<f64> = solved
        .parameters
        .iter()
        .enumerate()
        .map(|(k, t)| t + shift * ((k + 1) as f64).cos())
        .collect();
    let prepared = problem
        .prepare_state(&solved.indices, &theta, true)
        .unwrap();
    let psi = prepared.complex128().unwrap();
    let ansatz = Ansatz {
        indices: &solved.indices,
        parameters: &theta,
        product_form: true,
    };
    let atoms = setup.centers.len();
    let cells: usize = mesh.iter().product();
    let size = 3 * atoms * cells;
    let directions = DMatrix::from_fn(size, 3 * atoms, |row, column| {
        let (atom, axis) = ((row / 3) % atoms, row % 3);
        if atom * 3 + axis == column {
            1.
        } else {
            0.
        }
    });
    let reference = oracle::gradient(
        &base,
        &state,
        &model,
        &problem,
        psi,
        &oracle::Options {
            active_spec: active,
            ansatz: Some(ansatz),
            directions: Some(&directions),
            ..Default::default()
        },
    )
    .unwrap();
    let analytic = response::gradient(
        &base,
        &state,
        &model,
        &problem,
        psi,
        &response::Options {
            ansatz: Some(ansatz),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(analytic.scheme, "analytic");
    assert_eq!(analytic.gradient.as_slice().len(), 3 * atoms);
    let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
    let (mut worst, mut net) = (0f64, [0f64; 3]);
    for component in 0..3 * atoms {
        let a = analytic.gradient.as_slice()[component];
        // The oracle's row of image 0 holds the collective derivative of the supercell's energy.
        let b = reference.gradient.as_slice()[component] / cells as f64;
        println!("{name} component {component}: analytic {a:+.10e} oracle {b:+.10e} Ha/Bohr, miss {:.2e} eV/A", (a - b).abs() * to_ev);
        worst = worst.max((a - b).abs() * to_ev);
        net[component % 3] += a;
    }
    let estimate = reference.estimate.unwrap() / cells as f64 * to_ev;
    let net = net.iter().fold(0f64, |x, v| x.max(v.abs()));
    println!(
        "{name} {mesh:?} {kind} {particles:?} active {active:?} ({} qubits): worst miss {worst:.3e} eV/A (oracle estimate {estimate:.3e} eV/A), reference stationarity {:.2e}, CPHF residual {:.2e}, forms {}; analytic net force {net:.2e} Ha/Bohr",
        2 * model.m, analytic.reference_stationarity, analytic.cphf_residual, analytic.forms
    );
    (worst, net)
}

/// G3 and G4 of Bloch-VQE stage 6a (`BLOCH_VQE_PLAN.md` 6.6): the H2 chain (2,1,1), full space;
/// LiH (2,1,1) with its lowest orbital frozen and an energy selection (the Z-vector); LiH (3,1,1)
/// with its `+-k` pairs whole (the in-cluster parallel-transport gauge, `lowdin_gauge`: 9.9e-5
/// eV/A without it, 1.1e-9 with it, 2026-11-15); the triplet H2 chain (UHF). Gates: every component within the
/// oracle's own estimate or 1e-6 eV/A (the plan's 1e-7 eV/A is reported against the number), the
/// net force reported against G4's 1e-10 Ha/Bohr (the grid's egg box sets its floor, S4-6).
#[test]
#[ignore = "slow: seven cases (about 12 min, 2026-11-15), 24 displaced SCF each; run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_variational_forces -- --ignored g3 --nocapture"]
fn g3_the_analytic_force_of_a_paw_supercell_matches_the_oracle() {
    // LiH (2,1,1) SZ has 4 orbitals (levels -0.108, -0.067 | 0.091, 0.234 Ha): its lowest frozen,
    // 3 active by energy. LiH (3,1,1): 6 orbitals (-0.264 | the pair -0.216, the pair 0.128, 0.235):
    // the lowest frozen, the two pairs whole and the top level active; its parameter Hessian is
    // indefinite off the optimum (-9.4e-3 at a shift of 0.02, -1.3e-3 at 0.005), so it is gated at
    // the optimum (no Z-vector; the case is about the pairs).
    let cases: Vec<(
        &str,
        paw_supercell_support::Primitive,
        [usize; 3],
        &'static str,
        i64,
        Option<&str>,
        f64,
    )> = vec![
        (
            "H2 chain",
            hydrogen_chain_cell(),
            [2, 1, 1],
            "rhf",
            0,
            None,
            0.02,
        ),
        (
            "LiH",
            lithium_hydride_cell(),
            [2, 1, 1],
            "rhf",
            0,
            Some(r#"{"orbitals": 3, "frozen": [0]}"#),
            0.02,
        ),
        (
            "LiH",
            lithium_hydride_cell(),
            [3, 1, 1],
            "rhf",
            0,
            Some(r#"{"orbitals": 5, "frozen": [0]}"#),
            0.,
        ),
        (
            "triplet H2 chain",
            hydrogen_chain_cell(),
            [2, 1, 1],
            "uhf",
            2,
            None,
            0.02,
        ),
        (
            "triplet H2 chain",
            hydrogen_chain_cell(),
            [3, 1, 1],
            "uhf",
            2,
            None,
            0.02,
        ),
        // The +-k pairs of the (3,1,1) mesh inside the full space, through the restricted and
        // the unrestricted relaxed densities (a closed-shell UHF keeps the pairs exact).
        (
            "RHF H2 chain",
            hydrogen_chain_cell(),
            [3, 1, 1],
            "rhf",
            0,
            None,
            0.02,
        ),
        (
            "UHF H2 chain",
            hydrogen_chain_cell(),
            [3, 1, 1],
            "uhf",
            0,
            None,
            0.02,
        ),
    ];
    let mut failures = Vec::new();
    let only = std::env::var("G3_CASES").ok();
    for (name, cell, mesh, kind, spin, active, shift) in cases {
        if only.as_deref().is_some_and(|o| !o.contains(name)) {
            continue;
        }
        // Each case runs on its own: a refused or failing case is reported, the others still run.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g3_case(name, cell, mesh, kind, spin, active, shift)
        })) {
            Ok((worst, net)) => {
                if !(worst < 1e-6) {
                    failures.push(format!("{name} {mesh:?}: {worst:e} eV/A"));
                }
                println!("G4 {name} {mesh:?}: net {net:.2e} Ha/Bohr against 1e-10");
            }
            Err(_) => failures.push(format!(
                "{name} {mesh:?}: the case did not run (see its panic above)"
            )),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// G4, G5 and G7 of Bloch-VQE stage 6a on the RHF determinant (`MolecularIntegrals::gradient`
/// of a PAW-LCAO supercell, the derivative of the energy per cell): G4, the net force of the H2
/// chain (2,1,1) (reported against 1e-10 Ha/Bohr: the grid's egg box sets its floor, S4-6); G5,
/// the acoustic sum rule `sum_a dF_a/dtau_b`, by central differences of the analytic forces under
/// a collective displacement of `b` (2 mBohr; reported against 1e-8); G7, the degenerate Fermi
/// level refused by name: the half-filled H chain on a (4,1,1) mesh (its `+-1/4` pair), and the
/// same RHF state with its lowest empty level moved onto its highest occupied one.
#[test]
fn g4_g5_g7_the_supercell_force_sums_rules_and_refusals() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let options = ScfOptions {
        tolerance: 1e-12,
        ..ScfOptions::default()
    };
    let mesh = [2, 1, 1];
    let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
    let force = |integrals: &mandacaru_core::molecular::MolecularIntegrals| {
        let n = integrals.neutral_electrons();
        let rhf = integrals
            .mean_field("rhf", [n / 2, n / 2], &options)
            .unwrap();
        assert!(rhf.converged);
        integrals.gradient(&rhf).unwrap().total.as_slice().to_vec()
    };
    let base = force(&displaced(&ctx, &setup, mesh, 0, [0.; 3]));
    // The value: N times the per-cell RHF gradient is the relaxed RHF oracle of G1 (displaced SCF,
    // Richardson of 8, 4, 2, 1 mBohr, `the_two_body_derivative_of_a_paw_supercell_closes_g1`),
    // frozen from G1's run on the library installed 2026-11-16 (`lda-sr/H.parquet` sha256
    // 730c4c02d4e03748, see DECISIONS).
    let mut value = 0f64;
    for (component, oracle) in [
        (0, 0.216126835779),
        (1, 0.024487603768),
        (3, -0.216126682096),
    ] {
        let miss = (2. * base[component] - oracle).abs();
        println!("RHF value, component {component}: N g {:+.12} oracle {oracle:+.12} miss {miss:.2e} Ha/Bohr", 2. * base[component]);
        value = value.max(miss);
    }
    // The UHF branch: the triplet H2 chain against central differences of displaced UHF energies,
    // at the RHF tolerance 1e-12: every displaced triplet converges from each of the three starts
    // in 23 iterations to residuals of 2e-15 to 1e-13 (measured 2026-11-15; an earlier state of
    // the integrals stalled one displaced triplet at 2.8e-12, and the branch ran at 1e-11).
    let uhf_options = options.clone();
    let uhf = |integrals: &mandacaru_core::molecular::MolecularIntegrals| {
        let n = integrals.neutral_electrons();
        let state = integrals
            .mean_field("uhf", [n / 2 + 1, n / 2 - 1], &uhf_options)
            .unwrap();
        assert!(state.converged);
        state
    };
    let triplet = displaced(&ctx, &setup, mesh, 0, [0.; 3]);
    let g = triplet
        .gradient(&uhf(&triplet))
        .unwrap()
        .total
        .as_slice()
        .to_vec();
    for (atom, axis) in [(0, 0), (1, 1)] {
        let mut differences = Vec::new();
        for &step in &[4e-3, 2e-3, 1e-3] {
            let mut sides = Vec::new();
            for sign in [1., -1.] {
                let mut shift = [0.; 3];
                shift[axis] = sign * step;
                let moved = displaced(&ctx, &setup, mesh, atom, shift);
                sides.push(uhf(&moved).electronic_energy + moved.nuclear);
            }
            differences.push((sides[0] - sides[1]) / (2. * step));
        }
        let oracle = richardson(&differences) / 2.;
        let miss = (g[3 * atom + axis] - oracle).abs();
        println!("UHF triplet value, atom {atom} axis {axis}: per-cell g {:+.12} oracle {oracle:+.12} miss {miss:.2e} Ha/Bohr", g[3 * atom + axis]);
        value = value.max(miss);
    }
    println!("mean-field gradient value worst miss {value:.2e} Ha/Bohr");
    assert!(value < 1e-9, "{value:e}");
    let net: Vec<f64> = (0..3).map(|x| base[x] + base[3 + x]).collect();
    println!("G4 H2 chain (2,1,1) RHF: per-cell gradient {base:.10?} Ha/Bohr, net {net:.2?} against 1e-10");
    let step = 2e-3;
    let mut worst = 0f64;
    for atom in 0..2 {
        for axis in 0..2 {
            let mut shift = [0.; 3];
            shift[axis] = step;
            let plus = force(&displaced(&ctx, &setup, mesh, atom, shift));
            shift[axis] = -step;
            let minus = force(&displaced(&ctx, &setup, mesh, atom, shift));
            let rule: Vec<f64> = (0..3)
                .map(|x| (plus[x] + plus[3 + x] - minus[x] - minus[3 + x]) / (2. * step))
                .collect();
            let constant = (plus[3 * atom + axis] - minus[3 * atom + axis]) / (2. * step);
            println!("G5 b = atom {atom} axis {axis}: sum_a dF_a/dtau_b {rule:.3?} (the force constant d g_b/d tau_b {constant:+.6e}) Ha/Bohr^2");
            worst = worst.max(rule.iter().fold(0f64, |a, v| a.max(v.abs())));
        }
    }
    println!("G5 worst acoustic sum {worst:.2e} Ha/Bohr^2 against 1e-8");
    // G7: the half-filled H chain on a (4,1,1) mesh.
    let chain = paw_supercell_support::Primitive {
        lattice: nalgebra::Matrix3::new(2.8, 0.3, 0.2, 0., 7., 0.4, 0., 0., 7.),
        shape: [8, 20, 20],
        atoms: vec![("H", [0.1, 0.2, 0.15])],
    };
    let metal = displaced(
        &ctx,
        &cells_of(&ctx, "lda-sr", false, &chain, 1, [0.; 3]),
        [4, 1, 1],
        0,
        [0.; 3],
    );
    let mut rhf = metal.mean_field("rhf", [2, 2], &options).unwrap();
    let gap = rhf.energies[0][2] - rhf.energies[0][1];
    let answer = metal.gradient(&rhf);
    println!(
        "G7 H chain (4,1,1) RHF {:.10} (converged {}), levels {:.6?}, gap {gap:.3e} Ha: {}",
        rhf.electronic_energy,
        rhf.converged,
        &rhf.energies[0][..4],
        answer
            .as_ref()
            .map_or_else(|e| format!("refused: {e}"), |_| "a force".into())
    );
    if rhf.converged && gap < 1e-8 {
        assert!(matches!(answer, Err(mandacaru_core::Error::Unsupported(_))));
    }
    rhf.converged = true;
    rhf.energies[0][2] = rhf.energies[0][1];
    let error = metal
        .gradient(&rhf)
        .err()
        .expect("a degenerate Fermi level is refused");
    println!("G7 degenerate Fermi level: {error}");
    assert!(
        matches!(error, mandacaru_core::Error::Unsupported(_))
            && error.to_string().contains("Fermi level"),
        "{error}"
    );
}

/// MG-REST item 8: the packed pair matrix `(pr|qs)` of a PAW-LCAO supercell's Lowdin functions,
/// the Cholesky columns (smooth products and compensation charges) read one by one through
/// [`PairColumns`] and the one-center remainder (`PawSupercellSource::correction`) apart: the
/// columns' lowest eigenvalues (a Coulomb form: positive semidefinite), the total's (the remainder
/// is not), the diagonal the factorization starts from against the columns' own, the residual a
/// pivoted Cholesky of the columns leaves at 1e-10, and the factored Hamiltonian's integrals
/// against the dense total. Measured 2026-11-15: the total's eigenvalues reach -4.7e-6 Ha on the
/// N2 chain (2,1,1) and -3.7e-6 on (3,1,1) (36 below -1e-10), which no pivoted Cholesky of the
/// total survives; the columns alone are positive semidefinite (lowest +7.8e-10 and +2.0e-13).
#[test]
#[ignore = "slow: H2 chain (3,1,1), N2 chain (2,1,1) and (3,1,1), every Coulomb column twice (2163 s measured 2026-11-15); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_variational_forces -- --ignored pair_matrix --nocapture"]
fn the_pair_matrix_of_a_paw_supercell_is_measured_against_its_cholesky() {
    let ctx = Context::new(Some(4), 3 << 30).unwrap();
    let cases: Vec<(&str, paw_supercell_support::Primitive, [usize; 3])> =
        match std::env::var("PAIR_CASES").as_deref() {
            Ok("N2-3") => vec![("N2 chain", nitrogen_chain_cell(), [3, 1, 1])],
            _ => vec![
                ("H2 chain", hydrogen_chain_cell(), [3, 1, 1]),
                ("N2 chain", nitrogen_chain_cell(), [2, 1, 1]),
                ("N2 chain", nitrogen_chain_cell(), [3, 1, 1]),
            ],
        };
    for (name, cell, mesh) in cases {
        let setup = cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]);
        let copy = setup
            .with_positions(&ctx, &setup.fractional().unwrap())
            .unwrap();
        let integrals = mandacaru_core::molecular::MolecularIntegrals::paw_supercell(
            &ctx,
            std::sync::Arc::new(copy),
            mesh,
            None,
        )
        .unwrap();
        let source = integrals.paw_supercell.as_ref().unwrap();
        let columns = source.columns.as_ref();
        let n = (columns.m() * (columns.m() + 1)) / 2;
        let diagonal = columns.diagonal().unwrap();
        let mut a = DMatrix::<f64>::zeros(n, n);
        for k in 0..n {
            let column = columns.column(k).unwrap();
            for (j, v) in column.iter().enumerate() {
                a[(j, k)] = *v;
            }
        }
        let (imaginary, asymmetry) = columns.departures();
        let start = (0..n)
            .map(|k| (diagonal[k] - a[(k, k)]).abs())
            .fold(0f64, f64::max);
        let transposed = (&a - a.transpose()).abs().max();
        let symmetric = (&a + a.transpose()) * 0.5;
        let m = columns.m();
        let indices: Vec<(usize, usize)> =
            (0..m).flat_map(|p| (0..=p).map(move |r| (p, r))).collect();
        let correction = &source.correction;
        let total = DMatrix::from_fn(n, n, |x, y| {
            symmetric[(x, y)]
                + correction.value(indices[x].0, indices[x].1, indices[y].0, indices[y].1)
        });
        let mut whole: Vec<f64> = total
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .collect();
        whole.sort_by(f64::total_cmp);
        println!(
            "{name} {mesh:?}: with the remainder ({} signed terms) eigenvalues lowest {:?}; below -1e-10: {}",
            correction.factors.len(), whole.iter().take(4).map(|v| format!("{v:.3e}")).collect::<Vec<_>>(), whole.iter().filter(|v| **v < -1e-10).count()
        );
        let eri = integrals.eri().unwrap();
        let report = eri.report();
        let gap = (0..n)
            .flat_map(|x| (0..n).map(move |y| (x, y)))
            .fold(0f64, |w, (x, y)| {
                let ((p, r), (q, s)) = (indices[x], indices[y]);
                w.max((eri.value(p, q, r, s) - total[(x, y)]).abs())
            });
        println!("{name} {mesh:?}: the Hamiltonian's factors {} (rank {:?}): largest |(pr|qs) - total| {gap:.2e} Ha", report.representation, report.cholesky);
        assert!(gap < 1e-8, "{name}: {gap:e}");
        let mut values: Vec<f64> = symmetric
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .collect();
        values.sort_by(f64::total_cmp);
        let below = |t: f64| values.iter().filter(|v| **v < -t).count();
        println!(
            "{name} {mesh:?}: {} functions, {n} pairs, overlap min {:.3e}, metric error {:.1e}; departures imaginary {imaginary:.1e} (pr|qs)-(pr|sq) {asymmetry:.1e}; |diagonal - column diagonal| {start:.1e}; |A - A^T| {transposed:.1e}",
            columns.m(), integrals.overlap_min, source.metric_error
        );
        println!(
            "{name} {mesh:?}: eigenvalues lowest {:?} largest {:.6e}; below -1e-10: {}, -1e-9: {}, -1e-8: {}, -1e-6: {}",
            values.iter().take(6).map(|v| format!("{v:.3e}")).collect::<Vec<_>>(), values[n - 1], below(1e-10), below(1e-9), below(1e-8), below(1e-6)
        );
        // The production pivoted Cholesky on the dense columns, from either diagonal.
        for (label, first) in [
            ("diagonal()", diagonal.clone()),
            (
                "column diagonal",
                (0..n).map(|k| a[(k, k)]).collect::<Vec<_>>(),
            ),
        ] {
            let mut residual = first;
            let mut used = vec![false; n];
            let mut factors: Vec<Vec<f64>> = Vec::new();
            loop {
                let pivot = (0..n)
                    .max_by(|&x, &y| residual[x].total_cmp(&residual[y]))
                    .unwrap();
                let top = residual[pivot];
                if top <= 1e-10 || factors.len() == n {
                    break;
                }
                let scale = 1. / top.sqrt();
                let factor: Vec<f64> = (0..n)
                    .map(|j| {
                        (a[(j, pivot)] - factors.iter().map(|f| f[j] * f[pivot]).sum::<f64>())
                            * scale
                    })
                    .collect();
                for j in 0..n {
                    residual[j] -= factor[j] * factor[j];
                }
                used[pivot] = true;
                factors.push(factor);
            }
            let worst = (0..n)
                .min_by(|&x, &y| residual[x].total_cmp(&residual[y]))
                .unwrap();
            println!(
                "{name} {mesh:?}: Cholesky at 1e-10 from the {label}: rank {}, lowest residual {:.3e} on pair {worst} ({} pivot), negative below -1e-9 on {} pairs",
                factors.len(), residual[worst], if used[worst] { "a used" } else { "an unused" }, residual.iter().filter(|r| **r < -1e-9).count()
            );
        }
    }
}
