//! Stationary PAW-LCAO forces against independently converged total energies.
//!
//! The PAW input is a synthetic hydrogen-like dataset built here from smooth
//! analytic radial tables, so the test needs no dataset files. It exercises
//! the full molecular PAW engine (projectors, overlap and compensation
//! augmentation, signed one-center Coulomb factors, local potential). Real
//! datasets, including the closed-1s core exchange of lithium, are covered
//! through `Mandacaru(...).get_forces()` in `test/migration/test_native_dft.py`.
use mandacaru_core::{
    math::{storage::Precision, SpatialGrid},
    molecular::{MeanField, MolecularIntegrals, ScfOptions},
    paw::molecular::{build, Arrays},
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};
use std::f64::consts::PI;

const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = 27.211386245988 / 0.529177210903;
const SPACING: f64 = 0.5;
const CELL: f64 = 14.;
const RADIUS: f64 = 1.;

fn bessel(l: usize, x: f64) -> f64 {
    match l {
        0 => x.sin() / x,
        _ => x.sin() / (x * x) - x.cos() / x,
    }
}
fn smooth_cutoff(r: f64, edge: f64) -> f64 {
    if r >= edge {
        0.
    } else {
        (1. - (r / edge).powi(2)).powi(3)
    }
}
fn table(ctx: &Context, arrays: &mut Arrays, values: &[f64]) -> Value {
    let key = format!("a{}", arrays.len());
    arrays.insert(key.clone(), ctx.copy_slice(values).unwrap());
    Value::String(key)
}
/// A hydrogen-like species with s and p channels, two partial waves each.
fn species(ctx: &Context, arrays: &mut Arrays, functional: &str) -> Value {
    let r: Vec<f64> = (1..=700).map(|k| 0.02 * k as f64).collect();
    let n = r.len();
    let mut channels = Vec::new();
    for l in 0..2usize {
        let power = |r: f64| r.powi(l as i32);
        let ae: Vec<Vec<f64>> = vec![
            r.iter().map(|&r| 1.5 * power(r) * (-r).exp()).collect(),
            r.iter()
                .map(|&r| 1.1 * power(r) * (-0.6 * r).exp())
                .collect(),
        ];
        let projectors: Vec<Vec<f64>> = (0..2)
            .map(|i| {
                r.iter()
                    .map(|&r| {
                        let x = r / RADIUS;
                        if x >= 1. {
                            0.
                        } else {
                            (1.2 + 0.5 * i as f64 * x) * power(r) * (1. - x * x).powi(2)
                        }
                    })
                    .collect()
            })
            .collect();
        let mut wavevectors = Vec::new();
        let mut coefficients = Vec::new();
        for wave in &ae {
            let q: Vec<f64> = (1..=8).map(|z| PI * z as f64 / RADIUS).collect();
            let points: Vec<usize> = (0..n).filter(|&k| r[k] <= RADIUS && k % 2 == 0).collect();
            let design =
                DMatrix::from_fn(points.len(), q.len(), |a, z| bessel(l, q[z] * r[points[a]]));
            let target = DVector::from_fn(points.len(), |a, _| wave[points[a]]);
            let fit = design
                .svd(true, true)
                .solve(&target, 1e-10)
                .unwrap()
                .as_slice()
                .to_vec();
            wavevectors.push(table(ctx, arrays, &q));
            coefficients.push(table(ctx, arrays, &fit));
        }
        let flat = |rows: &[Vec<f64>]| rows.iter().flatten().copied().collect::<Vec<_>>();
        channels.push(json!({
            "l": l,
            "principal_n": l + 1,
            "radius": RADIUS,
            "occupations": if l == 0 { vec![1., 0.] } else { vec![0., 0.] },
            "ae": table(ctx, arrays, &flat(&ae)),
            "projectors": table(ctx, arrays, &flat(&projectors)),
            "coupling": table(ctx, arrays, &[-0.08, -0.15, -0.15, -0.30]),
            "overlap": table(ctx, arrays, &[0.014, 0.026, 0.026, 0.050]),
            "vanderbilt": table(ctx, arrays, &[-0.063, -0.138, -0.125, -0.276]),
            "wavevectors": wavevectors,
            "wave_coefficients": coefficients,
        }));
    }
    let ionic: Vec<f64> = r.iter().map(|&r| -libm::erf(r / 0.35) / r).collect();
    let valence: Vec<f64> = r.iter().map(|&r| (-2. * r).exp() / PI).collect();
    json!({
        "atomic_number": 1,
        "spin_orbit": false,
        "frozen_subshells": [],
        "semicore_subshells": [],
        "r": table(ctx, arrays, &r),
        "ionic": table(ctx, arrays, &ionic),
        "core": table(ctx, arrays, &vec![0.; n]),
        "valence": table(ctx, arrays, &valence),
        "xc_core": table(ctx, arrays, &vec![0.; n]),
        "charge": 1.0,
        "local_radius": 0.9,
        "compensation_radius": 1.0,
        "functional": functional,
        "one_center_energy": 0.002,
        "relativistic_exchange": false,
        "channels": channels,
    })
}
fn integrals(ctx: &Context, positions: &[[f64; 3]], functional: &str) -> MolecularIntegrals {
    let geometry = Geometry::new(
        vec![1; positions.len()],
        positions.to_vec(),
        [[CELL, 0., 0.], [0., CELL, 0.], [0., 0., CELL]],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap();
    let grid = SpatialGrid::from_geometry(&geometry, [SPACING; 3], false, false).unwrap();
    let mut arrays = Arrays::new();
    let species = species(ctx, &mut arrays, functional);
    let mut basis = Vec::new();
    let radii: Vec<f64> = (0..=350).map(|k| 0.02 * k as f64).collect();
    let tables: Vec<(usize, Value, Value)> = [0usize, 1]
        .into_iter()
        .map(|l| {
            let values: Vec<f64> = radii
                .iter()
                .map(|&r| r.powi(l as i32) * (-1.1 * r).exp() * smooth_cutoff(r, 7.))
                .collect();
            (
                l,
                table(ctx, &mut arrays, &radii),
                table(ctx, &mut arrays, &values),
            )
        })
        .collect();
    // The s function on every atom shares one table; the p set is added so
    // that the basis can polarize, which the forces need.
    for atom in 0..positions.len() {
        for (l, r, values) in &tables {
            for m in -(*l as i32)..=*l as i32 {
                basis.push(json!({"atom": atom, "l": l, "m": m, "r": r, "values": values}));
            }
        }
    }
    // Radial keys are consumed on first use; later atoms hit the value cache.
    let metadata = json!({
        "format": "molecular-paw-input",
        "version": 1,
        "length_unit": "bohr",
        "energy_unit": "Ha",
        "projector_basis": "dual",
        "atoms": vec![0; positions.len()],
        "species": [species],
        "basis": basis,
    });
    build(
        ctx,
        geometry,
        grid,
        &metadata.to_string(),
        arrays,
        "spectral",
        1e-12,
    )
    .unwrap()
}
fn options() -> ScfOptions {
    ScfOptions {
        tolerance: 1e-12,
        max_iterations: 400,
        ..Default::default()
    }
}
#[derive(Clone, Copy)]
enum Method {
    Hf(&'static str),
    Ks(&'static str),
}
fn solve(integrals: &MolecularIntegrals, method: Method, particles: [usize; 2]) -> MeanField {
    let state = match method {
        Method::Hf(kind) => integrals.mean_field(kind, particles, &options()).unwrap(),
        Method::Ks(name) => integrals
            .kohn_sham(name, particles, particles[0] != particles[1], &options())
            .unwrap(),
    };
    assert!(state.converged, "SCF did not converge");
    state
}
fn displaced(
    base: &MolecularIntegrals,
    atom: usize,
    axis: usize,
    delta: f64,
) -> MolecularIntegrals {
    let mut positions = base.geometry.positions().to_vec();
    positions[atom][axis] += delta;
    let geometry = Geometry::new(
        base.geometry.numbers().to_vec(),
        positions,
        *base.geometry.cell(),
        base.geometry.pbc(),
        LengthUnit::Bohr,
    )
    .unwrap();
    base.at_geometry(geometry).unwrap()
}
/// Richardson-extrapolated derivative of independently converged total
/// energies on the same fixed grid, in Hartree per Bohr.
fn energy_derivative(
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    atom: usize,
    axis: usize,
) -> f64 {
    let energy = |delta: f64| {
        let moved = displaced(base, atom, axis, delta);
        solve(&moved, method, particles).electronic_energy + moved.nuclear
    };
    let h = 1e-3;
    let coarse = (energy(h) - energy(-h)) / (2. * h);
    let fine = (energy(0.5 * h) - energy(-0.5 * h)) / h;
    (4. * fine - coarse) / 3.
}
/// Largest |F_stationary - F_finite_difference| in eV/Angstrom over the
/// listed components; the measured value is printed for the record.
fn check(
    label: &str,
    base: &MolecularIntegrals,
    method: Method,
    particles: [usize; 2],
    components: &[(usize, usize)],
    tolerance_ev: f64,
) -> Vec<f64> {
    let state = solve(base, method, particles);
    let gradient = base.gradient(&state).unwrap();
    assert!(gradient.hellmann_feynman.is_none() && gradient.pulay.is_none());
    assert_eq!(gradient.step, Some(1e-3));
    let mut worst = 0f64;
    for &(atom, axis) in components {
        let reference = energy_derivative(base, method, particles, atom, axis);
        let analytic = gradient.total.as_slice()[3 * atom + axis];
        worst = worst.max((analytic - reference).abs() * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM);
    }
    println!(
        "{label}: max |F - F_fd| = {worst:.3e} eV/A, step estimate {:.2e} Ha/Bohr, max |F| = {:.4} eV/A",
        gradient.estimate.unwrap(),
        gradient.total.as_slice().iter().fold(0f64, |a, v| a.max(v.abs()))
            * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    );
    assert!(worst < tolerance_ev, "{label}: {worst:.3e} eV/A");
    gradient.total.as_slice().to_vec()
}
const H2: [[f64; 3]; 2] = [[0., 0., -0.8], [0., 0., 0.8]];

#[test]
fn closed_shell_h2_forces_follow_total_energies_for_hf_lda_and_pbe() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        for (functional, method) in [
            ("lda", Method::Hf("rhf")),
            ("lda", Method::Ks("lda")),
            ("pbe", Method::Ks("pbe")),
        ] {
            let base = integrals(&ctx, &H2, functional);
            let g = check(
                &format!(
                    "H2 {}",
                    match method {
                        Method::Hf(_) => "RHF".into(),
                        Method::Ks(n) => n.to_uppercase(),
                    }
                ),
                &base,
                method,
                [1, 1],
                &[(0, 2), (1, 2), (1, 0)],
                1e-5,
            );
            // The midpoint is a grid node, so the grid is mirror symmetric:
            // the bond force is antisymmetric and transverse forces vanish.
            let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
            assert!((g[2] + g[5]).abs() * to_ev < 1e-6, "net bond force");
            assert!([g[0], g[1], g[3], g[4]]
                .iter()
                .all(|v| v.abs() * to_ev < 1e-6));
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn meta_gga_h2_forces_follow_total_energies() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "pbe");
        check(
            "H2 r2SCAN",
            &base,
            Method::Ks("r2scan"),
            [1, 1],
            &[(0, 2), (1, 0)],
            1e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn open_shell_forces_follow_total_energies_for_uhf_and_uks() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // Stretched H2 triplet, and a bent H3 doublet that has net transverse
        // and bond-angle forces.
        let triplet = integrals(&ctx, &[[0., 0., -1.1], [0., 0., 1.1]], "lda");
        check(
            "H2 triplet UHF",
            &triplet,
            Method::Hf("uhf"),
            [2, 0],
            &[(0, 2), (1, 0)],
            1e-5,
        );
        check(
            "H2 triplet UKS-LDA",
            &triplet,
            Method::Ks("lda"),
            [2, 0],
            &[(1, 2)],
            1e-5,
        );
        let bent = integrals(
            &ctx,
            &[[0., 0., 0.], [1.7, 0., 0.3], [0.4, 1.6, -0.5]],
            "pbe",
        );
        check(
            "H3 UHF",
            &bent,
            Method::Hf("uhf"),
            [2, 1],
            &[(0, 0), (1, 2), (2, 1)],
            1e-5,
        );
        check(
            "H3 UKS-PBE",
            &bent,
            Method::Ks("pbe"),
            [2, 1],
            &[(1, 0), (2, 2)],
            1e-5,
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn bond_force_does_not_depend_on_the_cartesian_axis_of_the_bond() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        // The angular quadratures of the PAW spheres are not cubic-symmetric,
        // so a rotated molecule differs from the original by the (tiny)
        // quadrature error of the energy itself. The force must inherit
        // exactly that error, and no more: the axis bookkeeping is right when
        // the force difference stays at the size of the energy difference
        // scaled by the bond length, not at the size of the force.
        let along_z = integrals(&ctx, &H2, "pbe");
        let along_x = integrals(&ctx, &[[-0.8, 0., 0.], [0.8, 0., 0.]], "pbe");
        let z_state = solve(&along_z, Method::Ks("pbe"), [1, 1]);
        let x_state = solve(&along_x, Method::Ks("pbe"), [1, 1]);
        let z = along_z.gradient(&z_state).unwrap();
        let x = along_x.gradient(&x_state).unwrap();
        let to_ev = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM;
        let force = (z.total.as_slice()[5] - x.total.as_slice()[3]).abs() * to_ev;
        let energy = (z_state.electronic_energy - x_state.electronic_energy).abs();
        println!("axis-permuted bond force difference: {force:.3e} eV/A, energy difference {energy:.3e} Ha");
        assert!(force < 1e-4 && energy < 1e-5);
        let (z, x) = (z.total.as_slice(), x.total.as_slice());
        let transverse = [z[0], z[1], z[3], z[4], x[1], x[2], x[4], x[5]]
            .iter()
            .fold(0f64, |a, v| a.max(v.abs()))
            * to_ev;
        assert!(transverse < 1e-6, "{transverse:.3e}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
#[test]
fn unsupported_states_raise_instead_of_returning_forces() {
    let ctx = Context::new(Some(2), 512 << 20).unwrap();
    {
        let base = integrals(&ctx, &H2, "lda");
        let spinor = base.mean_field("ghf", [1, 1], &options()).unwrap();
        let error = base.gradient(&spinor).err().unwrap().to_string();
        assert!(error.contains("spinor"), "{error}");
        let mut state = solve(&base, Method::Hf("rhf"), [1, 1]);
        state.converged = false;
        let error = base.gradient(&state).err().unwrap().to_string();
        assert!(error.contains("converged"), "{error}");
        // Screened exchange and nonlocal correlation have no PAW force.
        for name in ["hse06", "pbe+rvv10"] {
            let mut state = solve(&base, Method::Ks("pbe"), [1, 1]);
            state.functional = Some(name.into());
            let error = base.gradient(&state).err().unwrap().to_string();
            assert!(error.contains(name), "{error}");
        }
        // Single-precision samples cannot carry a finite difference.
        let single = integrals(&ctx, &H2, "lda")
            .with_precision(Precision::Single)
            .unwrap();
        let good = solve(&base, Method::Hf("rhf"), [1, 1]);
        let error = single.gradient(&good).err().unwrap().to_string();
        assert!(error.contains("double-precision"), "{error}");
        // A state from other integrals does not reproduce its energy here.
        let other = integrals(&ctx, &[[0., 0., -1.2], [0., 0., 1.2]], "lda");
        let foreign = solve(&other, Method::Hf("rhf"), [1, 1]);
        let error = base.gradient(&foreign).err().unwrap().to_string();
        assert!(error.contains("not computed with them"), "{error}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
