//! SH-F1: the explicit exact-exchange terms of a SPINOR (spin-orbit) hybrid crystal on a complex-k
//! mesh, one term at a time, against the production energy. The oracle is the central difference
//! of `Crystal::hybrid_exchange` of crystals rebuilt at the moved geometry, the converged
//! `2m x 2m` density matrices held fixed (the displaced-crystal oracle of
//! `periodic_gradient_hybrid_mesh.rs`, the spinor path of the K builder: no code shared with
//! `hybrid_gradient::pass`). Pieces compared: the phase of the compensation shapes, the pair
//! functions (bra and ket, both spin components) and the projection chain (the response `Z`,
//! `p x m` summed over the two components, through the one-center response with the full
//! spin-structured operator).
//!
//! Also here: the explicit terms are covariant under a global spin rotation `1 x u` of the density
//! matrices (a time-reversal-symmetric ensemble stays magnetization free, so the sphere remainder
//! stays absent), and a rotation that is NOT a spin rotation changes them. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::{
    math::storage::{MatrixSource, StoredMatrix},
    periodic::{
        hybrid_gradient::{explicit, projection_forces, Gradient},
        screened_exchange::Screening,
        setup::pulay::{projection_gradient, Mesh},
        Crystal,
    },
    quantum::C64,
};
use nalgebra::DMatrix;
use periodic_gradient_crystals_support::{mesh::*, *};

const CASE: &str = "si_hsefr_tr3";

type Matrix = DMatrix<C64>;

/// phase + functions + projection chain, per atom and axis.
fn totals(data: &Data, gradient: &Gradient) -> Vec<[f64; 3]> {
    let c = &data.crystal;
    let gradients: Vec<_> = c
        .points
        .iter()
        .map(|p| {
            projection_gradient(
                &c.context,
                &data.shells,
                &data.lattice,
                p.k,
                &data.projectors,
            )
            .unwrap()
        })
        .collect();
    let chain = projection_forces(
        &gradient.response(),
        &gradients,
        &data.owner,
        &data.projectors.iter().map(|p| p.atom).collect::<Vec<_>>(),
        data.atoms.len(),
    )
    .unwrap();
    let functions = gradient.functions.as_ref().unwrap().total();
    (0..data.atoms.len())
        .map(|a| {
            std::array::from_fn(|x| {
                gradient.phase[a][x] + functions[a][x] + chain.rows[a][x] + chain.columns[a][x]
            })
        })
        .collect()
}

fn explicit_of(
    data: &Data,
    mesh: &Mesh,
    matrices: &[StoredMatrix],
) -> mandacaru_core::Result<Gradient> {
    let c = &data.crystal;
    explicit(
        c,
        matrices,
        Screening::of(c).unwrap(),
        data.options.exxdiv_treatment,
        &c.hybrid_augmentation().unwrap(),
        Some(&ShellMotion { data, mesh }),
    )
}

fn complex_mesh_of(c: &Crystal) -> bool {
    c.points.iter().any(|p| {
        p.k.iter()
            .any(|x| (x.abs() - 0.5).abs() > 1e-9 && x.abs() > 1e-9)
    })
}

#[test]
fn the_spinor_explicit_terms_on_the_complex_k_mesh_are_the_displaced_crystal_derivative() {
    let data = Data::load(CASE);
    let c = &data.crystal;
    assert!(c.spinor && c.time_reversal && c.spins == 1);
    assert!(complex_mesh_of(c), "the mesh must hold a non-TRIM point");
    let result = data.solve();
    let mesh = Mesh::new(&c.context, &c.grid).unwrap();
    let gradient = explicit_of(&data, &mesh, &result.matrices).unwrap();
    // The derivative belongs to the production energy, and the two spin components add up.
    let production = c
        .hybrid_exchange(&result.matrices, data.options.exxdiv_treatment)
        .unwrap()
        .energy;
    let tie = gradient.energy + gradient.one_center_energy - production;
    eprintln!("explicit energy - hybrid_exchange energy = {tie:e}");
    assert!(tie.abs() < 1e-12, "energy tie {tie:e}");
    assert_eq!(gradient.components.len(), 2);
    let analytic = totals(&data, &gradient);
    // Coordinates: both atoms, every axis; the slow ones are ignored in the default run.
    let list: Vec<(usize, usize)> = vec![(1, 0), (0, 2)];
    for (atom, axis) in list {
        let (slope, floor) = finite_difference(&data, &mesh, &result.matrices, atom, axis);
        let parts: Vec<f64> = vec![
            gradient.phase[atom][axis],
            gradient.functions.as_ref().unwrap().total()[atom][axis],
        ];
        let miss = (analytic[atom][axis] - slope).abs();
        let gate = (100. * floor).max(1e-10);
        eprintln!(
            "atom {atom} axis {axis}: analytic {:+.9e}, displaced crystals {slope:+.9e}, miss {miss:.2e}, \
             step floor {floor:.1e}, gate {gate:.1e}; phase {:+.3e}, functions {:+.3e}",
            analytic[atom][axis], parts[0], parts[1]
        );
        assert!(
            miss < gate,
            "atom {atom} axis {axis}: {miss:e} against {gate:e}"
        );
        assert!(
            parts[1].abs() > 100. * gate,
            "the pair-function term is above 100 gates"
        );
    }
}

#[test]
#[ignore = "four more coordinates of six crystal builds each: 161 s measured on 2026-11-02 (loaded machine); run: cargo test --release -p mandacaru-core --test periodic_gradient_spinor_hybrid_mesh -- --ignored"]
fn the_other_coordinates_of_the_spinor_mesh_oracle() {
    let data = Data::load(CASE);
    let c = &data.crystal;
    let result = data.solve();
    let mesh = Mesh::new(&c.context, &c.grid).unwrap();
    let analytic = totals(&data, &explicit_of(&data, &mesh, &result.matrices).unwrap());
    for (atom, axis) in [(1, 1), (1, 2), (0, 0), (0, 1)] {
        let (slope, floor) = finite_difference(&data, &mesh, &result.matrices, atom, axis);
        let miss = (analytic[atom][axis] - slope).abs();
        let gate = (100. * floor).max(1e-10);
        eprintln!("atom {atom} axis {axis}: miss {miss:.2e}, gate {gate:.1e}");
        assert!(miss < gate, "atom {atom} axis {axis}: {miss:e}");
    }
}

/// `(1 x u) D (1 x u)^+` with `u` the SU(2) rotation of angle `theta` about the unit axis `n`.
fn spin_rotated(data: &Data, matrices: &[StoredMatrix], u: &Matrix) -> Vec<StoredMatrix> {
    let c = &data.crystal;
    let m = c.m;
    let mut big = Matrix::zeros(2 * m, 2 * m);
    for s in 0..2 {
        for t in 0..2 {
            for mu in 0..m {
                big[(s * m + mu, t * m + mu)] = u[(s, t)];
            }
        }
    }
    matrices
        .iter()
        .map(|d| {
            let d = d.matrix(&c.context).unwrap();
            StoredMatrix::new(&c.context, &big * &*d * big.adjoint(), false).unwrap()
        })
        .collect()
}

fn su2(theta: f64, n: [f64; 3]) -> Matrix {
    let (c, s) = ((theta / 2.).cos(), (theta / 2.).sin());
    let i = C64::i();
    let [x, y, z] = n;
    Matrix::from_row_slice(
        2,
        2,
        &[
            C64::new(c, 0.) - i * (s * z),
            -i * (s * x) - (s * y) * C64::new(1., 0.),
            -i * (s * x) + (s * y) * C64::new(1., 0.),
            C64::new(c, 0.) + i * (s * z),
        ],
    )
}

#[test]
fn the_explicit_terms_are_su2_covariant_and_a_spatial_rotation_is_not_a_symmetry() {
    let data = Data::load(CASE);
    let c = &data.crystal;
    let result = data.solve();
    let mesh = Mesh::new(&c.context, &c.grid).unwrap();
    let base = totals(&data, &explicit_of(&data, &mesh, &result.matrices).unwrap());
    let peak = base.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    // The cross-spin blocks must be populated, or the rotation is not a test: a spin-orbit state
    // has `D_ud != 0`.
    let (mut cross, mut all) = (0f64, 0f64);
    let m = c.m;
    for d in &result.matrices {
        let d = d.matrix(&c.context).unwrap();
        cross = cross.max(d.view((0, m), (m, m)).norm());
        all = all.max(d.norm());
    }
    eprintln!("cross-spin block {cross:.3e} of {all:.3e}; peak explicit term {peak:.3e}");
    // How much the rotation moves the density matrices, which is what the covariance is tested on
    // (the cross blocks of a weak spin-orbit state are a fraction of a percent).
    assert!(cross > 1e-3 * all, "the cross-spin blocks are populated");
    let u = su2(0.9, [0.48, -0.6, 0.64]);
    let rotated = spin_rotated(&data, &result.matrices, &u);
    // The rotation moves the density matrices by this much: that is the size of what the
    // covariance is tested on (a non-magnetic ensemble is invariant but for its spin-orbit blocks).
    let moved_d = result
        .matrices
        .iter()
        .zip(&rotated)
        .map(|(a, b)| (&*a.matrix(&c.context).unwrap() - &*b.matrix(&c.context).unwrap()).norm())
        .fold(0f64, f64::max);
    eprintln!("the rotation moves the density matrices by {moved_d:.3e} (of {all:.3e})");
    assert!(moved_d > 1e-3 * all, "the rotation is not vacuous");
    let turned = totals(&data, &explicit_of(&data, &mesh, &rotated).unwrap());
    let gap = base
        .iter()
        .zip(&turned)
        .flat_map(|(a, b)| (0..3).map(move |x| (a[x] - b[x]).abs()))
        .fold(0f64, f64::max);
    eprintln!("SU(2) covariance of the explicit terms: {gap:.2e} Ha/Bohr");
    assert!(gap < 1e-10, "{gap:e}");
    // A rotation that mixes two spatial functions (the same on both spin components, so the
    // ensemble stays time-reversal symmetric and magnetization free) is not a symmetry of the
    // force: the same comparison must miss by far more.
    let mut mix = Matrix::identity(2 * m, 2 * m);
    for block in [0, m] {
        mix[(block, block)] = C64::new(0.8, 0.);
        mix[(block, block + 1)] = C64::new(0.6, 0.);
        mix[(block + 1, block)] = C64::new(-0.6, 0.);
        mix[(block + 1, block + 1)] = C64::new(0.8, 0.);
    }
    let wrong: Vec<StoredMatrix> = result
        .matrices
        .iter()
        .map(|d| {
            let d = d.matrix(&c.context).unwrap();
            StoredMatrix::new(&c.context, &mix * &*d * mix.adjoint(), false).unwrap()
        })
        .collect();
    // What the covariance is sensitive to: the explicit terms with the cross-spin blocks of `D`
    // dropped (still time-reversal symmetric) differ from the full ones by far more than the
    // covariance gap.
    let no_cross: Vec<StoredMatrix> = result
        .matrices
        .iter()
        .map(|d| {
            let mut d = (*d.matrix(&c.context).unwrap()).clone();
            d.view_mut((0, m), (m, m)).fill(C64::default());
            d.view_mut((m, 0), (m, m)).fill(C64::default());
            StoredMatrix::new(&c.context, d, false).unwrap()
        })
        .collect();
    let dropped = totals(&data, &explicit_of(&data, &mesh, &no_cross).unwrap());
    let sensitivity = base
        .iter()
        .zip(&dropped)
        .flat_map(|(a, b)| (0..3).map(move |x| (a[x] - b[x]).abs()))
        .fold(0f64, f64::max);
    eprintln!("the cross-spin blocks of D move the explicit terms by {sensitivity:.2e} Ha/Bohr");
    assert!(
        sensitivity > 1e-8 && sensitivity > 1e6 * gap.max(1e-16),
        "{sensitivity:e}"
    );
    // The rotation is not a symmetry: it gives the ensemble a spherical-average magnetization,
    // and the time-reversal flag of the crystal makes the one-center model refuse it (it would
    // otherwise drop the remainder silently).
    match explicit_of(&data, &mesh, &wrong) {
        Err(mandacaru_core::Error::Invalid(message)) => {
            eprintln!("a spatial mixing of two functions: refused: {message}");
            assert!(message.contains("magnetization"), "{message}");
        }
        other => panic!(
            "the mixed ensemble must be refused, got {:?}",
            other.map(|g| g.energy)
        ),
    }
}
