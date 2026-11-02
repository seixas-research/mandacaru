//! The explicit exact-exchange terms on a k MESH (`k != q` pairs, the Nyquist planes of an even
//! grid) against central differences of the production energy `Crystal::hybrid_exchange` of
//! crystals rebuilt at the moved geometry, with the density matrices of the converged SCF held
//! fixed: the displaced-crystal oracle of `periodic_hybrid_gradient_motion.rs` (Gamma only)
//! extended to the 2x2x2 Si mesh of the frozen HSE06 case. The pieces compared are the ones the
//! engine adds: the phase, the pair functions (bra and ket) and the projection chain. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::{
    periodic::{
        hybrid_gradient::{explicit, projection_forces, Motion},
        screened_exchange::Screening,
        setup::{
            pulay::{projection, projection_gradient, KSamples, Mesh},
            shells::{DenseRows, Sampling, Shells},
            spheres::Projector,
        },
        Crystal,
    },
    quantum::C64,
    Buffer, Result,
};
use periodic_gradient_crystals_support::*;
use serde_json::{json, Value};

/// The center derivatives of the Bloch sums of the shells at the crystal's own k-points.
struct ShellMotion<'a> {
    data: &'a Data,
    mesh: &'a Mesh,
}

impl Motion for ShellMotion<'_> {
    fn owner(&self) -> &[usize] {
        &self.data.owner
    }
    fn center_derivative(&self, c: &Crystal, k: usize) -> Result<[Buffer<C64>; 3]> {
        let d = self.data;
        let samples = KSamples::new(&c.context, &d.shells, &d.lattice, self.mesh, c.points[k].k)?;
        let len = samples.slope(0).len();
        let mut out = [
            c.context.allocate::<C64>(len)?,
            c.context.allocate::<C64>(len)?,
            c.context.allocate::<C64>(len)?,
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            for (x, y) in to.as_mut_slice().iter_mut().zip(samples.slope(axis)) {
                *x = -y;
            }
        }
        Ok(out)
    }
}

/// The shells and projectors with `atom` moved by `delta` along `axis`.
fn moved(data: &Data, atom: usize, axis: usize, delta: f64) -> (Shells, Vec<Projector>) {
    let shells = data
        .shells
        .shells()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            if data.owner[s.rows[0]] == atom {
                s.center[axis] += delta;
            }
            s
        })
        .collect();
    let projectors = data
        .projectors
        .iter()
        .map(|p| {
            let mut p = p.clone();
            if p.atom == atom {
                p.center[axis] += delta;
            }
            p
        })
        .collect();
    (Shells::new(shells).unwrap(), projectors)
}

/// The crystal of the frozen case with `atom` moved: Bloch samples and projections of every
/// k-point rebuilt from the moved shells, the sphere centers moved, the rest of the snapshot kept.
fn displaced(data: &Data, mesh: &Mesh, atom: usize, axis: usize, delta: f64) -> Crystal {
    let (shells, projectors) = moved(data, atom, axis, delta);
    let ctx = &data.crystal.context;
    let (m, n, p) = (
        data.crystal.m,
        data.crystal.grid.index.points(),
        data.crystal.p,
    );
    let mut value: Value = snapshot_of(data.case, &data.manifest);
    value["sites"]["centers"][atom][axis] = json!(data.atoms[atom].center[axis] + delta);
    for (index, point) in data.crystal.points.iter().enumerate() {
        let mut psi = vec![C64::default(); m * n];
        shells
            .values(
                ctx,
                &Sampling {
                    lattice: &data.lattice,
                    region: mesh.region(),
                    kpoints: &[point.k],
                },
                mesh.points(),
                &mut DenseRows::new(&mut psi, m, n),
            )
            .unwrap();
        let c = projection(ctx, &shells, &data.lattice, point.k, &projectors).unwrap();
        value["kpoints"][index]["psi"] =
            json!(psi.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>());
        value["kpoints"][index]["projections"] = json!((0..m * p)
            .map(|x| [c[(x / p, x % p)].re, c[(x / p, x % p)].im])
            .collect::<Vec<_>>());
    }
    Crystal::from_json(ctx, &value.to_string()).unwrap()
}

/// `dE/dR` of the exchange energy at fixed density matrices: central differences of the rebuilt
/// crystals, Richardson in `step` and `2 step`, and the floor `|R(step) - R(step / 2)|`.
fn finite_difference(
    data: &Data,
    mesh: &Mesh,
    matrices: &[mandacaru_core::math::storage::StoredMatrix],
    atom: usize,
    axis: usize,
) -> (f64, f64) {
    let exxdiv = data.options.exxdiv_treatment;
    let energy = |delta: f64| {
        displaced(data, mesh, atom, axis, delta)
            .hybrid_exchange(matrices, exxdiv)
            .unwrap()
            .energy
    };
    let central = |step: f64| (energy(step) - energy(-step)) / (2. * step);
    let richardson = |step: f64| {
        let (fine, coarse) = (central(step), central(2. * step));
        fine + (fine - coarse) / 3.
    };
    let (full, half) = (richardson(2e-3), richardson(1e-3));
    (half, (full - half).abs())
}

/// The measured step floors are 1e-13 .. 1e-11 Ha/Bohr; the gate is 100 x the floor of each
/// coordinate, no tighter than 1e-10 (the SCF-free energies are good to about 1e-14 per build).
fn check(coordinates: &[(usize, usize)]) {
    let data = Data::load("si_hse06");
    let result = data.solve();
    let c = &data.crystal;
    let mesh = Mesh::new(&c.context, &c.grid).unwrap();
    let augmentation = c.hybrid_augmentation().unwrap();
    let motion = ShellMotion {
        data: &data,
        mesh: &mesh,
    };
    let gradient = explicit(
        c,
        &result.matrices,
        Screening::of(c).unwrap(),
        data.options.exxdiv_treatment,
        &augmentation,
        Some(&motion),
    )
    .unwrap();
    // The energy the derivative belongs to is the production one.
    let production = c
        .hybrid_exchange(&result.matrices, data.options.exxdiv_treatment)
        .unwrap()
        .energy;
    let tie = gradient.energy + gradient.one_center_energy - production;
    eprintln!("explicit energy - hybrid_exchange energy = {tie:e}");
    assert!(tie.abs() < 1e-12);
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
    for &(atom, axis) in coordinates {
        let analytic = gradient.phase[atom][axis]
            + functions[atom][axis]
            + chain.rows[atom][axis]
            + chain.columns[atom][axis];
        let (slope, floor) = finite_difference(&data, &mesh, &result.matrices, atom, axis);
        eprintln!(
            "atom {atom} axis {axis}: analytic {analytic:+.9e}, displaced crystals {slope:+.9e}, \
             miss {:.2e}, step floor {floor:.1e}; phase {:+.3e}, functions {:+.3e}, rows {:+.3e}, columns {:+.3e}",
            (analytic - slope).abs(),
            gradient.phase[atom][axis],
            functions[atom][axis],
            chain.rows[atom][axis],
            chain.columns[atom][axis]
        );
        let gate = (100. * floor).max(1e-10);
        assert!(
            (analytic - slope).abs() < gate,
            "atom {atom} axis {axis}: gate {gate:e}"
        );
        // Not vacuous: every piece of the exchange force is far above the gate on this coordinate.
        assert!(
            [
                gradient.phase[atom][axis],
                functions[atom][axis],
                chain.rows[atom][axis],
                chain.columns[atom][axis]
            ]
            .iter()
            .all(|t| t.abs() > 100. * gate),
            "a term of atom {atom} axis {axis} is not above 100 gates"
        );
    }
}

/// One coordinate, the one the frozen Si oracle misses most (atom 1, x), in the default suite:
/// every build re-reads the datasets and the screened tables (about 15 s each).
#[test]
fn the_explicit_exchange_terms_on_the_k_mesh_are_the_displaced_crystal_derivative() {
    check(&[(1, 0)]);
}

#[test]
#[ignore = "three more coordinates of 15 s builds, six per coordinate: 383 s for four measured on 2026-11-02"]
fn the_other_coordinates_of_the_mesh_oracle() {
    check(&[(1, 1), (0, 1), (1, 2)]);
}
