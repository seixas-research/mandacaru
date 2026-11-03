//! The displaced-crystal oracle of the explicit exchange terms on a k MESH, shared by the hybrid
//! mesh tests (`periodic_gradient_hybrid_mesh.rs`, `periodic_gradient_spinor_hybrid_mesh.rs`): the
//! crystal of a frozen case rebuilt with one atom moved (Bloch samples and projections of every
//! k-point from the moved shells, the sphere centers moved, the rest of the snapshot kept), and the
//! central-difference derivative of `Crystal::hybrid_exchange` at FIXED density matrices.
#![allow(dead_code)]

use super::{snapshot_of, Data};
use mandacaru_core::{
    periodic::{
        hybrid_gradient::Motion,
        setup::{
            pulay::{projection, KSamples, Mesh},
            shells::{DenseRows, Sampling, Shells},
            spheres::Projector,
        },
        Crystal,
    },
    quantum::C64,
    Buffer, Result,
};
use serde_json::{json, Value};

/// The center derivatives of the Bloch sums of the shells at the crystal's own k-points.
pub struct ShellMotion<'a> {
    pub data: &'a Data,
    pub mesh: &'a Mesh,
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
pub fn moved(data: &Data, atom: usize, axis: usize, delta: f64) -> (Shells, Vec<Projector>) {
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
pub fn displaced(data: &Data, mesh: &Mesh, atom: usize, axis: usize, delta: f64) -> Crystal {
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
pub fn finite_difference(
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
