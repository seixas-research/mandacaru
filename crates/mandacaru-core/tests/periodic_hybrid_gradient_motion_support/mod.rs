//! A DISPLACED hybrid Si crystal built from Rust ingredients: the Bloch samples of the shells
//! (`setup::shells`) and the projections of the projectors (`setup::pulay::projection`) at the
//! moved centers, the moved `sites.centers`, and everything else (the moment blocks, the
//! datasets, the occupied coefficients) of the Python-built phase-5 snapshot, which does not
//! depend on the positions. The energy of such a crystal through `Crystal::hybrid_exchange` is
//! the oracle of the whole explicit chain (HX-1).
#![allow(dead_code)]

use crate::{
    periodic_hybrid_gradient_support::OFF_SITE,
    periodic_screened_augmentation_support::{fixture, Rig},
    setup_gradients_support::context,
    setup_pulay_support::{bloch_values, Fixture},
    setup_support::{library_of, sites},
};
use mandacaru_core::{
    periodic::{
        hybrid_gradient::Motion,
        setup::{
            pulay::{projection, KSamples, Matrix, Mesh},
            shells::Shells,
            spheres::Projector,
        },
        Crystal,
    },
    quantum::C64,
    Buffer, Context, Result,
};
use serde_json::{json, Value};

pub const FIXTURE: &str = "si_crystal.json";

pub struct Si {
    pub ctx: Context,
    pub fx: Fixture,
    pub rig: Rig,
    /// The diamond centers of the phase-5 snapshot (Bohr).
    pub ideal: Vec<[f64; 3]>,
}

impl Si {
    pub fn new() -> Self {
        let ctx = context();
        let rig = Rig::new(FIXTURE, true, &[4]);
        let ideal = rig.geometry.atoms.iter().map(|a| a.center).collect();
        let fx = Fixture::new(&ctx, "si");
        Self {
            ctx,
            fx,
            rig,
            ideal,
        }
    }

    /// The ideal centers, atom 1 off its site by [`OFF_SITE`], then `atom` moved by `delta`.
    pub fn centers(&self, off_site: bool, atom: usize, axis: usize, delta: f64) -> Vec<[f64; 3]> {
        let mut centers = self.ideal.clone();
        if off_site {
            for a in 0..3 {
                centers[1][a] += OFF_SITE[a];
            }
        }
        centers[atom][axis] += delta;
        centers
    }

    pub fn shells(&self, centers: &[[f64; 3]]) -> Shells {
        let shells = self
            .fx
            .shells
            .shells()
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.center = centers[self.fx.owner[s.rows[0]]];
                s
            })
            .collect();
        Shells::new(shells).unwrap()
    }

    pub fn projectors(&self, centers: &[[f64; 3]]) -> Vec<Projector> {
        self.fx
            .projectors
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.center = centers[p.atom];
                p
            })
            .collect()
    }

    /// The Gamma-point samples (`m x N`, function-major) and projections (`m x p`).
    pub fn arrays(&self, centers: &[[f64; 3]]) -> (Vec<C64>, Matrix) {
        let shells = self.shells(centers);
        let mesh = self.fx.mesh(&self.ctx);
        let psi = bloch_values(&self.ctx, &self.fx, &shells, &mesh, [0.; 3]);
        let c = projection(
            &self.ctx,
            &shells,
            &self.fx.lattice,
            [0.; 3],
            &self.projectors(centers),
        )
        .unwrap();
        (psi, c)
    }

    /// The crystal with atoms at `centers`: the snapshot with those arrays and sites replaced.
    pub fn crystal(&self, centers: &[[f64; 3]]) -> Crystal {
        let (psi, c) = self.arrays(centers);
        let mut value: Value = fixture(FIXTURE);
        value["functional"] = json!("hse06");
        value["sites"] = sites(&value, library_of(FIXTURE));
        value["sites"]["centers"] = json!(centers);
        let mut first = value["kpoints"][0].clone();
        first["weight"] = json!(1.0);
        first["psi"] = json!(psi.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>());
        let (m, p) = c.shape();
        first["projections"] = json!((0..m * p)
            .map(|x| [c[(x / p, x % p)].re, c[(x / p, x % p)].im])
            .collect::<Vec<_>>());
        value["kpoints"] = json!([first]);
        value["permutations"] = json!([]);
        Crystal::from_json(
            &Context::new(Some(1), 512 << 20).unwrap(),
            &value.to_string(),
        )
        .unwrap()
    }
}

/// The center derivatives of the shells' Bloch sums at the crystal's own k-points.
pub struct ShellMotion<'a> {
    pub ctx: &'a Context,
    pub fx: &'a Fixture,
    pub shells: &'a Shells,
    pub mesh: &'a Mesh,
}

impl Motion for ShellMotion<'_> {
    fn owner(&self) -> &[usize] {
        &self.fx.owner
    }
    fn center_derivative(&self, c: &Crystal, k: usize) -> Result<[Buffer<C64>; 3]> {
        let samples = KSamples::new(
            self.ctx,
            self.shells,
            &self.fx.lattice,
            self.mesh,
            c.points[k].k,
        )?;
        let mut out = [
            c.context.allocate::<C64>(samples.slope(0).len())?,
            c.context.allocate::<C64>(samples.slope(0).len())?,
            c.context.allocate::<C64>(samples.slope(0).len())?,
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            for (x, y) in to.as_mut_slice().iter_mut().zip(samples.slope(axis)) {
                *x = -y;
            }
        }
        Ok(out)
    }
}
