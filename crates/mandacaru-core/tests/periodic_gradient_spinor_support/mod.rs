//! Shared by the `periodic_gradient_spinor*` tests (CS-3): a REAL converged spinor SCF crystal and
//! the Rust setup for it, with the spin-orbit union projectors of the dataset.
//!
//! * `Spinor::load("si_ldafr")`: the snapshot the Python setup exported for the `lda-fr` Si cell
//!   (`freeze_periodic_gradients.py`, whole 2x2x2 mesh, every point time-reversal invariant, so
//!   nothing is halved and the moment symmetrization is the identity), with the Richardson
//!   oracle of the same discretized spinor free energy;
//! * `Spinor::zero_soc_twin(case)`: a scalar frozen case rewritten as a time-reversal spinor
//!   crystal with a zero spin-orbit matrix at every k-point and a zero coupling. A spinor
//!   eigenproblem of zero coupling is two copies of the scalar one, so its force is the scalar
//!   force of the same case (and the scalar oracle is its oracle);
//! * the union list and couplings are built from the `lda-fr` record exactly as the setup
//!   tests do (`setup_shells_support::spin_orbit_list`, here per atom).
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{paw_root, provider, root, snapshot_of, Data};
use mandacaru_core::{
    math::storage::MatrixSource,
    paw::record::PawDataset,
    periodic::setup::{
        provider::{Core, SetupDerivatives},
        shells::Radial,
        spheres::Projector,
        spin_orbit::Coupling,
    },
    periodic::{spinor::component, PeriodicResult},
    Context,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};

pub mod altered;
pub mod oracle;

pub type Couplings = BTreeMap<(usize, usize), Coupling>;

/// The spin-orbit union projectors (every `l` of the record, dual tables) of every atom and the
/// `(atom, l)` couplings; `dataset[atom]` is the record of that atom.
pub fn union_list(
    ctx: &Context,
    datasets: &[&PawDataset],
    centers: &[[f64; 3]],
) -> (Vec<Projector>, Couplings) {
    let (mut out, mut couplings) = (Vec::new(), BTreeMap::new());
    for (atom, center) in centers.iter().enumerate() {
        let ds = datasets[atom];
        for (&l, entry) in &ds.spin_orbit {
            let r_cut = ds.channels[&l].r_cut.max(ds.r_cut_local);
            let radials: Vec<_> = entry
                .projectors
                .iter()
                .map(|t| Arc::new(Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).unwrap()))
                .collect();
            for m in -(l as i32)..=l as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    out.push(Projector {
                        atom,
                        l: l as usize,
                        m,
                        index,
                        center: *center,
                        r_cut,
                        radial: radial.clone(),
                    });
                }
            }
            couplings.insert(
                (atom, l as usize),
                Coupling {
                    radial: entry.coupling.shape()[0],
                    values: entry.coupling.data().to_vec(),
                },
            );
        }
    }
    (out, couplings)
}

pub fn manifest_of(case: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join(format!("{case}.json"))).unwrap())
        .unwrap()
}

/// The direction the initial magnetization of a frozen case is rotated to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// A spinor case: the converged-crystal data and the spin-orbit union list.
pub struct Spinor {
    pub data: Data,
    pub projectors: Vec<Projector>,
    pub couplings: Couplings,
}

impl Spinor {
    pub fn load(case: &'static str) -> Self {
        Self::from_data(Data::load(case))
    }

    fn from_data(data: Data) -> Self {
        assert!(data.crystal.spinor, "{} is a spinor snapshot", data.case);
        let refs: Vec<&PawDataset> = data.datasets.iter().collect();
        let (projectors, couplings) = union_list(&data.crystal.context, &refs, &data.centers());
        Self {
            data,
            projectors,
            couplings,
        }
    }

    /// The scalar frozen case `scalar` as a spinor crystal with no spin-orbit coupling: the
    /// crystal's `h^SO` is zero at every k-point, the couplings are the `lda-fr` ones times zero
    /// (the projections and the operator `B` are still formed, `B = 0`).
    pub fn zero_soc_twin(scalar: &'static str) -> Self {
        let manifest = manifest_of(scalar);
        let mut snapshot = snapshot_of(scalar, &manifest);
        let (m, n) = (
            snapshot["m"].as_u64().unwrap() as usize,
            snapshot["initial"].as_array().unwrap().len(),
        );
        snapshot["spinor"] = json!(true);
        snapshot["time_reversal"] = json!(true);
        snapshot["initial_m"] = json!(vec![0.0; 3 * n]);
        for point in snapshot["kpoints"].as_array_mut().unwrap() {
            point["spin_orbit"] = json!(vec![[0.0, 0.0]; 4 * m * m]);
        }
        let data = Data::build(scalar, manifest, snapshot);
        let ctx = &data.crystal.context;
        // The `lda-fr` Si record is a stand-in for the union projector list on every center (any
        // list gives `B = 0` at zero coupling); its coupling is zeroed.
        let library = PawDataset::read(ctx, &paw_root().join("lda-fr/Si.parquet")).unwrap();
        let refs: Vec<&PawDataset> = data.atoms.iter().map(|_| &library).collect();
        let (projectors, mut couplings) = union_list(ctx, &refs, &data.centers());
        for c in couplings.values_mut() {
            c.values.iter_mut().for_each(|v| *v = 0.0);
        }
        Self {
            data,
            projectors,
            couplings,
        }
    }

    /// The provider tied to the crystal, with the spin-orbit data. `cores` must outlive it.
    pub fn provider<'a>(&'a self, cores: &'a [Option<Core<'a>>]) -> SetupDerivatives<'a> {
        provider(&self.data, cores)
            .with_spin_orbit(&self.projectors, &self.couplings)
            .unwrap()
    }
}

impl Spinor {
    /// The frozen case with its initial magnetization turned to `axis`: the same SCF problem started
    /// from another spin direction. (`initial_m` holds `m_x, m_y, m_z` as three blocks of `n`; the
    /// frozen cases start along `z`.) With `spin_orbit` false the crystal's `h^SO` is zero at every
    /// k-point and the couplings are zero, so the problem is invariant under a global spin rotation.
    pub fn rotated(case: &'static str, axis: Axis, spin_orbit: bool) -> Self {
        let manifest = manifest_of(case);
        let mut snapshot = snapshot_of(case, &manifest);
        let initial = snapshot["initial_m"].as_array().unwrap().clone();
        let n = initial.len() / 3;
        let z = &initial[2 * n..];
        assert!(
            initial[..2 * n].iter().all(|v| v.as_f64() == Some(0.)),
            "{case} starts along z"
        );
        let mut turned = vec![json!(0.0); 3 * n];
        let block = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
        turned[block * n..(block + 1) * n].clone_from_slice(z);
        snapshot["initial_m"] = json!(turned);
        let m = snapshot["m"].as_u64().unwrap() as usize;
        if !spin_orbit {
            for point in snapshot["kpoints"].as_array_mut().unwrap() {
                point["spin_orbit"] = json!(vec![[0.0, 0.0]; 4 * m * m]);
            }
        }
        let mut spinor = Self::from_data(Data::build(case, manifest, snapshot));
        if !spin_orbit {
            for c in spinor.couplings.values_mut() {
                c.values.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        spinor
    }
}

/// The converged magnetic moment `(m_x, m_y, m_z)` in the PAW overlap metric, from the
/// density matrices of `result` (the vector `spinor::magnetic_moment` takes the norm of).
pub fn moment(data: &Data, result: &PeriodicResult) -> [f64; 3] {
    let c = &data.crystal;
    std::array::from_fn(|a| {
        c.points
            .iter()
            .zip(&result.matrices)
            .map(|(point, d)| {
                let d = d.matrix(&c.context).unwrap();
                point.weight * (component(&d, c.m, a + 1) * &point.overlap).trace().re
            })
            .sum()
    })
}
