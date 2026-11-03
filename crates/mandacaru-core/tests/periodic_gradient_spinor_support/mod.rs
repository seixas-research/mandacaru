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

    /// The scalar SPIN-POLARIZED frozen case `scalar` (`spins = 2`) as a MAGNETIC spinor crystal
    /// (`time_reversal = false`: four live fields, the eigenframe functional) with no spin-orbit
    /// coupling, whose initial magnetization points along `direction`. With the coupling off the
    /// energy is invariant under a global spin rotation, so every converged state is the scalar
    /// unrestricted one turned to `direction`: the free energy and the force are the scalar oracle's
    /// whatever the direction (F2b gates).
    pub fn magnetic_twin(scalar: &'static str, direction: [f64; 3]) -> Self {
        let manifest = manifest_of(scalar);
        let mut snapshot = snapshot_of(scalar, &manifest);
        assert_eq!(
            snapshot["spins"].as_u64(),
            Some(2),
            "{scalar} is spin-polarized"
        );
        let (m, n) = (
            snapshot["m"].as_u64().unwrap() as usize,
            snapshot["initial"].as_array().unwrap().len(),
        );
        let scalar_m: Vec<f64> = snapshot["initial_m"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        snapshot["spinor"] = json!(true);
        snapshot["time_reversal"] = json!(false);
        snapshot["spins"] = json!(1);
        snapshot["initial_m"] = json!(direction
            .iter()
            .flat_map(|d| scalar_m.iter().map(move |v| v * d / norm))
            .collect::<Vec<f64>>());
        assert_eq!(snapshot["initial_m"].as_array().unwrap().len(), 3 * n);
        for point in snapshot["kpoints"].as_array_mut().unwrap() {
            point["spin_orbit"] = json!(vec![[0.0, 0.0]; 4 * m * m]);
        }
        let data = Data::build(scalar, manifest, snapshot);
        let ctx = &data.crystal.context;
        // The `lda-fr` Si record is a stand-in for the union projector list on every center (any
        // list gives `B = 0` at zero coupling); its coupling is zeroed, as in `zero_soc_twin`.
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

    /// A time-reversal-symmetric spinor case (`case`) flagged MAGNETIC (`time_reversal = false`) with
    /// a zero initial moment: the four-field branch of the functional and of the spheres runs, the
    /// moment stays zero, and the answer is the time-reversal crystal's.
    pub fn magnetic_flagged(case: &'static str) -> Self {
        let manifest = manifest_of(case);
        let mut snapshot = snapshot_of(case, &manifest);
        let n = snapshot["initial"].as_array().unwrap().len();
        snapshot["time_reversal"] = json!(false);
        snapshot["initial_m"] = json!(vec![0.0; 3 * n]);
        Self::from_data(Data::build(case, manifest, snapshot))
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

/// The length of a vector.
pub fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The converged `moment` is a magnetic state pointing along `direction` (the start of the SCF),
/// asserted before any comparison so that it cannot be vacuous: its norm is above `size`, its unit
/// direction equals the start's to `tolerance`, and every Cartesian component the start has is
/// above `floor` of the unit vector (so the transverse spin blocks are populated).
pub fn assert_direction(
    moment: [f64; 3],
    direction: [f64; 3],
    size: f64,
    floor: f64,
    tolerance: f64,
) {
    let length = norm(moment);
    assert!(length > size, "the moment must survive: {moment:?}");
    let unit = direction.map(|d| d / norm(direction));
    for a in 0..3 {
        let along = moment[a] / length;
        assert!(
            (along - unit[a]).abs() < tolerance,
            "component {a} of the converged direction {along} against the start {}",
            unit[a]
        );
        assert!(
            unit[a] == 0. || along.abs() > floor,
            "component {a} of the direction is below the floor: {along}"
        );
    }
}

/// The largest entry of the difference of two force lists.
pub fn largest_gap(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i] - b[i]).abs()))
        .fold(0f64, f64::max)
}
