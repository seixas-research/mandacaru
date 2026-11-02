//! Shared by the `periodic_gradient_crystals*` tests: a REAL converged SCF crystal and the Rust
//! setup built for the same structure.
//!
//! * the crystal is the snapshot `reference/periodic_gradients/<case>.snapshot.json` the Python
//!   setup exported (`freeze_periodic_gradients.py`; a manifest with a `snapshot` entry names the
//!   case whose snapshot it shares), solved by the Rust SCF at the oracle tolerances recorded in
//!   the manifest, with its exact-exchange divergence treatment for a hybrid. The dataset files
//!   of the snapshot's `sites` block are joined to `MANDACARU_PAW_PATH` (the snapshot keeps
//!   library-relative names);
//! * the derivative data come from the Rust setup built from the snapshot's `sites` block (atom
//!   centers, channel labels, dataset files checked by SHA-256), the datasets, and the sidecar
//!   of the manifest (the radial tables and the function list of the basis, which the Python
//!   basis generation still owns: WP-M moves them to Rust). `q` and `D_ion` are assembled here
//!   from the dataset blocks in the raw projector basis (`B^-1 X B^-T`), as the Python does.
//!
//! `setup_consistency` of `periodic_gradient_crystals_setup.rs` proves the two descriptions are
//! the same crystal (matrices, grids, lattice sums, core), so the force is a force of THAT energy.
#![allow(dead_code)]

use mandacaru_core::{
    math::storage::MatrixSource,
    paw::{digest::sha256_file, local_split::grid_spacing, record::PawDataset},
    periodic::{
        gradient::{scf, Bands, Layout, State},
        screened_exchange::ExxDivTreatment,
        setup::{
            basis_gradient::{short_range_cross, ShortRange},
            electrostatics::max_wavenumber,
            ions::LocalPotential,
            provider::{Core, Inputs, Operators, SetupDerivatives},
            pulay::Matrix,
            shapes::{Atom, Channel, Shapes, Sites, COMPENSATION_CUTOFF},
            shells::Shells,
            spheres::Projector,
        },
        Crystal, Options, PeriodicResult,
    },
    Context,
};
use nalgebra::Matrix3;
use serde_json::Value;
use std::path::PathBuf;

/// `SIGMA_FACTOR` of `pseudopotentials/local_split.py`: the Gaussian ion width over the grid's
/// coarsest spacing (checked against the manifest's sigma).
pub const SIGMA_FACTOR: f64 = 1.4;

pub mod altered;
pub mod build;
pub mod gates;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/periodic_gradients")
}

pub fn paw_root() -> PathBuf {
    std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .expect("MANDACARU_PAW_PATH is unset: the gradient tests read the shipped datasets")
}

/// The snapshot of a case with the dataset files of its `sites` block joined to
/// `MANDACARU_PAW_PATH`. A manifest without a `snapshot` entry owns the snapshot of its own name.
pub fn snapshot_of(case: &str, manifest: &Value) -> Value {
    let owner = manifest["snapshot"].as_str().unwrap_or(case);
    let path = root().join(format!("{owner}.snapshot.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut snapshot: Value = serde_json::from_str(&text).unwrap();
    for (atom, d) in snapshot["sites"]["datasets"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let relative = manifest["setup"]["datasets"][atom]["file"]
            .as_str()
            .unwrap();
        d["file"] = Value::String(paw_root().join(relative).to_str().unwrap().into());
    }
    snapshot
}

pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Everything about one frozen case that does not borrow from itself.
pub struct Data {
    pub case: &'static str,
    pub manifest: Value,
    pub crystal: Crystal,
    pub options: Options,
    pub kpoints: Vec<[f64; 3]>,
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
    pub sigma: f64,
    pub datasets: Vec<PawDataset>,
    pub owner: Vec<usize>,
    pub shells: Shells,
    pub projectors: Vec<Projector>,
    pub potentials: Vec<LocalPotential>,
    pub shapes: Shapes,
    pub blocks: Vec<Matrix>,
    pub augmentation: Matrix,
    pub ion_operator: Matrix,
}

impl Data {
    pub fn load(case: &'static str) -> Self {
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(root().join(format!("{case}.json")))
                .unwrap_or_else(|e| panic!("{case}.json: {e}")),
        )
        .unwrap();
        let snapshot = snapshot_of(case, &manifest);
        Self::build(case, manifest, snapshot)
    }

    /// The same, from a manifest and a snapshot already in hand (a snapshot a test rewrote).
    pub fn build(case: &'static str, manifest: Value, snapshot: Value) -> Self {
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        let crystal = Crystal::from_json(&ctx, &snapshot.to_string()).unwrap();
        let scf = &manifest["scf"];
        // The divergence treatment of the oracle: named by every hybrid manifest (a missing key
        // would silently run the default), absent from a semilocal one.
        let exxdiv = match scf["exxdiv_treatment"].as_str() {
            Some(name) => ExxDivTreatment::parse(name).unwrap(),
            None => {
                assert_ne!(
                    manifest["functional"].as_str().unwrap(),
                    "hse06",
                    "a hybrid manifest names the exxdiv treatment of its oracle"
                );
                ExxDivTreatment::None
            }
        };
        let options = Options {
            max_iterations: scf["max_iterations"].as_u64().unwrap() as usize,
            tolerance: scf["tolerance"].as_f64().unwrap(),
            density_tolerance: scf["density_tolerance"].as_f64().unwrap(),
            smearing: mandacaru_core::periodic::spectrum::Smearing::FermiDirac,
            width: scf["width_ha"].as_f64().unwrap(),
            mixing: scf["mixing"].as_f64().unwrap(),
            history: scf["history"].as_u64().unwrap() as usize,
            exxdiv_treatment: exxdiv,
        };
        let kpoints = crystal.points.iter().map(|p| p.k).collect();
        let setup = &manifest["setup"];
        let lattice = Matrix3::from_row_slice(&reals(&setup["lattice_bohr"]["re"]));
        let centers: Vec<[f64; 3]> = snapshot["sites"]["centers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let c = reals(c);
                [c[0], c[1], c[2]]
            })
            .collect();
        let datasets: Vec<PawDataset> = snapshot["sites"]["datasets"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(atom, d)| {
                let file = paw_root().join(setup["datasets"][atom]["file"].as_str().unwrap());
                assert_eq!(
                    sha256_file(&file).unwrap(),
                    d["sha256"].as_str().unwrap(),
                    "{file:?} is not the dataset file the snapshot was built from"
                );
                PawDataset::read(&crystal.context, &file).unwrap()
            })
            .collect();
        let atoms: Vec<Atom> = centers
            .iter()
            .zip(&datasets)
            .map(|(center, d)| Atom {
                center: *center,
                charge: d.valence_charge,
                compensation_radius: d.compensation_radius,
            })
            .collect();
        let channels: Vec<Channel> = snapshot["channel_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c[0].as_u64().unwrap() as usize,
                    c[1].as_u64().unwrap() as usize,
                    c[2].as_i64().unwrap() as i32,
                )
            })
            .collect();
        let mut step = [0.; 9];
        for (i, value) in step.iter_mut().enumerate() {
            *value = crystal.grid.step[(i / 3, i % 3)];
        }
        let sigma = SIGMA_FACTOR * grid_spacing(&step);
        assert_eq!(
            sigma,
            setup["sigma_bohr"].as_f64().unwrap(),
            "the ion width"
        );
        let (shells, owner) = build::basis(&crystal.context, setup);
        let mut projectors = Vec::new();
        for (atom, center) in centers.iter().enumerate() {
            for mut p in build::projector_list(&crystal.context, &datasets[atom], *center) {
                p.atom = atom;
                projectors.push(p);
            }
        }
        let potentials = datasets
            .iter()
            .map(|ds| {
                LocalPotential::new(
                    &crystal.context,
                    ds.r.as_slice(),
                    ds.v_local.as_slice(),
                    ds.valence_charge,
                    ds.r_cut_local,
                )
                .unwrap()
            })
            .collect();
        let sites = Sites {
            atoms: &atoms,
            channels: &channels,
            lattice,
            sigma,
        };
        let q_max = max_wavenumber(&crystal.context, &crystal.grid).unwrap();
        let shapes = Shapes::new(&crystal.context, &sites, q_max).unwrap();
        let p = projectors.len();
        let blocks = (0..crystal.channels)
            .map(|a| Matrix::from_fn(p, p, |i, j| crystal.blocks.as_slice()[(a * p + i) * p + j]))
            .collect();
        let augmentation = build::assemble(&projectors, &datasets, |c| c.overlap_correction.data());
        let ion_operator = build::assemble(&projectors, &datasets, |c| c.coupling.data());
        Self {
            case,
            manifest,
            crystal,
            options,
            kpoints,
            atoms,
            channels,
            lattice,
            sigma,
            datasets,
            owner,
            shells,
            projectors,
            potentials,
            shapes,
            blocks,
            augmentation,
            ion_operator,
        }
    }

    /// The atom centers (Bohr).
    pub fn centers(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|a| a.center).collect()
    }

    pub fn sites(&self) -> Sites<'_> {
        Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: self.sigma,
        }
    }

    pub fn solve(&self) -> PeriodicResult {
        let result = self.crystal.solve(&self.options).unwrap();
        assert!(result.converged, "the oracle SCF must converge");
        result
    }

    /// The converged bands as the engine's state: all `m` bands of every `(spin, k)`, occupations
    /// per spin channel.
    pub fn state(&self, r: &PeriodicResult) -> State {
        let nk = self.crystal.points.len();
        let degeneracy = self.crystal.degeneracy();
        State {
            bands: (0..self.crystal.spins)
                .map(|s| {
                    (0..nk)
                        .map(|k| Bands {
                            coefficients: (*r.coefficients[s * nk + k]
                                .matrix(&self.crystal.context)
                                .unwrap())
                            .clone(),
                            eigenvalues: r.eigenvalues[s * nk + k].clone(),
                            occupations: r.occupations[s * nk + k]
                                .iter()
                                .map(|f| f / degeneracy)
                                .collect(),
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn layout(&self) -> Layout {
        Layout {
            atoms: self.atoms.len(),
            m: self.crystal.m,
            p: self.crystal.p,
            spins: self.crystal.spins,
            degeneracy: self.crystal.degeneracy(),
            weights: self.crystal.points.iter().map(|p| p.weight).collect(),
            row_owner: self.owner.clone(),
            projector_owner: self.projectors.iter().map(|p| p.atom).collect(),
            reference: self.crystal.reference,
            request: scf::request(&self.crystal),
        }
    }

    pub fn operators(&self) -> Operators {
        Operators {
            blocks: self.blocks.clone(),
            augmentation: self.augmentation.clone(),
            ion_operator: self.ion_operator.clone(),
        }
    }

    /// The smooth core tables the datasets were unscreened with, per atom.
    pub fn cores(&self) -> Vec<Option<Core<'_>>> {
        self.datasets
            .iter()
            .map(|ds| {
                let smooth = ds.smooth_core_density.as_slice();
                mandacaru_core::periodic::setup::fields::core_table(
                    ds.nlcc.applied,
                    ds.nlcc.source.as_deref().unwrap_or(""),
                    smooth,
                )
                .unwrap()
                .map(|values| Core {
                    radius: ds.r.as_slice(),
                    values,
                })
            })
            .collect()
    }
}

/// The plain inputs of the geometry of `data`; `cores` must outlive them.
pub fn inputs<'a>(data: &'a Data, cores: &'a [Option<Core<'a>>]) -> Inputs<'a> {
    Inputs {
        grid: &data.crystal.grid,
        shells: &data.shells,
        projectors: &data.projectors,
        kpoints: &data.kpoints,
        sites: data.sites(),
        shapes: &data.shapes,
        potentials: &data.potentials,
        cores,
        dense_cutoff: COMPENSATION_CUTOFF,
    }
}

/// The provider for `data`, tied to its SCF crystal. `cores` must outlive it.
pub fn provider<'a>(data: &'a Data, cores: &'a [Option<Core<'a>>]) -> SetupDerivatives<'a> {
    SetupDerivatives::new(
        &data.crystal.context,
        inputs(data, cores),
        data.layout(),
        data.operators(),
    )
    .unwrap()
    .with_scf(&data.crystal, data.options.clone())
    .unwrap()
}

/// The short-range matrix of every atom's sphere summed, at `k`.
pub fn short_range_total(data: &Data, k: [f64; 3]) -> Matrix {
    let ctx = &data.crystal.context;
    let m = data.crystal.m;
    let mut total = Matrix::zeros(m, m);
    for (atom, a) in data.atoms.iter().enumerate() {
        let sphere = ShortRange {
            center: a.center,
            potential: &data.potentials[atom],
            sigma: data.sigma,
        };
        let block = short_range_cross(
            ctx,
            &data.lattice,
            &[k],
            &sphere,
            &data.shells,
            None,
            &data.shells,
        )
        .unwrap();
        total += Matrix::from_row_slice(m, m, block.as_slice());
    }
    total
}

/// Largest entry of `a - b` over the largest entry of `b`.
pub fn gap(a: &Matrix, b: &Matrix) -> f64 {
    let peak = |m: &Matrix| m.iter().map(|x| x.norm()).fold(0., f64::max);
    peak(&(a - b)) / peak(b).max(1e-300)
}
