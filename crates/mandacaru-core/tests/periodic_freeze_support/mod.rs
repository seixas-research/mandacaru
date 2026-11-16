//! Shared by the freeze tools (`periodic_freeze_hybrid.rs`, `periodic_freeze_snapshot.rs`): the
//! setup request of a frozen crystal, its dataset files joined to `MANDACARU_PAW_PATH`, the
//! displaced setups the finite-difference oracle solves, the resumable ledger of the SCF energies
//! and the Richardson protocol of `PAW_COMPLETION_PLAN.md` 14.4 item 1.
#![allow(dead_code)]

use mandacaru_core::{
    paw::{digest::dataset_content_sha256, record::PawDataset},
    periodic::{
        setup::{create_setup, parse_input_json, PeriodicSetup},
        Options,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write, path::PathBuf, sync::Arc, time::Instant};

pub fn reference() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference")
}
/// The setup request of a case with its lattice and atoms (Bohr, columns are lattice vectors).
pub struct Case {
    pub request: Value,
    pub lattice: Matrix3<f64>,
    pub cart: Vec<[f64; 3]>,
    pub ctx: Context,
    pub datasets: Vec<Arc<PawDataset>>,
    pub files: Vec<(String, String)>,
}

impl Case {
    pub fn load(request: Value) -> Self {
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        let set = request["_dataset_set"].as_str().unwrap().to_string();
        let numbers: Vec<f64> = request["lattice"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let lattice = Matrix3::from_row_slice(&numbers);
        let atoms = request["atoms"].as_array().unwrap();
        let cart = atoms
            .iter()
            .map(|a| {
                let f: Vec<f64> = a["fractional"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                let c = lattice * Vector3::new(f[0], f[1], f[2]);
                [c[0], c[1], c[2]]
            })
            .collect();
        // One file per species, in species order (the setup indexes them by it).
        let root = std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset");
        let mut by_species: BTreeMap<usize, String> = BTreeMap::new();
        for a in atoms {
            by_species.insert(
                a["species"].as_u64().unwrap() as usize,
                a["symbol"].as_str().unwrap().to_string(),
            );
        }
        let mut datasets = Vec::new();
        let mut species_files = Vec::new();
        for symbol in by_species.values() {
            let path = PathBuf::from(&root)
                .join(&set)
                .join(format!("{symbol}.parquet"));
            datasets.push(Arc::new(PawDataset::read(&ctx, &path).unwrap()));
            species_files.push((
                path.to_str().unwrap().to_string(),
                dataset_content_sha256(&ctx, &path).unwrap(),
            ));
        }
        let files = atoms
            .iter()
            .map(|a| species_files[a["species"].as_u64().unwrap() as usize].clone())
            .collect();
        Self {
            request,
            lattice,
            cart,
            ctx,
            datasets,
            files,
        }
    }

    /// The setup of the cell with atom `moved` displaced by `delta` (Bohr) along `axis`.
    pub fn setup(&self, moved: Option<(usize, usize, f64)>) -> PeriodicSetup {
        let mut request = self.request.clone();
        if let Some((atom, axis, delta)) = moved {
            let mut c = self.cart[atom];
            c[axis] += delta;
            let f = self.lattice.try_inverse().unwrap() * Vector3::new(c[0], c[1], c[2]);
            request["atoms"][atom]["fractional"] = json!([f[0], f[1], f[2]]);
        }
        request.as_object_mut().unwrap().remove("_dataset_set");
        let mut input = parse_input_json(&request).unwrap();
        input.dataset_files = Some(self.files.clone());
        create_setup(&self.ctx, input, self.datasets.clone()).unwrap()
    }
}

/// The converged free energy of `setup`'s crystal; a state that does not converge fails the run.
pub fn free_energy(case: &Case, setup: &PeriodicSetup, options: &Options) -> f64 {
    let crystal = setup.build(&case.ctx).unwrap();
    let result = crystal.solve(options).unwrap();
    assert!(
        result.converged,
        "the SCF does not converge to the oracle's tolerances"
    );
    result.free_energy
}

pub struct Ledger {
    pub path: PathBuf,
    pub known: BTreeMap<String, f64>,
}
impl Ledger {
    pub fn open(path: PathBuf) -> Self {
        let mut known = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let v: Value = serde_json::from_str(line).unwrap();
                known.insert(
                    v["label"].as_str().unwrap().into(),
                    v["free_energy"].as_f64().unwrap(),
                );
            }
        }
        Self { path, known }
    }
    pub fn get(&mut self, label: &str, solve: &dyn Fn() -> f64) -> f64 {
        if let Some(v) = self.known.get(label) {
            return *v;
        }
        let start = Instant::now();
        let e = solve();
        println!(
            "{label}: {e:.12} Ha ({:.1} s)",
            start.elapsed().as_secs_f64()
        );
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .unwrap();
        writeln!(file, "{}", json!({"label": label, "free_energy": e})).unwrap();
        self.known.insert(label.into(), e);
        e
    }
}

/// `central`, the two Richardson levels, the extrapolation and the floors of one derivative from
/// the three central differences (finest first): `R(a, b) = c_a + (c_a - c_b) / 3`,
/// `RR = R(1, 2) + (R(1, 2) - R(2, 3)) / 15`.
pub fn richardson(central: [f64; 3]) -> (f64, f64, f64, f64, f64) {
    let r12 = central[0] + (central[0] - central[1]) / 3.;
    let r24 = central[1] + (central[1] - central[2]) / 3.;
    let rr = r12 + (r12 - r24) / 15.;
    (r12, r24, rr, r12 - r24, rr - r12)
}

pub fn energies_of(
    ledger: &mut Ledger,
    prefix: &str,
    steps: &[f64; 3],
    energy_at: &dyn Fn(f64) -> f64,
) -> (Value, [f64; 3]) {
    let mut map = serde_json::Map::new();
    let mut central = [0.; 3];
    for (k, h) in steps.iter().enumerate() {
        let plus = ledger.get(&format!("{prefix}+{k}"), &|| energy_at(*h));
        let minus = ledger.get(&format!("{prefix}-{k}"), &|| energy_at(-*h));
        map.insert(format!("+{k}"), json!(plus));
        map.insert(format!("-{k}"), json!(minus));
        central[k] = (plus - minus) / (2. * h);
    }
    (Value::Object(map), central)
}
