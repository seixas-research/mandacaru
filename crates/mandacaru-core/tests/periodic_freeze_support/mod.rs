//! Shared by the freeze tools (`periodic_freeze_hybrid.rs`, `periodic_freeze_snapshot.rs`,
//! `periodic_freeze_symmetry.rs`, `periodic_freeze_hybrid_spinor.rs`): the setup request of a
//! frozen crystal (`reference/periodic_requests/`: `<case>.json` for a force case,
//! `strain_<case>.json` for a stress case, `symmetry_<case>_<full|reduced>.json`,
//! `hybrid_spinor_<case>.json`), its dataset files joined to `MANDACARU_PAW_PATH`, the displaced
//! and strained setups the finite-difference oracle solves, the resumable ledger of the SCF
//! energies and the Richardson protocol of `PAW_COMPLETION_PLAN.md` 14.4 item 1.
#![allow(dead_code)]

use mandacaru_core::{
    paw::{digest::dataset_content_sha256, record::PawDataset},
    periodic::{
        screened_exchange::ExxDivTreatment,
        setup::{create_setup, parse_input_json, PeriodicSetup},
        spectrum::{mesh, Smearing},
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
/// The captured setup request `reference/periodic_requests/<file>.json`.
pub fn request_of(file: &str) -> Value {
    let path = reference()
        .join("periodic_requests")
        .join(format!("{file}.json"));
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
    )
    .unwrap()
}

/// The SCF options of a manifest's `scf` object (the oracle's tolerances); a hybrid manifest
/// names the exact-exchange divergence treatment of its oracle.
pub fn options_of(manifest: &Value) -> Options {
    let scf = &manifest["scf"];
    let hybrid = manifest["functional"].as_str() == Some("hse06");
    assert_eq!(
        hybrid,
        scf["exxdiv_treatment"].is_string(),
        "a hybrid manifest names the exxdiv treatment of its oracle, a semilocal one does not"
    );
    Options {
        max_iterations: scf["max_iterations"].as_u64().unwrap() as usize,
        tolerance: scf["tolerance"].as_f64().unwrap(),
        density_tolerance: scf["density_tolerance"].as_f64().unwrap(),
        smearing: Smearing::FermiDirac,
        width: scf["width_ha"].as_f64().unwrap(),
        mixing: scf["mixing"].as_f64().unwrap(),
        history: scf["history"].as_u64().unwrap() as usize,
        exxdiv_treatment: scf["exxdiv_treatment"]
            .as_str()
            .map_or(ExxDivTreatment::None, |name| {
                ExxDivTreatment::parse(name).unwrap()
            }),
        ..Options::default()
    }
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
    pub fn load(mut request: Value) -> Self {
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        // A case whose frozen crystal keeps every point of a time-reversal mesh (`_whole_mesh`):
        // the setup halves a named mesh, so the whole one is listed.
        if request
            .as_object_mut()
            .unwrap()
            .remove("_whole_mesh")
            .is_some_and(|flag| flag.as_bool() == Some(true))
        {
            let size: Vec<usize> = request["mesh"]["size"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            let gamma = request["mesh"]["gamma"].as_bool().unwrap();
            let whole = mesh(&ctx, [size[0], size[1], size[2]], gamma, &[], false).unwrap();
            request.as_object_mut().unwrap().remove("mesh");
            request["kpoints"] = Value::Array(
                whole
                    .points
                    .iter()
                    .zip(&whole.weights)
                    .map(|(k, w)| json!({"fractional": k, "weight": w}))
                    .collect(),
            );
        }
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
        self.build(request)
    }

    fn build(&self, mut request: Value) -> PeriodicSetup {
        request.as_object_mut().unwrap().remove("_dataset_set");
        let mut input = parse_input_json(&request).unwrap();
        input.dataset_files = Some(self.files.clone());
        create_setup(&self.ctx, input, self.datasets.clone()).unwrap()
    }

    /// The setup of the cell strained by `eps` (`x' = (1 + eps) x`, fractional coordinates kept).
    /// With no pin released it is `base.with_cell`, which holds every pin of the analytic stress
    /// (node counts, ion width, filter wave-vector, dense list). A released pin (`"sigma"`,
    /// `"shape"` or `"dense"`, the negative controls of the oracle) is a fresh setup of the strained
    /// request that lets that one discretization follow the strained cell: the ion width from the
    /// strained grid (`"sigma"`), the node counts `round(|a_m| / h)` of the strained vectors at the
    /// case's grid spacing `spacing_angstrom` (`"shape"`), the dense list enumerated again
    /// (`"dense"`, which every fresh setup does).
    pub fn strained(
        &self,
        base: &PeriodicSetup,
        eps: Matrix3<f64>,
        released: &[String],
        spacing_angstrom: f64,
    ) -> PeriodicSetup {
        let cell = (Matrix3::identity() + eps) * self.lattice;
        if released.is_empty() {
            return base.with_cell(&self.ctx, &cell).unwrap();
        }
        let mut request = self.request.clone();
        let rows: Vec<f64> = (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| cell[(r, c)])
            .collect();
        request["lattice"] = json!(rows);
        if released.iter().any(|pin| pin == "shape") {
            let h = spacing_angstrom / mandacaru_core::units::BOHR_TO_ANGSTROM;
            let counts: Vec<usize> = (0..3)
                .map(|m| ((cell.column(m).norm() / h).round() as usize).max(1))
                .collect();
            request["grid_shape"] = json!(counts);
        }
        if !released.iter().any(|pin| pin == "sigma") {
            request["sigma"] = json!(base.pins.sigma);
        }
        self.build(request)
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

/// Whether `floor` is above `limit` (the plan's gate of the coordinate): such a coordinate cannot
/// be gated, so the freeze reports it and the step family is halved.
fn flag(floor: f64, limit: f64) -> &'static str {
    if floor.abs() > limit {
        "  ABOVE THE PLAN GATE"
    } else {
        ""
    }
}

/// The displaced-atom oracle of every `coordinates` entry of a force manifest: the free energies
/// of fresh SCFs at the manifest's `steps_bohr` (finest first), their central differences and the
/// Richardson levels. `limit` is the plan's gate (Ha/Bohr) the floors are reported against.
pub fn gradient_oracle(
    manifest: &mut Value,
    case: &Case,
    options: &Options,
    ledger: &mut Ledger,
    limit: f64,
) {
    let steps_bohr: Vec<f64> = manifest["steps_bohr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let steps = [steps_bohr[0], steps_bohr[1], steps_bohr[2]];
    for entry in manifest["coordinates"].as_array_mut().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap() as usize,
            entry["axis"].as_u64().unwrap() as usize,
        );
        let (energies, central) = energies_of(ledger, &format!("g{atom}{axis}"), &steps, &|h| {
            free_energy(case, &case.setup(Some((atom, axis, h))), options)
        });
        let (r24, r48, rr, floor, extrapolated) = richardson(central);
        println!(
            "atom {atom} axis {axis}: rr {:+.9e} -> {rr:+.9e}, floors {:.2e} -> {:.2e}, {:.2e} -> {:.2e}{}",
            entry["rr"].as_f64().unwrap_or(f64::NAN),
            entry["floor_richardson"].as_f64().unwrap_or(f64::NAN).abs(),
            floor.abs(),
            entry["floor_extrapolated"].as_f64().unwrap_or(f64::NAN).abs(),
            extrapolated.abs(),
            flag(floor, limit)
        );
        entry["central"] = json!(central);
        entry["r24"] = json!(r24);
        entry["r48"] = json!(r48);
        entry["rr"] = json!(rr);
        entry["floor_richardson"] = json!(floor);
        entry["floor_extrapolated"] = json!(extrapolated);
        entry["free_energies_ha"] = energies;
    }
}

/// The strain oracle of a stress manifest: `dF/d eps_ab` of the nine entries and of the isotropic
/// series, Richardson over the manifest's `steps` (finest first), with the pins the manifest
/// releases (`pins.unpinned`). `limit(a, b)` is the plan's gate of an entry (Ha), `trace_limit`
/// that of the isotropic series.
pub fn strain_oracle(
    manifest: &mut Value,
    case: &Case,
    base: &PeriodicSetup,
    options: &Options,
    ledger: &mut Ledger,
    limit: &dyn Fn(usize, usize) -> f64,
    trace_limit: f64,
) {
    let released: Vec<String> = manifest["pins"]["unpinned"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pin| pin.as_str().unwrap().to_string())
        .collect();
    let spacing = manifest["grid_spacing_A"].as_f64().unwrap();
    let strained = |eps: Matrix3<f64>| -> f64 {
        free_energy(case, &case.strained(base, eps, &released, spacing), options)
    };
    let steps_f: Vec<f64> = manifest["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let steps = [steps_f[0], steps_f[1], steps_f[2]];
    for entry in manifest["entries"].as_array_mut().unwrap() {
        let (a, b) = (
            entry["a"].as_u64().unwrap() as usize,
            entry["b"].as_u64().unwrap() as usize,
        );
        let (energies, central) = energies_of(ledger, &format!("e{a}{b}"), &steps, &|h| {
            let mut eps = Matrix3::zeros();
            eps[(a, b)] = h;
            strained(eps)
        });
        let (r12, r24, rr, floor, extrapolated) = richardson(central);
        println!(
            "eps {a}{b}: rr {:+.9e} -> {rr:+.9e}, floors {:.2e} -> {:.2e}, {:.2e} -> {:.2e}{}",
            entry["rr"].as_f64().unwrap_or(f64::NAN),
            entry["floor_richardson"].as_f64().unwrap_or(f64::NAN).abs(),
            floor.abs(),
            entry["floor_extrapolated"]
                .as_f64()
                .unwrap_or(f64::NAN)
                .abs(),
            extrapolated.abs(),
            flag(floor, limit(a, b))
        );
        entry["central"] = json!(central);
        entry["r12"] = json!(r12);
        entry["r24"] = json!(r24);
        entry["rr"] = json!(rr);
        entry["floor_richardson"] = json!(floor);
        entry["floor_extrapolated"] = json!(extrapolated);
        entry["free_energies_ha"] = energies;
    }
    let (energies, central) = energies_of(ledger, "iso", &steps, &|h| {
        strained(Matrix3::identity() * h)
    });
    let (r12, r24, rr, floor, extrapolated) = richardson(central);
    println!(
        "isotropic: rr {rr:+.9e}, floors {:.2e}, {:.2e}{}",
        floor.abs(),
        extrapolated.abs(),
        flag(floor, trace_limit)
    );
    let iso = &mut manifest["isotropic"];
    iso["central"] = json!(central);
    iso["r12"] = json!(r12);
    iso["r24"] = json!(r24);
    iso["rr"] = json!(rr);
    iso["floor_richardson"] = json!(floor);
    iso["floor_extrapolated"] = json!(extrapolated);
    iso["free_energies_ha"] = energies;
}
