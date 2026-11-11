//! Generation tool, not a gate: re-freezes the reconverged finite-difference oracles of the
//! HSE06 crystal cases (`reference/periodic_gradients/<case>.json`, displaced atoms, and
//! `reference/periodic_strain/<case>.json`, strained cells) on the Rust energy. The oracle stays
//! independent of the analytic gradient and stress: it is the central difference of the converged
//! free energy of fresh SCFs, Richardson extrapolated over the manifest's three steps.
//!
//! Each SCF is a fresh crystal of the live setup (`parse_input_json`, `create_setup`, `build`)
//! from the request the case was set up with (`reference/periodic_requests/<snapshot>.json`, one
//! per frozen snapshot: the recipes, the mesh, the filter wave-vector, the node counts):
//!   * gradient: the same setup request with the displaced atom (fractional coordinates of the
//!     displaced Cartesian position), the ion width again `1.4 x` the coarsest step of the
//!     unchanged cell;
//!   * strain: the unstrained setup with `PeriodicSetup::with_cell((1 + eps) L)`, which holds the
//!     pins of the analytic stress (ion width, filter wave-vector, node counts, shape tables)
//!     and strains the lattice, the atoms, the grid step and the grid origin. THE PINS MATTER:
//!     a strain difference that lets the ion width follow the strained grid (the width is 1.4 x
//!     the coarsest grid step, here the step along the third lattice vector) differs from the
//!     analytic stress by a term proportional to the strain derivative of that step, 4e-6 eV/A^3
//!     in the zz stress of the skew LiH cell.
//!
//! Every SCF energy is appended to `<case>.energies.jsonl` in `MANDACARU_FREEZE_WORK` as it is
//! found, so an interrupted run resumes; the manifest is rewritten (its keys sorted) when all are
//! in. One SCF at a time, one thread.
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_KIND=strain MANDACARU_FREEZE_CASE=lih_hse06 \
//!   MANDACARU_FREEZE_WORK=<scratch dir> \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_hybrid -- --ignored --nocapture
//! ```
//! `KIND` is `gradient` or `strain`; `CASE` the manifest name (`si_hsefr_tr3_b`, ...).
use mandacaru_core::{
    paw::{digest::sha256_file, record::PawDataset},
    periodic::{
        screened_exchange::ExxDivTreatment,
        setup::{create_setup, parse_input_json, PeriodicSetup},
        Options,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write, path::PathBuf, sync::Arc, time::Instant};

fn reference() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference")
}

fn options_of(manifest: &Value) -> Options {
    let scf = &manifest["scf"];
    Options {
        max_iterations: scf["max_iterations"].as_u64().unwrap() as usize,
        tolerance: scf["tolerance"].as_f64().unwrap(),
        density_tolerance: scf["density_tolerance"].as_f64().unwrap(),
        smearing: mandacaru_core::periodic::spectrum::Smearing::FermiDirac,
        width: scf["width_ha"].as_f64().unwrap(),
        mixing: scf["mixing"].as_f64().unwrap(),
        history: scf["history"].as_u64().unwrap() as usize,
        exxdiv_treatment: ExxDivTreatment::parse(scf["exxdiv_treatment"].as_str().unwrap())
            .unwrap(),
        ..Options::default()
    }
}

/// The setup request of a case with its lattice and atoms (Bohr, columns are lattice vectors).
struct Case {
    request: Value,
    lattice: Matrix3<f64>,
    cart: Vec<[f64; 3]>,
    ctx: Context,
    datasets: Vec<Arc<PawDataset>>,
    files: Vec<(String, String)>,
}

impl Case {
    fn load(request: Value) -> Self {
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
                sha256_file(&path).unwrap(),
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
    fn setup(&self, moved: Option<(usize, usize, f64)>) -> PeriodicSetup {
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
fn free_energy(case: &Case, setup: &PeriodicSetup, options: &Options) -> f64 {
    let crystal = setup.build(&case.ctx).unwrap();
    let result = crystal.solve(options).unwrap();
    assert!(
        result.converged,
        "the SCF does not converge to the oracle's tolerances"
    );
    result.free_energy
}

struct Ledger {
    path: PathBuf,
    known: BTreeMap<String, f64>,
}
impl Ledger {
    fn open(path: PathBuf) -> Self {
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
    fn get(&mut self, label: &str, solve: &dyn Fn() -> f64) -> f64 {
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
fn richardson(central: [f64; 3]) -> (f64, f64, f64, f64, f64) {
    let r12 = central[0] + (central[0] - central[1]) / 3.;
    let r24 = central[1] + (central[1] - central[2]) / 3.;
    let rr = r12 + (r12 - r24) / 15.;
    (r12, r24, rr, r12 - r24, rr - r12)
}

fn energies_of(
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

#[test]
#[ignore = "generation tool: minutes to hours of SCFs; needs MANDACARU_FREEZE_KIND, _CASE and _WORK"]
fn refreeze_a_hybrid_crystal_oracle() {
    let (Ok(kind), Ok(name), Ok(work)) = (
        std::env::var("MANDACARU_FREEZE_KIND"),
        std::env::var("MANDACARU_FREEZE_CASE"),
        std::env::var("MANDACARU_FREEZE_WORK"),
    ) else {
        println!("nothing to do: MANDACARU_FREEZE_KIND, _CASE and _WORK are not all set");
        return;
    };
    let folder = match kind.as_str() {
        "gradient" => "periodic_gradients",
        "strain" => "periodic_strain",
        other => panic!("MANDACARU_FREEZE_KIND is gradient or strain, not {other:?}"),
    };
    let path = reference().join(folder).join(format!("{name}.json"));
    let mut manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let owner = manifest["snapshot"].as_str().unwrap_or(&name).to_string();
    let request: Value = serde_json::from_str(
        &std::fs::read_to_string(
            reference()
                .join("periodic_requests")
                .join(format!("{owner}.json")),
        )
        .unwrap(),
    )
    .unwrap();
    let case = Case::load(request);
    let options = options_of(&manifest);
    let mut ledger =
        Ledger::open(PathBuf::from(&work).join(format!("{folder}_{name}.energies.jsonl")));
    let start = Instant::now();
    let old_energy = manifest["free_energy_ha"].as_f64().unwrap();
    let base = case.setup(None);
    let energy = ledger.get("base", &|| free_energy(&case, &base, &options));
    println!("{name}: free energy {old_energy:.12} -> {energy:.12}");
    manifest["free_energy_ha"] = json!(energy);

    if kind == "gradient" {
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
            let (energies, central) =
                energies_of(&mut ledger, &format!("g{atom}{axis}"), &steps, &|h| {
                    free_energy(&case, &case.setup(Some((atom, axis, h))), &options)
                });
            let (r24, r48, rr, floor, extrapolated) = richardson(central);
            entry["central"] = json!(central);
            entry["r24"] = json!(r24);
            entry["r48"] = json!(r48);
            entry["rr"] = json!(rr);
            entry["floor_richardson"] = json!(floor);
            entry["floor_extrapolated"] = json!(extrapolated);
            entry["free_energies_ha"] = energies;
        }
    } else {
        let strained = |eps: Matrix3<f64>| -> f64 {
            let cell = (Matrix3::identity() + eps) * case.lattice;
            let setup = base.with_cell(&case.ctx, &cell).unwrap();
            free_energy(&case, &setup, &options)
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
            let (energies, central) = energies_of(&mut ledger, &format!("e{a}{b}"), &steps, &|h| {
                let mut eps = Matrix3::zeros();
                eps[(a, b)] = h;
                strained(eps)
            });
            let (r12, r24, rr, floor, extrapolated) = richardson(central);
            entry["central"] = json!(central);
            entry["r12"] = json!(r12);
            entry["r24"] = json!(r24);
            entry["rr"] = json!(rr);
            entry["floor_richardson"] = json!(floor);
            entry["floor_extrapolated"] = json!(extrapolated);
            entry["free_energies_ha"] = energies;
        }
        let (energies, central) = energies_of(&mut ledger, "iso", &steps, &|h| {
            strained(Matrix3::identity() * h)
        });
        let (r12, r24, rr, floor, extrapolated) = richardson(central);
        let iso = &mut manifest["isotropic"];
        iso["central"] = json!(central);
        iso["r12"] = json!(r12);
        iso["r24"] = json!(r24);
        iso["rr"] = json!(rr);
        iso["floor_richardson"] = json!(floor);
        iso["floor_extrapolated"] = json!(extrapolated);
        iso["free_energies_ha"] = energies;
    }
    manifest["seconds"] = json!(start.elapsed().as_secs_f64());
    manifest["refrozen_2026_11_10"] = json!(format!(
        "HSE06 smooth-core offset fix: re-frozen on the Rust energy by periodic_freeze_hybrid.rs; free energy {old_energy} -> {energy}"
    ));
    std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).unwrap();
    println!("{name}: written, {:.0} s", start.elapsed().as_secs_f64());
}
