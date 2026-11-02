//! Helpers shared by the `paw_record*` test binaries: library discovery, the
//! frozen A0 references and the dataset array walk in the order of the
//! Python `dataset_arrays` of `test/migration/freeze_paw_generation.py`.
#![allow(dead_code)]

pub mod sha;

use std::path::{Path, PathBuf};

use mandacaru_core::paw::record::PawDataset;
use serde_json::Value;

pub const SETS: [&str; 4] = ["lda-sr", "pbe-sr", "lda-fr", "pbe-fr"];

/// `$MANDACARU_PAW_PATH`; a missing variable FAILS the test, it never skips.
pub fn library() -> PathBuf {
    match std::env::var("MANDACARU_PAW_PATH") {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path),
        _ => panic!(
            "MANDACARU_PAW_PATH is not set: point it at the mandacaru-paw checkout \
             (`mandacaru-paw --set DIR`)"
        ),
    }
}

/// The UPAW-LCAO library: `$MANDACARU_UPAW_PATH`, else the `mandacaru-upaw`
/// checkout beside the PAW one. `None` when neither exists.
pub fn upaw_library() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("MANDACARU_UPAW_PATH") {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    let beside = library().parent()?.join("mandacaru-upaw");
    beside.is_dir().then_some(beside)
}

/// `(set, path)` of every dataset file of a library root, sorted.
pub fn files_of(root: &Path, sets: &[&str]) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    for set in sets {
        let folder = root.join(set);
        let mut names: Vec<PathBuf> = std::fs::read_dir(&folder)
            .unwrap_or_else(|e| panic!("{}: {e}", folder.display()))
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
            .collect();
        names.sort();
        found.extend(names.into_iter().map(|path| (set.to_string(), path)));
    }
    found
}

pub fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/paw_generate")
}

pub fn reference(name: &str) -> Value {
    let path = reference_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("frozen reference {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn sha256_of_f64(values: &[f64]) -> String {
    let mut hash = sha::Sha256::new();
    let mut bytes = Vec::with_capacity(1 << 16);
    for chunk in values.chunks(1 << 13) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        hash.update(&bytes);
    }
    hash.finish()
}

/// Every float array of a record, named and ordered as `dataset_arrays` of
/// the A0 freeze: row-major, one entry per array.
pub fn arrays_of(ds: &PawDataset) -> Vec<(String, Vec<usize>, &[f64])> {
    let mut out: Vec<(String, Vec<usize>, &[f64])> = Vec::new();
    let n = ds.r.as_slice().len();
    for (name, values) in [
        ("r", ds.r.as_slice()),
        ("v_local", ds.v_local.as_slice()),
        ("v_local_screened", ds.v_local_screened.as_slice()),
        ("valence_density", ds.valence_density.as_slice()),
        ("core_density", ds.core_density.as_slice()),
        ("smooth_core_density", ds.smooth_core_density.as_slice()),
    ] {
        assert_eq!(values.len(), n);
        out.push((name.to_string(), vec![n], values));
    }
    for (l, ch) in &ds.channels {
        for i in 0..ch.pseudo_waves.len() {
            for (kind, table) in [
                ("ae_wave", &ch.ae_waves[i]),
                ("pseudo_wave", &ch.pseudo_waves[i]),
                ("projector", &ch.projectors[i]),
                ("raw_projector", &ch.raw_projectors[i]),
            ] {
                out.push((format!("{kind}_l{l}_{i}"), vec![n], table.as_slice()));
            }
            for (kind, a) in [
                ("wavevectors", &ch.wavevectors[i]),
                ("wave_coefficients", &ch.wave_coefficients[i]),
            ] {
                out.push((format!("{kind}_l{l}_{i}"), a.shape().to_vec(), a.data()));
            }
        }
        for (kind, a) in [
            ("coupling", &ch.coupling),
            ("coupling_screened", &ch.coupling_screened),
            ("overlap_correction", &ch.overlap_correction),
            ("norm_correction", &ch.norm_correction),
            ("kinetic_difference", &ch.kinetic_difference),
            ("potential_difference", &ch.potential_difference),
            ("vanderbilt", &ch.vanderbilt),
            ("reference_energies", &ch.reference_energies),
            ("occupations", &ch.occupations),
            ("residual_kinetic", &ch.residual_kinetic),
        ] {
            out.push((format!("{kind}_l{l}"), a.shape().to_vec(), a.data()));
        }
    }
    for (l, so) in &ds.spin_orbit {
        for (k, p) in so.projectors.iter().enumerate() {
            out.push((
                format!("so_projector_l{l}_{k}"),
                vec![p.as_slice().len()],
                p.as_slice(),
            ));
        }
        for (kind, a) in [
            ("average", &so.average),
            ("average_screened", &so.average_screened),
            ("coupling", &so.coupling),
            ("coupling_screened", &so.coupling_screened),
            ("overlap_average", &so.overlap_average),
            ("overlap_coupling", &so.overlap_coupling),
            ("level_errors", &so.level_errors),
            ("reference_energies", &so.reference_energies),
        ] {
            out.push((format!("so_{kind}_l{l}"), a.shape().to_vec(), a.data()));
        }
    }
    out
}

/// The frozen `(name, sha256, shape, max)` of one A0 library sample, or
/// `None` for a file that is not in the sample.
pub fn frozen_arrays(
    set: &str,
    element: &str,
) -> Option<(Value, Vec<(String, String, Value, Value)>)> {
    let path = reference_dir().join(format!("lib_{set}_{element}.json"));
    if !path.exists() {
        return None;
    }
    let doc = reference(&format!("lib_{set}_{element}.json"));
    let names_doc = reference("lib_names.json");
    let id = doc["arrays"]["names_id"].as_str().unwrap();
    let names = names_doc[id].as_array().unwrap();
    let a = &doc["arrays"];
    let rows = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.as_str().unwrap().to_string(),
                a["sha256"][i].as_str().unwrap().to_string(),
                a["shape"][i].clone(),
                a["max"][i].clone(),
            )
        })
        .collect();
    Some((doc, rows))
}

/// The frozen shape is `n` for a vector and a list for a matrix.
pub fn shape_matches(frozen: &Value, shape: &[usize]) -> bool {
    match frozen {
        Value::Number(n) => shape.len() == 1 && n.as_u64() == Some(shape[0] as u64),
        Value::Array(list) => {
            list.len() == shape.len()
                && list
                    .iter()
                    .zip(shape)
                    .all(|(a, b)| a.as_u64() == Some(*b as u64))
        }
        _ => false,
    }
}

/// A scratch folder removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mandacaru-paw-record-{label}-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The path of one library file, `<root>/<set>/<element>.parquet`.
pub fn dataset_path(set: &str, element: &str) -> PathBuf {
    library().join(set).join(format!("{element}.parquet"))
}

/// The context of the tests: two threads, 512 MB.
pub fn context() -> mandacaru_core::Context {
    mandacaru_core::Context::new(Some(2), 512 << 20).unwrap()
}

/// The JSON payload of a table read from a dataset file.
pub fn payload_of(table: &mandacaru_core::parquet::Table) -> Value {
    serde_json::from_str(&table.metadata[mandacaru_core::paw::record::METADATA_KEY]).unwrap()
}

/// A JSON list of numbers (the string `"inf"` is infinity).
pub fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v {
            Value::String(text) if text == "inf" => f64::INFINITY,
            other => other.as_f64().unwrap(),
        })
        .collect()
}
