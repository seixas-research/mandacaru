//! Helpers shared by the `paw_record*` test binaries: library discovery, the dataset array walk
//! and a SHA-256.
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

/// Every float array of a record, named by table, channel and projector:
/// row-major, one entry per array.
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

pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    out[n - 1] = stop;
    out
}
