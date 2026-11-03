//! What the diagnostics tests share: the shipped libraries, the dataset-level cases the A0
//! freeze recorded on shipped files, and the NaN-propagating fold of a deviation.
//! The freeze reader (`Npz`) is `paw_assemble_support` (A2).
#![allow(dead_code)]

use crate::paw_assemble_support::{directory, Npz};
use mandacaru_core::paw::{diagnose::view::Smooth, record::PawDataset};
use mandacaru_core::{paw::digest::sha256_file, Context};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The larger of two deviations, with a NaN in either kept: `f64::max` would return the other
/// operand and a NaN deviation would then pass its gate.
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

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

/// The UPAW-LCAO library: `$MANDACARU_UPAW_PATH`, else the checkout beside the PAW one.
pub fn upaw_library() -> PathBuf {
    if let Ok(path) = std::env::var("MANDACARU_UPAW_PATH") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    let beside = library().parent().unwrap().join("mandacaru-upaw");
    assert!(
        beside.is_dir(),
        "the UPAW-LCAO library is not at {}: set MANDACARU_UPAW_PATH",
        beside.display()
    );
    beside
}

pub fn context() -> Context {
    Context::new(Some(2), 768 << 20).unwrap()
}

pub fn dataset_path(set: &str, element: &str) -> PathBuf {
    library().join(set).join(format!("{element}.parquet"))
}

pub fn read(ctx: &Context, path: &Path) -> PawDataset {
    PawDataset::read(ctx, path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn smooth_of(ctx: &Context, ds: &PawDataset) -> Smooth {
    Smooth::scalar(ctx, ds).unwrap()
}

/// One `lib_<set>_<element>.json` of the freeze: the shipped file it was taken on, its recorded
/// dataset-level functions and the archive of their large outputs.
pub struct LibraryCase {
    pub set: String,
    pub element: String,
    pub doc: Value,
    pub data: Npz,
}

impl LibraryCase {
    pub fn open(set: &str, element: &str) -> Self {
        let base = directory().join(format!("lib_{set}_{element}"));
        let doc: Value = serde_json::from_str(
            &std::fs::read_to_string(base.with_extension("json"))
                .unwrap_or_else(|e| panic!("frozen lib_{set}_{element}.json is missing: {e}")),
        )
        .unwrap();
        let restore = "cp test/migration/reference/paw_generate_archive/*.npz test/migration/reference/paw_generate/".to_string();
        let data = Npz::open(Path::new(&format!("{}_data.npz", base.display())), &restore);
        Self {
            set: set.into(),
            element: element.into(),
            doc,
            data,
        }
    }

    /// The shipped file, checked against the digest the freeze took the numbers on: a library
    /// that changed since fails loudly instead of comparing other numbers.
    pub fn file(&self) -> PathBuf {
        let path = dataset_path(&self.set, &self.element);
        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            digest,
            self.doc["file"]["sha256"].as_str().unwrap(),
            "{}: the shipped file is not the one the freeze was taken on",
            path.display()
        );
        path
    }

    pub fn function(&self, name: &str) -> &Value {
        &self.doc["functions"][name]
    }

    /// The gate the freeze recorded for a function (100 x its 1-ulp floor).
    pub fn floor_gate(&self, name: &str) -> f64 {
        self.doc["functions_floor"]["tol_100x"][name]
            .as_f64()
            .unwrap()
    }
}

/// Every `(set, element)` that has recorded dataset-level functions.
pub fn library_cases() -> Vec<(String, String)> {
    let mut names = std::fs::read_dir(directory())
        .unwrap()
        .filter_map(|e| {
            let file = e.unwrap().file_name().into_string().unwrap();
            let stem = file.strip_prefix("lib_")?.strip_suffix(".json")?;
            let (set, element) = stem.split_once('_')?;
            Some((set.to_string(), element.to_string()))
        })
        .collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .filter(|(s, e)| {
            let doc: Value = serde_json::from_str(
                &std::fs::read_to_string(directory().join(format!("lib_{s}_{e}.json"))).unwrap(),
            )
            .unwrap();
            doc.get("functions").is_some()
        })
        .collect()
}

pub fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// One shipped file diagnosed: the stored defects against the recomputed ones.
#[derive(Default)]
pub struct Comparison {
    /// Channels flagged on one side only, and a `residuals` entry (no check produces one).
    pub keys: Vec<String>,
    /// Largest absolute difference (rad, or Hartree for a ghost) of the recorded numbers present
    /// on both sides, by kind.
    pub ghosts: f64,
    pub unprojected: f64,
    pub phases: f64,
    pub s_miss: f64,
}

impl Comparison {
    /// The largest of the four.
    pub fn numbers(&self) -> f64 {
        [self.ghosts, self.unprojected, self.phases, self.s_miss]
            .into_iter()
            .fold(0., nan_max)
    }
}

/// Compare the defects a file stores with the ones diagnosed again from the file's own tables.
pub fn compare_defects(
    stored: &mandacaru_core::paw::record::Defects,
    got: &mandacaru_core::paw::record::Defects,
) -> Comparison {
    use std::collections::BTreeMap;
    let mut out = Comparison::default();
    let mut map = |name: &str, a: &BTreeMap<u32, f64>, b: &BTreeMap<u32, f64>, worst: &mut f64| {
        for l in a.keys().chain(b.keys()) {
            match (a.get(l), b.get(l)) {
                (Some(x), Some(y)) => *worst = nan_max(*worst, (x - y).abs()),
                (Some(x), None) => out
                    .keys
                    .push(format!("{name} l={l}: stored {x:+.4}, not found")),
                (None, Some(y)) => out
                    .keys
                    .push(format!("{name} l={l}: found {y:+.4}, not stored")),
                (None, None) => unreachable!(),
            }
        }
    };
    let (mut ghosts, mut unprojected, mut phases, mut s_miss) = (0f64, 0f64, 0f64, 0f64);
    map("ghost", &stored.ghosts, &got.ghosts, &mut ghosts);
    map(
        "unprojected",
        &stored.unprojected,
        &got.unprojected,
        &mut unprojected,
    );
    let side = |d: &BTreeMap<u32, [f64; 2]>, i: usize| d.iter().map(|(l, v)| (*l, v[i])).collect();
    map(
        "phase near",
        &side(&stored.phases, 0),
        &side(&got.phases, 0),
        &mut phases,
    );
    map(
        "phase far",
        &side(&stored.phases, 1),
        &side(&got.phases, 1),
        &mut phases,
    );
    match (stored.s_miss, got.s_miss) {
        (Some(x), Some(y)) => s_miss = nan_max(s_miss, (x - y).abs()),
        (Some(x), None) => out.keys.push(format!("s_miss: stored {x:.4}, not found")),
        (None, Some(y)) => out.keys.push(format!("s_miss: found {y:.4}, not stored")),
        (None, None) => {}
    }
    if !stored.residuals.is_empty() {
        out.keys
            .push("residuals are stored; no check produces them".into());
    }
    out.ghosts = ghosts;
    out.unprojected = unprojected;
    out.phases = phases;
    out.s_miss = s_miss;
    out
}

/// The lowest `count` eigenvalues of a channel's generalized problem on the interior nodes
/// `nodes`, by a dense symmetric-definite solve (Cholesky of the metric, then a Jacobi-class
/// symmetric eigensolver): an independent oracle of the separable solver the spectrum uses.
pub fn dense_levels(
    ctx: &Context,
    smooth: &Smooth,
    channel: &mandacaru_core::paw::diagnose::view::Channel,
    nodes: &[f64],
    count: usize,
) -> Vec<f64> {
    use nalgebra::DMatrix;
    let prepared = mandacaru_core::paw::diagnose::spectrum::prepared(ctx, smooth, channel).unwrap();
    let (v, p) = prepared.operator_on(ctx, nodes).unwrap();
    let (n, rank) = (nodes.len(), channel.rank);
    let h = nodes[0];
    let u = DMatrix::from_fn(n, rank, |i, j| h.sqrt() * p.as_slice()[j * n + i]);
    let d = DMatrix::from_fn(rank, rank, |i, j| channel.coupling[i * rank + j]);
    let q = DMatrix::from_fn(rank, rank, |i, j| channel.overlap[i * rank + j]);
    let mut hamiltonian = &u * &d * u.transpose();
    let mut metric = &u * &q * u.transpose();
    for i in 0..n {
        hamiltonian[(i, i)] += 1. / (h * h) + v.as_slice()[i];
        metric[(i, i)] += 1.;
        if i + 1 < n {
            hamiltonian[(i, i + 1)] += -0.5 / (h * h);
            hamiltonian[(i + 1, i)] += -0.5 / (h * h);
        }
    }
    let cholesky = metric.cholesky().expect("the metric is positive definite");
    let l = cholesky.l();
    let inverse = l.clone().try_inverse().unwrap();
    let standard = &inverse * hamiltonian * inverse.transpose();
    let mut levels = standard.symmetric_eigen().eigenvalues.as_slice().to_vec();
    levels.sort_by(f64::total_cmp);
    levels.truncate(count);
    levels
}

/// A copy of a channel with its `k`-th projector removed (its row and column of `D`, `q` and
/// the norm correction, its table).
pub fn without_projector(
    ctx: &Context,
    channel: &mandacaru_core::paw::diagnose::view::Channel,
    k: usize,
    n: usize,
) -> mandacaru_core::paw::diagnose::view::Channel {
    use mandacaru_core::paw::diagnose::view::Channel;
    let rank = channel.rank;
    let keep = (0..rank).filter(|&i| i != k).collect::<Vec<_>>();
    let matrix = |m: &[f64]| {
        keep.iter()
            .flat_map(|&i| keep.iter().map(move |&j| m[i * rank + j]))
            .collect::<Vec<_>>()
    };
    let mut projectors = ctx.allocate::<f64>(keep.len() * n).unwrap();
    for (row, &i) in keep.iter().enumerate() {
        projectors.as_mut_slice()[row * n..(row + 1) * n]
            .copy_from_slice(&channel.projectors.as_slice()[i * n..(i + 1) * n]);
    }
    let copy = |waves: &[mandacaru_core::Buffer<f64>]| {
        waves
            .iter()
            .map(|w| ctx.copy_slice(w.as_slice()).unwrap())
            .collect::<Vec<_>>()
    };
    Channel {
        l: channel.l,
        kappa: channel.kappa,
        r_cut: channel.r_cut,
        energies: channel.energies.clone(),
        occupations: channel.occupations.clone(),
        rank: keep.len(),
        projectors,
        coupling: matrix(&channel.coupling),
        overlap: matrix(&channel.overlap),
        norm: matrix(&channel.norm),
        pseudo_waves: copy(&channel.pseudo_waves),
        ae_waves: copy(&channel.ae_waves),
    }
}

/// A copy of `smooth` on every `factor`-th node of its table (`r[factor - 1]`, `r[2 factor - 1]`,
/// ...): a table that is `factor` times coarser, to measure how a quantity depends on the stride
/// the shipped files were written at.
pub fn decimated(ctx: &Context, smooth: &Smooth, factor: usize) -> Smooth {
    use mandacaru_core::paw::diagnose::view::Channel;
    let pick = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .skip(factor - 1)
            .step_by(factor)
            .copied()
            .collect()
    };
    let buffer = |values: &[f64]| ctx.copy_slice(&pick(values)).unwrap();
    let n = smooth.r.as_slice().len();
    let channels = smooth
        .channels
        .iter()
        .map(|(key, c)| {
            let rows = c
                .projectors
                .as_slice()
                .chunks(n)
                .map(|row| pick(row))
                .collect::<Vec<_>>();
            let flat = rows.concat();
            (
                *key,
                Channel {
                    l: c.l,
                    kappa: c.kappa,
                    r_cut: c.r_cut,
                    energies: c.energies.clone(),
                    occupations: c.occupations.clone(),
                    rank: c.rank,
                    projectors: ctx.copy_slice(&flat).unwrap(),
                    coupling: c.coupling.clone(),
                    overlap: c.overlap.clone(),
                    norm: c.norm.clone(),
                    pseudo_waves: c
                        .pseudo_waves
                        .iter()
                        .map(|w| buffer(w.as_slice()))
                        .collect(),
                    ae_waves: c.ae_waves.iter().map(|w| buffer(w.as_slice())).collect(),
                },
            )
        })
        .collect();
    Smooth {
        symbol: smooth.symbol.clone(),
        atomic_number: smooth.atomic_number,
        r: buffer(smooth.r.as_slice()),
        v_local_screened: buffer(smooth.v_local_screened.as_slice()),
        r_cut_local: smooth.r_cut_local,
        channels,
        frozen: smooth.frozen.clone(),
        semicore: smooth.semicore.clone(),
    }
}
