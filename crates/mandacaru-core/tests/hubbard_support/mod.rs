//! Loader for the DFT+U crystal fixtures written by
//! `test/migration/export_hubbard_crystal.py`: a small metadata JSON beside a git-ignored
//! `.npz` that holds the arrays the metadata names by key. Only what `np.savez` writes is
//! read (stored entries, optional zip64 extra field, `.npy` version 1 or 2).
#![allow(dead_code)]

use mandacaru_core::{
    math::storage::{ComplexSamples, Precision},
    periodic::input::{InputArray, InputArrays},
    periodic::Crystal,
    quantum::C64,
    Context,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

pub const HARTREE_EV: f64 = 27.211386245988;

pub fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/hubbard")
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}
fn u32_at(b: &[u8], at: usize) -> u64 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as u64
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// `(name, descr, bytes)` of every `.npy` member of an `.npz`.
fn members(bytes: &[u8]) -> Vec<(String, String, Vec<u8>)> {
    let mut out = vec![];
    let mut at = 0;
    while at + 30 <= bytes.len() && u32_at(bytes, at) == 0x0403_4b50 {
        assert_eq!(
            u16_at(bytes, at + 8),
            0,
            "compressed npz members are not read"
        );
        let (mut size, name_len, extra_len) = (
            u32_at(bytes, at + 22),
            u16_at(bytes, at + 26),
            u16_at(bytes, at + 28),
        );
        let name = String::from_utf8(bytes[at + 30..at + 30 + name_len].to_vec()).unwrap();
        let extra = at + 30 + name_len;
        if size == 0xFFFF_FFFF {
            let mut e = extra;
            while e + 4 <= extra + extra_len {
                let (id, len) = (u16_at(bytes, e), u16_at(bytes, e + 2));
                if id == 1 {
                    size = u64_at(bytes, e + 4);
                    break;
                }
                e += 4 + len;
            }
        }
        let start = extra + extra_len;
        let npy = &bytes[start..start + size as usize];
        assert_eq!(&npy[..6], b"\x93NUMPY");
        let (header_len, header_at) = if npy[6] == 1 {
            (u16_at(npy, 8), 10)
        } else {
            (u32_at(npy, 8) as usize, 12)
        };
        let header = std::str::from_utf8(&npy[header_at..header_at + header_len]).unwrap();
        assert!(
            header.contains("'fortran_order': False"),
            "{name}: Fortran order"
        );
        let descr = header
            .split("'descr': '")
            .nth(1)
            .unwrap()
            .split('\'')
            .next()
            .unwrap();
        out.push((
            name.trim_end_matches(".npy").to_string(),
            descr.to_string(),
            npy[header_at + header_len..].to_vec(),
        ));
        at = start + size as usize;
    }
    out
}

/// Loads `<case>.json` and `<case>.npz` into a crystal on `ctx`.
pub fn load(ctx: &Context, case: &str) -> Crystal {
    try_load(ctx, case, |_| {}).unwrap()
}

/// [`load`] after `edit` has changed the metadata (a refusal test changes one key).
pub fn try_load(
    ctx: &Context,
    case: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> mandacaru_core::Result<Crystal> {
    let dir = directory().join("crystals");
    let json = dir.join(format!("{case}.json"));
    let npz = dir.join(format!("{case}.npz"));
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate the fixtures with: python test/migration/export_hubbard_crystal.py",
            json.display()
        )
    });
    let bytes = std::fs::read(&npz).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate the fixtures with: python test/migration/export_hubbard_crystal.py",
            npz.display()
        )
    });
    let mut metadata: serde_json::Value = serde_json::from_str(&text).unwrap();
    edit(&mut metadata);
    let source = metadata.to_string();
    let mut samples: BTreeSet<String> = metadata["kpoints"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|k| k["psi"].as_str().map(str::to_string))
        .collect();
    if let Some(hats) = metadata["hats"].as_str() {
        samples.insert(hats.to_string());
    }
    let mut arrays = InputArrays::default();
    for (name, descr, data) in members(&bytes) {
        let owned = match descr.as_str() {
            "<f8" => {
                let v: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                InputArray::Real(ctx.copy_slice(&v).unwrap())
            }
            "<u8" => {
                let v: Vec<u64> = data
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                InputArray::Indices(ctx.copy_slice(&v).unwrap())
            }
            "<c16" => {
                let v: Vec<C64> = data
                    .chunks_exact(16)
                    .map(|c| {
                        C64::new(
                            f64::from_le_bytes(c[..8].try_into().unwrap()),
                            f64::from_le_bytes(c[8..].try_into().unwrap()),
                        )
                    })
                    .collect();
                if samples.contains(&name) {
                    let precision = if metadata["hats"].as_str() == Some(name.as_str()) {
                        Precision::Float64
                    } else {
                        Precision::parse(metadata["precision"].as_str().unwrap_or("float64"))
                            .unwrap()
                    };
                    InputArray::Samples(ComplexSamples::copy(ctx, &v, precision).unwrap())
                } else {
                    InputArray::Complex(ctx.copy_slice(&v).unwrap())
                }
            }
            other => panic!("{name}: unsupported dtype {other}"),
        };
        arrays.0.insert(name, owned);
    }
    Crystal::from_arrays(ctx, &source, arrays)
}
