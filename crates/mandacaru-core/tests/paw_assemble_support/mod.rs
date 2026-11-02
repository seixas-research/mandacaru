//! Reader of the A0 freeze of the Python PAW generator
//! (`test/migration/reference/paw_generate/gen_<set>_<element>{.json,_inputs.npz,_data.npz}`,
//! `test/migration/freeze_paw_generation.py`): the tracked JSON carries every recorded function
//! case, the git-ignored `.npz` files its inputs at full resolution and its large outputs
//! decimated at `a[stride - 1::stride]`. Members are deflated, hence [`inflate`].
#![allow(dead_code)]

pub mod call;
mod inflate;

use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/paw_generate")
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

struct Member {
    method: usize,
    compressed: usize,
    size: usize,
    local: usize,
}

/// A `np.savez`/`np.savez_compressed` archive of `<f8` vectors, read lazily member by member.
pub struct Npz {
    bytes: Vec<u8>,
    members: BTreeMap<String, Member>,
}

impl Npz {
    pub fn open(path: &std::path::Path, restore: &str) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| {
            panic!(
                "frozen input {} is missing ({e}); restore it with `{restore}` or copy it back \
                 from test/migration/reference/paw_generate_archive/",
                path.display()
            )
        });
        let end = (0..=bytes.len() - 22)
            .rev()
            .find(|&i| u32_at(&bytes, i) == 0x0605_4b50)
            .expect("zip end of central directory");
        let (entries, mut at) = (u16_at(&bytes, end + 10), u32_at(&bytes, end + 16) as usize);
        let mut members = BTreeMap::new();
        for _ in 0..entries {
            assert_eq!(u32_at(&bytes, at), 0x0201_4b50, "zip central directory");
            let (method, mut compressed, mut size, name_len, extra_len, comment_len) = (
                u16_at(&bytes, at + 10),
                u32_at(&bytes, at + 20),
                u32_at(&bytes, at + 24),
                u16_at(&bytes, at + 28),
                u16_at(&bytes, at + 30),
                u16_at(&bytes, at + 32),
            );
            let mut local = u32_at(&bytes, at + 42);
            let name = String::from_utf8(bytes[at + 46..at + 46 + name_len].to_vec()).unwrap();
            let mut e = at + 46 + name_len;
            while e + 4 <= at + 46 + name_len + extra_len {
                let (id, len) = (u16_at(&bytes, e), u16_at(&bytes, e + 2));
                if id == 1 {
                    // zip64: only the fields that overflowed 32 bits, in this order
                    let mut field = e + 4;
                    for value in [&mut size, &mut compressed, &mut local] {
                        if *value == 0xFFFF_FFFF {
                            *value = u64_at(&bytes, field);
                            field += 8;
                        }
                    }
                }
                e += 4 + len;
            }
            members.insert(
                name.trim_end_matches(".npy").to_string(),
                Member {
                    method,
                    compressed: compressed as usize,
                    size: size as usize,
                    local: local as usize,
                },
            );
            at += 46 + name_len + extra_len + comment_len;
        }
        Self { bytes, members }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.members.contains_key(key)
    }

    pub fn array(&self, key: &str) -> Vec<f64> {
        let m = self
            .members
            .get(key)
            .unwrap_or_else(|| panic!("frozen array {key} is not in the archive"));
        let start =
            m.local + 30 + u16_at(&self.bytes, m.local + 26) + u16_at(&self.bytes, m.local + 28);
        let raw = &self.bytes[start..start + m.compressed];
        let npy = match m.method {
            0 => raw.to_vec(),
            8 => inflate::inflate(raw, m.size),
            other => panic!("zip method {other}"),
        };
        assert_eq!(npy.len(), m.size, "{key}: inflated size");
        assert_eq!(&npy[..6], b"\x93NUMPY");
        let (header_len, header_at) = if npy[6] == 1 {
            (u16_at(&npy, 8), 10)
        } else {
            (u32_at(&npy, 8) as usize, 12)
        };
        let header = std::str::from_utf8(&npy[header_at..header_at + header_len]).unwrap();
        assert!(header.contains("'<f8'") && header.contains("'fortran_order': False"));
        npy[header_at + header_len..]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// An output of a recorded function: whole (small outputs) or decimated.
pub struct Reference {
    pub values: Vec<f64>,
    pub stride: usize,
    pub scale: f64,
    pub whole: bool,
}

impl Reference {
    /// Largest `|ours - frozen|` over the frozen positions, relative to the frozen maximum.
    pub fn deviation(&self, ours: &[f64]) -> f64 {
        let diff = self
            .values
            .iter()
            .enumerate()
            .map(|(k, b)| (ours[(k + 1) * self.stride - 1] - b).abs())
            .fold(0., f64::max);
        diff / self.scale.max(1e-300)
    }
}

/// One frozen dataset: the JSON and the two archives.
pub struct Frozen {
    pub name: String,
    pub doc: Value,
    inputs: Npz,
    outputs: Npz,
}

/// The names (`lda-sr_O`) of every frozen generated dataset in the folder.
pub fn datasets() -> Vec<String> {
    let mut names = std::fs::read_dir(directory())
        .unwrap()
        .filter_map(|e| {
            let file = e.unwrap().file_name().into_string().unwrap();
            let stem = file.strip_prefix("gen_")?.strip_suffix(".json")?;
            Some(stem.to_string())
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}

impl Frozen {
    pub fn open(name: &str) -> Self {
        let base = directory().join(format!("gen_{name}"));
        let restore = format!(
            "python test/migration/freeze_paw_generation.py run gen:{}",
            name.replace('_', ":")
        );
        let doc = serde_json::from_str(
            &std::fs::read_to_string(base.with_extension("json")).expect("frozen JSON"),
        )
        .unwrap();
        let archive = |suffix: &str| {
            Npz::open(
                &PathBuf::from(format!("{}{suffix}", base.display())),
                &restore,
            )
        };
        Self {
            name: name.to_string(),
            doc,
            inputs: archive("_inputs.npz"),
            outputs: archive("_data.npz"),
        }
    }

    /// The recorded cases of one Python function, in the order they were recorded.
    pub fn cases(&self, function: &str) -> Vec<(&String, &Value)> {
        self.doc["function_cases"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| v["function"] == function)
            .collect()
    }

    /// A recorded input: inline numbers or an array of the inputs archive.
    pub fn input(&self, spec: &Value) -> Vec<f64> {
        match spec.get("values") {
            Some(v) => v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect(),
            None => self.inputs.array(spec["blob"].as_str().unwrap()),
        }
    }

    pub fn output(&self, spec: &Value) -> Reference {
        if let Some(v) = spec.get("values") {
            let values = v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>();
            let scale = values.iter().fold(0., |m: f64, x| m.max(x.abs()));
            return Reference {
                values,
                stride: 1,
                scale,
                whole: true,
            };
        }
        let key = format!("o_{}", &spec["sha256"].as_str().unwrap()[..20]);
        let n = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .product::<u64>() as usize;
        Reference {
            values: self.outputs.array(&key),
            stride: n.div_ceil(3000).max(1),
            scale: spec["scale"].as_f64().unwrap(),
            whole: false,
        }
    }
}

/// The floor group of a recorded output (the grouping of the freeze script).
pub fn group_of(name: &str) -> &'static str {
    let base = name.rsplit('.').next().unwrap();
    if [
        "projector",
        "raw_projector",
        "pseudo_wave",
        "smooth",
        "derivatives",
        "e_xc_density",
        "v_xc",
    ]
    .iter()
    .any(|p| base.starts_with(p))
    {
        "tables"
    } else if [
        "duality_error",
        "asymmetry",
        "consistency_error",
        "norm_error",
    ]
    .contains(&base)
    {
        "diagnostics"
    } else if [
        "overlap_correction",
        "norm_correction",
        "kinetic_difference",
        "potential_difference",
        "coupling_screened",
        "vanderbilt",
    ]
    .contains(&base)
    {
        "matrices"
    } else {
        "scalars"
    }
}
