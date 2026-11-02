//! What `paw::record` refuses, and that nothing makes it panic: mutated copies
//! of real files, the plain-JSON route and a structural payload fuzz.

mod paw_record_support;

use mandacaru_core::parquet::{self, Codec, Column, Table, Utf8Column, WriteOptions};
use mandacaru_core::paw::record::{PawDataset, METADATA_KEY};
use mandacaru_core::{Context, Error};
use paw_record_support::*;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Refusals: a mutated copy of a real file.
// ---------------------------------------------------------------------------

/// Writes a copy of `source` whose payload and table are edited by `edit`.
fn mutated(
    ctx: &Context,
    dir: &TempDir,
    source: &std::path::Path,
    edit: impl FnOnce(&Context, &mut Table, &mut Value),
) -> std::path::PathBuf {
    let mut table = parquet::read_table(ctx, source).unwrap();
    let mut payload = payload_of(&table);
    edit(ctx, &mut table, &mut payload);
    table
        .metadata
        .insert(METADATA_KEY.into(), payload.to_string());
    let path = dir.file("mutated.parquet");
    let options = WriteOptions {
        codec: Codec::Uncompressed,
        ..WriteOptions::default()
    };
    parquet::write_table(ctx, &path, &table, &options).unwrap();
    path
}

fn refused(ctx: &Context, path: &std::path::Path) -> String {
    let error = PawDataset::read(ctx, path).expect_err("the file must be refused");
    assert!(matches!(error, Error::Invalid(_)), "{error:?}");
    assert_eq!(ctx.memory_used(), 0, "a refusal releases the budget");
    error.to_string()
}

#[test]
fn malformed_datasets_are_refused_with_the_reason() {
    let ctx = context();
    let dir = TempDir::new("refuse");
    let na = dataset_path("lda-fr", "Na");
    let hydrogen = dataset_path("lda-sr", "H");
    type Edit = Box<dyn FnOnce(&Context, &mut Table, &mut Value)>;
    let cases: Vec<(&str, std::path::PathBuf, Edit, &str)> = vec![
        (
            "bare-matrix spin-orbit",
            na.clone(),
            Box::new(|_, _, p| p["spin_orbit"]["1"] = json!([[1.0, 0.0], [0.0, 1.0]])),
            "first-order spin-orbit term",
        ),
        (
            "spin-orbit projectors off the grid",
            na.clone(),
            Box::new(|_, _, p| {
                p["spin_orbit"]["1"]["projectors"][0]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }),
            "not on its 4125-point radial grid",
        ),
        (
            "version 3",
            hydrogen.clone(),
            Box::new(|_, _, p| p["version"] = json!(3)),
            "newer than this build (2)",
        ),
        (
            "version missing",
            hydrogen.clone(),
            Box::new(|_, _, p| {
                p.as_object_mut().unwrap().remove("version");
            }),
            "\"version\" is missing",
        ),
        (
            "unknown family",
            hydrogen.clone(),
            Box::new(|_, _, p| p["family"] = json!("ncpp")),
            "family \"ncpp\" is not readable",
        ),
        (
            "wrong format tag",
            hydrogen.clone(),
            Box::new(|_, _, p| p["format"] = json!("mandacaru-paw")),
            "is not \"mandacaru-pseudopotential\"",
        ),
        (
            "missing norm_correction",
            hydrogen.clone(),
            Box::new(|_, _, p| {
                p["channels"]["0"]
                    .as_object_mut()
                    .unwrap()
                    .remove("norm_correction");
            }),
            "\"norm_correction\" is missing",
        ),
        (
            "missing required scalar",
            hydrogen.clone(),
            Box::new(|_, _, p| {
                p.as_object_mut().unwrap().remove("xc");
            }),
            "\"xc\" is missing",
        ),
        (
            "null where a number belongs",
            hydrogen.clone(),
            Box::new(|_, _, p| p["valence_charge"] = Value::Null),
            "expected a number, found null",
        ),
        (
            "string where a number belongs",
            hydrogen.clone(),
            Box::new(|_, _, p| p["q_cut"] = json!("5")),
            "expected a number, found a string",
        ),
        (
            "unknown top-level key",
            hydrogen.clone(),
            Box::new(|_, _, p| p["ncpp_radius"] = json!(1.0)),
            "\"ncpp_radius\" is not part of the schema",
        ),
        (
            "unknown channel key",
            hydrogen.clone(),
            Box::new(|_, _, p| p["channels"]["0"]["extra"] = json!(1.0)),
            "\"extra\" is not part of the schema",
        ),
        (
            "ragged matrix",
            hydrogen.clone(),
            Box::new(|_, _, p| p["channels"]["0"]["coupling"] = json!([[1.0, 2.0], [3.0]])),
            "ragged",
        ),
        (
            "coupling of the wrong size",
            hydrogen.clone(),
            Box::new(|_, _, p| p["channels"]["0"]["coupling"] = json!([[1.0]])),
            "[1, 1] where [2, 2] was expected",
        ),
        (
            "angular momentum out of range",
            hydrogen.clone(),
            Box::new(|_, _, p| {
                let c = p["channels"]["0"].clone();
                p["channels"]["99"] = c;
            }),
            "not an angular momentum",
        ),
        (
            "a radial table that is text",
            hydrogen.clone(),
            Box::new(|ctx, t, _| {
                let rows = t.num_rows();
                t.columns[1].1 =
                    Column::Utf8(Utf8Column::from_strs(ctx, &vec!["x"; rows]).unwrap());
            }),
            "not a float64 column",
        ),
        (
            "a column that belongs to no channel",
            hydrogen.clone(),
            Box::new(|ctx, t, _| {
                let c = ctx.copy_slice(&vec![1.0; t.num_rows()]).unwrap();
                t.columns.push(("projector_l7_0".into(), Column::Double(c)));
            }),
            "belongs to no channel",
        ),
        (
            "a column missing",
            hydrogen.clone(),
            Box::new(|_, t, _| {
                t.columns.retain(|(n, _)| n != "core_density");
            }),
            "\"core_density\" is missing",
        ),
        (
            "a grid that is not increasing",
            hydrogen.clone(),
            Box::new(|_, t, _| {
                if let Column::Double(r) = &mut t.columns[0].1 {
                    r.as_mut_slice().swap(3, 4);
                }
            }),
            "strictly increasing",
        ),
        (
            "the payload key absent",
            hydrogen.clone(),
            Box::new(|_, t, _| {
                t.metadata.insert("other".into(), "{}".into());
            }),
            "no \"mandacaru.pseudopotential\" metadata",
        ),
    ];
    let count = cases.len();
    for (name, source, edit, expected) in cases {
        let path = if name == "the payload key absent" {
            let mut table = parquet::read_table(&ctx, &source).unwrap();
            table.metadata.clear();
            table.metadata.insert("other".into(), "{}".into());
            let path = dir.file("mutated.parquet");
            let options = WriteOptions {
                codec: Codec::Uncompressed,
                ..WriteOptions::default()
            };
            parquet::write_table(&ctx, &path, &table, &options).unwrap();
            path
        } else {
            mutated(&ctx, &dir, &source, edit)
        };
        let message = refused(&ctx, &path);
        assert!(
            message.contains(expected),
            "{name}: {message:?} lacks {expected:?}"
        );
        println!("refused ({name}): {message}");
    }
    println!("{count} refusals, each with its reason, none leaking budget");
}

#[test]
fn a_plain_json_dataset_reads_like_its_parquet_twin() {
    let ctx = context();
    let dir = TempDir::new("json");
    let source = dataset_path("lda-fr", "Na");
    let table = parquet::read_table(&ctx, &source).unwrap();
    let mut document = payload_of(&table);
    let mut tables = serde_json::Map::new();
    for (name, column) in &table.columns {
        let values: Vec<Value> = column
            .as_f64()
            .unwrap()
            .iter()
            .map(|v| Value::Number(serde_json::Number::from_f64(*v).unwrap()))
            .collect();
        tables.insert(name.clone(), Value::Array(values));
    }
    // A non-finite entry is a string in JSON.
    let last = tables["core_density"].as_array().unwrap().len() - 1;
    tables.get_mut("core_density").unwrap()[last] = json!("inf");
    document["radial_tables"] = Value::Object(tables);
    let path = dir.file("Na.json");
    std::fs::write(&path, document.to_string()).unwrap();
    let from_json = PawDataset::read(&ctx, &path).unwrap();
    let from_parquet = PawDataset::read(&ctx, &source).unwrap();
    let (a, b) = (arrays_of(&from_json), arrays_of(&from_parquet));
    assert_eq!(a.len(), b.len());
    for ((name, shape, x), (_, _, y)) in a.iter().zip(&b) {
        let n = x.len() - usize::from(name == "core_density");
        assert!(
            x[..n]
                .iter()
                .zip(&y[..n])
                .all(|(p, q)| p.to_bits() == q.to_bits()),
            "{name} {shape:?}"
        );
    }
    assert_eq!(
        *from_json.core_density.as_slice().last().unwrap(),
        f64::INFINITY
    );
    assert_eq!(from_json.symbol, "Na");
    drop((from_json, from_parquet, table));
    assert_eq!(ctx.memory_used(), 0);
    // A JSON file without its tables is refused.
    document.as_object_mut().unwrap().remove("radial_tables");
    std::fs::write(&path, document.to_string()).unwrap();
    assert!(refused(&ctx, &path).contains("radial_tables is missing"));
    std::fs::write(&path, "not json").unwrap();
    assert!(refused(&ctx, &path).contains("not valid JSON"));
}

// ---------------------------------------------------------------------------
// Structural fuzz: whatever the payload says, the reader answers, never panics.
// ---------------------------------------------------------------------------

struct Xorshift(u64);

impl Xorshift {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Paths to every node of `value`, looking at no more than 40 items of an array.
fn node_paths(value: &Value, here: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    out.push(here.clone());
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                here.push(key.clone());
                node_paths(child, here, out);
                here.pop();
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate().take(40) {
                here.push(i.to_string());
                node_paths(child, here, out);
                here.pop();
            }
        }
        _ => {}
    }
}

#[test]
fn random_payload_edits_never_panic_and_never_leak_budget() {
    let ctx = context();
    let dir = TempDir::new("fuzz");
    let source = dataset_path("lda-fr", "Na");
    let table = parquet::read_table(&ctx, &source).unwrap();
    let original = payload_of(&table);
    let mut paths = Vec::new();
    node_paths(&original, &mut Vec::new(), &mut paths);
    paths.retain(|p| {
        !p.is_empty()
            && !(p.len() > 3 && p[0] == "spin_orbit" && p[2] == "projectors" && p.len() > 4)
    });
    let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
    let path = dir.file("f.parquet");
    let options = WriteOptions {
        codec: Codec::Uncompressed,
        ..WriteOptions::default()
    };
    let (mut read, mut refused_count) = (0usize, 0usize);
    let mut table = table;
    // The low-level table stays alive through the loop; nothing else may.
    let held = ctx.memory_used();
    for case in 0..1500 {
        let mut payload = original.clone();
        let target = &paths[rng.below(paths.len())];
        let mut node = &mut payload;
        for (i, step) in target.iter().enumerate() {
            let last = i + 1 == target.len();
            let edit = rng.below(7);
            let replacement = match edit {
                0 => Value::Null,
                1 => json!("text"),
                2 => json!(-1),
                3 => json!([]),
                4 => json!(1.0e308),
                5 => json!([[1.0, 2.0], [3.0]]),
                _ => json!({"x": 1}),
            };
            if last {
                match node {
                    Value::Object(map) if edit == 0 && rng.below(2) == 0 => {
                        map.remove(step);
                    }
                    Value::Object(map) => {
                        map.insert(step.clone(), replacement);
                    }
                    Value::Array(items) => {
                        let at: usize = step.parse().unwrap();
                        if edit == 0 {
                            items.remove(at);
                        } else {
                            items[at] = replacement;
                        }
                    }
                    _ => {}
                }
                break;
            }
            node = match node {
                Value::Object(map) => map.get_mut(step).unwrap(),
                Value::Array(items) => &mut items[step.parse::<usize>().unwrap()],
                _ => unreachable!(),
            };
        }
        table
            .metadata
            .insert(METADATA_KEY.into(), payload.to_string());
        parquet::write_table(&ctx, &path, &table, &options).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PawDataset::read(&ctx, &path).map(|d| d.points())
        }));
        match outcome {
            Err(_) => panic!("case {case}: editing {target:?} panicked"),
            Ok(Ok(_)) => read += 1,
            Ok(Err(e)) => {
                assert!(
                    matches!(
                        e,
                        Error::Invalid(_) | Error::Budget { .. } | Error::Overflow
                    ),
                    "case {case} {target:?}: {e:?}"
                );
                refused_count += 1;
            }
        }
        assert_eq!(
            ctx.memory_used(),
            held,
            "case {case}: {target:?} leaked budget"
        );
    }
    println!(
        "payload fuzz: 1500 edits, {read} still read, {refused_count} refused, no panic, no leak"
    );
    assert!(refused_count > 500);
}
