//! Generation tool, not a gate: re-exports the DFT+U crystal fixtures of
//! `reference/hubbard/crystals/` (read by `periodic_hubbard.rs` through `hubbard_support`)
//! against the installed PAW library (`MANDACARU_PAW_PATH`), with the Rust setup alone.
//!
//! Each case is the setup of its request (`reference/periodic_requests/hubbard_<case>.json`,
//! captured from the calculator's own request with `hubbard_u` at U = 0, so the setup writes the
//! shell table the tests set U on), built (`create_setup`, `build`) and saved. The saved crystal
//! is then split as the fixtures have always been: a small metadata `<case>.json` and a
//! git-ignored `<case>.npz` (stored members, C order) holding every array the metadata names by
//! key. The keys and dtypes are those of the fixture on disk (or of `MANDACARU_FREEZE_AGAINST`, a
//! directory holding the old pair), so the layout does not change; a large array the template does
//! not name fails the run. The `setup` and `sites` blocks of the saved crystal are left out, as
//! the fixtures never carried them; the `provenance` names this tool, the date and the library
//! digests, and the `fixture` block keeps the case's settings.
//!
//! ```text
//! MANDACARU_PAW_PATH=... MANDACARU_FREEZE_WORK=<scratch dir> \
//!   [MANDACARU_FREEZE_CASES=nio_afm2,ni_atom_7,cuh_7,cuh_7_spin] \
//!   [MANDACARU_FREEZE_OUT=<dir, default reference/hubbard/crystals>] [MANDACARU_FREEZE_AGAINST=<dir>] \
//!   cargo test --profile release-test -p mandacaru-core --test periodic_freeze_hubbard -- --ignored --nocapture
//! ```
mod hubbard_support;
mod periodic_freeze_support;

use hubbard_support::{members, try_load_in};
use mandacaru_core::{
    parquet::crc32::crc32,
    paw::digest::{dataset_content_sha256, sha256},
    Context,
};
use periodic_freeze_support::{reference, Case};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf};

const CASES: [&str; 4] = ["nio_afm2", "ni_atom_7", "cuh_7", "cuh_7_spin"];
/// An inline array longer than this that the template keeps in the npz is a layout change.
const INLINE_LIMIT: usize = 64;

/// Every number of `v`, flattened in document order (a complex value is its `[re, im]` pair).
fn numbers(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(a) => a.iter().for_each(|x| numbers(x, out)),
        _ => {}
    }
}

/// One `.npy` (version 1) of a flat array of `descr`.
fn npy(descr: &str, values: &[f64]) -> Vec<u8> {
    let (count, bytes): (usize, Vec<u8>) = match descr {
        "<f8" => (
            values.len(),
            values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        "<c16" => (
            values.len() / 2,
            values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        "<u8" => (
            values.len(),
            values
                .iter()
                .flat_map(|x| {
                    assert!(x.fract() == 0. && *x >= 0., "{x} is not an index");
                    (*x as u64).to_le_bytes()
                })
                .collect(),
        ),
        other => panic!("unsupported dtype {other}"),
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({count},), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(bytes);
    out
}

/// A zip of stored members (what `np.savez` writes, without zip64: every member is under 4 GiB).
fn npz(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for (name, data) in entries {
        let name = format!("{name}.npy");
        let (offset, crc, size) = (out.len() as u32, crc32(data), data.len() as u32);
        let common = |b: &mut Vec<u8>| {
            b.extend(20u16.to_le_bytes()); // version needed
            b.extend(0u16.to_le_bytes()); // flags
            b.extend(0u16.to_le_bytes()); // stored
            b.extend(0u16.to_le_bytes()); // time
            b.extend(0x21u16.to_le_bytes()); // date 1980-01-01
            b.extend(crc.to_le_bytes());
            b.extend(size.to_le_bytes());
            b.extend(size.to_le_bytes());
            b.extend((name.len() as u16).to_le_bytes());
            b.extend(0u16.to_le_bytes()); // extra
        };
        out.extend(0x0403_4b50u32.to_le_bytes());
        common(&mut out);
        out.extend(name.as_bytes());
        out.extend(data);
        central.extend(0x0201_4b50u32.to_le_bytes());
        central.extend(20u16.to_le_bytes()); // version made by
        common(&mut central);
        central.extend([0u8; 8]); // comment length, disk, internal attributes
        central.extend(0u32.to_le_bytes()); // external attributes
        central.extend(offset.to_le_bytes());
        central.extend(name.as_bytes());
    }
    let (start, length) = (out.len() as u32, central.len() as u32);
    out.extend(central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend([0u8; 4]);
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend(length.to_le_bytes());
    out.extend(start.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

/// Moves every array the template names by key from `new` into `arrays` (dtype of the old member),
/// leaving the key in its place.
fn split(
    template: &Value,
    new: &mut Value,
    dtypes: &BTreeMap<String, String>,
    arrays: &mut Vec<(String, Vec<u8>)>,
    path: &str,
) {
    match (template, &mut *new) {
        (Value::String(key), value) if dtypes.contains_key(key) => {
            let mut values = Vec::new();
            numbers(value, &mut values);
            arrays.push((key.clone(), npy(&dtypes[key], &values)));
            *value = json!(key);
        }
        (Value::Object(t), Value::Object(n)) => {
            for (k, v) in t {
                match n.get_mut(k) {
                    Some(x) => split(v, x, dtypes, arrays, &format!("{path}/{k}")),
                    None if k == "fixture" || k == "provenance" => {}
                    None => panic!("{path}/{k}: in the old fixture, not in the saved crystal"),
                }
            }
        }
        (Value::Array(t), Value::Array(n)) if t.len() == n.len() => {
            for (i, (v, x)) in t.iter().zip(n.iter_mut()).enumerate() {
                split(v, x, dtypes, arrays, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

/// Fails on an inline array larger than [`INLINE_LIMIT`] numbers where the template has none
/// (a large array the old fixture did not carry inline is a layout change).
fn check_inline(template: &Value, v: &Value, path: &str) {
    match v {
        Value::Object(o) => o
            .iter()
            .filter(|(k, _)| *k != "hubbard")
            .for_each(|(k, x)| check_inline(&template[k.as_str()], x, &format!("{path}/{k}"))),
        Value::Array(a) => {
            let mut values = Vec::new();
            numbers(v, &mut values);
            if a.iter().all(|x| x.is_number() || x.is_array()) {
                assert!(
                    values.len() <= INLINE_LIMIT || template.is_array(),
                    "{path}: {} inline numbers",
                    values.len()
                );
            } else {
                a.iter().enumerate().for_each(|(i, x)| {
                    check_inline(
                        template.get(i).unwrap_or(&Value::Null),
                        x,
                        &format!("{path}[{i}]"),
                    )
                });
            }
        }
        _ => {}
    }
}

fn report(
    old: &Value,
    new: &Value,
    old_npz: &[(String, String, Vec<u8>)],
    new_npz: &[(String, String, Vec<u8>)],
) {
    for key in [
        "m",
        "p",
        "channels",
        "electrons",
        "spins",
        "eion",
        "ion_constant",
        "constant",
        "reference",
        "permutation_count",
    ] {
        println!("  {key:>17}: {} -> {}", old[key], new[key]);
    }
    println!("  {:>17}: {} -> {}", "grid", old["grid"], new["grid"]);
    let shells = |v: &Value| -> Vec<(Value, Value, Value, Value)> {
        v["hubbard"]["shells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["atom"].clone(),
                    s["projectors"].clone(),
                    s["weights"].clone(),
                    s["seed"].clone(),
                )
            })
            .collect()
    };
    for ((a0, p0, w0, s0), (a1, p1, w1, s1)) in shells(old).into_iter().zip(shells(new)) {
        println!(
            "  shell atom {a0} -> {a1}: projectors {} | weights {w0} -> {w1} | seed {s0} -> {s1}",
            if p0 == p1 {
                "equal".to_string()
            } else {
                format!("{p0} -> {p1}")
            }
        );
    }
    let floats = |d: &str, b: &[u8]| -> Vec<f64> {
        match d {
            "<u8" => b
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
            _ => b
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        }
    };
    println!("  npz member: old length, new length, largest |new - old|, largest |old|");
    for (name, descr, data) in new_npz {
        let new_values = floats(descr, data);
        match old_npz.iter().find(|(n, _, _)| n == name) {
            Some((_, d, b)) => {
                let old_values = floats(d, b);
                let peak = old_values.iter().fold(0f64, |m, x| m.max(x.abs()));
                let diff = if old_values.len() == new_values.len() {
                    old_values
                        .iter()
                        .zip(&new_values)
                        .fold(0f64, |m, (x, y)| m.max((x - y).abs()))
                } else {
                    f64::NAN
                };
                println!(
                    "  {name:>17}: {:>9} {:>9}  {diff:.3e}  {peak:.3e}",
                    old_values.len(),
                    new_values.len()
                );
            }
            None => println!("  {name:>17}: new member"),
        }
    }
}

/// `YYYY-MM-DD` of today (UTC), from the system clock (civil-from-days, H. Hinnant).
fn today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        / 86400;
    let z = days + 719468;
    let (era, doe) = (z.div_euclid(146097), z.rem_euclid(146097));
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    format!(
        "{:04}-{month:02}-{day:02}",
        yoe + era * 400 + i64::from(month <= 2)
    )
}

#[test]
#[ignore = "generation tool: needs MANDACARU_FREEZE_WORK (see the module doc); four setups, no SCF"]
fn reexport_the_hubbard_crystal_fixtures() {
    let Ok(work) = std::env::var("MANDACARU_FREEZE_WORK") else {
        println!("nothing to do: MANDACARU_FREEZE_WORK is not set");
        return;
    };
    let folder = reference().join("hubbard").join("crystals");
    let out = std::env::var("MANDACARU_FREEZE_OUT").map_or(folder.clone(), PathBuf::from);
    let against = std::env::var("MANDACARU_FREEZE_AGAINST").map_or(folder.clone(), PathBuf::from);
    let cases: Vec<String> = std::env::var("MANDACARU_FREEZE_CASES")
        .map_or(CASES.iter().map(|c| c.to_string()).collect(), |v| {
            v.split(',').map(str::to_string).collect()
        });
    let read = |path: PathBuf| -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
        )
        .unwrap()
    };
    std::fs::create_dir_all(&work).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    for name in cases {
        let request = read(
            reference()
                .join("periodic_requests")
                .join(format!("hubbard_{name}.json")),
        );
        let case = Case::load(request);
        let crystal = case.setup(None).build(&case.ctx).unwrap();
        let raw = PathBuf::from(&work).join(format!("{name}.saved.json"));
        crystal.save(raw.to_str().unwrap()).unwrap();
        // The raw save stays in the work folder (the full crystal, every block inline). The
        // fixture keeps the layout the tests read: the `setup` description (the basis tables) and
        // the absolute dataset paths of `sites` are left out; `provenance` names the files.
        let mut new = read(raw.clone());
        for block in ["setup", "sites"] {
            new.as_object_mut().unwrap().remove(block);
        }
        let old = read(against.join(format!("{name}.json")));
        let old_npz = members(&std::fs::read(against.join(format!("{name}.npz"))).unwrap());
        let dtypes: BTreeMap<String, String> = old_npz
            .iter()
            .map(|(n, d, _)| (n.clone(), d.clone()))
            .collect();
        // A keyed permutation table carries its count (the loader cannot size it from a key).
        new["permutation_count"] = json!(new["permutations"].as_array().unwrap().len());
        let mut arrays = Vec::new();
        split(&old, &mut new, &dtypes, &mut arrays, "");
        check_inline(&old, &new, "");
        let mut names: Vec<&String> = arrays.iter().map(|(n, _)| n).collect();
        names.sort();
        let mut expected: Vec<&String> = dtypes.keys().collect();
        expected.sort();
        assert_eq!(
            names, expected,
            "{name}: the npz members are the template's"
        );

        let set = case.request["_dataset_set"].as_str().unwrap();
        let mut species: Vec<String> = Vec::new();
        for atom in case.request["atoms"].as_array().unwrap() {
            let symbol = atom["symbol"].as_str().unwrap().to_string();
            if !species.contains(&symbol) {
                species.push(symbol);
            }
        }
        let library: Vec<Value> = species
            .iter()
            .map(|symbol| {
                let relative = format!("{set}/{symbol}.parquet");
                let file = PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap()).join(&relative);
                let bytes = std::fs::read(&file).unwrap();
                json!({"file": relative, "symbol": symbol, "bytes": bytes.len(), "sha256": sha256(&bytes),
                       "content_sha256": dataset_content_sha256(&case.ctx, &file).unwrap()})
            })
            .collect();
        let mut provenance = old["provenance"].clone();
        let object = provenance.as_object_mut().unwrap();
        object.remove("dataset_sources");
        object.insert("setup_engine".into(), json!("rust-setup"));
        object.insert(
            "generator".into(),
            json!("crates/mandacaru-core/tests/periodic_freeze_hubbard.rs"),
        );
        object.insert("date".into(), json!(today()));
        object.insert("library".into(), json!(library));
        new["provenance"] = provenance;
        let mut fixture = old["fixture"].clone();
        fixture["generator"] = json!("crates/mandacaru-core/tests/periodic_freeze_hubbard.rs");
        new["fixture"] = fixture;

        let new_npz_bytes = npz(&arrays);
        let new_npz = members(&new_npz_bytes);
        println!("{name}:");
        report(&old, &new, &old_npz, &new_npz);
        std::fs::write(out.join(format!("{name}.npz")), &new_npz_bytes).unwrap();
        std::fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_string_pretty(&new).unwrap() + "\n",
        )
        .unwrap();

        // The pair reads back through the tests' loader.
        let ctx = Context::new(Some(1), 1 << 30).unwrap();
        let back = try_load_in(&ctx, &out, &name, |_| {}).unwrap();
        assert_eq!((back.m, back.p), (crystal.m, crystal.p));
        println!(
            "  written to {out:?}: npz {:.1} MB",
            new_npz_bytes.len() as f64 / 1e6
        );
    }
}
