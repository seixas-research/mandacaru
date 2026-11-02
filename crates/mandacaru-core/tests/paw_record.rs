//! The typed PAW dataset record (`paw::record`): the 10-digit rule, the
//! reader against the frozen A0 digests, the census of the shipped library,
//! and the column order. The write path, the refusals and the local potential
//! have their own files (`paw_record_write`, `paw_record_refusals`,
//! `paw_record_values`).
//!
//! References are the frozen A0 files in `test/migration/reference/paw_generate`
//! and the shipped library (`$MANDACARU_PAW_PATH`, and the UPAW-LCAO library
//! beside it); a missing input FAILS, it never skips. Every test processes one
//! file at a time, `pbe-fr/Pb.parquet` (15.5 MB) first.

mod paw_record_support;

use std::collections::BTreeMap;

use mandacaru_core::parquet;
use mandacaru_core::paw::record::{digits, PawDataset, METADATA_KEY};
use paw_record_support::*;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// The tools of the comparison.
// ---------------------------------------------------------------------------

#[test]
fn sha256_matches_the_nist_vectors() {
    assert_eq!(
        sha::Sha256::digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha::Sha256::digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(
        sha::Sha256::digest(long),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

fn expected_bits(value: &Value) -> u64 {
    match value {
        Value::String(text) if text == "inf" => f64::INFINITY.to_bits(),
        other => other.as_f64().unwrap().to_bits(),
    }
}

#[test]
fn the_digits_rule_reproduces_every_frozen_table_case() {
    let misc = reference("misc.json");
    let table = &misc["_table"];
    assert_eq!(table["digits"], 10);
    let cases = &table["cases"];
    let edge = floats(&cases["edge_stride1"]["input"]);
    let random = floats(&cases["random_97"]["input"]);
    let mut checked = 0usize;
    let mut compare = |input: &[f64], stride: usize, expected: &Value, label: &str| {
        let got = digits::decimate(input, stride).unwrap();
        let want = expected.as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), expected_bits(w), "{label}[{i}]: {g:e} vs {w}");
            checked += 1;
        }
    };
    compare(&edge, 1, &cases["edge_stride1"]["output"], "edge stride 1");
    for stride in 2..=4 {
        let key = format!("edge_stride{stride}");
        compare(&edge, stride, &cases[key.as_str()]["output"], &key);
    }
    for (key, expected) in cases["random_97"]["output"].as_object().unwrap() {
        let stride: usize = key.parse().unwrap();
        compare(
            &random,
            stride,
            expected,
            &format!("random_97 stride {stride}"),
        );
    }
    println!("digits rule: {checked} values equal the frozen Python output bit for bit");
    // The overflow edge: a value that rounds past f64::MAX becomes infinity,
    // exactly as float('%.10g' % v) does; a tie at the eleventh digit rounds
    // to even like the C library.
    assert_eq!(digits::round_digits(f64::MAX), f64::INFINITY);
    assert_eq!(digits::round_digits(2f64.powi(-15)), 3.051757812e-5);
    assert_eq!(digits::round_digits(f64::NAN).to_bits(), f64::NAN.to_bits());
    assert!(digits::decimate(&edge, 0).is_err(), "stride 0 is refused");
}

// ---------------------------------------------------------------------------
// The A0 sample: every array of every frozen file matches its digest.
// ---------------------------------------------------------------------------

/// Checks `ds` against the frozen arrays of `(set, element)`; returns the
/// number of arrays compared.
fn check_against_frozen(ds: &PawDataset, set: &str, element: &str) -> usize {
    let (doc, rows) = frozen_arrays(set, element).expect("a frozen sample file");
    let arrays = arrays_of(ds);
    let names: Vec<&str> = arrays.iter().map(|(n, _, _)| n.as_str()).collect();
    let frozen_names: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
    assert_eq!(
        names, frozen_names,
        "{set}/{element}: array names and order"
    );
    for ((name, shape, values), (_, sha, frozen_shape, frozen_max)) in arrays.iter().zip(&rows) {
        assert!(
            shape_matches(frozen_shape, shape),
            "{set}/{element} {name}: shape {shape:?}"
        );
        assert_eq!(
            &sha256_of_f64(values),
            sha,
            "{set}/{element} {name}: sha256"
        );
        match frozen_max.as_f64() {
            Some(max) => {
                let got = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(got.to_bits(), max.to_bits(), "{set}/{element} {name}: max");
            }
            None => assert!(
                values.is_empty(),
                "{set}/{element} {name}: frozen max is null"
            ),
        }
    }
    let grid = &doc["grid"];
    let r = ds.r.as_slice();
    assert_eq!(r.len() as u64, grid["n"].as_u64().unwrap());
    assert_eq!(r[0], grid["first"].as_f64().unwrap());
    assert_eq!(*r.last().unwrap(), grid["last"].as_f64().unwrap());
    assert_eq!(r[1] - r[0], grid["h_first"].as_f64().unwrap());
    assert_eq!(
        ds.step(),
        r[0],
        "the step is read from r[0], not from points(Z)/4"
    );
    check_scalars(ds, &doc["scalars"], &format!("{set}/{element}"));
    arrays.len()
}

fn nlcc_value(ds: &PawDataset) -> Value {
    let n = &ds.nlcc;
    let mut map = serde_json::Map::new();
    map.insert("applied".into(), json!(n.applied));
    map.insert("r_nlcc".into(), json!(n.r_nlcc));
    if let Some(v) = &n.source {
        map.insert("source".into(), json!(v));
    }
    if let Some(v) = n.core_electrons {
        map.insert("core_electrons".into(), json!(v));
    }
    if let Some(v) = n.partial_core_electrons {
        map.insert("partial_core_electrons".into(), json!(v));
    }
    if let Some(v) = &n.reason {
        map.insert("reason".into(), json!(v));
    }
    Value::Object(map)
}

fn defects_value(ds: &PawDataset) -> Value {
    let d = &ds.defects;
    let keyed = |m: &BTreeMap<u32, f64>| {
        Value::Object(m.iter().map(|(l, v)| (l.to_string(), json!(v))).collect())
    };
    let mut map = serde_json::Map::new();
    map.insert("ghosts".into(), keyed(&d.ghosts));
    map.insert(
        "phases".into(),
        Value::Object(
            d.phases
                .iter()
                .map(|(l, v)| (l.to_string(), json!(v)))
                .collect(),
        ),
    );
    if !d.residuals.is_empty() {
        map.insert("residuals".into(), keyed(&d.residuals));
    }
    if let Some(v) = d.s_miss {
        map.insert("s_miss".into(), json!(v));
    }
    if !d.unprojected.is_empty() {
        map.insert("unprojected".into(), keyed(&d.unprojected));
    }
    Value::Object(map)
}

/// The scalar block of the A0 freeze (`dataset_scalars`) as the record sees it.
fn scalars_value(ds: &PawDataset) -> Value {
    let channels: serde_json::Map<String, Value> = ds
        .channels
        .iter()
        .map(|(l, ch)| {
            let entry = json!({
                "n": ch.n, "r_cut": ch.r_cut, "occupation": ch.occupation,
                "duality_error": ch.duality_error, "overlap_minimum": ch.overlap_minimum,
                "asymmetry": ch.asymmetry, "consistency_error": ch.consistency_error,
                "q_cut": ch.q_cut, "n_projectors": ch.projectors.len(),
            });
            (l.to_string(), entry)
        })
        .collect();
    json!({
        "symbol": ds.symbol, "atomic_number": ds.atomic_number, "family": ds.family.name(),
        "valence_charge": ds.valence_charge, "r_cut_local": ds.r_cut_local,
        "local_shift": ds.local_shift, "compensation_radius": ds.compensation_radius,
        "compensation_charge": ds.compensation_charge,
        "hartree_screening": ds.hartree_screening, "one_center_energy": ds.one_center_energy,
        "energies": ds.energies, "q_cut": ds.q_cut, "energy_offset": ds.energy_offset,
        "norm_deficit": ds.norm_deficit, "xc": ds.xc, "relativity": ds.relativity,
        "relativistic_exchange": ds.relativistic_exchange, "extra_l": ds.extra_l,
        "nlcc": nlcc_value(ds), "frozen_subshells": ds.frozen_subshells,
        "semicore_subshells": ds.semicore_subshells, "defects": defects_value(ds),
        "n_points": ds.points(), "channels": channels,
    })
}

/// Every key of `frozen` exists in `mine` and holds the same value; numbers
/// agree bit for bit (the frozen JSON prints the shortest round-trip form).
fn subset_equal(frozen: &Value, mine: &Value, path: &str, compared: &mut usize) {
    match (frozen, mine) {
        (Value::Object(f), Value::Object(m)) => {
            for (key, value) in f {
                let there = m
                    .get(key)
                    .unwrap_or_else(|| panic!("{path}.{key} is missing"));
                subset_equal(value, there, &format!("{path}.{key}"), compared);
            }
        }
        (Value::Array(f), Value::Array(m)) => {
            assert_eq!(f.len(), m.len(), "{path}: length");
            for (i, (a, b)) in f.iter().zip(m).enumerate() {
                subset_equal(a, b, &format!("{path}[{i}]"), compared);
            }
        }
        (Value::Number(f), Value::Number(m)) => {
            assert_eq!(
                f.as_f64().unwrap().to_bits(),
                m.as_f64().unwrap().to_bits(),
                "{path}"
            );
            *compared += 1;
        }
        (f, m) => {
            assert_eq!(f, m, "{path}");
            *compared += 1;
        }
    }
}

fn check_scalars(ds: &PawDataset, frozen: &Value, label: &str) {
    let mut compared = 0;
    subset_equal(frozen, &scalars_value(ds), label, &mut compared);
    assert!(compared >= 15, "{label}: only {compared} scalars compared");
}

#[test]
fn pb_pbe_fr_is_read_first_and_matches_its_frozen_digests() {
    let ctx = context();
    let path = library().join("pbe-fr/Pb.parquet");
    let pb = reference("pb_parquet.json");
    let facts = &pb["pbe-fr/Pb"];
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        facts["bytes"].as_u64().unwrap()
    );
    let ds = PawDataset::read(&ctx, &path).unwrap();
    // A0 finding 7: this file was written at stride 2 (r[0] = 2 h_gen).
    let h = ds.step();
    assert_eq!(h, facts["r_first"].as_f64().unwrap());
    // lda-fr/Pb is stride 4 (r[0] = 4 h_gen); this file is stride 2, so its
    // step is half of that one's, and neither is points(Z)/4.
    let h4 = reference("lib_lda-fr_Pb.json")["grid"]["first"]
        .as_f64()
        .unwrap();
    assert!(
        (2.0 * h - h4).abs() < 1e-12,
        "stride 2 against stride 4: {h:e} vs {h4:e}"
    );
    assert_eq!(
        ds.r.as_slice().len() as u64,
        facts["rows"][0].as_u64().unwrap()
    );
    assert_eq!(
        ds.tables().len() as u64,
        facts["n_columns"].as_u64().unwrap()
    );
    assert_eq!(ds.relativity, facts["relativity"].as_str().unwrap());
    assert_eq!(ds.xc, facts["xc"].as_str().unwrap());
    let compared = check_against_frozen(&ds, "pbe-fr", "Pb");
    println!("pbe-fr/Pb: {compared} arrays equal their frozen sha256, h = r[0] = {h:e}");
    drop(ds);
    assert_eq!(
        ctx.memory_used(),
        0,
        "dropping the record releases the budget"
    );
}

#[test]
fn the_a0_sample_matches_its_frozen_digests() {
    let ctx = context();
    let root = library();
    let mut files = 0usize;
    let mut arrays = 0usize;
    for (set, path) in files_of(&root, &SETS) {
        let element = path.file_stem().unwrap().to_str().unwrap().to_string();
        if frozen_arrays(&set, &element).is_none() {
            continue;
        }
        let ds = PawDataset::read(&ctx, &path).unwrap();
        arrays += check_against_frozen(&ds, &set, &element);
        files += 1;
        drop(ds);
        assert_eq!(ctx.memory_used(), 0);
    }
    println!("A0 sample: {files} files, {arrays} arrays, every sha256, shape and max equal");
    assert!(
        files >= 19,
        "the A0 sample has at least 19 library files, found {files}"
    );
}

// ---------------------------------------------------------------------------
// The census of the whole library, one file at a time.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Census {
    files: usize,
    reordered: usize,
    channel_entries: usize,
    spin_orbit_entries: usize,
    semicore_defaults: usize,
    occupation_defaults: usize,
    refusals: Vec<String>,
    foreign_keys: BTreeMap<String, usize>,
    families: BTreeMap<String, usize>,
}

/// Reads every file of `files` (PAW first, then UPAW), checks the columns of the
/// record against the file's, and counts.
fn census(files: &[(String, std::path::PathBuf)]) -> Census {
    let ctx = context();
    let mut c = Census::default();
    for (set, path) in files {
        let label = format!("{set}/{}", path.file_name().unwrap().to_string_lossy());
        let footer = parquet::read_metadata(&ctx, path).unwrap();
        for key in footer
            .metadata
            .keys()
            .filter(|k| k.as_str() != METADATA_KEY)
        {
            *c.foreign_keys.entry(key.clone()).or_default() += 1;
        }
        drop(footer.metadata);
        match PawDataset::read(&ctx, path) {
            Ok(ds) => {
                let names: Vec<String> = ds.tables().into_iter().map(|(n, _)| n).collect();
                let columns: Vec<String> = footer.columns.iter().map(|(n, _)| n.clone()).collect();
                if names != columns {
                    // Python wrote the channels in generation order (lda-sr/As
                    // lists l = 2 first); the record writes them ascending.
                    c.reordered += 1;
                    let (mut a, mut b) = (names.clone(), columns.clone());
                    a.sort();
                    b.sort();
                    assert_eq!(a, b, "{label}: the record's columns are the file's");
                }
                assert_eq!(footer.num_rows as usize, ds.points(), "{label}");
                c.files += 1;
                c.channel_entries += ds.channels.len();
                c.spin_orbit_entries += ds.spin_orbit.len();
                *c.families.entry(ds.family.name().to_string()).or_default() += 1;
                for d in &ds.defaults_applied {
                    if d == "semicore_subshells" {
                        c.semicore_defaults += 1;
                    } else if d.ends_with(".occupations") {
                        c.occupation_defaults += 1;
                    } else {
                        panic!("{label}: unexpected default {d}");
                    }
                }
            }
            Err(e) => c.refusals.push(format!("{label}: {e}")),
        }
        assert_eq!(ctx.memory_used(), 0, "{label}: the budget is released");
    }
    c
}

#[test]
fn every_shipped_file_is_read_and_the_defaults_are_the_two_the_census_found() {
    let mut files = files_of(&library(), &SETS);
    let paw_files = files.len();
    let upaw = upaw_library();
    if let Some(root) = &upaw {
        files.extend(files_of(root, &["lda-sr"]));
    }
    let c = census(&files);
    println!(
        "census: {} files read of {} found ({} PAW-LCAO, {} UPAW-LCAO), {} refusals; \
         channel entries {}, spin-orbit entries {}; defaults applied: semicore_subshells {}, \
         occupations {}; foreign metadata keys ignored: {:?}; families {:?}; files whose channels are not in ascending l: {}",
        c.files,
        files.len(),
        paw_files,
        files.len() - paw_files,
        c.refusals.len(),
        c.channel_entries,
        c.spin_orbit_entries,
        c.semicore_defaults,
        c.occupation_defaults,
        c.foreign_keys,
        c.families,
        c.reordered
    );
    assert!(c.refusals.is_empty(), "{:#?}", c.refusals);
    assert_eq!(paw_files, 368, "4 sets x 92 PAW-LCAO files");
    assert_eq!(
        upaw.map(|_| files.len() - paw_files),
        Some(63),
        "the UPAW-LCAO library has 63 files"
    );
    assert_eq!(c.files, files.len());
    // The census of the Python side found exactly these two absences.
    assert_eq!(c.semicore_defaults, 11);
    assert_eq!(c.occupation_defaults, 21);
    assert_eq!(c.spin_orbit_entries, 273);
}
