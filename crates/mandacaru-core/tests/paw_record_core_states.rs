//! The core states of a dataset record (format version 3) and the content digest a crystal
//! snapshot pins a dataset by (`PawDataset::content_sha256`).
//!
//! The fixture is `lda-sr/Li` of the library: a version-2 file with a single 1s core, whose
//! orbital `u = r sqrt(4 pi n_c / 2)` the stored core density gives exactly, so a version-3
//! copy can be written without the generator.

mod paw_record_support;

use mandacaru_core::parquet::{self, Codec, Table, WriteOptions};
use mandacaru_core::paw::digest::dataset_content_sha256;
use mandacaru_core::paw::record::{
    CoreState, PawDataset, WriteSpec, CORE_FREE_VERSION, CORE_SCHEMES, FORMAT_VERSION,
    MAX_CORE_STATES, METADATA_KEY,
};
use mandacaru_core::{Context, Error};
use paw_record_support::*;
use serde_json::{json, Value};
use std::f64::consts::PI;

const STORED: WriteSpec = WriteSpec {
    stride: 1,
    codec: Codec::Zstd,
};

/// The 1s core state of a record with a closed 1s core, from its stored core density.
fn one_s(ctx: &Context, ds: &PawDataset) -> CoreState {
    let orbital: Vec<f64> =
        ds.r.as_slice()
            .iter()
            .zip(ds.core_density.as_slice())
            .map(|(r, n)| r * (4. * PI * n / 2.).sqrt())
            .collect();
    CoreState {
        n: 1,
        l: 0,
        occupation: 2.,
        energy: -2.0,
        orbital: ctx.copy_slice(&orbital).unwrap(),
    }
}

/// `lda-sr/Li` written as version 3 with its 1s core state into `dir`.
fn lithium_v3(ctx: &Context, dir: &TempDir) -> std::path::PathBuf {
    let mut ds = PawDataset::read(ctx, &dataset_path("lda-sr", "Li")).unwrap();
    assert!(ds.core_states.is_none(), "the shipped library is version 2");
    let state = one_s(ctx, &ds);
    ds.core_states = Some(vec![state]);
    let path = dir.file("Li.parquet");
    ds.write(ctx, &path, &STORED).unwrap();
    path
}

#[test]
fn a_version_two_file_has_no_core_states_and_writes_back_as_version_two() {
    let ctx = context();
    let dir = TempDir::new("core-v2");
    let source = dataset_path("lda-sr", "Li");
    let ds = PawDataset::read(&ctx, &source).unwrap();
    assert!(ds.core_states.is_none());
    assert_eq!(ds.format_version(), CORE_FREE_VERSION);
    let path = dir.file("Li.parquet");
    ds.write(&ctx, &path, &STORED).unwrap();
    let table = parquet::read_table(&ctx, &path).unwrap();
    assert_eq!(payload_of(&table)["version"], json!(CORE_FREE_VERSION));
    assert!(payload_of(&table).get("core_states").is_none());
    drop(table);
    let back = PawDataset::read(&ctx, &path).unwrap();
    for ((name, a), (other, b)) in ds.tables().into_iter().zip(back.tables()) {
        assert_eq!(name, other);
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name} changed"
        );
    }
    drop((ds, back));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn version_three_adds_the_core_states_and_leaves_every_other_column_bit_identical() {
    let ctx = context();
    let dir = TempDir::new("core-v3");
    let source = dataset_path("lda-sr", "Li");
    let original = PawDataset::read(&ctx, &source).unwrap();
    let path = lithium_v3(&ctx, &dir);
    let back = PawDataset::read(&ctx, &path).unwrap();
    assert_eq!(back.format_version(), FORMAT_VERSION);
    let states = back
        .core_states
        .as_ref()
        .expect("version 3 carries its core");
    assert_eq!(states.len(), 1);
    let (state, want) = (&states[0], one_s(&ctx, &original));
    assert_eq!((state.n, state.l), (1, 0));
    assert_eq!(state.occupation.to_bits(), 2f64.to_bits());
    assert_eq!(state.energy.to_bits(), (-2f64).to_bits());
    // The orbital is stored under the ten-digit rule of every table.
    for (got, raw) in state.orbital.as_slice().iter().zip(want.orbital.as_slice()) {
        let rounded = mandacaru_core::paw::record::digits::round_digits(*raw);
        assert_eq!(got.to_bits(), rounded.to_bits());
    }
    // The upgrade identity: the version-2 columns, the payload without the new key and the
    // typed record are those of the source, to the bit.
    let old = original.tables();
    let new = back.tables();
    assert_eq!(new.len(), old.len() + 1);
    assert_eq!(new.last().unwrap().0, PawDataset::core_table(1, 0));
    for ((name, a), (other, b)) in old.iter().zip(&new) {
        assert_eq!(name, other);
        assert_eq!(a.len(), b.len(), "{name}");
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name} is not bit-identical"
        );
    }
    let source_table = parquet::read_table(&ctx, &source).unwrap();
    let upgraded_table = parquet::read_table(&ctx, &path).unwrap();
    let mut source_payload = payload_of(&source_table);
    let mut upgraded_payload = payload_of(&upgraded_table);
    assert_eq!(upgraded_payload["version"], json!(FORMAT_VERSION));
    for payload in [&mut source_payload, &mut upgraded_payload] {
        let map = payload.as_object_mut().unwrap();
        map.remove("version");
        map.remove("core_states");
    }
    assert_eq!(source_payload, upgraded_payload);
    drop((source_table, upgraded_table, want, original, back));
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_content_digest_is_the_same_for_a_file_and_its_version_three_upgrade() {
    let ctx = context();
    let dir = TempDir::new("core-digest");
    let source = dataset_path("lda-sr", "Li");
    let upgraded = lithium_v3(&ctx, &dir);
    let before = dataset_content_sha256(&ctx, &source).unwrap();
    let after = dataset_content_sha256(&ctx, &upgraded).unwrap();
    assert_eq!(before.len(), 64);
    assert!(before
        .bytes()
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    assert_eq!(before, after, "the core states are not part of the content");
    // A different codec is a different file with the same content.
    let ds = PawDataset::read(&ctx, &source).unwrap();
    let plain = dir.file("Li-plain.parquet");
    ds.write(
        &ctx,
        &plain,
        &WriteSpec {
            stride: 1,
            codec: Codec::Uncompressed,
        },
    )
    .unwrap();
    assert_ne!(
        std::fs::read(&plain).unwrap(),
        std::fs::read(&source).unwrap()
    );
    assert_eq!(dataset_content_sha256(&ctx, &plain).unwrap(), before);
    assert_eq!(ds.content_sha256(&ctx).unwrap(), before);
    // The radial scheme an upgrade records is bookkeeping too: it reads back, and the digest
    // ignores it.
    for scheme in CORE_SCHEMES {
        let mut named = PawDataset::read(&ctx, &upgraded).unwrap();
        named.core_scheme = Some(scheme.to_string());
        let path = dir.file(&format!("Li-{scheme}.parquet"));
        named.write(&ctx, &path, &STORED).unwrap();
        let back = PawDataset::read(&ctx, &path).unwrap();
        assert_eq!(back.core_scheme.as_deref(), Some(scheme));
        assert_eq!(back.content_sha256(&ctx).unwrap(), before);
        drop((named, back));
    }
    // One bit of one table, or one scalar of the payload, is another content.
    let mut changed = PawDataset::read(&ctx, &source).unwrap();
    let first = changed.v_local.as_slice()[0];
    changed.v_local.as_mut_slice()[0] = f64::from_bits(first.to_bits() ^ 1);
    assert_ne!(changed.content_sha256(&ctx).unwrap(), before);
    let mut shifted = PawDataset::read(&ctx, &source).unwrap();
    shifted.local_shift += 1e-12;
    assert_ne!(shifted.content_sha256(&ctx).unwrap(), before);
    drop((ds, changed, shifted));
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// Refusals: a mutated copy of the version-3 file.
// ---------------------------------------------------------------------------

fn mutated(
    ctx: &Context,
    dir: &TempDir,
    source: &std::path::Path,
    edit: impl FnOnce(&mut Table, &mut Value),
) -> std::path::PathBuf {
    let mut table = parquet::read_table(ctx, source).unwrap();
    let mut payload = payload_of(&table);
    edit(&mut table, &mut payload);
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

fn refusal(ctx: &Context, path: &std::path::Path) -> String {
    let error = PawDataset::read(ctx, path).expect_err("the file must be refused");
    assert!(matches!(error, Error::Invalid(_)), "{error:?}");
    assert_eq!(ctx.memory_used(), 0, "a refusal releases the budget");
    error.to_string()
}

#[test]
fn every_bound_of_the_core_states_is_refused_by_name() {
    let ctx = context();
    let dir = TempDir::new("core-refuse");
    let v3 = lithium_v3(&ctx, &dir);
    let v2 = dataset_path("lda-sr", "Li");
    let state = |p: &mut Value| p["core_states"][0].clone();
    type Edit = Box<dyn FnOnce(&mut Table, &mut Value)>;
    let many: Vec<Value> = (0..=MAX_CORE_STATES).map(|_| json!({})).collect();
    let cases: Vec<(&str, std::path::PathBuf, Edit, String)> = vec![
        (
            "version 3 without core states",
            v3.clone(),
            Box::new(|_, p| {
                p.as_object_mut().unwrap().remove("core_states");
            }),
            "\"core_states\" is missing".into(),
        ),
        (
            "core states in a version-2 file",
            v2.clone(),
            Box::new(|_, p| p["core_states"] = json!([])),
            "core_states".into(),
        ),
        (
            "a core scheme in a version-2 file",
            v2.clone(),
            Box::new(|_, p| p["core_scheme"] = json!("atom-solver")),
            "\"core_scheme\" is not part of the schema".into(),
        ),
        (
            "a core scheme that is not one of the radial schemes",
            v3.clone(),
            Box::new(|_, p| p["core_scheme"] = json!("hartree")),
            "\"hartree\" is not one of".into(),
        ),
        (
            "a core scheme that is not a string",
            v3.clone(),
            Box::new(|_, p| p["core_scheme"] = json!(1)),
            "core_scheme".into(),
        ),
        (
            "a version newer than this build",
            v3.clone(),
            Box::new(|_, p| p["version"] = json!(FORMAT_VERSION + 1)),
            format!("newer than this build ({FORMAT_VERSION})"),
        ),
        (
            "not a list",
            v3.clone(),
            Box::new(|_, p| p["core_states"] = json!({"n": 1})),
            "expected a list".into(),
        ),
        (
            "more states than a core holds",
            v3.clone(),
            Box::new(move |_, p| p["core_states"] = Value::Array(many)),
            format!("a core holds at most {MAX_CORE_STATES} subshells"),
        ),
        (
            "not a subshell",
            v3.clone(),
            Box::new(|_, p| p["core_states"][0]["l"] = json!(1)),
            "is not a subshell".into(),
        ),
        (
            "an occupation above the subshell's capacity",
            v3.clone(),
            Box::new(|_, p| p["core_states"][0]["occupation"] = json!(3.0)),
            "outside (0, 2]".into(),
        ),
        (
            "a table that is not the state's column",
            v3.clone(),
            Box::new(|_, p| p["core_states"][0]["table"] = json!("core_wave_n2_l0")),
            "is not the state's column".into(),
        ),
        (
            "a state declared twice",
            v3.clone(),
            Box::new(move |_, p| {
                let again = state(p);
                p["core_states"].as_array_mut().unwrap().push(again);
            }),
            "declared twice".into(),
        ),
        (
            "an unknown key",
            v3.clone(),
            Box::new(|_, p| p["core_states"][0]["j"] = json!(0.5)),
            "\"j\"".into(),
        ),
        (
            "more core electrons than Z",
            v3.clone(),
            Box::new(|_, p| {
                p["core_states"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"n": 2, "l": 1, "occupation": 6.0, "energy": -1.0,
                                 "table": "core_wave_n2_l1"}));
            }),
            "more than Z = 3".into(),
        ),
        (
            "a missing column",
            v3.clone(),
            Box::new(|t, _| {
                let name = PawDataset::core_table(1, 0);
                t.columns.retain(|(n, _)| *n != name);
            }),
            "\"core_wave_n1_l0\" is missing".into(),
        ),
    ];
    for (label, source, edit, expected) in cases {
        let path = mutated(&ctx, &dir, &source, edit);
        let message = refusal(&ctx, &path);
        assert!(
            message.contains(&expected),
            "{label}: {message:?} does not name {expected:?}"
        );
    }
}
