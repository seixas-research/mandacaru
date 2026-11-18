//! The core states of a dataset record (format 3, the only format) and the content digest a
//! crystal snapshot pins a dataset by (`PawDataset::content_sha256`).
//!
//! The fixture is `lda-sr/Li` of the installed library, copied into a temporary folder: a record
//! with a single 1s core state.

mod paw_record_support;

use mandacaru_core::parquet::{self, Codec, Table, WriteOptions};
use mandacaru_core::paw::digest::dataset_content_sha256;
use mandacaru_core::paw::record::{
    PawDataset, WriteSpec, FORMAT_VERSION, MAX_CORE_STATES, METADATA_KEY,
};
use mandacaru_core::{Context, Error};
use paw_record_support::*;
use serde_json::{json, Value};

const STORED: WriteSpec = WriteSpec {
    stride: 1,
    codec: Codec::Zstd,
};

/// `lda-sr/Li` of the library written into `dir`.
fn lithium(ctx: &Context, dir: &TempDir) -> std::path::PathBuf {
    let ds = PawDataset::read(ctx, &dataset_path("lda-sr", "Li")).unwrap();
    assert_eq!(ds.format_version(), FORMAT_VERSION);
    assert_eq!(
        ds.core_states.as_ref().map(Vec::len),
        Some(1),
        "Li has one 1s core state"
    );
    let path = dir.file("Li.parquet");
    ds.write(ctx, &path, &STORED).unwrap();
    path
}

#[test]
fn a_record_without_core_states_is_not_written() {
    let ctx = context();
    let dir = TempDir::new("core-none");
    let mut ds = PawDataset::read(&ctx, &lithium(&ctx, &dir)).unwrap();
    ds.core_states = None;
    let path = dir.file("none.parquet");
    let error = ds
        .write(&ctx, &path, &STORED)
        .expect_err("no core states, no file");
    assert!(error.to_string().contains("core_states"), "{error}");
    assert!(!path.exists());
    drop(ds);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_content_digest_ignores_the_codec_and_the_core_scheme() {
    let ctx = context();
    let dir = TempDir::new("core-digest");
    let source = lithium(&ctx, &dir);
    let before = dataset_content_sha256(&ctx, &source).unwrap();
    assert_eq!(before.len(), 64);
    assert!(before
        .bytes()
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
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
// Refusals: a mutated copy of the record.
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
    let v3 = lithium(&ctx, &dir);
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
            "an older format version",
            v3.clone(),
            Box::new(|_, p| p["version"] = json!(2)),
            format!("format version 2 is not supported: only format {FORMAT_VERSION} is read"),
        ),
        (
            "a legacy core_scheme field is refused as unknown",
            v3.clone(),
            Box::new(|_, p| p["core_scheme"] = json!("variational")),
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
