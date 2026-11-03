//! A record in the layout of a dataset file (`PawDataset::parts` / `from_parts`) and the plain
//! JSON writer (`write_json`): what the Python boundary hands across, and the formats it writes.
//! The dataset is generated here (hydrogen, LDA, scalar-relativistic: about a second), so the
//! test needs no library.

mod paw_record_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::parquet::Codec;
use mandacaru_core::paw::generate::{generate, recipe::Recipe};
use mandacaru_core::paw::record::{PawDataset, WriteSpec};
use mandacaru_core::Context;
use paw_record_support::TempDir;

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

fn hydrogen(ctx: &Context) -> PawDataset {
    let recipe = Recipe::for_dataset("H", "lda", Treatment::Scalar).unwrap();
    generate(ctx, &recipe).unwrap().dataset
}

fn copies(
    ctx: &Context,
    parts: &mandacaru_core::paw::record::Parts<'_>,
) -> Vec<(String, mandacaru_core::Buffer<f64>)> {
    parts
        .tables
        .iter()
        .map(|(name, values)| (name.clone(), ctx.copy_slice(values).unwrap()))
        .collect()
}

#[test]
fn a_record_survives_parts_bit_for_bit_and_unrounded() {
    let ctx = context();
    let ds = hydrogen(&ctx);
    let parts = ds.parts(&ctx).unwrap();
    let back = PawDataset::from_parts(&ctx, &parts.text, copies(&ctx, &parts)).unwrap();
    let again = back.parts(&ctx).unwrap();
    assert_eq!(parts.text, again.text);
    assert_eq!(parts.tables.len(), again.tables.len());
    for ((a, x), (b, y)) in parts.tables.iter().zip(&again.tables) {
        assert_eq!(a, b);
        assert!(
            x.iter()
                .zip(y.iter())
                .all(|(p, q)| p.to_bits() == q.to_bits()),
            "table {a} changed"
        );
    }
    // The generated tables are handed out as they are: ten digits would change some of them.
    let r = parts.tables.iter().find(|(n, _)| n == "v_local").unwrap().1;
    assert!(
        r.iter()
            .any(|&v| v != mandacaru_core::paw::record::digits::round_digits(v)),
        "the in-memory record is not rounded to the storage digits"
    );
}

#[test]
fn from_parts_refuses_a_duplicate_a_missing_and_a_foreign_table() {
    let ctx = context();
    let ds = hydrogen(&ctx);
    let parts = ds.parts(&ctx).unwrap();

    let mut twice = copies(&ctx, &parts);
    let first = twice[0].0.clone();
    twice.push((first, ctx.copy_slice(&[0.0]).unwrap()));
    let error = PawDataset::from_parts(&ctx, &parts.text, twice).unwrap_err();
    assert!(error.to_string().contains("given twice"), "{error}");

    let mut short = copies(&ctx, &parts);
    short.pop();
    assert!(PawDataset::from_parts(&ctx, &parts.text, short).is_err());

    let mut foreign = copies(&ctx, &parts);
    foreign.push(("not_a_table".into(), ctx.copy_slice(&[0.0]).unwrap()));
    assert!(PawDataset::from_parts(&ctx, &parts.text, foreign).is_err());
}

#[test]
fn the_json_file_and_the_parquet_file_hold_the_same_record() {
    let ctx = context();
    let ds = hydrogen(&ctx);
    let dir = TempDir::new("parts");
    let (json, parquet) = (dir.file("H.json"), dir.file("H.parquet"));
    ds.write_json(&ctx, &json, 4).unwrap();
    ds.write(
        &ctx,
        &parquet,
        &WriteSpec {
            stride: 4,
            codec: Codec::Zstd,
        },
    )
    .unwrap();
    assert_eq!(
        dir.names(),
        ["H.json", "H.parquet"],
        "the staged files are gone"
    );
    let (a, b) = (
        PawDataset::read(&ctx, &json).unwrap(),
        PawDataset::read(&ctx, &parquet).unwrap(),
    );
    let (pa, pb) = (a.parts(&ctx).unwrap(), b.parts(&ctx).unwrap());
    assert_eq!(pa.text, pb.text);
    for ((na, x), (nb, y)) in pa.tables.iter().zip(&pb.tables) {
        assert_eq!(na, nb);
        assert!(
            x.iter()
                .zip(y.iter())
                .all(|(p, q)| p.to_bits() == q.to_bits()),
            "{na}"
        );
    }
    assert_eq!(a.points(), (ds.points() - 3).div_ceil(4));
    assert!(ds.write_json(&ctx, &dir.file("zero.json"), 0).is_err());
    assert_eq!(
        dir.names(),
        ["H.json", "H.parquet"],
        "a refused write leaves nothing"
    );
}
