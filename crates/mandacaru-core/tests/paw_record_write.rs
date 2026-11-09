//! The write path of `paw::record`: stride-1 rewrite of the whole library,
//! stride 4, the staged `.new` file and its failures.

mod paw_record_support;

use mandacaru_core::parquet::{self, Codec};
use mandacaru_core::paw::record::{digits, PawDataset, WriteSpec};
use mandacaru_core::Context;
use paw_record_support::*;
use serde_json::Value;

// ---------------------------------------------------------------------------
// The write path: stride-1 rewrite, stride 4, the staged file.
// ---------------------------------------------------------------------------

/// Drops from a rewritten payload the keys the reader had to default, so it
/// can be compared with the original file's payload.
fn without_defaults(mut payload: Value, defaults: &[String]) -> Value {
    for name in defaults {
        match name.strip_prefix("channels.") {
            Some(rest) => {
                let l = rest.strip_suffix(".occupations").unwrap();
                payload["channels"][l]
                    .as_object_mut()
                    .unwrap()
                    .remove("occupations");
            }
            None => {
                payload.as_object_mut().unwrap().remove(name.as_str());
            }
        }
    }
    payload
}

/// Reads `path`, writes it at stride 1 and checks the new file: the columns are
/// the input's bit for bit, the payload is the input's as parsed JSON.
/// Returns `(bytes in, bytes out)`.
fn rewrite_at_stride_1(
    ctx: &Context,
    path: &std::path::Path,
    dir: &TempDir,
    codec: Codec,
) -> (u64, u64) {
    let label = path.display().to_string();
    let ds = PawDataset::read(ctx, path).unwrap();
    let out = dir.file("rewritten.parquet");
    ds.write(ctx, &out, &WriteSpec { stride: 1, codec })
        .unwrap();
    assert_eq!(
        dir.names(),
        ["rewritten.parquet"],
        "{label}: the staged file is gone"
    );
    let (before, after) = (
        parquet::read_table(ctx, path).unwrap(),
        parquet::read_table(ctx, &out).unwrap(),
    );
    // The same columns by name; the record writes channels in ascending l.
    let (mut names_before, mut names_after) = (before.column_names(), after.column_names());
    names_before.sort();
    names_after.sort();
    assert_eq!(names_before, names_after, "{label}");
    for (name, a) in &before.columns {
        let a = a.as_f64().unwrap();
        let b = after.column(name).unwrap().as_f64().unwrap();
        assert!(
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{label}: column {name} is not bit for bit"
        );
    }
    assert_eq!(
        payload_of(&before),
        without_defaults(payload_of(&after), &ds.defaults_applied),
        "{label}: the payload as parsed JSON"
    );
    let sizes = (
        std::fs::metadata(path).unwrap().len(),
        std::fs::metadata(&out).unwrap().len(),
    );
    std::fs::remove_file(&out).unwrap();
    sizes
}

#[test]
fn pb_pbe_fr_rewrites_at_stride_1_bit_for_bit_with_both_codecs() {
    let ctx = context();
    let dir = TempDir::new("pb");
    let path = dataset_path("pbe-fr", "Pb");
    for codec in [Codec::Uncompressed, Codec::Zstd] {
        let (bytes_in, bytes_out) = rewrite_at_stride_1(&ctx, &path, &dir, codec);
        println!(
            "pbe-fr/Pb stride-1 rewrite ({}): columns bit for bit, payload equal as JSON; {bytes_in} B in, {bytes_out} B out",
            codec.name()
        );
        assert_eq!(ctx.memory_used(), 0);
    }
}

#[test]
#[ignore = "slow: 29 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_record_write -- --ignored the_whole_library_rewrites_at_stride_1_one_file_at_a_time"]
fn the_whole_library_rewrites_at_stride_1_one_file_at_a_time() {
    let ctx = context();
    let dir = TempDir::new("all");
    let mut files = files_of(&library(), &SETS);
    if let Some(root) = upaw_library() {
        files.extend(files_of(&root, &["lda-sr"]));
    }
    let mut bytes = 0u64;
    for (_, path) in &files {
        bytes += rewrite_at_stride_1(&ctx, path, &dir, Codec::Uncompressed).0;
    }
    println!(
        "stride-1 rewrite: {} files ({} MB read) reproduce their columns bit for bit and their payload as parsed JSON",
        files.len(),
        bytes >> 20
    );
}

#[test]
fn stride_4_keeps_every_fourth_point_to_ten_digits() {
    let ctx = context();
    let dir = TempDir::new("stride");
    let ds = PawDataset::read(&ctx, &dataset_path("lda-fr", "Na")).unwrap();
    let out = dir.file("Na4.parquet");
    ds.write(
        &ctx,
        &out,
        &WriteSpec {
            stride: 4,
            codec: Codec::Zstd,
        },
    )
    .unwrap();
    let back = PawDataset::read(&ctx, &out).unwrap();
    assert_eq!(
        back.r.as_slice(),
        digits::decimate(ds.r.as_slice(), 4).unwrap()
    );
    assert_eq!(back.points(), (ds.points() - 3).div_ceil(4));
    assert_eq!(back.step(), digits::round_digits(ds.r.as_slice()[3]));
    for (l, so) in &ds.spin_orbit {
        for (k, p) in so.projectors.iter().enumerate() {
            let kept = digits::decimate(p.as_slice(), 4).unwrap();
            assert_eq!(back.spin_orbit[l].projectors[k].as_slice(), kept);
        }
    }
    // A second pass at stride 1 changes nothing: the rule is idempotent.
    let again = dir.file("Na4b.parquet");
    back.write(
        &ctx,
        &again,
        &WriteSpec {
            stride: 1,
            codec: Codec::Uncompressed,
        },
    )
    .unwrap();
    let (a, b) = (
        parquet::read_table(&ctx, &out).unwrap(),
        parquet::read_table(&ctx, &again).unwrap(),
    );
    for ((_, x), (_, y)) in a.columns.iter().zip(&b.columns) {
        let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
    assert!(ds
        .write(
            &ctx,
            &dir.file("none.parquet"),
            &WriteSpec {
                stride: 0,
                codec: Codec::Zstd
            }
        )
        .is_err());
    assert!(
        ds.write(
            &ctx,
            &dir.file("none.parquet"),
            &WriteSpec {
                stride: 10_000,
                codec: Codec::Zstd
            }
        )
        .is_err(),
        "a stride past the table leaves nothing to store"
    );
}

#[test]
fn a_failed_write_leaves_the_target_alone_and_no_staged_file() {
    let ctx = context();
    let dir = TempDir::new("staged");
    let mut ds = PawDataset::read(&ctx, &dataset_path("lda-fr", "Na")).unwrap();
    let target = dir.file("Na.parquet");
    let spec = WriteSpec {
        stride: 1,
        codec: Codec::Uncompressed,
    };
    ds.write(&ctx, &target, &spec).unwrap();
    let good = std::fs::read(&target).unwrap();
    // A scalar JSON cannot hold.
    ds.valence_charge = f64::NAN;
    let error = ds.write(&ctx, &target, &spec).unwrap_err();
    assert!(error.to_string().contains("cannot be stored"), "{error}");
    ds.valence_charge = 9.0;
    // Spin-orbit projectors off the radial grid: the read-back refuses them.
    let off_grid = ctx.allocate::<f64>(ds.r.as_slice().len() - 1).unwrap();
    let (l, so) = ds.spin_orbit.iter_mut().next().unwrap();
    let l = *l;
    so.projectors[0] = off_grid;
    let error = ds.write(&ctx, &target, &spec).unwrap_err();
    assert!(error.to_string().contains("radial grid"), "l={l}: {error}");
    assert_eq!(
        dir.names(),
        ["Na.parquet"],
        "no staged file survives a failure"
    );
    assert_eq!(
        std::fs::read(&target).unwrap(),
        good,
        "the target is untouched"
    );
}
