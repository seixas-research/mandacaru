//! The typed PAW dataset record (`paw::record`): the 10-digit rule, the census of the shipped
//! library, and the column order. The write path, the refusals and the local potential have
//! their own files (`paw_record_write`, `paw_record_refusals`, `paw_record_values`).
//!
//! The input is the shipped library (`$MANDACARU_PAW_PATH`); a missing input FAILS, it never skips. Every test processes one file at a time,
//! `pbe-fr/Pb.parquet` (15.5 MB) first.

mod paw_record_support;

use std::collections::BTreeMap;

use mandacaru_core::parquet;
use mandacaru_core::paw::record::{digits, PawDataset, METADATA_KEY};
use paw_record_support::*;

// ---------------------------------------------------------------------------
// The tools.
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

#[test]
fn the_digits_rule_follows_the_c_library_at_its_edges() {
    // The overflow edge: a value that rounds past f64::MAX becomes infinity,
    // exactly as float('%.10g' % v) does; a tie at the eleventh digit rounds
    // to even like the C library.
    assert_eq!(digits::round_digits(f64::MAX), f64::INFINITY);
    assert_eq!(digits::round_digits(2f64.powi(-15)), 3.051757812e-5);
    assert_eq!(digits::round_digits(f64::NAN).to_bits(), f64::NAN.to_bits());
    assert!(
        digits::decimate(&[1.0, 2.0], 0).is_err(),
        "stride 0 is refused"
    );
}

/// A file's radial step is read from the file (`r[0]`), never derived from the generation grid:
/// the library once held a pbe-fr Pb written at stride 2 (A0 finding 7). That file was replaced on
/// 2026-11-09 by one written at stride 4, so the case is now made here: the shipped pbe-fr Pb is
/// written at stride 2 of its own table into a scratch file and read back, and the step, the row
/// count and every column follow the stride.
#[test]
fn a_file_written_at_another_stride_is_read_with_its_own_step() {
    use mandacaru_core::parquet::Codec;
    use mandacaru_core::paw::record::WriteSpec;
    let ctx = context();
    let ds = PawDataset::read(&ctx, &library().join("pbe-fr/Pb.parquet")).unwrap();
    let dir = TempDir::new("stride");
    let out = dir.file("Pb.parquet");
    ds.write(
        &ctx,
        &out,
        &WriteSpec {
            stride: 2,
            codec: Codec::Zstd,
        },
    )
    .unwrap();
    let half = PawDataset::read(&ctx, &out).unwrap();
    let (h, h2) = (ds.step(), half.step());
    println!("pbe-fr/Pb: step {h:e}, written at stride 2: {h2:e}");
    // The writer keeps ten significant digits (`digits::round_digits`), and `decimate` keeps every
    // second point from the second on.
    assert!((h2 - 2.0 * h).abs() < 1e-9 * h, "{h2:e} against 2 x {h:e}");
    assert_eq!(half.points(), ds.points() / 2);
    assert_eq!(half.tables().len(), ds.tables().len());
    assert_eq!(half.relativity, ds.relativity);
    drop((ds, half));
    assert_eq!(
        ctx.memory_used(),
        0,
        "dropping the records releases the budget"
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

/// Reads every file of `files` checks the columns of the
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
    let files = files_of(&library(), &SETS);
    let c = census(&files);
    println!(
        "census: {} files read of {} found, {} refusals; \
         channel entries {}, spin-orbit entries {}; defaults applied: semicore_subshells {}, \
         occupations {}; foreign metadata keys ignored: {:?}; families {:?}; files whose channels are not in ascending l: {}",
        c.files,
        files.len(),
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
    assert_eq!(files.len(), 368, "4 sets x 92 PAW-LCAO files");
    assert_eq!(c.files, files.len());
    // The keys older files lack. The census of the Python side found 11 and 21, and 273
    // spin-orbit entries; the files regenerated on 2026-11-09 and 10 by the Rust generator carry
    // the full schema and the fr ones the spin-orbit entries of today's partition, so the counts
    // measured on the library since are these (363 after the 106 files of 2026-11-09, 366 after
    // Cr, Zn and lda-fr Cu on 2026-11-10). The library regenerated and installed on 2026-11-16
    // carries the full schema in every file (no default applied) and 396 spin-orbit entries.
    assert_eq!(c.semicore_defaults, 0);
    assert_eq!(c.occupation_defaults, 0);
    assert_eq!(c.spin_orbit_entries, 396);
}
