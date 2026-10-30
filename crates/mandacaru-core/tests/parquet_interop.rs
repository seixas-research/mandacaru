//! Interoperability: files written by other libraries are read bit for bit,
//! the 431 shipped pseudopotential files match the digests frozen from the
//! previous reader, and files written here have exactly the layout other
//! readers (pyarrow, fastparquet, pandas, DuckDB) depend on, checked by an
//! independent walk of the Thrift bytes.
//!
//! The foreign files are the frozen fixtures under
//! `test/migration/reference/parquet/`. The shipped libraries are read from
//! `MANDACARU_PAW_PATH` and `MANDACARU_UPAW_PATH`; a test that needs an unset
//! library says so on stderr and passes without it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use mandacaru_core::parquet::{
    format::{
        self, codec as codec_id, encoding, page_type, physical, repetition, FileMetaData,
        PageHeader,
    },
    read_metadata, read_table, write_table, Codec, Column, ColumnKind, Table, Utf8Column,
    WriteOptions, CREATED_BY, MAGIC,
};
use mandacaru_core::Context;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

static SERIAL: AtomicUsize = AtomicUsize::new(0);

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mandacaru-parquet-interop-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/parquet")
}

fn json(name: &str) -> Value {
    let text = fs::read_to_string(fixtures().join(name)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

// ---------------------------------------------------------------------------
// SHA-256 (streaming; checked against known digests in `parquet.rs`).
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

struct Sha256 {
    h: [u32; 8],
    block: Vec<u8>,
    length: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: Vec::with_capacity(64),
            length: 0,
        }
    }

    fn compress(h: &mut [u32; 8], block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = *h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (slot, value) in h.iter_mut().zip(v) {
            *slot = slot.wrapping_add(value);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length += data.len() as u64;
        if !self.block.is_empty() {
            let take = (64 - self.block.len()).min(data.len());
            self.block.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.block.len() == 64 {
                Self::compress(&mut self.h, &self.block);
                self.block.clear();
            }
        }
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            Self::compress(&mut self.h, block);
        }
        self.block.extend_from_slice(blocks.remainder());
    }

    fn finish(mut self) -> String {
        let bits = self.length * 8;
        let mut tail = std::mem::take(&mut self.block);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bits.to_be_bytes());
        for block in tail.chunks_exact(64) {
            Self::compress(&mut self.h, block);
        }
        self.h.iter().map(|word| format!("{word:08x}")).collect()
    }
}

fn sha256(data: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(data);
    hash.finish()
}

fn double_digest(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    let mut buffer = Vec::with_capacity(1 << 16);
    for chunk in values.chunks(1 << 13) {
        buffer.clear();
        for value in chunk {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        hash.update(&buffer);
    }
    hash.finish()
}

fn string_digest(column: &Utf8Column) -> String {
    let mut hash = Sha256::new();
    for i in 0..column.len() {
        if i > 0 {
            hash.update(&[0]);
        }
        hash.update(column.bytes(i));
    }
    hash.finish()
}

// ---------------------------------------------------------------------------
// Foreign writers: pyarrow and fastparquet files against what the previous
// reader returned (`old_api_results.json`).
// ---------------------------------------------------------------------------

/// Parses Python's `float.hex()` text into the bit pattern it denotes
/// (`None` for `nan`, whose payload Python does not keep).
fn parse_hex_float(text: &str) -> Option<u64> {
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => (1u64 << 63, rest),
        None => (0, text),
    };
    match rest {
        "inf" => return Some(sign | (0x7ffu64 << 52)),
        "nan" => return None,
        _ => {}
    }
    let rest = rest.strip_prefix("0x").unwrap_or_else(|| panic!("{text}"));
    let (mantissa, exponent) = rest.split_once('p').unwrap();
    let exponent: i64 = exponent.parse().unwrap();
    let (int_part, fraction) = mantissa.split_once('.').unwrap();
    // Python pads every nonzero value to 13 hex digits and writes zero as `0x0.0p+0`.
    assert!(fraction.len() == 13 || fraction == "0", "{text}");
    let fraction = u64::from_str_radix(fraction, 16).unwrap();
    match int_part {
        // Zero or a subnormal: always printed with exponent -1022 (or +0 for zero).
        "0" => {
            assert!(fraction == 0 || exponent == -1022, "{text}");
            Some(sign | fraction)
        }
        "1" => Some(sign | (((exponent + 1023) as u64) << 52) | fraction),
        other => panic!("{text}: integer part {other}"),
    }
}

/// Python's `float.hex()` of a value (`nan` and `inf` as Python prints them).
fn python_hex(x: f64) -> String {
    let bits = x.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return format!("{sign}inf");
    }
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & ((1u64 << 52) - 1);
    if exponent == 0 && mantissa == 0 {
        format!("{sign}0x0.0p+0")
    } else if exponent == 0 {
        format!("{sign}0x0.{mantissa:013x}p-1022")
    } else {
        format!("{sign}0x1.{mantissa:013x}p{:+}", exponent - 1023)
    }
}

#[test]
fn the_hex_float_parser_is_right() {
    for value in [
        0.0f64,
        -0.0,
        1.0,
        -1.5,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(0x800f_ffff_ffff_ffff),
        f64::INFINITY,
        f64::NEG_INFINITY,
        std::f64::consts::PI,
        1.0 / 3.0,
        -1.234e-300,
        6.02e23,
    ] {
        // Python's float.hex() form, produced here by hand from the bits.
        let bits = value.to_bits();
        let sign = if bits >> 63 == 1 { "-" } else { "" };
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & 0xf_ffff_ffff_ffff;
        let text = if value.is_infinite() {
            format!("{sign}inf")
        } else if exp == 0 {
            if frac == 0 {
                format!("{sign}0x0.0p+0")
            } else {
                format!("{sign}0x0.{frac:013x}p-1022")
            }
        } else {
            format!("{sign}0x1.{frac:013x}p{:+}", exp - 1023)
        };
        assert_eq!(parse_hex_float(&text), Some(bits), "{text}");
    }
    assert_eq!(parse_hex_float("nan"), None);
}

fn check_column_against_old(
    name: &str,
    column_name: &str,
    column: &Column,
    old: &Value,
    engine: &str,
) {
    let label = format!("{name} {column_name} ({engine})");
    assert_eq!(
        column.len() as u64,
        old["rows"].as_u64().unwrap(),
        "{label}"
    );
    if column.is_empty() {
        // An empty column has no values to compare, and the previous reader
        // labelled an empty float column a string column.
        assert!(
            old["values"].as_array().is_none_or(|v| v.is_empty()),
            "{label}"
        );
        return;
    }
    if old["values"].is_null() {
        // Long columns were frozen as a digest only.
        match column {
            Column::Double(values) => {
                let lines: Vec<String> = values.as_slice().iter().map(|v| python_hex(*v)).collect();
                assert_eq!(
                    sha256(lines.join("\n").as_bytes()),
                    old["sha256_hex_lines"].as_str().unwrap(),
                    "{label}"
                );
            }
            Column::Utf8(values) => assert_eq!(
                string_digest(values),
                old["sha256_nul_joined"].as_str().unwrap(),
                "{label}"
            ),
        }
        return;
    }
    match (column, old["kind"].as_str().unwrap()) {
        (Column::Double(values), "double") => {
            let old_values = old["values"].as_array().unwrap();
            assert_eq!(values.as_slice().len(), old_values.len(), "{label}");
            for (i, (got, want)) in values.as_slice().iter().zip(old_values).enumerate() {
                match parse_hex_float(want.as_str().unwrap()) {
                    Some(bits) => assert_eq!(got.to_bits(), bits, "{label}[{i}]"),
                    None => assert!(got.is_nan(), "{label}[{i}] should be NaN"),
                }
            }
        }
        (Column::Utf8(values), "string") => {
            let old_values = old["values"].as_array().unwrap();
            assert_eq!(values.len(), old_values.len(), "{label}");
            for (i, want) in old_values.iter().enumerate() {
                let want = want.as_str().unwrap();
                match want.strip_prefix("bytes:") {
                    // A BYTE_ARRAY column without a string annotation: the
                    // previous reader returned bytes, this one returns the
                    // same bytes as text.
                    Some(hex) => {
                        let got: String =
                            values.bytes(i).iter().map(|b| format!("{b:02x}")).collect();
                        assert_eq!(got, hex, "{label}[{i}]");
                    }
                    None => assert_eq!(values.get(i), want, "{label}[{i}]"),
                }
            }
            assert_eq!(
                string_digest(values),
                old["sha256_nul_joined"].as_str().unwrap(),
                "{label}"
            );
        }
        (_, kind) => panic!("{label}: the old reader saw a {kind} column"),
    }
}

#[test]
fn foreign_files_read_exactly_as_the_previous_reader_returned_them() {
    let old = json("old_api_results.json");
    let references = json("references.json");
    let ctx = context();
    let mut files = 0;
    let mut columns = 0;
    let mut disagreements = Vec::new();
    for (name, entry) in references["fixtures"].as_object().unwrap() {
        if entry["kind"] == "negative" {
            continue;
        }
        let raw = &old["fixtures"][name]["raw_columns_and_metadata"];
        let table =
            read_table(&ctx, &fixtures().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut seen_keys = BTreeSet::new();
        for engine in ["fastparquet", "pyarrow"] {
            if raw[engine]["ok"] != true {
                continue;
            }
            let result = &raw[engine]["result"];
            let old_columns = result["columns"].as_object().unwrap();
            assert_eq!(old_columns.len(), table.columns.len(), "{name} ({engine})");
            for (column_name, old_column) in old_columns {
                let column = table
                    .column(column_name)
                    .unwrap_or_else(|| panic!("{name}: no column {column_name}"));
                let agreed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    check_column_against_old(name, column_name, column, old_column, engine);
                }))
                .is_ok();
                if !agreed {
                    disagreements.push(format!("{name} {column_name} ({engine})"));
                }
                columns += 1;
            }
            for (key, value) in result["meta"].as_object().unwrap() {
                seen_keys.insert(key.clone());
                let got = table
                    .metadata
                    .get(key)
                    .unwrap_or_else(|| panic!("{name}: the key {key} ({engine}) is missing"));
                match value {
                    Value::String(text) => assert_eq!(got, text, "{name} {key} ({engine})"),
                    long => {
                        assert_eq!(
                            got.len() as u64,
                            long["bytes"].as_u64().unwrap(),
                            "{name} {key}"
                        );
                        assert_eq!(
                            sha256(got.as_bytes()),
                            long["sha256"].as_str().unwrap(),
                            "{name} {key}"
                        );
                    }
                }
            }
        }
        assert!(!seen_keys.is_empty() || table.metadata.is_empty() || raw["pyarrow"]["ok"] != true);
        files += 1;
    }
    assert_eq!(ctx.memory_used(), 0);
    // The previous reader's fastparquet engine mis-decoded some pyarrow files
    // (the plan records that fastparquet 2026.5.0 cannot read large pyarrow
    // caches); the pyarrow engine never disagrees, and the new reader agrees
    // with pyarrow's own decoding of every fixture (`parquet.rs`).
    eprintln!("columns where the previous reader disagreed: {disagreements:?}");
    assert_eq!(
        disagreements,
        ["hamiltonian/pyarrow_dictionary_fallback.parquet pauli (fastparquet)"],
        "the previous reader's known misreads are exactly one column"
    );
    assert!(
        files >= 38 && columns >= 150,
        "{files} files, {columns} columns"
    );
    eprintln!(
        "compared {files} foreign files ({columns} column readings) with the previous reader"
    );
}

#[test]
fn the_fixtures_cover_every_variant_the_reader_claims_to_support() {
    let references = json("references.json");
    let mut codecs = BTreeSet::new();
    let mut pages = BTreeSet::new();
    let mut encodings = BTreeSet::new();
    let mut writers = BTreeSet::new();
    let mut multi_group = false;
    let mut multi_page = false;
    let mut zero_rows = false;
    let mut fallback = false;
    let mut checksum = false;
    let mut footer_versions = BTreeSet::new();
    for (name, entry) in references["fixtures"].as_object().unwrap() {
        if entry["expect_new_reader"] != "ok" {
            continue;
        }
        writers.insert(entry["writer"].as_str().unwrap().to_owned());
        let structure = &entry["structure"];
        footer_versions.insert(structure["format_version"].as_str().unwrap().to_owned());
        multi_group |= structure["num_row_groups"].as_u64().unwrap() > 1;
        zero_rows |= structure["num_rows"].as_u64().unwrap() == 0;
        multi_page |= entry["max_pages_per_chunk"].as_u64().unwrap() > 1;
        checksum |= entry["page_crc_present"] == true;
        for chunk in structure["chunks"].as_array().unwrap() {
            codecs.insert(chunk["codec"].as_str().unwrap().to_owned());
            for e in chunk["encodings"].as_array().unwrap() {
                encodings.insert(e.as_str().unwrap().to_owned());
            }
        }
        let summary: Vec<&str> = entry["page_summary"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        for page in &summary {
            pages.insert((*page).to_owned());
        }
        fallback |= summary.iter().any(|p| p.starts_with("DICTIONARY_PAGE"))
            && summary.contains(&"DATA_PAGE:PLAIN");
        let _ = name;
    }
    for codec in ["ZSTD", "SNAPPY", "UNCOMPRESSED"] {
        assert!(codecs.contains(codec), "no {codec} fixture: {codecs:?}");
    }
    assert!(
        pages.iter().any(|p| p.starts_with("DATA_PAGE_V2")),
        "{pages:?}"
    );
    assert!(
        pages.iter().any(|p| p.starts_with("DICTIONARY_PAGE")),
        "{pages:?}"
    );
    assert!(pages.contains("DATA_PAGE:PLAIN"), "{pages:?}");
    assert!(
        encodings.contains("PLAIN_DICTIONARY")
            || pages.iter().any(|p| p.contains("PLAIN_DICTIONARY")),
        "no PLAIN_DICTIONARY (format 1.0) fixture"
    );
    assert!(
        pages.iter().any(|p| p.contains("RLE_DICTIONARY")),
        "{pages:?}"
    );
    assert!(
        writers.contains("pyarrow") && writers.contains("fastparquet"),
        "{writers:?}"
    );
    assert!(multi_group && multi_page && zero_rows && fallback && checksum);
    assert!(footer_versions.len() >= 2, "{footer_versions:?}");
}

// ---------------------------------------------------------------------------
// The shipped libraries.
// ---------------------------------------------------------------------------

fn library_root(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

/// Where a corpus name (`paw/<set>/<Symbol>.parquet`, `upaw/lda-sr/...`) lives.
fn library_path(name: &str) -> Option<PathBuf> {
    if let Some(rest) = name.strip_prefix("paw/") {
        library_root("MANDACARU_PAW_PATH").map(|root| root.join(rest))
    } else if let Some(rest) = name.strip_prefix("upaw/") {
        library_root("MANDACARU_UPAW_PATH").map(|root| root.join(rest))
    } else {
        None
    }
}

/// The footer of a shipped file equals what was frozen.
fn check_footer(name: &str, entry: &Value, ctx: &Context, path: &Path) {
    let footer = read_metadata(ctx, path).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(footer.num_rows, entry["rows"].as_u64().unwrap(), "{name}");
    assert_eq!(
        footer.created_by,
        entry["created_by"].as_str().unwrap(),
        "{name}"
    );
    assert_eq!(footer.num_row_groups, 1, "{name}");
    let expected = entry["columns"].as_array().unwrap();
    assert_eq!(footer.columns.len(), expected.len(), "{name}");
    for ((got, kind), want) in footer.columns.iter().zip(expected) {
        assert_eq!(got, want[0].as_str().unwrap(), "{name}");
        assert_eq!(*kind, ColumnKind::Double, "{name} {got}");
    }
    let kv = &entry["kv"];
    let text = &footer.metadata[kv["key"].as_str().unwrap()];
    assert_eq!(text.len() as u64, kv["bytes"].as_u64().unwrap(), "{name}");
    for (key, other) in entry["kv_other_keys"].as_object().unwrap() {
        let text = &footer.metadata[key];
        assert_eq!(
            text.len() as u64,
            other["bytes"].as_u64().unwrap(),
            "{name} {key}"
        );
        assert_eq!(
            sha256(text.as_bytes()),
            other["sha256"].as_str().unwrap(),
            "{name} {key}"
        );
    }
    assert_eq!(
        footer.metadata.len(),
        1 + entry["kv_other_keys"].as_object().unwrap().len(),
        "{name}"
    );
}

/// Every column and the pseudopotential text of a shipped file hash to the
/// frozen digests. Returns the table for further use.
fn check_digests(name: &str, entry: &Value, ctx: &Context, path: &Path) -> Table {
    let table = read_table(ctx, path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let expected = entry["columns"].as_array().unwrap();
    assert_eq!(table.columns.len(), expected.len(), "{name}");
    for ((got_name, column), want) in table.columns.iter().zip(expected) {
        assert_eq!(got_name, want[0].as_str().unwrap(), "{name}");
        assert_eq!(
            column.len() as u64,
            want[1].as_u64().unwrap(),
            "{name} {got_name}"
        );
        assert_eq!(
            double_digest(column.as_f64().unwrap()),
            want[2].as_str().unwrap(),
            "{name} {got_name}"
        );
    }
    let kv = &entry["kv"];
    let text = &table.metadata[kv["key"].as_str().unwrap()];
    assert_eq!(text.len() as u64, kv["bytes"].as_u64().unwrap(), "{name}");
    assert_eq!(
        sha256(text.as_bytes()),
        kv["sha256"].as_str().unwrap(),
        "{name}"
    );
    table
}

#[test]
fn every_shipped_footer_matches_the_frozen_references() {
    let references = json("references.json");
    let ctx = context();
    let mut checked = BTreeMap::<String, usize>::new();
    for (name, entry) in references["corpus_files"].as_object().unwrap() {
        let Some(path) = library_path(name).filter(|p| p.exists()) else {
            continue;
        };
        check_footer(name, entry, &ctx, &path);
        let library = name.rsplit_once('/').unwrap().0.to_owned();
        *checked.entry(library).or_default() += 1;
    }
    assert_eq!(ctx.memory_used(), 0);
    let total: usize = checked.values().sum();
    eprintln!("shipped footers checked: {checked:?} (total {total} of 431)");
    for (library, count) in &checked {
        assert_eq!(
            *count,
            references["corpus_totals"]["files_per_set"][library]
                .as_u64()
                .unwrap() as usize,
            "{library}"
        );
    }
}

/// Every shipped file, every column, bit for bit. Debug builds decode zstd
/// slowly, so they check five files (one per library); release builds check
/// all that are installed.
#[test]
fn every_shipped_file_matches_the_frozen_digests() {
    let references = json("references.json");
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let sample = [
        "paw/lda-sr/Cu.parquet",
        "paw/lda-fr/Ac.parquet",
        "paw/pbe-sr/H.parquet",
        "paw/pbe-fr/Pb.parquet",
        "upaw/lda-sr/Ag.parquet",
    ];
    let mut checked = 0usize;
    let mut bytes = 0u64;
    for (name, entry) in references["corpus_files"].as_object().unwrap() {
        if cfg!(debug_assertions) && !sample.contains(&name.as_str()) {
            continue;
        }
        let Some(path) = library_path(name).filter(|p| p.exists()) else {
            continue;
        };
        let table = check_digests(name, entry, &ctx, &path);
        drop(table);
        assert_eq!(ctx.memory_used(), 0, "{name}");
        bytes += entry["bytes"].as_u64().unwrap();
        checked += 1;
    }
    eprintln!(
        "shipped files with identical digests: {checked} ({} MB)",
        bytes >> 20
    );
    if checked == 0 {
        eprintln!("MANDACARU_PAW_PATH and MANDACARU_UPAW_PATH are unset: no shipped file was read");
    }
}

#[test]
fn resaving_a_shipped_file_keeps_every_digest() {
    let references = json("references.json");
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let dir = Directory::new();
    let mut checked = 0;
    for name in [
        "paw/lda-sr/Cu.parquet",
        "paw/pbe-fr/Pb.parquet",
        "paw/pbe-sr/H.parquet",
        "upaw/lda-sr/Ag.parquet",
    ] {
        let Some(path) = library_path(name).filter(|p| p.exists()) else {
            continue;
        };
        let entry = &references["corpus_files"][name];
        let table = check_digests(name, entry, &ctx, &path);
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            let out = dir.file("resaved.parquet");
            write_table(
                &ctx,
                &out,
                &table,
                &WriteOptions {
                    codec,
                    ..WriteOptions::default()
                },
            )
            .unwrap();
            let size = fs::metadata(&out).unwrap().len();
            let again = check_digests(name, entry, &ctx, &out);
            assert_eq!(again.metadata, table.metadata, "{name}");
            eprintln!(
                "{name}: original {} B, re-saved {} {} B",
                entry["bytes"],
                codec.name(),
                size
            );
            drop(again);
        }
        drop(table);
        assert_eq!(ctx.memory_used(), 0, "{name}");
        checked += 1;
    }
    if checked == 0 {
        eprintln!("no shipped library is set: nothing was re-saved");
    }
}

// ---------------------------------------------------------------------------
// The layout of the files this crate writes.
// ---------------------------------------------------------------------------

/// A minimal, independent walk of the Thrift compact protocol. It checks that
/// every struct's field ids strictly ascend (an out-of-order id broke a
/// foreign reader once) and returns the position after the value.
struct Walker<'a> {
    bytes: &'a [u8],
    at: usize,
    structs: usize,
    fields: usize,
}

impl Walker<'_> {
    fn byte(&mut self) -> u8 {
        let b = self.bytes[self.at];
        self.at += 1;
        b
    }

    fn varint(&mut self) -> u64 {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let b = self.byte();
            value |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return value;
            }
            shift += 7;
            assert!(shift < 70, "varint too long");
        }
    }

    fn value(&mut self, ty: u8) {
        match ty {
            1 | 2 => {} // booleans live in the field header
            3 => self.at += 1,
            4..=6 => {
                self.varint();
            }
            7 => self.at += 8,
            8 => {
                let n = self.varint() as usize;
                self.at += n;
            }
            9 | 10 => {
                let header = self.byte();
                let mut n = usize::from(header >> 4);
                if n == 15 {
                    n = self.varint() as usize;
                }
                let element = header & 0x0f;
                for _ in 0..n {
                    if element == 1 || element == 2 {
                        self.at += 1; // a boolean element is a byte
                    } else {
                        self.value(element);
                    }
                }
            }
            11 => {
                let n = self.varint() as usize;
                if n > 0 {
                    let types = self.byte();
                    for _ in 0..n {
                        self.value(types >> 4);
                        self.value(types & 0x0f);
                    }
                }
            }
            12 => self.strict_struct(),
            other => panic!("unknown Thrift type {other}"),
        }
    }

    fn strict_struct(&mut self) {
        self.structs += 1;
        let mut last = 0i64;
        loop {
            let header = self.byte();
            if header == 0 {
                return;
            }
            let delta = i64::from(header >> 4);
            let ty = header & 0x0f;
            let id = if delta == 0 {
                let zigzag = self.varint();
                ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64)
            } else {
                last + delta
            };
            assert!(id > last, "field id {id} does not ascend after {last}");
            last = id;
            self.fields += 1;
            self.value(ty);
        }
    }
}

/// Splits a file into its body, footer bytes and checks the frame.
fn frame(bytes: &[u8]) -> (&[u8], usize) {
    assert_eq!(&bytes[..4], MAGIC);
    assert_eq!(&bytes[bytes.len() - 4..], MAGIC);
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    (&bytes[n - 8 - footer_len..n - 8], n - 8 - footer_len)
}

fn utf8(strings: &[String]) -> Column {
    let ctx = context();
    Column::Utf8(Utf8Column::from_strs(&ctx, strings).unwrap())
}

fn layout_table(ctx: &Context, rows: usize) -> Table {
    let d: Vec<f64> = (0..rows).map(|i| (i % 97) as f64 * 0.5).collect();
    let e: Vec<f64> = (0..rows).map(|i| -(i as f64)).collect();
    let s: Vec<String> = (0..rows).map(|i| format!("IXYZ{}", i % 13)).collect();
    Table {
        columns: vec![
            ("pauli".to_owned(), utf8(&s)),
            (
                "real".to_owned(),
                Column::Double(ctx.copy_slice(&d).unwrap()),
            ),
            (
                "imag".to_owned(),
                Column::Double(ctx.copy_slice(&e).unwrap()),
            ),
        ],
        metadata: BTreeMap::from([
            ("mandacaru.version".to_owned(), "1".to_owned()),
            ("mandacaru.format".to_owned(), "x".to_owned()),
            ("a".to_owned(), "b".to_owned()),
        ]),
    }
}

#[test]
fn written_files_have_the_layout_other_readers_depend_on() {
    let ctx = context();
    let dir = Directory::new();
    let rows = 200_000usize;
    let table = layout_table(&ctx, rows);
    for codec in [Codec::Zstd, Codec::Uncompressed] {
        for (row_group_rows, page_rows) in [(1usize << 20, 1usize << 16), (150_000, 40_000)] {
            let path = dir.file("layout.parquet");
            write_table(
                &ctx,
                &path,
                &table,
                &WriteOptions {
                    codec,
                    row_group_rows,
                    page_rows,
                },
            )
            .unwrap();
            let bytes = fs::read(&path).unwrap();
            let (footer_bytes, footer_start) = frame(&bytes);

            // The independent walk: ascending ids in the footer and in every page header.
            let mut walker = Walker {
                bytes: footer_bytes,
                at: 0,
                structs: 0,
                fields: 0,
            };
            walker.strict_struct();
            assert_eq!(
                walker.at,
                footer_bytes.len(),
                "trailing bytes in the footer"
            );
            assert!(walker.structs > 10 && walker.fields > 30);

            let meta = FileMetaData::from_bytes(footer_bytes).unwrap();
            assert_eq!(meta.version, 1);
            assert_eq!(meta.created_by.as_deref(), Some(CREATED_BY));
            assert_eq!(meta.num_rows as usize, rows);

            // Schema: a root group and REQUIRED leaves.
            let root = &meta.schema[0];
            assert_eq!(root.name, "schema");
            assert_eq!(root.physical_type, None);
            assert_eq!(root.repetition, None);
            assert_eq!(root.num_children, Some(3));
            assert_eq!(meta.schema.len(), 4);
            for (leaf, (name, physical_type, text)) in meta.schema[1..].iter().zip([
                ("pauli", physical::BYTE_ARRAY, true),
                ("real", physical::DOUBLE, false),
                ("imag", physical::DOUBLE, false),
            ]) {
                assert_eq!(leaf.name, name);
                assert_eq!(leaf.physical_type, Some(physical_type));
                assert_eq!(leaf.repetition, Some(repetition::REQUIRED), "{name}");
                assert_eq!(leaf.type_length, None, "{name}: no type_length");
                assert_eq!(leaf.num_children, None);
                // Without both annotations pyarrow reads binary, DuckDB a BLOB.
                assert_eq!(
                    leaf.converted_type,
                    text.then_some(format::CONVERTED_UTF8),
                    "{name}"
                );
                assert_eq!(leaf.logical_string, text, "{name}");
            }

            // Key-value metadata: sorted, and nothing a foreign writer would add.
            let keys: Vec<&str> = meta
                .key_value_metadata
                .iter()
                .map(|kv| kv.key.as_str())
                .collect();
            assert_eq!(keys, ["a", "mandacaru.format", "mandacaru.version"]);
            assert!(meta.key_value_metadata.iter().all(|kv| kv.value.is_some()));

            // Row groups, chunks and pages.
            assert_eq!(meta.row_groups.len(), rows.div_ceil(row_group_rows));
            let mut total_rows = 0i64;
            for group in &meta.row_groups {
                assert!(group.num_rows as usize <= row_group_rows);
                total_rows += group.num_rows;
                assert_eq!(group.columns.len(), 3);
                let mut group_bytes = 0i64;
                for chunk in &group.columns {
                    let m = &chunk.meta_data;
                    assert!(chunk.file_path.is_none());
                    assert_eq!(m.num_values, group.num_rows);
                    assert!(m.dictionary_page_offset.is_none() && m.index_page_offset.is_none());
                    assert_eq!(m.data_page_offset, m.start_offset());
                    assert!(
                        m.codec == codec_id::UNCOMPRESSED
                            || (m.codec == codec_id::ZSTD && codec == Codec::Zstd)
                    );
                    assert_eq!(m.encodings, vec![encoding::PLAIN], "PLAIN only, no levels");
                    assert!(m.key_value_metadata.is_empty());
                    let (mut at, end) = (
                        m.start_offset() as usize,
                        m.start_offset() as usize + m.total_compressed_size as usize,
                    );
                    assert!(end <= footer_start);
                    let (mut values, mut compressed, mut uncompressed) = (0i64, 0i64, 0i64);
                    let mut page_count = 0;
                    while at < end {
                        let (header, header_len) = PageHeader::from_bytes(&bytes[at..end]).unwrap();
                        // A page header is a Thrift struct too.
                        let mut walker = Walker {
                            bytes: &bytes[at..at + header_len],
                            at: 0,
                            structs: 0,
                            fields: 0,
                        };
                        walker.strict_struct();
                        assert_eq!(walker.at, header_len);
                        assert_eq!(header.page_type, page_type::DATA_PAGE, "v1 data pages only");
                        assert!(header.dictionary_page.is_none() && header.data_page_v2.is_none());
                        let body_start = at + header_len;
                        let body_end = body_start + header.compressed_page_size as usize;
                        assert_eq!(
                            header.crc,
                            Some(
                                mandacaru_core::parquet::crc32::crc32(&bytes[body_start..body_end])
                                    as i32
                            ),
                            "every page carries the CRC32 of its stored body"
                        );
                        let data = header.data_page.unwrap();
                        assert_eq!(data.encoding, encoding::PLAIN);
                        assert!(data.num_values as usize <= page_rows);
                        assert!(data.num_values > 0);
                        values += i64::from(data.num_values);
                        compressed += (header_len + header.compressed_page_size as usize) as i64;
                        uncompressed +=
                            (header_len + header.uncompressed_page_size as usize) as i64;
                        if m.codec == codec_id::UNCOMPRESSED {
                            assert_eq!(header.compressed_page_size, header.uncompressed_page_size);
                        }
                        at += header_len + header.compressed_page_size as usize;
                        page_count += 1;
                    }
                    assert_eq!(at, end, "the pages fill the chunk exactly");
                    assert_eq!(values, m.num_values);
                    assert_eq!(compressed, m.total_compressed_size);
                    assert_eq!(uncompressed, m.total_uncompressed_size);
                    assert_eq!(page_count, (m.num_values as usize).div_ceil(page_rows));
                    group_bytes += m.total_uncompressed_size;
                }
                assert_eq!(group.total_byte_size, group_bytes);
            }
            assert_eq!(total_rows as usize, rows);
            // The first chunk starts right after the magic; chunks follow each other.
            let first = &meta.row_groups[0].columns[0].meta_data;
            assert_eq!(first.start_offset(), 4);
            let mut expected_start = 4i64;
            for group in &meta.row_groups {
                for chunk in &group.columns {
                    assert_eq!(chunk.meta_data.start_offset(), expected_start);
                    expected_start += chunk.meta_data.total_compressed_size;
                }
            }
            assert_eq!(
                expected_start as usize, footer_start,
                "no gap before the footer"
            );
            assert_eq!(read_table(&ctx, &path).unwrap().num_rows(), rows);
        }
    }
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_chunk_that_does_not_shrink_is_stored_uncompressed_and_still_reads() {
    let ctx = context();
    let dir = Directory::new();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let noise: Vec<f64> = (0..3000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64::from_bits(state)
        })
        .collect();
    let table = Table {
        columns: vec![(
            "noise".to_owned(),
            Column::Double(ctx.copy_slice(&noise).unwrap()),
        )],
        metadata: BTreeMap::new(),
    };
    let path = dir.file("noise.parquet");
    write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
    let bytes = fs::read(&path).unwrap();
    let (footer_bytes, _) = frame(&bytes);
    let meta = FileMetaData::from_bytes(footer_bytes).unwrap();
    let chunk = &meta.row_groups[0].columns[0].meta_data;
    assert_eq!(
        chunk.codec,
        codec_id::UNCOMPRESSED,
        "incompressible data stays raw"
    );
    let back = read_table(&ctx, &path).unwrap();
    assert_eq!(
        back.column("noise").unwrap().as_f64().unwrap().len(),
        noise.len()
    );
    for (a, b) in back
        .column("noise")
        .unwrap()
        .as_f64()
        .unwrap()
        .iter()
        .zip(&noise)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn the_dictionary_pages_of_a_foreign_file_are_not_copied_into_ours() {
    // Reading a dictionary-encoded pyarrow file and writing it back gives the
    // plain layout above, never a dictionary.
    let ctx = context();
    let dir = Directory::new();
    let source = fixtures().join("hamiltonian/pyarrow_multi_page_dictionary.parquet");
    let table = read_table(&ctx, &source).unwrap();
    let path = dir.file("plain.parquet");
    write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap();
    let bytes = fs::read(&path).unwrap();
    let (footer_bytes, _) = frame(&bytes);
    let meta = FileMetaData::from_bytes(footer_bytes).unwrap();
    for group in &meta.row_groups {
        for chunk in &group.columns {
            assert!(chunk.meta_data.dictionary_page_offset.is_none());
            assert_eq!(chunk.meta_data.encodings, vec![encoding::PLAIN]);
        }
    }
    let keys: Vec<&String> = table.metadata.keys().collect();
    let back = read_table(&ctx, &path).unwrap();
    assert_eq!(back.metadata.keys().collect::<Vec<_>>(), keys);
}

// ---------------------------------------------------------------------------
// Files for external readers.
// ---------------------------------------------------------------------------

/// Writes the files an outside reader should open (pyarrow, fastparquet,
/// pandas, DuckDB) into `MANDACARU_PARQUET_EMIT_DIR`. Ignored by default: it
/// asserts nothing about the readers, which live outside this crate; a script
/// opens the files afterwards. Run it with
/// `MANDACARU_PARQUET_EMIT_DIR=DIR cargo test --release -p mandacaru-core
/// --test parquet_interop -- --ignored emit`.
#[test]
#[ignore]
fn emit_files_for_external_readers() {
    use mandacaru_core::persistence::{Document, HAMILTONIAN};
    let Some(out) = library_root("MANDACARU_PARQUET_EMIT_DIR") else {
        eprintln!("MANDACARU_PARQUET_EMIT_DIR is unset: nothing emitted");
        return;
    };
    fs::create_dir_all(&out).unwrap();
    let ctx = Context::new(Some(1), 1500 << 20).unwrap();
    let letters = [b'I', b'X', b'Y', b'Z'];
    // Hamiltonians through the document layer: 0, 1, 4000 x 12, 100,000 x 24
    // and 1.1 million x 24 terms (two row groups).
    for (rows, qubits) in [
        (0usize, 6usize),
        (1, 4),
        (4000, 12),
        (100_000, 24),
        (1_100_000, 24),
    ] {
        let mask = (1u64 << (2 * qubits)) - 1;
        let terms: Vec<(String, f64, f64)> = (0..rows as u64)
            .map(|i| {
                let code = i.wrapping_mul(0x9E37_79B9_7F4A_7C15 | 1) & mask;
                let label: String = (0..qubits)
                    .map(|q| letters[((code >> (2 * q)) & 3) as usize] as char)
                    .collect();
                (
                    label,
                    (i as f64) * 1e-3 - 0.25,
                    if i % 5 == 0 { 0.5 } else { 0.0 },
                )
            })
            .collect();
        let labels: Vec<&str> = terms.iter().map(|t| t.0.as_str()).collect();
        let real: Vec<f64> = terms.iter().map(|t| t.1).collect();
        let imag: Vec<f64> = terms.iter().map(|t| t.2).collect();
        let mut metadata = BTreeMap::new();
        metadata.insert("mandacaru.format".to_owned(), HAMILTONIAN.to_owned());
        metadata.insert("mandacaru.version".to_owned(), "1".to_owned());
        metadata.insert("mandacaru.num_qubits".to_owned(), qubits.to_string());
        metadata.insert(
            "mandacaru.metadata".to_owned(),
            "{\"energy_unit\":\"Ha\",\"note\":\"café\"}".to_owned(),
        );
        let table = Table {
            columns: vec![
                (
                    "pauli".into(),
                    Column::Utf8(Utf8Column::from_strs(&ctx, &labels).unwrap()),
                ),
                (
                    "real".into(),
                    Column::Double(ctx.copy_slice(&real).unwrap()),
                ),
                (
                    "imag".into(),
                    Column::Double(ctx.copy_slice(&imag).unwrap()),
                ),
            ],
            metadata,
        };
        let document = Document::from_table(&ctx, &table, HAMILTONIAN, "'emit'").unwrap();
        drop(table);
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            if rows == 1_100_000 && codec == Codec::Uncompressed {
                continue;
            }
            let path = out.join(format!("hamiltonian_{rows}_{}.parquet", codec.name()));
            document.save_parquet(&ctx, &path, codec).unwrap();
        }
    }
    // Pseudopotential-shaped files: float64 columns and one large JSON entry,
    // re-saved from shipped files when the libraries are set.
    for (name, label) in [
        ("paw/lda-sr/Cu.parquet", "Cu"),
        ("upaw/lda-sr/Ag.parquet", "Ag_upaw"),
    ] {
        let Some(path) = library_path(name).filter(|p| p.exists()) else {
            continue;
        };
        let table = read_table(&ctx, &path).unwrap();
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            let target = out.join(format!("pseudopotential_{label}_{}.parquet", codec.name()));
            write_table(
                &ctx,
                &target,
                &table,
                &WriteOptions {
                    codec,
                    ..WriteOptions::default()
                },
            )
            .unwrap();
        }
    }
    // A foreign-shaped generic table with the special values.
    let values = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE / 4.0,
        1.5,
    ];
    let strings = ["", "I", "áéîõü", "∑∫∂", "🌵", "a\0b"];
    let table = Table {
        columns: vec![
            (
                "s".into(),
                Column::Utf8(Utf8Column::from_strs(&ctx, &strings).unwrap()),
            ),
            ("d".into(), Column::Double(ctx.copy_slice(&values).unwrap())),
        ],
        metadata: BTreeMap::from([
            ("k.one".to_owned(), "1".to_owned()),
            ("k.empty".to_owned(), String::new()),
        ]),
    };
    for codec in [Codec::Zstd, Codec::Uncompressed] {
        write_table(
            &ctx,
            &out.join(format!("generic_{}.parquet", codec.name())),
            &table,
            &WriteOptions {
                codec,
                ..WriteOptions::default()
            },
        )
        .unwrap();
    }
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
    eprintln!("emitted into {}", out.display());
}
