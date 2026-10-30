//! Hamiltonian document to Parquet table glue.
//!
//! A qubit-Hamiltonian document is a JSON-shaped payload; its Parquet form is
//! a three-column table (`pauli`, `real`, `imag`, rows sorted by label) plus
//! seven key-value entries in the footer:
//!
//! | key                             | value                                  |
//! |---------------------------------|----------------------------------------|
//! | `mandacaru.format`              | `mandacaru-qubit-hamiltonian`          |
//! | `mandacaru.version`             | `1`                                    |
//! | `mandacaru.num_qubits`          | the register width, as text            |
//! | `mandacaru.mapping`             | mapping name (default `jordan_wigner`) |
//! | `mandacaru.num_particles`       | compact JSON, `null` or `[alpha,beta]` |
//! | `mandacaru.n_spatial_orbitals`  | compact JSON, `null` or an integer     |
//! | `mandacaru.metadata`            | compact JSON object (default `{}`)     |
//!
//! The two conversions here only move data between the table and the payload.
//! Every rule about what a Hamiltonian may contain (tag, version, qubit limit,
//! energy unit, mapping, particle consistency, finite coefficients, Pauli
//! letters and lengths) stays in `Document::validate`, which runs on the
//! rebuilt payload exactly as it does for a JSON document.
//!
//! Memory: the payload built from a table is reserved from the `Context`
//! before it is built, and the reservation travels with the `Document`.

use crate::{
    parquet::{Column, Table, Utf8Column},
    persistence::{integer, HAMILTONIAN, MAX_FILE_QUBITS},
    product, Context, Error, Reservation, Result,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Footer key naming the document kind.
pub const KEY_FORMAT: &str = "mandacaru.format";
/// Footer key of the schema version.
pub const KEY_VERSION: &str = "mandacaru.version";
/// Footer key of the register width.
pub const KEY_NUM_QUBITS: &str = "mandacaru.num_qubits";
/// Footer key of the fermion-to-qubit mapping name.
pub const KEY_MAPPING: &str = "mandacaru.mapping";
/// Footer key of the particle counts (JSON).
pub const KEY_NUM_PARTICLES: &str = "mandacaru.num_particles";
/// Footer key of the spatial orbital count (JSON).
pub const KEY_N_SPATIAL_ORBITALS: &str = "mandacaru.n_spatial_orbitals";
/// Footer key of the free-form metadata object (JSON).
pub const KEY_METADATA: &str = "mandacaru.metadata";
/// Column holding the Pauli labels.
pub const COLUMN_PAULI: &str = "pauli";
/// Column holding the real parts.
pub const COLUMN_REAL: &str = "real";
/// Column holding the imaginary parts.
pub const COLUMN_IMAG: &str = "imag";

/// Schema version this build reads and writes.
const VERSION: u64 = 1;
/// Bytes charged per term for the JSON payload: one outer array element, the
/// inner three-element array, the label's `String` header and allocator slack.
const TERM_OVERHEAD: usize = 192;
/// Bytes charged per footer byte for the parsed JSON values of the metadata.
const KV_FACTOR: usize = 32;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// A Python-list style rendering of `names`, for error messages.
fn listing(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    format!("[{}]", quoted.join(", "))
}

fn key_value<'a>(table: &'a Table, key: &str) -> Option<&'a str> {
    table.metadata.get(key).map(String::as_str)
}

fn json_key(source: &str, key: &str, text: &str) -> Result<Value> {
    serde_json::from_str(text)
        .map_err(|e| invalid(format!("{source} has an unreadable {key} entry: {e}")))
}

/// Longest text of a particle-count or orbital-count entry. The legitimate
/// forms are `null`, an integer and `[a,b]`.
const SMALL_ENTRY_BYTES: usize = 256;

/// `json_key` for an entry that holds a count or a pair of counts.
fn small_json_key(source: &str, key: &str, text: &str) -> Result<Value> {
    if text.len() > SMALL_ENTRY_BYTES {
        return Err(invalid(format!(
            "{source} has an unreadable {key} entry: {} bytes where at most {SMALL_ENTRY_BYTES} are allowed",
            text.len()
        )));
    }
    json_key(source, key, text)
}

fn unsigned_key(source: &str, key: &str, text: Option<&str>) -> Result<u64> {
    let text = text.ok_or_else(|| invalid(format!("{source} lacks the {key} entry")))?;
    text.trim().parse::<u64>().map_err(|_| {
        invalid(format!(
            "{source} has a {key} entry that is not a nonnegative integer: {text:?}"
        ))
    })
}

/// Builds the Hamiltonian payload of `table`, reserving its memory first.
///
/// `source` names the file in messages (already quoted, for example
/// `'/path/h.parquet'`). Reads the footer entries and the three columns only;
/// no Hamiltonian rule is checked here except those needed to read the table
/// at all (the kind, the version ceiling, the entries and columns present with
/// the right types, finite coefficients that JSON can carry). The caller runs
/// `Document::validate` on the result.
pub(crate) fn payload_from_table(
    ctx: &Context,
    table: &Table,
    kind: &str,
    source: &str,
) -> Result<(Value, Reservation)> {
    if kind != HAMILTONIAN {
        return Err(invalid(format!(
            "{source} is a Parquet file; only a qubit Hamiltonian is stored as Parquet, not a {kind} snapshot"
        )));
    }
    if key_value(table, KEY_FORMAT) != Some(HAMILTONIAN) {
        return Err(invalid(format!(
            "{source} is not a Mandacaru qubit-Hamiltonian Parquet file (expected {KEY_FORMAT} = '{HAMILTONIAN}')"
        )));
    }
    let version = unsigned_key(source, KEY_VERSION, key_value(table, KEY_VERSION))?;
    if version > VERSION {
        return Err(invalid(format!(
            "{source} uses Hamiltonian format version {version}, but this build understands up to {VERSION}"
        )));
    }
    let num_qubits = unsigned_key(source, KEY_NUM_QUBITS, key_value(table, KEY_NUM_QUBITS))?;
    let mapping = key_value(table, KEY_MAPPING).unwrap_or("jordan_wigner");

    let found = table.column_names();
    let mut sorted = found.clone();
    sorted.sort_unstable();
    let require = |name: &str| {
        table.column(name).ok_or_else(|| {
            invalid(format!(
                "{source} is missing the '{name}' column (found {})",
                listing(&sorted)
            ))
        })
    };
    let (pauli, real, imag) = (
        require(COLUMN_PAULI)?,
        require(COLUMN_REAL)?,
        require(COLUMN_IMAG)?,
    );
    let labels = pauli.as_utf8().ok_or_else(|| {
        invalid(format!(
            "{source}: the '{COLUMN_PAULI}' column must hold UTF-8 text"
        ))
    })?;
    let real = real.as_f64().ok_or_else(|| {
        invalid(format!(
            "{source}: the '{COLUMN_REAL}' column must hold float64 values"
        ))
    })?;
    let imag = imag.as_f64().ok_or_else(|| {
        invalid(format!(
            "{source}: the '{COLUMN_IMAG}' column must hold float64 values"
        ))
    })?;
    let rows = labels.len();
    if real.len() != rows || imag.len() != rows {
        return Err(invalid(format!(
            "{source}: the '{COLUMN_PAULI}', '{COLUMN_REAL}' and '{COLUMN_IMAG}' columns differ in length"
        )));
    }

    let metadata_bytes: usize = table
        .metadata
        .values()
        .map(String::len)
        .try_fold(0usize, usize::checked_add)
        .ok_or(Error::Overflow)?;
    let per_term = TERM_OVERHEAD
        .checked_add(
            usize::try_from(num_qubits)
                .map_err(|_| Error::Overflow)?
                .min(MAX_FILE_QUBITS + 1),
        )
        .ok_or(Error::Overflow)?;
    let memory = ctx.reserve(
        product(&[rows, per_term])?
            .checked_add(product(&[metadata_bytes, KV_FACTOR])?)
            .ok_or(Error::Overflow)?,
    )?;

    // The memory is reserved above, before any footer entry is parsed into a
    // JSON tree (a value such as `[0,0,0,...]` expands about 16-fold).
    let num_particles = match key_value(table, KEY_NUM_PARTICLES) {
        Some(text) => small_json_key(source, KEY_NUM_PARTICLES, text)?,
        None => Value::Null,
    };
    let n_spatial_orbitals = match key_value(table, KEY_N_SPATIAL_ORBITALS) {
        Some(text) => small_json_key(source, KEY_N_SPATIAL_ORBITALS, text)?,
        None => Value::Null,
    };
    let metadata = match key_value(table, KEY_METADATA) {
        Some(text) => json_key(source, KEY_METADATA, text)?,
        None => json!({}),
    };
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(rows)
        .map_err(|_| Error::Allocation)?;
    for i in 0..rows {
        // A non-finite double has no JSON form; refuse it here with the row
        // instead of letting it become `null` and fail later without context.
        if !real[i].is_finite() || !imag[i].is_finite() {
            return Err(invalid(format!(
                "{source}: row {i} has a non-finite coefficient"
            )));
        }
        terms.push(json!([labels.get(i), real[i], imag[i]]));
    }
    let mut payload = Map::new();
    payload.insert("format".into(), json!(HAMILTONIAN));
    payload.insert("version".into(), json!(version));
    payload.insert("num_qubits".into(), json!(num_qubits));
    payload.insert("mapping".into(), json!(mapping));
    payload.insert("num_particles".into(), num_particles);
    payload.insert("n_spatial_orbitals".into(), n_spatial_orbitals);
    payload.insert("metadata".into(), metadata);
    payload.insert("terms".into(), Value::Array(terms));
    Ok((Value::Object(payload), memory))
}

/// The header of a Hamiltonian handed over as terms, not as a JSON payload.
pub struct TermsHeader<'a> {
    pub num_qubits: usize,
    pub mapping: &'a str,
    pub num_particles: Option<[usize; 2]>,
    pub n_spatial_orbitals: Option<usize>,
    /// The compact JSON text of the free-form metadata object.
    pub metadata_json: String,
}

/// Builds the Parquet table of a Hamiltonian given as `(label, real, imag)`
/// terms, with no JSON payload in between: the file is the one
/// `table_from_payload` gives for the same document. The caller guarantees
/// what the operator guarantees (labels of `num_qubits` letters I/X/Y/Z, the
/// particle counts consistent with the orbitals); what a table can still get
/// wrong, the register width and non-finite coefficients, is refused here.
pub fn table_from_terms(
    ctx: &Context,
    header: &TermsHeader<'_>,
    mut terms: Vec<(String, f64, f64)>,
) -> Result<Table> {
    if header.num_qubits > MAX_FILE_QUBITS {
        return Err(invalid("Pauli-string files support at most 50 qubits"));
    }
    if terms
        .iter()
        .any(|(_, re, im)| !re.is_finite() || !im.is_finite())
    {
        return Err(invalid("real coefficient must be finite"));
    }
    let optional = |value: Option<Value>| {
        serde_json::to_string(&value.unwrap_or(Value::Null)).map_err(|e| invalid(e.to_string()))
    };
    let mut metadata = BTreeMap::new();
    metadata.insert(KEY_FORMAT.to_owned(), HAMILTONIAN.to_owned());
    metadata.insert(KEY_VERSION.to_owned(), VERSION.to_string());
    metadata.insert(KEY_NUM_QUBITS.to_owned(), header.num_qubits.to_string());
    metadata.insert(KEY_MAPPING.to_owned(), header.mapping.to_owned());
    metadata.insert(
        KEY_NUM_PARTICLES.to_owned(),
        optional(header.num_particles.map(|p| json!(p)))?,
    );
    metadata.insert(
        KEY_N_SPATIAL_ORBITALS.to_owned(),
        optional(header.n_spatial_orbitals.map(|n| json!(n)))?,
    );
    metadata.insert(KEY_METADATA.to_owned(), header.metadata_json.clone());
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    let labels: Vec<&str> = terms.iter().map(|t| t.0.as_str()).collect();
    let pauli = Utf8Column::from_strs(ctx, &labels)?;
    drop(labels);
    let mut real = ctx.allocate::<f64>(terms.len())?;
    let mut imag = ctx.allocate::<f64>(terms.len())?;
    for ((slot_re, slot_im), term) in real
        .as_mut_slice()
        .iter_mut()
        .zip(imag.as_mut_slice())
        .zip(&terms)
    {
        *slot_re = term.1;
        *slot_im = term.2;
    }
    Ok(Table {
        columns: vec![
            (COLUMN_PAULI.to_owned(), Column::Utf8(pauli)),
            (COLUMN_REAL.to_owned(), Column::Double(real)),
            (COLUMN_IMAG.to_owned(), Column::Double(imag)),
        ],
        metadata,
    })
}

/// Builds the Parquet table of a Hamiltonian payload.
///
/// Rows are sorted by label bytes (`I < X < Y < Z`, the order the Python
/// writer used), so equal documents give equal files. Terms are written as
/// they are: duplicates are not merged. Everything that can be refused is
/// refused here, before any file exists.
pub(crate) fn table_from_payload(ctx: &Context, payload: &Value) -> Result<Table> {
    if payload["format"].as_str() != Some(HAMILTONIAN) {
        return Err(invalid(
            "only a qubit-Hamiltonian snapshot can be written as Parquet",
        ));
    }
    let version = integer(&payload["version"], "schema version")?;
    if version != VERSION as usize {
        return Err(invalid("unsupported snapshot schema version"));
    }
    let n = integer(&payload["num_qubits"], "qubit count")?;
    if n > MAX_FILE_QUBITS {
        return Err(invalid("Pauli-string files support at most 50 qubits"));
    }
    let rows = payload["terms"]
        .as_array()
        .ok_or_else(|| invalid("Pauli terms must be [label, real, imag] triples"))?;

    let mut metadata = BTreeMap::new();
    metadata.insert(KEY_FORMAT.to_owned(), HAMILTONIAN.to_owned());
    metadata.insert(KEY_VERSION.to_owned(), VERSION.to_string());
    metadata.insert(KEY_NUM_QUBITS.to_owned(), n.to_string());
    metadata.insert(
        KEY_MAPPING.to_owned(),
        payload
            .get("mapping")
            .and_then(Value::as_str)
            .unwrap_or("jordan_wigner")
            .to_owned(),
    );
    let compact = |value: Option<&Value>, default: Value| -> Result<String> {
        serde_json::to_string(value.filter(|v| !v.is_null()).unwrap_or(&default))
            .map_err(|e| invalid(e.to_string()))
    };
    metadata.insert(
        KEY_NUM_PARTICLES.to_owned(),
        compact(payload.get("num_particles"), Value::Null)?,
    );
    metadata.insert(
        KEY_N_SPATIAL_ORBITALS.to_owned(),
        compact(payload.get("n_spatial_orbitals"), Value::Null)?,
    );
    metadata.insert(
        KEY_METADATA.to_owned(),
        compact(payload.get("metadata"), json!({}))?,
    );

    // One working vector of (label, real, imag) per row, charged first.
    let _work = ctx.reserve(product(&[rows.len(), 40])?)?;
    let mut entries: Vec<(&str, f64, f64)> = Vec::new();
    entries
        .try_reserve_exact(rows.len())
        .map_err(|_| Error::Allocation)?;
    for row in rows {
        let triple = row
            .as_array()
            .filter(|r| r.len() == 3)
            .ok_or_else(|| invalid("Pauli terms must be [label, real, imag] triples"))?;
        let label = triple[0]
            .as_str()
            .ok_or_else(|| invalid("Pauli label must be a string"))?;
        let part = |value: &Value, name: &str| {
            value
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid(format!("{name} must be finite")))
        };
        entries.push((
            label,
            part(&triple[1], "real coefficient")?,
            part(&triple[2], "imaginary coefficient")?,
        ));
    }
    if !entries.windows(2).all(|w| w[0].0 <= w[1].0) {
        entries.sort_by(|a, b| a.0.cmp(b.0));
    }

    let labels: Vec<&str> = entries.iter().map(|e| e.0).collect();
    let pauli = Utf8Column::from_strs(ctx, &labels)?;
    drop(labels);
    let mut real = ctx.allocate::<f64>(entries.len())?;
    let mut imag = ctx.allocate::<f64>(entries.len())?;
    for ((slot_re, slot_im), entry) in real
        .as_mut_slice()
        .iter_mut()
        .zip(imag.as_mut_slice())
        .zip(&entries)
    {
        *slot_re = entry.1;
        *slot_im = entry.2;
    }
    Ok(Table {
        columns: vec![
            (COLUMN_PAULI.to_owned(), Column::Utf8(pauli)),
            (COLUMN_REAL.to_owned(), Column::Double(real)),
            (COLUMN_IMAG.to_owned(), Column::Double(imag)),
        ],
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parquet::{self, Codec},
        persistence::Document,
    };
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    static SERIAL: AtomicUsize = AtomicUsize::new(0);

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "mandacaru-snapshot-table-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            Self(p)
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

    fn context() -> Context {
        Context::new(Some(1), 256 << 20).unwrap()
    }

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/parquet")
    }

    fn payload() -> Value {
        json!({
            "format": HAMILTONIAN,
            "version": 1,
            "num_qubits": 4,
            "mapping": "jordan_wigner",
            "num_particles": [1, 1],
            "n_spatial_orbitals": 2,
            "metadata": {"energy_unit": "Ha", "driver": "test", "note": "caf\u{e9} \u{1F335}"},
            "terms": [
                ["IIII", 0.75, 0.0],
                ["IIIZ", -0.125, 0.0],
                ["XXYY", 0.0625, -0.5],
                ["ZZII", 1.5, 2.0],
            ],
        })
    }

    fn document(ctx: &Context, payload: &Value) -> Document {
        Document::parse(ctx, &payload.to_string(), HAMILTONIAN).unwrap()
    }

    #[test]
    fn json_and_parquet_documents_are_equal_after_validate() {
        let ctx = context();
        let dir = Directory::new();
        {
            let original = document(&ctx, &payload());
            let json_path = dir.file("h.json");
            original.save(&json_path).unwrap();
            for (name, codec) in [
                ("zstd.parquet", Codec::Zstd),
                ("none.parquet", Codec::Uncompressed),
            ] {
                let path = dir.file(name);
                original.save_parquet(&ctx, &path, codec).unwrap();
                let from_parquet = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
                let from_json = Document::read(&ctx, &json_path, HAMILTONIAN).unwrap();
                assert_eq!(from_parquet.payload(), from_json.payload());
                assert_eq!(from_parquet.payload(), original.payload());
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn writing_is_deterministic_and_sorts_rows() {
        let ctx = context();
        let dir = Directory::new();
        let mut shuffled = payload();
        shuffled["terms"].as_array_mut().unwrap().reverse();
        let sorted = document(&ctx, &payload());
        let reversed = document(&ctx, &shuffled);
        let (a, b, c) = (
            dir.file("a.parquet"),
            dir.file("b.parquet"),
            dir.file("c.parquet"),
        );
        sorted.save_parquet(&ctx, &a, Codec::Zstd).unwrap();
        sorted.save_parquet(&ctx, &b, Codec::Zstd).unwrap();
        reversed.save_parquet(&ctx, &c, Codec::Zstd).unwrap();
        assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
        assert_eq!(fs::read(&a).unwrap(), fs::read(&c).unwrap());
        let back = Document::read(&ctx, &c, HAMILTONIAN).unwrap();
        assert_eq!(back.payload(), sorted.payload());
        drop((back, sorted, reversed));
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn footer_entries_are_the_documented_compact_text() {
        let ctx = context();
        let dir = Directory::new();
        let path = dir.file("h.parquet");
        document(&ctx, &payload())
            .save_parquet(&ctx, &path, Codec::Zstd)
            .unwrap();
        let footer = parquet::read_metadata(&ctx, &path).unwrap();
        assert_eq!(footer.num_rows, 4);
        assert_eq!(footer.created_by, parquet::CREATED_BY);
        assert_eq!(footer.metadata[KEY_FORMAT], HAMILTONIAN);
        assert_eq!(footer.metadata[KEY_VERSION], "1");
        assert_eq!(footer.metadata[KEY_NUM_QUBITS], "4");
        assert_eq!(footer.metadata[KEY_MAPPING], "jordan_wigner");
        assert_eq!(footer.metadata[KEY_NUM_PARTICLES], "[1,1]");
        assert_eq!(footer.metadata[KEY_N_SPATIAL_ORBITALS], "2");
        assert!(footer.metadata[KEY_METADATA].contains("\"energy_unit\":\"Ha\""));
        assert_eq!(footer.metadata.len(), 7);
    }

    #[test]
    fn zero_terms_keeps_the_register_width() {
        let ctx = context();
        let dir = Directory::new();
        let mut empty = payload();
        empty["terms"] = json!([]);
        empty["num_qubits"] = json!(4);
        let doc = document(&ctx, &empty);
        for codec in [Codec::Zstd, Codec::Uncompressed] {
            let path = dir.file(&format!("empty-{}.parquet", codec.name()));
            doc.save_parquet(&ctx, &path, codec).unwrap();
            let back = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
            assert_eq!(back.payload()["num_qubits"], json!(4));
            assert_eq!(back.payload()["terms"], json!([]));
        }
        drop(doc);
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn a_failed_conversion_creates_no_file() {
        let ctx = context();
        let dir = Directory::new();
        // 51 qubits cannot even be parsed.
        let mut wide = payload();
        wide["num_qubits"] = json!(51);
        wide["num_particles"] = Value::Null;
        wide["n_spatial_orbitals"] = Value::Null;
        wide["terms"] = json!([]);
        assert!(Document::parse(&ctx, &wide.to_string(), HAMILTONIAN).is_err());
        // A document altered after validation (a non-finite coefficient has no
        // JSON form, so `null` stands for it) and a 51-qubit register are
        // refused by the conversion, before the stage file exists.
        let mut nan = document(&ctx, &payload());
        nan.payload["terms"][1][1] = Value::Null;
        let path = dir.file("nan.parquet");
        let error = nan.save_parquet(&ctx, &path, Codec::Zstd).unwrap_err();
        assert!(error.to_string().contains("must be finite"), "{error}");
        let mut too_wide = document(&ctx, &payload());
        too_wide.payload["num_qubits"] = json!(51);
        let path51 = dir.file("wide.parquet");
        let error = too_wide
            .save_parquet(&ctx, &path51, Codec::Zstd)
            .unwrap_err();
        assert!(error.to_string().contains("at most 50 qubits"), "{error}");
        assert!(!path.exists() && !path51.exists());
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
        let mut kind = document(&ctx, &payload());
        kind.payload["format"] = json!("mandacaru-wavefunction-checkpoint");
        assert!(kind.save_parquet(&ctx, &path, Codec::Zstd).is_err());
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[test]
    fn a_failing_write_keeps_the_old_file() {
        let ctx = context();
        let dir = Directory::new();
        let path = dir.file("h.parquet");
        fs::write(&path, b"previous").unwrap();
        let mut bad = document(&ctx, &payload());
        bad.payload["terms"][0][0] = json!(7);
        assert!(bad.save_parquet(&ctx, &path, Codec::Zstd).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    fn read_error(name: &str) -> String {
        let ctx = context();
        let path = fixtures().join(name);
        let error = Document::read(&ctx, &path, HAMILTONIAN)
            .err()
            .unwrap_or_else(|| panic!("{name} should not load"));
        assert_eq!(ctx.memory_used(), 0, "{name}");
        error.to_string()
    }

    #[test]
    fn a_missing_column_names_what_was_found() {
        let message = read_error("negative/hamiltonian_missing_imag.parquet");
        assert!(
            message.ends_with("is missing the 'imag' column (found ['pauli', 'real'])"),
            "{message}"
        );
        assert!(message.starts_with('\''), "{message}");
    }

    #[test]
    fn a_foreign_file_is_not_a_hamiltonian() {
        for name in [
            "negative/hamiltonian_no_kv.parquet",
            "negative/hamiltonian_kv_wrong_format.parquet",
            "negative/hamiltonian_kv_missing_format.parquet",
        ] {
            let message = read_error(name);
            assert!(
                message.contains("is not a Mandacaru qubit-Hamiltonian Parquet file"),
                "{name}: {message}"
            );
        }
    }

    #[test]
    fn the_hamiltonian_rules_still_run_on_a_parquet_payload() {
        let expected = [
            (
                "hamiltonian_qubits_51",
                "Pauli-string files support at most 50 qubits",
            ),
            ("hamiltonian_bad_letter", "invalid Pauli letter"),
            (
                "hamiltonian_energy_unit_ev",
                "snapshot energies must be in Ha",
            ),
            (
                "hamiltonian_kv_version_0",
                "unsupported snapshot schema version",
            ),
            (
                "hamiltonian_kv_version_2",
                "uses Hamiltonian format version 2, but this build understands up to 1",
            ),
            (
                "hamiltonian_kv_unknown_mapping",
                "unsupported snapshot mapping",
            ),
            (
                "hamiltonian_kv_num_qubits_mismatch",
                "snapshot qubits disagree with mapping and orbitals",
            ),
            (
                "hamiltonian_wrong_pauli_length",
                "Pauli labels must match the register and coefficients must be finite",
            ),
            (
                "hamiltonian_kv_missing_version",
                "lacks the mandacaru.version entry",
            ),
            (
                "hamiltonian_kv_missing_num_qubits",
                "lacks the mandacaru.num_qubits entry",
            ),
            (
                "hamiltonian_kv_metadata_not_json",
                "unreadable mandacaru.metadata entry",
            ),
            ("hamiltonian_nonfinite_inf", "non-finite coefficient"),
        ];
        for (name, fragment) in expected {
            let message = read_error(&format!("negative/{name}.parquet"));
            assert!(message.contains(fragment), "{name}: {message}");
        }
        for name in [
            "null_in_pauli_pyarrow",
            "null_in_real_pyarrow",
            "null_in_real_fastparquet",
        ] {
            let message = read_error(&format!("negative/{name}.parquet"));
            assert!(message.contains("null"), "{name}: {message}");
        }
    }

    #[test]
    fn a_checkpoint_kind_refuses_a_parquet_file() {
        let ctx = context();
        let path = fixtures().join("hamiltonian/fastparquet_zstd.parquet");
        let error = Document::read(&ctx, &path, crate::persistence::CHECKPOINT)
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("qubit Hamiltonian is stored as Parquet"));
        assert_eq!(ctx.memory_used(), 0);
    }

    #[test]
    fn missing_optional_entries_take_their_defaults() {
        let ctx = context();
        let path = fixtures().join("hamiltonian/semantic_kv_defaults_only_required.parquet");
        let doc = Document::read(&ctx, &path, HAMILTONIAN).unwrap();
        let p = doc.payload();
        assert_eq!(p["mapping"], json!("jordan_wigner"));
        assert!(p["num_particles"].is_null() && p["n_spatial_orbitals"].is_null());
        assert_eq!(p["metadata"], json!({}));
    }

    #[test]
    fn the_budget_refuses_before_the_payload_is_built() {
        let ctx = context();
        let dir = Directory::new();
        let mut big = payload();
        let terms: Vec<Value> = (0..2000)
            .map(|i| {
                json!([
                    format!("{:04}", i)
                        .replace(['8', '9'], "X")
                        .replace(['0'], "I")
                        .replace('1', "Y")
                        .replace('2', "Z")
                        .replace('3', "X")
                        .replace(['4', '5', '6', '7'], "Z"),
                    0.5,
                    0.0
                ])
            })
            .collect();
        big["terms"] = Value::Array(terms);
        let path = dir.file("big.parquet");
        document(&ctx, &big)
            .save_parquet(&ctx, &path, Codec::Zstd)
            .unwrap();
        // The table sits in the roomy context; the payload built from it is
        // charged to the small one and refused before it exists.
        let roomy = context();
        let table = parquet::read_table(&roomy, &path).unwrap();
        let small = Context::new(Some(1), 100 << 10).unwrap();
        match Document::from_table(&small, &table, HAMILTONIAN, "'big.parquet'") {
            Err(Error::Budget { .. }) => {}
            other => panic!("expected a budget error, got {:?}", other.err()),
        }
        assert_eq!(small.memory_used(), 0);
        drop(table);
        // A whole-file read under a budget below the table is refused too.
        let tiny = Context::new(Some(1), 8 << 10).unwrap();
        match Document::read(&tiny, &path, HAMILTONIAN) {
            Err(Error::Budget { .. }) => {}
            other => panic!("expected a budget error, got {:?}", other.err()),
        }
        assert_eq!(tiny.memory_used(), 0);
        assert_eq!(roomy.memory_used(), 0);
    }

    // ----- bit-for-bit comparison with the frozen old-reader results -----

    fn python_hex(x: f64) -> String {
        let bits = x.to_bits();
        let sign = if bits >> 63 == 1 { "-" } else { "" };
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

    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut message = data.to_vec();
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in message.chunks(64) {
            let mut w = [0u32; 64];
            for (i, word) in block.chunks(4).enumerate() {
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
            let mut v = h;
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
            for (a, b) in h.iter_mut().zip(v) {
                *a = a.wrapping_add(b);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }

    #[test]
    fn the_sha256_helper_is_correct() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(python_hex(0.75), "0x1.8000000000000p-1");
        assert_eq!(python_hex(0.0), "0x0.0p+0");
        assert_eq!(python_hex(-2.0), "-0x1.0000000000000p+1");
    }

    /// Terms of a payload in the old reader's shape: duplicate labels summed in
    /// file order, sorted by label, as `[label, real.hex(), imag.hex()]`.
    fn merged_terms(payload: &Value) -> Vec<(String, String, String)> {
        let mut sums: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for term in payload["terms"].as_array().unwrap() {
            let entry = sums
                .entry(term[0].as_str().unwrap().to_owned())
                .or_default();
            entry.0 += term[1].as_f64().unwrap();
            entry.1 += term[2].as_f64().unwrap();
        }
        sums.into_iter()
            .map(|(label, (re, im))| (label, python_hex(re), python_hex(im)))
            .collect()
    }

    #[test]
    fn every_frozen_hamiltonian_fixture_matches_the_old_reader_bit_for_bit() {
        let frozen: Value =
            serde_json::from_slice(&fs::read(fixtures().join("old_api_results.json")).unwrap())
                .unwrap();
        let mut compared = 0;
        let mut names: Vec<_> = frozen["fixtures"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        names.sort();
        for name in names.iter().filter(|n| n.starts_with("hamiltonian/")) {
            let results = &frozen["fixtures"][name]["load_hamiltonian"];
            let ctx = context();
            let path = fixtures().join(name);
            let loaded = Document::read(&ctx, &path, HAMILTONIAN);
            let expected = ["pyarrow", "fastparquet"]
                .iter()
                .map(|engine| &results[*engine])
                .find(|r| r["ok"] == json!(true));
            let Some(expected) = expected else {
                // Neither old engine could load it (unannotated binary, missing
                // optional entries): the native reader is more tolerant, never
                // wrong. Only check that it loads.
                assert!(loaded.is_ok(), "{name}: {:?}", loaded.err());
                continue;
            };
            let expected = &expected["result"];
            let doc = loaded.unwrap_or_else(|e| panic!("{name}: {e}"));
            let p = doc.payload();
            assert_eq!(p["num_qubits"], expected["num_qubits"], "{name}");
            assert_eq!(p["mapping"], expected["mapping"], "{name}");
            assert_eq!(p["num_particles"], expected["num_particles"], "{name}");
            assert_eq!(
                p["n_spatial_orbitals"], expected["n_spatial_orbitals"],
                "{name}"
            );
            assert_eq!(p["metadata"], expected["metadata"], "{name}");
            let terms = merged_terms(p);
            assert_eq!(
                terms.len() as u64,
                expected["n_terms"].as_u64().unwrap(),
                "{name}"
            );
            let text: Vec<String> = terms
                .iter()
                .map(|(label, re, im)| format!("{label} {re} {im}"))
                .collect();
            assert_eq!(
                sha256_hex(text.join("\n").as_bytes()),
                expected["terms_sha256"].as_str().unwrap(),
                "{name}"
            );
            if let Some(list) = expected["terms"].as_array() {
                assert_eq!(list.len(), terms.len(), "{name}");
                for (want, got) in list.iter().zip(&terms) {
                    assert_eq!(want, &json!([got.0, got.1, got.2]), "{name}");
                }
            }
            drop(doc);
            assert_eq!(ctx.memory_used(), 0, "{name}");
            compared += 1;
        }
        assert!(compared >= 20, "only {compared} fixtures compared");
    }
}
