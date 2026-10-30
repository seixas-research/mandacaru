//! Parquet round trips, determinism, atomic writes, budgets and the frozen
//! fixtures, through the public API of `mandacaru_core::parquet`.
//!
//! Hostile input lives in `parquet_errors.rs`; foreign writers, the shipped
//! libraries and the layout other readers rely on live in
//! `parquet_interop.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use mandacaru_core::parquet::{
    self, format::FileMetaData, read_metadata, read_table, write_table, Codec, Column, ColumnKind,
    Table, Utf8Column, WriteOptions, CREATED_BY,
};
use mandacaru_core::{Context, Error};

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

static SERIAL: AtomicUsize = AtomicUsize::new(0);

/// A scratch directory removed on drop, also when a test panics.
struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mandacaru-parquet-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn entries(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
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

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

enum Spec<'a> {
    D(&'a [f64]),
    S(&'a [String]),
}

fn try_make(
    ctx: &Context,
    columns: &[(&str, Spec)],
    metadata: &[(&str, &str)],
) -> Result<Table, Error> {
    let mut built = Vec::new();
    for (name, spec) in columns {
        let column = match spec {
            Spec::D(values) => Column::Double(ctx.copy_slice(values)?),
            Spec::S(values) => Column::Utf8(Utf8Column::from_strs(ctx, values)?),
        };
        built.push(((*name).to_owned(), column));
    }
    Ok(Table {
        columns: built,
        metadata: metadata
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
    })
}

fn make(ctx: &Context, columns: &[(&str, Spec)], metadata: &[(&str, &str)]) -> Table {
    try_make(ctx, columns, metadata).unwrap()
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| (*s).to_owned()).collect()
}

/// Bit-exact equality of two tables: names, kinds, every double's bits, every
/// string's bytes and the metadata.
fn assert_same(a: &Table, b: &Table, what: &str) {
    assert_eq!(a.column_names(), b.column_names(), "{what}: column names");
    assert_eq!(a.metadata, b.metadata, "{what}: metadata");
    for ((name, left), (_, right)) in a.columns.iter().zip(&b.columns) {
        match (left, right) {
            (Column::Double(x), Column::Double(y)) => {
                let (x, y) = (x.as_slice(), y.as_slice());
                assert_eq!(x.len(), y.len(), "{what}: {name} rows");
                for (i, (p, q)) in x.iter().zip(y).enumerate() {
                    assert_eq!(p.to_bits(), q.to_bits(), "{what}: {name}[{i}]");
                }
            }
            (Column::Utf8(x), Column::Utf8(y)) => {
                assert_eq!(x.len(), y.len(), "{what}: {name} rows");
                for i in 0..x.len() {
                    assert_eq!(x.bytes(i), y.bytes(i), "{what}: {name}[{i}]");
                }
            }
            _ => panic!("{what}: {name} changed kind"),
        }
    }
}

fn options(codec: Codec, row_group_rows: usize, page_rows: usize) -> WriteOptions {
    WriteOptions {
        codec,
        row_group_rows,
        page_rows,
    }
}

const CODECS: [Codec; 2] = [Codec::Zstd, Codec::Uncompressed];

// ---------------------------------------------------------------------------
// Round trips.
// ---------------------------------------------------------------------------

fn special_doubles() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8_0000_dead_beef),
        f64::from_bits(0xfff4_0000_0000_0001),
        f64::from_bits(1),
        f64::from_bits(0x800f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 4.0,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        std::f64::consts::PI,
        1.0 / 3.0,
    ]
}

fn special_strings() -> Vec<String> {
    vec![
        String::new(),
        "I".into(),
        "IIXZ".into(),
        "áéîõü".into(),
        "∑∫∂".into(),
        "🌵 mandacaru".into(),
        "tab\tnew\nline\r\n".into(),
        "nul\0inside\0".into(),
        "\0".into(),
        "a".repeat(10_000),
        "ß".repeat(3000),
        "end".into(),
        String::new(),
        "\u{10ffff}\u{ffff}\u{80}\u{7f}".into(),
        "\u{feff}bom".into(),
    ]
}

#[test]
fn special_values_round_trip_bit_exact_in_every_codec() {
    let ctx = context();
    let dir = Directory::new();
    {
        let d = special_doubles();
        let reversed: Vec<f64> = d.iter().rev().copied().collect();
        let mut s = special_strings();
        s.truncate(d.len());
        while s.len() < d.len() {
            s.push("pad".into());
        }
        let table = make(
            &ctx,
            &[
                ("s", Spec::S(&s)),
                ("d", Spec::D(&d)),
                ("e", Spec::D(&reversed)),
            ],
            &[
                ("k.one", "1"),
                ("k.empty", ""),
                ("k.unicode", "ção 🌵"),
                ("mandacaru.format", "x"),
            ],
        );
        for codec in CODECS {
            let path = dir.file(&format!("special-{}.parquet", codec.name()));
            write_table(&ctx, &path, &table, &options(codec, 1 << 20, 1 << 16)).unwrap();
            let back = read_table(&ctx, &path).unwrap();
            assert_same(&table, &back, codec.name());
            assert_eq!(back.num_rows(), d.len());
            let footer = read_metadata(&ctx, &path).unwrap();
            assert_eq!(footer.num_rows as usize, d.len());
            assert_eq!(footer.num_row_groups, 1);
            assert_eq!(footer.created_by, CREATED_BY);
            assert_eq!(
                footer.columns,
                vec![
                    ("s".to_owned(), ColumnKind::Utf8),
                    ("d".to_owned(), ColumnKind::Double),
                    ("e".to_owned(), ColumnKind::Double),
                ]
            );
            assert_eq!(footer.metadata, table.metadata);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn every_double_bit_pattern_survives() {
    let ctx = context();
    let dir = Directory::new();
    {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let mut d: Vec<f64> = (0..70_000).map(|_| f64::from_bits(rng.next())).collect();
        // Exponent extremes and every subnormal-boundary pattern.
        for exp in [0u64, 1, 0x3ff, 0x7fe, 0x7ff] {
            for sign in [0u64, 1] {
                for frac in [0u64, 1, 0x8_0000_0000_0000, 0xf_ffff_ffff_ffff] {
                    d.push(f64::from_bits((sign << 63) | (exp << 52) | frac));
                }
            }
        }
        let table = make(&ctx, &[("d", Spec::D(&d))], &[]);
        for codec in CODECS {
            let path = dir.file("bits.parquet");
            write_table(&ctx, &path, &table, &options(codec, 40_000, 1 << 12)).unwrap();
            assert_same(&table, &read_table(&ctx, &path).unwrap(), codec.name());
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

fn random_text(rng: &mut Rng) -> String {
    const POOL: [&str; 12] = [
        "I", "X", "Y", "Z", "é", "ß", "∑", "🌵", "\0", "\n", " ", "abc",
    ];
    let length = match rng.below(4) {
        0 => 0,
        1 => rng.below(5),
        2 => rng.below(40),
        _ => rng.below(400),
    };
    (0..length).map(|_| POOL[rng.below(POOL.len())]).collect()
}

#[test]
fn random_tables_round_trip_across_row_group_and_page_sizes() {
    let ctx = context();
    let dir = Directory::new();
    {
        let mut rng = Rng(0xfeed_face_cafe_beef);
        for rows in [0usize, 1, 2, 7, 8, 9, 63, 64, 65, 100, 1001, 5000] {
            let d: Vec<f64> = (0..rows).map(|_| f64::from_bits(rng.next())).collect();
            let e: Vec<f64> = (0..rows).map(|i| (i as f64) * 0.25 - 3.0).collect();
            let s: Vec<String> = (0..rows).map(|_| random_text(&mut rng)).collect();
            let t: Vec<String> = (0..rows).map(|i| format!("row-{}", i % 17)).collect();
            let table = make(
                &ctx,
                &[
                    ("d", Spec::D(&d)),
                    ("s", Spec::S(&s)),
                    ("e", Spec::D(&e)),
                    ("t", Spec::S(&t)),
                ],
                &[("rows", &rows.to_string()), ("z", "last"), ("a", "first")],
            );
            for (row_group_rows, page_rows) in [
                (1usize, 1usize),
                (2, 1),
                (3, 2),
                (7, 3),
                (8, 8),
                (64, 5),
                (1000, 64),
                (1 << 20, 1 << 16),
            ] {
                for codec in CODECS {
                    let path = dir.file("t.parquet");
                    let opts = options(codec, row_group_rows, page_rows);
                    write_table(&ctx, &path, &table, &opts).unwrap();
                    let back = read_table(&ctx, &path).unwrap();
                    let label = format!(
                        "{rows} rows, groups of {row_group_rows}, pages of {page_rows}, {}",
                        codec.name()
                    );
                    assert_same(&table, &back, &label);
                    let footer = read_metadata(&ctx, &path).unwrap();
                    assert_eq!(footer.num_rows as usize, rows, "{label}");
                    assert_eq!(
                        footer.num_row_groups,
                        rows.div_ceil(row_group_rows),
                        "{label}"
                    );
                    assert_eq!(footer.columns.len(), 4, "{label}");
                }
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
    assert_eq!(dir.entries(), ["t.parquet"]);
}

#[test]
fn an_empty_table_keeps_its_columns_and_metadata() {
    let ctx = context();
    let dir = Directory::new();
    {
        let table = make(
            &ctx,
            &[
                ("pauli", Spec::S(&[])),
                ("real", Spec::D(&[])),
                ("imag", Spec::D(&[])),
            ],
            &[("mandacaru.num_qubits", "6")],
        );
        for codec in CODECS {
            let path = dir.file("empty.parquet");
            write_table(&ctx, &path, &table, &options(codec, 1 << 20, 1 << 16)).unwrap();
            let back = read_table(&ctx, &path).unwrap();
            assert_same(&table, &back, "empty");
            assert_eq!(back.num_rows(), 0);
            let footer = read_metadata(&ctx, &path).unwrap();
            assert_eq!(footer.num_rows, 0);
            assert_eq!(footer.metadata["mandacaru.num_qubits"], "6");
            assert_eq!(footer.columns.len(), 3);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_five_megabyte_metadata_value_round_trips() {
    // The largest shipped key-value value is 4,477,748 bytes.
    let ctx = context();
    let dir = Directory::new();
    {
        let big: String = (0..5_000_000usize)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let d = [1.0, 2.0];
        let table = make(
            &ctx,
            &[("d", Spec::D(&d))],
            &[("big", &big), ("small", "x")],
        );
        for codec in CODECS {
            let path = dir.file("kv.parquet");
            write_table(&ctx, &path, &table, &options(codec, 1 << 20, 1 << 16)).unwrap();
            let back = read_table(&ctx, &path).unwrap();
            assert_eq!(back.metadata["big"].len(), 5_000_000);
            assert_same(&table, &back, "big kv");
            assert_eq!(read_metadata(&ctx, &path).unwrap().metadata, table.metadata);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn compression_shrinks_what_compresses_and_leaves_noise_alone() {
    let ctx = context();
    let dir = Directory::new();
    {
        let steady: Vec<f64> = (0..50_000).map(|i| f64::from(i % 10)).collect();
        let mut rng = Rng(7);
        let noise: Vec<f64> = (0..50_000).map(|_| f64::from_bits(rng.next())).collect();
        let table = make(
            &ctx,
            &[("steady", Spec::D(&steady)), ("noise", Spec::D(&noise))],
            &[],
        );
        let zstd = dir.file("zstd.parquet");
        let raw = dir.file("raw.parquet");
        write_table(&ctx, &zstd, &table, &options(Codec::Zstd, 1 << 20, 1 << 16)).unwrap();
        write_table(
            &ctx,
            &raw,
            &table,
            &options(Codec::Uncompressed, 1 << 20, 1 << 16),
        )
        .unwrap();
        let (z, r) = (
            fs::metadata(&zstd).unwrap().len(),
            fs::metadata(&raw).unwrap().len(),
        );
        assert!(z < r, "zstd {z} bytes, raw {r}");
        // Noise cannot shrink, so the file cannot be much below the raw payload.
        assert!(z as f64 > 0.5 * 8.0 * 50_000.0);
        assert_same(&table, &read_table(&ctx, &zstd).unwrap(), "zstd");
        assert_same(&table, &read_table(&ctx, &raw).unwrap(), "raw");
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The codec stored for each column chunk of the first row group, by column.
fn chunk_codecs(bytes: &[u8]) -> Vec<(String, i32)> {
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let meta = FileMetaData::from_bytes(&bytes[n - 8 - footer_len..n - 8]).unwrap();
    meta.row_groups[0]
        .columns
        .iter()
        .map(|c| (c.meta_data.path_in_schema[0].clone(), c.meta_data.codec))
        .collect()
}

#[test]
fn a_tiny_tail_page_does_not_decide_the_codec_of_the_chunk() {
    let ctx = context();
    let dir = Directory::new();
    {
        let page = 1usize << 16;
        let mut rng = Rng(11);
        let noise_all: Vec<f64> = (0..3 * page + 8)
            .map(|_| f64::from_bits(rng.next()))
            .collect();
        let steady_all: Vec<f64> = (0..3 * page + 8)
            .map(|i| f64::from((i % 10) as u32))
            .collect();
        for pages in [1usize, 2] {
            // The size of the same data with a full last page is the yardstick.
            let full = pages * page;
            let table = make(
                &ctx,
                &[
                    ("steady", Spec::D(&steady_all[..full])),
                    ("noise", Spec::D(&noise_all[..full])),
                ],
                &[],
            );
            let base = dir.file("base.parquet");
            write_table(&ctx, &base, &table, &options(Codec::Zstd, 1 << 22, page)).unwrap();
            let base_size = fs::metadata(&base).unwrap().len();
            for tail in [1usize, 2, 3, 4, 8] {
                let rows = full + tail;
                let table = make(
                    &ctx,
                    &[
                        ("steady", Spec::D(&steady_all[..rows])),
                        ("noise", Spec::D(&noise_all[..rows])),
                    ],
                    &[],
                );
                let path = dir.file("tail.parquet");
                write_table(&ctx, &path, &table, &options(Codec::Zstd, 1 << 22, page)).unwrap();
                let bytes = fs::read(&path).unwrap();
                let size = bytes.len() as u64;
                // The compressible column keeps zstd, the noise column (whose
                // frames cannot shrink in total) does not.
                let codecs = chunk_codecs(&bytes);
                assert_eq!(
                    codecs,
                    vec![
                        ("steady".to_owned(), parquet::format::codec::ZSTD),
                        ("noise".to_owned(), parquet::format::codec::UNCOMPRESSED)
                    ],
                    "{pages} pages + {tail} rows"
                );
                // The tail adds its own rows and little else: before the fix
                // the steady column doubled the file.
                assert!(
                    size < base_size + 8 * 2 * tail as u64 + 600,
                    "{pages} pages + {tail} rows: {size} bytes against {base_size} without the tail"
                );
                assert_same(&table, &read_table(&ctx, &path).unwrap(), "tail");
                // And the bytes are a function of the table.
                let again = dir.file("again.parquet");
                write_table(&ctx, &again, &table, &options(Codec::Zstd, 1 << 22, page)).unwrap();
                assert_eq!(fs::read(&again).unwrap(), bytes);
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn many_small_zstd_pages_decode_without_a_thread_per_page() {
    // 20,000 pages of 16 rows in one chunk. The serial read used to spawn and
    // join a large-stack thread for every page (30-40 microseconds each,
    // 0.6 s for this file); now one thread serves the whole read.
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let dir = Directory::new();
    {
        let rows = 20_000 * 16;
        let steady: Vec<f64> = (0..rows).map(|i| f64::from((i % 10) as u32)).collect();
        let table = make(&ctx, &[("steady", Spec::D(&steady))], &[]);
        let path = dir.file("pages.parquet");
        write_table(&ctx, &path, &table, &options(Codec::Zstd, 1 << 22, 16)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            chunk_codecs(&bytes),
            vec![("steady".to_owned(), parquet::format::codec::ZSTD)]
        );
        let mut best = std::time::Duration::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let back = read_table(&ctx, &path).unwrap();
            best = best.min(started.elapsed());
            assert_eq!(back.num_rows(), rows);
        }
        eprintln!("20,000 zstd pages, one thread: best of three {best:?}");
        assert_same(&table, &read_table(&ctx, &path).unwrap(), "pages");
        assert!(
            best < std::time::Duration::from_millis(300),
            "20,000 small zstd pages took {best:?}"
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// Determinism.
// ---------------------------------------------------------------------------

#[test]
fn writing_is_deterministic_and_idempotent() {
    let ctx = context();
    let dir = Directory::new();
    {
        let mut rng = Rng(99);
        let rows = 20_000;
        let d: Vec<f64> = (0..rows).map(|i| ((i * 37) % 101) as f64 * 0.125).collect();
        let s: Vec<String> = (0..rows).map(|_| random_text(&mut rng)).collect();
        let table = make(
            &ctx,
            &[("d", Spec::D(&d)), ("s", Spec::S(&s))],
            &[("b", "2"), ("a", "1"), ("c", "3")],
        );
        for codec in CODECS {
            for (rg, pg) in [(1 << 20, 1 << 16), (4096, 700)] {
                let opts = options(codec, rg, pg);
                let (a, b, c) = (
                    dir.file("a.parquet"),
                    dir.file("b.parquet"),
                    dir.file("c.parquet"),
                );
                write_table(&ctx, &a, &table, &opts).unwrap();
                write_table(&ctx, &b, &table, &opts).unwrap();
                assert_eq!(
                    fs::read(&a).unwrap(),
                    fs::read(&b).unwrap(),
                    "two writes differ"
                );
                // Overwriting in place gives the same bytes again.
                write_table(&ctx, &a, &table, &opts).unwrap();
                assert_eq!(
                    fs::read(&a).unwrap(),
                    fs::read(&b).unwrap(),
                    "an overwrite differs"
                );
                // Read and write again: the same bytes (a re-save leaves a file unchanged).
                let back = read_table(&ctx, &a).unwrap();
                write_table(&ctx, &c, &back, &opts).unwrap();
                assert_eq!(
                    fs::read(&a).unwrap(),
                    fs::read(&c).unwrap(),
                    "a re-save differs"
                );
            }
        }
        // Metadata comes out sorted by key whatever the insertion order was.
        let path = dir.file("a.parquet");
        let n = fs::metadata(&path).unwrap().len() as usize;
        let bytes = fs::read(&path).unwrap();
        let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
        let meta = FileMetaData::from_bytes(&bytes[n - 8 - footer_len..n - 8]).unwrap();
        let keys: Vec<&str> = meta
            .key_value_metadata
            .iter()
            .map(|kv| kv.key.as_str())
            .collect();
        assert_eq!(keys, ["a", "b", "c"]);
    }
    assert_eq!(ctx.memory_used(), 0);
    assert_eq!(dir.entries().len(), 3);
}

// ---------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------

#[test]
fn codec_names_are_parsed_strictly() {
    for (name, expected) in [
        ("zstd", Codec::Zstd),
        ("ZSTD", Codec::Zstd),
        ("Zstd", Codec::Zstd),
        ("uncompressed", Codec::Uncompressed),
        ("UNCOMPRESSED", Codec::Uncompressed),
        ("none", Codec::Uncompressed),
        ("None", Codec::Uncompressed),
    ] {
        assert_eq!(Codec::parse(name).unwrap(), expected, "{name}");
    }
    for name in [
        "snappy", "gzip", "brotli", "lz4", "lz4_raw", "lzo", "", " zstd", "zstd ", "zstd-1",
        "default", "auto",
    ] {
        match Codec::parse(name) {
            Err(Error::Invalid(message)) => {
                for accepted in ["zstd", "uncompressed", "none"] {
                    assert!(message.contains(accepted), "{name}: {message}");
                }
                assert!(message.starts_with("parquet: "), "{message}");
            }
            other => panic!("{name:?} was accepted: {other:?}"),
        }
    }
    assert_eq!(Codec::Zstd.name(), "zstd");
    assert_eq!(Codec::Uncompressed.name(), "uncompressed");
    assert_eq!(Codec::parse(Codec::Zstd.name()).unwrap(), Codec::Zstd);
    assert_eq!(
        Codec::parse(Codec::Uncompressed.name()).unwrap(),
        Codec::Uncompressed
    );
    let defaults = WriteOptions::default();
    assert_eq!(defaults.codec, Codec::Zstd);
    assert_eq!(defaults.row_group_rows, 1 << 20);
    assert_eq!(defaults.page_rows, 1 << 16);
}

#[test]
fn a_table_that_cannot_be_written_creates_nothing() {
    let ctx = context();
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    fs::write(&path, b"previous").unwrap();
    let d = [1.0, 2.0, 3.0];
    let short = [1.0];
    let s = strings(&["a", "b", "c"]);
    let good = make(&ctx, &[("d", Spec::D(&d)), ("s", Spec::S(&s))], &[]);
    let refused: Vec<(&str, Table, WriteOptions)> = vec![
        (
            "unequal lengths",
            make(&ctx, &[("d", Spec::D(&d)), ("e", Spec::D(&short))], &[]),
            WriteOptions::default(),
        ),
        (
            "unequal kinds",
            make(&ctx, &[("e", Spec::D(&short)), ("s", Spec::S(&s))], &[]),
            WriteOptions::default(),
        ),
        (
            "repeated names",
            make(&ctx, &[("d", Spec::D(&d)), ("d", Spec::D(&d))], &[]),
            WriteOptions::default(),
        ),
        (
            "empty name",
            make(&ctx, &[("", Spec::D(&d))], &[]),
            WriteOptions::default(),
        ),
        (
            "no columns",
            Table {
                columns: vec![],
                metadata: BTreeMap::new(),
            },
            WriteOptions::default(),
        ),
        (
            "zero row group",
            make(&ctx, &[("d", Spec::D(&d))], &[]),
            options(Codec::Zstd, 0, 10),
        ),
        (
            "zero page",
            make(&ctx, &[("d", Spec::D(&d))], &[]),
            options(Codec::Zstd, 10, 0),
        ),
        (
            "huge page",
            make(&ctx, &[("d", Spec::D(&d))], &[]),
            options(Codec::Zstd, 10, usize::MAX),
        ),
    ];
    for (what, table, opts) in &refused {
        let error = write_table(&ctx, &path, table, opts).expect_err(what);
        assert!(
            matches!(&error, Error::Invalid(m) if m.starts_with("parquet: ")),
            "{what}: {error:?}"
        );
        assert_eq!(fs::read(&path).unwrap(), b"previous", "{what}");
        assert_eq!(dir.entries(), ["t.parquet"], "{what}");
    }
    // A key-value block too large for the footer.
    let mut huge = make(&ctx, &[("d", Spec::D(&d))], &[]);
    huge.metadata.insert("k".into(), "x".repeat(40 << 20));
    assert!(write_table(&ctx, &path, &huge, &WriteOptions::default()).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"previous");
    assert_eq!(dir.entries(), ["t.parquet"]);
    drop((refused, huge));
    // The good table, on the same path, replaces the old file.
    write_table(&ctx, &path, &good, &WriteOptions::default()).unwrap();
    assert_same(&good, &read_table(&ctx, &path).unwrap(), "replacement");
    assert_eq!(dir.entries(), ["t.parquet"]);
    drop(good);
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// Atomic failure (pattern of tests/persistence.rs).
// ---------------------------------------------------------------------------

fn try_sample(ctx: &Context, rows: usize) -> Result<Table, Error> {
    let d: Vec<f64> = (0..rows).map(|i| (i % 7) as f64 + 0.5).collect();
    let s: Vec<String> = (0..rows).map(|i| format!("label-{}", i % 11)).collect();
    try_make(
        ctx,
        &[("d", Spec::D(&d)), ("s", Spec::S(&s))],
        &[("k", "v")],
    )
}

fn sample(ctx: &Context, rows: usize) -> Table {
    try_sample(ctx, rows).unwrap()
}

#[test]
fn a_failure_during_the_write_keeps_the_old_file_and_leaves_no_stage_file() {
    let big = context();
    let dir = Directory::new();
    let path = dir.file("t.parquet");
    let old = sample(&big, 3000);
    write_table(&big, &path, &old, &options(Codec::Zstd, 1 << 20, 1 << 16)).unwrap();
    let before = fs::read(&path).unwrap();

    // A context that holds the table and the footer budget but not the zstd
    // encoder's working memory: the failure happens after the stage file was
    // created. The same table is written without trouble when it is not
    // compressed.
    let tight = Context::new(Some(1), 1 << 20).unwrap();
    {
        let table = sample(&tight, 3000);
        let error = write_table(
            &tight,
            &path,
            &table,
            &options(Codec::Zstd, 1 << 20, 1 << 16),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Budget { .. }), "{error:?}");
        assert_eq!(fs::read(&path).unwrap(), before, "the old file changed");
        assert_eq!(dir.entries(), ["t.parquet"], "a stage file was left");
        let other = dir.file("raw.parquet");
        write_table(
            &tight,
            &other,
            &table,
            &options(Codec::Uncompressed, 1 << 20, 1 << 16),
        )
        .unwrap();
        assert_same(
            &table,
            &read_table(&tight, &other).unwrap(),
            "uncompressed under 1 MiB",
        );
    }
    assert_eq!(tight.memory_used(), 0);
    fs::remove_file(dir.file("raw.parquet")).unwrap();

    // A path whose parent is a regular file, and a path that is a directory.
    let blocked = path.join("inner.parquet");
    assert!(write_table(&big, &blocked, &old, &WriteOptions::default()).is_err());
    let folder = dir.file("folder.parquet");
    fs::create_dir(&folder).unwrap();
    assert!(write_table(&big, &folder, &old, &WriteOptions::default()).is_err());
    assert!(folder.is_dir());
    assert_eq!(dir.entries(), ["folder.parquet", "t.parquet"]);
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_dir(&folder).unwrap();

    // A budget sweep: whatever the limit, the write is all or nothing.
    let mut succeeded = 0;
    let mut refused = 0;
    let mut limit = 1usize << 10;
    while limit < 1 << 26 {
        for codec in CODECS {
            let ctx = Context::new(Some(1), limit).unwrap();
            // The table itself may not fit this limit.
            let Ok(table) = try_sample(&ctx, 200) else {
                continue;
            };
            let outcome = write_table(&ctx, &path, &table, &options(codec, 128, 50));
            match outcome {
                Ok(()) => {
                    succeeded += 1;
                    // Reading needs room for a second copy of the table.
                    match read_table(&ctx, &path) {
                        Ok(back) => assert_same(&table, &back, "swept write"),
                        Err(Error::Budget { .. }) => {}
                        Err(other) => panic!("limit {limit}: {other:?}"),
                    }
                    fs::write(&path, &before).unwrap();
                }
                Err(Error::Budget { .. }) => {
                    refused += 1;
                    assert_eq!(fs::read(&path).unwrap(), before, "limit {limit}");
                }
                Err(other) => panic!("limit {limit}: {other:?}"),
            }
            assert_eq!(dir.entries(), ["t.parquet"], "limit {limit}");
            drop(table);
            assert_eq!(ctx.memory_used(), 0, "limit {limit}");
        }
        limit += limit / 4 + 1;
    }
    assert!(
        succeeded > 0 && refused > 0,
        "{succeeded} ok, {refused} refused"
    );
    drop(old);
    assert_eq!(big.memory_used(), 0);
}

#[test]
fn writing_creates_missing_directories_and_replaces_atomically() {
    let ctx = context();
    let dir = Directory::new();
    let table = sample(&ctx, 10);
    let nested = dir.file("a/b/c/t.parquet");
    write_table(&ctx, &nested, &table, &WriteOptions::default()).unwrap();
    assert_same(&table, &read_table(&ctx, &nested).unwrap(), "nested");
    assert_eq!(fs::read_dir(nested.parent().unwrap()).unwrap().count(), 1);
    assert!(parquet::is_parquet(&nested).unwrap());
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// The reader's accounting.
// ---------------------------------------------------------------------------

#[test]
fn read_metadata_touches_only_the_trailer_and_the_footer() {
    let ctx = context();
    let dir = Directory::new();
    let table = sample(&ctx, 5000);
    let path = dir.file("t.parquet");
    write_table(&ctx, &path, &table, &options(Codec::Zstd, 1000, 300)).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    // Every data page is destroyed, the magic and the footer are not.
    for byte in &mut bytes[4..n - 8 - footer_len] {
        *byte = 0xA5;
    }
    fs::write(&path, &bytes).unwrap();
    let footer = read_metadata(&ctx, &path).unwrap();
    assert_eq!(footer.num_rows, 5000);
    assert_eq!(footer.num_row_groups, 5);
    assert_eq!(footer.metadata["k"], "v");
    assert_eq!(footer.created_by, CREATED_BY);
    assert!(read_table(&ctx, &path).is_err());
    // A file that is only a footer opens for metadata and fails for data.
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_budget_decides_and_nothing_is_left_behind() {
    let ctx = context();
    let dir = Directory::new();
    let rows = 200_000usize;
    let table = {
        let d: Vec<f64> = (0..rows).map(|i| (i % 1000) as f64).collect();
        make(&ctx, &[("d", Spec::D(&d))], &[("k", "v")])
    };
    for codec in CODECS {
        let path = dir.file("big.parquet");
        write_table(&ctx, &path, &table, &options(codec, 50_000, 10_000)).unwrap();
        let table_bytes = rows * 8;
        // The measured peak of an unconstrained read.
        let probe = Context::new(Some(1), 1 << 30).unwrap();
        let back = read_table(&probe, &path).unwrap();
        assert_same(&table, &back, "unconstrained");
        drop(back);
        let peak = probe.memory_peak();
        assert_eq!(probe.memory_used(), 0);
        eprintln!(
            "{}: table {table_bytes} B, read peak {peak} B",
            codec.name()
        );
        assert!(peak >= table_bytes);
        assert!(
            peak <= table_bytes + (12 << 20),
            "peak {peak} is more than the table plus one chunk's working memory"
        );
        // Exactly that budget works; far below the table it is refused.
        let exact = Context::new(Some(1), peak).unwrap();
        drop(read_table(&exact, &path).unwrap());
        assert_eq!(exact.memory_used(), 0);
        for limit in [1, 2, 1000, table_bytes / 2, table_bytes - 1] {
            let small = Context::new(Some(1), limit).unwrap();
            let error = read_table(&small, &path).unwrap_err();
            assert!(
                matches!(error, Error::Budget { .. }),
                "limit {limit}: {error:?}"
            );
            assert_eq!(small.memory_used(), 0);
        }
        // Between the table and the exact peak it either reads or is refused.
        let mut limit = table_bytes;
        while limit < peak {
            let ctx = Context::new(Some(1), limit).unwrap();
            match read_table(&ctx, &path) {
                Ok(t) => assert_eq!(t.num_rows(), rows),
                Err(Error::Budget { .. }) => {}
                Err(other) => panic!("limit {limit}: {other:?}"),
            }
            assert_eq!(ctx.memory_used(), 0);
            limit += (peak - table_bytes) / 7 + 1;
        }
    }
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// The frozen fixtures.
// ---------------------------------------------------------------------------

fn sha256(data: &[u8]) -> String {
    Sha256::digest(data)
}

struct Sha256 {
    h: [u32; 8],
    block: Vec<u8>,
    length: u64,
}

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

    fn digest(data: &[u8]) -> String {
        let mut hash = Self::new();
        hash.update(data);
        hash.finish()
    }
}

/// SHA-256 of the doubles as 8 little-endian bytes each.
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

/// SHA-256 of the UTF-8 values joined by NUL.
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

#[test]
fn the_sha256_helper_is_correct() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Streaming in odd pieces equals one shot, across the block boundary.
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
    let mut hash = Sha256::new();
    for piece in data.chunks(37) {
        hash.update(piece);
    }
    assert_eq!(hash.finish(), sha256(&data));
    assert_eq!(
        sha256(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

fn references() -> serde_json::Value {
    let text = fs::read_to_string(fixtures().join("references.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn every_readable_fixture_matches_the_digests_pyarrow_recorded() {
    let references = references();
    let mut read = 0;
    for (name, entry) in references["fixtures"].as_object().unwrap() {
        if entry["expect_new_reader"] != "ok" {
            continue;
        }
        let ctx = context();
        let path = fixtures().join(name);
        {
            let table = read_table(&ctx, &path).unwrap_or_else(|e| panic!("{name}: {e}"));
            let decoded = entry["decoded_by_pyarrow"].as_object().unwrap();
            assert_eq!(table.columns.len(), decoded.len(), "{name}");
            for (column_name, expect) in decoded {
                let column = table
                    .column(column_name)
                    .unwrap_or_else(|| panic!("{name}: no column {column_name}"));
                assert_eq!(
                    column.len() as u64,
                    expect["rows"].as_u64().unwrap(),
                    "{name} {column_name}"
                );
                match column {
                    Column::Double(values) => assert_eq!(
                        double_digest(values.as_slice()),
                        expect["sha256_float64_bytes"].as_str().unwrap(),
                        "{name} {column_name}"
                    ),
                    Column::Utf8(values) => assert_eq!(
                        string_digest(values),
                        expect["sha256_utf8_nul_joined"].as_str().unwrap(),
                        "{name} {column_name}"
                    ),
                }
            }
            // The columns come in the file's order.
            let schema: Vec<&str> = entry["structure"]["schema"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c["name"].as_str().unwrap())
                .collect();
            assert_eq!(table.column_names(), schema, "{name}");
            // The key-value metadata equals what pyarrow reports.
            let mut expected = BTreeMap::new();
            for (key, value) in entry["kv"].as_object().unwrap() {
                match value {
                    serde_json::Value::String(text) => {
                        expected.insert(key.clone(), text.clone());
                    }
                    long => {
                        let got = &table.metadata[key];
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
                        expected.insert(key.clone(), got.clone());
                    }
                }
            }
            assert_eq!(table.metadata, expected, "{name}");
            // A fixture re-saved by this writer reads back to the same table.
            let dir = Directory::new();
            let again = dir.file("again.parquet");
            for codec in CODECS {
                write_table(&ctx, &again, &table, &options(codec, 3, 2)).unwrap();
                assert_same(&table, &read_table(&ctx, &again).unwrap(), name);
            }
        }
        assert_eq!(ctx.memory_used(), 0, "{name}");
        read += 1;
    }
    assert!(read >= 38, "only {read} fixtures read");
}

#[test]
fn the_fixture_folder_is_what_the_references_describe() {
    // A fixture added or removed without regenerating `references.json` would
    // leave the sweeps and the digests testing a stale set.
    let references = references();
    let listed: Vec<&String> = references["fixtures"].as_object().unwrap().keys().collect();
    let mut found = Vec::new();
    for folder in ["generic", "hamiltonian", "negative", "pseudopotential"] {
        for entry in fs::read_dir(fixtures().join(folder)).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if name.ends_with(".parquet") {
                found.push(format!("{folder}/{name}"));
            }
        }
    }
    found.sort();
    let mut listed: Vec<String> = listed.into_iter().cloned().collect();
    listed.sort();
    assert_eq!(found, listed);
    for (name, entry) in references["fixtures"].as_object().unwrap() {
        let bytes = fs::read(fixtures().join(name)).unwrap();
        assert!(bytes.len() < 20_000, "{name} has {} bytes", bytes.len());
        assert_eq!(
            bytes.len() as u64,
            entry["bytes"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(sha256(&bytes), entry["sha256"].as_str().unwrap(), "{name}");
    }
}

#[test]
fn is_parquet_agrees_with_the_fixtures() {
    for entry in fs::read_dir(fixtures().join("hamiltonian")).unwrap() {
        let path = entry.unwrap().path();
        assert!(parquet::is_parquet(&path).unwrap(), "{}", path.display());
    }
    assert!(!parquet::is_parquet(&fixtures().join("references.json")).unwrap());
    assert!(!parquet::is_parquet(Path::new(
        &fixtures().join("negative/container_zero_bytes.parquet")
    ))
    .unwrap());
}

// ---------------------------------------------------------------------------
// Many columns, many threads, small stacks.
// ---------------------------------------------------------------------------

/// A table of `doubles` double columns and two text columns with enough rows
/// for several row groups and pages, compressible so zstd really runs.
fn wide_table(ctx: &Context, doubles: usize, rows: usize) -> Table {
    let mut columns = Vec::new();
    for c in 0..doubles {
        let values: Vec<f64> = (0..rows)
            .map(|i| ((i * (c + 3)) % 97) as f64 * 0.25 + c as f64)
            .collect();
        columns.push((
            format!("d{c:03}"),
            Column::Double(ctx.copy_slice(&values).unwrap()),
        ));
    }
    for c in 0..2 {
        let values: Vec<String> = (0..rows)
            .map(|i| format!("{}{}", "IXYZ".as_bytes()[(i + c) % 4] as char, i % 13))
            .collect();
        columns.push((
            format!("s{c}"),
            Column::Utf8(Utf8Column::from_strs(ctx, &values).unwrap()),
        ));
    }
    Table {
        columns,
        metadata: BTreeMap::from([("k".to_owned(), "v".to_owned())]),
    }
}

#[test]
fn reading_on_many_threads_gives_the_table_one_thread_gives() {
    let ctx = context();
    let dir = Directory::new();
    let table = wide_table(&ctx, 24, 6_000);
    for codec in CODECS {
        let path = dir.file("wide.parquet");
        write_table(&ctx, &path, &table, &options(codec, 2_500, 700)).unwrap();
        for threads in [1usize, 2, 3, 8, 64] {
            let reader = Context::new(Some(threads), 1 << 30).unwrap();
            let back = read_table(&reader, &path).unwrap();
            assert_same(
                &table,
                &back,
                &format!("{} on {threads} threads", codec.name()),
            );
            drop(back);
            assert_eq!(reader.memory_used(), 0, "{threads} threads left memory");
        }
    }
}

#[test]
fn a_budget_that_fits_one_worker_still_reads_with_many_threads() {
    let ctx = context();
    let dir = Directory::new();
    let table = wide_table(&ctx, 16, 30_000);
    let path = dir.file("wide.parquet");
    write_table(&ctx, &path, &table, &options(Codec::Zstd, 10_000, 5_000)).unwrap();
    // The smallest budget that reads on one thread.
    let reads_on = |threads: usize, limit: usize| {
        let reader = Context::new(Some(threads), limit).unwrap();
        let outcome = read_table(&reader, &path);
        let read = outcome.is_ok();
        drop(outcome);
        assert_eq!(reader.memory_used(), 0);
        assert!(reader.memory_peak() <= limit);
        read
    };
    let (mut low, mut high) = (1usize, 1usize << 28);
    assert!(reads_on(1, high));
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if reads_on(1, middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    assert!(!reads_on(8, high - 1), "one byte less must be refused");
    for threads in [2, 8] {
        assert!(
            reads_on(threads, high),
            "{threads} threads must fall back to the workers the budget allows"
        );
    }
}

fn retarget(bytes: &[u8], edit: impl FnOnce(&mut FileMetaData)) -> Vec<u8> {
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let start = n - 8 - footer_len;
    let mut meta = FileMetaData::from_bytes(&bytes[start..start + footer_len]).unwrap();
    edit(&mut meta);
    let footer = meta.to_bytes().unwrap();
    let mut out = bytes[..start].to_vec();
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(b"PAR1");
    out
}

#[test]
fn the_first_failing_column_is_the_error_whatever_the_thread_count() {
    let ctx = context();
    let dir = Directory::new();
    let table = wide_table(&ctx, 20, 3_000);
    let path = dir.file("wide.parquet");
    write_table(
        &ctx,
        &path,
        &table,
        &options(Codec::Uncompressed, 1_500, 500),
    )
    .unwrap();
    let bytes = fs::read(&path).unwrap();
    // Column 4 claims GZIP and column 9 claims Brotli in every row group.
    let bad = retarget(&bytes, |meta| {
        for group in &mut meta.row_groups {
            group.columns[4].meta_data.codec = 2;
            group.columns[9].meta_data.codec = 4;
        }
    });
    let bad_path = dir.file("bad.parquet");
    fs::write(&bad_path, bad).unwrap();
    for threads in [1usize, 2, 4, 8, 16] {
        for _ in 0..10 {
            let reader = Context::new(Some(threads), 1 << 30).unwrap();
            let error = read_table(&reader, &bad_path).unwrap_err().to_string();
            assert!(error.contains("GZIP"), "{threads} threads: {error}");
            assert_eq!(reader.memory_used(), 0);
        }
    }
}

#[test]
fn zstd_runs_on_the_stack_of_a_small_caller_thread() {
    // ruzstd needs more than 192 KiB to encode and more than 96 KiB to
    // decode; a caller with a 128 KiB stack must still get a table or an
    // error, never a killed process.
    let dir = Directory::new();
    let path = dir.file("small.parquet");
    let original = {
        let ctx = context();
        wide_table(&ctx, 3, 40_000)
    };
    for threads in [1usize, 4] {
        let (path, original_ref) = (path.clone(), &original);
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(128 << 10)
                .spawn_scoped(scope, move || {
                    let ctx = Context::new(Some(threads), 1 << 30).unwrap();
                    write_table(
                        &ctx,
                        &path,
                        original_ref,
                        &options(Codec::Zstd, 1 << 20, 1 << 16),
                    )
                    .unwrap();
                    let back = read_table(&ctx, &path).unwrap();
                    assert_same(original_ref, &back, "on a small stack");
                })
                .unwrap()
                .join()
                .unwrap();
        });
    }
}

#[test]
fn a_file_with_tens_of_thousands_of_columns_is_handled_in_linear_time() {
    // The duplicate-name check was quadratic: 65,536 columns took minutes.
    let ctx = context();
    let dir = Directory::new();
    let count = parquet::MAX_COLUMNS;
    let columns: Vec<(String, Column)> = (0..count)
        .map(|c| {
            (
                format!("column-{c:06}"),
                Column::Double(ctx.copy_slice(&[c as f64]).unwrap()),
            )
        })
        .collect();
    let table = Table {
        columns,
        metadata: BTreeMap::new(),
    };
    let path = dir.file("wide.parquet");
    let started = std::time::Instant::now();
    write_table(
        &ctx,
        &path,
        &table,
        &options(Codec::Uncompressed, 1 << 20, 1 << 16),
    )
    .unwrap();
    let footer = read_metadata(&ctx, &path).unwrap();
    assert_eq!(footer.columns.len(), count);
    let back = read_table(&ctx, &path).unwrap();
    assert_same(&table, &back, "65,536 columns");
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "write, metadata and read of {count} columns took {elapsed:?}"
    );
}

#[test]
fn more_columns_than_the_cap_are_refused_by_writer_and_reader() {
    let ctx = context();
    let dir = Directory::new();
    let count = parquet::MAX_COLUMNS + 1;
    let columns: Vec<(String, Column)> = (0..count)
        .map(|c| {
            (
                format!("c{c}"),
                Column::Double(ctx.copy_slice(&[0.0]).unwrap()),
            )
        })
        .collect();
    let table = Table {
        columns,
        metadata: BTreeMap::new(),
    };
    let path = dir.file("too-wide.parquet");
    let error = write_table(&ctx, &path, &table, &WriteOptions::default()).unwrap_err();
    assert!(error.to_string().contains("at most 65536"), "{error}");
    assert!(!path.exists());
    // A footer that lists them is refused before anything is built.
    use mandacaru_core::parquet::format::SchemaElement;
    let mut schema = vec![SchemaElement::root(count).unwrap()];
    schema.extend((0..count).map(|c| SchemaElement::leaf(&format!("c{c}"), 5, false)));
    let meta = FileMetaData {
        version: 1,
        schema,
        num_rows: 0,
        row_groups: vec![],
        key_value_metadata: vec![],
        created_by: None,
    };
    let footer = meta.to_bytes().unwrap();
    let mut file = b"PAR1".to_vec();
    file.extend_from_slice(&footer);
    file.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    file.extend_from_slice(b"PAR1");
    let forged = dir.file("forged.parquet");
    fs::write(&forged, file).unwrap();
    for error in [
        read_table(&ctx, &forged).unwrap_err(),
        read_metadata(&ctx, &forged).map(|_| ()).unwrap_err(),
    ] {
        assert!(error.to_string().contains("at most 65536"), "{error}");
    }
    drop(table);
    assert_eq!(ctx.memory_used(), 0);
}

// ---------------------------------------------------------------------------
// A file replaced while it is read.
// ---------------------------------------------------------------------------

#[test]
fn a_file_replaced_during_a_threaded_read_is_never_a_mix_of_both() {
    // Two files of the same shape, every double 1.0 or every double 2.0, are
    // swapped in by atomic rename while another thread reads with several
    // workers. A reader that re-opened the path per worker would apply the
    // footer of one file to the bytes of the other; the table must be wholly
    // one file or the other.
    let dir = Directory::new();
    let path = dir.file("swap.parquet");
    let rows = 200_000usize;
    let ctx = context();
    let tables: Vec<Table> = [1.0f64, 2.0]
        .iter()
        .map(|&v| {
            let values = vec![v; rows];
            make(
                &ctx,
                &[
                    ("a", Spec::D(&values)),
                    ("b", Spec::D(&values)),
                    ("c", Spec::D(&values)),
                ],
                &[],
            )
        })
        .collect();
    let plain = options(Codec::Uncompressed, 1 << 20, 1 << 16);
    write_table(&ctx, &path, &tables[0], &plain).unwrap();
    let finished = std::sync::atomic::AtomicBool::new(false);
    let torn = std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            let writer_ctx = context();
            for round in 0..60 {
                write_table(&writer_ctx, &path, &tables[round % 2], &plain).unwrap();
            }
            finished.store(true, Ordering::Release);
        });
        let reader_ctx = Context::new(Some(4), 1 << 30).unwrap();
        let (mut torn, mut reads) = (0usize, 0usize);
        while !finished.load(Ordering::Acquire) || reads < 5 {
            let table = read_table(&reader_ctx, &path).unwrap();
            let first = table.column("a").unwrap().as_f64().unwrap()[0];
            let uniform = table.columns.iter().all(|(_, column)| {
                column
                    .as_f64()
                    .unwrap()
                    .iter()
                    .all(|v| v.to_bits() == first.to_bits())
            });
            torn += usize::from(!uniform);
            reads += 1;
        }
        writer.join().unwrap();
        torn
    });
    assert_eq!(torn, 0, "reads that mixed the two files");
}
