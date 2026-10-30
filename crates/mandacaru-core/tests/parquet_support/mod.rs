//! Helpers shared by the Parquet test binaries (`parquet_errors.rs`,
//! `parquet_memory.rs`): scratch directories, the frozen fixtures and the
//! seeded file mutator. Each binary uses a different subset.
#![allow(dead_code)]

use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use mandacaru_core::parquet::format::{codec as codec_id, FileMetaData, PageHeader};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

pub struct Directory(PathBuf);

impl Directory {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mandacaru-parquet-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    pub fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration/reference/parquet")
}

pub fn fixture(name: &str) -> Vec<u8> {
    fs::read(fixtures().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// The part of a file a mutation aims at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Head,
    Footer,
    Trailer,
    PageHeader,
    ZstdBody,
    SnappyBody,
    PlainBody,
}

pub struct Target {
    pub name: String,
    pub bytes: Vec<u8>,
    pub parts: Vec<(Part, Range<usize>)>,
}

/// Splits a well-formed file into the regions a mutation aims at, using the
/// public footer and page-header parsers.
pub fn target(name: &str) -> Target {
    let bytes = fixture(name);
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let footer_start = n - 8 - footer_len;
    let mut parts = vec![
        (Part::Head, 0..4),
        (Part::Footer, footer_start..n - 8),
        (Part::Trailer, n - 8..n),
    ];
    let meta = FileMetaData::from_bytes(&bytes[footer_start..n - 8]).unwrap();
    for group in &meta.row_groups {
        for chunk in &group.columns {
            let m = &chunk.meta_data;
            let mut at = m.start_offset() as usize;
            let end = at + m.total_compressed_size as usize;
            while at < end {
                let (header, header_len) = PageHeader::from_bytes(&bytes[at..end]).unwrap();
                parts.push((Part::PageHeader, at..at + header_len));
                let body = at + header_len..at + header_len + header.compressed_page_size as usize;
                let kind = match m.codec {
                    codec_id::ZSTD => Part::ZstdBody,
                    codec_id::SNAPPY => Part::SnappyBody,
                    _ => Part::PlainBody,
                };
                at = body.end;
                if !body.is_empty() {
                    parts.push((kind, body));
                }
            }
        }
    }
    Target {
        name: name.to_owned(),
        bytes,
        parts,
    }
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Values that sit on the edge of a length, count or size field.
const EDGES: [u8; 9] = [0x00, 0x01, 0x7f, 0x80, 0x81, 0xfe, 0xff, 0x0f, 0xf0];

/// Applies one random mutation inside `range` (insertions and removals shift
/// the rest of the file, so offsets and lengths go stale as well).
pub fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>, range: &Range<usize>) -> &'static str {
    let at = range.start + rng.below(range.len());
    match rng.below(8) {
        0 => {
            for _ in 0..1 + rng.below(4) {
                let p = range.start + rng.below(range.len());
                bytes[p] = rng.next() as u8;
            }
            "random bytes"
        }
        1 => {
            bytes[at] ^= 1 << rng.below(8);
            "bit flip"
        }
        2 => {
            bytes[at] = EDGES[rng.below(EDGES.len())];
            "edge byte"
        }
        3 => {
            let length = 1 + rng.below(32.min(range.end - at));
            let value = EDGES[rng.below(EDGES.len())];
            bytes[at..at + length].fill(value);
            "range fill"
        }
        4 => {
            let length = 1 + rng.below(16);
            let junk: Vec<u8> = (0..length).map(|_| rng.next() as u8).collect();
            bytes.splice(at..at, junk);
            "insertion"
        }
        5 => {
            let length = 1 + rng.below(16.min(range.end - at));
            bytes.drain(at..at + length);
            "removal"
        }
        6 => {
            bytes.truncate(at);
            "truncation"
        }
        _ => {
            // Overwrite a little-endian u32 with a large number.
            let end = (at + 4).min(bytes.len());
            let value = [0xffu8, 0xff, 0xff, 0x7f];
            bytes[at..end].copy_from_slice(&value[..end - at]);
            "large u32"
        }
    }
}
