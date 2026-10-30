//! Page compression dispatch.
//!
//! The reader first calls `claim` with the Parquet `CompressionCodec` number of
//! the column chunk, the compressed page body and the uncompressed size the
//! page header claims: it refuses a body that cannot regenerate that size,
//! allocating nothing proportional to the claim. Only then does the reader
//! size an output slice of exactly that length and call `Claim::decompress`
//! (`decompress` does both for callers that already hold the output). The
//! caller owns and has already charged the output; the only memory a codec
//! reserves itself is zstd's decoder working set, charged to the `Context`
//! before the decoder is built.
//!
//! `compress` is the writer's single entry. A Parquet codec is a property of
//! the whole column chunk, so the writer compresses every page of a chunk and
//! keeps the frames of a zstd chunk even when a small page comes back `None`
//! (its frame is not smaller than the raw page); it stores the chunk
//! `UNCOMPRESSED` only when the frames of the whole chunk are not smaller.

pub mod snappy;
pub mod zstd;
mod zstd_audit;

use std::cell::Cell;
use std::thread::{Builder, Scope, ScopedJoinHandle};

use crate::parquet::format::codec;
use crate::parquet::{invalid, Codec};
use crate::{Context, Error, Result};

/// Stack of every thread that runs the zstd codec. ruzstd keeps large state on
/// the stack (its encoder needs more than 192 KiB and its decoder more than
/// 96 KiB, measured), so a caller on a small thread stack (a pool, musl, a
/// `threading.stack_size`) would be killed by SIGBUS, which no `catch_unwind`
/// can stop. The codec therefore never runs on the caller's stack.
pub(crate) const LARGE_STACK_BYTES: usize = 2 << 20;

thread_local! {
    static LARGE_STACK: Cell<bool> = const { Cell::new(false) };
}

/// Spawns a scoped thread with [`LARGE_STACK_BYTES`] of stack that knows it
/// is on one, so the codec runs on it directly.
pub(crate) fn spawn_large<'scope, 'env, T: Send + 'scope>(
    scope: &'scope Scope<'scope, 'env>,
    name: &str,
    work: impl FnOnce() -> T + Send + 'scope,
) -> Result<ScopedJoinHandle<'scope, T>> {
    Builder::new()
        .name(name.to_owned())
        .stack_size(LARGE_STACK_BYTES)
        .spawn_scoped(scope, move || {
            LARGE_STACK.with(|flag| flag.set(true));
            work()
        })
        .map_err(|_| Error::Allocation)
}

/// Runs `work` on a thread with a large stack: directly when the current
/// thread already is one, on a scoped thread (joined before returning)
/// otherwise. A panic of `work` becomes `Error::Invalid`.
pub(crate) fn on_large_stack<T: Send>(work: impl FnOnce() -> T + Send) -> Result<T> {
    if LARGE_STACK.with(Cell::get) {
        return Ok(work());
    }
    std::thread::scope(|scope| {
        spawn_large(scope, "mandacaru-parquet-codec", work)?
            .join()
            .map_err(|_| invalid("the compression codec failed (internal failure)"))
    })
}

/// The Parquet `CompressionCodec` number a write-side codec stores.
pub fn codec_id(codec: Codec) -> i32 {
    match codec {
        Codec::Uncompressed => codec::UNCOMPRESSED,
        Codec::Zstd => codec::ZSTD,
    }
}

/// A page body checked against the uncompressed size its header claims, before
/// any output buffer exists.
///
/// The claim is untrusted: a few bytes of body can claim hundreds of
/// megabytes, and a reader that sizes (and zero-fills) a buffer from the claim
/// lets a tiny file drive a huge allocation. [`claim`] proves, from the body
/// alone, that the body can regenerate that many bytes (zstd: the frame's
/// blocks add up to exactly the claim; snappy: the body is long enough at the
/// largest expansion a snappy element has), so the caller reserves only what
/// the body justifies.
pub enum Claim {
    Stored,
    Zstd(zstd::Frame),
    Snappy,
}

/// Checks `body` against the `uncompressed` size of its page header.
pub fn claim(codec_number: i32, body: &[u8], uncompressed: usize) -> Result<Claim> {
    match codec_number {
        codec::UNCOMPRESSED => {
            if body.len() != uncompressed {
                return Err(invalid(format!(
                    "uncompressed page holds {} bytes but its header says {uncompressed}",
                    body.len()
                )));
            }
            Ok(Claim::Stored)
        }
        codec::ZSTD => zstd::inspect(body, uncompressed).map(Claim::Zstd),
        codec::SNAPPY => snappy::inspect(body, uncompressed).map(|()| Claim::Snappy),
        codec::GZIP | codec::LZO | codec::BROTLI | codec::LZ4 | codec::LZ4_RAW => {
            Err(invalid(format!(
                "compression codec {} is not supported; re-save the file with zstd",
                codec::name(codec_number)
            )))
        }
        other => Err(invalid(format!(
            "{} is not supported; re-save the file with zstd",
            codec::name(other)
        ))),
    }
}

impl Claim {
    /// Fills `output`, whose length is the claimed size, from `body`.
    pub fn decompress(&self, ctx: &Context, body: &[u8], output: &mut [u8]) -> Result<()> {
        match self {
            Claim::Stored => {
                output.copy_from_slice(body);
                Ok(())
            }
            Claim::Zstd(frame) => frame.decode(ctx, body, output),
            Claim::Snappy => snappy::decompress(body, output),
        }
    }
}

/// Decompresses one page body into `output`, whose length is the exact
/// uncompressed size the page header declares.
pub fn decompress(ctx: &Context, codec_number: i32, input: &[u8], output: &mut [u8]) -> Result<()> {
    claim(codec_number, input, output.len())?.decompress(ctx, input, output)
}

/// Compresses one page. `Ok(None)` means "store this page as it is": the
/// codec is `Uncompressed`. A compressed page may be larger than its input
/// (incompressible data); the writer compares whole chunks.
pub fn compress(ctx: &Context, codec: Codec, input: &[u8]) -> Result<Option<Vec<u8>>> {
    match codec {
        Codec::Uncompressed => Ok(None),
        Codec::Zstd => zstd::compress(ctx, input).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context::new(Some(1), 1 << 30).unwrap()
    }

    #[test]
    fn dispatch_routes_each_codec() {
        let c = ctx();
        let data: Vec<u8> = (0..20_000u32)
            .flat_map(|i| ((i % 50) as f64).to_le_bytes())
            .collect();
        let frame = compress(&c, Codec::Zstd, &data).unwrap().unwrap();
        let mut out = vec![0u8; data.len()];
        decompress(&c, codec::ZSTD, &frame, &mut out).unwrap();
        assert_eq!(out, data);
        let mut raw = vec![0u8; data.len()];
        decompress(&c, codec::UNCOMPRESSED, &data, &mut raw).unwrap();
        assert_eq!(raw, data);
        assert!(compress(&c, Codec::Uncompressed, &data).unwrap().is_none());
        let mut out = [0u8; 3];
        decompress(&c, codec::SNAPPY, &[3, 8, 1, 2, 3], &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn exact_size_is_enforced_for_every_codec() {
        let c = ctx();
        let mut out = [0u8; 4];
        assert!(decompress(&c, codec::UNCOMPRESSED, &[1, 2, 3], &mut out).is_err());
        assert!(decompress(&c, codec::UNCOMPRESSED, &[1, 2, 3, 4, 5], &mut out).is_err());
        assert!(decompress(&c, codec::SNAPPY, &[3, 8, 1, 2, 3], &mut out).is_err());
    }

    #[test]
    fn unsupported_codecs_say_to_resave_with_zstd() {
        let c = ctx();
        let mut out = [0u8; 1];
        for number in [
            codec::GZIP,
            codec::LZO,
            codec::BROTLI,
            codec::LZ4,
            codec::LZ4_RAW,
            99,
        ] {
            let message = format!("{:?}", decompress(&c, number, &[0], &mut out).unwrap_err());
            assert!(message.contains("re-save the file with zstd"), "{message}");
        }
    }
}
