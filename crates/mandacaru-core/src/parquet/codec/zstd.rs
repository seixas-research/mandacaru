//! zstd adapter, the only file that touches `ruzstd`.
//!
//! Decoding audits the frame before ruzstd sees a byte, so the page budget is
//! known and every unsupported or hostile frame is refused up front:
//!
//! - exactly one frame, no skippable frame, nothing after it;
//! - `Frame_Content_Size` present and equal to the page's uncompressed size;
//! - no dictionary id and no content checksum (the `hash` feature of ruzstd is
//!   off, so a checksum could not be verified and is refused instead);
//! - a window of at most `MAX_ZSTD_WINDOW_BYTES`;
//! - blocks that add up to exactly the content size, none above 128 KiB
//!   (`zstd_audit`: ruzstd enforces neither, and a 3 KB block of sequences
//!   expands to hundreds of megabytes in its ring buffer). That bound is what
//!   makes the memory charged to the `Context` before the decoder is created
//!   an upper bound of what ruzstd allocates, for any frame that passes.
//!
//! [`inspect`] does the audit and allocates nothing but a few kilobytes of
//! tables, so the reader can ask whether a page body can regenerate the size
//! its header claims before it reserves or zero-fills that size.
//!
//! Every ruzstd call runs inside `catch_unwind` because the decoder has panic
//! sites; a panic becomes `Error::Invalid`. It also runs on a thread with a
//! 2 MiB stack (`codec::on_large_stack`), because ruzstd's state does not fit
//! the stack of a small caller thread. The encoder only ever runs at
//! `CompressionLevel::Fastest` (the other levels are `unimplemented!()` in
//! ruzstd 0.8.1) and its frame is rewritten to carry `Frame_Content_Size`,
//! which ruzstd omits and this module's own decoder requires.

use std::io::Read;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ruzstd::decoding::StreamingDecoder;
use ruzstd::encoding::{compress_to_vec, CompressionLevel};

use crate::parquet::codec::on_large_stack;
use crate::parquet::codec::zstd_audit::{self, BLOCK_BYTES};
use crate::parquet::{invalid, MAX_ZSTD_WINDOW_BYTES};
use crate::{Context, Error, Result};

/// Little-endian magic number of a zstd frame.
const FRAME_MAGIC: u32 = 0xFD2F_B528;
/// First and last magic number of a skippable frame.
const SKIPPABLE_FIRST: u32 = 0x184D_2A50;
const SKIPPABLE_LAST: u32 = 0x184D_2A5F;
/// Smallest window a frame can declare (`windowLog = 10`).
const MIN_WINDOW_BYTES: u64 = 1 << 10;
/// Fixed working memory charged on top of the window: Huffman and FSE tables,
/// the literals and sequences scratch and the decoder bookkeeping. Measured
/// at well under this by `decoder_allocation_stays_inside_the_charge`.
const DECODER_FIXED_BYTES: usize = 2 << 20;
/// Fixed working memory charged for the encoder (match generator tables and
/// the block scratch), on top of the size-proportional part.
const ENCODER_FIXED_BYTES: usize = 4 << 20;

/// What the frame header declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Bytes the header occupies (magic included).
    pub header_len: usize,
    /// Declared window size.
    pub window: u64,
    /// Declared `Frame_Content_Size`.
    pub content_size: u64,
}

/// Parses and validates the frame header; never reads past it.
pub fn parse_frame_header(input: &[u8]) -> Result<FrameHeader> {
    let magic = input
        .get(..4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| invalid("zstd frame is shorter than its magic number"))?;
    if (SKIPPABLE_FIRST..=SKIPPABLE_LAST).contains(&magic) {
        return Err(invalid("zstd skippable frames are not supported"));
    }
    if magic != FRAME_MAGIC {
        return Err(invalid("page is not a zstd frame (bad magic number)"));
    }
    let descriptor = *input
        .get(4)
        .ok_or_else(|| invalid("zstd frame header is truncated"))?;
    let fcs_flag = descriptor >> 6;
    let single_segment = descriptor & 0x20 != 0;
    if descriptor & 0x08 != 0 {
        return Err(invalid("zstd frame header sets a reserved bit"));
    }
    if descriptor & 0x04 != 0 {
        return Err(invalid(
            "zstd frames with a content checksum are not supported; re-save with zstd without checksums",
        ));
    }
    if descriptor & 0x03 != 0 {
        return Err(invalid(
            "zstd frames that use a dictionary are not supported",
        ));
    }
    let mut at = 5usize;
    let window_descriptor = if single_segment {
        None
    } else {
        let byte = *input
            .get(at)
            .ok_or_else(|| invalid("zstd frame header is truncated"))?;
        at += 1;
        Some(byte)
    };
    let fcs_bytes =
        match (fcs_flag, single_segment) {
            (0, true) => 1,
            (0, false) => return Err(invalid(
                "zstd frame does not declare its content size; re-save with a zstd that records it",
            )),
            (1, _) => 2,
            (2, _) => 4,
            _ => 8,
        };
    let field = input
        .get(at..at + fcs_bytes)
        .ok_or_else(|| invalid("zstd frame header is truncated"))?;
    at += fcs_bytes;
    let mut raw = [0u8; 8];
    raw[..fcs_bytes].copy_from_slice(field);
    let mut content_size = u64::from_le_bytes(raw);
    if fcs_bytes == 2 {
        content_size += 256;
    }
    let window = match window_descriptor {
        None => content_size,
        Some(byte) => {
            let exponent = u32::from(byte >> 3);
            let mantissa = u64::from(byte & 7);
            let base = 1u64 << (10 + exponent);
            base + (base / 8) * mantissa
        }
    };
    if window < MIN_WINDOW_BYTES && window_descriptor.is_some() {
        return Err(invalid("zstd frame window is below the 1 KiB minimum"));
    }
    if window > MAX_ZSTD_WINDOW_BYTES as u64 {
        return Err(invalid(format!(
            "zstd frame window of {window} bytes exceeds the {MAX_ZSTD_WINDOW_BYTES}-byte limit"
        )));
    }
    Ok(FrameHeader {
        header_len: at,
        window,
        content_size,
    })
}

/// Bytes charged to the `Context` while a frame whose decode ring buffer
/// holds `resident` bytes is decoded: the ring buffer ruzstd grows to the next
/// power of two above what it needs (the resident bytes plus the block being
/// decoded and the block read ahead; it may briefly hold two copies while it
/// reallocates), plus the fixed scratch.
fn charge_for(resident: u64) -> Result<usize> {
    let needed = resident.saturating_add(2 * BLOCK_BYTES as u64);
    let ring = usize::try_from(needed)
        .ok()
        .and_then(usize::checked_next_power_of_two)
        .ok_or(Error::Overflow)?;
    ring.checked_mul(3)
        .and_then(|b| b.checked_add(DECODER_FIXED_BYTES))
        .ok_or(Error::Overflow)
}

/// The most [`Frame::charge`] can be for a frame of `content` bytes; the
/// reader sizes its worker buffers with it.
pub fn charge_bound(content: u64) -> Result<usize> {
    charge_for(content)
}

/// A frame whose header and blocks have been audited against the size the
/// page declares: it regenerates exactly that many bytes, and ruzstd cannot
/// allocate more than [`Frame::charge`] to do it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    header: FrameHeader,
}

impl Frame {
    /// Bytes the decoder is charged: the ring buffer never holds more than the
    /// window or the content, whichever is smaller.
    pub fn charge(&self) -> Result<usize> {
        charge_for(self.header.window.min(self.header.content_size))
    }

    /// Decodes the frame into `output` (exactly the audited content size, and
    /// already charged to the `Context`). Runs on a large-stack thread (see
    /// `codec::LARGE_STACK_BYTES`).
    pub fn decode(&self, ctx: &Context, input: &[u8], output: &mut [u8]) -> Result<()> {
        if self.header.content_size != output.len() as u64 {
            return Err(invalid(format!(
                "zstd frame declares {} content bytes but the page header says {}",
                self.header.content_size,
                output.len()
            )));
        }
        on_large_stack(|| self.decode_here(ctx, input, output))?
    }

    fn decode_here(&self, ctx: &Context, input: &[u8], output: &mut [u8]) -> Result<()> {
        let _scratch = ctx.reserve(self.charge()?)?;
        let outcome = catch_unwind(AssertUnwindSafe(|| -> std::result::Result<(), String> {
            let mut source: &[u8] = input;
            let mut decoder = StreamingDecoder::new(&mut source).map_err(|e| e.to_string())?;
            // Read block by block: one big read makes ruzstd decode the whole
            // frame into its ring buffer before it hands anything out.
            for piece in output.chunks_mut(BLOCK_BYTES) {
                decoder.read_exact(piece).map_err(|e| {
                    format!("zstd frame holds fewer bytes than declared or is corrupt ({e})")
                })?;
            }
            let mut probe = [0u8; 1];
            match decoder.read(&mut probe) {
                Ok(0) => {}
                Ok(_) => return Err("zstd frame holds more bytes than declared".to_string()),
                Err(e) => return Err(format!("zstd frame is corrupt ({e})")),
            }
            drop(decoder);
            if !source.is_empty() {
                return Err(
                    "bytes follow the zstd frame (multiple frames are not supported)".to_string(),
                );
            }
            Ok(())
        }));
        match outcome {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => Err(invalid(message)),
            Err(_) => Err(invalid("zstd decoder rejected the page (internal failure)")),
        }
    }
}

/// Audits `input` as the one zstd frame of a page that declares `expected`
/// uncompressed bytes. Allocates nothing proportional to `expected`: a frame
/// that cannot regenerate that many bytes is refused here, before the caller
/// reserves or fills a buffer of that size.
pub fn inspect(input: &[u8], expected: usize) -> Result<Frame> {
    let header = parse_frame_header(input)?;
    if header.content_size != expected as u64 {
        return Err(invalid(format!(
            "zstd frame declares {} content bytes but the page header says {expected}",
            header.content_size
        )));
    }
    let audited = catch_unwind(AssertUnwindSafe(|| {
        zstd_audit::blocks(&input[header.header_len..], header.content_size)
    }));
    match audited {
        Ok(result) => result?,
        Err(_) => return Err(invalid("zstd frame audit failed (internal failure)")),
    }
    Ok(Frame { header })
}

/// Decodes one zstd frame into `output`, which the caller sized to the exact
/// uncompressed page length (and already charged to the `Context`).
pub fn decompress(ctx: &Context, input: &[u8], output: &mut [u8]) -> Result<()> {
    inspect(input, output.len())?.decode(ctx, input, output)
}

/// Compresses `input` into one zstd frame at the `Fastest` level.
///
/// The frame carries `Frame_Content_Size`, no dictionary and no checksum. It
/// is returned even when it is not smaller than `input` (incompressible data
/// grows by the frame header and block headers); the writer decides per column
/// chunk whether the frames pay.
pub fn compress(ctx: &Context, input: &[u8]) -> Result<Vec<u8>> {
    on_large_stack(|| compress_here(ctx, input))?
}

fn compress_here(ctx: &Context, input: &[u8]) -> Result<Vec<u8>> {
    let charge = input
        .len()
        .checked_mul(2)
        .and_then(|b| b.checked_add(ENCODER_FIXED_BYTES))
        .ok_or(Error::Overflow)?;
    let _scratch = ctx.reserve(charge)?;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        compress_to_vec(input, CompressionLevel::Fastest)
    }));
    let mut frame = outcome.map_err(|_| invalid("zstd encoder failed (internal failure)"))?;
    add_content_size(&mut frame, input.len() as u64)?;
    Ok(frame)
}

/// ruzstd writes `Single_Segment = 0`, no checksum, no dictionary and no
/// `Frame_Content_Size`. Rewrites the descriptor to a 4- or 8-byte
/// `Frame_Content_Size` and inserts the field after the window descriptor.
fn add_content_size(frame: &mut Vec<u8>, content_size: u64) -> Result<()> {
    if frame.len() < 6 || frame[..4] != FRAME_MAGIC.to_le_bytes() {
        return Err(invalid("zstd encoder produced an unexpected frame header"));
    }
    let descriptor = frame[4];
    if descriptor != 0x00 {
        return Err(invalid(
            "zstd encoder produced an unexpected frame descriptor",
        ));
    }
    // Window descriptor is the byte at index 5; the field goes right after it.
    if let Ok(small) = u32::try_from(content_size) {
        frame[4] = 2 << 6;
        frame.splice(6..6, small.to_le_bytes());
    } else {
        frame[4] = 3 << 6;
        frame.splice(6..6, content_size.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    // A counting allocator that only counts on threads that opt in, so the
    // other tests of the crate are unaffected.
    struct Counting;

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static LIVE: Cell<usize> = const { Cell::new(0) };
        static PEAK: Cell<usize> = const { Cell::new(0) };
    }

    fn note_alloc(size: usize) {
        if ACTIVE.try_with(|a| a.get()).unwrap_or(false) {
            let _ = LIVE.try_with(|live| {
                let now = live.get() + size;
                live.set(now);
                let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
            });
        }
    }

    fn note_free(size: usize) {
        if ACTIVE.try_with(|a| a.get()).unwrap_or(false) {
            let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(size)));
        }
    }

    // SAFETY: every call is forwarded unchanged to `System`, which upholds the
    // `GlobalAlloc` contract; the additions are thread-local counters that never
    // allocate and are read with `try_with`, so a call made during thread teardown
    // is ignored. Test-only: see the allocator contract in
    // `crates/mandacaru-kernels/SAFETY.md`.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note_alloc(layout.size());
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            note_free(layout.size());
            System.dealloc(ptr, layout)
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            note_free(layout.size());
            note_alloc(new_size);
            System.realloc(ptr, layout, new_size)
        }
    }

    #[global_allocator]
    static COUNTING: Counting = Counting;

    /// Peak live bytes allocated on this thread while `f` runs.
    fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
        LIVE.with(|l| l.set(0));
        PEAK.with(|p| p.set(0));
        ACTIVE.with(|a| a.set(true));
        let value = f();
        ACTIVE.with(|a| a.set(false));
        (value, PEAK.with(|p| p.get()))
    }

    fn ctx() -> Context {
        Context::new(Some(1), 1 << 30).unwrap()
    }

    /// Deterministic pseudo-random bytes (xorshift).
    fn noise(n: usize, mut seed: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect()
    }

    /// Sample that compresses: repeated structured f64 values.
    fn structured(n_doubles: usize) -> Vec<u8> {
        (0..n_doubles)
            .flat_map(|i| (((i % 97) as f64) * 0.25).to_le_bytes())
            .collect()
    }

    /// A frame with explicit header fields and one raw block holding `body`.
    fn raw_frame(descriptor: u8, window: Option<u8>, header_tail: &[u8], body: &[u8]) -> Vec<u8> {
        let mut frame = FRAME_MAGIC.to_le_bytes().to_vec();
        frame.push(descriptor);
        frame.extend(window);
        frame.extend_from_slice(header_tail);
        let block = ((body.len() as u32) << 3) | 1; // last block, raw
        frame.extend_from_slice(&block.to_le_bytes()[..3]);
        frame.extend_from_slice(body);
        frame
    }

    fn decode(input: &[u8], expected: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; expected];
        decompress(&ctx(), input, &mut out)?;
        Ok(out)
    }

    #[test]
    fn round_trips_compressible_and_incompressible_data() {
        let c = ctx();
        for data in [structured(40_000), noise(5_000, 7), vec![9u8; 300_000]] {
            let frame = compress(&c, &data).unwrap();
            let header = parse_frame_header(&frame).unwrap();
            assert_eq!(header.content_size, data.len() as u64);
            assert_eq!(decode(&frame, data.len()).unwrap(), data);
        }
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn incompressible_data_gets_a_frame_that_is_not_smaller() {
        let data = noise(4_096, 3);
        let frame = compress(&ctx(), &data).unwrap();
        assert!(frame.len() >= data.len());
        assert_eq!(decode(&frame, data.len()).unwrap(), data);
    }

    #[test]
    fn empty_input_round_trips() {
        let c = ctx();
        let empty = compress(&c, &[]).unwrap();
        assert_eq!(decode(&empty, 0).unwrap(), Vec::<u8>::new());
        let frame = raw_frame(0x20, None, &[0], &[]);
        assert_eq!(decode(&frame, 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn multi_block_page_round_trips() {
        let c = ctx();
        let data = structured(100_000); // 800 kB, several 128 KiB blocks
        let frame = compress(&c, &data).unwrap();
        assert_eq!(decode(&frame, data.len()).unwrap(), data);
    }

    #[test]
    fn lying_content_size_is_rejected() {
        let c = ctx();
        let data = structured(2_000);
        let frame = compress(&c, &data).unwrap();
        let err = decode(&frame, data.len() + 1).unwrap_err();
        assert!(format!("{err:?}").contains("content bytes"), "{err:?}");
        let err = decode(&frame, data.len() - 1).unwrap_err();
        assert!(format!("{err:?}").contains("content bytes"), "{err:?}");
    }

    #[test]
    fn frame_with_more_data_than_declared_is_rejected() {
        // Declares 4 content bytes, holds 8.
        let frame = raw_frame(0x20, None, &[4], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let err = decode(&frame, 4).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
    }

    #[test]
    fn frame_with_less_data_than_declared_is_rejected() {
        let frame = raw_frame(0x20, None, &[8], &[1, 2, 3, 4]);
        let err = decode(&frame, 8).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
    }

    #[test]
    fn huge_window_dictionary_and_checksum_are_rejected() {
        // Window descriptor 0xF8: exponent 31, a 2 TiB window.
        let frame = raw_frame(0x80, Some(0xF8), &[4, 0, 0, 0], &[1, 2, 3, 4]);
        let err = decode(&frame, 4).unwrap_err();
        assert!(format!("{err:?}").contains("window"), "{err:?}");
        // 256 MiB window (exponent 18).
        let frame = raw_frame(0x80, Some(18 << 3), &[4, 0, 0, 0], &[1, 2, 3, 4]);
        assert!(format!("{:?}", decode(&frame, 4).unwrap_err()).contains("window"));
        // Single segment whose content size is the window: 1 GiB.
        let frame = raw_frame(0xE0, None, &(1u64 << 30).to_le_bytes(), &[1, 2, 3, 4]);
        assert!(format!("{:?}", decode(&frame, 4).unwrap_err()).contains("window"));
        // Dictionary id present.
        let frame = raw_frame(0x81, Some(0x00), &[7, 4, 0, 0, 0], &[1, 2, 3, 4]);
        assert!(format!("{:?}", decode(&frame, 4).unwrap_err()).contains("dictionary"));
        // Content checksum flag.
        let frame = raw_frame(0x84, Some(0x00), &[4, 0, 0, 0], &[1, 2, 3, 4]);
        assert!(format!("{:?}", decode(&frame, 4).unwrap_err()).contains("checksum"));
        // No content size on a windowed frame.
        let frame = raw_frame(0x00, Some(0x00), &[], &[1, 2, 3, 4]);
        assert!(format!("{:?}", decode(&frame, 4).unwrap_err()).contains("content size"));
        // Reserved bit.
        let frame = raw_frame(0xA8, None, &[4, 0, 0, 0], &[1, 2, 3, 4]);
        assert!(matches!(decode(&frame, 4).unwrap_err(), Error::Invalid(_)));
    }

    #[test]
    fn skippable_and_bad_magic_frames_are_rejected() {
        let mut skippable = SKIPPABLE_FIRST.to_le_bytes().to_vec();
        skippable.extend_from_slice(&[0, 0, 0, 0]);
        assert!(format!("{:?}", decode(&skippable, 0).unwrap_err()).contains("skippable"));
        assert!(format!("{:?}", decode(b"not zstd at all", 4).unwrap_err()).contains("magic"));
        assert!(decode(&[], 0).is_err());
        assert!(decode(&[0x28, 0xB5, 0x2F], 0).is_err());
    }

    #[test]
    fn data_after_the_frame_is_rejected() {
        let c = ctx();
        let data = structured(2_000);
        let mut frame = compress(&c, &data).unwrap();
        frame.extend_from_slice(&[0, 1, 2]);
        let err = decode(&frame, data.len()).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
    }

    #[test]
    fn truncation_at_every_length_returns_an_error() {
        let c = ctx();
        let data = structured(3_000);
        let frame = compress(&c, &data).unwrap();
        for cut in 0..frame.len() {
            let mut out = vec![0u8; data.len()];
            assert!(
                decompress(&c, &frame[..cut], &mut out).is_err(),
                "cut at {cut} decoded"
            );
        }
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn corrupted_frames_never_panic_out_of_the_adapter() {
        let c = ctx();
        let data = structured(3_000);
        let frame = compress(&c, &data).unwrap();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..3_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let mut bad = frame.clone();
            let at = (seed as usize) % bad.len();
            bad[at] ^= 1 << ((seed >> 40) % 8);
            let mut out = vec![0u8; data.len()];
            // Either a clean error or a (rare) valid decode; never a panic.
            let _ = decompress(&c, &bad, &mut out);
        }
        assert_eq!(c.memory_used(), 0);
    }

    #[test]
    fn decoder_budget_is_charged_and_refused() {
        let data = structured(40_000);
        let frame = compress(&ctx(), &data).unwrap();
        let tiny = Context::new(Some(1), 1 << 20).unwrap();
        let mut out = vec![0u8; data.len()];
        let err = decompress(&tiny, &frame, &mut out).unwrap_err();
        assert!(matches!(err, Error::Budget { .. }), "{err:?}");
        assert_eq!(tiny.memory_used(), 0);
        let err = compress(&tiny, &data).unwrap_err();
        assert!(matches!(err, Error::Budget { .. }), "{err:?}");
        assert_eq!(tiny.memory_used(), 0);
    }

    #[test]
    fn decoder_allocation_stays_inside_the_charge() {
        let c = ctx();
        for n in [1_000usize, 40_000, 400_000, 2_000_000] {
            let data = structured(n);
            let frame = compress(&c, &data).unwrap();
            let audited = inspect(&frame, data.len()).unwrap();
            let charge = audited.charge().unwrap();
            let mut out = vec![0u8; data.len()];
            let (result, peak) = peak_of(|| audited.decode_here(&c, &frame, &mut out));
            result.unwrap();
            assert!(
                peak <= charge,
                "{} B page: decoder peak {peak} > charge {charge}",
                data.len()
            );
        }
    }

    #[test]
    fn lying_small_content_with_a_large_window_allocates_little() {
        // FCS 4 bytes, window 64 MiB (exponent 16). The ring buffer must not
        // grow to the declared window.
        let frame = raw_frame(0x80, Some(16 << 3), &[4, 0, 0, 0], &[1, 2, 3, 4]);
        let audited = inspect(&frame, 4).unwrap();
        let charge = audited.charge().unwrap();
        let c = ctx();
        let mut out = vec![0u8; 4];
        let (result, peak) = peak_of(|| audited.decode_here(&c, &frame, &mut out));
        result.unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        assert!(peak <= charge, "peak {peak} > charge {charge}");
        assert!(peak < (16 << 20), "peak {peak} follows the declared window");
    }

    #[test]
    fn encoder_allocation_stays_inside_the_charge() {
        let c = ctx();
        for n in [1_000usize, 40_000, 400_000, 2_000_000] {
            for data in [structured(n), noise(n * 8, 11)] {
                let charge = data.len() * 2 + ENCODER_FIXED_BYTES;
                let (result, peak) = peak_of(|| compress_here(&c, &data));
                result.unwrap();
                assert!(
                    peak <= charge,
                    "{} B input: encoder peak {peak} > charge {charge}",
                    data.len()
                );
            }
        }
    }

    /// A frame of `blocks` 128 KiB run-length blocks, with the given window
    /// descriptor and content size (4-byte field).
    fn run_frame(window: u8, content: u32, blocks: usize) -> Vec<u8> {
        let mut frame = FRAME_MAGIC.to_le_bytes().to_vec();
        frame.push(0x80);
        frame.push(window);
        frame.extend_from_slice(&content.to_le_bytes());
        for index in 0..blocks {
            let header = ((BLOCK_BYTES as u32) << 3) | (1 << 1) | u32::from(index + 1 == blocks);
            frame.extend_from_slice(&header.to_le_bytes()[..3]);
            frame.push(7);
        }
        frame
    }

    #[test]
    fn ring_buffer_stays_inside_the_charge_for_every_window_and_size() {
        let c = ctx();
        let mut worst = 0.0f64;
        // Windows from the 1 KiB minimum up to a few MiB, contents from one
        // block to 8 MiB: the window may be smaller or larger than the content.
        for window in [0x00u8, 0x28, 0x58, 0x68, 0x70, 0x78, 0x80] {
            for blocks in [1usize, 2, 3, 5, 8, 17, 64] {
                let content = (blocks * BLOCK_BYTES) as u32;
                let frame = run_frame(window, content, blocks);
                let audited = inspect(&frame, content as usize).unwrap();
                let charge = audited.charge().unwrap();
                let mut out = vec![0u8; content as usize];
                let (result, peak) = peak_of(|| audited.decode_here(&c, &frame, &mut out));
                result.unwrap();
                assert!(out.iter().all(|b| *b == 7));
                worst = worst.max(peak as f64 / charge as f64);
                assert!(
                    peak <= charge,
                    "window {window:#04x}, {blocks} blocks: peak {peak} > charge {charge}"
                );
            }
        }
        eprintln!("ring buffer: worst peak/charge {worst:.3}");
    }

    /// A compressed block of `n` sequences that each match three bytes, all
    /// three tables in run-length mode (no bits per sequence), around
    /// `literals` raw literal bytes.
    fn match_block(n: usize, literals: usize) -> Vec<u8> {
        let mut b = Vec::new();
        let header = (1u32 << 2) | ((literals as u32) << 4); // raw, 2-byte header
        assert!(literals < 4096);
        b.extend_from_slice(&header.to_le_bytes()[..2]);
        b.extend(std::iter::repeat_n(0x41u8, literals));
        if n < 128 {
            b.push(n as u8);
        } else if n < 0x7F00 {
            b.push((n >> 8) as u8 + 128);
            b.push((n & 255) as u8);
        } else {
            let m = n - 0x7F00;
            b.extend_from_slice(&[255, (m & 255) as u8, (m >> 8) as u8]);
        }
        b.push(0b01_01_01_00);
        b.extend_from_slice(&[0, 0, 0]); // literal length 0, offset code 0, match 3
        b.push(0x01); // the end marker of an empty bit stream
        b
    }

    fn one_block_frame(content: u32, block: &[u8]) -> Vec<u8> {
        let mut frame = FRAME_MAGIC.to_le_bytes().to_vec();
        frame.push(0x80);
        frame.push(0x00);
        frame.extend_from_slice(&content.to_le_bytes());
        let header = ((block.len() as u32) << 3) | (2 << 1) | 1;
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(block);
        frame
    }

    #[test]
    fn the_audit_counts_matches_exactly_and_refuses_what_exceeds_the_block() {
        // 3 literals and 10 matches of 3 bytes: 33 bytes.
        let frame = one_block_frame(33, &match_block(10, 3));
        assert!(inspect(&frame, 33).is_ok());
        for wrong in [32usize, 34] {
            let frame = one_block_frame(wrong as u32, &match_block(10, 3));
            assert!(inspect(&frame, wrong).is_err(), "{wrong}");
        }
        // 43,690 matches of three bytes are 131,070: the largest legal block.
        let frame = one_block_frame(131_070, &match_block(43_690, 0));
        assert!(inspect(&frame, 131_070).is_ok());
        // One more match is past 128 KiB, whatever the declared size.
        let frame = one_block_frame(131_073, &match_block(43_691, 0));
        let message = format!("{:?}", inspect(&frame, 131_073).unwrap_err());
        assert!(message.contains("128 KiB"), "{message}");
    }

    #[test]
    fn the_largest_sequence_section_stays_inside_the_fixed_charge() {
        // The audit passes; ruzstd then fails on the empty history (the first
        // match has nothing to copy), after allocating its sequence table,
        // literals and block buffers: the worst fixed working set.
        let c = ctx();
        let frame = one_block_frame(131_070, &match_block(43_690, 0));
        let audited = inspect(&frame, 131_070).unwrap();
        let charge = audited.charge().unwrap();
        let mut out = vec![0u8; 131_070];
        let (result, peak) = peak_of(|| audited.decode_here(&c, &frame, &mut out));
        assert!(result.is_err());
        eprintln!("largest sequence section: peak {peak} B, charge {charge} B");
        assert!(peak <= charge, "peak {peak} > charge {charge}");
    }

    #[test]
    fn mutated_frames_that_pass_the_audit_stay_inside_the_charge() {
        let c = ctx();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut sources = Vec::new();
        for data in [structured(20_000), structured(120_000)] {
            sources.push((compress(&c, &data).unwrap(), data.len()));
        }
        let mut accepted = 0usize;
        for case in 0..20_000 {
            let (frame, size) = &sources[case % sources.len()];
            let mut bad = frame.clone();
            for _ in 0..1 + next() % 3 {
                let at = (next() as usize) % bad.len();
                bad[at] = match next() % 3 {
                    0 => bad[at] ^ (1 << (next() % 8)),
                    1 => next() as u8,
                    _ => [0u8, 1, 0x7F, 0x80, 0xFF][(next() % 5) as usize],
                };
            }
            let Ok(audited) = inspect(&bad, *size) else {
                continue;
            };
            accepted += 1;
            let charge = audited.charge().unwrap();
            let mut out = vec![0u8; *size];
            let (_result, peak) = peak_of(|| audited.decode_here(&c, &bad, &mut out));
            assert!(peak <= charge, "case {case}: peak {peak} > charge {charge}");
        }
        assert_eq!(c.memory_used(), 0);
        eprintln!("mutated frames: {accepted} of 20000 passed the audit");
        assert!(accepted > 0, "no mutated frame passed the audit");
    }

    #[test]
    fn hostile_frames_are_refused_by_the_audit_without_decoding() {
        // Expands to megabytes per sequence in ruzstd; one honest content size.
        let mut block = Vec::new();
        let n = 1000usize;
        let header = (3u32 << 2) | ((n as u32) << 4);
        block.extend_from_slice(&header.to_le_bytes()[..3]);
        block.extend(std::iter::repeat_n(0x41u8, n));
        block.extend_from_slice(&[
            ((n >> 8) as u8) + 128,
            (n & 255) as u8,
            0b01_01_01_00,
            1,
            0,
            52,
        ]);
        for _ in 0..n {
            block.extend_from_slice(&0xFFFFu16.to_le_bytes());
        }
        block.push(1);
        let frame = one_block_frame(16, &block);
        let message = format!("{:?}", inspect(&frame, 16).unwrap_err());
        assert!(message.contains("128 KiB"), "{message}");
        // Honest content size, then more blocks.
        let frame = run_frame(0x88, 16, 1024);
        let message = format!("{:?}", inspect(&frame, 16).unwrap_err());
        assert!(message.contains("more bytes than declared"), "{message}");
        // Run-length literals that claim a megabyte.
        let mut block = vec![1 | (3 << 2) | (0xF << 4), 0xFF, 0xFF]; // RLE literals, size format 3
        block.push(0x41);
        block.push(0);
        let frame = one_block_frame(16, &block);
        assert!(inspect(&frame, 16).is_err());
    }
}
