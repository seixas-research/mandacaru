//! RLE / bit-packed hybrid decoder, used for definition levels (bit width 1)
//! and dictionary indices (bit width 0 to 32).
//!
//! Layout of the encoding (Parquet "RLE" encoding): a sequence of runs, each
//! starting with an unsigned varint header.
//!
//! - header bit 0 = 0: an RLE run of `header >> 1` copies of one value, stored
//!   in `ceil(width / 8)` little-endian bytes;
//! - header bit 0 = 1: a bit-packed run of `(header >> 1) * 8` values, packed
//!   least-significant bit first, `(header >> 1) * width` bytes long.
//!
//! The decoder never trusts a run length. It decodes exactly the number of
//! values the caller asks for: a run longer than that is clamped (the tail is
//! never expanded), and a bit-packed run needs only the bytes of the values
//! actually consumed, so a writer that truncates the last group is accepted.
//! A header that promises more than the data holds is an error only when
//! the values are needed.

use crate::parquet::invalid;
use crate::Result;

/// Widest value the hybrid decoder supports (a dictionary index).
pub const MAX_BIT_WIDTH: u32 = 32;

/// Number of values the level checker decodes at a time.
const CHUNK: usize = 4096;

#[derive(Debug, Clone, Copy)]
enum Run {
    None,
    Rle {
        value: u32,
        left: u64,
    },
    Packed {
        start: usize,
        index: u64,
        total: u64,
    },
}

/// Sequential reader over one hybrid-encoded byte range.
#[derive(Debug)]
pub struct HybridDecoder<'a> {
    data: &'a [u8],
    position: usize,
    width: u32,
    run: Run,
}

impl<'a> HybridDecoder<'a> {
    /// `width` must be in `0..=32`.
    pub fn new(data: &'a [u8], width: u32) -> Result<Self> {
        if width > MAX_BIT_WIDTH {
            return Err(invalid(format!(
                "hybrid bit width {width} exceeds the supported maximum of {MAX_BIT_WIDTH}"
            )));
        }
        Ok(Self {
            data,
            position: 0,
            width,
            run: Run::None,
        })
    }

    fn mask(&self) -> u64 {
        (1u64 << self.width) - 1
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let Some(&byte) = self.data.get(self.position) else {
                return Err(invalid("hybrid run header runs past the end of the data"));
            };
            self.position += 1;
            let payload = u64::from(byte & 0x7F);
            if shift == 63 && payload > 1 {
                return Err(invalid("hybrid run header varint overflows 64 bits"));
            }
            value |= payload << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid("hybrid run header varint is longer than 10 bytes"))
    }

    fn next_run(&mut self) -> Result<()> {
        let header = self.read_varint()?;
        let count = header >> 1;
        if header & 1 == 1 {
            let total = count
                .checked_mul(8)
                .ok_or_else(|| invalid("hybrid bit-packed run length overflows"))?;
            self.run = Run::Packed {
                start: self.position,
                index: 0,
                total,
            };
        } else {
            let bytes = self.width.div_ceil(8) as usize;
            let Some(raw) = self.data.get(self.position..self.position + bytes) else {
                return Err(invalid(
                    "hybrid RLE run value runs past the end of the data",
                ));
            };
            self.position += bytes;
            let mut value = 0u64;
            for (i, &byte) in raw.iter().enumerate() {
                value |= u64::from(byte) << (8 * i);
            }
            if value > self.mask() {
                return Err(invalid(format!(
                    "hybrid RLE run value {value} does not fit in {} bits",
                    self.width
                )));
            }
            self.run = Run::Rle {
                value: value as u32,
                left: count,
            };
        }
        Ok(())
    }

    /// One bit-packed value: `index` counts values from the start of the run.
    fn packed_value(&self, start: usize, index: u64) -> Result<u32> {
        if self.width == 0 {
            return Ok(0);
        }
        let bit = index
            .checked_mul(u64::from(self.width))
            .ok_or_else(|| invalid("hybrid bit offset overflows"))?;
        let byte = usize::try_from(bit / 8)
            .ok()
            .and_then(|offset| start.checked_add(offset))
            .ok_or_else(|| invalid("hybrid bit offset overflows"))?;
        let shift = (bit % 8) as u32;
        let needed = (shift + self.width).div_ceil(8) as usize;
        let word = match self.data.get(byte..) {
            Some(tail) if tail.len() >= 8 => u64::from_le_bytes([
                tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6], tail[7],
            ]),
            Some(tail) if tail.len() >= needed => {
                let mut word = 0u64;
                for (i, &b) in tail.iter().take(needed).enumerate() {
                    word |= u64::from(b) << (8 * i);
                }
                word
            }
            _ => return Err(invalid("hybrid bit-packed run is truncated")),
        };
        Ok(((word >> shift) & self.mask()) as u32)
    }

    /// Fill `out` completely with the next `out.len()` values.
    pub fn read(&mut self, out: &mut [u32]) -> Result<()> {
        let mut filled = 0usize;
        while filled < out.len() {
            let wanted = (out.len() - filled) as u64;
            match self.run {
                Run::None => self.next_run()?,
                Run::Rle { value, left } => {
                    let take = left.min(wanted);
                    out[filled..filled + take as usize].fill(value);
                    filled += take as usize;
                    self.run = if take == left {
                        Run::None
                    } else {
                        Run::Rle {
                            value,
                            left: left - take,
                        }
                    };
                }
                Run::Packed {
                    start,
                    index,
                    total,
                } => {
                    let take = (total - index).min(wanted);
                    for k in 0..take {
                        out[filled + k as usize] = self.packed_value(start, index + k)?;
                    }
                    filled += take as usize;
                    let index = index + take;
                    if index == total {
                        // Whole run consumed: move past its declared bytes
                        // (`total` is a multiple of 8, so this is exact).
                        let bytes = total / 8 * u64::from(self.width);
                        self.position = usize::try_from(bytes)
                            .ok()
                            .and_then(|b| start.checked_add(b))
                            .map_or(self.data.len(), |end| end.min(self.data.len()));
                        self.run = Run::None;
                    } else {
                        self.run = Run::Packed {
                            start,
                            index,
                            total,
                        };
                    }
                }
            }
        }
        Ok(())
    }
}

/// Split a DATA_PAGE v1 level block: a little-endian `u32` byte length, then
/// that many bytes of hybrid-encoded levels. Returns `(levels, rest)`.
pub fn split_length_prefixed(data: &[u8]) -> Result<(&[u8], &[u8])> {
    let Some(prefix) = data.get(..4) else {
        return Err(invalid("level block is shorter than its 4-byte length"));
    };
    let length = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
    let body = &data[4..];
    if length > body.len() {
        return Err(invalid(format!(
            "level block claims {length} bytes but only {} remain",
            body.len()
        )));
    }
    Ok(body.split_at(length))
}

/// Check that the first `count` definition levels (width 1) are all 1, that
/// is, that no value is null. Fails with a message containing "null" at the
/// first level 0. A single RLE run of 1 covering `count` is accepted without
/// expanding it.
pub fn require_all_present(levels: &[u8], count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let mut decoder = HybridDecoder::new(levels, 1)?;
    decoder.next_run()?;
    if let Run::Rle { value, left } = decoder.run {
        if value == 1 && left >= count as u64 {
            return Ok(());
        }
    }
    let mut buffer = [0u32; CHUNK];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(CHUNK);
        decoder.read(&mut buffer[..take])?;
        if buffer[..take].iter().any(|&level| level != 1) {
            return Err(invalid(
                "null values are not supported (definition level 0 found)",
            ));
        }
        remaining -= take;
    }
    Ok(())
}

/// DATA_PAGE v1 helper: split off the definition-level block of an OPTIONAL
/// column, require every value present, and return the bytes after it.
pub fn skip_present_levels_v1(data: &[u8], count: usize) -> Result<&[u8]> {
    let (levels, rest) = split_length_prefixed(data)?;
    require_all_present(levels, count)?;
    Ok(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn rle_run(count: u64, value: u32, width: u32) -> Vec<u8> {
        let mut out = varint(count << 1);
        let bytes = width.div_ceil(8) as usize;
        out.extend_from_slice(&value.to_le_bytes()[..bytes]);
        out
    }

    /// Bit-pack `values` (length a multiple of 8) into one bit-packed run.
    fn packed_run(values: &[u32], width: u32) -> Vec<u8> {
        assert_eq!(values.len() % 8, 0);
        let mut out = varint(((values.len() / 8) as u64) << 1 | 1);
        let mut bits = 0u128;
        let mut count = 0u32;
        for &value in values {
            bits |= u128::from(value) << count;
            count += width;
            while count >= 8 {
                out.push((bits & 0xFF) as u8);
                bits >>= 8;
                count -= 8;
            }
        }
        assert_eq!(count, 0);
        out
    }

    fn decode(data: &[u8], width: u32, n: usize) -> Result<Vec<u32>> {
        let mut out = vec![0u32; n];
        HybridDecoder::new(data, width)?.read(&mut out)?;
        Ok(out)
    }

    fn message(error: crate::Error) -> String {
        format!("{error}")
    }

    #[test]
    fn single_rle_run() {
        assert_eq!(decode(&rle_run(5, 3, 3), 3, 5).unwrap(), vec![3; 5]);
    }

    #[test]
    fn packed_run_round_trips_at_every_width() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        for width in 0..=32u32 {
            let mask = if width == 0 {
                0
            } else {
                ((1u64 << width) - 1) as u32
            };
            let values: Vec<u32> = (0..64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed as u32) & mask
                })
                .collect();
            let data = packed_run(&values, width);
            assert_eq!(decode(&data, width, 64).unwrap(), values, "width {width}");
            // Read in odd-sized pieces to exercise the resumable state.
            let mut decoder = HybridDecoder::new(&data, width).unwrap();
            let mut got = Vec::new();
            for piece in [1usize, 2, 3, 5, 8, 13, 32] {
                let mut part = vec![0u32; piece];
                decoder.read(&mut part).unwrap();
                got.extend(part);
            }
            assert_eq!(got, values[..got.len()], "width {width}");
        }
    }

    #[test]
    fn mixed_runs_continue_across_reads() {
        let mut data = rle_run(3, 2, 2);
        data.extend(packed_run(&[1, 0, 3, 2, 1, 0, 3, 2], 2));
        data.extend(rle_run(4, 1, 2));
        let all = decode(&data, 2, 15).unwrap();
        assert_eq!(all, [2, 2, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 1, 1, 1]);
    }

    #[test]
    fn huge_rle_run_is_clamped_not_expanded() {
        // 2^62 copies promised; only 10 are produced.
        let data = rle_run(1u64 << 62, 1, 1);
        assert_eq!(decode(&data, 1, 10).unwrap(), vec![1; 10]);
    }

    #[test]
    fn huge_packed_run_is_clamped() {
        let mut data = varint(((1u64 << 40) << 1) | 1);
        data.extend([0b0000_0101, 0b1111_0000]);
        let got = decode(&data, 1, 12).unwrap();
        assert_eq!(got, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn packed_run_overflowing_length_is_an_error() {
        let data = varint(u64::MAX >> 1 << 1 | 1);
        let error = decode(&data, 1, 1).unwrap_err();
        assert!(message(error).contains("overflows"));
    }

    #[test]
    fn truncated_data_is_an_error_never_a_panic() {
        let mut data = rle_run(3, 5, 8);
        data.extend(packed_run(&[1, 2, 3, 4, 5, 6, 7, 8], 8));
        for length in 0..data.len() {
            let _ = decode(&data[..length], 8, 11);
        }
        assert!(decode(&data[..2], 8, 11).is_err());
        assert!(decode(&data, 8, 12).is_err());
        assert!(decode(&[], 1, 1).is_err());
    }

    #[test]
    fn rle_value_wider_than_the_width_is_an_error() {
        let data = rle_run(4, 4, 3);
        assert!(message(decode(&data, 2, 4).unwrap_err()).contains("does not fit"));
    }

    #[test]
    fn width_zero() {
        // RLE run of 7 with no value bytes, then a bit-packed run of 8.
        let mut data = varint(7 << 1);
        data.extend(varint(1 << 1 | 1));
        assert_eq!(decode(&data, 0, 15).unwrap(), vec![0; 15]);
    }

    #[test]
    fn width_above_32_is_refused() {
        assert!(HybridDecoder::new(&[], 33).is_err());
    }

    #[test]
    fn varint_longer_than_ten_bytes_is_an_error() {
        let data = [0x80u8; 11];
        assert!(decode(&data, 1, 1).is_err());
        let overflow = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        assert!(decode(&overflow, 1, 1).is_err());
    }

    #[test]
    fn trailing_pad_bytes_are_never_read() {
        let mut data = packed_run(&[1, 0, 1, 1, 0, 0, 1, 0], 1);
        data.extend([0u8; 8]);
        assert_eq!(decode(&data, 1, 8).unwrap(), [1, 0, 1, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn truncated_last_group_is_accepted_when_the_values_fit() {
        // Declares 8 values at width 3 (3 bytes) but stores only 2 bytes,
        // which still hold 5 whole values.
        let mut data = varint(1 << 1 | 1);
        data.extend([0b1010_1001, 0b0000_0101]);
        let got = decode(&data, 3, 5).unwrap();
        assert_eq!(got, [1, 5, 6, 2, 0]);
        assert!(decode(&data, 3, 6).is_err());
    }

    #[test]
    fn mutation_sweep_never_panics() {
        let mut base = rle_run(3, 5, 8);
        base.extend(packed_run(&[9, 8, 7, 6, 5, 4, 3, 2], 8));
        base.extend(rle_run(100, 1, 8));
        let mut seed = 0xDEAD_BEEF_CAFE_F00Du64;
        for _ in 0..20_000 {
            let mut data = base.clone();
            for _ in 0..3 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let at = (seed as usize) % data.len();
                data[at] = (seed >> 32) as u8;
            }
            for width in [0, 1, 7, 8, 32] {
                let _ = decode(&data, width, 20);
            }
        }
    }

    // Definition levels.

    #[test]
    fn single_run_of_ones_is_the_fast_path() {
        // 2^50 values promised, never expanded.
        let data = rle_run(1u64 << 50, 1, 1);
        require_all_present(&data, 100_000).unwrap();
        require_all_present(&[], 0).unwrap();
    }

    #[test]
    fn level_zero_raises_null() {
        let error = require_all_present(&rle_run(10, 0, 1), 10).unwrap_err();
        assert!(message(error).contains("null"));
    }

    #[test]
    fn level_zero_inside_a_packed_run_raises_null() {
        let mut data = rle_run(2, 1, 1);
        data.extend(packed_run(&[1, 1, 1, 0, 1, 1, 1, 1], 1));
        let error = require_all_present(&data, 10).unwrap_err();
        assert!(message(error).contains("null"));
        // The same bytes are fine when the null lies beyond the page's count.
        require_all_present(&data, 5).unwrap();
    }

    #[test]
    fn short_run_of_ones_followed_by_more_ones() {
        let mut data = rle_run(5000, 1, 1);
        data.extend(rle_run(5000, 1, 1));
        require_all_present(&data, 10_000).unwrap();
        assert!(require_all_present(&data, 10_001).is_err());
    }

    #[test]
    fn level_block_split_and_skip() {
        let levels = rle_run(6, 1, 1);
        let mut page = (levels.len() as u32).to_le_bytes().to_vec();
        page.extend(&levels);
        page.extend([7u8; 48]);
        page.extend([0u8; 8]);
        let rest = skip_present_levels_v1(&page, 6).unwrap();
        assert_eq!(rest.len(), 56);
        assert_eq!(rest[0], 7);
    }

    #[test]
    fn level_block_longer_than_the_page_is_an_error() {
        let mut page = 100u32.to_le_bytes().to_vec();
        page.extend([0u8; 10]);
        assert!(split_length_prefixed(&page).is_err());
        assert!(split_length_prefixed(&[1, 0]).is_err());
    }
}
