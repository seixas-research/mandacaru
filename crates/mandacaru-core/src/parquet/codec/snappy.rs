//! Snappy raw-block decoder (the format Parquet stores, not the framing
//! format). Read-only: the writer never emits Snappy.
//!
//! The block starts with the uncompressed length as a little-endian varint,
//! which must equal the page's expected size; elements follow until exactly
//! that many bytes have been produced. Every length and copy offset is range
//! checked against the input and the output, so nothing is read or written
//! out of bounds and the decoder never allocates.

use crate::parquet::invalid;
use crate::Result;

/// The most bytes a snappy element expands to per input byte: a three-byte
/// copy produces up to 64 bytes (21.33 per byte), so 22 bounds every block.
const MAX_EXPANSION: usize = 22;

/// Reads the preamble: the uncompressed length as a varint of at most five
/// bytes. Returns the length and the bytes it occupies.
fn preamble(input: &[u8]) -> Result<(u64, usize)> {
    let mut at = 0usize;
    let mut declared: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = *input
            .get(at)
            .ok_or_else(|| invalid("snappy block is truncated"))?;
        at += 1;
        declared |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(invalid("snappy block length is not a valid 32-bit varint"));
        }
    }
    Ok((declared, at))
}

/// Checks, without producing a byte, that `input` can regenerate `expected`
/// bytes: its preamble says so and the body is long enough to hold that much
/// at the largest expansion a snappy element has. A page header that claims
/// far more than its body can hold is refused here, before the caller sizes a
/// buffer from the claim.
pub fn inspect(input: &[u8], expected: usize) -> Result<()> {
    let (declared, at) = preamble(input)?;
    if declared != expected as u64 {
        return Err(invalid(format!(
            "snappy block declares {declared} bytes but the page header says {expected}"
        )));
    }
    let body = input.len() - at;
    if expected / MAX_EXPANSION > body {
        return Err(invalid(format!(
            "snappy block of {body} bytes cannot regenerate the {expected} bytes its page header claims"
        )));
    }
    Ok(())
}

/// Decodes `input` into `output`, which the caller sized to the exact
/// uncompressed length.
pub fn decompress(input: &[u8], output: &mut [u8]) -> Result<()> {
    let truncated = || invalid("snappy block is truncated");
    let (declared, mut at) = preamble(input)?;
    if declared != output.len() as u64 {
        return Err(invalid(format!(
            "snappy block declares {declared} bytes but the page header says {}",
            output.len()
        )));
    }

    let mut out = 0usize;
    while out < output.len() {
        let tag = *input.get(at).ok_or_else(truncated)?;
        at += 1;
        match tag & 3 {
            0 => {
                // Literal.
                let mut len = usize::from(tag >> 2);
                if len >= 60 {
                    let extra = len - 59; // 1..=4 length bytes
                    let bytes = input.get(at..at + extra).ok_or_else(truncated)?;
                    at += extra;
                    len = 0;
                    for (i, b) in bytes.iter().enumerate() {
                        len |= usize::from(*b) << (8 * i);
                    }
                }
                len = len
                    .checked_add(1)
                    .ok_or_else(|| invalid("snappy literal length overflows"))?;
                let source = at
                    .checked_add(len)
                    .and_then(|end| input.get(at..end))
                    .ok_or_else(truncated)?;
                let target = out
                    .checked_add(len)
                    .and_then(|end| output.get_mut(out..end))
                    .ok_or_else(|| invalid("snappy block holds more bytes than declared"))?;
                target.copy_from_slice(source);
                at += len;
                out += len;
            }
            kind => {
                // Copy of `len` bytes from `offset` bytes back.
                let (len, offset) = match kind {
                    1 => {
                        let low = *input.get(at).ok_or_else(truncated)?;
                        at += 1;
                        (
                            4 + usize::from((tag >> 2) & 7),
                            (usize::from(tag >> 5) << 8) | usize::from(low),
                        )
                    }
                    2 => {
                        let b = input.get(at..at + 2).ok_or_else(truncated)?;
                        at += 2;
                        (
                            1 + usize::from(tag >> 2),
                            usize::from(u16::from_le_bytes([b[0], b[1]])),
                        )
                    }
                    _ => {
                        let b = input.get(at..at + 4).ok_or_else(truncated)?;
                        at += 4;
                        let offset = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        (1 + usize::from(tag >> 2), offset as usize)
                    }
                };
                if offset == 0 || offset > out {
                    return Err(invalid("snappy copy reaches before the start of the block"));
                }
                if len > output.len() - out {
                    return Err(invalid("snappy block holds more bytes than declared"));
                }
                // Byte by byte: the source may overlap the bytes being written.
                for i in 0..len {
                    output[out + i] = output[out + i - offset];
                }
                out += len;
            }
        }
    }
    if at != input.len() {
        return Err(invalid("bytes follow the snappy block"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(input: &[u8], n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        decompress(input, &mut out)?;
        Ok(out)
    }

    #[test]
    fn literal_only_block() {
        // length 5, literal tag (len-1)<<2 = 16.
        let block = [5, 16, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(decode(&block, 5).unwrap(), b"hello");
    }

    #[test]
    fn overlapping_copy_repeats_the_pattern() {
        // length 10: literal "ab" then copy len 8 offset 2 (tag 2: (8-1)<<2|2).
        let block = [10, 4, b'a', b'b', (7 << 2) | 2, 2, 0];
        assert_eq!(decode(&block, 10).unwrap(), b"ababababab");
        // 1-byte-offset form: tag = ((len-4)<<2)|1, then the low offset byte.
        let block = [6, 4, b'a', b'b', 1, 2];
        assert_eq!(decode(&block, 6).unwrap(), b"ababab");
    }

    #[test]
    fn four_byte_offset_copy_and_long_literal() {
        // 70-byte literal uses tag 60 (one length byte).
        let lit: Vec<u8> = (0..70u8).collect();
        let mut block = vec![74, 60 << 2, 69];
        block.extend_from_slice(&lit);
        // copy len 4 offset 70 with a 4-byte offset: tag (3<<2)|3.
        block.extend_from_slice(&[(3 << 2) | 3, 70, 0, 0, 0]);
        let out = decode(&block, 74).unwrap();
        assert_eq!(&out[..70], &lit[..]);
        assert_eq!(&out[70..], &lit[..4]);
    }

    #[test]
    fn rejections() {
        // Wrong preamble.
        assert!(decode(&[4, 12, 1, 2, 3, 4], 5).is_err());
        // Offset zero and offset beyond output.
        assert!(decode(&[4, 0, 1, 2, 0, 0], 5).is_err());
        assert!(decode(&[5, 0, 1, (3 << 2) | 2, 5, 0], 5).is_err());
        // Truncated literal, truncated preamble, empty input.
        assert!(decode(&[5, 16, 1, 2], 5).is_err());
        assert!(decode(&[0x80], 0).is_err());
        assert!(decode(&[], 0).is_err());
        // More data than declared (literal runs past the end).
        assert!(decode(&[2, 16, 1, 2, 3, 4, 5], 2).is_err());
        // Trailing bytes after the declared length is reached.
        assert!(decode(&[1, 0, 9, 0], 1).is_err());
        // Oversized varint.
        assert!(decode(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01], 0).is_err());
        // Empty block is valid.
        assert_eq!(decode(&[0], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn every_truncation_of_a_valid_block_fails() {
        let mut block = vec![74, 60 << 2, 69];
        block.extend((0..70u8).collect::<Vec<_>>());
        block.extend_from_slice(&[(3 << 2) | 3, 70, 0, 0, 0]);
        for cut in 0..block.len() {
            assert!(decode(&block[..cut], 74).is_err(), "cut {cut}");
        }
    }
}
