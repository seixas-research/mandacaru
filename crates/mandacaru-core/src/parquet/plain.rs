//! PLAIN value decoding (DOUBLE and BYTE_ARRAY) and dictionary
//! materialization.
//!
//! None of these functions allocates: the caller reserves the budget, sizes the
//! output and passes it in. Functions that push into `Vec`s check the spare
//! capacity first and fail instead of growing, so no byte is ever allocated
//! outside the reservation. Every byte read is bounds-checked; bytes after the
//! values a page declares (fastparquet appends 8 zero bytes to every page) are
//! ignored, but too few bytes are an error.
//!
//! Strings are stored flat: `data` holds the concatenated UTF-8 bytes and
//! `offsets` holds `count + 1` end positions, the first being 0. Every value is
//! validated as UTF-8 on its own, so no value can straddle a boundary.

use crate::parquet::hybrid::HybridDecoder;
use crate::parquet::invalid;
use crate::Result;

/// Indices decoded per step when a dictionary page is materialized.
const INDEX_CHUNK: usize = 4096;

/// Decode `out.len()` PLAIN doubles (explicit little-endian). Returns the
/// number of bytes used; trailing bytes are ignored.
pub fn decode_f64(data: &[u8], out: &mut [f64]) -> Result<usize> {
    let needed = out
        .len()
        .checked_mul(8)
        .ok_or_else(|| invalid("PLAIN double count overflows"))?;
    let Some(body) = data.get(..needed) else {
        return Err(invalid(format!(
            "PLAIN double page holds {} bytes but {} values need {needed}",
            data.len(),
            out.len()
        )));
    };
    for (value, bytes) in out.iter_mut().zip(body.chunks_exact(8)) {
        *value = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
    }
    Ok(needed)
}

/// Result of walking the length prefixes of a PLAIN BYTE_ARRAY page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteArrayScan {
    /// Bytes of the page the `count` values occupy, prefixes included.
    pub consumed: usize,
    /// Total payload bytes (the sum of the value lengths).
    pub bytes: usize,
}

/// Walk `count` length-prefixed values and report how many payload bytes they
/// hold, so the caller can reserve before it allocates. Does not check UTF-8.
/// The loop ends at the first value that does not fit, so a lying `count`
/// costs at most `data.len() / 4` steps.
pub fn scan_byte_array(data: &[u8], count: usize) -> Result<ByteArrayScan> {
    let mut position = 0usize;
    let mut bytes = 0usize;
    for index in 0..count {
        let Some(prefix) = data.get(position..position + 4) else {
            return Err(invalid(format!(
                "BYTE_ARRAY page ends before the length of value {index} of {count}"
            )));
        };
        let length = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
        position += 4;
        if length > data.len() - position {
            return Err(invalid(format!(
                "BYTE_ARRAY value {index} claims {length} bytes but only {} remain",
                data.len() - position
            )));
        }
        position += length;
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| invalid("BYTE_ARRAY total length overflows"))?;
    }
    Ok(ByteArrayScan {
        consumed: position,
        bytes,
    })
}

fn require_spare(
    out_data: &Vec<u8>,
    out_offsets: &Vec<u64>,
    count: usize,
    bytes: usize,
) -> Result<()> {
    // `offsets` starts with a 0 sentinel, pushed here when the vector is empty.
    let offsets_needed = count
        .checked_add(usize::from(out_offsets.is_empty()))
        .ok_or_else(|| invalid("string offset count overflows"))?;
    if out_data.capacity() - out_data.len() < bytes
        || out_offsets.capacity() - out_offsets.len() < offsets_needed
    {
        return Err(invalid(
            "string output was not reserved for these values (internal sizing error)",
        ));
    }
    Ok(())
}

fn push_value(out_data: &mut Vec<u8>, out_offsets: &mut Vec<u64>, value: &[u8]) {
    out_data.extend_from_slice(value);
    out_offsets.push(out_data.len() as u64);
}

/// Decode `count` PLAIN BYTE_ARRAY values as UTF-8 and append them to
/// `out_data` / `out_offsets` (a leading 0 offset is added to an empty
/// `out_offsets`). The vectors must already have the spare capacity the
/// values need (use [`scan_byte_array`]); they are never reallocated. Returns
/// the page bytes used.
pub fn append_byte_array(
    data: &[u8],
    count: usize,
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<u64>,
) -> Result<usize> {
    let scan = scan_byte_array(data, count)?;
    require_spare(out_data, out_offsets, count, scan.bytes)?;
    if out_offsets.is_empty() {
        out_offsets.push(out_data.len() as u64);
    }
    let mut position = 0usize;
    for index in 0..count {
        let length = u32::from_le_bytes([
            data[position],
            data[position + 1],
            data[position + 2],
            data[position + 3],
        ]) as usize;
        let value = &data[position + 4..position + 4 + length];
        if std::str::from_utf8(value).is_err() {
            return Err(invalid(format!(
                "BYTE_ARRAY value {index} is not valid UTF-8"
            )));
        }
        push_value(out_data, out_offsets, value);
        position += 4 + length;
    }
    Ok(scan.consumed)
}

/// Dictionary of doubles: gather `out.len()` values by index.
pub fn gather_f64(dictionary: &[f64], indices: &[u32], out: &mut [f64]) -> Result<()> {
    if indices.len() != out.len() {
        return Err(invalid("dictionary gather output length mismatch"));
    }
    for (slot, &index) in out.iter_mut().zip(indices) {
        let Some(&value) = dictionary.get(index as usize) else {
            return Err(index_out_of_range(index, dictionary.len()));
        };
        *slot = value;
    }
    Ok(())
}

fn index_out_of_range(index: u32, entries: usize) -> crate::Error {
    invalid(format!(
        "dictionary index {index} is out of range for a dictionary of {entries} entries"
    ))
}

/// The first byte of a dictionary-encoded data page is the index bit width.
fn index_decoder(data: &[u8]) -> Result<HybridDecoder<'_>> {
    let Some((&width, indices)) = data.split_first() else {
        return Err(invalid(
            "dictionary-encoded page is empty (no bit width byte)",
        ));
    };
    HybridDecoder::new(indices, u32::from(width))
}

/// Decode a dictionary-encoded page of doubles (bit-width byte, then hybrid
/// indices) into `out`, looking each index up in `dictionary`.
pub fn decode_dictionary_f64(data: &[u8], dictionary: &[f64], out: &mut [f64]) -> Result<()> {
    let mut decoder = index_decoder(data)?;
    let mut indices = [0u32; INDEX_CHUNK];
    for piece in out.chunks_mut(INDEX_CHUNK) {
        let chunk = &mut indices[..piece.len()];
        decoder.read(chunk)?;
        gather_f64(dictionary, chunk, piece)?;
    }
    Ok(())
}

fn entry<'a>(dict_data: &'a [u8], dict_offsets: &[u64], index: u32) -> Result<&'a [u8]> {
    let slot = index as usize;
    let (Some(&start), Some(&end)) = (dict_offsets.get(slot), dict_offsets.get(slot + 1)) else {
        return Err(index_out_of_range(
            index,
            dict_offsets.len().saturating_sub(1),
        ));
    };
    usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| dict_data.get(start..end))
        .ok_or_else(|| invalid("string dictionary offsets are inconsistent"))
}

/// Total payload bytes a dictionary-encoded page of `count` strings expands
/// to, with every index range-checked, so the caller can reserve first.
pub fn scan_dictionary_byte_array(
    data: &[u8],
    count: usize,
    dict_data: &[u8],
    dict_offsets: &[u64],
) -> Result<usize> {
    let mut decoder = index_decoder(data)?;
    let mut indices = [0u32; INDEX_CHUNK];
    let mut bytes = 0usize;
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(INDEX_CHUNK);
        decoder.read(&mut indices[..take])?;
        for &index in &indices[..take] {
            bytes = bytes
                .checked_add(entry(dict_data, dict_offsets, index)?.len())
                .ok_or_else(|| invalid("string dictionary expansion overflows"))?;
        }
        remaining -= take;
    }
    Ok(bytes)
}

/// Append `count` strings from a dictionary-encoded page to `out_data` /
/// `out_offsets`. The dictionary entries were validated as UTF-8 when the
/// dictionary page was loaded. Same capacity contract as
/// [`append_byte_array`]; size it with [`scan_dictionary_byte_array`].
pub fn append_dictionary_byte_array(
    data: &[u8],
    count: usize,
    dict_data: &[u8],
    dict_offsets: &[u64],
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<u64>,
) -> Result<()> {
    let bytes = scan_dictionary_byte_array(data, count, dict_data, dict_offsets)?;
    require_spare(out_data, out_offsets, count, bytes)?;
    if out_offsets.is_empty() {
        out_offsets.push(out_data.len() as u64);
    }
    let mut decoder = index_decoder(data)?;
    let mut indices = [0u32; INDEX_CHUNK];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(INDEX_CHUNK);
        decoder.read(&mut indices[..take])?;
        for &index in &indices[..take] {
            push_value(
                out_data,
                out_offsets,
                entry(dict_data, dict_offsets, index)?,
            );
        }
        remaining -= take;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_f64(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn plain_strings(values: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for value in values {
            out.extend((value.len() as u32).to_le_bytes());
            out.extend(*value);
        }
        out
    }

    fn message(error: crate::Error) -> String {
        format!("{error}")
    }

    /// Load strings into flat storage with the capacity a reader would reserve.
    fn load(page: &[u8], count: usize) -> Result<(Vec<u8>, Vec<u64>)> {
        let scan = scan_byte_array(page, count)?;
        let mut data = Vec::with_capacity(scan.bytes);
        let mut offsets = Vec::with_capacity(count + 1);
        append_byte_array(page, count, &mut data, &mut offsets)?;
        Ok((data, offsets))
    }

    #[test]
    fn doubles_keep_every_bit_pattern() {
        let values = [
            0.0,
            -0.0,
            1.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 4.0,
            f64::from_bits(0x7FF8_0000_0000_1234),
        ];
        let page = plain_f64(&values);
        let mut out = vec![0.0; values.len()];
        assert_eq!(decode_f64(&page, &mut out).unwrap(), 64);
        for (got, want) in out.iter().zip(values) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn doubles_tolerate_eight_trailing_pad_bytes() {
        let mut page = plain_f64(&[1.0, 2.0, 3.0]);
        page.extend([0u8; 8]);
        let mut out = [0.0; 3];
        assert_eq!(decode_f64(&page, &mut out).unwrap(), 24);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn doubles_with_too_few_bytes_are_an_error() {
        let page = plain_f64(&[1.0, 2.0]);
        let mut out = [0.0; 3];
        assert!(decode_f64(&page, &mut out).is_err());
        assert!(decode_f64(&page[..15], &mut [0.0; 2]).is_err());
        assert_eq!(decode_f64(&[], &mut []).unwrap(), 0);
    }

    #[test]
    fn strings_round_trip_with_non_ascii_and_empty_values() {
        let values: [&[u8]; 4] = ["IXYZ".as_bytes(), b"", "ção 𝛼".as_bytes(), b"ZZ"];
        let page = plain_strings(&values);
        let (data, offsets) = load(&page, 4).unwrap();
        assert_eq!(offsets.len(), 5);
        assert_eq!(offsets[0], 0);
        for (i, value) in values.iter().enumerate() {
            assert_eq!(&data[offsets[i] as usize..offsets[i + 1] as usize], *value);
        }
    }

    #[test]
    fn strings_tolerate_trailing_pad_bytes() {
        let mut page = plain_strings(&[b"ab", b"c"]);
        let used = page.len();
        page.extend([0u8; 8]);
        let scan = scan_byte_array(&page, 2).unwrap();
        assert_eq!(
            scan,
            ByteArrayScan {
                consumed: used,
                bytes: 3
            }
        );
        load(&page, 2).unwrap();
    }

    #[test]
    fn strings_append_across_pages_keep_absolute_offsets() {
        let first = plain_strings(&[b"ab", b"cde"]);
        let second = plain_strings(&[b"f"]);
        let mut data = Vec::with_capacity(6);
        let mut offsets = Vec::with_capacity(4);
        append_byte_array(&first, 2, &mut data, &mut offsets).unwrap();
        append_byte_array(&second, 1, &mut data, &mut offsets).unwrap();
        assert_eq!(data, b"abcdef");
        assert_eq!(offsets, [0, 2, 5, 6]);
    }

    #[test]
    fn strings_truncated_everywhere_are_errors() {
        let page = plain_strings(&[b"hello", b"world"]);
        for length in 0..page.len() {
            assert!(scan_byte_array(&page[..length], 2).is_err(), "{length}");
        }
        scan_byte_array(&page, 2).unwrap();
    }

    #[test]
    fn string_length_prefix_beyond_the_page_is_an_error() {
        let mut page = u32::MAX.to_le_bytes().to_vec();
        page.extend(b"abc");
        assert!(message(scan_byte_array(&page, 1).unwrap_err()).contains("claims"));
    }

    #[test]
    fn lying_count_ends_at_the_first_missing_value() {
        let page = plain_strings(&[b"a"]);
        assert!(scan_byte_array(&page, usize::MAX).is_err());
        assert!(scan_byte_array(&page, 2).is_err());
    }

    #[test]
    fn invalid_utf8_is_refused() {
        let page = plain_strings(&[b"ok", &[0xFF, 0xFE]]);
        let error = load(&page, 2).unwrap_err();
        assert!(message(error).contains("not valid UTF-8"));
        // A split multi-byte character across two values is invalid for each.
        let split = "é".as_bytes();
        let page = plain_strings(&[&split[..1], &split[1..]]);
        assert!(load(&page, 2).is_err());
    }

    #[test]
    fn append_refuses_to_outgrow_its_reservation() {
        let page = plain_strings(&[b"abcdef"]);
        let mut data = Vec::with_capacity(3);
        let mut offsets = Vec::with_capacity(2);
        assert!(append_byte_array(&page, 1, &mut data, &mut offsets).is_err());
        assert!(data.is_empty());
    }

    // Dictionary pages.

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

    /// Dictionary data page: width byte plus one bit-packed run of `indices`
    /// (padded to a multiple of 8 with zeros).
    fn index_page(indices: &[u32], width: u32) -> Vec<u8> {
        let mut padded = indices.to_vec();
        while padded.len() % 8 != 0 {
            padded.push(0);
        }
        let mut out = vec![width as u8];
        out.extend(varint(((padded.len() / 8) as u64) << 1 | 1));
        let mut bits = 0u128;
        let mut count = 0u32;
        for &value in &padded {
            bits |= u128::from(value) << count;
            count += width;
            while count >= 8 {
                out.push((bits & 0xFF) as u8);
                bits >>= 8;
                count -= 8;
            }
        }
        out
    }

    #[test]
    fn dictionary_of_doubles_materializes() {
        let dictionary = [10.0, 20.0, 30.0];
        let mut page = index_page(&[2, 0, 1, 1, 2], 2);
        page.extend([0u8; 8]);
        let mut out = [0.0; 5];
        decode_dictionary_f64(&page, &dictionary, &mut out).unwrap();
        assert_eq!(out, [30.0, 10.0, 20.0, 20.0, 30.0]);
    }

    #[test]
    fn width_zero_dictionary_always_picks_entry_zero() {
        // Width byte 0, then an RLE run of 6 with no value bytes.
        let mut page = vec![0u8];
        page.extend(varint(6 << 1));
        let mut out = [0.0; 6];
        decode_dictionary_f64(&page, &[42.5], &mut out).unwrap();
        assert_eq!(out, [42.5; 6]);
    }

    #[test]
    fn width_zero_with_an_empty_dictionary_is_out_of_range() {
        let mut page = vec![0u8];
        page.extend(varint(3 << 1));
        let mut out = [0.0; 3];
        let error = decode_dictionary_f64(&page, &[], &mut out).unwrap_err();
        assert!(message(error).contains("out of range"));
    }

    #[test]
    fn dictionary_index_out_of_range_is_an_error() {
        let page = index_page(&[0, 1, 3], 2);
        let mut out = [0.0; 3];
        let error = decode_dictionary_f64(&page, &[1.0, 2.0, 3.0], &mut out).unwrap_err();
        assert!(message(error).contains("index 3 is out of range"));
    }

    #[test]
    fn dictionary_page_without_a_width_byte_or_with_a_wide_one_is_an_error() {
        assert!(decode_dictionary_f64(&[], &[1.0], &mut [0.0; 1]).is_err());
        assert!(decode_dictionary_f64(&[33, 2, 0], &[1.0], &mut [0.0; 1]).is_err());
    }

    #[test]
    fn dictionary_spanning_several_index_chunks() {
        let dictionary: Vec<f64> = (0..300).map(f64::from).collect();
        let indices: Vec<u32> = (0..10_000u32).map(|i| (i * 7) % 300).collect();
        let page = index_page(&indices, 9);
        let mut out = vec![0.0; indices.len()];
        decode_dictionary_f64(&page, &dictionary, &mut out).unwrap();
        for (got, index) in out.iter().zip(&indices) {
            assert_eq!(*got, f64::from(*index));
        }
    }

    #[test]
    fn dictionary_of_strings_materializes() {
        let entries: [&[u8]; 3] = [b"IIII", b"XXYY", "ç".as_bytes()];
        let (dict_data, dict_offsets) = load(&plain_strings(&entries), 3).unwrap();
        let indices = [2u32, 0, 0, 1, 2, 2];
        let mut page = index_page(&indices, 2);
        page.extend([0u8; 8]);
        let bytes = scan_dictionary_byte_array(&page, 6, &dict_data, &dict_offsets).unwrap();
        assert_eq!(bytes, 2 + 4 + 4 + 4 + 2 + 2);
        let mut data = Vec::with_capacity(bytes);
        let mut offsets = Vec::with_capacity(7);
        append_dictionary_byte_array(&page, 6, &dict_data, &dict_offsets, &mut data, &mut offsets)
            .unwrap();
        let expected: Vec<&[u8]> = indices.iter().map(|&i| entries[i as usize]).collect();
        assert_eq!(offsets.len(), 7);
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(&data[offsets[i] as usize..offsets[i + 1] as usize], *want);
        }
    }

    #[test]
    fn dictionary_of_strings_index_out_of_range_is_an_error() {
        let (dict_data, dict_offsets) = load(&plain_strings(&[b"a", b"b"]), 2).unwrap();
        let page = index_page(&[0, 1, 2], 2);
        let error = scan_dictionary_byte_array(&page, 3, &dict_data, &dict_offsets).unwrap_err();
        assert!(message(error).contains("out of range"));
        let mut data = Vec::with_capacity(8);
        let mut offsets = Vec::with_capacity(8);
        assert!(append_dictionary_byte_array(
            &page,
            3,
            &dict_data,
            &dict_offsets,
            &mut data,
            &mut offsets
        )
        .is_err());
        assert!(data.is_empty() && offsets.is_empty());
    }

    #[test]
    fn dictionary_string_count_larger_than_the_indices_is_an_error() {
        let (dict_data, dict_offsets) = load(&plain_strings(&[b"a"]), 1).unwrap();
        let page = index_page(&[0, 0], 1);
        // The run holds 8 indices (two real, six pad); ask for more than 8.
        assert!(scan_dictionary_byte_array(&page, 9, &dict_data, &dict_offsets).is_err());
        assert!(scan_dictionary_byte_array(&page, usize::MAX, &dict_data, &dict_offsets).is_err());
    }

    #[test]
    fn dictionary_string_append_refuses_to_outgrow_its_reservation() {
        let (dict_data, dict_offsets) = load(&plain_strings(&[b"abcd"]), 1).unwrap();
        let page = index_page(&[0, 0], 1);
        let mut data = Vec::with_capacity(4);
        let mut offsets = Vec::with_capacity(3);
        assert!(append_dictionary_byte_array(
            &page,
            2,
            &dict_data,
            &dict_offsets,
            &mut data,
            &mut offsets
        )
        .is_err());
    }

    #[test]
    fn mutation_sweep_never_panics() {
        let page = plain_strings(&[b"alpha", "beta ç".as_bytes(), b"", b"gamma"]);
        let mut seed = 0x0123_4567_89AB_CDEFu64;
        for _ in 0..20_000 {
            let mut bytes = page.clone();
            for _ in 0..3 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let at = (seed as usize) % bytes.len();
                bytes[at] = (seed >> 32) as u8;
            }
            let _ = load(&bytes, 4);
            let mut out = [0.0; 2];
            let _ = decode_dictionary_f64(&bytes, &[1.0, 2.0], &mut out);
            let _ = scan_dictionary_byte_array(&bytes, 9, b"abc", &[0, 1, 3]);
        }
    }
}
