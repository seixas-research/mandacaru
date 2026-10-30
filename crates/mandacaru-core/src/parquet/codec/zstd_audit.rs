//! Size audit of a zstd frame, run before `ruzstd` sees a byte.
//!
//! `ruzstd` 0.8.1 decodes a whole compressed block into its ring buffer before
//! it hands out anything, and it never checks that the block stays within the
//! 128 KiB the format allows (a block of 3,000 bytes can ask for gigabytes).
//! Neither the window nor the declared content size bounds that. So the
//! adapter walks the frame first and proves, from the frame alone, how many
//! bytes every block regenerates:
//!
//! - raw and run-length blocks say it in their 3-byte header;
//! - a compressed block regenerates its literals plus the sum of the match
//!   lengths of its sequences. The audit decodes the sequence section the way
//!   the format defines it (finite-state-entropy tables, the backward bit
//!   stream, the extra-bits tables) but executes nothing: it keeps two
//!   counters and no output.
//!
//! A frame passes only when every block regenerates at most 128 KiB, the
//! blocks add up to exactly the declared content size, and nothing follows the
//! last block. Everything `ruzstd` then allocates is bounded by the content
//! size, which is the page size the file header declared and the `Context`
//! charged. Because the audit sees the real structure of the body, it also
//! tells, before any buffer exists, whether a page can regenerate the size its
//! header claims.
//!
//! The audit never panics: it only indexes through `get` and uses checked or
//! saturating arithmetic.

use crate::parquet::invalid;
use crate::Result;

/// Largest block of a frame (`Block_Maximum_Size`).
pub const BLOCK_BYTES: usize = 128 << 10;

/// Literal-length codes: (baseline, extra bits).
const LL_CODES: [(u32, u8); 36] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 1),
    (18, 1),
    (20, 1),
    (22, 1),
    (24, 2),
    (28, 2),
    (32, 3),
    (40, 3),
    (48, 4),
    (64, 6),
    (128, 7),
    (256, 8),
    (512, 9),
    (1024, 10),
    (2048, 11),
    (4096, 12),
    (8192, 13),
    (16384, 14),
    (32768, 15),
    (65536, 16),
];

/// Match-length codes: (baseline, extra bits).
const ML_CODES: [(u32, u8); 53] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 0),
    (17, 0),
    (18, 0),
    (19, 0),
    (20, 0),
    (21, 0),
    (22, 0),
    (23, 0),
    (24, 0),
    (25, 0),
    (26, 0),
    (27, 0),
    (28, 0),
    (29, 0),
    (30, 0),
    (31, 0),
    (32, 0),
    (33, 0),
    (34, 0),
    (35, 1),
    (37, 1),
    (39, 1),
    (41, 1),
    (43, 2),
    (47, 2),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 5),
    (131, 7),
    (259, 8),
    (515, 9),
    (1027, 10),
    (2051, 11),
    (4099, 12),
    (8195, 13),
    (16387, 14),
    (32771, 15),
    (65539, 16),
];

/// Largest offset code (`Offset_Value` has at most 31 extra bits).
const MAX_OFFSET_CODE: u8 = 31;

/// Predefined distributions (accuracy log 6, 6 and 5).
const LL_DEFAULT: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const ML_DEFAULT: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_DEFAULT: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

fn truncated() -> crate::Error {
    invalid("zstd frame is truncated")
}

/// One state of a decoding table.
#[derive(Clone, Copy, Default)]
struct Entry {
    symbol: u8,
    bits: u8,
    base: u16,
}

/// A finite-state-entropy decoding table.
#[derive(Clone)]
struct Table {
    log: u8,
    entries: Vec<Entry>,
}

impl Table {
    /// The table of a distribution of normalized counts (`-1` is "less than
    /// one"), spread over `1 << log` states as the format prescribes.
    fn build(counts: &[i16], log: u8) -> Result<Table> {
        let size = 1usize << log;
        let mut entries = vec![Entry::default(); size];
        let mut high = size;
        let mut next = [0u32; 64];
        for (symbol, &count) in counts.iter().enumerate() {
            match count {
                -1 => {
                    high = high
                        .checked_sub(1)
                        .ok_or_else(|| invalid("zstd table distribution overflows its table"))?;
                    entries[high] = Entry {
                        symbol: symbol as u8,
                        bits: log,
                        base: 0,
                    };
                    next[symbol] = 1;
                }
                c if c > 0 => next[symbol] = c as u32,
                _ => {}
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mut position = 0usize;
        for (symbol, &count) in counts.iter().enumerate() {
            for _ in 0..count.max(0) {
                entries[position].symbol = symbol as u8;
                position = (position + step) & (size - 1);
                while position >= high {
                    position = (position + step) & (size - 1);
                }
            }
        }
        if position != 0 {
            return Err(invalid("zstd table distribution does not fill its table"));
        }
        for entry in entries.iter_mut().take(high) {
            let symbol = entry.symbol as usize;
            let state = next[symbol];
            next[symbol] += 1;
            let bits = log
                .checked_sub(31 - state.leading_zeros() as u8)
                .ok_or_else(|| invalid("zstd table distribution is inconsistent"))?;
            let base = (state << bits) - size as u32;
            *entry = Entry {
                symbol: entry.symbol,
                bits,
                base: u16::try_from(base)
                    .map_err(|_| invalid("zstd table distribution is inconsistent"))?,
            };
        }
        Ok(Table { log, entries })
    }

    /// The table of a run-length mode: one state, always `symbol`.
    fn run(symbol: u8) -> Table {
        Table {
            log: 0,
            entries: vec![Entry {
                symbol,
                bits: 0,
                base: 0,
            }],
        }
    }
}

/// Bits read from the front of a byte slice, least significant bit first.
struct Forward<'a> {
    data: &'a [u8],
    at: usize,
}

impl Forward<'_> {
    /// The next `n <= 16` bits, zero-padded past the end.
    fn peek(&self, n: u32) -> u32 {
        let byte = self.at / 8;
        let mut word = 0u32;
        for i in 0..4 {
            if let Some(&b) = self.data.get(byte + i) {
                word |= u32::from(b) << (8 * i);
            }
        }
        (word >> (self.at % 8)) & ((1u32 << n) - 1)
    }

    fn skip(&mut self, n: u32) -> Result<()> {
        self.at += n as usize;
        if self.at > self.data.len() * 8 {
            return Err(truncated());
        }
        Ok(())
    }

    fn take(&mut self, n: u32) -> Result<u32> {
        let value = self.peek(n);
        self.skip(n)?;
        Ok(value)
    }
}

/// Reads a table description: its accuracy log, the normalized counts and the
/// bytes the description occupies.
fn read_distribution(
    input: &[u8],
    max_log: u8,
    max_symbols: usize,
) -> Result<(Vec<i16>, u8, usize)> {
    let mut bits = Forward { data: input, at: 0 };
    let log = 5 + bits.take(4)? as u8;
    if log > max_log {
        return Err(invalid("zstd table accuracy log is above the format limit"));
    }
    let size = 1i32 << log;
    let mut remaining = size + 1;
    let mut threshold = size;
    let mut width = u32::from(log) + 1;
    let mut counts: Vec<i16> = Vec::new();
    let mut after_zero = false;
    while remaining > 1 {
        if counts.len() >= max_symbols {
            return Err(invalid("zstd table describes too many symbols"));
        }
        if after_zero {
            // Runs of zero counts: two-bit repeat flags, 3 means "more".
            loop {
                let repeat = bits.take(2)? as usize;
                counts.resize(counts.len() + repeat, 0);
                if repeat != 3 {
                    break;
                }
                if counts.len() > max_symbols {
                    return Err(invalid("zstd table describes too many symbols"));
                }
            }
            if counts.len() >= max_symbols {
                return Err(invalid("zstd table describes too many symbols"));
            }
        }
        let limit = 2 * threshold - 1 - remaining;
        let low = bits.peek(width - 1) as i32;
        let count = if low < limit {
            bits.skip(width - 1)?;
            low
        } else {
            let wide = bits.peek(width) as i32;
            bits.skip(width)?;
            if wide >= threshold {
                wide - limit
            } else {
                wide
            }
        } - 1;
        remaining -= count.abs();
        if remaining < 1 {
            return Err(invalid("zstd table distribution overshoots its total"));
        }
        counts.push(count as i16);
        after_zero = count == 0;
        while remaining < threshold {
            width -= 1;
            threshold >>= 1;
        }
    }
    if remaining != 1 {
        return Err(invalid("zstd table distribution does not add up"));
    }
    Ok((counts, log, bits.at.div_ceil(8)))
}

/// Bits read from the back of a byte slice, as a sequence bit stream is: the
/// last byte holds a 1 marker above the data, and reading goes downward.
struct Backward<'a> {
    data: &'a [u8],
    /// Data bits that are left.
    left: usize,
}

impl<'a> Backward<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        let last = *data.last().ok_or_else(truncated)?;
        if last == 0 {
            return Err(invalid("zstd sequence stream has no end marker"));
        }
        let marker = 7 - last.leading_zeros() as usize;
        Ok(Self {
            data,
            left: (data.len() - 1) * 8 + marker,
        })
    }

    /// The next `n <= 32` bits.
    fn take(&mut self, n: u32) -> Result<u32> {
        let n = n as usize;
        if n > self.left {
            return Err(invalid("zstd sequence stream ends early"));
        }
        let start = self.left - n;
        let byte = start / 8;
        let mut word = 0u64;
        for i in 0..5 {
            if let Some(&b) = self.data.get(byte + i) {
                word |= u64::from(b) << (8 * i);
            }
        }
        self.left = start;
        Ok(((word >> (start % 8)) & ((1u64 << n) - 1)) as u32)
    }
}

/// One of the three symbol streams of a sequence section.
struct Stream<'a> {
    table: &'a Table,
    state: usize,
}

impl<'a> Stream<'a> {
    fn start(table: &'a Table, bits: &mut Backward<'_>) -> Result<Self> {
        let state = bits.take(u32::from(table.log))? as usize;
        Ok(Self { table, state })
    }

    fn entry(&self) -> Result<Entry> {
        self.table
            .entries
            .get(self.state)
            .copied()
            .ok_or_else(|| invalid("zstd table state is out of range"))
    }

    fn symbol(&self) -> Result<usize> {
        Ok(usize::from(self.entry()?.symbol))
    }

    fn advance(&mut self, bits: &mut Backward<'_>) -> Result<()> {
        let entry = self.entry()?;
        self.state = usize::from(entry.base) + bits.take(u32::from(entry.bits))? as usize;
        Ok(())
    }
}

/// The three tables of a frame; a `Repeat` mode reuses the previous block's.
#[derive(Default)]
struct Tables {
    literal_lengths: Option<Table>,
    offsets: Option<Table>,
    match_lengths: Option<Table>,
}

/// How a table is given (`Symbol_Compression_Mode`).
fn read_table(
    mode: u8,
    input: &[u8],
    used: &mut usize,
    slot: &mut Option<Table>,
    default: (&[i16], u8),
    max_log: u8,
    max_symbols: usize,
) -> Result<()> {
    let rest = input.get(*used..).ok_or_else(truncated)?;
    match mode {
        0 => *slot = Some(Table::build(default.0, default.1)?),
        1 => {
            let symbol = *rest.first().ok_or_else(truncated)?;
            if usize::from(symbol) >= max_symbols {
                return Err(invalid("zstd run-length table symbol is out of range"));
            }
            *used += 1;
            *slot = Some(Table::run(symbol));
        }
        2 => {
            let (counts, log, bytes) = read_distribution(rest, max_log, max_symbols)?;
            *used += bytes;
            *slot = Some(Table::build(&counts, log)?);
        }
        _ => {
            if slot.is_none() {
                return Err(invalid("zstd block repeats a table that was never given"));
            }
        }
    }
    Ok(())
}

/// Bytes a compressed block regenerates, or why it cannot be trusted.
fn compressed_block(block: &[u8], tables: &mut Tables) -> Result<usize> {
    // Literals section.
    let first = *block.first().ok_or_else(truncated)?;
    let kind = first & 3;
    let format = (first >> 2) & 3;
    let byte = |i: usize| block.get(i).copied().map(u32::from).ok_or_else(truncated);
    let (header, regenerated, stored) = if kind < 2 {
        let (header, regenerated) = match format {
            0 | 2 => (1, u32::from(first >> 3)),
            1 => (2, u32::from(first >> 4) | (byte(1)? << 4)),
            _ => (
                3,
                u32::from(first >> 4) | (byte(1)? << 4) | (byte(2)? << 12),
            ),
        };
        (header, regenerated, if kind == 0 { regenerated } else { 1 })
    } else {
        let (header, width) = match format {
            0 | 1 => (3usize, 10u32),
            2 => (4, 14),
            _ => (5, 18),
        };
        let mut value = 0u64;
        for i in 0..header {
            value |= u64::from(byte(i)?) << (8 * i);
        }
        let mask = (1u64 << width) - 1;
        (
            header,
            ((value >> 4) & mask) as u32,
            ((value >> (4 + width)) & mask) as u32,
        )
    };
    let literals = regenerated as usize;
    if literals > BLOCK_BYTES {
        return Err(invalid(
            "zstd block regenerates more than 128 KiB of literals",
        ));
    }
    let after_literals = header
        .checked_add(stored as usize)
        .filter(|end| *end <= block.len())
        .ok_or_else(truncated)?;

    // Sequences section header.
    let rest = &block[after_literals..];
    let lead = *rest.first().ok_or_else(truncated)?;
    let (count, modes_at) = match lead {
        0 => (0usize, 1usize),
        1..=127 => (usize::from(lead), 1),
        128..=254 => {
            let low = *rest.get(1).ok_or_else(truncated)?;
            ((usize::from(lead) - 128) * 256 + usize::from(low), 2)
        }
        255 => {
            let low = *rest.get(1).ok_or_else(truncated)?;
            let high = *rest.get(2).ok_or_else(truncated)?;
            (usize::from(low) + (usize::from(high) << 8) + 0x7F00, 3)
        }
    };
    if count == 0 {
        // No sequences: the literals are the block, and nothing may follow.
        if rest.len() != modes_at {
            return Err(invalid(
                "zstd block has bytes after an empty sequence section",
            ));
        }
        return Ok(literals);
    }
    let modes = *rest.get(modes_at).ok_or_else(truncated)?;
    let body = &rest[modes_at + 1..];
    let mut used = 0usize;
    read_table(
        modes >> 6,
        body,
        &mut used,
        &mut tables.literal_lengths,
        (&LL_DEFAULT, 6),
        9,
        LL_CODES.len(),
    )?;
    read_table(
        (modes >> 4) & 3,
        body,
        &mut used,
        &mut tables.offsets,
        (&OF_DEFAULT, 5),
        8,
        usize::from(MAX_OFFSET_CODE) + 1,
    )?;
    read_table(
        (modes >> 2) & 3,
        body,
        &mut used,
        &mut tables.match_lengths,
        (&ML_DEFAULT, 6),
        9,
        ML_CODES.len(),
    )?;
    let stream = body.get(used..).ok_or_else(truncated)?;
    let (Some(ll), Some(of), Some(ml)) = (
        tables.literal_lengths.as_ref(),
        tables.offsets.as_ref(),
        tables.match_lengths.as_ref(),
    ) else {
        return Err(invalid("zstd block is missing a sequence table"));
    };

    // The sequences: their literal lengths must fit the literals, and the
    // block regenerates the literals plus every match.
    let mut bits = Backward::new(stream)?;
    let mut ll_stream = Stream::start(ll, &mut bits)?;
    let mut of_stream = Stream::start(of, &mut bits)?;
    let mut ml_stream = Stream::start(ml, &mut bits)?;
    let mut literal_total = 0usize;
    let mut regenerated_total = literals;
    for index in 0..count {
        let (ll_base, ll_bits) = *LL_CODES
            .get(ll_stream.symbol()?)
            .ok_or_else(|| invalid("zstd literal-length code is out of range"))?;
        let (ml_base, ml_bits) = *ML_CODES
            .get(ml_stream.symbol()?)
            .ok_or_else(|| invalid("zstd match-length code is out of range"))?;
        let offset_bits = of_stream.symbol()? as u32;
        if offset_bits > u32::from(MAX_OFFSET_CODE) {
            return Err(invalid("zstd offset code is out of range"));
        }
        // Extra bits come offset first, then match length, then literal length.
        bits.take(offset_bits)?;
        let match_length = ml_base as usize + bits.take(u32::from(ml_bits))? as usize;
        let literal_length = ll_base as usize + bits.take(u32::from(ll_bits))? as usize;
        literal_total += literal_length;
        regenerated_total += match_length;
        if literal_total > literals {
            return Err(invalid("zstd block uses more literals than it holds"));
        }
        if regenerated_total > BLOCK_BYTES {
            return Err(invalid("zstd block regenerates more than 128 KiB"));
        }
        if index + 1 < count {
            ll_stream.advance(&mut bits)?;
            ml_stream.advance(&mut bits)?;
            of_stream.advance(&mut bits)?;
        }
    }
    if bits.left != 0 {
        return Err(invalid("zstd sequence stream has unused bits"));
    }
    Ok(regenerated_total)
}

/// Checks that the blocks of a frame (the bytes after the frame header)
/// regenerate exactly `content` bytes, each block at most 128 KiB, and that
/// nothing follows the last block.
pub fn blocks(input: &[u8], content: u64) -> Result<()> {
    let mut at = 0usize;
    let mut total = 0u64;
    let mut tables = Tables::default();
    loop {
        let head = input.get(at..at + 3).ok_or_else(truncated)?;
        at += 3;
        let word = u32::from(head[0]) | u32::from(head[1]) << 8 | u32::from(head[2]) << 16;
        let last = word & 1 == 1;
        let size = (word >> 3) as usize;
        if size > BLOCK_BYTES {
            return Err(invalid("zstd block is larger than 128 KiB"));
        }
        let regenerated = match (word >> 1) & 3 {
            0 => {
                at = at
                    .checked_add(size)
                    .filter(|end| *end <= input.len())
                    .ok_or_else(truncated)?;
                size
            }
            1 => {
                if at >= input.len() {
                    return Err(truncated());
                }
                at += 1;
                size
            }
            2 => {
                let end = at
                    .checked_add(size)
                    .filter(|end| *end <= input.len())
                    .ok_or_else(truncated)?;
                let regenerated = compressed_block(&input[at..end], &mut tables)?;
                at = end;
                regenerated
            }
            _ => return Err(invalid("zstd frame holds a reserved block type")),
        };
        total += regenerated as u64;
        if total > content {
            return Err(invalid("zstd frame holds more bytes than declared"));
        }
        if last {
            break;
        }
    }
    if at != input.len() {
        return Err(invalid(
            "bytes follow the zstd frame (multiple frames are not supported)",
        ));
    }
    if total != content {
        return Err(invalid(
            "zstd frame holds fewer bytes than declared or is corrupt",
        ));
    }
    Ok(())
}
