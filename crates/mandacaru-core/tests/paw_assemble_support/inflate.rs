//! DEFLATE decoder (RFC 1951) for the compressed `.npz` members of the A0 freeze, after the
//! `puff` reference decoder: canonical Huffman codes read bit by bit. Test-only; the crate
//! has no zlib dependency and the frozen inputs are written with `np.savez_compressed`.
#![allow(dead_code)]

struct Bits<'a> {
    data: &'a [u8],
    at: usize,
    buffer: u32,
    count: u32,
}

impl Bits<'_> {
    fn take(&mut self, need: u32) -> u32 {
        let mut value = self.buffer;
        while self.count < need {
            value |= (self.data[self.at] as u32) << self.count;
            self.at += 1;
            self.count += 8;
        }
        self.buffer = value >> need;
        self.count -= need;
        value & ((1u32 << need) - 1)
    }
}

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offsets[l as usize] as usize] = s as u16;
                offsets[l as usize] += 1;
            }
        }
        Self { count, symbol }
    }

    fn decode(&self, bits: &mut Bits) -> usize {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= bits.take(1) as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                return self.symbol[(index + code - first) as usize] as usize;
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        panic!("invalid Huffman code in a compressed npz member");
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn codes(bits: &mut Bits, out: &mut Vec<u8>, literal: &Huffman, distance: &Huffman) {
    loop {
        let symbol = literal.decode(bits);
        if symbol < 256 {
            out.push(symbol as u8);
        } else if symbol == 256 {
            return;
        } else {
            let s = symbol - 257;
            let length = LENGTH_BASE[s] as usize + bits.take(LENGTH_EXTRA[s] as u32) as usize;
            let d = distance.decode(bits);
            let back = DISTANCE_BASE[d] as usize + bits.take(DISTANCE_EXTRA[d] as u32) as usize;
            assert!(back <= out.len(), "deflate distance beyond the output");
            for _ in 0..length {
                out.push(out[out.len() - back]);
            }
        }
    }
}

/// The bytes a raw DEFLATE stream expands to.
pub fn inflate(data: &[u8], expected: usize) -> Vec<u8> {
    let mut bits = Bits {
        data,
        at: 0,
        buffer: 0,
        count: 0,
    };
    let mut out = Vec::with_capacity(expected);
    loop {
        let last = bits.take(1);
        match bits.take(2) {
            0 => {
                bits.buffer = 0;
                bits.count = 0;
                let len = u16::from_le_bytes([data[bits.at], data[bits.at + 1]]) as usize;
                out.extend_from_slice(&data[bits.at + 4..bits.at + 4 + len]);
                bits.at += 4 + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                for (i, l) in lengths.iter_mut().enumerate() {
                    *l = match i {
                        0..=143 => 8,
                        144..=255 => 9,
                        256..=279 => 7,
                        _ => 8,
                    };
                }
                codes(
                    &mut bits,
                    &mut out,
                    &Huffman::new(&lengths),
                    &Huffman::new(&[5u8; 30]),
                );
            }
            2 => {
                let (nlen, ndist, ncode) = (
                    bits.take(5) as usize + 257,
                    bits.take(5) as usize + 1,
                    bits.take(4) as usize + 4,
                );
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut lengths = [0u8; 320];
                for &slot in ORDER.iter().take(ncode) {
                    lengths[slot] = bits.take(3) as u8;
                }
                let code_lengths = Huffman::new(&lengths[..19]);
                let mut lengths = [0u8; 320];
                let mut index = 0;
                while index < nlen + ndist {
                    let symbol = code_lengths.decode(&mut bits);
                    if symbol < 16 {
                        lengths[index] = symbol as u8;
                        index += 1;
                    } else {
                        let (value, repeat) = match symbol {
                            16 => (lengths[index - 1], 3 + bits.take(2)),
                            17 => (0, 3 + bits.take(3)),
                            _ => (0, 11 + bits.take(7)),
                        };
                        for _ in 0..repeat {
                            lengths[index] = value;
                            index += 1;
                        }
                    }
                }
                codes(
                    &mut bits,
                    &mut out,
                    &Huffman::new(&lengths[..nlen]),
                    &Huffman::new(&lengths[nlen..nlen + ndist]),
                );
            }
            _ => panic!("invalid deflate block type"),
        }
        if last == 1 {
            return out;
        }
    }
}
