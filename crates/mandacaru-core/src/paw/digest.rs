//! SHA-256 in lowercase hex: of bytes, and of a dataset's
//! content ([`dataset_content_sha256`], `PawDataset::content_sha256`), the digest a crystal
//! snapshot pins a dataset by. The content digest covers what the record holds for the physics
//! (its payload and tables without the format version and the core states), not the bytes of the
//! file, so a version-2 dataset and its version-3 upgrade have the same one.
use super::invalid;
use crate::Result;
use std::path::Path;

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

/// A streaming SHA-256.
pub(crate) struct Hasher {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    length: u64,
}

impl Hasher {
    pub(crate) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            filled: 0,
            length: 0,
        }
    }
    fn compress(&mut self) {
        let mut w = [0u32; 64];
        for (i, word) in self.block.chunks_exact(4).enumerate() {
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
        let mut v = self.state;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let choice = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(majority);
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
        for (s, x) in self.state.iter_mut().zip(v) {
            *s = s.wrapping_add(x);
        }
    }
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.length += bytes.len() as u64;
        while !bytes.is_empty() {
            let take = (64 - self.filled).min(bytes.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == 64 {
                self.compress();
                self.filled = 0;
            }
        }
    }
    pub(crate) fn finish(mut self) -> String {
        let bits = self.length * 8;
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state.iter().map(|x| format!("{x:08x}")).collect()
    }
}

/// SHA-256 of `bytes`, lowercase hex.
pub fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finish()
}

/// The content digest of the dataset file at `path` (`PawDataset::content_sha256`), the record
/// read under `ctx`'s budget like any dataset.
pub fn dataset_content_sha256(ctx: &crate::Context, path: &Path) -> Result<String> {
    super::record::PawDataset::read(ctx, path)?.content_sha256(ctx)
}

/// The content digest a crystal snapshot's dataset entry pins (its `content_sha256`), `""` when the
/// entry names none (a dataset that was not read from a file: the snapshot loads, and a force or a
/// hybrid sphere that must read the file again refuses it). An entry that still pins the file's
/// bytes (`sha256`, the layout before content digests) is refused with the remedy: such a crystal
/// is saved again. A saved crystal's energy stands regardless: the crystal loader
/// (`periodic::input`) sets the snapshot's description aside on this refusal and raises it only
/// when a force, a stress or a hybrid sphere asks for the files; a new setup request
/// (`periodic::setup::parse_input_json`) is refused outright.
pub fn pinned_content(entry: &serde_json::Value) -> Result<&str> {
    match (entry.get("content_sha256"), entry.get("sha256")) {
        (Some(value), _) => value
            .as_str()
            .ok_or_else(|| invalid("a dataset's content_sha256 must be a string")),
        (None, Some(_)) => Err(invalid(
            "this crystal was saved when snapshots pinned a dataset by the SHA-256 of its file's \
             bytes; datasets are now pinned by their content (content_sha256): build the crystal \
             again from its structure and save it again",
        )),
        (None, None) => Ok(""),
    }
}
