//! Shared atomic shell metadata used by basis and reference-atom models.
pub const AUFBAU: [(usize, usize); 19] = [
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
];
pub fn configuration(z: u8) -> Vec<(usize, usize, usize)> {
    let mut left = z as usize;
    let mut out = Vec::new();
    for (n, l) in AUFBAU {
        if left == 0 {
            break;
        }
        let occupied = left.min(2 * (2 * l + 1));
        left -= occupied;
        out.push((n, l, occupied));
    }
    out.sort_unstable();
    out
}
