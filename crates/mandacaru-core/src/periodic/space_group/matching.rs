//! Space group identification by matching conventional operations to the catalog.
use crate::math::lattice::{det3, unimodular_inverse};
use crate::periodic::space_group::catalog::{centering_vectors, iter_groups};
use crate::periodic::space_group::classify::CrystalSystem;
use crate::Result;
use nalgebra::Matrix3;
use std::collections::HashMap;

/// Result of space group identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub number: usize,
    pub symbol: &'static str,
}

/// The 24 proper signed permutations of standard axes (det = +1).
pub fn proper_signed_permutations() -> Vec<Matrix3<i32>> {
    let perms = [
        [0, 1, 2], [1, 2, 0], [2, 0, 1],
        [0, 2, 1], [2, 1, 0], [1, 0, 2],
    ];
    let signs = [-1, 1];
    let mut out = Vec::with_capacity(24);
    for p in perms {
        for &s0 in &signs {
            for &s1 in &signs {
                for &s2 in &signs {
                    let mut m = Matrix3::zeros();
                    m[(0, p[0])] = s0;
                    m[(1, p[1])] = s1;
                    m[(2, p[2])] = s2;
                    if det3(&m) == 1 {
                        out.push(m);
                    }
                }
            }
        }
    }
    out
}

/// Proper tetragonal lattice transforms (fixing the 4-fold z-axis or flipping it).
pub fn tetragonal_transforms() -> Vec<Matrix3<i32>> {
    let mut out = Vec::new();
    let in_plane = [
        Matrix3::new(1, 0, 0, 0, 1, 0, 0, 0, 1),
        Matrix3::new(0, -1, 0, 1, 0, 0, 0, 0, 1),
        Matrix3::new(-1, 0, 0, 0, -1, 0, 0, 0, 1),
        Matrix3::new(0, 1, 0, -1, 0, 0, 0, 0, 1),
        Matrix3::new(1, 0, 0, 0, -1, 0, 0, 0, -1),
        Matrix3::new(-1, 0, 0, 0, 1, 0, 0, 0, -1),
        Matrix3::new(0, 1, 0, 1, 0, 0, 0, 0, -1),
        Matrix3::new(0, -1, 0, -1, 0, 0, 0, 0, -1),
    ];
    for m in in_plane {
        if det3(&m) == 1 {
            out.push(m);
        }
    }
    out
}

/// Proper hexagonal lattice transforms (powers of 60° rotation about z and flips).
pub fn hexagonal_transforms() -> Vec<Matrix3<i32>> {
    let rot60 = Matrix3::new(1, -1, 0, 1, 0, 0, 0, 0, 1);
    let flip = Matrix3::new(0, 1, 0, 1, 0, 0, 0, 0, -1);
    let mut out = Vec::with_capacity(12);

    let mut curr = Matrix3::identity();
    for _ in 0..6 {
        if det3(&curr) == 1 {
            out.push(curr);
        }
        let flipped = curr * flip;
        if det3(&flipped) == 1 {
            out.push(flipped);
        }
        curr = curr * rot60;
    }
    out
}

/// Proper monoclinic transforms preserving unique axis b (y-axis).
pub fn monoclinic_transforms() -> Vec<Matrix3<i32>> {
    let mut out = Vec::new();
    let ac_maps = [
        [1, 0, 0, 1],
        [-1, 0, 0, -1],
        [0, 1, -1, 0],
        [0, -1, 1, 0],
        [1, 1, 0, 1],
        [1, -1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, -1, 1],
    ];
    for &[a, b, c, d] in &ac_maps {
        let m = Matrix3::new(a, 0, b, 0, 1, 0, c, 0, d);
        if det3(&m) == 1 {
            out.push(m);
        }
        let m_flip = Matrix3::new(-a, 0, -b, 0, -1, 0, c, 0, d);
        if det3(&m_flip) == 1 {
            out.push(m_flip);
        }
    }
    out
}

/// Get the set of candidate coordinate transforms for a crystal system.
pub fn candidate_transforms(system: CrystalSystem) -> Vec<Matrix3<i32>> {
    match system {
        CrystalSystem::Triclinic => vec![Matrix3::identity()],
        CrystalSystem::Monoclinic => monoclinic_transforms(),
        CrystalSystem::Orthorhombic => proper_signed_permutations(),
        CrystalSystem::Tetragonal => tetragonal_transforms(),
        CrystalSystem::Trigonal | CrystalSystem::Hexagonal => hexagonal_transforms(),
        CrystalSystem::Cubic => proper_signed_permutations(),
    }
}

/// Match conventional operations against the 230 space-group catalog.
pub fn match_space_group(
    conv_ops: &[(Matrix3<i32>, [i32; 3])],
    system: CrystalSystem,
    point_group: &str,
    centering: char,
) -> Result<Identity> {
    let transforms = candidate_transforms(system);
    let mut matches = Vec::new();

    for row in iter_groups() {
        if row.crystal_system != system || row.point_group != point_group || row.centering != centering {
            continue;
        }

        let c_trans = centering_vectors(row.centering);
        let row_ops = row.operations;
        let mut row_map: HashMap<[i32; 9], [i32; 3]> = HashMap::new();
        for (r, t) in row_ops {
            row_map.insert(*r, *t);
        }

        let mut row_matched = false;

        for c_mat in &transforms {
            let c_inv = match unimodular_inverse(c_mat) {
                Ok(inv) => inv,
                Err(_) => continue,
            };

            // Conjugate data operations: W' = C * W * C^-1, t' = C * t
            let mut transformed_ops = Vec::with_capacity(conv_ops.len());
            let mut rot_match = true;

            for (w, t) in conv_ops {
                let w_prime = c_mat * w * c_inv;
                let t_prime = [
                    (c_mat[(0, 0)] * t[0] + c_mat[(0, 1)] * t[1] + c_mat[(0, 2)] * t[2]).rem_euclid(24),
                    (c_mat[(1, 0)] * t[0] + c_mat[(1, 1)] * t[1] + c_mat[(1, 2)] * t[2]).rem_euclid(24),
                    (c_mat[(2, 0)] * t[0] + c_mat[(2, 1)] * t[1] + c_mat[(2, 2)] * t[2]).rem_euclid(24),
                ];
                let key = [
                    w_prime[(0, 0)], w_prime[(0, 1)], w_prime[(0, 2)],
                    w_prime[(1, 0)], w_prime[(1, 1)], w_prime[(1, 2)],
                    w_prime[(2, 0)], w_prime[(2, 1)], w_prime[(2, 2)],
                ];
                if !row_map.contains_key(&key) {
                    rot_match = false;
                    break;
                }
                transformed_ops.push((w_prime, t_prime));
            }

            if !rot_match || transformed_ops.len() != row_map.len() {
                continue;
            }

            // Find origin shift s in (Z/24Z)^3
            // Condition for all ops k: (I - W'_k) s == C t_k - t_{row, k} + c (mod 24)
            // Filter on first operation with W'_k != I
            let non_trivial = transformed_ops
                .iter()
                .find(|(w, _)| *w != Matrix3::identity());

            let mut candidate_shifts = Vec::new();

            if let Some((w_first, t_first)) = non_trivial {
                let key_first = [
                    w_first[(0, 0)], w_first[(0, 1)], w_first[(0, 2)],
                    w_first[(1, 0)], w_first[(1, 1)], w_first[(1, 2)],
                    w_first[(2, 0)], w_first[(2, 1)], w_first[(2, 2)],
                ];
                let t_row_first = row_map[&key_first];

                for sx in 0..24 {
                    for sy in 0..24 {
                        for sz in 0..24 {
                            let s = [sx, sy, sz];
                            let ws = [
                                (w_first[(0, 0)] * s[0] + w_first[(0, 1)] * s[1] + w_first[(0, 2)] * s[2] - s[0] + t_first[0]).rem_euclid(24),
                                (w_first[(1, 0)] * s[0] + w_first[(1, 1)] * s[1] + w_first[(1, 2)] * s[2] - s[1] + t_first[1]).rem_euclid(24),
                                (w_first[(2, 0)] * s[0] + w_first[(2, 1)] * s[1] + w_first[(2, 2)] * s[2] - s[2] + t_first[2]).rem_euclid(24),
                            ];
                            let mut ok = false;
                            for c in c_trans {
                                let expected = [
                                    (t_row_first[0] + c[0]).rem_euclid(24),
                                    (t_row_first[1] + c[1]).rem_euclid(24),
                                    (t_row_first[2] + c[2]).rem_euclid(24),
                                ];
                                if ws == expected {
                                    ok = true;
                                    break;
                                }
                            }
                            if ok {
                                candidate_shifts.push(s);
                            }
                        }
                    }
                }
            } else {
                candidate_shifts.push([0, 0, 0]);
            }

            // Test candidate shifts on all remaining operations
            for s in candidate_shifts {
                let mut all_ok = true;
                for (w, t) in &transformed_ops {
                    let key = [
                        w[(0, 0)], w[(0, 1)], w[(0, 2)],
                        w[(1, 0)], w[(1, 1)], w[(1, 2)],
                        w[(2, 0)], w[(2, 1)], w[(2, 2)],
                    ];
                    let t_row = row_map[&key];
                    let ws = [
                        (w[(0, 0)] * s[0] + w[(0, 1)] * s[1] + w[(0, 2)] * s[2] - s[0] + t[0]).rem_euclid(24),
                        (w[(1, 0)] * s[0] + w[(1, 1)] * s[1] + w[(1, 2)] * s[2] - s[1] + t[1]).rem_euclid(24),
                        (w[(2, 0)] * s[0] + w[(2, 1)] * s[1] + w[(2, 2)] * s[2] - s[2] + t[2]).rem_euclid(24),
                    ];
                    let mut op_ok = false;
                    for c in c_trans {
                        let expected = [
                            (t_row[0] + c[0]).rem_euclid(24),
                            (t_row[1] + c[1]).rem_euclid(24),
                            (t_row[2] + c[2]).rem_euclid(24),
                        ];
                        if ws == expected {
                            op_ok = true;
                            break;
                        }
                    }
                    if !op_ok {
                        all_ok = false;
                        break;
                    }
                }

                if all_ok {
                    row_matched = true;
                    break;
                }
            }

            if row_matched {
                break;
            }
        }

        if row_matched {
            matches.push(Identity {
                number: row.number,
                symbol: row.symbol,
            });
        }
    }

    if matches.len() == 1 {
        Ok(matches[0])
    } else if matches.is_empty() {
        Ok(Identity {
            number: 1,
            symbol: "P1",
        })
    } else {
        // Deterministic resolution if enantiomorphs or origin choices
        Ok(matches[0])
    }
}
