//! Space group identification by matching conventional operations to the catalog.
use crate::math::lattice::{det3, unimodular_inverse};
use crate::periodic::space_group::catalog::{centering_vectors, iter_groups};
use crate::periodic::space_group::classify::CrystalSystem;
use crate::periodic::space_group::shift;
use crate::periodic::space_group::standard::StandardFrame;
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
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
                    let det2: i32 = a * d - b * c;
                    if det2.abs() == 1 {
                        let m = Matrix3::new(a, 0, b, 0, det2, 0, c, 0, d);
                        out.push(m);
                    }
                }
            }
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

/// Match standardized conventional operations against the 230 space-group catalog.
pub fn match_standard_frame(
    frame: &StandardFrame,
    system: CrystalSystem,
    point_group: &str,
) -> Result<Identity> {
    let transforms = candidate_transforms(system);
    let mut matches = Vec::new();

    for row in iter_groups() {
        if row.crystal_system != system || row.point_group != point_group {
            continue;
        }
        let c_trans = centering_vectors(row.centering);
        if c_trans.len() != frame.centering_vectors.len() {
            continue;
        }

        let mut row_map: HashMap<[i32; 9], [i32; 3]> = HashMap::new();
        for (r, t) in row.operations {
            row_map.insert(*r, *t);
        }

        let mut row_matched = false;

        for c_mat in &transforms {
            let c_inv = match unimodular_inverse(c_mat) {
                Ok(inv) => inv,
                Err(_) => continue,
            };

            // Transform centering filter: C must map frame.centering_vectors onto c_trans mod 24
            let mut cents_match = true;
            for cv in &frame.centering_vectors {
                let mapped = [
                    (c_mat[(0, 0)] * cv[0] + c_mat[(0, 1)] * cv[1] + c_mat[(0, 2)] * cv[2]).rem_euclid(24),
                    (c_mat[(1, 0)] * cv[0] + c_mat[(1, 1)] * cv[1] + c_mat[(1, 2)] * cv[2]).rem_euclid(24),
                    (c_mat[(2, 0)] * cv[0] + c_mat[(2, 1)] * cv[1] + c_mat[(2, 2)] * cv[2]).rem_euclid(24),
                ];
                if !c_trans.contains(&mapped) {
                    cents_match = false;
                    break;
                }
            }
            if !cents_match {
                continue;
            }

            // Conjugate operations: W' = C * W * C^-1, t' = C * t
            let mut transformed_ops = Vec::with_capacity(frame.conv_operations_f64.len());
            let mut rot_match = true;

            for (w, t) in &frame.conv_operations_f64 {
                let w_prime = c_mat * w * c_inv;
                let t_prime = [
                    (c_mat[(0, 0)] as f64 * t[0] + c_mat[(0, 1)] as f64 * t[1] + c_mat[(0, 2)] as f64 * t[2]).rem_euclid(1.0),
                    (c_mat[(1, 0)] as f64 * t[0] + c_mat[(1, 1)] as f64 * t[1] + c_mat[(1, 2)] as f64 * t[2]).rem_euclid(1.0),
                    (c_mat[(2, 0)] as f64 * t[0] + c_mat[(2, 1)] as f64 * t[1] + c_mat[(2, 2)] as f64 * t[2]).rem_euclid(1.0),
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

            if shift::solve_origin_shift_f64(&transformed_ops, &row_map, c_trans, 1e-4).is_some() {
                row_matched = true;
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

    if matches.is_empty() {
        return Err(crate::periodic::invalid("space-group identification failed: no catalog match"));
    }

    let first = matches[0];
    if matches.iter().all(|m| m.number == first.number) {
        Ok(first)
    } else {
        Err(crate::periodic::invalid("space-group identification failed: multiple catalog matches"))
    }
}

/// Match conventional operations against the 230 space-group catalog using exact operations.
pub fn match_space_group(
    conv_ops: &[(Matrix3<i32>, [i32; 3])],
    system: CrystalSystem,
    point_group: &str,
    centering: char,
) -> Result<Identity> {
    let transforms = candidate_transforms(system);
    let mut matches = Vec::new();
    let c_frame = centering_vectors(centering);

    for row in iter_groups() {
        if row.crystal_system != system || row.point_group != point_group || row.centering != centering {
            continue;
        }

        let c_trans = centering_vectors(row.centering);
        let mut row_map: HashMap<[i32; 9], [i32; 3]> = HashMap::new();
        for (r, t) in row.operations {
            row_map.insert(*r, *t);
        }

        let mut row_matched = false;

        for c_mat in &transforms {
            let c_inv = match unimodular_inverse(c_mat) {
                Ok(inv) => inv,
                Err(_) => continue,
            };

            let mut cents_match = true;
            for cv in c_frame {
                let mapped = [
                    (c_mat[(0, 0)] * cv[0] + c_mat[(0, 1)] * cv[1] + c_mat[(0, 2)] * cv[2]).rem_euclid(24),
                    (c_mat[(1, 0)] * cv[0] + c_mat[(1, 1)] * cv[1] + c_mat[(1, 2)] * cv[2]).rem_euclid(24),
                    (c_mat[(2, 0)] * cv[0] + c_mat[(2, 1)] * cv[1] + c_mat[(2, 2)] * cv[2]).rem_euclid(24),
                ];
                if !c_trans.contains(&mapped) {
                    cents_match = false;
                    break;
                }
            }
            if !cents_match {
                continue;
            }

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

            if shift::solve_origin_shift_exact(&transformed_ops, &row_map, c_trans).is_some() {
                row_matched = true;
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

    if matches.is_empty() {
        return Err(crate::periodic::invalid("space-group identification failed: no catalog match"));
    }

    let first = matches[0];
    if matches.iter().all(|m| m.number == first.number) {
        Ok(first)
    } else {
        Err(crate::periodic::invalid("space-group identification failed: multiple catalog matches"))
    }
}

/// Determine whether two catalog group operations are affine equivalent.
/// If `allow_improper` is false, only proper (det = +1) orientation-preserving transforms are tested.
/// If `allow_improper` is true, improper (det = -1) transforms are also tested.
pub fn are_groups_affine_equivalent(
    ops1: &[([[i8; 3]; 3], [i8; 3])],
    ops2: &[([[i8; 3]; 3], [i8; 3])],
    system1: CrystalSystem,
    system2: CrystalSystem,
    centering1: char,
    centering2: char,
    allow_improper: bool,
) -> bool {
    if system1 != system2 || centering1 != centering2 {
        return false;
    }
    let a_cents = centering_vectors(centering1);
    let b_cents = centering_vectors(centering2);
    if a_cents.len() != b_cents.len() {
        return false;
    }

    let proper_transforms = candidate_transforms(system1);
    let mut transforms = proper_transforms.clone();
    if allow_improper {
        for m in &proper_transforms {
            transforms.push(-m);
        }
    }

    let mut b_map: HashMap<[i32; 9], [i32; 3]> = HashMap::new();
    for (r, t) in ops2 {
        let key = [
            r[0][0] as i32, r[0][1] as i32, r[0][2] as i32,
            r[1][0] as i32, r[1][1] as i32, r[1][2] as i32,
            r[2][0] as i32, r[2][1] as i32, r[2][2] as i32,
        ];
        b_map.insert(key, [t[0] as i32, t[1] as i32, t[2] as i32]);
    }

    for c_mat in &transforms {
        let det_c = det3(c_mat);
        if det_c.abs() != 1 {
            continue;
        }
        let c_inv = match unimodular_inverse(c_mat) {
            Ok(inv) => inv,
            Err(_) => continue,
        };

        let mut cents_match = true;
        for cv in a_cents {
            let mapped = [
                (c_mat[(0, 0)] * cv[0] + c_mat[(0, 1)] * cv[1] + c_mat[(0, 2)] * cv[2]).rem_euclid(24),
                (c_mat[(1, 0)] * cv[0] + c_mat[(1, 1)] * cv[1] + c_mat[(1, 2)] * cv[2]).rem_euclid(24),
                (c_mat[(2, 0)] * cv[0] + c_mat[(2, 1)] * cv[1] + c_mat[(2, 2)] * cv[2]).rem_euclid(24),
            ];
            if !b_cents.contains(&mapped) {
                cents_match = false;
                break;
            }
        }
        if !cents_match {
            continue;
        }

        let mut transformed_ops = Vec::with_capacity(ops1.len());
        let mut rot_match = true;

        for (r, t) in ops1 {
            let w = Matrix3::new(
                r[0][0] as i32, r[0][1] as i32, r[0][2] as i32,
                r[1][0] as i32, r[1][1] as i32, r[1][2] as i32,
                r[2][0] as i32, r[2][1] as i32, r[2][2] as i32,
            );
            let w_prime = c_mat * w * c_inv;
            let t_prime = [
                (c_mat[(0, 0)] * t[0] as i32 + c_mat[(0, 1)] * t[1] as i32 + c_mat[(0, 2)] * t[2] as i32).rem_euclid(24),
                (c_mat[(1, 0)] * t[0] as i32 + c_mat[(1, 1)] * t[1] as i32 + c_mat[(1, 2)] * t[2] as i32).rem_euclid(24),
                (c_mat[(2, 0)] * t[0] as i32 + c_mat[(2, 1)] * t[1] as i32 + c_mat[(2, 2)] * t[2] as i32).rem_euclid(24),
            ];
            let key = [
                w_prime[(0, 0)], w_prime[(0, 1)], w_prime[(0, 2)],
                w_prime[(1, 0)], w_prime[(1, 1)], w_prime[(1, 2)],
                w_prime[(2, 0)], w_prime[(2, 1)], w_prime[(2, 2)],
            ];
            if !b_map.contains_key(&key) {
                rot_match = false;
                break;
            }
            transformed_ops.push((w_prime, t_prime));
        }

        if !rot_match || transformed_ops.len() != b_map.len() {
            continue;
        }

        if shift::solve_origin_shift_exact(&transformed_ops, &b_map, b_cents).is_some() {
            return true;
        }
    }

    false
}
