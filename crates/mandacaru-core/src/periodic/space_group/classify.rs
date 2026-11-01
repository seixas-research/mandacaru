//! Point group, crystal system, and Laue class classification from (det, trace) signatures.
use crate::math::lattice::det3;
use crate::periodic::invalid;
use crate::Result;
use nalgebra::Matrix3;
use std::collections::HashMap;

/// The seven 3-D crystal systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrystalSystem {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Trigonal,
    Hexagonal,
    Cubic,
}

impl CrystalSystem {
    pub fn name(self) -> &'static str {
        match self {
            Self::Triclinic => "triclinic",
            Self::Monoclinic => "monoclinic",
            Self::Orthorhombic => "orthorhombic",
            Self::Tetragonal => "tetragonal",
            Self::Trigonal => "trigonal",
            Self::Hexagonal => "hexagonal",
            Self::Cubic => "cubic",
        }
    }
}

/// Point group information and crystallographic classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointGroupInfo {
    pub symbol: &'static str,
    pub crystal_system: CrystalSystem,
    pub laue_class: &'static str,
    pub order: usize,
}

const POINT_GROUPS: &[(&str, CrystalSystem, &str, usize, &[((i32, i32), usize)])] = &[
    ("-1", CrystalSystem::Triclinic, "-1", 2, &[((-1, -3), 1), ((1, 3), 1)]),
    ("-3", CrystalSystem::Trigonal, "-3", 6, &[((-1, -3), 1), ((-1, 0), 2), ((1, 0), 2), ((1, 3), 1)]),
    ("-3m", CrystalSystem::Trigonal, "-3m", 12, &[((-1, -3), 1), ((-1, 0), 2), ((-1, 1), 3), ((1, -1), 3), ((1, 0), 2), ((1, 3), 1)]),
    ("-4", CrystalSystem::Tetragonal, "4/m", 4, &[((-1, -1), 2), ((1, -1), 1), ((1, 3), 1)]),
    ("-42m", CrystalSystem::Tetragonal, "4/mmm", 8, &[((-1, -1), 2), ((-1, 1), 2), ((1, -1), 3), ((1, 3), 1)]),
    ("-43m", CrystalSystem::Cubic, "m-3m", 24, &[((-1, -1), 6), ((-1, 1), 6), ((1, -1), 3), ((1, 0), 8), ((1, 3), 1)]),
    ("-6", CrystalSystem::Hexagonal, "6/m", 6, &[((-1, -2), 2), ((-1, 1), 1), ((1, 0), 2), ((1, 3), 1)]),
    ("-6m2", CrystalSystem::Hexagonal, "6/mmm", 12, &[((-1, -2), 2), ((-1, 1), 4), ((1, -1), 3), ((1, 0), 2), ((1, 3), 1)]),
    ("1", CrystalSystem::Triclinic, "-1", 1, &[((1, 3), 1)]),
    ("2", CrystalSystem::Monoclinic, "2/m", 2, &[((1, -1), 1), ((1, 3), 1)]),
    ("2/m", CrystalSystem::Monoclinic, "2/m", 4, &[((-1, -3), 1), ((-1, 1), 1), ((1, -1), 1), ((1, 3), 1)]),
    ("222", CrystalSystem::Orthorhombic, "mmm", 4, &[((1, -1), 3), ((1, 3), 1)]),
    ("23", CrystalSystem::Cubic, "m-3", 12, &[((1, -1), 3), ((1, 0), 8), ((1, 3), 1)]),
    ("3", CrystalSystem::Trigonal, "-3", 3, &[((1, 0), 2), ((1, 3), 1)]),
    ("32", CrystalSystem::Trigonal, "-3m", 6, &[((1, -1), 3), ((1, 0), 2), ((1, 3), 1)]),
    ("3m", CrystalSystem::Trigonal, "-3m", 6, &[((-1, 1), 3), ((1, 0), 2), ((1, 3), 1)]),
    ("4", CrystalSystem::Tetragonal, "4/m", 4, &[((1, -1), 1), ((1, 1), 2), ((1, 3), 1)]),
    ("4/m", CrystalSystem::Tetragonal, "4/m", 8, &[((-1, -3), 1), ((-1, -1), 2), ((-1, 1), 1), ((1, -1), 1), ((1, 1), 2), ((1, 3), 1)]),
    ("4/mmm", CrystalSystem::Tetragonal, "4/mmm", 16, &[((-1, -3), 1), ((-1, -1), 2), ((-1, 1), 5), ((1, -1), 5), ((1, 1), 2), ((1, 3), 1)]),
    ("422", CrystalSystem::Tetragonal, "4/mmm", 8, &[((1, -1), 5), ((1, 1), 2), ((1, 3), 1)]),
    ("432", CrystalSystem::Cubic, "m-3m", 24, &[((1, -1), 9), ((1, 0), 8), ((1, 1), 6), ((1, 3), 1)]),
    ("4mm", CrystalSystem::Tetragonal, "4/mmm", 8, &[((-1, 1), 4), ((1, -1), 1), ((1, 1), 2), ((1, 3), 1)]),
    ("6", CrystalSystem::Hexagonal, "6/m", 6, &[((1, -1), 1), ((1, 0), 2), ((1, 2), 2), ((1, 3), 1)]),
    ("6/m", CrystalSystem::Hexagonal, "6/m", 12, &[((-1, -3), 1), ((-1, -2), 2), ((-1, 0), 2), ((-1, 1), 1), ((1, -1), 1), ((1, 0), 2), ((1, 2), 2), ((1, 3), 1)]),
    ("6/mmm", CrystalSystem::Hexagonal, "6/mmm", 24, &[((-1, -3), 1), ((-1, -2), 2), ((-1, 0), 2), ((-1, 1), 7), ((1, -1), 7), ((1, 0), 2), ((1, 2), 2), ((1, 3), 1)]),
    ("622", CrystalSystem::Hexagonal, "6/mmm", 12, &[((1, -1), 7), ((1, 0), 2), ((1, 2), 2), ((1, 3), 1)]),
    ("6mm", CrystalSystem::Hexagonal, "6/mmm", 12, &[((-1, 1), 6), ((1, -1), 1), ((1, 0), 2), ((1, 2), 2), ((1, 3), 1)]),
    ("m", CrystalSystem::Monoclinic, "2/m", 2, &[((-1, 1), 1), ((1, 3), 1)]),
    ("m-3", CrystalSystem::Cubic, "m-3", 24, &[((-1, -3), 1), ((-1, 0), 8), ((-1, 1), 3), ((1, -1), 3), ((1, 0), 8), ((1, 3), 1)]),
    ("m-3m", CrystalSystem::Cubic, "m-3m", 48, &[((-1, -3), 1), ((-1, -1), 6), ((-1, 0), 8), ((-1, 1), 9), ((1, -1), 9), ((1, 0), 8), ((1, 1), 6), ((1, 3), 1)]),
    ("mm2", CrystalSystem::Orthorhombic, "mmm", 4, &[((-1, 1), 2), ((1, -1), 1), ((1, 3), 1)]),
    ("mmm", CrystalSystem::Orthorhombic, "mmm", 8, &[((-1, -3), 1), ((-1, 1), 3), ((1, -1), 3), ((1, 3), 1)]),
];

/// Classify a group from its distinct rotation matrices.
pub fn classify_point_group(rotations: &[Matrix3<i32>]) -> Result<PointGroupInfo> {
    let mut counts: HashMap<(i32, i32), usize> = HashMap::new();
    for r in rotations {
        let det = det3(r);
        let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
        *counts.entry((det, tr)).or_default() += 1;
    }

    let mut sorted_counts: Vec<((i32, i32), usize)> = counts.into_iter().collect();
    sorted_counts.sort_by_key(|&(key, _)| key);

    for &(sym, sys, laue, order, sig) in POINT_GROUPS {
        if sorted_counts.len() == sig.len() {
            let mut match_all = true;
            for (actual, expected) in sorted_counts.iter().zip(sig.iter()) {
                if actual != expected {
                    match_all = false;
                    break;
                }
            }
            if match_all {
                return Ok(PointGroupInfo {
                    symbol: sym,
                    crystal_system: sys,
                    laue_class: laue,
                    order,
                });
            }
        }
    }

    Err(invalid(format!(
        "unrecognized point group signature: {:?}",
        sorted_counts
    )))
}
