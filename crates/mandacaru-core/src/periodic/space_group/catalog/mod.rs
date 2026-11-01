//! Crystallographic catalog of all 230 standard space groups.
pub mod hexagonal_cubic;
pub mod orthorhombic;
pub mod tetragonal;
pub mod triclinic_monoclinic;
pub mod trigonal;

use crate::periodic::space_group::classify::CrystalSystem;

/// An entry in the crystallographic catalog.
#[derive(Clone, Copy, Debug)]
pub struct GroupEntry {
    pub number: usize,
    pub symbol: &'static str,
    pub crystal_system: CrystalSystem,
    pub point_group: &'static str,
    pub centering: char,
    /// Coset representative operations (rotation matrix, translation in 1/24 units).
    pub operations: &'static [([i32; 9], [i32; 3])],
}

/// Look up a group entry by space group number (1..=230).
pub fn lookup(number: usize) -> Option<&'static GroupEntry> {
    if number == 0 || number > 230 {
        return None;
    }
    if number <= 15 {
        Some(&triclinic_monoclinic::GROUPS[number - 1])
    } else if number <= 74 {
        Some(&orthorhombic::GROUPS[number - 16])
    } else if number <= 142 {
        Some(&tetragonal::GROUPS[number - 75])
    } else if number <= 167 {
        Some(&trigonal::GROUPS[number - 143])
    } else {
        Some(&hexagonal_cubic::GROUPS[number - 168])
    }
}

/// Return an iterator over all 230 space group entries.
pub fn iter_groups() -> impl Iterator<Item = &'static GroupEntry> {
    triclinic_monoclinic::GROUPS
        .iter()
        .chain(orthorhombic::GROUPS.iter())
        .chain(tetragonal::GROUPS.iter())
        .chain(trigonal::GROUPS.iter())
        .chain(hexagonal_cubic::GROUPS.iter())
}

/// Centering vectors for each centering type in units of 1/24.
pub fn centering_vectors(centering: char) -> &'static [[i32; 3]] {
    match centering {
        'P' => &[[0, 0, 0]],
        'I' => &[[0, 0, 0], [12, 12, 12]],
        'F' => &[[0, 0, 0], [0, 12, 12], [12, 0, 12], [12, 12, 0]],
        'A' => &[[0, 0, 0], [0, 12, 12]],
        'B' => &[[0, 0, 0], [12, 0, 12]],
        'C' => &[[0, 0, 0], [12, 12, 0]],
        'R' => &[[0, 0, 0], [16, 8, 8], [8, 16, 16]],
        _ => &[[0, 0, 0]],
    }
}
