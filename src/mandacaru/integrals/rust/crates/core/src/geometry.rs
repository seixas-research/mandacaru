use crate::{finite, Error, LengthUnit, Result};

/// Owned geometry in Bohr. ASE and Python objects never enter the engine.
#[derive(Clone, Debug)]
pub struct Geometry {
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    pbc: [bool; 3],
}

impl Geometry {
    pub fn new(
        numbers: Vec<u8>,
        mut positions: Vec<[f64; 3]>,
        mut cell: [[f64; 3]; 3],
        pbc: [bool; 3],
        units: LengthUnit,
    ) -> Result<Self> {
        if numbers.is_empty()
            || numbers.len() != positions.len()
            || numbers.iter().any(|&z| z == 0 || z > 118)
        {
            return Err(Error::Invalid(
                "geometry needs matching nonempty atomic numbers (1..118) and positions".into(),
            ));
        }
        for row in positions.iter_mut().chain(cell.iter_mut()) {
            finite(row, "geometry")?;
            for value in row.iter_mut() {
                *value = units.to_bohr(*value);
            }
            finite(row, "converted geometry")?;
        }
        // Gram-Schmidt only on periodic vectors: a molecule may have a zero
        // cell, and a partially periodic geometry may have zero unused axes.
        let mut orthogonal: Vec<[f64; 3]> = Vec::new();
        for axis in 0..3 {
            if !pbc[axis] {
                continue;
            }
            let mut vector = cell[axis];
            for basis in &orthogonal {
                let projection: f64 = vector.iter().zip(basis).map(|(a, b)| a * b).sum();
                for i in 0..3 {
                    vector[i] -= projection * basis[i];
                }
            }
            let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !norm.is_finite() || norm <= 1e-12 {
                return Err(Error::Invalid(
                    "periodic cell vectors must be nonzero and linearly independent".into(),
                ));
            }
            for x in &mut vector {
                *x /= norm;
            }
            orthogonal.push(vector);
        }
        Ok(Self {
            numbers,
            positions,
            cell,
            pbc,
        })
    }
    pub fn numbers(&self) -> &[u8] {
        &self.numbers
    }
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }
    pub fn cell(&self) -> &[[f64; 3]; 3] {
        &self.cell
    }
    pub fn pbc(&self) -> [bool; 3] {
        self.pbc
    }
}
