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

// ---------------------------------------------------------------------------------------------
// Lattice geometry on plain arrays: fractional coordinates, supercells and neighbor vectors. The
// functions take any one length unit and return it; the cell vectors are the ROWS of `cell`.
// ---------------------------------------------------------------------------------------------

type Rows = [[f64; 3]; 3];

fn matrix(cell: &Rows) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::from_fn(|i, j| cell[i][j])
}

/// `x` modulo one with the sign of the divisor (Python's `%`): `[0, 1)`, a negative zero or a
/// value that rounds up to one giving the nearest representable result of that rule.
fn modulo_one(x: f64) -> f64 {
    let r = x % 1.0;
    if r == 0. {
        0.
    } else if r < 0. {
        r + 1.0
    } else {
        r
    }
}

/// The cell with each vanishing vector completed as a basis needs it: none missing is the cell,
/// all three the unit cell, one the unit vector orthogonal to the other two (right-handed with
/// them). Two missing vectors have no unique completion and are refused.
pub fn complete_cell(cell: &Rows) -> Result<Rows> {
    let missing: Vec<usize> = (0..3).filter(|&i| cell[i].iter().all(|&v| v == 0.)).collect();
    let mut out = *cell;
    match missing.len() {
        0 => {}
        3 => out = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        1 => {
            let i = missing[0];
            let (a, b) = (cell[(i + 1) % 3], cell[(i + 2) % 3]);
            // ASE completes row i as cross(row i - 2, row i - 1) = cross(row i + 1, row i + 2)
            let cross = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            if norm == 0. {
                return Err(Error::Invalid("the two cell vectors are parallel".into()));
            }
            out[i] = [cross[0] / norm, cross[1] / norm, cross[2] / norm];
        }
        _ => {
            return Err(Error::Invalid(
                "a cell with two zero vectors has no unique completion: give a full cell".into(),
            ))
        }
    }
    Ok(out)
}

/// The fractional coordinates of `positions` in the basis of the cell vectors (the cell completed
/// by [`complete_cell`]); with `wrap`, the periodic coordinates are brought into `[0, 1)`.
pub fn fractional_coordinates(
    cell: &Rows,
    positions: &[[f64; 3]],
    pbc: [bool; 3],
    wrap: bool,
) -> Result<Vec<[f64; 3]>> {
    let full = matrix(&complete_cell(cell)?);
    let lu = full.transpose().lu();
    let mut out = Vec::with_capacity(positions.len());
    for p in positions {
        finite(p, "positions")?;
        let x = lu
            .solve(&nalgebra::Vector3::new(p[0], p[1], p[2]))
            .ok_or_else(|| Error::Invalid("the cell is singular".into()))?;
        let mut f = [x[0], x[1], x[2]];
        if wrap {
            for axis in 0..3 {
                if pbc[axis] {
                    // twice, as the second pass takes a value that rounded up to one back to zero
                    f[axis] = modulo_one(modulo_one(f[axis]));
                }
            }
        }
        out.push(f);
    }
    Ok(out)
}

/// A cell repeated `repeats` times along its vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Repeated {
    pub numbers: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    pub cell: Rows,
    /// The index in the primitive cell of each atom.
    pub source: Vec<usize>,
}

/// The supercell of `repeats` copies along the three cell vectors: the atoms of the primitive
/// cell in order, then shifted by `m0 a0 + m1 a1 + m2 a2` with `m0` the slowest index and `m2` the
/// fastest, and the cell vectors multiplied by the counts.
pub fn repeat_cell(cell: &Rows, positions: &[[f64; 3]], numbers: &[u8], repeats: [usize; 3]) -> Result<Repeated> {
    if repeats.iter().any(|&m| m == 0) || positions.len() != numbers.len() {
        return Err(Error::Invalid("a supercell needs positive repeats and one number per atom".into()));
    }
    for row in cell.iter().chain(positions) {
        finite(row, "supercell geometry")?;
    }
    let n = positions.len();
    let copies = repeats[0] * repeats[1] * repeats[2];
    let mut out = Repeated {
        numbers: Vec::with_capacity(n * copies),
        positions: Vec::with_capacity(n * copies),
        cell: [[0.; 3]; 3],
        source: Vec::with_capacity(n * copies),
    };
    for m0 in 0..repeats[0] {
        for m1 in 0..repeats[1] {
            for m2 in 0..repeats[2] {
                let shift: [f64; 3] = std::array::from_fn(|j| {
                    m0 as f64 * cell[0][j] + m1 as f64 * cell[1][j] + m2 as f64 * cell[2][j]
                });
                for (i, p) in positions.iter().enumerate() {
                    out.numbers.push(numbers[i]);
                    out.positions.push([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
                    out.source.push(i);
                }
            }
        }
    }
    for axis in 0..3 {
        for j in 0..3 {
            out.cell[axis][j] = repeats[axis] as f64 * cell[axis][j];
        }
    }
    Ok(out)
}

/// The vectors from each atom to every atom (periodic images included, not itself) closer than
/// `cutoff`: for atom `i`, `r_j + S . cell - r_i` of every neighbor `j` and lattice shift `S`
/// (nonzero only along periodic axes) with a length below `cutoff`, ordered by `j` then by `S`.
pub fn neighbor_vectors(cell: &Rows, pbc: [bool; 3], positions: &[[f64; 3]], cutoff: f64) -> Result<Vec<Vec<[f64; 3]>>> {
    if !cutoff.is_finite() || cutoff <= 0. {
        return Err(Error::Invalid("the neighbor cutoff must be positive and finite".into()));
    }
    let fractional = fractional_coordinates(cell, positions, pbc, false)?;
    let full = matrix(&complete_cell(cell)?);
    // The interplanar spacing along each axis bounds how many cells reach `cutoff`.
    let inverse = full.try_inverse().ok_or_else(|| Error::Invalid("the cell is singular".into()))?;
    let reach: [f64; 3] = std::array::from_fn(|a| cutoff * inverse.column(a).norm());
    let mut out = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        let mut found = Vec::new();
        for j in 0..positions.len() {
            let delta: [f64; 3] = std::array::from_fn(|a| fractional[j][a] - fractional[i][a]);
            let range: [(i64, i64); 3] = std::array::from_fn(|a| {
                if pbc[a] {
                    ((-delta[a] - reach[a]).floor() as i64, (-delta[a] + reach[a]).ceil() as i64)
                } else {
                    (0, 0)
                }
            });
            for s0 in range[0].0..=range[0].1 {
                for s1 in range[1].0..=range[1].1 {
                    for s2 in range[2].0..=range[2].1 {
                        let shift: [f64; 3] = std::array::from_fn(|c| {
                            s0 as f64 * cell[0][c] + s1 as f64 * cell[1][c] + s2 as f64 * cell[2][c]
                        });
                        let d: [f64; 3] =
                            std::array::from_fn(|c| positions[j][c] + shift[c] - positions[i][c]);
                        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if length < cutoff && !(i == j && s0 == 0 && s1 == 0 && s2 == 0) {
                            found.push(d);
                        }
                    }
                }
            }
        }
        out.push(found);
    }
    Ok(out)
}
