//! The Born-von Karman supercell as the box of a fragment's Coulomb interaction: its grid, its
//! reciprocal vectors and the Coulomb kernel truncated at a radius (Spencer and Alavi 2008).
//!
//! The kernel `v_c(G) = 4 pi / G^2 (1 - cos(|G| R_c))` is the Fourier series of `theta(R_c - r) / r`
//! on the supercell, so a charge that fits inside a sphere of radius `R_c` does not interact with
//! the periodic images of its partner; its `G = 0` limit is `2 pi R_c^2`. Atomic units.
use crate::{
    math::fft::frequency,
    periodic::{invalid, wannier::sampling},
    Result,
};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// `4 pi / G^2 (1 - cos(|G| R_c))`, finite at `G = 0` (`2 pi R_c^2`).
pub fn truncated(g2: f64, radius: f64) -> f64 {
    if g2 > 0. {
        4. * PI / g2 * (1. - (g2.sqrt() * radius).cos())
    } else {
        2. * PI * radius * radius
    }
}

/// The supercell of a mesh: the cell grid repeated `size` times along each axis.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Cells along each axis, and in all.
    pub size: [usize; 3],
    pub cells: usize,
    /// The grid points of one cell, and of the supercell, along each axis.
    pub shape: [usize; 3],
    pub total: [usize; 3],
    /// The cell lattice, vectors as columns (Bohr).
    pub lattice: Matrix3<f64>,
    pub supercell: Matrix3<f64>,
    /// The supercell's reciprocal vectors as columns, `2 pi A^-T`.
    pub reciprocal: Matrix3<f64>,
    /// Supercell volume, volume element of the grid (Bohr^3) and the position of its first node.
    pub volume: f64,
    pub dv: f64,
    pub origin: [f64; 3],
    /// Half the shortest supercell lattice vector: the default truncation radius.
    pub half_shortest: f64,
}

impl Cell {
    pub fn new(
        lattice: Matrix3<f64>,
        shape: [usize; 3],
        size: [usize; 3],
        dv: f64,
        origin: [f64; 3],
    ) -> Result<Self> {
        if size.contains(&0) || shape.contains(&0) || !(dv.is_finite() && dv > 0.) {
            return Err(invalid(
                "the supercell needs positive sizes and grid volume",
            ));
        }
        let supercell = lattice * Matrix3::from_diagonal(&Vector3::from(size.map(|n| n as f64)));
        let reciprocal = 2.
            * PI
            * supercell
                .try_inverse()
                .ok_or_else(|| invalid("singular supercell"))?
                .transpose();
        let mut shortest = f64::INFINITY;
        for a in -1i32..=1 {
            for b in -1i32..=1 {
                for c in -1i32..=1 {
                    if (a, b, c) != (0, 0, 0) {
                        let v = supercell * Vector3::new(a as f64, b as f64, c as f64);
                        shortest = shortest.min(v.norm());
                    }
                }
            }
        }
        Ok(Self {
            size,
            cells: size[0] * size[1] * size[2],
            shape,
            total: std::array::from_fn(|a| size[a] * shape[a]),
            lattice,
            supercell,
            reciprocal,
            volume: supercell.determinant().abs(),
            dv,
            origin,
            half_shortest: 0.5 * shortest,
        })
    }
    /// The cell of a crystal's grid repeated `size` times.
    pub fn of(crystal: &crate::periodic::Crystal, size: [usize; 3]) -> Result<Self> {
        Self::new(
            sampling::lattice(&crystal.grid),
            crystal.grid.index.shape(),
            size,
            crystal.grid.index.dv(),
            crystal.grid.origin,
        )
    }
    /// Grid points of one cell.
    pub fn cell_points(&self) -> usize {
        self.shape.iter().product()
    }
    /// Grid points of the supercell.
    pub fn points(&self) -> usize {
        self.total.iter().product()
    }
    /// The signed integer frequencies of a supercell grid index (C order).
    pub fn frequencies(&self, index: usize) -> [i64; 3] {
        let [_, ny, nz] = self.total;
        let at = [index / (ny * nz), (index / nz) % ny, index % nz];
        std::array::from_fn(|a| frequency(at[a], self.total[a]) as i64)
    }
    /// The wave vector (Bohr^-1) of those frequencies.
    pub fn vector(&self, m: [i64; 3]) -> [f64; 3] {
        (self.reciprocal * Vector3::new(m[0] as f64, m[1] as f64, m[2] as f64)).into()
    }
    /// The wave vector of a supercell grid index.
    pub fn wavevector(&self, index: usize) -> [f64; 3] {
        self.vector(self.frequencies(index))
    }
    /// Whether a frequency sits on a Nyquist plane of the supercell grid (an even grid's
    /// `-N/2`, which has no `+N/2` partner).
    pub fn at_nyquist(&self, m: [i64; 3]) -> bool {
        (0..3).any(|a| self.total[a] % 2 == 0 && 2 * m[a].unsigned_abs() as usize == self.total[a])
    }
    /// The index of the cell `T` that the frequencies `m` of the lattice sums name: `m mod size`,
    /// in C order.
    pub fn cell_of(&self, m: [i64; 3]) -> usize {
        let u: [usize; 3] = std::array::from_fn(|a| m[a].rem_euclid(self.size[a] as i64) as usize);
        (u[0] * self.size[1] + u[1]) * self.size[2] + u[2]
    }
    /// The integer vector of the cell with C-order index `t`.
    pub fn cell_vector(&self, t: usize) -> [i64; 3] {
        let [_, s1, s2] = self.size;
        [
            (t / (s1 * s2)) as i64,
            ((t / s2) % s1) as i64,
            (t % s2) as i64,
        ]
    }
    /// The supercell grid index (C order) of the point `j` of the cell `T`.
    pub fn point_of(&self, cell: [usize; 3], local: [usize; 3]) -> usize {
        let x: [usize; 3] = std::array::from_fn(|a| cell[a] * self.shape[a] + local[a]);
        (x[0] * self.total[1] + x[1]) * self.total[2] + x[2]
    }
    /// The radius of the sphere inscribed in the cell grid's reciprocal box, `pi / h` of its
    /// coarsest axis: the wave number up to which a grid carries a compact charge.
    pub fn grid_sphere(&self) -> f64 {
        (0..3)
            .map(|a| PI * self.shape[a] as f64 / self.lattice.column(a).norm())
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_kernel_is_finite_at_the_origin_and_matches_its_closed_form() {
        assert!((truncated(0., 3.) - 2. * PI * 9.).abs() < 1e-14);
        let (g, r) = (0.7_f64, 4.0_f64);
        let value = truncated(g * g, r);
        assert!((value - 4. * PI / (g * g) * (1. - (g * r).cos())).abs() < 1e-13);
        // continuity of the limit: 4 pi (1 - cos(g R)) / g^2 -> 2 pi R^2 as g -> 0
        assert!((truncated(1e-6, r) - 2. * PI * r * r).abs() < 1e-3);
    }
}
