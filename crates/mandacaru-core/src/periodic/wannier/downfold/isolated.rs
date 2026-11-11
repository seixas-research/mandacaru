//! The fragment alone in its box: `coulomb="isolated"`.
//!
//! Each function, Born-von Karman periodic on the supercell, is kept on the supercell's
//! Wigner-Seitz cell centered on the fragment and set in a supercell twice as large in every
//! direction, zero elsewhere; each atom copy's projector coefficients are kept the same way. The
//! truncation radius of that box is the original supercell's shortest lattice vector, twice the
//! plain one, so the fragment's charge stays clear of its copies up to that diameter. The grid has
//! eight times the points.
use super::{
    coulomb::Cell,
    functions::{axis_sources, shifted_coefficients, Functions},
    pairs::Mode,
};
use crate::{
    periodic::{invalid, response::setup::Setup, setup::remainder::projector_columns, Crystal},
    product,
    quantum::C64,
    Result,
};
use nalgebra::{Matrix3, Vector3};

/// For each point, `1 / (number of equally near images)` when the point is (one of) the nearest
/// image(s) of `center` among its translations by the supercell's lattice vectors, within the
/// double supercell `period`; `0` otherwise: the Wigner-Seitz cell of the supercell centered
/// there, ties shared.
fn nearest_image(
    points: impl Iterator<Item = Vector3<f64>>,
    center: Vector3<f64>,
    supercell: &Matrix3<f64>,
    period: &Matrix3<f64>,
) -> Result<Vec<f64>> {
    let inverse = period
        .try_inverse()
        .ok_or_else(|| invalid("a singular doubled supercell"))?;
    let shifts: Vec<Vector3<f64>> = (-1..=1)
        .flat_map(|a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| (a, b, c))))
        .map(|(a, b, c)| supercell * Vector3::new(a as f64, b as f64, c as f64))
        .collect();
    Ok(points
        .map(|x| {
            let f = inverse * (x - center);
            let d = period * f.map(|v| v - v.round());
            let own = d.norm_squared();
            let others: Vec<f64> = shifts.iter().map(|s| (d + s).norm_squared()).collect();
            let nearest = others.iter().cloned().fold(f64::INFINITY, f64::min);
            let ties = others
                .iter()
                .filter(|o| **o <= own * (1. + 1e-9) + 1e-12)
                .count()
                .max(1);
            if own <= nearest * (1. + 1e-9) + 1e-12 {
                1. / ties as f64
            } else {
                0.
            }
        })
        .collect())
}

/// The fragment `modes` of `functions` in the doubled box centered at `center` (Bohr): the doubled
/// cell, the padded functions (the modes of the new set are `0..modes.len()` at the origin cell)
/// and their identity modes.
pub(in crate::periodic) fn padded(
    crystal: &Crystal,
    setup: &Setup,
    cell: &Cell,
    functions: &Functions,
    modes: &[Mode],
    center: [f64; 3],
) -> Result<(Cell, Functions, Vec<Mode>)> {
    let ctx = &crystal.context;
    let big = Cell::new(
        cell.lattice,
        cell.shape,
        std::array::from_fn(|a| 2 * cell.size[a]),
        cell.dv,
        cell.origin,
    )?;
    let center = Vector3::from(center);
    let period = big.supercell;
    // the mask of the grid points
    let [nx, ny, nz] = big.total;
    let step = crystal.grid.step;
    let origin = Vector3::from(cell.origin);
    let points = (0..big.points()).map(|i| {
        let x = Vector3::new(
            (i / (ny * nz)) as f64,
            ((i / nz) % ny) as f64,
            (i % nz) as f64,
        );
        origin + step * x
    });
    let _ = nx;
    let mask = nearest_image(points, center, &cell.supercell, &period)?;
    // the mask of the atom copies
    let atoms = setup.sites().atoms.len();
    let owners = setup.projector_atoms();
    let cells: Vec<Vector3<f64>> = (0..big.cells)
        .map(|t| {
            let v = big.cell_vector(t);
            cell.lattice * Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64)
        })
        .collect();
    let mut column_mask = vec![0.; big.cells * crystal.p];
    for atom in 0..atoms {
        let position = Vector3::from(setup.sites().atoms[atom].center);
        let weights = nearest_image(
            cells.iter().map(|t| position + t),
            center,
            &cell.supercell,
            &period,
        )?;
        for (t, w) in weights.iter().enumerate() {
            for j in projector_columns(&owners, atom) {
                column_mask[t * crystal.p + j] = *w;
            }
        }
    }
    let n = modes.len();
    let p = crystal.p;
    let mut values = (0..n)
        .map(|_| ctx.allocate::<C64>(big.points()))
        .collect::<Result<Vec<_>>>()?;
    let mut coefficients = (0..n)
        .map(|_| ctx.allocate::<C64>(product(&[big.cells, p])?))
        .collect::<Result<Vec<_>>>()?;
    for (i, mode) in modes.iter().enumerate() {
        let source = axis_sources(cell, mode.cell);
        let home = functions.values[mode.at].as_slice();
        let small = cell.total;
        let out = values[i].as_mut_slice();
        for x0 in 0..big.total[0] {
            for x1 in 0..big.total[1] {
                for x2 in 0..big.total[2] {
                    let at = (x0 * big.total[1] + x1) * big.total[2] + x2;
                    if mask[at] == 0. {
                        continue;
                    }
                    let s = (source[0][x0 % small[0]] * small[1] + source[1][x1 % small[1]])
                        * small[2]
                        + source[2][x2 % small[2]];
                    out[at] = home[s] * mask[at];
                }
            }
        }
        let shifted = shifted_coefficients(functions, cell, mode.at, mode.cell);
        for t in 0..big.cells {
            let v = big.cell_vector(t);
            let wrapped = cell.cell_of(v);
            for j in 0..p {
                coefficients[i].as_mut_slice()[t * p + j] =
                    shifted[wrapped * p + j] * column_mask[t * p + j];
            }
        }
    }
    let padded = Functions {
        ids: (0..n).collect(),
        values,
        coefficients,
        projectors: p,
    };
    let identity = (0..n).map(|at| Mode { at, cell: [0; 3] }).collect();
    Ok((big, padded, identity))
}
