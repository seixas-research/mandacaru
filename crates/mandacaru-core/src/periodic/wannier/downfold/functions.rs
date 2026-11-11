//! The Wannier functions of one spin channel on the Born-von Karman supercell.
//!
//! The function `n` of the home cell is, on the supercell grid and on the projectors,
//!
//! ```text
//! w_n(r + T)            = (1/N) sum_k exp(i k.T) sum_m U_mn(k) psi_mk(r)
//! <p_{iT}|w_n>          = (1/N) sum_k exp(i k.T) sum_m U_mn(k) <p_i|psi_mk>
//! ```
//!
//! with `psi_mk` the pseudo Bloch states of the frame and `U(k)` the localized gauge. The function
//! of the cell `R` is the home one moved by `R`; the supercell grid wraps (Born-von Karman).
use super::coulomb::Cell;
use crate::{
    periodic::{
        invalid,
        wannier::{model::Channel, Matrix},
        Crystal,
    },
    product,
    quantum::C64,
    Buffer, Result,
};
use std::f64::consts::PI;

/// The home-cell functions `ids` of a channel: their supercell values and the coefficients on the
/// projectors of every cell.
pub struct Functions {
    pub ids: Vec<usize>,
    /// `values[i]`: the supercell grid values (C order) of function `ids[i]`.
    pub values: Vec<Buffer<C64>>,
    /// `coefficients[i][t * p + j]`: `<p_{j, T_t}|w_{ids[i]}>`.
    pub coefficients: Vec<Buffer<C64>>,
    pub projectors: usize,
}

/// The home source of every supercell index along each axis for the function moved by `shift`
/// cells: `source[a][x] = (x - shift_a * shape_a) mod total_a`.
pub fn axis_sources(cell: &Cell, shift: [i64; 3]) -> [Vec<usize>; 3] {
    std::array::from_fn(|a| {
        (0..cell.total[a])
            .map(|x| {
                (x as i64 - shift[a] * cell.shape[a] as i64).rem_euclid(cell.total[a] as i64)
                    as usize
            })
            .collect()
    })
}

/// The projector coefficients of the function `ids[at]` moved by `shift`, `[t * p + j]`.
pub fn shifted_coefficients(
    functions: &Functions,
    cell: &Cell,
    at: usize,
    shift: [i64; 3],
) -> Vec<C64> {
    let p = functions.projectors;
    let mut out = vec![C64::default(); cell.cells * p];
    for t in 0..cell.cells {
        let v = cell.cell_vector(t);
        let source = cell.cell_of(std::array::from_fn(|a| v[a] - shift[a]));
        out[t * p..(t + 1) * p]
            .copy_from_slice(&functions.coefficients[at].as_slice()[source * p..(source + 1) * p]);
    }
    out
}

/// The supercell functions `ids` of `channel`. Every frame is read once; the supercell values and
/// the projector coefficients are reserved from the crystal's context before they are filled.
pub fn supercell_functions(
    crystal: &Crystal,
    channel: &Channel,
    cell: &Cell,
    ids: &[usize],
) -> Result<Functions> {
    let ctx = &crystal.context;
    let functions = channel.localized.gauge.first().map_or(0, |g| g.ncols());
    if ids.is_empty() || ids.iter().any(|n| *n >= functions) {
        return Err(invalid(
            "a fragment names a Wannier function that does not exist",
        ));
    }
    if crystal.spinor {
        return Err(crate::Error::Unsupported(
            "Wannier downfolding of a spinor crystal is not available: the two-component \
             functions have no spin-resolved fragment model"
                .into(),
        ));
    }
    if channel.frames.len() != cell.cells || channel.localized.gauge.len() != cell.cells {
        return Err(invalid("the channel does not fill the supercell's mesh"));
    }
    let n = crystal.grid.index.points();
    let p = crystal.p;
    let bands = channel.localized.gauge[0].nrows();
    let mut values = ids
        .iter()
        .map(|_| ctx.allocate::<C64>(product(&[cell.cells, n])?))
        .collect::<Result<Vec<_>>>()?;
    let mut coefficients = ids
        .iter()
        .map(|_| ctx.allocate::<C64>(product(&[cell.cells, p])?))
        .collect::<Result<Vec<_>>>()?;
    let _work = ctx.reserve(product(&[ids.len() + 2, n, 16])?)?;
    let local: Vec<[usize; 3]> = (0..n)
        .map(|j| {
            let [_, ny, nz] = cell.shape;
            [j / (ny * nz), (j / nz) % ny, j % nz]
        })
        .collect();
    let cells: Vec<[i64; 3]> = (0..cell.cells).map(|t| cell.cell_vector(t)).collect();
    let mut phi = vec![vec![C64::default(); n]; ids.len()];
    for (k, frame) in channel.frames.iter().enumerate() {
        let gauge: &Matrix = &channel.localized.gauge[k];
        if frame.components != 1 || gauge.nrows() != bands {
            return Err(invalid(
                "the frames of a channel must share their band space",
            ));
        }
        for row in phi.iter_mut() {
            row.fill(C64::default());
        }
        for m in 0..bands {
            let sample = frame.samples.read_range(ctx, m * n, n)?;
            for (i, id) in ids.iter().enumerate() {
                let u = gauge[(m, *id)];
                if u.norm_sqr() == 0. {
                    continue;
                }
                for (to, from) in phi[i].iter_mut().zip(sample.iter()) {
                    *to += u * from;
                }
            }
        }
        // <p_j|w> of this k: conj(<psi_m|p_j>) U_mn
        for (i, id) in ids.iter().enumerate() {
            let mut a = vec![C64::default(); p];
            for m in 0..bands {
                let u = gauge[(m, *id)];
                for (j, slot) in a.iter_mut().enumerate() {
                    *slot += u * frame.projections[(m, j)].conj();
                }
            }
            for (t, vector) in cells.iter().enumerate() {
                let angle = 2.
                    * PI
                    * (0..3)
                        .map(|a| frame.fractional[a] * vector[a] as f64)
                        .sum::<f64>();
                let phase = C64::from_polar(1. / cell.cells as f64, angle);
                for (j, value) in a.iter().enumerate() {
                    coefficients[i].as_mut_slice()[t * p + j] += phase * value;
                }
                let target = values[i].as_mut_slice();
                let t3 = [vector[0] as usize, vector[1] as usize, vector[2] as usize];
                for (j, value) in phi[i].iter().enumerate() {
                    target[cell.point_of(t3, local[j])] += phase * value;
                }
            }
        }
    }
    Ok(Functions {
        ids: ids.to_vec(),
        values,
        coefficients,
        projectors: p,
    })
}
