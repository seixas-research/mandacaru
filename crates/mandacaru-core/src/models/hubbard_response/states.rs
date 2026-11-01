//! The bare response as a sum over states, from the eigenpairs of the ground state.
use crate::models::invalid;
use crate::Result;
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// Two levels closer than this (the energy unit of the caller) are one degenerate level:
/// the quotient `(f_n - f_m) / (e_n - e_m)` is replaced by the slope `df/de`.
pub const DEGENERATE: f64 = 1e-10;
/// A sum of slopes below this is a gapped system at tiny width: the Fermi-level term has
/// nothing to move and is skipped instead of divided.
pub const NO_SLOPE: f64 = 1e-300;

/// The states of one k-point (or the one set of a molecule).
#[derive(Clone, Copy)]
pub struct StateBlock<'a> {
    /// k-point weight.
    pub weight: f64,
    pub levels: &'a [f64],
    pub occupations: &'a [f64],
    /// `df/de` at the levels.
    pub slopes: &'a [f64],
    /// Per shell, `<n|P_I|m>` row-major over the bands.
    pub elements: &'a [&'a [C64]],
}

/// `chi0 = pairs + fermi`. The second term is the shift of the Fermi level that keeps the
/// electron number fixed; without it the result is the response at fixed potential.
pub struct StateSum {
    pub pairs: DMatrix<f64>,
    pub fermi: DMatrix<f64>,
}

impl StateSum {
    pub fn total(&self) -> DMatrix<f64> {
        &self.pairs + &self.fermi
    }
}

/// ```text
/// chi0_IJ = g sum_k w_k sum_nm c_nm Re[M_I(n,m) M_J(m,n)]  -  s_I s_J / S
/// c_nm = (f_n - f_m) / (e_n - e_m),  c_nn = f'_n
/// s_I = g sum_k w_k sum_n f'_n Re M_I(n,n),  S = g sum_k w_k sum_n f'_n
/// ```
/// `degeneracy` is the spin degeneracy `g` of the occupations (2 when restricted).
pub fn sum_over_states(blocks: &[StateBlock], degeneracy: f64) -> Result<StateSum> {
    if !degeneracy.is_finite() || degeneracy <= 0. {
        return Err(invalid("the spin degeneracy must be positive"));
    }
    let shells = blocks.first().map(|b| b.elements.len()).unwrap_or(0);
    if shells == 0 {
        return Err(invalid(
            "a sum over states needs at least one block and one shell",
        ));
    }
    let mut pairs = DMatrix::zeros(shells, shells);
    let mut s = vec![0.; shells];
    let mut total_slope = 0.;
    for b in blocks {
        let bands = b.levels.len();
        let finite = b
            .levels
            .iter()
            .chain(b.occupations)
            .chain(b.slopes)
            .all(|v| v.is_finite());
        if !b.weight.is_finite() || b.weight < 0. || !finite {
            return Err(invalid(
                "block weights must be finite and not negative, levels finite",
            ));
        }
        if bands == 0
            || b.occupations.len() != bands
            || b.slopes.len() != bands
            || b.elements.len() != shells
            || b.elements.iter().any(|m| m.len() != bands * bands)
        {
            return Err(invalid(format!(
                "inconsistent block: {bands} levels, {} occupations, {} slopes, {} shells of {} matrix elements",
                b.occupations.len(),
                b.slopes.len(),
                b.elements.len(),
                b.elements.first().map_or(0, |m| m.len())
            )));
        }
        let scale = b.weight * degeneracy;
        for n in 0..bands {
            total_slope += scale * b.slopes[n];
            for i in 0..shells {
                s[i] += scale * b.slopes[n] * b.elements[i][n * bands + n].re;
            }
            for m in 0..bands {
                let gap = b.levels[n] - b.levels[m];
                let c = if gap.abs() < DEGENERATE {
                    0.5 * (b.slopes[n] + b.slopes[m])
                } else {
                    (b.occupations[n] - b.occupations[m]) / gap
                };
                for i in 0..shells {
                    let a = b.elements[i][n * bands + m];
                    for j in 0..shells {
                        pairs[(i, j)] += scale * c * (a * b.elements[j][m * bands + n]).re;
                    }
                }
            }
        }
    }
    let fermi = if total_slope.abs() < NO_SLOPE {
        DMatrix::zeros(shells, shells)
    } else {
        DMatrix::from_fn(shells, shells, |i, j| -s[i] * s[j] / total_slope)
    };
    Ok(StateSum { pairs, fermi })
}
