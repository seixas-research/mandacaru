//! Radial functions of the pseudo-atomic orbitals that need no dataset: the
//! quasi-Gaussian polarization shell and the polarization shape split off the
//! outermost orbital, with the radial quadrature they share.
use super::invalid;
use crate::Result;

/// Trapezoid `int y dx` on the increasing nodes `x`, in numpy's order.
pub fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(x, y)| (x[1] - x[0]) * (y[1] + y[0]) / 2.)
        .sum()
}

/// `int R^2 r^2 dr` of a radial table.
pub fn radial_norm(r: &[f64], values: &[f64]) -> f64 {
    let density: Vec<f64> = r.iter().zip(values).map(|(&r, &v)| v * v * r * r).collect();
    trapezoid(r, &density)
}

/// The quasi-Gaussian `exp(-alpha r^2) - (a - b r^2)` inside `r_cut`, zero
/// beyond, with `a = (1 + alpha r_c^2) exp(-alpha r_c^2)` and
/// `b = alpha exp(-alpha r_c^2)`: value and slope vanish at the cutoff.
pub fn quasi_gaussian(r: &[f64], alpha: f64, r_cut: f64) -> Vec<f64> {
    let decay = (-alpha * r_cut.powi(2)).exp();
    let a = (1. + alpha * r_cut.powi(2)) * decay;
    let b = alpha * decay;
    r.iter()
        .map(|&x| {
            if x < r_cut {
                let r2 = x * x;
                (-alpha * r2).exp() - (a - b * r2)
            } else {
                0.
            }
        })
        .collect()
}

fn unit_norm(table: &[f64], values: Vec<f64>) -> Result<Vec<f64>> {
    let norm = radial_norm(table, &values).sqrt();
    if !(norm.is_finite() && norm > 0.) {
        return Err(invalid(
            "a radial function with zero norm cannot be normalized",
        ));
    }
    Ok(values.into_iter().map(|v| v / norm).collect())
}

fn checked_table(table: &[f64], what: &str) -> Result<()> {
    if table.len() < 4 || table.windows(2).any(|w| w[1] <= w[0]) || table[0] < 0. {
        return Err(invalid(format!("{what} needs an increasing radial table")));
    }
    Ok(())
}

/// The polarization shell `R = r^l [exp(-r^2/r_char^2) - (a - b r^2)]` cut at
/// `r_cut`, normalized, on the dataset grid `table`.
pub fn polarization_radial(table: &[f64], l: usize, r_cut: f64, r_char: f64) -> Result<Vec<f64>> {
    checked_table(table, "a polarization shell")?;
    if !(r_cut.is_finite() && r_cut > 0. && r_char.is_finite() && r_char > 0.) {
        return Err(invalid("the polarization shell needs positive radii"));
    }
    let gaussian = quasi_gaussian(table, 1. / r_char.powi(2), r_cut);
    unit_norm(
        table,
        table
            .iter()
            .zip(&gaussian)
            .map(|(&r, g)| r.powi(l as i32) * g)
            .collect(),
    )
}

/// The polarization shape `r^power R_outer(r)`, normalized: it keeps the
/// channel's range and vanishes as `r^l` at the nucleus.
pub fn power_shape(table: &[f64], outermost: &[f64], power: usize) -> Result<Vec<f64>> {
    checked_table(table, "a polarization shape")?;
    if outermost.len() != table.len() {
        return Err(invalid(
            "the outermost orbital must be tabulated on the grid",
        ));
    }
    unit_norm(
        table,
        table
            .iter()
            .zip(outermost)
            .map(|(&r, v)| r.powi(power as i32) * v)
            .collect(),
    )
}
