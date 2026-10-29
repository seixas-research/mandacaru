//! Minimum-cost square assignment by augmenting paths and dual potentials.
use crate::{product, Context, Error, Result};
use nalgebra::DMatrix;
pub fn minimum(context: &Context, cost: &DMatrix<f64>) -> Result<Vec<usize>> {
    let n = cost.nrows();
    if n != cost.ncols() || cost.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid(
            "assignment needs a finite square cost matrix".into(),
        ));
    }
    let _memory = context.reserve(product(&[n + 1, 64])?)?;
    let mut u = vec![0.; n + 1];
    let mut v = vec![0.; n + 1];
    let mut matched = vec![0; n + 1];
    let mut previous = vec![0; n + 1];
    for row in 1..=n {
        matched[0] = row;
        let mut column = 0;
        let mut distance = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let row = matched[column];
            let mut delta = f64::INFINITY;
            let mut next = 0;
            for j in 1..=n {
                if !used[j] {
                    let reduced = cost[(row - 1, j - 1)] - u[row] - v[j];
                    if reduced < distance[j] {
                        distance[j] = reduced;
                        previous[j] = column;
                    }
                    if distance[j] < delta {
                        delta = distance[j];
                        next = j;
                    }
                }
            }
            if !delta.is_finite() {
                return Err(Error::Invalid("assignment cost arithmetic overflow".into()));
            }
            for j in 0..=n {
                if used[j] {
                    u[matched[j]] += delta;
                    v[j] -= delta;
                } else {
                    distance[j] -= delta;
                }
            }
            column = next;
            if matched[column] == 0 {
                break;
            }
        }
        loop {
            let next = previous[column];
            matched[column] = matched[next];
            column = next;
            if column == 0 {
                break;
            }
        }
    }
    let mut result = vec![0; n];
    for column in 1..=n {
        result[matched[column] - 1] = column - 1;
    }
    Ok(result)
}
