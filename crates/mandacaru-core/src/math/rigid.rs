//! Internal Cartesian displacement directions orthogonal to rigid motions.
use crate::{product, Context, Error, Reservation, Result};
use nalgebra::DMatrix;
pub fn remove_translation(context: &Context, values: &[f64]) -> Result<crate::Buffer<f64>> {
    if values.is_empty() || values.len() % 3 != 0 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid(
            "translation projection needs a finite nonempty Cartesian array".into(),
        ));
    }
    let atoms = values.len() / 3;
    let mean: [f64; 3] =
        std::array::from_fn(|i| values.chunks_exact(3).map(|r| r[i]).sum::<f64>() / atoms as f64);
    let mut result = context.copy_slice(values)?;
    for row in result.as_mut_slice().chunks_exact_mut(3) {
        for axis in 0..3 {
            row[axis] -= mean[axis];
        }
    }
    Ok(result)
}
pub struct InternalDirections {
    pub matrix: DMatrix<f64>,
    _memory: Reservation,
}
fn project(vector: &mut [f64], basis: &[Vec<f64>]) {
    // Twice-reorthogonalized modified Gram–Schmidt is stable for linear
    // molecules, where one nominal rotation is exactly redundant.
    for _ in 0..2 {
        for q in basis {
            let dot = vector.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
            for (a, b) in vector.iter_mut().zip(q) {
                *a -= dot * b;
            }
        }
    }
}
pub fn internal(context: &Context, positions: &[[f64; 3]]) -> Result<InternalDirections> {
    if positions.is_empty() || positions.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::Invalid(
            "internal directions need finite nonempty positions".into(),
        ));
    }
    let size = product(&[positions.len(), 3])?;
    let memory = context.reserve(product(&[size, size + 16, 16])?)?;
    let center: [f64; 3] = std::array::from_fn(|a| {
        positions.iter().map(|r| r[a]).sum::<f64>() / positions.len() as f64
    });
    let mut candidates = Vec::new();
    for axis in 0..3 {
        candidates.push(
            (0..size)
                .map(|i| f64::from(i % 3 == axis))
                .collect::<Vec<_>>(),
        );
    }
    for axis in 0..3 {
        let mut rotation = vec![0.; size];
        for (atom, r) in positions.iter().enumerate() {
            let first = (axis + 1) % 3;
            let second = (axis + 2) % 3;
            rotation[3 * atom + first] = -(r[second] - center[second]);
            rotation[3 * atom + second] = r[first] - center[first];
        }
        // Unit scaling makes the rank decision independent of length units.
        let norm = rotation.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0. {
            for v in &mut rotation {
                *v /= norm;
            }
        }
        candidates.push(rotation);
    }
    let mut rigid = Vec::new();
    while !candidates.is_empty() {
        for vector in &mut candidates {
            project(vector, &rigid);
        }
        let (at, norm) = candidates
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.iter().map(|x| x * x).sum::<f64>().sqrt()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if norm <= 1e-11 {
            break;
        }
        let mut vector = candidates.swap_remove(at);
        for value in &mut vector {
            *value /= norm;
        }
        rigid.push(vector);
    }
    let mut complement = Vec::new();
    for axis in 0..size {
        let mut vector = vec![0.; size];
        vector[axis] = 1.;
        project(&mut vector, &rigid);
        project(&mut vector, &complement);
        let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-11 {
            for value in &mut vector {
                *value /= norm;
            }
            complement.push(vector);
        }
    }
    if complement.len() + rigid.len() != size {
        return Err(Error::Kernel(
            "rigid-motion complement lost numerical rank".into(),
        ));
    }
    let matrix = DMatrix::from_fn(size, complement.len(), |r, c| complement[c][r]);
    Ok(InternalDirections {
        matrix,
        _memory: memory,
    })
}
