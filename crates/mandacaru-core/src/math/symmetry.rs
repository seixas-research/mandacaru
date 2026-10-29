//! Orthogonal symmetry operations of finite labelled point arrangements.
use crate::{Error, Result};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
pub type Operation = Matrix3<f64>;
fn frame(a: Vector3<f64>, b: Vector3<f64>, handed: f64) -> Matrix3<f64> {
    let u = a.normalize();
    let v = (b - u * u.dot(&b)).normalize();
    Matrix3::from_columns(&[u, v, u.cross(&v) * handed])
}
fn maps(matrix: &Matrix3<f64>, points: &[Vector3<f64>], labels: &[u8], tolerance: f64) -> bool {
    let mut used = vec![false; points.len()];
    for (i, p) in points.iter().enumerate() {
        let image = matrix * p;
        let target = points
            .iter()
            .enumerate()
            .filter(|(j, _)| !used[*j] && labels[*j] == labels[i])
            .min_by(|(_, a), (_, b)| {
                (image - *a)
                    .norm_squared()
                    .total_cmp(&(image - *b).norm_squared())
            });
        if let Some((j, q)) = target {
            if (image - q).norm() > tolerance {
                return false;
            }
            used[j] = true;
        } else {
            return false;
        }
    }
    true
}
fn push(operations: &mut Vec<Operation>, matrix: Operation) {
    if !operations.iter().any(|r| (r - matrix).amax() < 1e-3) {
        operations.push(matrix);
    }
}
fn finite(points: &[Vector3<f64>], labels: &[u8], tolerance: f64) -> Result<Vec<Operation>> {
    let (a, outer) = points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_squared().total_cmp(&b.norm_squared()))
        .unwrap();
    let b = points
        .iter()
        .enumerate()
        .filter(|(_, p)| outer.cross(p).norm() > 0.1 * outer.norm() * p.norm())
        .max_by(|(_, a), (_, b)| {
            outer
                .cross(a)
                .norm_squared()
                .total_cmp(&outer.cross(b).norm_squared())
        })
        .map(|(i, _)| i)
        .ok_or_else(|| {
            Error::Invalid("nonlinear symmetry requires two independent point vectors".into())
        })?;
    let basis = frame(points[a], points[b], 1.);
    let mut operations = vec![];
    for i in 0..points.len() {
        for j in 0..points.len() {
            if labels[i] != labels[a]
                || labels[j] != labels[b]
                || (points[i].norm() - points[a].norm()).abs() > tolerance
                || (points[j].norm() - points[b].norm()).abs() > tolerance
                || (points[i].dot(&points[j]) - points[a].dot(&points[b])).abs()
                    > tolerance * (points[a].norm() + points[b].norm())
                || points[i].cross(&points[j]).norm() < 1e-10
            {
                continue;
            }
            for handed in [1., -1.] {
                let matrix = frame(points[i], points[j], handed) * basis.transpose();
                if maps(&matrix, points, labels, tolerance) {
                    push(&mut operations, matrix);
                }
            }
        }
    }
    Ok(operations)
}
pub fn operations(
    points: &[[f64; 3]],
    labels: &[u8],
    tolerance: f64,
) -> Result<(Vector3<f64>, Vec<Operation>)> {
    if points.is_empty()
        || points.len() != labels.len()
        || !tolerance.is_finite()
        || tolerance <= 0.
        || points.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(Error::Invalid(
            "symmetry needs finite labelled points and a positive tolerance".into(),
        ));
    }
    let total = labels.iter().map(|&z| z as f64).sum::<f64>();
    if total <= 0. {
        return Err(Error::Invalid(
            "symmetry labels must have a positive total weight".into(),
        ));
    }
    let center = points
        .iter()
        .zip(labels)
        .fold(Vector3::zeros(), |sum, (p, &z)| {
            sum + Vector3::from(*p) * (z as f64 / total)
        });
    let points: Vec<_> = points.iter().map(|p| Vector3::from(*p) - center).collect();
    let outer = points
        .iter()
        .max_by(|a, b| a.norm_squared().total_cmp(&b.norm_squared()))
        .unwrap();
    let mut result = if outer.norm() < tolerance {
        let phi = 0.5 * (1. + 5_f64.sqrt());
        let mut vertices = vec![];
        for a in [-1., 1.] {
            for b in [-1., 1.] {
                vertices.extend([
                    Vector3::new(0., a, b * phi),
                    Vector3::new(a, b * phi, 0.),
                    Vector3::new(b * phi, 0., a),
                ]);
            }
        }
        finite(&vertices, &[1; 12], 1e-7)?
    } else if points
        .iter()
        .all(|p| outer.cross(p).norm() < tolerance * outer.norm())
    {
        let axis = outer.normalize();
        let coordinate = (0..3)
            .min_by(|&a, &b| axis[a].abs().total_cmp(&axis[b].abs()))
            .unwrap();
        let mut trial = Vector3::zeros();
        trial[coordinate] = 1.;
        let normal = axis.cross(&trial).normalize();
        let mirror = Matrix3::identity() - 2. * normal * normal.transpose();
        let inversion = maps(&(-Matrix3::identity()), &points, labels, tolerance);
        let cross = Matrix3::new(
            0., -axis[2], axis[1], axis[2], 0., -axis[0], -axis[1], axis[0], 0.,
        );
        let mut result = vec![];
        for k in 0..8 {
            let angle = 2. * PI * k as f64 / 8.;
            let rotation =
                Matrix3::identity() + cross * angle.sin() + cross * cross * (1. - angle.cos());
            for operation in [rotation, rotation * mirror] {
                push(&mut result, operation);
                if inversion {
                    push(&mut result, -operation);
                }
            }
        }
        result
    } else {
        finite(&points, labels, tolerance)?
    };
    result.sort_by(|a, b| {
        (a - Matrix3::identity())
            .norm_squared()
            .total_cmp(&(b - Matrix3::identity()).norm_squared())
    });
    if result.is_empty() {
        return Err(Error::Kernel(
            "point-group construction found no identity".into(),
        ));
    }
    Ok((center, result))
}
