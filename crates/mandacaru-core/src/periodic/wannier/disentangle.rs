//! Frozen-window subspace selection by minimization of the invariant spread.
//! Optional band representations constrain the projectors to a space group.
use super::{invalid, Matrix, Mesh};
use crate::{
    math::{
        linalg::eigen,
        storage::{MatrixSource, StoredMatrix},
    },
    product,
    quantum::C64,
    Buffer, Context, Reservation, Result,
};

pub struct Windows {
    pub inside: Buffer<u8>,
    pub frozen: Buffer<u8>,
    pub functions: usize,
}
pub struct Symmetry {
    pub operations: usize,
    pub mapping: Buffer<usize>,
    pub rotations: Vec<StoredMatrix>,
}
pub struct Options {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub mixing: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            tolerance: 1e-10,
            mixing: 0.5,
        }
    }
}
pub struct Subspace {
    pub gauge: Vec<Matrix>,
    pub history: Vec<f64>,
    pub converged: bool,
    _memory: Reservation,
}
fn symmetric(
    ctx: &Context,
    mesh: &Mesh,
    values: &[Matrix],
    group: Option<&Symmetry>,
) -> Result<Vec<Matrix>> {
    let Some(group) = group else {
        return Ok(values.to_vec());
    };
    let mut out = vec![Matrix::zeros(mesh.bands, mesh.bands); mesh.points];
    for operation in 0..group.operations {
        for (k, value) in values.iter().enumerate() {
            let at = operation * mesh.points + k;
            let d = group.rotations[at].matrix(ctx)?;
            out[group.mapping.as_slice()[at]] +=
                &*d * value * d.adjoint() * C64::new(1. / group.operations as f64, 0.);
        }
    }
    Ok(out)
}
fn validate_symmetry(
    ctx: &Context,
    mesh: &Mesh,
    group: &Symmetry,
    windows: &Windows,
) -> Result<()> {
    if group.operations == 0
        || group.mapping.as_slice().len() != product(&[group.operations, mesh.points])?
        || group.rotations.len() != group.mapping.as_slice().len()
    {
        return Err(invalid("invalid Wannier symmetry dimensions"));
    }
    let mut seen = ctx.allocate::<u8>(mesh.points)?;
    for operation in 0..group.operations {
        seen.as_mut_slice().fill(0);
        for k in 0..mesh.points {
            let at = operation * mesh.points + k;
            let dest = group.mapping.as_slice()[at];
            if dest >= mesh.points || seen.as_slice()[dest] != 0 {
                return Err(invalid("Wannier symmetry must permute the k-point mesh"));
            }
            seen.as_mut_slice()[dest] = 1;
            let d = group.rotations[at].matrix(ctx)?;
            if d.shape() != (mesh.bands, mesh.bands)
                || d.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
                || (d.adjoint() * &*d - Matrix::identity(mesh.bands, mesh.bands)).norm() > 1e-4
            {
                return Err(invalid("Wannier band representation is not unitary"));
            }
            for i in 0..mesh.bands {
                for j in 0..mesh.bands {
                    if d[(i, j)].norm() > 1e-8
                        && [windows.inside.as_slice(), windows.frozen.as_slice()]
                            .iter()
                            .any(|mask| mask[dest * mesh.bands + i] != mask[k * mesh.bands + j])
                    {
                        return Err(invalid(
                            "Wannier symmetry mixes states across an outer or frozen window",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}
fn leading(matrix: &Matrix, functions: usize) -> Result<Matrix> {
    if functions == 0 {
        return Ok(Matrix::zeros(matrix.nrows(), 0));
    }
    let (_, vectors) = eigen((matrix + matrix.adjoint()) * C64::new(0.5, 0.))?;
    Ok(vectors
        .columns(matrix.nrows() - functions, functions)
        .into_owned())
}
pub fn select(
    ctx: &Context,
    mesh: &Mesh,
    trials: &[Matrix],
    windows: &Windows,
    group: Option<&Symmetry>,
    options: &Options,
) -> Result<Subspace> {
    mesh.validate(ctx)?;
    let functions = windows.functions;
    if functions == 0
        || functions > mesh.bands
        || windows.inside.as_slice().len() != product(&[mesh.points, mesh.bands])?
        || windows.frozen.as_slice().len() != windows.inside.as_slice().len()
        || windows
            .inside
            .as_slice()
            .iter()
            .chain(windows.frozen.as_slice())
            .any(|v| *v > 1)
        || trials.len() != mesh.points
        || trials.iter().any(|a| {
            a.shape() != (mesh.bands, functions)
                || a.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        })
        || options.max_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || !options.mixing.is_finite()
        || options.mixing <= 0.
        || options.mixing > 1.
    {
        return Err(invalid(
            "invalid Wannier windows, trial projections or disentanglement options",
        ));
    }
    let retained = product(&[mesh.points, mesh.bands, functions, 16])?
        .checked_add(product(&[
            options
                .max_iterations
                .checked_add(1)
                .ok_or(crate::Error::Overflow)?,
            8,
        ])?)
        .ok_or(crate::Error::Overflow)?;
    let mut memory = ctx.reserve(retained)?;
    let _work = ctx.reserve(
        product(&[mesh.points, mesh.bands, mesh.bands, 16, 7])?
            .checked_add(product(&[mesh.bands, mesh.bands, 16, 12])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    if let Some(group) = group {
        validate_symmetry(ctx, mesh, group, windows)?;
    }
    ctx.run(|| {
        let mut free = vec![];
        let mut fixed = vec![];
        let mut gauge = vec![Matrix::zeros(mesh.bands, functions); mesh.points];
        for (k, g) in gauge.iter_mut().enumerate() {
            let mut f = vec![];
            let mut r = vec![];
            for i in 0..mesh.bands {
                let inside = windows.inside.as_slice()[k * mesh.bands + i] != 0;
                let frozen = windows.frozen.as_slice()[k * mesh.bands + i] != 0;
                if frozen && !inside {
                    return Err(invalid(
                        "frozen Wannier states must lie in the outer window",
                    ));
                }
                if frozen {
                    f.push(i);
                } else if inside {
                    r.push(i);
                }
            }
            if f.len() > functions || f.len() + r.len() < functions {
                return Err(invalid(
                    "Wannier windows cannot contain the requested subspace dimension",
                ));
            }
            for (j, &i) in f.iter().enumerate() {
                g[(i, j)] = C64::new(1., 0.);
            }
            free.push(r);
            fixed.push(f);
        }
        let update = |matrices: &[Matrix], gauge: &mut [Matrix]| -> Result<()> {
            let matrices = symmetric(ctx, mesh, matrices, group)?;
            for k in 0..mesh.points {
                let rest = functions - fixed[k].len();
                if rest == 0 {
                    continue;
                }
                let block = Matrix::from_fn(free[k].len(), free[k].len(), |i, j| {
                    matrices[k][(free[k][i], free[k][j])]
                });
                let chosen = leading(&block, rest)?;
                for (i, &band) in free[k].iter().enumerate() {
                    for j in 0..rest {
                        gauge[k][(band, fixed[k].len() + j)] = chosen[(i, j)];
                    }
                }
            }
            Ok(())
        };
        // The trial projector fixes the initial subspace; its singular-vector
        // phases cannot influence the invariant objective.
        let projectors: Vec<_> = trials.iter().map(|a| a * a.adjoint()).collect();
        update(&projectors, &mut gauge)?;
        let mut history: Vec<f64> = vec![];
        let mut mixed: Option<Vec<Matrix>> = None;
        let mut converged = false;
        for iteration in 0..=options.max_iterations {
            let mut z = vec![Matrix::zeros(mesh.bands, mesh.bands); mesh.points];
            let mut invariant = 0.;
            for k in 0..mesh.points {
                for b in 0..mesh.vectors.as_slice().len() {
                    let at = k * mesh.vectors.as_slice().len() + b;
                    let link = mesh.links[at].matrix(ctx)?;
                    let x = &*link * &gauge[mesh.neighbors.as_slice()[at]];
                    let projected = gauge[k].adjoint() * &x;
                    let weight = mesh.weights.as_slice()[b];
                    invariant +=
                        weight * (functions as f64 - projected.norm_squared()) / mesh.points as f64;
                    z[k] += &x * x.adjoint() * C64::new(weight, 0.);
                }
            }
            let done = history
                .last()
                .is_some_and(|previous| (previous - invariant).abs() < options.tolerance);
            history.push(invariant);
            if done || iteration == options.max_iterations {
                converged = done;
                break;
            }
            if let Some(previous) = &mixed {
                for (new, old) in z.iter_mut().zip(previous) {
                    *new = &*new * C64::new(options.mixing, 0.)
                        + old * C64::new(1. - options.mixing, 0.);
                }
            }
            update(&z, &mut gauge)?;
            mixed = Some(z);
        }
        memory.shrink_to(
            product(&[mesh.points, mesh.bands, functions, 16])?
                .checked_add(product(&[history.len(), 8])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        Ok(Subspace {
            gauge,
            history,
            converged,
            _memory: memory,
        })
    })
}
