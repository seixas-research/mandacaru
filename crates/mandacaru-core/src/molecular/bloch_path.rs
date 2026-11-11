//! The spectral function of a Bloch run along a band path.
//!
//! [`crate::periodic::kpath::select`] names the mesh points on the path; this module resolves
//! `A(k, omega)` there with [`bloch_spectral`], once per distinct source point. Without a zone
//! the sources are the path's own mesh points, so every row is an exact Lehmann spectrum of the
//! state. With a [`Zone`] the sources are the irreducible representatives of their stars and a
//! row is that of its star's representative: the state is assumed symmetric under the point
//! group and time reversal. The assumption is audited, not trusted: the representative of the
//! largest star and one other member of it are both resolved, and the relative `L1` difference
//! of their broadened spectral functions is the [`Audit::residual`]. A variational state of a
//! degenerate shell (the restricted reference of a half-filled square lattice, whose X and Y
//! points are degenerate) breaks the symmetry and the residual shows it.
use super::periodic::{bloch_spectral, BlochSpectral};
use super::{transform::Model, MeanField, MolecularIntegrals};
use crate::periodic::invalid;
use crate::periodic::kpath::{PathPoints, Zone, AUDIT_TOLERANCE};
use crate::quantum::spectral::{OccupationState, Spectral};
use crate::{Context, Result};

/// The symmetry audit of a reduced spectral function.
#[derive(Clone, Debug, PartialEq)]
pub struct Audit {
    pub representative: [f64; 3],
    pub image: [f64; 3],
    /// Members of the audited star.
    pub star_size: usize,
    /// `sum |A(rep) - A(image)| / sum A(rep)` on the energy axis.
    pub residual: f64,
    /// `residual <= AUDIT_TOLERANCE`.
    pub symmetric: bool,
}

/// `A(k, omega)` at the points of a path.
pub struct PathSpectral {
    /// One row per path point (`kpoints` are the points of the path).
    pub spectral: BlochSpectral,
    pub axis: Vec<f64>,
    pub audit: Option<Audit>,
}

fn same(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| ((a[i] - b[i]) - (a[i] - b[i]).round()).abs() < 1e-8)
}

impl BlochSpectral {
    /// The rows `sources[i]` of `self`, one per entry, relabeled with `kpoints`.
    pub fn gather(&self, sources: &[usize], kpoints: Vec<[f64; 3]>) -> Result<BlochSpectral> {
        if sources.len() != kpoints.len() || sources.iter().any(|&s| s >= self.kpoints.len()) {
            return Err(invalid(
                "a gathered spectral function names a missing k-point",
            ));
        }
        let n = self.n_primitive;
        let pick = |rows: &[[crate::quantum::spectral::Poles; 2]]| {
            sources
                .iter()
                .flat_map(|&s| rows[s * n..(s + 1) * n].iter().cloned())
                .collect::<Vec<_>>()
        };
        Ok(BlochSpectral {
            kpoints,
            n_primitive: n,
            parts: Spectral {
                ground_energy: self.parts.ground_energy,
                addition: pick(&self.parts.addition),
                removal: pick(&self.parts.removal),
            },
        })
    }
}

/// `A(k, omega)` of `state` at the points of `path`; see the module documentation. `eta` is the
/// Lorentzian half-width of the audit and of the default axis (`axis` is used when given,
/// otherwise `n_axis` points spanning the poles).
#[allow(clippy::too_many_arguments)]
pub fn bloch_path_spectral(
    ctx: &Context,
    integrals: &MolecularIntegrals,
    mean_field: &MeanField,
    model: &Model,
    state: &OccupationState,
    path: &PathPoints,
    zone: Option<&Zone>,
    eta: f64,
    axis: Option<&[f64]>,
    n_axis: usize,
    krylov: usize,
) -> Result<PathSpectral> {
    let mut sources: Vec<[f64; 3]> = vec![];
    let mut row = Vec::with_capacity(path.kpoints.len());
    for k in &path.kpoints {
        let source = match zone {
            Some(zone) => zone.mesh.points[zone.star_of(*k)?],
            None => *k,
        };
        let index = match sources.iter().position(|s| same(*s, source)) {
            Some(i) => i,
            None => {
                sources.push(source);
                sources.len() - 1
            }
        };
        row.push(index);
    }
    let computed = bloch_spectral(ctx, integrals, mean_field, model, state, &sources, krylov)?;
    let axis = match axis {
        Some(a) => a.to_vec(),
        None => computed.default_axis(eta, n_axis)?,
    };
    let audit = match zone {
        Some(zone) => audit(
            ctx, integrals, mean_field, model, state, zone, &axis, eta, krylov,
        )?,
        None => None,
    };
    Ok(PathSpectral {
        spectral: computed.gather(&row, path.kpoints.clone())?,
        axis,
        audit,
    })
}

#[allow(clippy::too_many_arguments)]
fn audit(
    ctx: &Context,
    integrals: &MolecularIntegrals,
    mean_field: &MeanField,
    model: &Model,
    state: &OccupationState,
    zone: &Zone,
    axis: &[f64],
    eta: f64,
    krylov: usize,
) -> Result<Option<Audit>> {
    let mut sizes = vec![0usize; zone.mesh.points.len()];
    for &star in &zone.mesh.mapping {
        sizes[star] += 1;
    }
    let (star, &size) = sizes
        .iter()
        .enumerate()
        .max_by_key(|&(i, &s)| (s, std::cmp::Reverse(i)))
        .expect("a mesh has a point");
    if size < 2 {
        return Ok(None);
    }
    let representative = zone.mesh.points[star];
    let image = zone
        .mesh
        .full
        .iter()
        .zip(&zone.mesh.mapping)
        .find(|(k, &s)| s == star && !same(**k, representative))
        .map(|(k, _)| *k)
        .ok_or_else(|| invalid("a star of the mesh has one distinct point"))?;
    let pair = bloch_spectral(
        ctx,
        integrals,
        mean_field,
        model,
        state,
        &[representative, image],
        krylov,
    )?;
    let curves = pair.broadened(axis, eta)?;
    let (first, second) = curves.split_at(axis.len());
    let norm: f64 = first.iter().map(|v| v.abs()).sum();
    let residual = if norm > 0. {
        first
            .iter()
            .zip(second)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / norm
    } else {
        0.
    };
    Ok(Some(Audit {
        representative,
        image,
        star_size: size,
        residual,
        symmetric: residual <= AUDIT_TOLERANCE,
    }))
}
