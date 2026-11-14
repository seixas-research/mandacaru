//! The image-resolved projections of the Bloch sums on a projector list: everything the crystal
//! force and the crystal stress read of `C_k = <chi_k | p>` and of its derivatives, for EVERY
//! k-point, from one set of quadrature sweeps.
//!
//! A projection of a Bloch sum unfolds to a sum over the lattice images `T` of the function,
//!
//! ```text
//! C_{mu p}(k) = sum_T e^{-i k.T} B_{mu T p},    B_{mu T p} = sum_x w_x conj(phi_mu(x - c_mu - T)) p_p(x)
//! ```
//!
//! and the blocks `B` (and the same blocks with the Cartesian derivative `d_a phi` in place of `phi`)
//! do not depend on `k`: the k-point only enters through the phase of the image. The table holds the
//! blocks of the value and of the three derivatives, once, and forms from them
//!
//! * `C_k` ([`ProjectionTable::value`]),
//! * the row-center derivative `G_a(k) = -sum_T e^{-ik.T} B^a_T` ([`ProjectionTable::gradient`]),
//! * the strain derivative with the spheres rigid about their atoms
//!   ([`ProjectionTable::strain`]). A function evaluated at `x_p = R_A + s_p` moves, under
//!   `x -> (1 + eps) x` of the cell, by `d chi_mu(x_p)/d eps_ab = sum_T e^{ik.T} (d_a phi)(y) (y_b - s_b)`
//!   with `y - s = -Delta`, `Delta = c_mu + T - R_A`, a constant of the image: the nine tangents
//!   of every k-point are the three derivative blocks weighted by `-Delta_b`, so no strain sweep of
//!   the sphere is made at all.
//!
//! The quadrature work therefore does not grow with the number of k-points, and a force and a
//! stress that share the table share it entirely.
use super::{
    basis_gradient::ShortRange,
    images::{Images, Region},
    pulay::Matrix,
    shells::Shells,
    sphere_table::{SphereCache, SphereTable},
    spheres::{evaluate, Projector, Sphere, CHUNK_POINTS},
};
use crate::{
    kernels::{self, ShellImages},
    periodic::invalid,
    product, Buffer, Context, Error, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

/// A strain direction: `E_ab` multiplies `d / d eps_ab`.
pub type Direction = [[f64; 3]; 3];

/// The blocks of one shell of functions that reaches a sphere: `blocks[d]` is
/// `(image, component, member)` row-major for `d = 0` (the value) and `d = 1, 2, 3` (the
/// derivative along `x`, `y`, `z`).
struct ShellBlocks {
    center: [f64; 3],
    /// The position of each component among the functions that reach the sphere.
    local_rows: Vec<usize>,
    images: Vec<[f64; 3]>,
    components: usize,
    blocks: [Option<Buffer<Complex64>>; 4],
}

struct Group {
    center: [f64; 3],
    /// Global function index of each reached row, global projector index of each member.
    rows: Vec<usize>,
    members: Vec<usize>,
    shells: Vec<ShellBlocks>,
}

/// The image-resolved projection blocks of a projector list, for every k-point.
pub struct ProjectionTable {
    functions: usize,
    count: usize,
    groups: Vec<Group>,
}

impl ProjectionTable {
    /// Forms the blocks of the value (`derivatives` contains `None`) and of the derivative along
    /// each listed axis. Projectors of one atom with the same cutoff share one sphere.
    pub fn new(
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        projectors: &[Projector],
        derivatives: &[Option<usize>],
    ) -> Result<Self> {
        let (functions, count) = (shells.functions(), projectors.len());
        let mut spheres: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
        for (p, projector) in projectors.iter().enumerate() {
            if !(projector.r_cut.is_finite() && projector.r_cut > 0.) {
                return Err(Error::Invalid(
                    "a projector needs a positive cutoff radius".into(),
                ));
            }
            spheres
                .entry((projector.atom, (projector.r_cut * 1e12).round() as i64))
                .or_default()
                .push(p);
        }
        let mut groups = Vec::new();
        for members in spheres.values() {
            let first = &projectors[members[0]];
            let region = Region {
                center: first.center,
                radius: first.r_cut,
            };
            let reached = shells.reaching(ctx, lattice, &region)?;
            if reached.rows.is_empty() {
                continue;
            }
            let sphere = Sphere::projector(ctx, first.center, first.r_cut)?;
            let mut group = Group {
                center: first.center,
                rows: reached.rows.clone(),
                members: members.clone(),
                shells: Vec::new(),
            };
            let mut near_shells = Vec::new();
            for shell in reached.shells.shells() {
                let images =
                    Images::around(ctx, lattice, &region, shell.center, shell.radial.support())?;
                if images.is_empty() {
                    continue;
                }
                let size = product(&[images.len(), shell.ms.len(), members.len()])?;
                let mut blocks: [Option<Buffer<Complex64>>; 4] = [None, None, None, None];
                for derivative in derivatives {
                    let slot = derivative.map_or(0, |a| a + 1);
                    if slot > 3 {
                        return Err(invalid("a projection derivative axis is 0, 1 or 2"));
                    }
                    blocks[slot] = Some(ctx.allocate::<Complex64>(size)?);
                }
                group.shells.push(ShellBlocks {
                    center: shell.center,
                    local_rows: shell.rows.clone(),
                    images: images.vectors().to_vec(),
                    components: shell.ms.len(),
                    blocks,
                });
                near_shells.push(shell);
            }
            let mut columns =
                ctx.allocate::<Complex64>(product(&[members.len(), CHUNK_POINTS])?)?;
            for range in sphere.chunks(CHUNK_POINTS) {
                let points = sphere.points(range.clone());
                let weights = sphere.weights(range.clone());
                let n = weights.len();
                let slice = &mut columns.as_mut_slice()[..members.len() * n];
                for (row, &p) in slice.chunks_mut(n).zip(members) {
                    evaluate(ctx, &projectors[p], points, row)?;
                    for (value, w) in row.iter_mut().zip(weights) {
                        *value *= *w;
                    }
                }
                for (shell, table) in near_shells.iter().zip(&mut group.shells) {
                    let spline = shell.radial.spline(shell.l)?;
                    let unit = vec![Complex64::new(1., 0.); table.images.len()];
                    for derivative in derivatives {
                        let slot = derivative.map_or(0, |a| a + 1);
                        let block = kernels::bloch_project(
                            ctx,
                            &ShellImages {
                                xyz: points,
                                center: shell.center,
                                images: &table.images,
                                phases: &unit,
                                support: shell.radial.support(),
                                spline: &spline,
                                ms: &shell.ms,
                                derivative: derivative.map_or(-1, |a| a as i32),
                            },
                            slice,
                        )?;
                        let to = table.blocks[slot]
                            .as_mut()
                            .ok_or_else(|| invalid("a projection block was not allocated"))?;
                        for (x, y) in to.as_mut_slice().iter_mut().zip(block.as_slice()) {
                            *x += y;
                        }
                    }
                }
            }
            groups.push(group);
        }
        Ok(Self {
            functions,
            count,
            groups,
        })
    }

    pub fn functions(&self) -> usize {
        self.functions
    }

    pub fn projectors(&self) -> usize {
        self.count
    }

    /// `sum_T e^{-ik.T} weight(T) B^slot_T`, `m x p`.
    fn sum(
        &self,
        k: [f64; 3],
        slot: usize,
        weight: &dyn Fn(&Group, &ShellBlocks, [f64; 3]) -> f64,
    ) -> Result<Matrix> {
        let mut out = Matrix::zeros(self.functions, self.count);
        for group in &self.groups {
            for shell in &group.shells {
                let blocks = shell.blocks[slot]
                    .as_ref()
                    .ok_or_else(|| {
                        invalid("this projection table was not formed with that derivative")
                    })?
                    .as_slice();
                let members = group.members.len();
                for (r, image) in shell.images.iter().enumerate() {
                    let w = weight(group, shell, *image);
                    if w == 0. {
                        continue;
                    }
                    let angle = -(k[0] * image[0] + k[1] * image[1] + k[2] * image[2]);
                    let phase = Complex64::new(angle.cos(), angle.sin()) * w;
                    for (i, &local) in shell.local_rows.iter().enumerate() {
                        let mu = group.rows[local];
                        let block = &blocks[(r * shell.components + i) * members..][..members];
                        for (value, &p) in block.iter().zip(&group.members) {
                            out[(mu, p)] += phase * value;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// `C_k` (`m x p`).
    pub fn value(&self, k: [f64; 3]) -> Result<Matrix> {
        self.sum(k, 0, &|_, _, _| 1.)
    }

    /// `G_a(k) = dC/dc_mu`, the derivative with respect to the center of the ROW function (the
    /// projector and its sphere fixed): minus the derivative blocks.
    pub fn gradient(&self, k: [f64; 3]) -> Result<[Matrix; 3]> {
        let mut out = [
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
            Matrix::zeros(0, 0),
        ];
        for (axis, to) in out.iter_mut().enumerate() {
            *to = -self.sum(k, axis + 1, &|_, _, _| 1.)?;
        }
        Ok(out)
    }

    /// `sum_ab E_ab dC_ab` for each direction: the spheres rigid about their atoms, so that
    /// `d chi_mu(x_p) / d eps_ab = -Delta_b sum_T e^{ik.T} (d_a phi)(x_p - c_mu - T)` with
    /// `Delta = c_mu + T - R_A`.
    pub fn strain(&self, k: [f64; 3], directions: &[Direction]) -> Result<Vec<Matrix>> {
        directions
            .iter()
            .map(|e| {
                let mut total = Matrix::zeros(self.functions, self.count);
                for a in 0..3 {
                    if e[a].iter().all(|x| *x == 0.) {
                        continue;
                    }
                    total += self.sum(k, a + 1, &|group, shell, t| {
                        -(0..3)
                            .map(|b| e[a][b] * (shell.center[b] + t[b] - group.center[b]))
                            .sum::<f64>()
                    })?;
                }
                Ok(total)
            })
            .collect()
    }
}

/// The k-independent tables of a setup (the projections and the short-range spheres, image
/// resolved), formed on first use and shared by every provider that holds them: a force and a
/// stress of one converged crystal read the same quadrature.
pub struct SetupTables {
    scalar: OnceLock<ProjectionTable>,
    spin_orbit: OnceLock<ProjectionTable>,
    forming: Mutex<()>,
    spheres: SphereCache,
}

impl Default for SetupTables {
    fn default() -> Self {
        Self::new()
    }
}

impl SetupTables {
    pub fn new() -> Self {
        Self {
            scalar: OnceLock::new(),
            spin_orbit: OnceLock::new(),
            forming: Mutex::new(()),
            spheres: SphereCache::new(),
        }
    }

    /// One short-range table per sphere of `spheres`.
    pub fn short_range(
        &self,
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        spheres: &[ShortRange<'_>],
    ) -> Result<&[SphereTable]> {
        self.spheres.tables(ctx, shells, lattice, spheres)
    }

    /// The table of the scalar projectors, with the value and the three derivatives.
    pub fn scalar(
        &self,
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        projectors: &[Projector],
    ) -> Result<&ProjectionTable> {
        Self::table(
            &self.scalar,
            &self.forming,
            ctx,
            shells,
            lattice,
            projectors,
        )
    }

    /// The table of the spin-orbit union projectors of a spinor crystal.
    pub fn spin_orbit(
        &self,
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        projectors: &[Projector],
    ) -> Result<&ProjectionTable> {
        Self::table(
            &self.spin_orbit,
            &self.forming,
            ctx,
            shells,
            lattice,
            projectors,
        )
    }

    /// Formed under the lock, so that k-point lanes asking at once form it once.
    fn table<'a>(
        slot: &'a OnceLock<ProjectionTable>,
        forming: &Mutex<()>,
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        projectors: &[Projector],
    ) -> Result<&'a ProjectionTable> {
        if let Some(table) = slot.get() {
            return Ok(table);
        }
        let _forming = forming.lock().unwrap_or_else(|e| e.into_inner());
        if slot.get().is_none() {
            let table = ProjectionTable::new(
                ctx,
                shells,
                lattice,
                projectors,
                &[None, Some(0), Some(1), Some(2)],
            )?;
            let _ = slot.set(table);
        }
        slot.get()
            .ok_or_else(|| invalid("the projection table was not formed"))
    }
}
