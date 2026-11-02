//! Shared by `strain_spheres_short_range.rs` and `strain_spheres_linear.rs`: the cells (Si, Al2,
//! a skewed Al2 and the skew LiH cell), the strained rebuilds the short-range central difference
//! uses, and the wrong variants its controls compare with it.
//!
//! * `v_sr_at` rebuilds the short-range matrix of every atom's sphere on the cell carried by
//!   `x -> (1 + eps) x` (functions, centers and lattice carried, the sphere rule and the radial
//!   potential rigid about the carried atom);
//! * `v_variant` is the library's short-range strain written out with the tangent form of
//!   [`super::Form`] swapped, and `image_potential_term` is the plan's pre-correction `du_A`
//!   (the image potentials of the other atoms seen from the sphere) that does not belong to
//!   `V_sr`.
#![allow(dead_code)]

use super::{tangent_of, Form};
use crate::{
    periodic_gradient_crystals_support::Data,
    setup_gradients_support::Cell,
    setup_pulay_support::Fixture,
    strain_samples_support::{entry, Carried},
};
use mandacaru_core::{
    paw::local_split::{short_range_potential, short_range_potential_derivative},
    periodic::setup::{
        basis_gradient::{short_range_cross, ShortRange},
        ions::LocalPotential,
        pulay::Matrix,
        shapes::visit_lattice,
        shells::{DenseRows, Sampling, Shells},
        spheres::{
            chunk_size, Sphere, SHORT_RANGE_AZIMUTHAL, SHORT_RANGE_POLAR, SHORT_RANGE_RADIAL,
        },
        strain_spheres::sphere_gradients,
    },
    Context,
};
use nalgebra::{DMatrixView, Matrix3, Vector3};
use num_complex::Complex64;

/// What a control that depends on the geometry must do on a case.
#[derive(Clone, Copy, PartialEq)]
pub enum Control {
    /// Not run (a symmetric lattice cannot tell rows from columns).
    Skip,
    /// Run and reported: the geometry gives the control too little to bite.
    Reported,
    /// Run and asserted to miss by more than 100 gates.
    Asserted,
}

/// The lattice (vectors as columns) of the skewed Al2: tetragonal Al2 with three off-diagonal
/// entries, so that it is not its own transpose and the images reach every sphere.
pub fn skew_lattice() -> Matrix3<f64> {
    Matrix3::from_row_slice(&[5.4117645, 0.9, 0.5, 0., 5.4117645, 0.7, 0., 0., 7.6533908])
}

pub struct Local {
    pub name: &'static str,
    pub cell: Cell,
    pub shells: Shells,
    pub potentials: Vec<LocalPotential>,
}

impl Local {
    /// Si or Al2 of the WP-X freezes.
    pub fn wpx(ctx: &Context, name: &'static str) -> Self {
        let fx = Fixture::new(ctx, name);
        Self {
            name,
            cell: fx.case.cell(),
            shells: fx.shells,
            potentials: fx.potentials,
        }
    }
    /// Al2 with a skewed lattice (atoms and basis unchanged, Cartesian): a synthetic cell whose
    /// images reach every sphere and whose lattice is not its own transpose.
    pub fn al2_skew(ctx: &Context) -> Self {
        let mut local = Self::wpx(ctx, "al2");
        local.name = "al2_skew";
        local.cell.lattice = skew_lattice();
        local
    }
    /// The triclinic LiH cell of the crystal-force freeze.
    pub fn lih() -> Self {
        let d = Data::load("lih");
        Self {
            name: "lih",
            cell: Cell {
                atoms: d.atoms.clone(),
                channels: d.channels.clone(),
                lattice: d.lattice,
                sigma: d.sigma,
            },
            shells: d.shells,
            potentials: d.potentials,
        }
    }
    pub fn centers(&self) -> Vec<[f64; 3]> {
        self.cell.atoms.iter().map(|a| a.center).collect()
    }
    pub fn spheres(&self) -> Vec<ShortRange<'_>> {
        self.cell
            .atoms
            .iter()
            .zip(&self.potentials)
            .map(|(a, potential)| ShortRange {
                center: a.center,
                potential,
                sigma: self.cell.sigma,
            })
            .collect()
    }
}

/// The short-range matrix of every atom's sphere on the cell carried by `eps`.
pub fn v_sr_at(ctx: &Context, local: &Local, eps: &Matrix3<f64>, k: [f64; 3]) -> Vec<Matrix> {
    let c = Carried::new(&local.cell.lattice, &local.shells, &[], eps, k);
    let m = c.shells.functions();
    local
        .cell
        .atoms
        .iter()
        .zip(&local.potentials)
        .map(|(atom, potential)| {
            let sphere = ShortRange {
                center: (c.map * Vector3::from(atom.center)).into(),
                potential,
                sigma: local.cell.sigma,
            };
            let block =
                short_range_cross(ctx, &c.lattice, &[c.k], &sphere, &c.shells, None, &c.shells)
                    .unwrap();
            Matrix::from_row_slice(m, m, block.as_slice())
        })
        .collect()
}

/// The Richardson central difference of [`v_sr_at`] along the single entry `(a, b)` from the
/// steps `step` and `2 step` (truncation `O(step^4)`).
pub fn v_sr_derivative(
    ctx: &Context,
    local: &Local,
    k: [f64; 3],
    (a, b): (usize, usize),
    step: f64,
) -> Vec<Matrix> {
    let central = |s: f64| -> Vec<Matrix> {
        let plus = v_sr_at(ctx, local, &entry(a, b, s), k);
        let minus = v_sr_at(ctx, local, &entry(a, b, -s), k);
        plus.iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) / Complex64::new(2. * s, 0.))
            .collect()
    };
    central(step)
        .into_iter()
        .zip(central(2. * step))
        .map(|(one, two)| (one * Complex64::new(4., 0.) - two) / Complex64::new(3., 0.))
        .collect()
}

/// The rule of an atom's short-range sphere and its weights (`w_x`, and `w_x v^sr(r_x)`).
fn rule(ctx: &Context, sphere: &ShortRange<'_>) -> (Sphere, Vec<f64>, Vec<f64>) {
    let rule = Sphere::rule(
        ctx,
        sphere.center,
        sphere.potential.reach(sphere.sigma).unwrap(),
        SHORT_RANGE_RADIAL,
        SHORT_RANGE_POLAR,
        SHORT_RANGE_AZIMUTHAL,
        sphere.potential.cutoff,
    )
    .unwrap();
    let all = 0..rule.len();
    let radii = rule.radii(all.clone());
    let sampled: Vec<f64> = radii.iter().map(|r| sphere.potential.value(*r)).collect();
    let v = short_range_potential(&sampled, sphere.potential.charge, sphere.sigma, radii).unwrap();
    let plain = rule.weights(all).to_vec();
    let weighted = plain.iter().zip(&v).map(|(w, v)| w * v).collect();
    (rule, plain, weighted)
}

/// `dV[a][b]` of one sphere with the tangent form of `form`, for the pairs listed.
pub fn v_variant(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    sphere: &ShortRange<'_>,
    form: Form,
    pairs: &[(usize, usize)],
) -> [[Matrix; 3]; 3] {
    let m = shells.functions();
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, m)));
    let (rule, _, weights) = rule(ctx, sphere);
    let region = rule.region();
    let reached = shells.reaching(ctx, lattice, &region).unwrap();
    let near = reached.rows.len();
    let sampling = Sampling {
        lattice,
        region,
        kpoints: &[k],
    };
    for range in rule.chunks(chunk_size(1, 6 * near)) {
        let points = rule.points(range.clone());
        let n = range.len();
        let mut values = vec![Complex64::new(0., 0.); near * n];
        reached
            .shells
            .values(
                ctx,
                &sampling,
                points,
                &mut DenseRows::new(&mut values, near, n),
            )
            .unwrap();
        let w = &weights[range];
        let ket = Matrix::from_fn(n, near, |x, j| values[j * n + x] * w[x]);
        let g = sphere_gradients(ctx, &reached.shells, &sampling, points).unwrap();
        for &(a, b) in pairs {
            let tangent = tangent_of(
                ctx,
                &reached.shells,
                &sampling,
                points,
                sphere.center,
                &g,
                form,
                (a, b),
            );
            let half = DMatrixView::from_slice(&tangent, n, near).adjoint() * &ket;
            for (i, &mu) in reached.rows.iter().enumerate() {
                for (j, &nu) in reached.rows.iter().enumerate() {
                    out[a][b][(mu, nu)] += half[(i, j)] + half[(j, i)].conj();
                }
            }
        }
    }
    out
}

/// The plan's pre-correction `du_A` contracted with the functions on the sphere of `atom`:
/// `sum_x w_x conj(chi_i) du_ab(x) chi_j`, `du_ab = -sum_{B,T} (v_B^sr)'(|y|) y_a Delta_b / |y|`,
/// the image potentials of the other atoms (and the own images) seen from the sphere. It belongs
/// to the compensation sum, NOT to `V_sr`, whose potential is the atom's own and rigid.
pub fn image_potential_term(
    ctx: &Context,
    local: &Local,
    atom: usize,
    k: [f64; 3],
    pairs: &[(usize, usize)],
) -> [[Matrix; 3]; 3] {
    let m = local.shells.functions();
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, m)));
    let spheres = local.spheres();
    let sphere = &spheres[atom];
    let (rule, plain, _) = rule(ctx, sphere);
    let all = 0..rule.len();
    let points = rule.points(all.clone());
    let n = rule.len();
    let sigma = local.cell.sigma;
    let center = Vector3::from(sphere.center);
    let at = |p: usize| Vector3::new(points[0][p], points[1][p], points[2][p]);
    let mut du = vec![[0f64; 9]; n];
    let mut distance = vec![0.; n];
    let mut sampled = vec![0.; n];
    for (other, species) in local.potentials.iter().enumerate() {
        let reach = species.reach(sigma).unwrap() + rule.radius;
        let d0 = center - Vector3::from(local.cell.atoms[other].center);
        visit_lattice(
            &local.cell.lattice,
            reach + d0.norm(),
            false,
            4096,
            &mut |images| {
                for image in images {
                    let shift = Vector3::from(*image);
                    if other == atom && shift.norm() < 1e-10 || (d0 - shift).norm() > reach {
                        continue;
                    }
                    let origin = Vector3::from(local.cell.atoms[other].center) + shift;
                    for p in 0..n {
                        distance[p] = (at(p) - origin).norm();
                        sampled[p] = species.slope(distance[p]);
                    }
                    let slope = short_range_potential_derivative(
                        &sampled,
                        species.charge,
                        sigma,
                        &distance,
                    )?;
                    let delta = origin - center;
                    for p in 0..n {
                        let y = at(p) - origin;
                        for a in 0..3 {
                            for b in 0..3 {
                                du[p][a * 3 + b] -=
                                    slope[p] / distance[p].max(1e-300) * y[a] * delta[b];
                            }
                        }
                    }
                }
                Ok(())
            },
        )
        .unwrap();
    }
    let region = rule.region();
    let reached = local
        .shells
        .reaching(ctx, &local.cell.lattice, &region)
        .unwrap();
    let near = reached.rows.len();
    let mut values = vec![Complex64::new(0., 0.); near * n];
    reached
        .shells
        .values(
            ctx,
            &Sampling {
                lattice: &local.cell.lattice,
                region,
                kpoints: &[k],
            },
            points,
            &mut DenseRows::new(&mut values, near, n),
        )
        .unwrap();
    for &(a, b) in pairs {
        let bra = Matrix::from_fn(n, near, |x, i| values[i * n + x]);
        let ket = Matrix::from_fn(n, near, |x, j| {
            values[j * n + x] * (plain[x] * du[x][a * 3 + b])
        });
        let block = bra.adjoint() * ket;
        for (i, &mu) in reached.rows.iter().enumerate() {
            for (j, &nu) in reached.rows.iter().enumerate() {
                out[a][b][(mu, nu)] += block[(i, j)];
            }
        }
    }
    out
}
