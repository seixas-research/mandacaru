//! Shared by `strain_spheres.rs`: the projections of a projector list on its rigid quadrature
//! spheres, the strain tangent of them written two independent ways, and the wrong variants the
//! controls compare with the central difference.
//!
//! * `variant(Form::Right)` is the sphere form `A_ab - s_b G_a` per chunk of points (the library
//!   function written out, so the wrong forms can be swapped in);
//! * `image_resolved` is the identity `dC_ab = sum_T e^{-ik.T} (-Delta_b) <d_a phi_T | w p>` with
//!   `Delta = c + T - R_A`, from the per-image projection kernel (no weighted kernel, no points).
#![allow(dead_code)]

pub mod hubbard;
pub mod linear;
pub mod local;

use mandacaru_core::{
    kernels::{bloch_project, bloch_shell, ShellImages},
    periodic::setup::{
        images::{Images, Region},
        pulay::Matrix,
        shells::{DenseRows, Sampling, Shells},
        spheres::{chunk_size, Projector, Sphere},
        strain_samples::weighted_derivatives,
        strain_spheres::sphere_gradients,
    },
    Context,
};
use nalgebra::{DMatrixView, Matrix3};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// How `d chi / d eps_ab` on a sphere is formed from `A` (the weighted sum) and `G` (the summed
/// derivative), `s = x - R_A`.
#[derive(Clone, Copy, Debug)]
pub enum Form {
    /// `A_ab - s_b G_a`.
    Right,
    /// `A_ab`: the sphere stays rigid on nothing (the atom's own displacement dropped).
    NoSphereTerm,
    /// `G_a - s_b G_a`: the image weight `y_b` dropped.
    NoWeight,
    /// `-(A_ab - s_b G_a)`.
    Sign,
    /// `A_ba - s_a G_b`: the other component.
    Transposed,
    /// `G_a`: no weight at all.
    Unweighted,
}

/// The projectors that share a sphere (an atom and a cutoff), as `projection` groups them.
pub fn groups(projectors: &[Projector]) -> Vec<Vec<usize>> {
    let mut spheres: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for (p, projector) in projectors.iter().enumerate() {
        spheres
            .entry((projector.atom, (projector.r_cut * 1e12).round() as i64))
            .or_default()
            .push(p);
    }
    spheres.into_values().collect()
}

/// `p(x) w_x` for the members of one sphere on a chunk, `(member, point)`.
fn columns(
    ctx: &Context,
    projectors: &[Projector],
    members: &[usize],
    points: [&[f64]; 3],
    weights: &[f64],
) -> Vec<Complex64> {
    let n = weights.len();
    let mut out = vec![Complex64::new(0., 0.); members.len() * n];
    for (row, &p) in out.chunks_mut(n).zip(members) {
        let projector = &projectors[p];
        let spline = projector.radial.spline(projector.l).unwrap();
        // The function ends at the last node of its table.
        let support = f64::from_bits(projector.radial.extent().to_bits() + 1);
        bloch_shell(
            ctx,
            &ShellImages {
                xyz: points,
                center: projector.center,
                images: &[[0.; 3]],
                phases: &[Complex64::new(1., 0.)],
                support,
                spline: &spline,
                ms: &[projector.m],
                derivative: -1,
            },
            &[0],
            1,
            row,
        )
        .unwrap();
        for (v, w) in row.iter_mut().zip(weights) {
            *v *= *w;
        }
    }
    out
}

/// `d chi / d eps_ab` of every function at the points of a sphere about `center` in the form
/// `form` (`g` is the summed derivative `G` of the same points), `(function, point)`.
pub fn tangent_of(
    ctx: &Context,
    shells: &Shells,
    sampling: &Sampling<'_>,
    points: [&[f64]; 3],
    center: [f64; 3],
    g: &[mandacaru_core::Buffer<Complex64>; 3],
    form: Form,
    (a, b): (usize, usize),
) -> Vec<Complex64> {
    let (near, n) = (shells.functions(), points[0].len());
    let weighted = |axis: usize, weight: usize| {
        let mut rows = vec![Complex64::new(0., 0.); near * n];
        weighted_derivatives(
            ctx,
            shells,
            sampling,
            points,
            axis,
            weight,
            &mut DenseRows::new(&mut rows, near, n),
        )
        .unwrap();
        rows
    };
    let s = |c: usize, x: usize| points[c][x] - center[c];
    match form {
        Form::Right | Form::Sign => {
            let sign = if matches!(form, Form::Sign) { -1. } else { 1. };
            let rows = weighted(a, b);
            (0..near * n)
                .map(|i| (rows[i] - g[a].as_slice()[i] * s(b, i % n)) * sign)
                .collect()
        }
        Form::NoSphereTerm => weighted(a, b),
        Form::NoWeight => (0..near * n)
            .map(|i| g[a].as_slice()[i] * (1. - s(b, i % n)))
            .collect(),
        Form::Transposed => {
            let rows = weighted(b, a);
            (0..near * n)
                .map(|i| rows[i] - g[b].as_slice()[i] * s(a, i % n))
                .collect()
        }
        Form::Unweighted => g[a].as_slice().to_vec(),
    }
}

/// `dC[a][b]` of the projector list at `k` with the sphere tangent of `form`, for the pairs
/// listed (the others stay zero).
pub fn variant(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
    form: Form,
    pairs: &[(usize, usize)],
) -> [[Matrix; 3]; 3] {
    let (m, count) = (shells.functions(), projectors.len());
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, count)));
    for members in groups(projectors) {
        let first = &projectors[members[0]];
        let region = Region {
            center: first.center,
            radius: first.r_cut,
        };
        let reached = shells.reaching(ctx, lattice, &region).unwrap();
        let near = reached.rows.len();
        if near == 0 {
            continue;
        }
        let sphere = Sphere::projector(ctx, first.center, first.r_cut).unwrap();
        let sampling = Sampling {
            lattice,
            region,
            kpoints: &[k],
        };
        for range in sphere.chunks(chunk_size(1, 4 * near)) {
            let points = sphere.points(range.clone());
            let n = range.len();
            let cols = columns(ctx, projectors, &members, points, sphere.weights(range));
            let cols = DMatrixView::from_slice(&cols, n, members.len());
            let g = sphere_gradients(ctx, &reached.shells, &sampling, points).unwrap();
            for &(a, b) in pairs {
                {
                    let tangent = tangent_of(
                        ctx,
                        &reached.shells,
                        &sampling,
                        points,
                        first.center,
                        &g,
                        form,
                        (a, b),
                    );
                    let rows = DMatrixView::from_slice(&tangent, n, near);
                    let block = rows.adjoint() * cols;
                    for (i, &mu) in reached.rows.iter().enumerate() {
                        for (j, &p) in members.iter().enumerate() {
                            out[a][b][(mu, p)] += block[(i, j)];
                        }
                    }
                }
            }
        }
    }
    out
}

/// The weight the image-resolved identity gives the derivative of an image.
#[derive(Clone, Copy, Debug)]
pub enum Displacement {
    /// `Delta = c_mu + T - R_A`: the image translation `T` is part of it.
    Full,
    /// `c_mu - R_A`: the lattice vector `R` dropped (invisible at Gamma, wrong across cells).
    WithoutImage,
}

/// `dC[a][b]` by the image-resolved identity (the per-image projection kernel, the translation
/// weight and the projection phase applied outside it).
pub fn image_resolved(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
) -> [[Matrix; 3]; 3] {
    image_resolved_with(
        ctx,
        shells,
        lattice,
        k,
        projectors,
        Displacement::Full,
        &[0, 1, 2],
    )
}

/// [`image_resolved`] with the displacement of the weight chosen, for the derivative axes `axes`
/// only (the other `dC[a][b]` stay zero).
pub fn image_resolved_with(
    ctx: &Context,
    shells: &Shells,
    lattice: &Matrix3<f64>,
    k: [f64; 3],
    projectors: &[Projector],
    displacement: Displacement,
    axes: &[usize],
) -> [[Matrix; 3]; 3] {
    let (m, count) = (shells.functions(), projectors.len());
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(m, count)));
    for members in groups(projectors) {
        let first = &projectors[members[0]];
        let region = Region {
            center: first.center,
            radius: first.r_cut,
        };
        let reached = shells.reaching(ctx, lattice, &region).unwrap();
        if reached.rows.is_empty() {
            continue;
        }
        let sphere = Sphere::projector(ctx, first.center, first.r_cut).unwrap();
        for range in sphere.chunks(chunk_size(1, 4 * reached.rows.len())) {
            let points = sphere.points(range.clone());
            let cols = columns(ctx, projectors, &members, points, sphere.weights(range));
            for shell in reached.shells.shells() {
                let images =
                    Images::around(ctx, lattice, &region, shell.center, shell.radial.support())
                        .unwrap();
                if images.is_empty() {
                    continue;
                }
                let spline = shell.radial.spline(shell.l).unwrap();
                let nm = shell.ms.len();
                for &a in axes {
                    let blocks = bloch_project(
                        ctx,
                        &ShellImages {
                            xyz: points,
                            center: shell.center,
                            images: images.vectors(),
                            phases: &vec![Complex64::new(1., 0.); images.len()],
                            support: shell.radial.support(),
                            spline: &spline,
                            ms: &shell.ms,
                            derivative: a as i32,
                        },
                        &cols,
                    )
                    .unwrap();
                    for (r, t) in images.vectors().iter().enumerate() {
                        let angle = -(k[0] * t[0] + k[1] * t[1] + k[2] * t[2]);
                        let phase = Complex64::new(angle.cos(), angle.sin());
                        for b in 0..3 {
                            // -Delta_b, Delta = c + T - R_A.
                            let image = match displacement {
                                Displacement::Full => t[b],
                                Displacement::WithoutImage => 0.,
                            };
                            let minus_delta = first.center[b] - shell.center[b] - image;
                            for (i, &row) in shell.rows.iter().enumerate() {
                                for (j, &p) in members.iter().enumerate() {
                                    out[a][b][(reached.rows[row], p)] += phase
                                        * blocks.as_slice()[(r * nm + i) * members.len() + j]
                                        * minus_delta;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
