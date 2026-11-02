//! The row-center derivative of the basis in the crystal force: the slope of a tabulated radial
//! function, the Cartesian derivative `d(R_l Y_lm)/d x_axis` of a basis function as a function of
//! its own, and the short-range local matrix of one atom between two sets of functions, the bra
//! optionally differentiated (`<d_axis chi_mu | V^sr_A | chi_nu>`, the sphere held fixed).
//!
//! The matrix is `-(G + G^+)` summed over the moved atoms' rows by translation invariance of the
//! sphere rule (the ownership identity of the force engine); this module supplies `G` and the
//! plain cross block, not the contraction.
use super::{
    ions::LocalPotential,
    shells::{Shell, Shells},
    spheres::{
        weighted_cross, Sphere, SHORT_RANGE_AZIMUTHAL, SHORT_RANGE_POLAR, SHORT_RANGE_RADIAL,
    },
};
use crate::{
    basis::table,
    kernels::{self, ShellImages},
    paw::local_split::short_range_potential,
    periodic::invalid,
    product, Buffer, Context, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;

/// `dR/dr` of a tabulated radial function (the not-a-knot spline of its table) at `nodes`, with
/// the continuation `R(r_0) (r / r_0)^l` below the first node and zero past the last.
pub fn radial_slope(
    ctx: &Context,
    r: &[f64],
    values: &[f64],
    l: usize,
    nodes: &[f64],
) -> Result<Vec<f64>> {
    table::evaluate(ctx, r, values, l, nodes, true)
}

/// The Cartesian derivative along `axis` of the component `m` of `shell` (a single function at
/// the shell's center, no lattice images) at `points`:
/// `R'(r) (x_axis / r) Y_lm + R(r) d Y_lm / d x_axis`. At the nucleus only a p function has a
/// value, `R'(0)` times the constant gradient of its harmonic.
pub fn basis_gradient(
    ctx: &Context,
    shell: &Shell,
    m: i32,
    axis: usize,
    points: [&[f64]; 3],
) -> Result<Vec<Complex64>> {
    if axis > 2 || !shell.ms.contains(&m) {
        return Err(invalid(
            "a gradient needs an axis 0..=2 and a component of the shell",
        ));
    }
    let spline = shell.radial.spline(shell.l)?;
    // The function ends at the last node of its table (an endpoint-exclusive support).
    let support = f64::from_bits(shell.radial.extent().to_bits() + 1);
    let mut out = ctx.allocate::<Complex64>(points[0].len())?;
    kernels::bloch_shell(
        ctx,
        &ShellImages {
            xyz: points,
            center: shell.center,
            images: &[[0.; 3]],
            phases: &[Complex64::new(1., 0.)],
            support,
            spline: &spline,
            ms: &[m],
            derivative: axis as i32,
        },
        &[0],
        1,
        out.as_mut_slice(),
    )?;
    Ok(out.as_slice().to_vec())
}

/// The functions of `shells` at the positions `rows` (in this order), renumbered `0..rows.len()`;
/// a row may not repeat.
pub fn subset(shells: &Shells, rows: &[usize]) -> Result<Shells> {
    let mut place = vec![usize::MAX; shells.functions()];
    for (i, &row) in rows.iter().enumerate() {
        if row >= place.len() || std::mem::replace(&mut place[row], i) != usize::MAX {
            return Err(invalid("a subset takes each existing row once"));
        }
    }
    let kept = shells
        .shells()
        .iter()
        .filter_map(|shell| {
            let both: Vec<(i32, usize)> = shell
                .ms
                .iter()
                .zip(&shell.rows)
                .filter(|(_, &row)| place[row] != usize::MAX)
                .map(|(&m, &row)| (m, place[row]))
                .collect();
            (!both.is_empty()).then(|| Shell {
                ms: both.iter().map(|b| b.0).collect(),
                rows: both.iter().map(|b| b.1).collect(),
                ..shell.clone()
            })
        })
        .collect();
    Shells::new(kept)
}

/// One atom's short-range local potential: the dataset's local potential less the Gaussian ion
/// of width `sigma`, on a sphere about `center`.
pub struct ShortRange<'a> {
    pub center: [f64; 3],
    pub potential: &'a LocalPotential,
    pub sigma: f64,
}

/// `(k, bra, ket)` row-major: `<bra_k | V^sr | ket_k>` between the two sets of functions, zero
/// for a pair no image of which reaches the sphere. With `bra_axis` the bra functions are
/// replaced by their derivative along that axis.
pub fn short_range_cross(
    ctx: &Context,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    sphere: &ShortRange<'_>,
    bra: &Shells,
    bra_axis: Option<usize>,
    ket: &Shells,
) -> Result<Buffer<Complex64>> {
    let (nk, nb, nc) = (kpoints.len(), bra.functions(), ket.functions());
    let mut out = ctx.allocate::<Complex64>(product(&[nk, nb, nc])?)?;
    let radius = sphere.potential.reach(sphere.sigma)?;
    let rule = Sphere::rule(
        ctx,
        sphere.center,
        radius,
        SHORT_RANGE_RADIAL,
        SHORT_RANGE_POLAR,
        SHORT_RANGE_AZIMUTHAL,
        sphere.potential.cutoff,
    )?;
    let all = 0..rule.len();
    let radii = rule.radii(all.clone());
    let sampled: Vec<f64> = radii.iter().map(|r| sphere.potential.value(*r)).collect();
    let potential = short_range_potential(&sampled, sphere.potential.charge, sphere.sigma, radii)?;
    let weights: Vec<f64> = rule
        .weights(all)
        .iter()
        .zip(&potential)
        .map(|(w, v)| w * v)
        .collect();
    let gram = weighted_cross(
        ctx,
        bra,
        bra_axis,
        Some(ket),
        lattice,
        kpoints,
        &rule,
        &weights,
    )?;
    let width = gram.columns.len();
    for k in 0..nk {
        for (i, &row) in gram.rows.iter().enumerate() {
            for (j, &column) in gram.columns.iter().enumerate() {
                out.as_mut_slice()[(k * nb + row) * nc + column] =
                    gram.values.as_slice()[(k * gram.rows.len() + i) * width + j];
            }
        }
    }
    Ok(out)
}
