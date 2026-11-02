//! Checked entry points to existing numerical kernels.
//!
//! Every unsafe call below has validated dimensions and disjoint, live output
//! allocations. The legacy C ABI still requires its original caller contracts.

use crate::{finite, product, Buffer, Complex64, Context, Error, Grid, Result};
use mandacaru_kernels::{bloch, one_body, radial, two_center};

fn dimension(n: usize) -> Result<i32> {
    if n == 0 {
        return Err(Error::Invalid("kernel dimensions must be positive".into()));
    }
    i32::try_from(n).map_err(|_| Error::Overflow)
}
fn length(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{name} has {actual} entries; expected {expected}"
        )))
    }
}
fn complex_finite(values: &[Complex64], name: &str) -> Result<()> {
    if values.iter().all(|v| v.re.is_finite() && v.im.is_finite()) {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{name} must contain finite complex values"
        )))
    }
}
fn angular(l: i32) -> Result<()> {
    if (0..=10).contains(&l) {
        Ok(())
    } else {
        Err(Error::Invalid("angular momentum must be in 0..10".into()))
    }
}

pub fn one_body(
    ctx: &Context,
    grid: &Grid,
    orbitals: usize,
    psi: &[Complex64],
    potential: &[f64],
) -> Result<(Buffer<Complex64>, Buffer<Complex64>)> {
    let m = dimension(orbitals)?;
    length(psi.len(), product(&[orbitals, grid.points()])?, "orbitals")?;
    length(potential.len(), grid.points(), "potential")?;
    complex_finite(psi, "orbitals")?;
    finite(potential, "potential")?;
    let mut kinetic = ctx.allocate(product(&[orbitals, orbitals])?)?;
    let mut local = ctx.allocate(product(&[orbitals, orbitals])?)?;
    let _scratch = ctx.reserve(product(&[grid.points(), 16, ctx.threads().min(orbitals)])?)?;
    let [nx, ny, nz] = grid.shape().map(|n| n as i32);
    // SAFETY: input shapes and integer ranges are checked; output buffers are
    // independent M*M allocations. The kernel joins all scoped workers.
    ctx.run(|| unsafe {
        one_body::one_body_general(
            psi.as_ptr(),
            potential.as_ptr(),
            m,
            nx,
            ny,
            nz,
            grid.metric().as_ptr(),
            grid.dv(),
            kinetic.as_mut_slice().as_mut_ptr(),
            local.as_mut_slice().as_mut_ptr(),
        )
    });
    complex_finite(kinetic.as_slice(), "kinetic result")?;
    complex_finite(local.as_slice(), "local result")?;
    Ok((kinetic, local))
}

macro_rules! laplacian {
    ($name:ident, $scalar:ty, $kernel:ident) => {
        pub fn $name(
            ctx: &Context,
            grid: &Grid,
            fields: usize,
            values: &[$scalar],
        ) -> Result<Buffer<$scalar>> {
            let n = dimension(fields)?;
            length(values.len(), product(&[fields, grid.points()])?, "fields")?;
            if !values.iter().all(|x| x.is_finite()) {
                return Err(Error::Invalid("fields must be finite".into()));
            }
            let mut output = ctx.allocate(values.len())?;
            let [nx, ny, nz] = grid.shape().map(|n| n as i32);
            // SAFETY: all shapes are checked, and input and output cannot alias.
            ctx.run(|| unsafe {
                one_body::$kernel(
                    values.as_ptr(),
                    output.as_mut_slice().as_mut_ptr(),
                    n,
                    nx,
                    ny,
                    nz,
                    grid.metric().as_ptr(),
                )
            });
            if !output.as_slice().iter().all(|x| x.is_finite()) {
                return Err(Error::Kernel("nonfinite Laplacian result".into()));
            }
            Ok(output)
        }
    };
}
laplacian!(laplacian, f64, laplacian_real);
laplacian!(laplacian_f32, f32, laplacian_real_f32);

pub fn project(
    ctx: &Context,
    points: usize,
    orbitals: usize,
    projectors: usize,
    dv: f64,
    psi: &[Complex64],
    chi: &[Complex64],
) -> Result<Buffer<Complex64>> {
    let (m, p) = (dimension(orbitals)?, dimension(projectors)?);
    if points == 0 || !dv.is_finite() || dv <= 0.0 {
        return Err(Error::Invalid(
            "projection grid and volume must be positive".into(),
        ));
    }
    let g = i64::try_from(points).map_err(|_| Error::Overflow)?;
    length(psi.len(), product(&[orbitals, points])?, "orbitals")?;
    length(chi.len(), product(&[projectors, points])?, "projectors")?;
    complex_finite(psi, "orbitals")?;
    complex_finite(chi, "projectors")?;
    let mut output = ctx.allocate(product(&[orbitals, projectors])?)?;
    // SAFETY: checked lengths and live, nonaliasing output with M*P entries.
    ctx.run(|| unsafe {
        one_body::kb_project(
            psi.as_ptr(),
            chi.as_ptr(),
            m,
            p,
            g,
            dv,
            output.as_mut_slice().as_mut_ptr(),
        )
    });
    complex_finite(output.as_slice(), "projection result")?;
    Ok(output)
}

pub fn bessel(ctx: &Context, l: i32, r: &[f64], k: &[f64]) -> Result<Buffer<f64>> {
    angular(l)?;
    let (nr, nk) = (dimension(r.len())?, dimension(k.len())?);
    finite(r, "radii")?;
    finite(k, "wave numbers")?;
    let mut output = ctx.allocate(product(&[r.len(), k.len()])?)?;
    // SAFETY: dimensions fit int32 and match all slices/output storage.
    ctx.run(|| unsafe {
        two_center::mandacaru_spherical_bessel_matrix(
            l,
            r.as_ptr(),
            nr,
            k.as_ptr(),
            nk,
            output.as_mut_slice().as_mut_ptr(),
        )
    });
    finite(output.as_slice(), "Bessel result")?;
    Ok(output)
}

pub fn fourier_bessel(
    ctx: &Context,
    l: i32,
    r: &[f64],
    weighted: &[f64],
    k: &[f64],
) -> Result<Buffer<f64>> {
    angular(l)?;
    let (nr, nk) = (dimension(r.len())?, dimension(k.len())?);
    length(weighted.len(), r.len(), "weighted radial function")?;
    finite(r, "radii")?;
    finite(k, "wave numbers")?;
    finite(weighted, "weighted radial function")?;
    let mut output = ctx.allocate(k.len())?;
    // SAFETY: all read slices have checked lengths; each worker writes disjoint k entries.
    ctx.run(|| unsafe {
        two_center::mandacaru_fourier_bessel_transform(
            l,
            r.as_ptr(),
            weighted.as_ptr(),
            nr,
            k.as_ptr(),
            nk,
            0,
            output.as_mut_slice().as_mut_ptr(),
        )
    });
    finite(output.as_slice(), "Fourier-Bessel result")?;
    Ok(output)
}

pub fn eigenpair(
    ctx: &Context,
    diagonal: &[f64],
    off_diagonal: &[f64],
    index: usize,
) -> Result<(f64, Buffer<f64>)> {
    let n = dimension(diagonal.len())?;
    length(off_diagonal.len(), diagonal.len() - 1, "off diagonal")?;
    if index >= diagonal.len() {
        return Err(Error::Invalid("eigenvalue index outside matrix".into()));
    }
    finite(diagonal, "diagonal")?;
    finite(off_diagonal, "off diagonal")?;
    let mut vector = ctx.allocate(diagonal.len())?;
    let _scratch = ctx.reserve(product(&[diagonal.len(), 8, 20])?)?;
    let mut value = 0.0;
    // SAFETY: n>=1, index<n, n-1 off-diagonals and n independent output entries.
    let status = ctx.run(|| unsafe {
        radial::mandacaru_tridiagonal_eigenpair(
            n,
            diagonal.as_ptr(),
            off_diagonal.as_ptr(),
            index as i32,
            f64::NAN,
            &mut value,
            vector.as_mut_slice().as_mut_ptr(),
        )
    });
    if status != 0 {
        return Err(Error::Kernel(format!(
            "radial eigenpair failed with status {status}"
        )));
    }
    finite(&[value], "eigenvalue")?;
    finite(vector.as_slice(), "eigenvector")?;
    Ok((value, vector))
}

pub fn numerov(
    ctx: &Context,
    f: &[f64],
    source: &[f64],
    initial: &[f64],
    h2: f64,
    boundary: usize,
    inward: bool,
) -> Result<Buffer<f64>> {
    let n = dimension(f.len())?;
    length(initial.len(), f.len(), "initial solution")?;
    length(source.len(), f.len(), "Numerov source")?;
    if n < 3 || boundary >= f.len() - 1 || !h2.is_finite() || h2 <= 0.0 {
        return Err(Error::Invalid(
            "Numerov needs at least 3 points, a valid boundary and positive step squared".into(),
        ));
    }
    finite(f, "Numerov coefficient")?;
    finite(source, "Numerov source")?;
    finite(initial, "initial solution")?;
    if inward && source.iter().any(|&value| value != 0.0) {
        return Err(Error::Invalid(
            "inward Numerov supports only a zero source".into(),
        ));
    }
    if f.iter().any(|&x| 1.0 - h2 * x / 12.0 == 0.0) {
        return Err(Error::Invalid("singular Numerov denominator".into()));
    }
    let mut output = ctx.allocate(f.len())?;
    output.as_mut_slice().copy_from_slice(initial);
    // SAFETY: n>=3 prevents unsigned underflow, boundary<n-1, each array has n entries.
    ctx.run(|| unsafe {
        if inward {
            radial::mandacaru_numerov_inward(
                n,
                f.as_ptr(),
                h2,
                boundary as i32,
                output.as_mut_slice().as_mut_ptr(),
            );
        } else {
            radial::mandacaru_numerov_outward(
                n,
                f.as_ptr(),
                source.as_ptr(),
                h2,
                boundary as i32,
                output.as_mut_slice().as_mut_ptr(),
            );
        }
    });
    finite(output.as_slice(), "Numerov result")?;
    Ok(output)
}

pub struct Spline<'a> {
    breaks: &'a [f64],
    coefficients: &'a [f64],
    cutoff: f64,
    l: i32,
}
impl<'a> Spline<'a> {
    pub fn new(breaks: &'a [f64], coefficients: &'a [f64], cutoff: f64, l: i32) -> Result<Self> {
        angular(l)?;
        dimension(breaks.len())?;
        if breaks.len() < 2
            || breaks.windows(2).any(|w| w[1] <= w[0])
            || !cutoff.is_finite()
            || cutoff <= 0.0
        {
            return Err(Error::Invalid(
                "spline needs increasing knots and positive finite cutoff".into(),
            ));
        }
        finite(breaks, "spline knots")?;
        finite(coefficients, "spline coefficients")?;
        length(
            coefficients.len(),
            product(&[breaks.len() - 1, 4])?,
            "spline coefficients",
        )?;
        Ok(Self {
            breaks,
            coefficients,
            cutoff,
            l,
        })
    }
}

pub fn real_shell(
    ctx: &Context,
    xyz: [&[f64]; 3],
    center: [f64; 3],
    support: f64,
    spline: &Spline<'_>,
    ms: &[i32],
    derivative: i32,
) -> Result<Buffer<f64>> {
    let points = xyz[0].len();
    let nm = dimension(ms.len())?;
    if points == 0
        || !support.is_finite()
        || support <= 0.0
        || !(-1..=2).contains(&derivative)
        || ms.iter().any(|&m| m < -spline.l || m > spline.l)
    {
        return Err(Error::Invalid(
            "invalid shell support, angular component or derivative axis".into(),
        ));
    }
    for axis in xyz {
        length(axis.len(), points, "shell coordinates")?;
        finite(axis, "shell coordinates")?;
    }
    finite(&center, "shell center")?;
    let count = product(&[ms.len(), points])?;
    let points = i64::try_from(points).map_err(|_| Error::Overflow)?;
    let mut rows = ctx.allocate::<i64>(ms.len())?;
    for (i, value) in rows.as_mut_slice().iter_mut().enumerate() {
        *value = i as i64;
    }
    let mut output = ctx.allocate(count)?;
    // SAFETY: coordinates, spline dimensions, angular indices and destination
    // rows are checked. Output has nm*npoints entries and cannot alias inputs.
    ctx.run(|| unsafe {
        bloch::real_shell(
            xyz[0].as_ptr(),
            xyz[1].as_ptr(),
            xyz[2].as_ptr(),
            points,
            center.as_ptr(),
            support,
            spline.breaks.as_ptr(),
            spline.coefficients.as_ptr(),
            spline.breaks.len() as i32,
            spline.cutoff,
            spline.l,
            derivative,
            ms.as_ptr(),
            rows.as_slice().as_ptr(),
            nm,
            nm as i64,
            output.as_mut_slice().as_mut_ptr(),
        )
    });
    finite(output.as_slice(), "shell result")?;
    Ok(output)
}

use num_complex::Complex64 as C64;

// `num_complex::Complex64` and the kernels' `#[repr(C)]` complex are two `f64` pairs in the same
// order; the checked entry points below take the former and hand the latter's pointer type to the
// kernels.
const _: () = {
    assert!(std::mem::size_of::<C64>() == std::mem::size_of::<Complex64>());
    assert!(std::mem::align_of::<C64>() == std::mem::align_of::<Complex64>());
};

fn c64_finite(values: &[C64], name: &str) -> Result<()> {
    if values.iter().all(|v| v.re.is_finite() && v.im.is_finite()) {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{name} must contain finite complex values"
        )))
    }
}

/// What one atom-centered shell is sampled on: the points, the lattice images that reach them and
/// the radial function. `phases` is `(k, image)` row-major, one row per k-point.
pub struct ShellImages<'a> {
    pub xyz: [&'a [f64]; 3],
    pub center: [f64; 3],
    pub images: &'a [[f64; 3]],
    pub phases: &'a [C64],
    pub support: f64,
    pub spline: &'a Spline<'a>,
    pub ms: &'a [i32],
    /// `-1` the values, `0..=2` the Cartesian derivative.
    pub derivative: i32,
}

impl ShellImages<'_> {
    /// Validates the shape of the call; returns the number of k-points and of points.
    fn check(&self) -> Result<(usize, usize)> {
        let points = self.xyz[0].len();
        let (images, nm) = (self.images.len(), self.ms.len());
        dimension(points)?;
        dimension(images)?;
        dimension(nm)?;
        if !self.support.is_finite()
            || self.support <= 0.0
            || !(-1..=2).contains(&self.derivative)
            || self.ms.iter().any(|&m| m.abs() > self.spline.l)
        {
            return Err(Error::Invalid(
                "invalid shell support, angular component or derivative axis".into(),
            ));
        }
        for axis in self.xyz {
            length(axis.len(), points, "shell coordinates")?;
            finite(axis, "shell coordinates")?;
        }
        finite(&self.center, "shell center")?;
        if self.images.iter().any(|t| t.iter().any(|x| !x.is_finite())) {
            return Err(Error::Invalid("lattice images must be finite".into()));
        }
        if self.phases.is_empty() || self.phases.len() % images != 0 {
            return Err(Error::Invalid(format!(
                "{} phases do not make whole rows of {images} images",
                self.phases.len()
            )));
        }
        c64_finite(self.phases, "k-point phases")?;
        Ok((self.phases.len() / images, points))
    }
}

/// Adds `sum_R phase(k, R) phi_m(r - center - R)` to `out[(k * m_dim + rows[m]) * points + j]`.
///
/// `rows` places each `m` of the shell; the rows must be distinct and inside `m_dim`, and `out`
/// holds `k * m_dim * points` entries. The kernel accumulates, so `out` starts at zero for a plain
/// Bloch sum. A shell owns its rows: shells never share one.
pub fn bloch_shell(
    ctx: &Context,
    call: &ShellImages<'_>,
    rows: &[i64],
    m_dim: usize,
    out: &mut [C64],
) -> Result<()> {
    let (nk, points) = call.check()?;
    length(rows.len(), call.ms.len(), "shell rows")?;
    let distinct: std::collections::BTreeSet<_> = rows.iter().collect();
    if distinct.len() != rows.len() || rows.iter().any(|&r| r < 0 || r as usize >= m_dim) {
        return Err(Error::Invalid(
            "shell rows must be distinct and inside the destination".into(),
        ));
    }
    length(out.len(), product(&[nk, m_dim, points])?, "Bloch sums")?;
    let spline = call.spline;
    let m_dim_i = i64::try_from(m_dim).map_err(|_| Error::Overflow)?;
    let points_i = i64::try_from(points).map_err(|_| Error::Overflow)?;
    // SAFETY: coordinates, images, phases, spline, angular indices and rows are checked above;
    // `out` holds `nk * m_dim * points` entries and cannot alias the read-only inputs. Distinct
    // rows make the destination ranges of the shell's components disjoint.
    ctx.run(|| unsafe {
        bloch::bloch_shell(
            call.xyz[0].as_ptr(),
            call.xyz[1].as_ptr(),
            call.xyz[2].as_ptr(),
            points_i,
            call.center.as_ptr(),
            call.images.as_ptr().cast::<f64>(),
            call.images.len() as i32,
            call.phases.as_ptr().cast::<Complex64>(),
            nk as i32,
            call.support,
            spline.breaks.as_ptr(),
            spline.coefficients.as_ptr(),
            spline.breaks.len() as i32,
            spline.cutoff,
            spline.l,
            call.derivative,
            call.ms.as_ptr(),
            rows.as_ptr(),
            call.ms.len() as i32,
            m_dim_i,
            out.as_mut_ptr().cast::<Complex64>(),
        )
    });
    c64_finite(out, "Bloch sums")
}

/// `out[image, m, p] = sum_j conj(phi_m(r_j - center - R_image)) columns[p, j]` over the points
/// inside the support, for every image of `call` (its `phases` are not read; pass the images
/// alone through [`ShellImages::phases`] as one row of ones). The k-point phases are applied by
/// the caller, which is what makes the work independent of the number of k-points.
pub fn bloch_project(
    ctx: &Context,
    call: &ShellImages<'_>,
    columns: &[C64],
) -> Result<Buffer<C64>> {
    let (_, points) = call.check()?;
    let projectors = columns.len() / points;
    length(
        columns.len(),
        product(&[projectors, points])?,
        "projector columns",
    )?;
    dimension(projectors)?;
    c64_finite(columns, "projector columns")?;
    let size = product(&[call.images.len(), call.ms.len(), projectors])?;
    let mut output = ctx.allocate::<C64>(size)?;
    // Every worker sums into a private copy of the output, joined in worker order afterwards.
    let _scratch = ctx.reserve(product(&[ctx.threads(), size, 16])?)?;
    let spline = call.spline;
    let points_i = i64::try_from(points).map_err(|_| Error::Overflow)?;
    // SAFETY: coordinates, images, spline, angular indices and the `P * points` columns are
    // checked; the output holds `images * nm * P` entries (the kernel zeroes it first) and cannot
    // alias the inputs.
    ctx.run(|| unsafe {
        bloch::bloch_project(
            call.xyz[0].as_ptr(),
            call.xyz[1].as_ptr(),
            call.xyz[2].as_ptr(),
            points_i,
            call.center.as_ptr(),
            call.images.as_ptr().cast::<f64>(),
            call.images.len() as i32,
            call.support,
            spline.breaks.as_ptr(),
            spline.coefficients.as_ptr(),
            spline.breaks.len() as i32,
            spline.cutoff,
            spline.l,
            call.derivative,
            call.ms.as_ptr(),
            call.ms.len() as i32,
            columns.as_ptr().cast::<Complex64>(),
            projectors as i32,
            output.as_mut_slice().as_mut_ptr().cast::<Complex64>(),
        )
    });
    c64_finite(output.as_slice(), "Bloch projections")?;
    Ok(output)
}

/// [`bloch_shell`] with the derivative of every image multiplied by one component of that image's
/// own argument: adds `sum_R phase(k, R) (d_axis phi_m)(y) y_weight`, `y = r - center - R`, to
/// `out[(k * m_dim + rows[m]) * points + j]`, with `axis = call.derivative` in `0..=2` and
/// `weight` in `0..=2`. This is the strain tangent `d chi / d eps_{axis, weight}` of a Bloch sum
/// for nodes and image centers that move with the cell; the weight sits inside the image sum, so
/// it cannot be built from the summed derivatives. The checks are those of [`bloch_shell`] plus
/// the two bounds; `out` accumulates.
pub fn bloch_shell_weighted(
    ctx: &Context,
    call: &ShellImages<'_>,
    weight: usize,
    rows: &[i64],
    m_dim: usize,
    out: &mut [C64],
) -> Result<()> {
    if weight > 2 || !(0..=2).contains(&call.derivative) {
        return Err(Error::Invalid(
            "a weighted Bloch sum needs a derivative axis and a weight component in 0..=2".into(),
        ));
    }
    let (nk, points) = call.check()?;
    length(rows.len(), call.ms.len(), "shell rows")?;
    let distinct: std::collections::BTreeSet<_> = rows.iter().collect();
    if distinct.len() != rows.len() || rows.iter().any(|&r| r < 0 || r as usize >= m_dim) {
        return Err(Error::Invalid(
            "shell rows must be distinct and inside the destination".into(),
        ));
    }
    length(out.len(), product(&[nk, m_dim, points])?, "Bloch sums")?;
    let spline = call.spline;
    let m_dim_i = i64::try_from(m_dim).map_err(|_| Error::Overflow)?;
    let points_i = i64::try_from(points).map_err(|_| Error::Overflow)?;
    // SAFETY: the same checks as `bloch_shell` (coordinates, images, phases, spline, angular
    // indices, distinct rows inside `m_dim`, `out` of `nk * m_dim * points` entries that cannot
    // alias the inputs); the weight is a bounded by-value scalar, read through no pointer.
    ctx.run(|| unsafe {
        bloch::bloch_shell_weighted(
            call.xyz[0].as_ptr(),
            call.xyz[1].as_ptr(),
            call.xyz[2].as_ptr(),
            points_i,
            call.center.as_ptr(),
            call.images.as_ptr().cast::<f64>(),
            call.images.len() as i32,
            call.phases.as_ptr().cast::<Complex64>(),
            nk as i32,
            call.support,
            spline.breaks.as_ptr(),
            spline.coefficients.as_ptr(),
            spline.breaks.len() as i32,
            spline.cutoff,
            spline.l,
            call.derivative,
            weight as i32,
            call.ms.as_ptr(),
            rows.as_ptr(),
            call.ms.len() as i32,
            m_dim_i,
            out.as_mut_ptr().cast::<Complex64>(),
        )
    });
    c64_finite(out, "Bloch sums")
}
