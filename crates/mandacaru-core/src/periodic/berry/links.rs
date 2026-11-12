//! Model-free algebra of Bloch-state links `M(k, k') = <u_k|u_k'>` between band groups.
//!
//! Nothing here knows where a link came from: a crystal's PAW overlaps and a lattice model's
//! eigenvectors go through the same functions, which is how the tests pin the conventions.
//!
//! Conventions (`A = i <u|grad_k u>`, so `arg det M(k, k + dk) = -A.dk` to first order):
//! - Berry phase of a closed string: `gamma = -sum_s arg det U(k_s, k_{s+1})`, in (-pi, pi].
//! - Wilson loop `W = prod_s U(k_s, k_{s+1})` in string order; its eigenvalues `e^{i theta_n}`
//!   ([`unitary_spectrum`])
//!   give the Wannier charge centers `x_n = -theta_n / 2 pi` in [0, 1), summing to
//!   `gamma / 2 pi` modulo 1.
//! - Plaquette flux `phi = -arg[u_a(k) u_b(k + a) conj(u_a(k + b)) conj(u_b(k))]` of the link
//!   determinants, in (-pi, pi]: the lattice field strength of Fukui, Hatsugai and Suzuki, whose
//!   sum over a closed plane is `2 pi C`.
//!
//! `U` is the polar (unitary) part of a link, `U V^+` of its SVD `M = U S V^+`: it has the phase of
//! `det M` and unit modulus, so a long product neither underflows nor overflows.
use crate::math::linalg::eigen;
use crate::{periodic::invalid, quantum::C64, Result};
use nalgebra::DMatrix;
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

/// `angle` on (-pi, pi].
pub fn wrap(angle: f64) -> f64 {
    PI - (PI - angle).rem_euclid(2. * PI)
}

fn check(link: &Matrix) -> Result<()> {
    if link.nrows() == 0
        || link.nrows() != link.ncols()
        || link.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(invalid(
            "a Berry link must be a nonempty square matrix of finite values",
        ));
    }
    Ok(())
}

fn tolerance_ok(tolerance: f64) -> Result<()> {
    if !tolerance.is_finite() || tolerance <= 0. || tolerance >= 1. {
        return Err(invalid("a link tolerance must lie in (0, 1)"));
    }
    Ok(())
}

/// The polar part `U V^+` of `link` and its smallest singular value.
pub fn unitarize(link: &Matrix) -> Result<(Matrix, f64)> {
    check(link)?;
    let svd = link.clone().svd(true, true);
    let smallest = svd.singular_values.min();
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Err(invalid("the singular value decomposition of a link failed"));
    };
    Ok((u * v_t, smallest))
}

/// `det U` of the polar part of `link` (unit modulus) and the smallest singular value of `link`;
/// a link below `tolerance` raises: the overlap of the two band spaces is (nearly) singular, so its
/// phase is undefined.
pub fn link_phase(link: &Matrix, tolerance: f64) -> Result<(C64, f64)> {
    tolerance_ok(tolerance)?;
    let (unitary, smallest) = unitarize(link)?;
    if !smallest.is_finite() || smallest < tolerance {
        return Err(invalid(format!(
            "singular occupied-space link (smallest singular value {smallest:.3e}): the Berry \
             phase is undefined; refine the k mesh or check that the bands are isolated"
        )));
    }
    let det = unitary.lu().determinant();
    Ok((det / det.norm(), smallest))
}

fn same_size(links: &[Matrix]) -> Result<usize> {
    let n = links
        .first()
        .ok_or_else(|| invalid("a closed string needs at least one link"))?
        .nrows();
    for link in links {
        check(link)?;
        if link.nrows() != n {
            return Err(invalid(
                "the links of one string must share their band count",
            ));
        }
    }
    Ok(n)
}

/// Berry phase `-sum arg det U` of a closed string of links, on (-pi, pi], and the smallest
/// singular value of its links.
pub fn string_phase(links: &[Matrix], tolerance: f64) -> Result<(f64, f64)> {
    same_size(links)?;
    let mut phase = 0.;
    let mut smallest = f64::INFINITY;
    for link in links {
        let (det, s) = link_phase(link, tolerance)?;
        phase -= det.arg();
        smallest = smallest.min(s);
    }
    Ok((wrap(phase), smallest))
}

/// The Wannier charge centers `-theta_n / 2 pi` in [0, 1), ascending, of the Wilson loop of a
/// closed string: the eigenphases of the product of the unitarized links, in string order.
pub fn wannier_centers(links: &[Matrix], tolerance: f64) -> Result<Vec<f64>> {
    let n = same_size(links)?;
    let mut loop_matrix = Matrix::identity(n, n);
    for link in links {
        link_phase(link, tolerance)?;
        loop_matrix *= unitarize(link)?.0;
    }
    let values = unitary_spectrum(&loop_matrix)?;
    let mut centers: Vec<f64> = values
        .iter()
        .map(|v| (-v.arg() / (2. * PI)).rem_euclid(1.))
        .map(|x| if x >= 1. { 0. } else { x })
        .collect();
    centers.sort_by(f64::total_cmp);
    Ok(centers)
}

/// The eigenvalues of a unitary matrix by Hermitian eigensolves only: `W` is normal, so
/// `H = (W + W^+)/2` and `K = (W - W^+)/(2i)` commute. The eigenvectors of `H + alpha K` (alpha
/// irrational) are eigenvectors of `W` except inside a cluster of equal `H + alpha K` eigenvalues,
/// which the `K` block of that cluster splits. Each value is `v^+ W v`, checked for unit modulus,
/// and the spectrum against `tr W` and `det W`. (A complex Schur form need not converge on the
/// exactly degenerate spectra of flat bands.)
pub fn unitary_spectrum(w: &Matrix) -> Result<Vec<C64>> {
    check(w)?;
    let n = w.nrows();
    if (w.adjoint() * w - Matrix::identity(n, n)).norm() > 1e-8 * n as f64 {
        return Err(invalid("a Wilson loop must be unitary"));
    }
    let h = (w + w.adjoint()) * C64::new(0.5, 0.);
    let k = (w - w.adjoint()) * C64::new(0., -0.5);
    let alpha = C64::new(0.577_215_664_901_532_9, 0.);
    let (values, vectors) = eigen(&h + &k * alpha)?;
    let mut basis = Matrix::zeros(n, n);
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[end] - values[end - 1] < 1e-9 {
            end += 1;
        }
        let block = vectors.columns(start, end - start).into_owned();
        let m = end - start;
        let projected = block.adjoint() * &k * &block;
        let mean = projected.trace() / C64::new(m as f64, 0.);
        // A scalar block is a degenerate eigenvalue of W: every vector of the cluster is an
        // eigenvector, and a split would only diagonalize rounding noise.
        let anisotropic = (&projected - Matrix::identity(m, m) * mean).norm();
        let split = if m > 1 && anisotropic > 1e-10 {
            let (_, rotation) = eigen(projected)?;
            &block * rotation
        } else {
            block
        };
        basis.columns_mut(start, end - start).copy_from(&split);
        start = end;
    }
    let spectrum: Vec<C64> = (0..n)
        .map(|i| (basis.column(i).adjoint() * w * basis.column(i))[(0, 0)])
        .collect();
    let sum = spectrum.iter().fold(C64::default(), |p, v| p + v);
    let product = spectrum.iter().fold(C64::new(1., 0.), |p, v| p * v);
    let det = w.clone().lu().determinant();
    if spectrum.iter().any(|v| (v.norm() - 1.).abs() > 1e-8)
        || (sum - w.trace()).norm() > 1e-8 * n as f64
        || wrap(product.arg() - det.arg()).abs() > 1e-8
    {
        return Err(invalid(
            "the Wilson-loop eigenvalues are not a unit-modulus spectrum of its trace and \
             determinant",
        ));
    }
    Ok(spectrum)
}

/// The flux `-arg[u_a(k) u_b(k + a) conj(u_a(k + b)) conj(u_b(k))]` of one plaquette, on
/// (-pi, pi], from the unit-modulus link determinants around it.
pub fn plaquette_flux(a: C64, b_shifted: C64, a_shifted: C64, b: C64) -> f64 {
    -(a * b_shifted * a_shifted.conj() * b.conj()).arg()
}

/// A plaquette flux above this (radians) raises: near pi the lattice field aliases and the Chern
/// number would be a wrong integer without any sign of it.
pub const ALIASING: f64 = 0.75 * PI;

/// The lattice field strength of a periodic plane of link determinants.
pub struct Field {
    /// `flux[i * n_b + j]` of the plaquette at `(i, j)`, radians.
    pub flux: Vec<f64>,
    /// `(1/2 pi) sum flux`.
    pub chern: i64,
    /// The largest `|flux|`.
    pub max_flux: f64,
}

/// The plaquette fluxes and the Chern number of the periodic `n_a x n_b` plane whose unit-modulus
/// link determinants along its two directions are `along_a[i * n_b + j]` and `along_b[...]`.
/// A flux beyond [`ALIASING`] raises.
pub fn lattice_field(along_a: &[C64], along_b: &[C64], size: [usize; 2]) -> Result<Field> {
    let [na, nb] = size;
    let count = na
        .checked_mul(nb)
        .ok_or_else(|| invalid("the k plane is too large"))?;
    if na < 2
        || nb < 2
        || along_a.len() != count
        || along_b.len() != count
        || along_a
            .iter()
            .chain(along_b)
            .any(|v| !v.re.is_finite() || !v.im.is_finite() || (v.norm() - 1.).abs() > 1e-8)
    {
        return Err(invalid(
            "a lattice field needs unit-modulus link determinants on at least a 2 x 2 plane",
        ));
    }
    let u = |along: &[C64], i: usize, j: usize| along[(i % na) * nb + j % nb];
    let mut flux = Vec::with_capacity(count);
    let mut total = 0.;
    let mut max_flux = 0f64;
    for i in 0..na {
        for j in 0..nb {
            let phi = plaquette_flux(
                u(along_a, i, j),
                u(along_b, i + 1, j),
                u(along_a, i, j + 1),
                u(along_b, i, j),
            );
            max_flux = max_flux.max(phi.abs());
            total += phi;
            flux.push(phi);
        }
    }
    if max_flux > ALIASING {
        return Err(invalid(format!(
            "a Berry plaquette carries a flux of {max_flux:.3} rad, near pi, where the lattice \
             field aliases: refine the k mesh of the plane"
        )));
    }
    let chern = total / (2. * PI);
    if (chern - chern.round()).abs() > 1e-6 {
        return Err(invalid(format!(
            "the plaquette fluxes sum to {chern} flux quanta, not an integer"
        )));
    }
    Ok(Field {
        flux,
        chern: chern.round() as i64,
        max_flux,
    })
}

/// The mean of `phases` on the branch of their circular mean, and the largest distance of an
/// aligned phase from it. The mean is returned on (-pi, pi].
pub fn circular_average(phases: &[f64]) -> Result<(f64, f64)> {
    if phases.is_empty() || phases.iter().any(|v| !v.is_finite()) {
        return Err(invalid("a phase average needs finite phases"));
    }
    let sum = phases
        .iter()
        .fold(C64::default(), |s, v| s + C64::from_polar(1., *v));
    let reference = if sum.norm() > 1e-12 * phases.len() as f64 {
        sum.arg()
    } else {
        phases[0]
    };
    let aligned: Vec<f64> = phases
        .iter()
        .map(|v| reference + wrap(v - reference))
        .collect();
    let mean = aligned.iter().sum::<f64>() / aligned.len() as f64;
    let spread = aligned.iter().map(|v| (v - mean).abs()).fold(0., f64::max);
    Ok((wrap(mean), spread))
}
