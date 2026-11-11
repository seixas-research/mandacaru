//! Orbital occupations of a molecular Kohn-Sham channel.
//!
//! Fermi-Dirac occupations with one Fermi level per spin channel (owner decision 2026-11-10,
//! "Fractional occupation (MP-8)"; default D-14.9, "fermionic ensemble, one mu per spin"). A
//! gapped molecule is unaffected: at the default width a gap above about 80 widths leaves every
//! occupation exactly 0 or 1, so the density keeps the determinant's arithmetic. A degenerate shell at the Fermi level is shared equally by its members
//! (spin-restricted O2's pi*, the OH radical's beta pi on the axis), the zero-temperature
//! limit of the ensemble. A shell split by a geometry (OH with its hydrogen off the axis) takes
//! the fractions at which its levels meet: the integer state there has no aufbau solution, its
//! doubly occupied member rising above the empty one.
//!
//! The state minimizes the Mermin free energy `F = E - width S`; the analytic forces are `-dF/dR`
//! at fixed occupations. For discrete levels the Kohn-Sham energy `E` of the fractional state
//! is the zero-temperature estimate (its error is second order in the width, as `E` is
//! stationary in the occupations at the zero-width point), while `F` is first order.
use crate::periodic::spectrum::{fermi, Smearing};
use crate::quantum::C64;
use crate::Result;
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// Default Fermi-Dirac width of a molecular Kohn-Sham SCF, in Hartree (2.7 meV): a molecule whose
/// gap exceeds about 80 widths (0.008 Ha) has occupations exactly 0 or 1.
pub const DEFAULT_WIDTH: f64 = 1e-4;

/// Hartree within which neighboring levels form one degenerate shell and share one occupation:
/// a symmetry degeneracy split only by rounding, the tolerance of the aufbau test.
pub const DEGENERACY_WINDOW: f64 = super::scf::AUFBAU_TOLERANCE;

/// The width (Hartree) at which an annealed SCF starts, about 3 mHa (0.08 eV): wide enough that
/// DIIS converges a degenerate frontier shell from the core guess (restricted O2, OH).
pub const ANNEALING_START: f64 = 3e-3;

/// Fermi-Dirac occupations (0 to 1) of `levels` (any order) holding `n` electrons of one spin,
/// at `width` Hartree, with the electron count exact: the bisection's last 1e-13 Ha of the
/// Fermi level is distributed over the fractional levels in proportion to `f (1 - f)`.
pub fn fermi_dirac(levels: &[f64], n: usize, width: f64) -> Result<Vec<f64>> {
    let size = levels.len();
    if n > size || !width.is_finite() || width <= 0. {
        return Err(super::invalid(
            "invalid occupation count or Fermi-Dirac width",
        ));
    }
    if n == 0 || n == size {
        return Ok(vec![if n == 0 { 0. } else { 1. }; size]);
    }
    let fill = |mu: f64| -> Vec<f64> {
        levels
            .iter()
            .map(|e| Smearing::FermiDirac.occupation((e - mu) / width))
            .collect()
    };
    // A gapped spectrum: the Fermi level of the exact solution is the middle of the gap (equal
    // tails), where a gap above ~80 widths saturates every occupation to exactly 0 or 1. The
    // bisection alone stops where the count first reaches `n` in floating point, a few widths
    // from the highest occupied level.
    let mut sorted = levels.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut f = fill(0.5 * (sorted[n - 1] + sorted[n]));
    if (f.iter().sum::<f64>() - n as f64).abs() > 1e-12 {
        let mu = fermi(
            &[levels.to_vec()],
            &[1.],
            n as f64,
            Smearing::FermiDirac,
            width,
            1.,
        )?;
        f = fill(mu);
    }
    let deficit = n as f64 - f.iter().sum::<f64>();
    let softness: Vec<f64> = f.iter().map(|f| f * (1. - f)).collect();
    let total: f64 = softness.iter().sum();
    if deficit != 0. && total > 0. {
        for (f, s) in f.iter_mut().zip(&softness) {
            *f = (*f + deficit * s / total).clamp(0., 1.);
        }
    }
    // Levels within DEGENERACY_WINDOW of their neighbor are one degenerate shell, to which
    // Fermi-Dirac gives one occupation; averaging it removes the rounding-level splitting that
    // the SCF would otherwise amplify by about U / (4 width) per iteration and that breaks the
    // symmetry of the shell (restricted O2's pi*, OH's beta pi on the axis).
    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|&a, &b| levels[a].total_cmp(&levels[b]));
    let mut first = 0;
    while first < size {
        let mut last = first;
        while last + 1 < size && levels[order[last + 1]] - levels[order[last]] < DEGENERACY_WINDOW {
            last += 1;
        }
        if last > first {
            let shell = &order[first..=last];
            let mean = shell.iter().map(|&j| f[j]).sum::<f64>() / shell.len() as f64;
            for &j in shell {
                f[j] = mean;
            }
        }
        first = last + 1;
    }
    Ok(f)
}

/// The Fermi-Dirac entropy `sum_i -f_i ln f_i - (1 - f_i) ln(1 - f_i)` of one spin channel.
pub fn entropy(occupations: &[f64]) -> f64 {
    let term = |x: f64| if x > 0. { x * x.ln() } else { 0. };
    occupations.iter().map(|&f| -term(f) - term(1. - f)).sum()
}

/// Aufbau occupations of `size` orbitals holding `n` electrons of one spin.
pub fn aufbau(size: usize, n: usize) -> Vec<f64> {
    (0..size).map(|j| if j < n { 1. } else { 0. }).collect()
}

/// Whether every occupation is 0 or 1 (a determinant, not an ensemble).
pub fn is_integer(occupations: &[f64]) -> bool {
    occupations.iter().all(|&f| f == 0. || f == 1.)
}

/// `sum_j f_j c_j c_j^+`: the density of orbitals `c` (columns) with occupations `f`.
pub fn weighted_density(c: &Matrix, occupations: &[f64]) -> Matrix {
    let mut out = Matrix::zeros(c.nrows(), c.nrows());
    for (j, &f) in occupations.iter().enumerate() {
        if f != 0. {
            let column = c.column(j);
            out += column * column.adjoint() * C64::new(f, 0.);
        }
    }
    out
}

/// The density of orbitals `c` with `occupations`: an aufbau prefix keeps the determinant's
/// arithmetic `C_occ C_occ^+`, an ensemble sums its weighted columns.
pub fn density(c: &Matrix, occupations: &[f64]) -> Matrix {
    let n = occupations.iter().take_while(|&&f| f == 1.).count();
    if occupations[n..].iter().all(|&f| f == 0.) {
        let occupied = c.columns(0, n);
        occupied * occupied.adjoint()
    } else {
        weighted_density(c, occupations)
    }
}

/// The (more occupied, less occupied) pair of an occupation that is not aufbau: a level that
/// holds more than another (by more than 1e-9 of an electron) while lying above it by more
/// than `tolerance` Hartree, the pair with the largest such inversion. Fermi-Dirac occupations
/// decrease with the level and never form one; an integer state held by a level shift can.
pub fn non_aufbau(levels: &[f64], occupations: &[f64], tolerance: f64) -> Option<(usize, usize)> {
    let mut worst: Option<(usize, usize, f64)> = None;
    for i in 0..levels.len() {
        for a in 0..levels.len() {
            let inversion = levels[i] - levels[a];
            if occupations[i] > occupations[a] + 1e-9
                && inversion > tolerance
                && worst.is_none_or(|w| inversion > w.2)
            {
                worst = Some((i, a, inversion));
            }
        }
    }
    worst.map(|(i, a, _)| (i, a))
}
