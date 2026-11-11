//! Values the run log and the ASE results derive from a calculation's numbers: force norms and
//! their summary, the rows of the ADAPT iteration table, a cell's lengths and angles.
use crate::{math::numpy_rules, Error, Result};

/// The norms of the per-atom forces and their summary, as the `[FORCES]` block and the
/// optimization history report them.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceSummary {
    /// `|F_a|` of each atom.
    pub norms: Vec<f64>,
    /// The largest `|F_a|`.
    pub max: f64,
    /// `sqrt(mean |F_a|^2)`.
    pub rms: f64,
    /// The largest Cartesian component of `|sum_a F_a|`: zero for a free molecule up to the
    /// discretization, so it measures how usable the forces are.
    pub net: f64,
}

/// [`ForceSummary`] of `forces`, one `[x, y, z]` per atom (at least one atom).
pub fn force_summary(forces: &[[f64; 3]]) -> Result<ForceSummary> {
    if forces.is_empty() || forces.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("forces must be finite, one vector per atom".into()));
    }
    let norms: Vec<f64> = forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).collect();
    let max = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rms = (norms.iter().map(|n| n * n).sum::<f64>() / norms.len() as f64).sqrt();
    let total = forces.iter().fold([0.; 3], |acc, f| [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]);
    let net = total.iter().map(|v| v.abs()).fold(0., f64::max);
    Ok(ForceSummary { norms, max, rms, net })
}

/// What one row of the ADAPT iteration table reports beside the energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationValues {
    /// `energy - previous`, or `None` on the first row.
    pub delta: Option<f64>,
    /// The selected operator's `|gradient|`, in eV (the table's display unit).
    pub gradient_ev: f64,
    /// Whether `|gradient|` (Hartree) is strictly below the gradient threshold; `None` without one.
    pub gradient_met: Option<bool>,
    /// Whether `|delta|` is strictly below the energy threshold (in the energy's unit); `None`
    /// without one, and `false` on the first row (a dash never meets it).
    pub energy_met: Option<bool>,
}

/// The row values the solver's own convergence tests decide: the same strict comparisons.
pub fn iteration_values(
    energy: f64,
    previous: Option<f64>,
    gradient_ha: f64,
    gradient_threshold: Option<f64>,
    energy_threshold: Option<f64>,
) -> Result<IterationValues> {
    if !energy.is_finite() || !gradient_ha.is_finite() || previous.is_some_and(|p| !p.is_finite()) {
        return Err(Error::Invalid("iteration energies and gradients must be finite".into()));
    }
    let delta = previous.map(|p| energy - p);
    Ok(IterationValues {
        delta,
        gradient_ev: gradient_ha.abs() * crate::units::HARTREE_TO_EV,
        gradient_met: gradient_threshold.map(|t| gradient_ha.abs() < t),
        energy_met: energy_threshold.map(|t| delta.is_some_and(|d| d.abs() < t)),
    })
}

/// `(a, b, c, alpha, beta, gamma)` of a cell whose rows are its vectors: lengths in the cell's
/// units, angles in degrees (alpha between b and c, beta between a and c, gamma between a and
/// b), zero for an angle with a zero vector.
pub fn cell_parameters(cell: [[f64; 3]; 3]) -> Result<[f64; 6]> {
    if cell.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("the cell must be finite".into()));
    }
    let norm = |u: &[f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let angle = |u: &[f64; 3], v: &[f64; 3]| {
        let (nu, nv) = (norm(u), norm(v));
        if nu == 0. || nv == 0. {
            return 0.;
        }
        let cosine = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)).clamp(-1., 1.);
        cosine.acos().to_degrees()
    };
    let [a, b, c] = &cell;
    Ok([norm(a), norm(b), norm(c), angle(b, c), angle(a, c), angle(a, b)])
}

/// `values[i] - values[0]`: the energies of excited states above the first.
pub fn relative_to_first(values: &[f64]) -> Vec<f64> {
    let first = values.first().copied().unwrap_or(0.);
    values.iter().map(|v| v - first).collect()
}

/// `(a - b)` in user units: the correlation energy of a result, its energy above its reference.
pub fn difference_to_user(a: f64, b: f64, quantity: crate::units::Quantity, atomic: bool) -> Result<f64> {
    Ok(crate::units::to_user(&[a - b], quantity, atomic, 1)?[0])
}

/// The electrons of each spin of a crystal: `sum_k w_k sum_n f_snk` per spin channel, from the
/// occupations `(spins, kpoints, bands)` (row-major) and the k-point weights. A restricted
/// (one-channel) crystal splits its total evenly between the two spins.
pub fn electron_counts(occupations: &[f64], spins: usize, kpoints: usize, bands: usize, weights: &[f64]) -> Result<[f64; 2]> {
    if !(1..=2).contains(&spins) || weights.len() != kpoints || occupations.len() != spins * kpoints * bands {
        return Err(Error::Invalid("occupations must be (spins, kpoints, bands) with one weight per k-point".into()));
    }
    let count = |s: usize| -> f64 {
        (0..kpoints)
            .map(|k| weights[k] * occupations[(s * kpoints + k) * bands..(s * kpoints + k + 1) * bands].iter().sum::<f64>())
            .sum()
    };
    Ok(if spins == 2 { [count(0), count(1)] } else { let half = count(0) / 2.; [half, half] })
}

/// The sum of `values` as Python's `sum` of floats computes it (a run's wall times, for one): the
/// compensated [`python_sum`], which differs from an ordered sum in the last bit for three or more terms.
pub fn total(values: &[f64]) -> f64 {
    python_sum(values)
}

/// The fractional coordinates of a Gamma-centered `shape` mesh, `[i / n0, j / n1, k / n2]` with `k`
/// running fastest (the order of the Wannier setup's points). Each entry is the correctly rounded
/// quotient of two exactly representable integers, as Python's `i / n` is.
pub fn mesh_fractions(shape: [usize; 3]) -> Result<Vec<[f64; 3]>> {
    if shape.iter().any(|&n| n == 0) {
        return Err(Error::Invalid("every mesh dimension must be positive".into()));
    }
    let mut points = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                points.push([
                    i as f64 / shape[0] as f64,
                    j as f64 / shape[1] as f64,
                    k as f64 / shape[2] as f64,
                ]);
            }
        }
    }
    Ok(points)
}

/// The Euclidean norm of `vector`.
pub fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The distance between two points of the same dimension.
pub fn distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::Invalid("the two points must have the same dimension".into()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt())
}

/// `values[i + 1] - values[i]`: the gaps between consecutive levels.
pub fn gaps(values: &[f64]) -> Vec<f64> {
    values.windows(2).map(|w| w[1] - w[0]).collect()
}

/// `n` equal weights `1 / n`: a uniform k-point mesh.
pub fn uniform_weights(n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::Invalid("a mesh holds at least one k-point".into()));
    }
    Ok(vec![1.0 / n as f64; n])
}

/// The highest occupied level over the spin channels: `levels[s * bands + particles[s] - 1]`
/// of each channel holding electrons (`levels` row-major `(spins, bands)`, Hartree).
pub fn highest_occupied(levels: &[f64], bands: usize, particles: &[usize]) -> Result<f64> {
    if levels.len() != particles.len() * bands || particles.iter().any(|&p| p > bands) {
        return Err(Error::Invalid("levels must be (spins, bands) with at most one electron per level".into()));
    }
    particles
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0)
        .map(|(s, &p)| levels[s * bands + p - 1])
        .fold(None, |best: Option<f64>, v| Some(best.map_or(v, |b| b.max(v))))
        .ok_or_else(|| Error::Invalid("an empty molecule has no occupied Fermi level".into()))
}

/// `npoints` energies (eV) evenly spanning the levels (Hartree) widened by `scale * margin` (eV)
/// on each side (eight Gaussian widths, for one), formed as an even spacing is
/// (`start + i * step`, the last point exactly the end).
pub fn energy_axis(levels_ha: &[f64], npoints: usize, margin: f64, scale: f64) -> Result<Vec<f64>> {
    let margin = scale * margin;
    if levels_ha.is_empty() || npoints == 0 || !margin.is_finite() {
        return Err(Error::Invalid("an energy axis needs levels, points and a finite margin".into()));
    }
    let low = levels_ha.iter().cloned().fold(f64::INFINITY, f64::min) * crate::units::HARTREE_TO_EV;
    let high = levels_ha.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * crate::units::HARTREE_TO_EV;
    let (start, stop) = (low - margin, high + margin);
    if npoints == 1 {
        return Ok(vec![start]);
    }
    let step = (stop - start) / (npoints - 1) as f64;
    let mut axis: Vec<f64> = (0..npoints).map(|i| i as f64 * step + start).collect();
    axis[npoints - 1] = stop;
    Ok(axis)
}

/// The columns of `values` (row-major `(rows, labels.len())`) summed per label, the labels in
/// their first-occurrence order: the shells of a projected density of states.
pub fn column_groups(values: &[f64], rows: usize, labels: &[String]) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let cols = labels.len();
    if values.len() != rows * cols {
        return Err(Error::Invalid("values must be (rows, labels)".into()));
    }
    let mut names: Vec<String> = Vec::new();
    let mut sums: Vec<Vec<f64>> = Vec::new();
    for (column, label) in labels.iter().enumerate() {
        let at = match names.iter().position(|n| n == label) {
            Some(at) => at,
            None => {
                names.push(label.clone());
                sums.push(vec![0.; rows]);
                names.len() - 1
            }
        };
        for row in 0..rows {
            sums[at][row] += values[row * cols + column];
        }
    }
    Ok((names, sums))
}

/// `positions` with atom `index` moved by `sign * step` along `axis`.
pub fn displaced(mut positions: Vec<[f64; 3]>, index: usize, axis: usize, sign: f64, step: f64) -> Result<Vec<[f64; 3]>> {
    if index >= positions.len() || axis >= 3 {
        return Err(Error::Invalid("the displaced atom and axis must exist".into()));
    }
    positions[index][axis] += sign * step;
    Ok(positions)
}

/// `a / b`, refused for a zero `b`: a total per cell or per electron.
pub fn ratio(a: f64, b: f64) -> Result<f64> {
    if b == 0. || !b.is_finite() {
        return Err(Error::Invalid("the divisor must be finite and nonzero".into()));
    }
    Ok(a / b)
}

fn determinant(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// `(voxel volume, cell, cell volume)` of a grid of `shape` points with step matrix `step`: the
/// cell is the step matrix with each column `b` scaled by `shape[b]`, the volumes the absolute
/// determinants. A degenerate step is refused.
pub fn grid_cell(step: [[f64; 3]; 3], shape: [usize; 3]) -> Result<(f64, [[f64; 3]; 3], f64)> {
    if step.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("the grid step must be finite".into()));
    }
    let dv = determinant(&step).abs();
    if dv <= 0. {
        return Err(Error::Invalid("the voxel has zero volume (degenerate step)".into()));
    }
    let cell = std::array::from_fn(|a| std::array::from_fn(|b| step[a][b] * shape[b] as f64));
    Ok((dv, cell, determinant(&cell).abs()))
}

/// The Cartesian points of a grid, `(3, nx * ny * nz)` row-major: point `(i, j, k)` (flattened
/// as `(i * ny + j) * nz + k`) at `origin + step . (i, j, k)`.
pub fn grid_coordinates(origin: [f64; 3], step: [[f64; 3]; 3], shape: [usize; 3]) -> Vec<f64> {
    let n = shape[0] * shape[1] * shape[2];
    let mut out = vec![0.; 3 * n];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let p = (i * shape[1] + j) * shape[2] + k;
                let index = [i as f64, j as f64, k as f64];
                for a in 0..3 {
                    let offset: f64 = (0..3).map(|b| step[a][b] * index[b]).sum();
                    out[a * n + p] = origin[a] + offset;
                }
            }
        }
    }
    out
}

/// The decimal digits a tolerance resolves, `max(0, round(-log10(atol)))` with ties to even.
pub fn decimal_digits(atol: f64) -> Result<u32> {
    if !atol.is_finite() || atol <= 0. {
        return Err(Error::Invalid("the tolerance must be finite and positive".into()));
    }
    Ok((-atol.log10()).round_ties_even().max(0.) as u32)
}

/// `values` rounded to `decimals` places as numpy does: `rint(x * 10^d) / 10^d`, ties to even.
pub fn rounded(values: &[f64], decimals: i32) -> Vec<f64> {
    let scale = 10f64.powi(decimals);
    values.iter().map(|v| (v * scale).round_ties_even() / scale).collect()
}

/// The wavevector (1/Bohr) of a free electron of kinetic energy `energy_ev`:
/// `sqrt(2 * energy_ev * EV_TO_HARTREE)`, as `(2 E) * (1/H)` under the root.
pub fn free_wavevector(energy_ev: f64) -> Result<f64> {
    if !(energy_ev.is_finite() && energy_ev > 0.) {
        return Err(Error::Invalid("the kinetic energy must be finite and positive (eV)".into()));
    }
    Ok((2.0 * energy_ev * crate::units::EV_TO_HARTREE).sqrt())
}

/// The kinetic energy (eV) of a free electron of wavevector `k` (1/Bohr): `0.5 k^2 / EV_TO_HARTREE`.
pub fn free_energy_ev(k: f64) -> Result<f64> {
    if !k.is_finite() {
        return Err(Error::Invalid("the wavevector must be finite".into()));
    }
    Ok(0.5 * k * k / crate::units::EV_TO_HARTREE)
}

/// The confinement radius (Bohr) of an energy shift (eV): half a free wavelength at that energy,
/// `pi / free_wavevector(shift)`.
pub fn energy_shift_radius(shift_ev: f64) -> Result<f64> {
    Ok(std::f64::consts::PI / free_wavevector(shift_ev)?)
}

/// A fraction of the grid's Nyquist wavevector, `fraction * pi / spacing` (1/Bohr).
pub fn nyquist_wavevector(fraction: f64, spacing_bohr: f64) -> Result<f64> {
    if !(spacing_bohr.is_finite() && spacing_bohr > 0.) {
        return Err(Error::Invalid("the grid spacing must be finite and positive".into()));
    }
    if !(fraction.is_finite() && fraction > 0.) {
        return Err(Error::Invalid("the Nyquist fraction must be finite and positive".into()));
    }
    Ok(fraction * std::f64::consts::PI / spacing_bohr)
}

/// The weights `n, n - 1, ..., 1`: the default weights of `n` states in a subspace search.
pub fn descending_weights(n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::Invalid("a subspace holds at least one state".into()));
    }
    Ok((0..n).map(|i| (n - i) as f64).collect())
}

/// `max |v|` (NaN when a value is NaN), zero for no values.
pub fn max_abs(values: &[f64]) -> f64 {
    let magnitudes: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    numpy_rules::maximum(&magnitudes).unwrap_or(0.)
}

/// The times `i dt` of `n` uniform samples.
pub fn sample_times(n: usize, dt: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * dt).collect()
}

/// The spacing `2 pi / (n dt)` of the Fourier bins of `n` samples a time step `dt` apart.
pub fn fourier_bin_spacing(samples: usize, dt: f64) -> f64 {
    2. * std::f64::consts::PI / (samples as f64 * dt)
}

/// The Nyquist frequency `pi / dt` of samples a time step `dt` apart.
pub fn nyquist_frequency(dt: f64) -> f64 {
    std::f64::consts::PI / dt
}

/// `np.abs` of complex128 values (see [`numpy_rules::complex_abs`]).
pub fn magnitudes(values: &[crate::quantum::C64]) -> Vec<f64> {
    values.iter().map(|&z| numpy_rules::complex_abs(z)).collect()
}

/// `x ** 2` as Python evaluates it for a float, through the C library's `pow`: it can differ
/// from `x * x` in the last bit.
pub fn squared(x: f64) -> f64 {
    x.powf(std::hint::black_box(2.))
}

/// `abs(z) ** 2` of a Python complex (`hypot`, then [`squared`]): the fidelity of an amplitude.
pub fn modulus_squared(re: f64, im: f64) -> f64 {
    squared(re.hypot(im))
}

/// `|tau| * norm_bound`: the strength bound of a pulse of signed strength `tau`.
pub fn pulse_bound(tau: f64, norm_bound: f64) -> f64 {
    tau.abs() * norm_bound
}

/// Python's `sum` of floats, added in order with Neumaier's compensation (the algorithm of
/// CPython 3.12 and later), not the running sum: the norm bound of a Pauli sum, for one.
pub fn python_sum(values: &[f64]) -> f64 {
    let (mut total, mut compensation) = (0.0f64, 0.0f64);
    for &x in values {
        let t = total + x;
        compensation += if total.abs() >= x.abs() { (total - t) + x } else { (x - t) + total };
        total = t;
    }
    if compensation != 0. && compensation.is_finite() {
        total += compensation;
    }
    total
}
