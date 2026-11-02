//! Projectors and one-center matrices of one PAW channel for a given screened local potential.
//! Atomic units.
//!
//! With the smooth waves `phi~_i = sum_n c_in j_l(q_n r)` inside `r_cut`, the raw projectors
//! are `chi_i = (eps_i - T - v~) phi~_i` there, `(v_AE - v~) phi_i` in the shell between
//! `r_cut` and the radius `r_local` of the local potential (where the smooth wave *is* the
//! all-electron one), and zero beyond. `B_ij = <phi~_i|chi_j>`, the dual projectors are
//! `p~_i = sum_k (B^-1)_ki chi_k`, and `q`, `Delta T`, `Delta V` and `D^scr = B + q_norm eps`
//! follow. Every integral is Simpson's rule on the 801-node grid of the sphere (and of the
//! shell), the all-electron quantities cubic-resampled onto it.
use crate::{
    math::{radial::validate, spline::simpson},
    paw::{
        invalid,
        pseudize::PseudoWaves,
        pseudize_tables::{bessel_table, inner_grid, linspace, resample, INNER_POINTS},
    },
    product, Buffer, Context, Result,
};
use mandacaru_kernels::two_center::spherical_bessel as bessel;
use nalgebra::DMatrix;

/// Largest tolerated deviation of `<p_i|phi~_j>` from the identity.
pub const DUALITY_TOLERANCE: f64 = 1e-8;
/// Largest tolerated asymmetry of the screened coupling matrix (Hartree).
pub const COUPLING_ASYMMETRY_TOLERANCE: f64 = 1e-4;
/// Smallest tolerated eigenvalue of the overlap operator `1 + sum |p~> q <p~|` in the channel.
pub const OVERLAP_MINIMUM: f64 = 0.1;

/// The reference atom a channel is assembled in.
#[derive(Clone, Copy)]
pub struct Atomic<'a> {
    pub r: &'a [f64],
    /// The all-electron effective potential.
    pub v_ae: &'a [f64],
    /// The screened local potential of the dataset.
    pub v_loc: &'a [f64],
    /// Radius of the local potential.
    pub r_local: f64,
}

/// The partial waves of one channel: the all-electron references, their Bessel expansions and
/// the electrons each holds in the reference atom.
#[derive(Clone, Copy)]
pub struct Partial<'a> {
    pub l: usize,
    /// Principal quantum number of the first reference.
    pub n: usize,
    pub energies: &'a [f64],
    /// All-electron `R(r)` of the references, `count x len(r)` flat.
    pub waves: &'a [f64],
    pub smooth: &'a PseudoWaves,
    pub q_cut: f64,
    /// Electrons of each reference; missing entries are empty.
    pub occupations: &'a [f64],
}

/// One PAW-LCAO channel. All matrices are `count x count` over the references.
pub struct PawChannel {
    pub l: usize,
    pub n: usize,
    pub r_cut: f64,
    pub q_cut: f64,
    pub reference_energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub wavevectors: Vec<f64>,
    pub coefficients: DMatrix<f64>,
    pub residual_kinetic: Vec<f64>,
    /// `count x len(r)` flat: the smooth waves (the all-electron ones beyond `r_cut`).
    pub pseudo_waves: Buffer<f64>,
    /// The dual projectors.
    pub projectors: Buffer<f64>,
    /// The raw projectors `chi`.
    pub raw_projectors: Buffer<f64>,
    /// `q = norms - achieved`: the charge the smooth density is missing.
    pub overlap_correction: DMatrix<f64>,
    /// `q_norm = targets - achieved`: the norm the Vanderbilt condition conserved.
    pub norm_correction: DMatrix<f64>,
    pub kinetic_difference: DMatrix<f64>,
    pub potential_difference: DMatrix<f64>,
    pub coupling_screened: DMatrix<f64>,
    /// `B_ij = <phi~_i|chi_j>`, not symmetrized.
    pub vanderbilt: DMatrix<f64>,
    pub duality_error: f64,
    pub overlap_minimum: f64,
    pub asymmetry: f64,
    pub consistency_error: f64,
}

impl PawChannel {
    /// Energy of the first (bound) reference.
    pub fn eigenvalue(&self) -> f64 {
        self.reference_energies[0]
    }
    /// Electrons of the channel in the reference atom.
    pub fn occupation(&self) -> f64 {
        self.occupations.iter().sum()
    }
    /// `|q_00|`, the norm the first smooth wave gives up.
    pub fn norm_error(&self) -> f64 {
        self.overlap_correction[(0, 0)].abs()
    }
}

/// `sum_k weights_k * rows_k`, accumulated in order.
fn combine(weights: impl Iterator<Item = f64>, rows: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![0.; rows.first().map_or(0, Vec::len)];
    for (weight, row) in weights.zip(rows) {
        for (o, x) in out.iter_mut().zip(row) {
            *o += weight * x;
        }
    }
    out
}

fn symmetrize(a: &DMatrix<f64>) -> DMatrix<f64> {
    (a + a.transpose()) * 0.5
}

/// The projectors and one-center matrices of the channel `set` in `atom`. With `strict` a
/// duality error, an asymmetry or an overlap eigenvalue beyond its tolerance (or a NaN) raises.
pub fn assemble(
    ctx: &Context,
    atom: &Atomic<'_>,
    set: &Partial<'_>,
    strict: bool,
) -> Result<PawChannel> {
    let Atomic {
        r,
        v_ae,
        v_loc,
        r_local,
    } = *atom;
    validate(r, v_ae)?;
    validate(r, v_loc)?;
    let (l, smooth) = (set.l, set.smooth);
    let (n, count, terms) = (r.len(), set.energies.len(), smooth.wavevectors.len());
    let r_cut = smooth.r_cut;
    let square = |m: &DMatrix<f64>| m.nrows() == count && m.ncols() == count;
    if count == 0
        || set.waves.len() != product(&[count, n])?
        || smooth.coefficients.nrows() != count
        || smooth.coefficients.ncols() != terms
        || smooth.residuals.len() != count
        || !square(&smooth.norms)
        || !square(&smooth.achieved)
        || !square(&smooth.targets)
        || set.occupations.len() > count
        || !r_cut.is_finite()
        || r_cut <= 0.
        || !r_local.is_finite()
    {
        return Err(invalid(
            "inconsistent PAW channel: partial waves, Bessel expansions, norms or radii",
        ));
    }
    crate::finite(set.energies, "reference energies")?;
    crate::finite(set.waves, "all-electron partial waves")?;
    crate::finite(&smooth.wavevectors, "Bessel wave vectors")?;
    let _transient = ctx.reserve(product(&[10 * count + 8, INNER_POINTS, 8])?)?;
    let qs = &smooth.wavevectors;
    let c = &smooth.coefficients;
    let energies = set.energies;
    let wave = |i: usize| &set.waves[i * n..(i + 1) * n];

    let r_in = inner_grid(r_cut);
    let w = r_in.iter().map(|x| x * x).collect::<Vec<_>>();
    let v_loc_in = resample(ctx, r, v_loc, &r_in)?;
    let v_ae_in = resample(ctx, r, v_ae, &r_in)?;
    let ae_in = (0..count)
        .map(|i| resample(ctx, r, wave(i), &r_in))
        .collect::<Result<Vec<_>>>()?;
    let table = bessel_table(l, qs, &r_in);
    let m = r_in.len();
    let (mut pseudo_in, mut chi_in, mut kin_in) = (vec![], vec![], vec![]);
    for i in 0..count {
        let sum = |term: &dyn Fn(usize, usize) -> f64| {
            (0..m)
                .map(|p| {
                    let mut total = 0.;
                    for k in 0..terms {
                        total += term(k, p);
                    }
                    total
                })
                .collect::<Vec<_>>()
        };
        pseudo_in.push(sum(&|k, p| c[(i, k)] * table[k * m + p]));
        kin_in.push(sum(&|k, p| {
            0.5 * qs[k].powi(2) * c[(i, k)] * table[k * m + p]
        }));
        chi_in.push(sum(&|k, p| {
            let b = table[k * m + p];
            c[(i, k)] * ((energies[i] - 0.5 * qs[k].powi(2)) * b - v_loc_in[p] * b)
        }));
    }
    let inner = |a: &[f64], b: &[f64]| {
        let f = (0..m).map(|p| a[p] * b[p] * w[p]).collect::<Vec<_>>();
        simpson(&r_in, &f)
    };

    // The shell (r_cut, r_local] where v_loc still differs from v_AE: there phi~ = phi and
    // T phi = (eps - v_AE) phi, so chi = (v_AE - v_loc) phi and the shell adds
    // <phi_i|v_AE - v_loc|phi_j> to B and to Delta V alike (Delta T gets nothing).
    let shell = (r_local > r_cut).then(|| linspace(r_cut, r_local, INNER_POINTS));
    let mut shell_v = DMatrix::<f64>::zeros(count, count);
    let (mut shell_ae, mut shell_chi) = (vec![vec![]; count], vec![vec![]; count]);
    if let Some(r_shell) = &shell {
        let step = (0..n).map(|i| v_ae[i] - v_loc[i]).collect::<Vec<_>>();
        let dv = resample(ctx, r, &step, r_shell)?;
        for i in 0..count {
            shell_ae[i] = resample(ctx, r, wave(i), r_shell)?;
            shell_chi[i] = dv.iter().zip(&shell_ae[i]).map(|(d, a)| d * a).collect();
        }
        let weight = r_shell.iter().map(|x| x * x).collect::<Vec<_>>();
        for a in 0..count {
            for b in 0..count {
                let f = (0..r_shell.len())
                    .map(|p| shell_ae[a][p] * shell_chi[b][p] * weight[p])
                    .collect::<Vec<_>>();
                shell_v[(a, b)] = simpson(r_shell, &f)?;
            }
        }
    }
    let inner_all = |a: (&[f64], &[f64]), b: (&[f64], &[f64])| -> Result<f64> {
        let mut total = inner(a.0, b.0)?;
        if let Some(r_shell) = &shell {
            let f = (0..r_shell.len())
                .map(|p| a.1[p] * b.1[p] * r_shell[p] * r_shell[p])
                .collect::<Vec<_>>();
            total += simpson(r_shell, &f)?;
        }
        Ok(total)
    };

    let mut vanderbilt = DMatrix::<f64>::zeros(count, count);
    for i in 0..count {
        for j in 0..count {
            vanderbilt[(i, j)] = inner(&pseudo_in[i], &chi_in[j])? + shell_v[(i, j)];
        }
    }
    let inverse = vanderbilt
        .clone()
        .try_inverse()
        .ok_or_else(|| invalid("PAW projector overlap matrix B is singular"))?;
    let projectors_in = (0..count)
        .map(|i| combine((0..count).map(|k| inverse[(k, i)]), &chi_in))
        .collect::<Vec<_>>();
    let projectors_sh = (0..count)
        .map(|i| combine((0..count).map(|k| inverse[(k, i)]), &shell_chi))
        .collect::<Vec<_>>();
    let mut duality = DMatrix::<f64>::zeros(count, count);
    let mut gram = DMatrix::<f64>::zeros(count, count);
    for i in 0..count {
        for j in 0..count {
            duality[(i, j)] = inner_all(
                (&projectors_in[i], &projectors_sh[i]),
                (&pseudo_in[j], &shell_ae[j]),
            )?;
            gram[(i, j)] = inner_all(
                (&projectors_in[i], &projectors_sh[i]),
                (&projectors_in[j], &projectors_sh[j]),
            )?;
        }
    }
    let duality_error = (duality - DMatrix::identity(count, count)).amax();
    // Written so that a NaN fails: every comparison with it is false.
    if strict && !(duality_error <= DUALITY_TOLERANCE) {
        return Err(invalid(format!(
            "PAW-LCAO projectors of l={l} are not dual to the smooth partial waves (error {duality_error:.2e}); B is ill-conditioned"
        )));
    }

    // `q` is the charge the smooth density is missing: the overlap operator, the compensation
    // multipoles and the augmented density all count electrons with the plain inner product.
    // The coupling is an energy and takes the norm the Vanderbilt condition conserved,
    // `targets`; the two differ at O(c^-2) once the all-electron waves are relativistic.
    let q = &smooth.norms - &smooth.achieved;
    let q_norm = &smooth.targets - &smooth.achieved;
    let overlap_minimum = 1.
        + (&q * &gram)
            .complex_eigenvalues()
            .iter()
            .map(|e| e.re)
            .fold(f64::INFINITY, f64::min);
    if strict && !(overlap_minimum >= OVERLAP_MINIMUM) {
        return Err(invalid(format!(
            "the PAW-LCAO overlap operator of l={l} has an eigenvalue {overlap_minimum:.3} (the smooth partial waves carry too little or too much norm for their projectors); change r_cut, n_bessel or energy_offset"
        )));
    }

    // Kinetic energies inside r_c. The all-electron side is defined by the radial equation,
    // <phi_i|T|phi_j> = eps_j <phi_i|phi_j> - <phi_i|V|phi_j> with the norm the condition
    // conserved, which makes D^scr = dT + dV an identity and holds for a relativistic wave.
    let mut v_ae_matrix = DMatrix::<f64>::zeros(count, count);
    let mut t_pseudo = v_ae_matrix.clone();
    let mut v_pseudo = v_ae_matrix.clone();
    for i in 0..count {
        for j in 0..count {
            let v_phi = (0..m).map(|p| v_ae_in[p] * ae_in[j][p]).collect::<Vec<_>>();
            let v_loc_phi = (0..m)
                .map(|p| v_loc_in[p] * pseudo_in[j][p])
                .collect::<Vec<_>>();
            v_ae_matrix[(i, j)] = inner(&ae_in[i], &v_phi)?;
            t_pseudo[(i, j)] = inner(&pseudo_in[i], &kin_in[j])?;
            v_pseudo[(i, j)] = inner(&pseudo_in[i], &v_loc_phi)?;
        }
    }
    let t_ae = DMatrix::from_fn(count, count, |i, j| {
        energies[j] * smooth.targets[(i, j)] - v_ae_matrix[(i, j)]
    });
    let kinetic_difference = symmetrize(&(t_ae - t_pseudo));
    let potential_difference = symmetrize(&(&v_ae_matrix - v_pseudo)) + symmetrize(&shell_v);
    let raw = DMatrix::from_fn(count, count, |i, j| {
        vanderbilt[(i, j)] + q_norm[(i, j)] * energies[j]
    });
    let asymmetry = (&raw - raw.transpose()).amax();
    if strict && !(asymmetry <= COUPLING_ASYMMETRY_TOLERANCE) {
        return Err(invalid(format!(
            "the screened PAW-LCAO coupling of l={l} is asymmetric by {asymmetry:.2e} Ha (the partial waves do not satisfy the radial equation or the matching failed)"
        )));
    }
    let coupling_screened = symmetrize(&raw);
    let consistency_error =
        (&coupling_screened - (&kinetic_difference + &potential_difference)).amax();

    // Full-grid tables. A Bessel value is shared by every reference at a node.
    let mut pseudo_waves = ctx.allocate::<f64>(product(&[count, n])?)?;
    let mut raw_projectors = ctx.allocate::<f64>(product(&[count, n])?)?;
    let mut projectors = ctx.allocate::<f64>(product(&[count, n])?)?;
    let mut bessels = vec![0.; terms];
    for p in 0..n {
        let inside = r[p] <= r_cut;
        if inside {
            for (k, b) in bessels.iter_mut().enumerate() {
                *b = bessel(l as i32, qs[k] * r[p]);
            }
        }
        for i in 0..count {
            let (value, chi) = if inside {
                let mut value = 0.;
                let mut chi = 0.;
                for (k, b) in bessels.iter().enumerate() {
                    value += c[(i, k)] * b;
                    chi += c[(i, k)] * ((energies[i] - 0.5 * qs[k].powi(2)) * b - v_loc[p] * b);
                }
                (value, chi)
            } else if r[p] <= r_local {
                (wave(i)[p], (v_ae[p] - v_loc[p]) * wave(i)[p])
            } else {
                (wave(i)[p], 0.)
            };
            pseudo_waves.as_mut_slice()[i * n + p] = value;
            raw_projectors.as_mut_slice()[i * n + p] = chi;
        }
        for i in 0..count {
            let mut total = 0.;
            for k in 0..count {
                total += inverse[(k, i)] * raw_projectors.as_slice()[k * n + p];
            }
            projectors.as_mut_slice()[i * n + p] = total;
        }
    }

    let mut occupations = set.occupations.to_vec();
    occupations.resize(count, 0.);
    Ok(PawChannel {
        l,
        n: set.n,
        r_cut,
        q_cut: set.q_cut,
        reference_energies: energies.to_vec(),
        occupations,
        wavevectors: qs.clone(),
        coefficients: c.clone(),
        residual_kinetic: smooth.residuals.clone(),
        pseudo_waves,
        projectors,
        raw_projectors,
        overlap_correction: q,
        norm_correction: q_norm,
        kinetic_difference,
        potential_difference,
        coupling_screened,
        vanderbilt,
        duality_error,
        overlap_minimum,
        asymmetry,
        consistency_error,
    })
}
