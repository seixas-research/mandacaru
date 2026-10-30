//! Spherical all-electron reference atoms: LDA/PBE, nonrelativistic, scalar-relativistic or
//! Dirac, on a uniform radial grid, with density-metric DIIS.
//!
//! The Kohn-Sham problem collapses to one radial equation per subshell for `u = r R`. A
//! uniform grid keeps the nonrelativistic eigenproblem a symmetric tridiagonal matrix whose
//! `k`-th eigenpair is found directly. A relativistic atom solves `models::relativity`'s
//! operator instead (each `j` separately for Dirac) and finishes on `models::numerov`'s
//! fourth-order shoot for the valence, so the atom and the pseudization built from it agree
//! on where a relativistic s level is.
use super::{
    configuration::{self, valence_subshells},
    invalid,
    mixing::{density_change, PulayMixer, HISTORY},
    numerov,
    radial_xc::xc_potential,
    relativity::{self, PotentialTerms, Treatment},
    selection, spherical, xc,
};
use crate::{product, Buffer, Context, Reservation, Result};
use std::{collections::HashMap, f64::consts::PI};

/// Extra self-consistency rounds a relativistic atom takes with the Numerov shoot.
pub const POLISH_ROUNDS: usize = 12;

pub struct AtomOptions {
    pub points: usize,
    pub r_max: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub mixing: f64,
}

/// What is solved: the functional and the radial equation. `polish` is ignored when
/// `relativity` is `Treatment::None`.
pub struct AtomModel {
    pub functional: String,
    pub relativity: Treatment,
    pub polish: usize,
}

impl AtomModel {
    pub fn nonrelativistic(functional: &str) -> Self {
        Self {
            functional: functional.to_string(),
            relativity: Treatment::None,
            polish: 0,
        }
    }
}

/// One `(n, l, kappa)` level of a Dirac atom.
pub struct JState {
    pub n: usize,
    pub l: usize,
    pub kappa: i32,
    pub occupation: f64,
    pub eigenvalue: f64,
    pub orbital: Buffer<f64>,
}

pub struct Atom {
    pub r: Buffer<f64>,
    pub density: Buffer<f64>,
    pub hartree: Buffer<f64>,
    pub xc: Buffer<f64>,
    pub effective: Buffer<f64>,
    /// `u = r R` of each `(n, l)` entry of `occupations`; for a Dirac atom the
    /// `(2j+1)`-weighted average of its two `j` levels.
    pub orbitals: Vec<Buffer<f64>>,
    pub eigenvalues: Vec<f64>,
    pub occupations: Vec<(usize, usize, f64)>,
    /// Filled only by `Treatment::Dirac`.
    pub j_states: Vec<JState>,
    pub relativity: Treatment,
    pub functional: String,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    _memory: Reservation,
}

fn integrate(values: impl Iterator<Item = f64>, n: usize, h: f64) -> f64 {
    values
        .enumerate()
        .map(|(i, v)| v * if i == 0 || i == n - 1 { 0.5 } else { 1. })
        .sum::<f64>()
        * h
}

/// Hartree and exchange-correlation potentials of `density`, and the XC energy.
fn potentials(
    ctx: &Context,
    r: &[f64],
    density: &[f64],
    h: f64,
    functional: &str,
    relativistic: bool,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let n = r.len();
    let hartree = spherical::hartree(ctx, r, density)?.as_slice().to_vec();
    let (per_electron, vxc) = xc_potential(ctx, r, density, functional, relativistic)?;
    let exc = integrate(
        (0..n).map(|i| per_electron[i] * density[i] * 4. * PI * r[i] * r[i]),
        n,
        h,
    );
    Ok((hartree, vxc, exc))
}

/// Nonrelativistic LDA atom.
pub fn solve(
    ctx: &Context,
    z: u8,
    options: &AtomOptions,
    configuration_spec: Option<&[(usize, usize, f64)]>,
    confinement: Option<&[f64]>,
) -> Result<Atom> {
    solve_functional(ctx, z, options, configuration_spec, confinement, "lda")
}

/// Nonrelativistic atom with the named functional.
pub fn solve_functional(
    ctx: &Context,
    z: u8,
    options: &AtomOptions,
    configuration_spec: Option<&[(usize, usize, f64)]>,
    confinement: Option<&[f64]>,
    functional: &str,
) -> Result<Atom> {
    solve_model(
        ctx,
        z,
        options,
        &AtomModel::nonrelativistic(functional),
        configuration_spec,
        confinement,
    )
}

/// One orbital: its radial function and eigenvalue, for any treatment.
struct Solver<'a> {
    ctx: &'a Context,
    r: &'a [f64],
    z: f64,
    treatment: Treatment,
    guesses: HashMap<(usize, usize, i32), f64>,
}

impl Solver<'_> {
    fn state(
        &mut self,
        n: usize,
        l: usize,
        kappa: Option<i32>,
        potential: &[f64],
        terms: Option<&PotentialTerms>,
    ) -> Result<(Vec<f64>, f64)> {
        let key = (n, l, kappa.unwrap_or(0));
        let (u, eps) = relativity::solve_radial(
            self.ctx,
            self.r,
            potential,
            l,
            n - l - 1,
            self.treatment,
            kappa,
            self.z,
            self.guesses.get(&key).copied(),
            terms,
        )?;
        self.guesses.insert(key, eps);
        Ok((u, eps))
    }
}

pub fn solve_model(
    ctx: &Context,
    z: u8,
    options: &AtomOptions,
    model: &AtomModel,
    configuration_spec: Option<&[(usize, usize, f64)]>,
    confinement: Option<&[f64]>,
) -> Result<Atom> {
    xc::functional(&model.functional)?;
    let functional = model.functional.as_str();
    let treatment = model.relativity;
    let relativistic = treatment.relativistic_exchange();
    let dirac = treatment == Treatment::Dirac;
    let n = options.points;
    let h = options.r_max / (n as f64 + 1.);
    if z == 0
        || z > 118
        || n < 4
        || n > i32::MAX as usize
        || !options.r_max.is_finite()
        || options.r_max <= 0.
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || !options.mixing.is_finite()
        || options.mixing <= 0.
        || options.mixing > 1.
        || options.max_iterations == 0
    {
        return Err(invalid("invalid spherical atom or SCF options"));
    }
    if confinement.is_some_and(|v| v.len() != n || v.iter().any(|v| !v.is_finite())) {
        return Err(invalid(
            "confinement must be a finite array on the radial grid",
        ));
    }
    let occupations = match configuration_spec {
        Some(v) => v.to_vec(),
        None => {
            selection::relaxed_configuration(ctx, z, functional, treatment, options.r_max, None)?
        }
    };
    if occupations.is_empty()
        || occupations.iter().any(|(nn, l, q)| {
            *nn == 0
                || *l >= *nn
                || *l > 8
                || nn - l > n
                || !q.is_finite()
                || *q <= 0.
                || *q > 2. * (2 * l + 1) as f64
        })
    {
        return Err(invalid("invalid atomic configuration"));
    }
    for (i, state) in occupations.iter().enumerate() {
        if occupations[..i]
            .iter()
            .any(|s| (s.0, s.1) == (state.0, state.1))
        {
            return Err(invalid("duplicate atomic subshell"));
        }
    }
    let memory = ctx.reserve(product(&[occupations.len(), 128])?)?;
    let _workspace = ctx.reserve(product(&[n, 8, 64])?)?;
    let electrons = occupations.iter().map(|s| s.2).sum::<f64>();
    let z_f = z as f64;
    let r = (1..=n).map(|i| i as f64 * h).collect::<Vec<_>>();
    let shell = r.iter().map(|r| 4. * PI * r * r).collect::<Vec<_>>();
    let nuclear = r.iter().map(|r| -z_f / r).collect::<Vec<_>>();
    let wall = confinement.map_or_else(|| vec![0.; n], |v| v.to_vec());

    // Thomas-Fermi-like start: a screened exponential holding Z electrons.
    let scale = z_f.cbrt().max(1.);
    let mut density = r
        .iter()
        .map(|r| z_f * scale.powi(3) / PI * (-2. * scale * r).exp())
        .collect::<Vec<_>>();
    let charge = integrate((0..n).map(|i| shell[i] * density[i]), n, h);
    for d in density.iter_mut() {
        *d *= z_f / charge.max(1e-30);
    }

    let occupations_j = if dirac {
        configuration::split_configuration(&occupations)
    } else {
        Vec::new()
    };
    let mut solver = Solver {
        ctx,
        r: &r,
        z: z_f,
        treatment,
        guesses: HashMap::new(),
    };
    let mut orbitals: Vec<Vec<f64>> = vec![Vec::new(); occupations.len()];
    let mut eigenvalues = vec![0.; occupations.len()];
    let mut orbitals_j: Vec<Vec<f64>> = vec![Vec::new(); occupations_j.len()];
    let mut eigenvalues_j = vec![0.; occupations_j.len()];
    let weight = shell.iter().map(|s| s * h).collect::<Vec<_>>();
    let mut mixer = PulayMixer::new(weight.clone(), options.mixing, electrons, HISTORY);
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 1..=options.max_iterations {
        let (hartree, vxc, _) = potentials(ctx, &r, &density, h, functional, relativistic)?;
        let effective = (0..n)
            .map(|i| nuclear[i] + hartree[i] + vxc[i])
            .collect::<Vec<_>>();
        let walled = (0..n).map(|i| effective[i] + wall[i]).collect::<Vec<_>>();
        let terms = if treatment == Treatment::None {
            None
        } else {
            Some(PotentialTerms::new(&r, &walled, z_f)?)
        };
        let mut next = vec![0.; n];
        if dirac {
            // Each j is its own radial equation; the (n, l) entries are the (2j+1) average.
            for (index, &(nn, l, kappa, share)) in occupations_j.iter().enumerate() {
                let (u, eps) = solver.state(nn, l, Some(kappa), &walled, terms.as_ref())?;
                if share > 0. {
                    for i in 0..n {
                        next[i] += share * u[i] * u[i] / shell[i];
                    }
                }
                orbitals_j[index] = u;
                eigenvalues_j[index] = eps;
            }
        } else {
            for (index, &(nn, l, q)) in occupations.iter().enumerate() {
                let (u, eps) = solver.state(nn, l, None, &walled, terms.as_ref())?;
                for i in 0..n {
                    next[i] += q * u[i] * u[i] / shell[i];
                }
                orbitals[index] = u;
                eigenvalues[index] = eps;
            }
        }
        let change = density_change(&density, &next, &r);
        density = mixer.mix(&density, &next);
        iterations = iteration;
        if change < options.tolerance {
            converged = true;
            break;
        }
    }

    if treatment != Treatment::None && model.polish > 0 {
        // Finish on Numerov orbitals: a relativistic s state, whose r^gamma cusp neither
        // discretization resolves well, lands ~10 mHa apart in the tridiagonal and shooting
        // solutions, and a pseudization built on a wave that is not an eigenstate of its
        // potential leaves the ionic potential off -Z_ion/r. Only the valence is re-solved.
        let wanted = valence_subshells(&occupations);
        let mut polisher = PulayMixer::new(weight.clone(), options.mixing, electrons, HISTORY);
        let states = if dirac {
            occupations_j
                .iter()
                .map(|s| (s.0, s.1, Some(s.2), s.3))
                .collect::<Vec<_>>()
        } else {
            occupations
                .iter()
                .map(|s| (s.0, s.1, None, s.2))
                .collect::<Vec<_>>()
        };
        let refresh = |density: &[f64],
                       orbitals: &mut [Vec<f64>],
                       eigenvalues: &mut [f64],
                       valence_only: bool|
         -> Result<Vec<f64>> {
            let (hartree, vxc, _) = potentials(ctx, &r, density, h, functional, relativistic)?;
            let walled = (0..n)
                .map(|i| nuclear[i] + hartree[i] + vxc[i] + wall[i])
                .collect::<Vec<_>>();
            let mut next = vec![0.; n];
            for (index, &(nn, l, kappa, q)) in states.iter().enumerate() {
                if q <= 0. {
                    continue;
                }
                if wanted.contains(&(nn, l)) {
                    if let Some((u, eps)) = numerov::bound_state(
                        ctx,
                        &r,
                        &walled,
                        l,
                        eigenvalues[index],
                        z_f,
                        numerov::DEFAULT_WINDOW,
                        treatment,
                        kappa,
                    )? {
                        orbitals[index] = u;
                        eigenvalues[index] = eps;
                    }
                }
                if !valence_only {
                    for i in 0..n {
                        next[i] += q * (orbitals[index][i] * orbitals[index][i]) / shell[i];
                    }
                }
            }
            Ok(next)
        };
        let (orbs, eigs) = if dirac {
            (&mut orbitals_j, &mut eigenvalues_j)
        } else {
            (&mut orbitals, &mut eigenvalues)
        };
        for _ in 0..model.polish {
            let next = refresh(&density, orbs, eigs, false)?;
            let change = density_change(&density, &next, &r);
            density = polisher.mix(&density, &next);
            if change < options.tolerance {
                break;
            }
        }
        // One last solve in the potential that will actually be reported, so the returned
        // eigenvalues do not belong to a potential one step behind the returned one.
        refresh(&density, orbs, eigs, true)?;
    }

    if dirac {
        for (index, &(nn, l, _)) in occupations.iter().enumerate() {
            let find = |kappa: i32| {
                occupations_j
                    .iter()
                    .position(|s| (s.0, s.1, s.2) == (nn, l, kappa))
                    .unwrap()
            };
            eigenvalues[index] = relativity::j_average(l, |k| eigenvalues_j[find(k)]);
            let total = relativity::kappa_values(l)
                .iter()
                .map(|&k| relativity::degeneracy(k) as f64)
                .sum::<f64>();
            let mut averaged = vec![0.; n];
            for k in relativity::kappa_values(l) {
                let w = relativity::degeneracy(k) as f64;
                for i in 0..n {
                    averaged[i] += w * orbitals_j[find(k)][i];
                }
            }
            for v in averaged.iter_mut() {
                *v /= total;
            }
            let norm = integrate((0..n).map(|i| averaged[i] * averaged[i]), n, h).sqrt();
            for v in averaged.iter_mut() {
                *v /= norm;
            }
            orbitals[index] = averaged;
        }
    }

    // Final potentials consistent with the converged density.
    let (hartree, vxc, exc) = potentials(ctx, &r, &density, h, functional, relativistic)?;
    let effective = (0..n)
        .map(|i| nuclear[i] + hartree[i] + vxc[i])
        .collect::<Vec<_>>();
    let band = if dirac {
        occupations_j
            .iter()
            .zip(&eigenvalues_j)
            .filter(|(s, _)| s.3 > 0.)
            .map(|(s, e)| s.3 * e)
            .sum::<f64>()
    } else {
        occupations
            .iter()
            .zip(&eigenvalues)
            .map(|(s, e)| s.2 * e)
            .sum::<f64>()
    };
    let correction = integrate(
        (0..n).map(|i| (0.5 * hartree[i] + vxc[i]) * density[i] * shell[i]),
        n,
        h,
    );
    let j_states = occupations_j
        .iter()
        .enumerate()
        .map(|(i, &(nn, l, kappa, occupation))| {
            Ok(JState {
                n: nn,
                l,
                kappa,
                occupation,
                eigenvalue: eigenvalues_j[i],
                orbital: ctx.copy_slice(&orbitals_j[i])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Atom {
        r: ctx.copy_slice(&r)?,
        density: ctx.copy_slice(&density)?,
        hartree: ctx.copy_slice(&hartree)?,
        xc: ctx.copy_slice(&vxc)?,
        effective: ctx.copy_slice(&effective)?,
        orbitals: orbitals
            .iter()
            .map(|u| ctx.copy_slice(u))
            .collect::<Result<Vec<_>>>()?,
        eigenvalues,
        occupations,
        j_states,
        relativity: treatment,
        functional: model.functional.clone(),
        energy: band - correction + exc,
        converged,
        iterations,
        _memory: memory,
    })
}
