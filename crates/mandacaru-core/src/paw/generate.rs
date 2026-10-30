//! Native spherical PAW dataset generation. Atomic units throughout.
use super::{
    channel::{self, ChannelInput},
    dataset::{Dataset, DatasetChannel},
    invalid,
    pseudize::{self, Equation, NormCondition, PseudoInput},
    radial,
    smooth::SmoothWaves,
};
use crate::{
    models::{
        atom::{self, AtomOptions},
        elements::configuration,
        relativity::Treatment,
    },
    product, Context, Result,
};
use serde_json::Value;
use std::{collections::BTreeMap, f64::consts::PI};

pub struct Options {
    pub atom: AtomOptions,
    pub functional: String,
    pub radii: BTreeMap<usize, f64>,
    pub radius_factor: f64,
    pub local_radius: Option<f64>,
    pub local_shift: f64,
    pub norm_deficit: f64,
    pub reference_gap: f64,
    pub bessel_count: usize,
    pub q_cut: f64,
    pub core_electrons: Option<usize>,
    pub nonlinear_core: bool,
}
impl Options {
    pub fn parse(source: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(source).map_err(|e| invalid(e.to_string()))?;
        let fields = value
            .as_object()
            .ok_or_else(|| invalid("PAW options must be an object"))?;
        let supported = [
            "functional",
            "points",
            "r_max_bohr",
            "max_iterations",
            "tolerance",
            "mixing",
            "radii_bohr",
            "radius_factor",
            "local_radius_bohr",
            "local_shift_ha",
            "norm_deficit",
            "reference_gap_ha",
            "bessel_count",
            "q_cut_inv_bohr",
            "core_electrons",
            "nonlinear_core",
        ];
        if fields.keys().any(|k| !supported.contains(&k.as_str())) {
            return Err(invalid("unknown native PAW generation option"));
        }
        let real = |name: &str, default: f64| -> Result<f64> {
            match value.get(name) {
                None => Ok(default),
                Some(v) => v
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid(format!("{name} must be finite"))),
            }
        };
        let integer = |name: &str, default: usize| -> Result<usize> {
            match value.get(name) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .ok_or_else(|| invalid(format!("{name} must be a nonnegative integer"))),
            }
        };
        let functional = value
            .get("functional")
            .map_or(Some("lda"), Value::as_str)
            .ok_or_else(|| invalid("functional must be a string"))?
            .to_owned();
        if !["lda", "pbe"].contains(&functional.as_str()) {
            return Err(invalid(
                "native spherical PAW generation supports lda and pbe",
            ));
        }
        let mut radii = BTreeMap::new();
        if let Some(v) = value.get("radii_bohr") {
            for (key, val) in v
                .as_object()
                .ok_or_else(|| invalid("radii_bohr maps angular momentum to a radius"))?
            {
                let l = key
                    .parse::<usize>()
                    .map_err(|_| invalid("angular momentum keys must be nonnegative integers"))?;
                let radius = val
                    .as_f64()
                    .filter(|x| x.is_finite() && *x > 0.)
                    .ok_or_else(|| invalid("PAW radius must be positive and finite"))?;
                if l > 8 {
                    return Err(invalid("PAW angular momentum exceeds eight"));
                }
                radii.insert(l, radius);
            }
        }
        let out = Self {
            atom: AtomOptions {
                points: integer("points", 6000)?,
                r_max: real("r_max_bohr", 30.)?,
                max_iterations: integer("max_iterations", 300)?,
                tolerance: real("tolerance", 1e-8)?,
                mixing: real("mixing", 0.25)?,
            },
            functional,
            radii,
            radius_factor: real("radius_factor", 1.3)?,
            local_radius: if value.get("local_radius_bohr").is_some() {
                Some(real("local_radius_bohr", 0.)?)
            } else {
                None
            },
            local_shift: real("local_shift_ha", 0.)?,
            norm_deficit: real("norm_deficit", 0.1)?,
            reference_gap: real("reference_gap_ha", 1.)?,
            bessel_count: integer("bessel_count", 8)?,
            q_cut: real("q_cut_inv_bohr", 5.)?,
            core_electrons: if value.get("core_electrons").is_some() {
                Some(integer("core_electrons", 0)?)
            } else {
                None
            },
            nonlinear_core: value
                .get("nonlinear_core")
                .map_or(Some(true), Value::as_bool)
                .ok_or_else(|| invalid("nonlinear_core must be boolean"))?,
        };
        if out.radius_factor <= 0.
            || !(0. ..1.).contains(&out.norm_deficit)
            || out.reference_gap <= 0.
            || out.q_cut <= 0.
            || !(6..=32).contains(&out.bessel_count)
            || out.local_radius.is_some_and(|r| r <= 0.)
        {
            return Err(invalid(
                "invalid PAW radius, deficit, reference gap or Bessel basis",
            ));
        }
        Ok(out)
    }
}

pub fn generate(ctx: &Context, z: u8, options: &Options) -> Result<Dataset> {
    let atom = atom::solve_functional(ctx, z, &options.atom, None, None, &options.functional)?;
    if !atom.converged {
        return Err(invalid("PAW reference atom did not converge"));
    }
    let n = atom.r.as_slice().len();
    let r = atom.r.as_slice();
    let core = options.core_electrons.unwrap_or_else(|| {
        [0, 2, 10, 18, 36, 54, 86]
            .into_iter()
            .filter(|&v| v < z as usize)
            .max()
            .unwrap()
    });
    if core >= z as usize {
        return Err(invalid("PAW frozen core must leave valence electrons"));
    }
    let config = configuration(z);
    let mut frozen = core;
    let mut valence = vec![];
    let mut core_indices = vec![];
    for (index, (_, _, q)) in config.iter().enumerate() {
        if frozen >= *q {
            frozen -= *q;
            core_indices.push(index);
        } else if frozen == 0 {
            valence.push(index);
        } else {
            return Err(invalid(
                "frozen core must contain complete reference subshells",
            ));
        }
    }
    if frozen != 0 {
        return Err(invalid("frozen core exceeds the reference configuration"));
    }
    let charge = valence.iter().map(|&i| atom.occupations[i].2).sum::<f64>();
    let mut by_l: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &index in &valence {
        by_l.entry(atom.occupations[index].1)
            .or_default()
            .push(index);
    }
    let mut radii = BTreeMap::new();
    for (&l, indices) in &by_l {
        let radial = atom.orbitals[*indices.last().unwrap()].as_slice();
        let peak = radial
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .0;
        let requested = options
            .radii
            .get(&l)
            .copied()
            .unwrap_or(options.radius_factor * r[peak]);
        let index = r.partition_point(|x| *x < requested).clamp(4, n - 5);
        if requested >= r[n - 4] {
            return Err(invalid("PAW radius exceeds the reference grid"));
        }
        radii.insert(l, r[index]);
    }
    if options.radii.keys().any(|l| !by_l.contains_key(l)) {
        return Err(invalid(
            "radii_bohr names an angular channel absent from the valence",
        ));
    }
    let local_radius = options
        .local_radius
        .unwrap_or(0.9 * radii.values().copied().fold(f64::INFINITY, f64::min));
    let screened = radial::local_potential(
        ctx,
        r,
        atom.effective.as_slice(),
        local_radius,
        options.local_shift,
        None,
    )?;
    let mut channels = vec![];
    for (&l, indices) in &by_l {
        if indices.len() > 4 {
            return Err(invalid(
                "a PAW angular channel has too many occupied references",
            ));
        }
        let count = indices.len().max(2);
        let radius = radii[&l];
        let mut references = ctx.allocate::<f64>(product(&[count, n])?)?;
        let mut energies = vec![];
        let mut occupations = vec![];
        let mut shells = vec![];
        for (wave_index, &index) in indices.iter().enumerate() {
            let principal = atom.occupations[index].0;
            let (wave, energy) = pseudize::bound_reference(
                ctx,
                r,
                atom.effective.as_slice(),
                l,
                atom.eigenvalues[index],
                &Equation {
                    treatment: Treatment::None,
                    kappa: None,
                    z: z as f64,
                },
                None,
            )?;
            let nodes = wave.windows(2).filter(|w| w[0] * w[1] < 0.).count();
            if energy >= 0. || nodes != principal - l - 1 {
                return Err(invalid(format!(
                    "PAW reference has {nodes} nodes at {energy:.6} Ha, expected {}",
                    principal - l - 1
                )));
            }
            references.as_mut_slice()[wave_index * n..(wave_index + 1) * n].copy_from_slice(&wave);
            energies.push(energy);
            occupations.push(atom.occupations[index].2);
            shells.push(atom.occupations[index].0);
        }
        if indices.len() == 1 {
            let energy = energies[0] + options.reference_gap;
            let wave = pseudize::scattering_reference(
                ctx,
                r,
                atom.effective.as_slice(),
                l,
                energy,
                &Equation {
                    treatment: Treatment::None,
                    kappa: None,
                    z: z as f64,
                },
                radius,
            )?;
            references.as_mut_slice()[n..2 * n].copy_from_slice(&wave);
            energies.push(energy);
            occupations.push(0.);
        }
        let optimized = pseudize::optimize(
            ctx,
            &PseudoInput {
                r,
                v_ae: atom.effective.as_slice(),
                l,
                waves: references.as_slice(),
                energies: &energies,
                r_cut: radius,
                q_cut: options.q_cut,
                n_bessel: options.bessel_count,
                norm: NormCondition::Factor(1. - options.norm_deficit),
                equation: Equation {
                    treatment: Treatment::None,
                    kappa: None,
                    z: z as f64,
                },
            },
        )?;
        let mut coefficients = ctx.allocate::<f64>(optimized.coefficients.len())?;
        for (wave, row) in optimized.coefficients.row_iter().enumerate() {
            for (term, value) in row.iter().enumerate() {
                coefficients.as_mut_slice()[wave * row.len() + term] = *value;
            }
        }
        let smooth = SmoothWaves {
            wavevectors: optimized.wavevectors,
            coefficients,
            norms: optimized.norms,
            residuals: optimized.residuals,
        };
        let channel = channel::construct(
            ctx,
            &ChannelInput {
                r,
                ae: references.as_slice(),
                vae: atom.effective.as_slice(),
                local: screened.as_slice(),
                l,
                radius,
                local_radius,
                energies: &energies,
                wavevectors: &smooth.wavevectors,
                coefficients: smooth.coefficients.as_slice(),
            },
        )?;
        let ionic = channel.screened.clone();
        channels.push(DatasetChannel {
            l,
            n: shells,
            energies,
            occupations,
            radius,
            ae: references,
            smooth,
            channel,
            ionic,
        });
    }
    let mut valence_density = ctx.allocate::<f64>(n)?;
    let mut ae_valence = ctx.allocate::<f64>(n)?;
    let mut core_density = ctx.allocate::<f64>(n)?;
    for &index in &core_indices {
        for (i, _) in r.iter().enumerate() {
            core_density.as_mut_slice()[i] += atom.occupations[index].2
                * atom.orbitals[index].as_slice()[i].powi(2)
                / (4. * PI * r[i] * r[i]);
        }
    }
    for ch in &channels {
        for (wave, &occupation) in ch.occupations.iter().enumerate() {
            for (i, _) in r.iter().enumerate() {
                valence_density.as_mut_slice()[i] +=
                    occupation * ch.channel.pseudo.as_slice()[wave * n + i].powi(2) / (4. * PI);
                ae_valence.as_mut_slice()[i] +=
                    occupation * ch.ae.as_slice()[wave * n + i].powi(2) / (4. * PI);
            }
        }
    }
    let compensation_radius = radii.values().copied().fold(0., f64::max);
    let mut integrand = ctx.allocate::<f64>(n)?;
    for (i, _) in r.iter().enumerate() {
        integrand.as_mut_slice()[i] = 4. * PI * r[i] * r[i] * valence_density.as_slice()[i];
    }
    let compensation_charge = charge - crate::math::radial::integrate(r, integrand.as_slice())?;
    let mut smooth_core = ctx.allocate::<f64>(n)?;
    smooth_core
        .as_mut_slice()
        .copy_from_slice(core_density.as_slice());
    if options.nonlinear_core && !core_indices.is_empty() {
        let d = crate::math::radial::derivatives(
            r,
            core_density.as_slice(),
            compensation_radius,
            2,
            8,
        )?;
        if d[0] > 0. {
            let radius = compensation_radius;
            let slope = d[1] / d[0];
            let curvature = d[2] / d[0] - slope * slope;
            let c = (curvature - slope / radius) / (8. * radius * radius);
            let b = slope / (2. * radius) - 2. * c * radius * radius;
            let a = d[0].ln() - b * radius * radius - c * radius.powi(4);
            for (i, _) in r.iter().enumerate() {
                if r[i] < radius {
                    smooth_core.as_mut_slice()[i] = (a + b * r[i] * r[i] + c * r[i].powi(4)).exp();
                }
            }
        }
    }
    let mut augmented = ctx.allocate::<f64>(n)?;
    let mut xc_density = ctx.allocate::<f64>(n)?;
    for (i, _) in r.iter().enumerate() {
        augmented.as_mut_slice()[i] = valence_density.as_slice()[i]
            + compensation_charge * radial::shape(r[i], compensation_radius);
        xc_density.as_mut_slice()[i] = valence_density.as_slice()[i]
            + if options.nonlinear_core {
                smooth_core.as_slice()[i]
            } else {
                0.
            };
    }
    let vh = crate::models::spherical::hartree(ctx, r, augmented.as_slice())?;
    let xc = crate::models::spherical::exchange_correlation(
        ctx,
        r,
        xc_density.as_slice(),
        &options.functional,
        false,
    )?;
    let mut ionic = ctx.allocate::<f64>(n)?;
    for (i, _) in r.iter().enumerate() {
        ionic.as_mut_slice()[i] =
            screened.as_slice()[i] - vh.as_slice()[i] - xc.potential.as_slice()[i];
        integrand.as_mut_slice()[i] =
            vh.as_slice()[i] * radial::shape(r[i], compensation_radius) * 4. * PI * r[i] * r[i];
    }
    let hartree_screening = crate::math::radial::integrate(r, integrand.as_slice())?;
    for ch in &mut channels {
        ch.ionic = &ch.channel.screened - &ch.channel.overlap * hartree_screening;
    }
    let mut terms = vec![];
    for density in [ae_valence.as_slice(), augmented.as_slice()] {
        let v = crate::models::spherical::hartree(ctx, r, density)?;
        for (i, _) in r.iter().enumerate() {
            integrand.as_mut_slice()[i] =
                0.5 * v.as_slice()[i] * density[i] * 4. * PI * r[i] * r[i];
        }
        terms.push(crate::math::radial::integrate(r, integrand.as_slice())?);
    }
    let mut xc_terms = vec![];
    for density in [ae_valence.as_slice(), valence_density.as_slice()] {
        let xc = crate::models::spherical::exchange_correlation(
            ctx,
            r,
            density,
            &options.functional,
            false,
        )?;
        for (i, _) in r.iter().enumerate() {
            integrand.as_mut_slice()[i] =
                xc.potential.as_slice()[i] * density[i] * 4. * PI * r[i] * r[i];
        }
        xc_terms.push(xc.energy - crate::math::radial::integrate(r, integrand.as_slice())?);
    }
    let one_center_energy = -terms[0] + xc_terms[0] + terms[1] - xc_terms[1];
    drop(augmented);
    drop(xc_density);
    drop(ae_valence);
    drop(integrand);
    drop(vh);
    drop(xc);
    let radial_grid = ctx.copy_slice(r)?;
    let reference_energy = atom.energy;
    Ok(Dataset {
        context: ctx.clone(),
        atomic_number: z,
        functional: options.functional.clone(),
        r: radial_grid,
        reference_energy,
        valence_charge: charge,
        local_radius,
        compensation_radius,
        compensation_charge,
        hartree_screening,
        one_center_energy,
        ionic,
        screened,
        valence_density,
        core_density,
        smooth_core,
        channels,
    })
}
