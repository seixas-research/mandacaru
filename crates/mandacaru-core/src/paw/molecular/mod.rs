//! Isolated PAW-LCAO integrals with exact valence one-center interactions.
//! Dataset decoding and basis recipes are outside this numerical boundary.
pub(super) mod data;
pub(super) mod onecenter;
use super::invalid;
use crate::{
    basis::{Basis, Orbital, Radial},
    math::{
        linalg::real_eigen, projection::PointAccumulator, radial::gauss_legendre, spline::Cubic,
        SpatialGrid,
    },
    molecular::{
        integrals::{MolecularCorrections, PairCorrection, PawKohnSham},
        MolecularIntegrals,
    },
    product, Buffer, Context, Geometry, Result,
};
pub use data::Arrays;
use data::{count, table, Species};
use nalgebra::DMatrix;
use onecenter::{angular_rule, harmonic, potential, shape};
use std::{collections::BTreeMap, f64::consts::PI, sync::Arc};

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}
fn long_range(charge: f64, sigma: f64, radius: f64) -> f64 {
    if radius < 1e-12 * sigma {
        -charge * (2. / PI).sqrt() / sigma
    } else {
        -charge * libm::erf(radius / (2f64.sqrt() * sigma)) / radius
    }
}
fn radial_rule(radius: f64, panel: f64, points: usize) -> Result<Vec<(f64, f64)>> {
    let nodes = gauss_legendre(points)?;
    let edges = if panel > 0. && panel < radius {
        vec![0., panel, radius]
    } else {
        vec![0., radius]
    };
    Ok(edges
        .windows(2)
        .flat_map(|edge| {
            nodes.iter().map(move |(x, w)| {
                let h = 0.5 * (edge[1] - edge[0]);
                let r = edge[0] + h * (x + 1.);
                (r, h * w * r * r)
            })
        })
        .collect())
}
fn sphere(
    radius: f64,
    panel: f64,
    radial: usize,
    polar: usize,
    azimuthal: usize,
    mut visit: impl FnMut([f64; 3], f64, f64) -> Result<()>,
) -> Result<()> {
    let angular = angular_rule(polar, azimuthal)?;
    for (r, w) in radial_rule(radius, panel, radial)? {
        for (u, v) in &angular {
            visit(u.map(|x| x * r), r, w * v)?;
        }
    }
    Ok(())
}
fn projection(
    ctx: &Context,
    basis: &Basis,
    species: &Species,
    terms: &onecenter::AtomTerms,
    center: [f64; 3],
) -> Result<DMatrix<f64>> {
    let m = basis.orbitals.len();
    let p = terms.projectors.len();
    let _memory = ctx.reserve(
        product(&[m, p, 8, 2])?
            .checked_add(product(&[m, 8, 2])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut accumulator = PointAccumulator::new(ctx, m, p)?;
    let mut phi = vec![0.; m];
    let mut target = vec![0.; p];
    for (index, ch) in species.channels.iter().enumerate() {
        sphere(
            ch.radius.max(species.local_radius),
            0.,
            64,
            24,
            48,
            |offset, r, weight| {
                let point = std::array::from_fn(|i| center[i] + offset[i]);
                for (value, orbital) in phi.iter_mut().zip(&basis.orbitals) {
                    *value = orbital.evaluate(point);
                }
                target.fill(0.);
                for (b, p) in terms
                    .projectors
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.channel == index)
                {
                    let value = ch.projectors[p.radial].value_slope(r).0
                        * harmonic(p.l, p.m, offset)
                        * weight;
                    target[b] = value;
                }
                accumulator.push(&phi, &target)
            },
        )?;
    }
    let values = accumulator.finish()?;
    Ok(DMatrix::from_column_slice(m, p, values.as_slice()))
}
struct GlobalMultipole {
    atom: usize,
    l: usize,
    m: i32,
    q: DMatrix<f64>,
}
/// Atom-independent PAW tables and one-center contractions owned by a
/// calculator. Geometry-dependent projections and Coulomb terms are rebuilt
/// from these same tables, without returning to a language adapter.
pub struct Setup {
    species: Vec<Species>,
    atoms: Vec<usize>,
    basis: Basis,
    species_terms: Vec<onecenter::AtomTerms>,
}

pub fn build(
    ctx: &Context,
    geometry: Geometry,
    grid: SpatialGrid,
    metadata: &str,
    mut arrays: Arrays,
    kinetic: &str,
    tolerance: f64,
) -> Result<MolecularIntegrals> {
    let metadata: serde_json::Value =
        serde_json::from_str(metadata).map_err(|e| invalid(e.to_string()))?;
    if metadata["format"] != "molecular-paw-input" || metadata["version"] != 1 {
        return Err(invalid("unsupported molecular PAW input"));
    }
    if metadata["length_unit"] != "bohr" || metadata["energy_unit"] != "Ha" {
        return Err(invalid("native PAW input must use Bohr and Hartree"));
    }
    if grid.periodic || geometry.pbc().iter().any(|v| *v) {
        return Err(invalid(
            "isolated PAW integrals need nonperiodic geometry and grid",
        ));
    }
    let raw = match metadata["projector_basis"].as_str() {
        Some("raw") => true,
        Some("dual") => false,
        _ => return Err(invalid("PAW projector basis must be raw or dual")),
    };
    let species = metadata["species"]
        .as_array()
        .ok_or_else(|| invalid("missing PAW species"))?
        .iter()
        .map(|s| Species::parse(ctx, s, &mut arrays, raw))
        .collect::<Result<Vec<_>>>()?;
    let atoms = metadata["atoms"]
        .as_array()
        .filter(|v| v.len() == geometry.numbers().len())
        .ok_or_else(|| invalid("PAW atom count disagrees with geometry"))?
        .iter()
        .map(count)
        .collect::<Result<Vec<_>>>()?;
    if atoms.iter().any(|&i| i >= species.len()) {
        return Err(invalid("invalid PAW species index"));
    }
    if atoms
        .iter()
        .enumerate()
        .any(|(a, &s)| species[s].atomic_number != geometry.numbers()[a])
    {
        return Err(invalid("PAW dataset atomic numbers disagree with geometry"));
    }
    let mut orbitals = Vec::new();
    let mut cache = BTreeMap::<String, Arc<Cubic>>::new();
    for entry in metadata["basis"]
        .as_array()
        .ok_or_else(|| invalid("missing PAW basis tables"))?
    {
        let atom = count(&entry["atom"])?;
        let l = count(&entry["l"])?;
        let m = entry["m"]
            .as_i64()
            .and_then(|i| i32::try_from(i).ok())
            .ok_or_else(|| invalid("invalid PAW magnetic quantum number"))?;
        if atom >= atoms.len() {
            return Err(invalid("basis atom outside geometry"));
        }
        let key = entry["values"]
            .as_str()
            .ok_or_else(|| invalid("basis values need an array key"))?
            .to_owned();
        let radial = if let Some(radial) = cache.get(&key) {
            radial.clone()
        } else {
            let r = table(&mut arrays, &entry["r"])?;
            let values = table(&mut arrays, &entry["values"])?;
            if r.as_slice().first().is_none_or(|r| *r < 0.) {
                return Err(invalid("basis radii must be nonnegative"));
            }
            let radial = Arc::new(Cubic::new(ctx, r, values)?);
            cache.insert(key, radial.clone());
            radial
        };
        orbitals.push(Orbital {
            atom,
            l,
            m,
            center: geometry.positions()[atom],
            radial: Radial::Tabulated(radial),
        });
    }
    let basis = Basis::from_orbitals(ctx, orbitals)?;
    let species_terms = species
        .iter()
        .map(|s| onecenter::build(ctx, s))
        .collect::<Result<Vec<_>>>()?;
    let setup = Arc::new(Setup {
        species,
        atoms,
        basis,
        species_terms,
    });
    setup.at_geometry(ctx, geometry, grid, kinetic, tolerance)
}
impl Setup {
    pub fn at_geometry(
        self: &Arc<Self>,
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        kinetic: &str,
        tolerance: f64,
    ) -> Result<MolecularIntegrals> {
        if geometry.numbers().len() != self.atoms.len()
            || self
                .atoms
                .iter()
                .enumerate()
                .any(|(a, &s)| self.species[s].atomic_number != geometry.numbers()[a])
            || grid.periodic
            || geometry.pbc().iter().any(|v| *v)
        {
            return Err(invalid(
                "PAW setup needs the same isolated atomic composition",
            ));
        }
        let mut orbitals = self.basis.orbitals.clone();
        for orbital in &mut orbitals {
            orbital.center = geometry.positions()[orbital.atom];
        }
        let basis = Basis::from_orbitals(ctx, orbitals)?;
        let species = &self.species;
        let atoms = &self.atoms;
        let species_terms = &self.species_terms;
        let m = basis.orbitals.len();
        let g = grid.index.points();
        let c = atoms
            .iter()
            .enumerate()
            .map(|(a, &s)| {
                projection(
                    ctx,
                    &basis,
                    &species[s],
                    &species_terms[s],
                    geometry.positions()[a],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut multipoles = Vec::new();
        for (a, &s) in atoms.iter().enumerate() {
            for ch in &species_terms[s].multipoles {
                multipoles.push(GlobalMultipole {
                    atom: a,
                    l: ch.l,
                    m: ch.m,
                    q: &c[a] * &ch.block * c[a].transpose(),
                });
            }
        }
        let nch = multipoles.len();
        let rank = atoms
            .iter()
            .map(|&s| species_terms[s].factors.len())
            .sum::<usize>()
            .checked_add(2 * nch)
            .ok_or(crate::Error::Overflow)?;
        let memory = ctx.reserve(
            product(&[rank.checked_add(4).ok_or(crate::Error::Overflow)?, m, m, 8])?
                .checked_add(product(&[rank, 64])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let _workspace = ctx.reserve(
            product(&[nch, m, m, 8, 5])?
                .checked_add(product(&[nch, nch, 8, 12])?)
                .ok_or(crate::Error::Overflow)?,
        )?;
        let mut overlap = DMatrix::zeros(m, m);
        let mut v = DMatrix::zeros(m, m);
        let mut ks_difference = DMatrix::zeros(m, m);
        let mut factors = Vec::new();
        let mut weights = Vec::new();
        for (a, &s) in atoms.iter().enumerate() {
            let terms = &species_terms[s];
            overlap += &c[a] * &terms.overlap * c[a].transpose();
            v += &c[a] * &terms.coupling * c[a].transpose();
            ks_difference += &c[a] * &terms.ks_difference * c[a].transpose();
            for (factor, &weight) in terms.factors.iter().zip(&terms.weights) {
                factors.push(&c[a] * factor * c[a].transpose());
                weights.push(weight);
            }
        }
        let sigma = 1.4 * grid.spacings().into_iter().fold(0., f64::max);
        let mut external = ctx.allocate::<f64>(g)?;
        for k in 0..g {
            external.as_mut_slice()[k] = atoms
                .iter()
                .enumerate()
                .map(|(a, &s)| {
                    long_range(
                        species[s].charge,
                        sigma,
                        distance(grid.point(k), geometry.positions()[a]),
                    )
                })
                .sum();
        }
        let mut phi = vec![0.; m];
        let mut weighted = vec![0.; m];
        let mut local = PointAccumulator::new(ctx, m, m)?;
        for (a, &s) in atoms.iter().enumerate() {
            let sp = &species[s];
            let tail = (0..7701)
                .map(|k| 0.3 + 7.7 * k as f64 / 7700.)
                .rfind(|&r| (sp.local(r) + sp.charge / r).abs() > 1e-5)
                .map_or(0., |r| r + 0.01);
            let radius = (6.5 * sigma).max(sp.local_radius).max(tail);
            sphere(radius, sp.local_radius, 64, 32, 64, |offset, r, weight| {
                let point = std::array::from_fn(|i| geometry.positions()[a][i] + offset[i]);
                let weight = weight * (sp.local(r) - long_range(sp.charge, sigma, r));
                for (value, orbital) in phi.iter_mut().zip(&basis.orbitals) {
                    *value = orbital.evaluate(point);
                }
                for (to, from) in weighted.iter_mut().zip(&phi) {
                    *to = weight * from;
                }
                local.push(&phi, &weighted)
            })?;
        }
        let values = local.finish()?;
        v += DMatrix::from_column_slice(m, m, values.as_slice());
        drop(values);
        let samples = basis.sample(ctx, &grid)?;
        let mut ws = vec![DMatrix::zeros(m, m); nch];
        let mut projector = if m >= 16 {
            Some(crate::math::projection::RealProjector::new(
                ctx,
                samples.as_slice(),
                m,
                g,
            )?)
        } else {
            None
        };
        let mut field = ctx.allocate::<f64>(g)?;
        for (a, ch) in multipoles.iter().enumerate() {
            let sp = &species[atoms[ch.atom]];
            for k in 0..g {
                let offset =
                    std::array::from_fn(|i| grid.point(k)[i] - geometry.positions()[ch.atom][i]);
                let weight = potential(distance(offset, [0.; 3]), sp.compensation_radius, ch.l)
                    * harmonic(ch.l, ch.m, offset)
                    * grid.index.dv();
                if projector.is_some() {
                    field.as_mut_slice()[k] = weight;
                } else {
                    for i in 0..m {
                        for j in 0..m {
                            ws[a][(i, j)] += weight
                                * samples.as_slice()[i * g + k]
                                * samples.as_slice()[j * g + k];
                        }
                    }
                }
            }
            if let Some(projector) = projector.as_mut() {
                ws[a]
                    .as_mut_slice()
                    .copy_from_slice(projector.project(field.as_slice())?);
            }
        }
        let mut electrostatic = PointAccumulator::new(ctx, nch, nch)?;
        let mut potentials = vec![0.; nch];
        let mut densities = vec![0.; nch];
        for (b, &s) in atoms.iter().enumerate() {
            let sp = &species[s];
            sphere(
                sp.compensation_radius,
                0.,
                50,
                24,
                48,
                |offset, r, weight| {
                    let point: [f64; 3] =
                        std::array::from_fn(|i| geometry.positions()[b][i] + offset[i]);
                    for (i, ci) in multipoles.iter().enumerate() {
                        let delta =
                            std::array::from_fn(|k| point[k] - geometry.positions()[ci.atom][k]);
                        potentials[i] = potential(
                            distance(delta, [0.; 3]),
                            species[atoms[ci.atom]].compensation_radius,
                            ci.l,
                        ) * harmonic(ci.l, ci.m, delta);
                    }
                    densities.fill(0.);
                    for (j, cj) in multipoles.iter().enumerate().filter(|(_, ch)| ch.atom == b) {
                        densities[j] = shape(r, sp.compensation_radius, cj.l)
                            * harmonic(cj.l, cj.m, offset)
                            * weight;
                    }
                    electrostatic.push(&potentials, &densities)
                },
            )?;
            let mut ionic = vec![0.; nch];
            sphere(
                sp.compensation_radius,
                0.,
                48,
                24,
                48,
                |offset, r, weight| {
                    let point = std::array::from_fn(|i| geometry.positions()[b][i] + offset[i]);
                    let value: f64 = atoms
                        .iter()
                        .enumerate()
                        .filter(|(a, _)| *a != b)
                        .map(|(a, &s)| species[s].local(distance(point, geometry.positions()[a])))
                        .sum();
                    for (i, ch) in multipoles.iter().enumerate().filter(|(_, ch)| ch.atom == b) {
                        ionic[i] += weight
                            * value
                            * shape(r, sp.compensation_radius, ch.l)
                            * harmonic(ch.l, ch.m, offset);
                    }
                    Ok(())
                },
            )?;
            for (ch, value) in multipoles.iter().zip(ionic) {
                v += &ch.q * value;
            }
        }
        let values = electrostatic.finish()?;
        let u = DMatrix::from_column_slice(nch, nch, values.as_slice());
        drop(values);
        let smooth_start = factors.len();
        if nch > 0 {
            let coupled = DMatrix::from_fn(2 * nch, 2 * nch, |a, b| {
                if a < nch && b < nch {
                    0.5 * (u[(a, b)] + u[(b, a)])
                } else if a.abs_diff(b) == nch {
                    1.
                } else {
                    0.
                }
            });
            let (values, vectors) = real_eigen(coupled)?;
            for (k, weight) in values.into_iter().enumerate() {
                if weight == 0. {
                    continue;
                }
                let mut factor = DMatrix::zeros(m, m);
                for a in 0..nch {
                    factor += &multipoles[a].q * vectors[(a, k)] + &ws[a] * vectors[(a + nch, k)];
                }
                factors.push(factor);
                weights.push(weight);
            }
        }
        let charges: Vec<_> = atoms.iter().map(|&s| species[s].charge).collect();
        let count: f64 = charges.iter().sum();
        if count.fract().abs() > 1e-8 || count > usize::MAX as f64 {
            return Err(invalid("PAW valence electron count must be integral"));
        }
        let corrections = MolecularCorrections {
            dft: Some(PawKohnSham {
                one: ks_difference,
                pairs: PairCorrection {
                    factors: factors[smooth_start..].to_vec(),
                    weights: weights[smooth_start..].to_vec(),
                    memory: ctx.reserve(product(&[factors.len() - smooth_start, m, m, 8])?)?,
                },
                core: {
                    let mut core = ctx.allocate::<f64>(g)?;
                    for k in 0..g {
                        core.as_mut_slice()[k] = atoms
                            .iter()
                            .enumerate()
                            .map(|(a, &s)| {
                                let radial = &species[s].xc_core;
                                let r = distance(grid.point(k), geometry.positions()[a]);
                                if r > *radial.knots.as_slice().last().unwrap() {
                                    0.
                                } else {
                                    radial
                                        .value_slope(r.max(radial.knots.as_slice()[0]))
                                        .0
                                        .max(0.)
                                }
                            })
                            .sum();
                    }
                    core
                },
                core_tau: {
                    let mut tau = ctx.allocate::<f64>(g)?;
                    for k in 0..g {
                        tau.as_mut_slice()[k] = atoms
                            .iter()
                            .enumerate()
                            .map(|(a, &s)| {
                                let radial = &species[s].xc_core_tau;
                                let r = distance(grid.point(k), geometry.positions()[a]);
                                if r > *radial.knots.as_slice().last().unwrap() {
                                    0.
                                } else {
                                    radial
                                        .value_slope(r.max(radial.knots.as_slice()[0]))
                                        .0
                                        .max(0.)
                                }
                            })
                            .sum();
                    }
                    tau
                },
                constant: atoms.iter().map(|&s| species[s].dft_constant).sum(),
                relativistic_exchange: species.iter().any(|s| s.relativistic_exchange),
                memory: ctx.reserve(product(&[m, m, 8])?)?,
            }),
            external,
            overlap,
            potential: v,
            pairs: Arc::new(PairCorrection {
                factors,
                weights,
                memory,
            }),
            charges,
            electrons: count.round() as usize,
        };
        let mut integrals = MolecularIntegrals::from_basis(
            ctx,
            geometry,
            grid,
            basis,
            kinetic,
            Some(tolerance),
            Some(corrections),
        )?;
        integrals.paw_setup = Some(self.clone());
        Ok(integrals)
    }
}
