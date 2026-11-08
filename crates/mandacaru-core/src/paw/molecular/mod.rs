//! Isolated PAW-LCAO integrals with exact valence one-center interactions.
//! Dataset decoding and basis recipes are outside this numerical boundary.
pub(super) mod data;
pub mod gradient;
pub(super) mod hubbard;
pub(super) mod onecenter;
mod screened;
pub use screened::{Screened, SpinTerms};
use super::invalid;
use crate::{
    basis::{Basis, Orbital, Radial},
    math::{
        linalg::real_eigen,
        projection::PointAccumulator,
        radial::gauss_legendre,
        spline::Cubic,
        storage::{Precision, Sample},
        SpatialGrid,
    },
    molecular::{
        integrals::{MolecularCorrections, PairCorrection, PawKohnSham},
        MolecularIntegrals,
    },
    product, Buffer, Context, Geometry, Result,
};
pub use data::Arrays;
use crate::paw::screened_atom::ScreenedAtom;
use data::{count, table, Species};
use nalgebra::DMatrix;
use crate::paw::pair_tables::{angular_rule, harmonic, potential, shape};
use std::{collections::BTreeMap, f64::consts::PI, sync::Arc};

/// The quadrature rules `[radial, polar, azimuthal]` of the spheres around an atom, shared by
/// the setup and its analytic gradient so that the gradient differentiates the rule in use.
pub(crate) const PROJECTION: [usize; 3] = [64, 24, 48];
pub(crate) const LOCAL: [usize; 3] = [64, 32, 64];
pub(crate) const ELECTROSTATIC: [usize; 3] = [50, 24, 48];
pub(crate) const IONIC: [usize; 3] = [48, 24, 48];

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
/// The quadrature of the projections of one atom, offsets from its center: the
/// projector values of every channel's sphere, already carrying the quadrature
/// weight, one entry per projector of the atom.
fn projection_points(
    species: &Species,
    terms: &onecenter::AtomTerms,
    mut visit: impl FnMut([f64; 3], &[f64]) -> Result<()>,
) -> Result<()> {
    let mut target = vec![0.; terms.projectors.len()];
    for (index, ch) in species.channels.iter().enumerate() {
        sphere(
            ch.radius.max(species.local_radius),
            0.,
            PROJECTION[0],
            PROJECTION[1],
            PROJECTION[2],
            |offset, r, weight| {
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
                visit(offset, &target)
            },
        )?;
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
    let max_radius = species
        .channels
        .iter()
        .map(|ch| ch.radius.max(species.local_radius))
        .fold(0.0f64, f64::max);
    let active_orbitals: Vec<(usize, f64)> = basis
        .orbitals
        .iter()
        .enumerate()
        .filter_map(|(i, orbital)| {
            let r_cut = match &orbital.radial {
                crate::basis::Radial::Tabulated(table) => {
                    *table.knots.as_slice().last().unwrap_or(&f64::INFINITY)
                }
                _ => f64::INFINITY,
            };
            let d = distance(center, orbital.center);
            if d <= max_radius + r_cut {
                Some((i, if r_cut.is_finite() { r_cut * r_cut } else { f64::INFINITY }))
            } else {
                None
            }
        })
        .collect();
    projection_points(species, terms, |offset, target| {
        if target.iter().all(|&t| t == 0.) {
            return Ok(());
        }
        let point = std::array::from_fn(|i| center[i] + offset[i]);
        phi.fill(0.);
        for &(i, support2) in &active_orbitals {
            let orbital = &basis.orbitals[i];
            let r2 = (point[0] - orbital.center[0]).powi(2)
                + (point[1] - orbital.center[1]).powi(2)
                + (point[2] - orbital.center[2]).powi(2);
            if r2 <= support2 {
                phi[i] = orbital.evaluate(point);
            }
        }
        accumulator.push(&phi, target)
    })?;
    let values = accumulator.finish()?;
    Ok(DMatrix::from_column_slice(m, p, values.as_slice()))
}
/// Radius of the sphere in which the short-range local potential of a species
/// is integrated: the Gaussian-filtered compensation tail, the dataset's local
/// radius or the radius past which its potential is its long-range tail.
fn local_sphere_radius(species: &Species, sigma: f64) -> f64 {
    let tail = (0..7701)
        .map(|k| 0.3 + 7.7 * k as f64 / 7700.)
        .rfind(|&r| (species.local(r) + species.charge / r).abs() > 1e-5)
        .map_or(0., |r| r + 0.01);
    (6.5 * sigma).max(species.local_radius).max(tail)
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
    /// The species-level terms of a screened hybrid, built once per screening.
    screened: std::sync::Mutex<Vec<(f64, Arc<Vec<ScreenedAtom>>)>>,
    hubbard: Option<crate::models::hubbard_request::Request>,
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    ctx: &Context,
    geometry: Geometry,
    grid: SpatialGrid,
    metadata: &str,
    mut arrays: Arrays,
    kinetic: &str,
    tolerance: f64,
    precision: Precision,
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
    let allow_heavy_dft = metadata.get("allow_heavy_dft").and_then(|v| v.as_bool()).unwrap_or(false);
    let hubbard = match metadata.get("hubbard") {
        Some(serde_json::Value::String(s)) => Some(crate::models::hubbard_request::Request::parse(s)?),
        _ => None,
    };
    let species = metadata["species"]
        .as_array()
        .ok_or_else(|| invalid("missing PAW species"))?
        .iter()
        .map(|s| {
            let allow_heavy = allow_heavy_dft || s.get("allow_heavy_dft").and_then(|v| v.as_bool()).unwrap_or(false);
            if let Some(file) = s.get("file").and_then(|f| f.as_str()) {
                let ds = crate::paw::record::PawDataset::read(ctx, std::path::Path::new(file))?;
                Species::from_dataset(ctx, &ds, raw, false, allow_heavy)
            } else {
                Species::parse_with_options(ctx, s, &mut arrays, raw, false, allow_heavy)
            }
        })
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
        screened: std::sync::Mutex::new(Vec::new()),
        hubbard,
    });
    setup.at_geometry(ctx, geometry, grid, kinetic, tolerance, precision)
}
impl Setup {
    pub fn at_geometry(
        self: &Arc<Self>,
        ctx: &Context,
        geometry: Geometry,
        grid: SpatialGrid,
        kinetic: &str,
        tolerance: f64,
        precision: Precision,
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
        let c = if atoms.len() <= 1 || ctx.threads() <= 1 {
            atoms
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
                .collect::<Result<Vec<_>>>()?
        } else {
            let basis_ref = &basis;
            let geom_ref = &geometry;
            ctx.run(|| {
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(atoms.len());
                    for (a, &s) in atoms.iter().enumerate() {
                        let sp = &species[s];
                        let st = &species_terms[s];
                        let pos = geom_ref.positions()[a];
                        handles.push(mandacaru_kernels::runtime::spawn(scope, move || {
                            projection(ctx, basis_ref, sp, st, pos)
                        }));
                    }
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Result<Vec<_>>>()
                })
            })?
        };
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
        let workers = ctx.threads().max(1).min(g.max(1));
        if workers <= 1 {
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
        } else {
            let chunk = g.div_ceil(workers);
            let ext_slice = external.as_mut_slice();
            let grid_ref = &grid;
            let geom_ref = &geometry;
            let species_ref = species;
            let atoms_ref = atoms;
            ctx.run(|| {
                std::thread::scope(|scope| {
                    for (w, part) in ext_slice.chunks_mut(chunk).enumerate() {
                        let start = w * chunk;
                        mandacaru_kernels::runtime::spawn(scope, move || {
                            for (i, val) in part.iter_mut().enumerate() {
                                let k = start + i;
                                *val = atoms_ref
                                    .iter()
                                    .enumerate()
                                    .map(|(a, &s)| {
                                        long_range(
                                            species_ref[s].charge,
                                            sigma,
                                            distance(grid_ref.point(k), geom_ref.positions()[a]),
                                        )
                                    })
                                    .sum();
                            }
                        });
                    }
                });
            });
        }
        let values = if atoms.len() <= 1 || ctx.threads() <= 1 {
            let mut phi = vec![0.; m];
            let mut weighted = vec![0.; m];
            let mut local = PointAccumulator::new(ctx, m, m)?;
            for (a, &s) in atoms.iter().enumerate() {
                let sp = &species[s];
                let radius = local_sphere_radius(sp, sigma);
                let active_orbitals: Vec<(usize, f64)> = basis
                    .orbitals
                    .iter()
                    .enumerate()
                    .filter_map(|(i, orbital)| {
                        let r_cut = match &orbital.radial {
                            crate::basis::Radial::Tabulated(table) => {
                                *table.knots.as_slice().last().unwrap_or(&f64::INFINITY)
                            }
                            _ => f64::INFINITY,
                        };
                        let d = distance(geometry.positions()[a], orbital.center);
                        if d <= radius + r_cut {
                            Some((i, if r_cut.is_finite() { r_cut * r_cut } else { f64::INFINITY }))
                        } else {
                            None
                        }
                    })
                    .collect();
                sphere(radius, sp.local_radius, LOCAL[0], LOCAL[1], LOCAL[2], |offset, r, weight| {
                    let g_r = sp.local(r) - long_range(sp.charge, sigma, r);
                    if g_r.abs() < 1e-15 {
                        return Ok(());
                    }
                    let weight = weight * g_r;
                    let point = std::array::from_fn(|i| geometry.positions()[a][i] + offset[i]);
                    phi.fill(0.);
                    for &(i, support2) in &active_orbitals {
                        let orbital = &basis.orbitals[i];
                        let r2 = (point[0] - orbital.center[0]).powi(2)
                            + (point[1] - orbital.center[1]).powi(2)
                            + (point[2] - orbital.center[2]).powi(2);
                        if r2 <= support2 {
                            phi[i] = orbital.evaluate(point);
                        }
                    }
                    for &(i, _) in &active_orbitals {
                        weighted[i] = weight * phi[i];
                    }
                    local.push(&phi, &weighted)
                })?;
            }
            local.finish()?
        } else {
            let basis_ref = &basis;
            let geom_ref = &geometry;
            ctx.run(|| {
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(atoms.len());
                    for (a, &s) in atoms.iter().enumerate() {
                        let sp = &species[s];
                        let pos = geom_ref.positions()[a];
                        handles.push(mandacaru_kernels::runtime::spawn(scope, move || {
                            let mut phi = vec![0.; m];
                            let mut weighted = vec![0.; m];
                            let mut local = PointAccumulator::new(ctx, m, m)?;
                            let radius = local_sphere_radius(sp, sigma);
                            let active_orbitals: Vec<(usize, f64)> = basis_ref
                                .orbitals
                                .iter()
                                .enumerate()
                                .filter_map(|(i, orbital)| {
                                    let r_cut = match &orbital.radial {
                                        crate::basis::Radial::Tabulated(table) => {
                                            *table.knots.as_slice().last().unwrap_or(&f64::INFINITY)
                                        }
                                        _ => f64::INFINITY,
                                    };
                                    let d = distance(pos, orbital.center);
                                    if d <= radius + r_cut {
                                        Some((i, if r_cut.is_finite() { r_cut * r_cut } else { f64::INFINITY }))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            sphere(radius, sp.local_radius, LOCAL[0], LOCAL[1], LOCAL[2], |offset, r, weight| {
                                let g_r = sp.local(r) - long_range(sp.charge, sigma, r);
                                if g_r.abs() < 1e-15 {
                                    return Ok(());
                                }
                                let weight = weight * g_r;
                                let point = std::array::from_fn(|i| pos[i] + offset[i]);
                                phi.fill(0.);
                                for &(i, support2) in &active_orbitals {
                                    let orbital = &basis_ref.orbitals[i];
                                    let r2 = (point[0] - orbital.center[0]).powi(2)
                                        + (point[1] - orbital.center[1]).powi(2)
                                        + (point[2] - orbital.center[2]).powi(2);
                                    if r2 <= support2 {
                                        phi[i] = orbital.evaluate(point);
                                    }
                                }
                                for &(i, _) in &active_orbitals {
                                    weighted[i] = weight * phi[i];
                                }
                                local.push(&phi, &weighted)
                            })?;
                            local.finish()
                        }));
                    }
                    let mut sum_buf = ctx.allocate::<f64>(m * m)?;
                    for h in handles {
                        let partial = h.join().unwrap()?;
                        for (tot, val) in sum_buf.as_mut_slice().iter_mut().zip(partial.as_slice()) {
                            *tot += val;
                        }
                    }
                    Ok(sum_buf)
                })
            })?
        };
        v += DMatrix::from_column_slice(m, m, values.as_slice());
        drop(values);
        // The stack is built once, in the calculation's precision, and handed
        // to the integrals below; the projector and its field scratch end with
        // this block.
        let samples = basis.sample(ctx, &grid, precision)?;
        let mut ws = vec![DMatrix::zeros(m, m); nch];
        crate::math::storage::with_rows!(samples.rows(), |stored| {
            let mut projector = if m >= 16 {
                Some(crate::math::projection::RealProjector::new(
                    ctx, stored, m, g,
                )?)
            } else {
                None
            };
            let mut field = ctx.allocate::<f64>(g)?;
            let workers = ctx.threads().max(1).min(g.max(1));
            let chunk = g.div_ceil(workers);
            for (a, ch) in multipoles.iter().enumerate() {
                let sp = &species[atoms[ch.atom]];
                let ch_atom_pos = geometry.positions()[ch.atom];
                let dv = grid.index.dv();
                let field_slice = field.as_mut_slice();
                if workers <= 1 {
                    for k in 0..g {
                        let offset = std::array::from_fn(|i| {
                            grid.point(k)[i] - ch_atom_pos[i]
                        });
                        field_slice[k] = potential(distance(offset, [0.; 3]), sp.compensation_radius, ch.l)
                            * harmonic(ch.l, ch.m, offset)
                            * dv;
                    }
                } else {
                    let grid_ref = &grid;
                    ctx.run(|| {
                        std::thread::scope(|scope| {
                            for (w, part) in field_slice.chunks_mut(chunk).enumerate() {
                                let start = w * chunk;
                                mandacaru_kernels::runtime::spawn(scope, move || {
                                    for (i, val) in part.iter_mut().enumerate() {
                                        let k = start + i;
                                        let offset = std::array::from_fn(|dim| {
                                            grid_ref.point(k)[dim] - ch_atom_pos[dim]
                                        });
                                        *val = potential(distance(offset, [0.; 3]), sp.compensation_radius, ch.l)
                                            * harmonic(ch.l, ch.m, offset)
                                            * dv;
                                    }
                                });
                            }
                        });
                    });
                }
                if let Some(projector) = projector.as_mut() {
                    ws[a]
                        .as_mut_slice()
                        .copy_from_slice(projector.project(field.as_slice())?);
                } else {
                    for i in 0..m {
                        for j in 0..m {
                            for k in 0..g {
                                ws[a][(i, j)] +=
                                    field_slice[k] * stored[i * g + k].widen() * stored[j * g + k].widen();
                            }
                        }
                    }
                }
            }
        });
        let mut total_ionic = vec![0.; nch];
        let (u, ionic) = if atoms.len() <= 1 || ctx.threads() <= 1 {
            let mut electrostatic = PointAccumulator::new(ctx, nch, nch)?;
            let mut potentials = vec![0.; nch];
            let mut densities = vec![0.; nch];
            let mut ionic = vec![0.; nch];
            for (b, &s) in atoms.iter().enumerate() {
                let sp = &species[s];
                sphere(
                    sp.compensation_radius,
                    0.,
                    ELECTROSTATIC[0],
                    ELECTROSTATIC[1],
                    ELECTROSTATIC[2],
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
                sphere(
                    sp.compensation_radius,
                    0.,
                    IONIC[0],
                    IONIC[1],
                    IONIC[2],
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
            }
            let values = electrostatic.finish()?;
            let u = DMatrix::from_column_slice(nch, nch, values.as_slice());
            (u, ionic)
        } else {
            let geom_ref = &geometry;
            let multipoles_ref = &multipoles;
            ctx.run(|| {
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(atoms.len());
                    for (b, &s) in atoms.iter().enumerate() {
                        let sp = &species[s];
                        let pos = geom_ref.positions()[b];
                        handles.push(mandacaru_kernels::runtime::spawn(scope, move || {
                            let mut electrostatic = PointAccumulator::new(ctx, nch, nch)?;
                            let mut potentials = vec![0.; nch];
                            let mut densities = vec![0.; nch];
                            let mut ionic = vec![0.; nch];
                            sphere(
                                sp.compensation_radius,
                                0.,
                                ELECTROSTATIC[0],
                                ELECTROSTATIC[1],
                                ELECTROSTATIC[2],
                                |offset, r, weight| {
                                    let point: [f64; 3] =
                                        std::array::from_fn(|i| pos[i] + offset[i]);
                                    for (i, ci) in multipoles_ref.iter().enumerate() {
                                        let delta =
                                            std::array::from_fn(|k| point[k] - geom_ref.positions()[ci.atom][k]);
                                        potentials[i] = potential(
                                            distance(delta, [0.; 3]),
                                            species[atoms[ci.atom]].compensation_radius,
                                            ci.l,
                                        ) * harmonic(ci.l, ci.m, delta);
                                    }
                                    densities.fill(0.);
                                    for (j, cj) in multipoles_ref.iter().enumerate().filter(|(_, ch)| ch.atom == b) {
                                        densities[j] = shape(r, sp.compensation_radius, cj.l)
                                            * harmonic(cj.l, cj.m, offset)
                                            * weight;
                                    }
                                    electrostatic.push(&potentials, &densities)
                                },
                            )?;
                            sphere(
                                sp.compensation_radius,
                                0.,
                                IONIC[0],
                                IONIC[1],
                                IONIC[2],
                                |offset, r, weight| {
                                    let point = std::array::from_fn(|i| pos[i] + offset[i]);
                                    let value: f64 = atoms
                                        .iter()
                                        .enumerate()
                                        .filter(|(a, _)| *a != b)
                                        .map(|(a, &s)| species[s].local(distance(point, geom_ref.positions()[a])))
                                        .sum();
                                    for (i, ch) in multipoles_ref.iter().enumerate().filter(|(_, ch)| ch.atom == b) {
                                        ionic[i] += weight
                                            * value
                                            * shape(r, sp.compensation_radius, ch.l)
                                            * harmonic(ch.l, ch.m, offset);
                                    }
                                    Ok(())
                                },
                            )?;
                            let values = electrostatic.finish()?;
                            Ok((values, ionic))
                        }));
                    }
                    let mut u_mat = DMatrix::zeros(nch, nch);
                    let mut tot_ionic = vec![0.; nch];
                    for h in handles {
                        let (values, ion) = h.join().unwrap()?;
                        let partial = DMatrix::from_column_slice(nch, nch, values.as_slice());
                        u_mat += partial;
                        for (tot, val) in tot_ionic.iter_mut().zip(&ion) {
                            *tot += val;
                        }
                    }
                    Ok((u_mat, tot_ionic))
                })
            })?
        };
        for (i, (ch, &value)) in multipoles.iter().zip(&ionic).enumerate() {
            v += &ch.q * value;
            total_ionic[i] += value;
        }
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
                    let shape = grid.index.shape();
                    let [_, ny, nz] = shape;
                    let inv_step = grid.step.try_inverse().unwrap_or(nalgebra::Matrix3::identity());
                    let row_norms = [
                        (inv_step[(0, 0)].powi(2) + inv_step[(0, 1)].powi(2) + inv_step[(0, 2)].powi(2)).sqrt(),
                        (inv_step[(1, 0)].powi(2) + inv_step[(1, 1)].powi(2) + inv_step[(1, 2)].powi(2)).sqrt(),
                        (inv_step[(2, 0)].powi(2) + inv_step[(2, 1)].powi(2) + inv_step[(2, 2)].powi(2)).sqrt(),
                    ];
                    for (a, &s) in atoms.iter().enumerate() {
                        let radial = &species[s].xc_core;
                        let r_cut = *radial.knots.as_slice().last().unwrap_or(&0.0);
                        if r_cut <= 0. {
                            continue;
                        }
                        let center = geometry.positions()[a];
                        let c_center = inv_step * (nalgebra::Vector3::from(center) - nalgebra::Vector3::from(grid.origin));
                        let mut min_idx = [0; 3];
                        let mut max_idx = [0; 3];
                        let mut inside = true;
                        for d in 0..3 {
                            let delta = r_cut * row_norms[d];
                            let mn = ((c_center[d] - delta).floor() as isize).max(0) as usize;
                            let mx = (c_center[d] + delta).ceil() as isize;
                            if mx < 0 {
                                inside = false;
                                break;
                            }
                            let mx = (mx as usize).min(shape[d] - 1);
                            if mn > mx {
                                inside = false;
                                break;
                            }
                            min_idx[d] = mn;
                            max_idx[d] = mx;
                        }
                        if !inside {
                            continue;
                        }
                        for ix in min_idx[0]..=max_idx[0] {
                            for iy in min_idx[1]..=max_idx[1] {
                                let base = (ix * ny + iy) * nz;
                                for iz in min_idx[2]..=max_idx[2] {
                                    let k = base + iz;
                                    let r = distance(grid.point(k), center);
                                    if r <= r_cut {
                                        core.as_mut_slice()[k] += radial
                                            .value_slope(r.max(radial.knots.as_slice()[0]))
                                            .0
                                            .max(0.);
                                    }
                                }
                            }
                        }
                    }
                    core
                },
                core_tau: {
                    let mut tau = ctx.allocate::<f64>(g)?;
                    let shape = grid.index.shape();
                    let [_, ny, nz] = shape;
                    let inv_step = grid.step.try_inverse().unwrap_or(nalgebra::Matrix3::identity());
                    let row_norms = [
                        (inv_step[(0, 0)].powi(2) + inv_step[(0, 1)].powi(2) + inv_step[(0, 2)].powi(2)).sqrt(),
                        (inv_step[(1, 0)].powi(2) + inv_step[(1, 1)].powi(2) + inv_step[(1, 2)].powi(2)).sqrt(),
                        (inv_step[(2, 0)].powi(2) + inv_step[(2, 1)].powi(2) + inv_step[(2, 2)].powi(2)).sqrt(),
                    ];
                    for (a, &s) in atoms.iter().enumerate() {
                        let radial = &species[s].xc_core_tau;
                        let r_cut = *radial.knots.as_slice().last().unwrap_or(&0.0);
                        if r_cut <= 0. {
                            continue;
                        }
                        let center = geometry.positions()[a];
                        let c_center = inv_step * (nalgebra::Vector3::from(center) - nalgebra::Vector3::from(grid.origin));
                        let mut min_idx = [0; 3];
                        let mut max_idx = [0; 3];
                        let mut inside = true;
                        for d in 0..3 {
                            let delta = r_cut * row_norms[d];
                            let mn = ((c_center[d] - delta).floor() as isize).max(0) as usize;
                            let mx = (c_center[d] + delta).ceil() as isize;
                            if mx < 0 {
                                inside = false;
                                break;
                            }
                            let mx = (mx as usize).min(shape[d] - 1);
                            if mn > mx {
                                inside = false;
                                break;
                            }
                            min_idx[d] = mn;
                            max_idx[d] = mx;
                        }
                        if !inside {
                            continue;
                        }
                        for ix in min_idx[0]..=max_idx[0] {
                            for iy in min_idx[1]..=max_idx[1] {
                                let base = (ix * ny + iy) * nz;
                                for iz in min_idx[2]..=max_idx[2] {
                                    let k = base + iz;
                                    let r = distance(grid.point(k), center);
                                    if r <= r_cut {
                                        tau.as_mut_slice()[k] += radial
                                            .value_slope(r.max(radial.knots.as_slice()[0]))
                                            .0
                                            .max(0.);
                                    }
                                }
                            }
                        }
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
            core_supported: species.iter().all(|s| s.core_supported),
            hubbard: self.hubbard.as_ref().and_then(|req| {
                hubbard::build(req, &species, &species_terms, &atoms, &geometry, &c, m).ok().flatten()
            }),
            projections: Some(c),
            electrostatic: Some(u),
            ionic: Some(total_ionic),
        };
        let mut integrals = MolecularIntegrals::from_basis(
            ctx,
            geometry,
            grid,
            basis,
            samples,
            kinetic,
            Some(tolerance),
            Some(corrections),
        )?;
        integrals.paw_setup = Some(self.clone());
        Ok(integrals)
    }
}
