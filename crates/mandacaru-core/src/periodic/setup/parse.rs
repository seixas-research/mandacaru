//! Parsing of periodic PAW setup specifications from JSON or structured data.
//!
//! Converts geometry (cell, atom positions, species), k-point grid, and options into
//! the Rust types needed to construct a [`super::PeriodicSetup`].

use super::{
    ions::LocalPotential,
    shells::{Radial, Shell, Shells},
    spheres::Projector,
    symmetry::{Geometry, Symmetry},
    PeriodicSetup, Pins,
};
use crate::{
    basis::filter,
    math::SpatialGrid,
    paw::{
        confinement::{orbital, ChannelData, Prepared},
        record::PawDataset,
    },
    periodic::invalid,
    Context, Result,
};
use nalgebra::Matrix3;
use serde_json::Value;
use std::sync::Arc;

/// A parsed atom specification.
#[derive(Clone, Debug)]
pub struct AtomSpec {
    pub species: usize,
    pub symbol: String,
    pub fractional: [f64; 3],
    pub moment: f64,
}

/// Structured input to construct a [`PeriodicSetup`].
#[derive(Clone, Debug)]
pub struct SetupInput {
    pub lattice: Matrix3<f64>,
    pub grid_shape: [usize; 3],
    pub atoms: Vec<AtomSpec>,
    pub kpoints_frac: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub functional: String,
    pub precision: String,
    /// Basis size, such as SZ, DZP or TZ2P.
    pub basis_size: String,
    pub spinor: bool,
    /// Request a collinear two-spin calculation even when moments are zero.
    pub spin: bool,
    pub symmetry: bool,
    pub filter_cutoff: Option<f64>,
    pub sigma: Option<f64>,
    pub dataset_files: Option<Vec<(String, String)>>,
    pub hubbard_u: Option<crate::models::hubbard_request::Request>,
}

/// Construct a [`PeriodicSetup`] from structured inputs and loaded datasets.
pub fn create_setup(
    ctx: &Context,
    input: SetupInput,
    datasets: Vec<Arc<PawDataset>>,
) -> Result<PeriodicSetup> {
    if input.atoms.is_empty() {
        return Err(invalid("periodic setup needs at least one atom"));
    }
    if input.kpoints_frac.is_empty() {
        return Err(invalid("periodic setup needs at least one k-point"));
    }
    if crate::models::functional(&input.functional)?.screened_exchange.is_some() {
        return Err(crate::Error::Unsupported(
            "native periodic setup does not yet build hybrid PAW spheres".into(),
        ));
    }
    if let Some(request) = &input.hubbard_u {
        request.require_scope(&input.functional, &input.precision)?;
    }
    let (zeta_count, polarization_count) = match input.basis_size.to_ascii_uppercase().as_str() {
        "SZ" => (1, 0), "SZP" => (1, 1),
        "DZ" => (2, 0), "DZP" => (2, 1), "DZ2P" => (2, 2),
        "TZ" => (3, 0), "TZP" => (3, 1), "TZ2P" => (3, 2),
        "QZ" => (4, 0), "QZP" => (4, 1), "QZ2P" => (4, 2),
        _ => return Err(crate::Error::Unsupported(
            "unsupported native periodic basis size".into(),
        )),
    };
    if input.grid_shape.contains(&0)
        || input.lattice.iter().any(|x| !x.is_finite())
        || input.lattice.determinant().abs() < 1e-12
        || input.atoms.iter().any(|a| {
            !a.moment.is_finite()
                || a.fractional.iter().any(|x| !x.is_finite())
        })
    {
        return Err(invalid("invalid lattice, grid or atom coordinates"));
    }
    if input.kpoints_frac.len() != input.weights.len()
        || input.kpoints_frac.iter().flatten().any(|x| !x.is_finite())
        || input.weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        || (input.weights.iter().sum::<f64>() - 1.).abs() > 1e-10
    {
        return Err(invalid("k-point coordinates and positive normalized weights must agree"));
    }
    if input.atoms.iter().any(|a| a.species >= datasets.len()) {
        return Err(invalid("atom dataset index is out of range"));
    }
    if input.atoms.iter().any(|a| !a.symbol.is_empty() && a.symbol != datasets[a.species].symbol) {
        return Err(invalid("atom symbol does not match its PAW dataset"));
    }
    let has_spin_orbit = input.atoms.iter().any(|a| !datasets[a.species].spin_orbit.is_empty());
    if let Some(request) = &input.hubbard_u {
        if has_spin_orbit {
            crate::periodic::hubbard_table::check_datasets(request, true, &[])?;
        }
    }
    if has_spin_orbit && !input.spinor {
        return Err(crate::Error::Unsupported(
            "native periodic setup cannot omit a dataset's spin-orbit operators".into(),
        ));
    }
    if input.spinor && !has_spin_orbit {
        return Err(crate::Error::Unsupported(
            "spinor setup needs a dataset with spin-orbit projectors".into(),
        ));
    }
    let relativistic = datasets[input.atoms[0].species].relativistic_exchange;
    if input.atoms.iter().any(|a| datasets[a.species].relativistic_exchange != relativistic) {
        return Err(invalid("datasets mix relativistic and non-relativistic exchange"));
    }
    if input.filter_cutoff.is_some_and(|v| !v.is_finite() || v <= 0.)
        || input.sigma.is_some_and(|v| !v.is_finite() || v <= 0.)
    {
        return Err(invalid("filter cutoff and sigma must be finite and positive"));
    }
    if input.dataset_files.as_ref().is_some_and(|files| {
        files.len() != input.atoms.len() && files.len() != datasets.len()
    }) {
        return Err(invalid("dataset provenance must match atoms or species"));
    }

    let n_atoms = input.atoms.len();
    let shape = input.grid_shape;

    // Grid step vectors are columns of lattice divided by node counts
    let mut step = [0.; 9];
    for row in 0..3 {
        for col in 0..3 {
            step[row * 3 + col] = input.lattice[(row, col)] / shape[col] as f64;
        }
    }
    let grid = SpatialGrid::new(shape, [0.; 3], step, true)?;

    let spacing = grid.step.column(0).norm()
        .max(grid.step.column(1).norm())
        .max(grid.step.column(2).norm());

    let k_c = input.filter_cutoff.unwrap_or_else(|| {
        std::f64::consts::PI / spacing.max(1e-12)
    });

    let sigma = input.sigma.unwrap_or(1.4 * spacing);

    // Compute Cartesian positions and centers
    let mut centers = Vec::with_capacity(n_atoms);
    let mut fractional_atoms = Vec::with_capacity(n_atoms);
    let mut atoms_species = Vec::with_capacity(n_atoms);
    let mut atomic_numbers = Vec::with_capacity(n_atoms);
    let mut moments = Vec::with_capacity(n_atoms);

    for a in &input.atoms {
        let f = nalgebra::Vector3::new(a.fractional[0], a.fractional[1], a.fractional[2]);
        let c = input.lattice * f;
        centers.push([c[0], c[1], c[2]]);
        fractional_atoms.push(a.fractional);
        atoms_species.push(a.species);
        let z = datasets[a.species].atomic_number;
        atomic_numbers.push(z);
        moments.push(a.moment);
    }

    // Build minimal SZ basis from datasets
    let mut shells_vec = Vec::new();
    let mut tables = Vec::new();
    let mut function_atom = Vec::new();
    let mut function_m = Vec::new();
    let mut function_table = Vec::new();
    let mut row_idx = 0;
    // Radial data are independent of the atom's position. Share the splines and
    // description tables for every atom using the same dataset.
    let mut species_shells: Vec<Option<Vec<(usize, usize, Arc<Radial>)>>> =
        (0..datasets.len()).map(|_| None).collect();

    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let center = centers[atom_idx];

        if species_shells[species_idx].is_none() {
            let mut cached = Vec::new();
            let mut states = Vec::new();
            for (&l, channel) in &ds.channels {
                for (state, &occupation) in channel.occupations.data().iter().enumerate() {
                    if occupation > 0. {
                        states.push((l, state));
                    }
                }
            }
            // Empty bound channels follow all occupied states, as in the
            // active basis builder. Li and Be retain their p channel solely
            // for scattering, even though it has a bound level.
            for (&l, channel) in &ds.channels {
                if channel.occupations.data().iter().any(|&o| o > 0.)
                    || (l == 1 && (ds.symbol == "Li" || ds.symbol == "Be"))
                {
                    continue;
                }
                if empty_channel_level(ctx, ds, l)? < -1e-3 {
                    states.push((l, 0));
                }
            }
            let mut valence = std::collections::BTreeMap::new();
            let mut cutoffs = std::collections::BTreeMap::new();
            for (l, state) in states {
                let l_usize = l as usize;
                let orbital = confined_orbital(ctx, ds, l, state, 0.1)?;
                let n = ds.channels[&l].n as i64 + state as i64;
                let semicore = ds.semicore_subshells.iter()
                    .any(|shell| shell.len() >= 2 && shell[0] == n && shell[1] == l as i64);
                if !semicore {
                    valence.insert(l, state);
                    cutoffs.insert((l, state), orbital.r_c);
                }
                let count = if semicore { 1 } else { zeta_count };
                let zetas = crate::basis::radial::zeta_values(
                    ctx, ds.r.as_slice(), &orbital.radial, l_usize, count, 0.15,
                    Some(&[0.15, 0.31, 0.54]),
                )?;
                for zeta in zetas {
                    let (filtered, _) = filter::filter(
                        ctx, ds.r.as_slice(), zeta.as_slice(), l_usize, k_c,
                        &filter::Options::default(),
                    )?;
                    let table_idx = tables.len();
                    let radial = Arc::new(Radial::new(ctx, ds.r.as_slice(), &filtered, None)?);
                    tables.push(crate::periodic::description::Table {
                        l: l_usize, r: ds.r.as_slice().to_vec(), values: filtered,
                        confinement: None,
                    });
                    cached.push((l_usize, table_idx, radial));
                }
            }
            if polarization_count > 0 {
                let max_l = *valence.keys().max()
                    .ok_or_else(|| invalid("polarization needs a valence channel"))?;
                let l = (0..=max_l).find(|l| !valence.contains_key(l)).unwrap_or(max_l + 1);
                let base = l.checked_sub(1)
                    .and_then(|l| valence.get(&l).map(|&state| (l, state)))
                    .ok_or_else(|| invalid("polarization needs the preceding valence channel"))?;
                let r_cut = cutoffs[&base];
                let reference = confined_orbital(ctx, ds, base.0, base.1, 0.3)?.r_c;
                let shape = crate::basis::pao::polarization_radial(
                    ds.r.as_slice(), l as usize, r_cut, 0.25 * reference,
                )?;
                let zetas = crate::basis::radial::zeta_values_by_fraction(
                    ctx, ds.r.as_slice(), &shape, l as usize, polarization_count,
                    0.15, Some(&[0.15, 0.31, 0.54]),
                )?;
                for zeta in zetas {
                    let (filtered, _) = filter::filter(
                        ctx, ds.r.as_slice(), zeta.as_slice(), l as usize, k_c,
                        &filter::Options::default(),
                    )?;
                    let table_idx = tables.len();
                    let radial = Arc::new(Radial::new(ctx, ds.r.as_slice(), &filtered, None)?);
                    tables.push(crate::periodic::description::Table {
                        l: l as usize, r: ds.r.as_slice().to_vec(), values: filtered,
                        confinement: None,
                    });
                    cached.push((l as usize, table_idx, radial));
                }
            }
            species_shells[species_idx] = Some(cached);
        }
        for &(l_usize, table_idx, ref radial) in species_shells[species_idx].as_ref().unwrap() {
            let ms: Vec<i32> = (-(l_usize as i32)..=l_usize as i32).collect();
            let n_m = ms.len();
            let rows: Vec<usize> = (row_idx..row_idx + n_m).collect();
            row_idx += n_m;

            for &m in &ms {
                function_atom.push(atom_idx);
                function_m.push(m);
                function_table.push(table_idx);
            }

            shells_vec.push(Shell {
                center,
                l: l_usize,
                ms,
                rows,
                radial: radial.clone(),
            });
        }
    }

    // The j-resolved spin-orbit projectors are distinct from the scalar raw
    // projectors. Retain their own radial tables and coupling blocks.
    let mut spin_orbit_projectors = Vec::new();
    let mut spin_orbit_couplings = std::collections::BTreeMap::new();
    if input.spinor {
        for (atom, &species) in atoms_species.iter().enumerate() {
            let ds = &datasets[species];
            for (&l, entry) in &ds.spin_orbit {
                let channel = ds.channels.get(&l)
                    .ok_or_else(|| invalid("spin-orbit channel has no scalar channel"))?;
                let radials: Vec<_> = entry.projectors.iter().map(|t| {
                    Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).map(Arc::new)
                }).collect::<Result<_>>()?;
                for m in -(l as i32)..=l as i32 {
                    for (index, radial) in radials.iter().enumerate() {
                        spin_orbit_projectors.push(Projector {
                            atom, l: l as usize, m, index, center: centers[atom],
                            r_cut: channel.r_cut.max(ds.r_cut_local), radial: radial.clone(),
                        });
                    }
                }
                spin_orbit_couplings.insert((atom, l as usize), super::spin_orbit::Coupling {
                    radial: entry.coupling.shape()[0], values: entry.coupling.data().to_vec(),
                });
            }
        }
    }
    let shells = Shells::new(shells_vec)?;

    // Build Projectors list
    let mut projectors = Vec::new();
    let mut proj_tuples = Vec::new();
    let mut species_projectors: Vec<Option<Vec<(usize, Vec<Arc<Radial>>)>>> =
        (0..datasets.len()).map(|_| None).collect();
    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let center = centers[atom_idx];
        if species_projectors[species_idx].is_none() {
            let mut cached = Vec::new();
            for (&l, channel) in &ds.channels {
                if channel.raw_projectors.len() != channel.projectors.len() {
                    return Err(invalid("raw projector count does not match the channel"));
                }
                let radials = channel.raw_projectors.iter()
                    .map(|t| Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).map(Arc::new))
                    .collect::<Result<Vec<_>>>()?;
                cached.push((l as usize, radials));
            }
            species_projectors[species_idx] = Some(cached);
        }
        for (l_usize, radials) in species_projectors[species_idx].as_ref().unwrap() {
            let channel = &ds.channels[&(*l_usize as u32)];

            for m in -(*l_usize as i32)..=*l_usize as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    proj_tuples.push((atom_idx, *l_usize, m, index));
                    projectors.push(Projector {
                        atom: atom_idx,
                        l: *l_usize,
                        m,
                        index,
                        center,
                        r_cut: channel.r_cut.max(ds.r_cut_local),
                        radial: radial.clone(),
                    });
                }
            }
        }
    }

    // Atoms of a species share the immutable local-potential spline.
    let mut potentials = Vec::with_capacity(n_atoms);
    let mut species_potentials: Vec<Option<LocalPotential>> =
        (0..datasets.len()).map(|_| None).collect();
    for (_atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        if species_potentials[species_idx].is_none() {
            species_potentials[species_idx] = Some(LocalPotential::new(
                ctx, ds.r.as_slice(), ds.v_local.as_slice(), ds.valence_charge,
                ds.r_cut_local,
            )?);
        }
        potentials.push(species_potentials[species_idx].as_ref().unwrap().clone());
    }

    // Convert fractional k-points to Cartesian k-points: k = 2 pi (lattice^-1)^T f
    let inv_lattice = input.lattice.try_inverse().ok_or_else(|| {
        invalid("singular crystal lattice matrix")
    })?;
    let rec_lattice = 2.0 * std::f64::consts::PI * inv_lattice.transpose();

    let mut kpoints_cart = Vec::with_capacity(input.kpoints_frac.len());
    for f in &input.kpoints_frac {
        let f_vec = nalgebra::Vector3::new(f[0], f[1], f[2]);
        let k = rec_lattice * f_vec;
        kpoints_cart.push([k[0], k[1], k[2]]);
    }

    // Symmetry operations
    let magnetic = input.atoms.iter().any(|a| a.moment != 0.);
    let symmetry = if input.symmetry && !input.spinor && !magnetic {
        let labels: Vec<u32> = atomic_numbers
            .iter()
            .zip(&moments)
            .map(|(&z, &m)| z ^ ((m.abs() * 1e4) as u32))
            .collect();
        let geom = Geometry {
            lattice: input.lattice,
            fractional: &fractional_atoms,
            species: &atomic_numbers,
            labels: &labels,
            grid: shape,
            symprec: 1e-4,
            max_l: 4.max(atoms_species.iter().map(|&species| {
                2 * datasets[species].channels.keys().copied().max().unwrap_or(0) as usize
            }).max().unwrap_or(0)),
        };
        Some(Symmetry::new(&geom)?)
    } else {
        None
    };

    // Compensation shapes channels: (atom, l, m)
    let mut channels = Vec::new();
    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let max_l = ds.channels.keys().copied().max().unwrap_or(0) as usize;
        let l_max = 2 * max_l;
        for l in 0..=l_max {
            for m in -(l as i32)..=l as i32 {
                channels.push((atom_idx, l, m));
            }
        }
    }

    // Pins
    let alias_ops = crate::math::reciprocal::operation_count(ctx, &grid)?;
    let pins = Pins {
        sigma,
        filter_cutoff: k_c,
        grid_shape: shape,
        alias_operations: alias_ops,
        dense_cutoff: super::shapes::COMPENSATION_CUTOFF,
        compensation_cutoff: super::shapes::COMPENSATION_CUTOFF,
    };

    let precision = crate::math::storage::Precision::parse(&input.precision)?;

    let point_group = if let Some(ref sym) = symmetry {
        Some(sym.to_point_group(n_atoms)?)
    } else {
        None
    };

    let dataset_files: Vec<(String, String)> = match input.dataset_files {
        Some(ref files) if files.len() == atoms_species.len() => files.clone(),
        Some(ref files) if files.len() == datasets.len() => {
            atoms_species.iter().map(|&sp| files[sp].clone()).collect()
        }
        _ => atoms_species
            .iter()
            .map(|&sp| (datasets[sp].symbol.clone(), String::new()))
            .collect(),
    };

    let hubbard = if let Some(ref request) = input.hubbard_u {
        use crate::periodic::hubbard_table::{build_paw, PawLayout, Projector as HubbardProjector};
        let labels: Vec<_> = projectors.iter().map(|p| HubbardProjector {
            symbol: datasets[atoms_species[p.atom]].symbol.clone(), basis: "raw".into(),
            atom: p.atom, l: p.l, m: p.m as i64, index: p.index,
        }).collect();
        let sources: Vec<_> = (0..datasets.len()).map(|species| {
            atoms_species.iter().position(|&s| s == species)
                .map(|atom| dataset_files[atom].0.clone()).unwrap_or_default()
        }).collect();
        Some(build_paw(ctx, request, &PawLayout { projectors: &labels,
            species_of_atom: &atoms_species, species: &datasets, sources: &sources,
            moments: &moments, spin_orbit: has_spin_orbit })?)
    } else { None };

    let description = Some(crate::periodic::description::Description::from_parts(
        ctx,
        sigma,
        tables,
        function_atom,
        function_m,
        function_table,
        proj_tuples,
        point_group,
        centers.clone(),
        dataset_files,
        "raw".to_string(),
        channels.clone(),
    )?);

    Ok(PeriodicSetup {
        grid,
        lattice: input.lattice,
        shells,
        projectors,
        potentials,
        datasets,
        kpoints_cart,
        kpoints_frac: input.kpoints_frac,
        weights: input.weights,
        atoms_species,
        centers,
        channels,
        symmetry,
        functional: input.functional,
        precision,
        relativistic,
        spins: if input.spinor { 1 } else if input.spin || magnetic { 2 } else { 1 },
        moments,
        spinor: input.spinor,
        time_reversal: input.spinor && !input.spin && !magnetic,
        spin_orbit_projectors,
        spin_orbit_couplings,
        pins,
        constant: 0.0,
        sigma,
        description,
        hubbard,
    })
}

fn empty_channel_level(ctx: &Context, ds: &PawDataset, l: u32) -> Result<f64> {
    let channel = &ds.channels[&l];
    let wave = channel.pseudo_waves.first()
        .ok_or_else(|| invalid("empty channel has no reference pseudo wave"))?;
    let projectors: Vec<f64> = channel.projectors.iter()
        .flat_map(|p| p.as_slice().iter().copied()).collect();
    let data = ChannelData {
        table: ds.r.as_slice(),
        local: ds.v_local_screened.as_slice(),
        projectors: &projectors,
        rank: channel.projectors.len(),
        coupling: channel.coupling_screened.data(),
        overlap: channel.overlap_correction.data(),
        wave: wave.as_slice(),
        l: l as usize,
        r_cut: channel.r_cut,
    };
    let (lo, hi) = crate::paw::diagnose::spectrum::EXTENT_BOUNDS;
    let extent = crate::paw::checks::spectrum_extent(
        data.table, data.wave, crate::paw::diagnose::spectrum::EXTENT_FLOOR, lo, hi,
    )?;
    crate::paw::pao::bound_level(ctx, data, extent)
}

fn confined_orbital(ctx: &Context, ds: &PawDataset, l: u32, state: usize, shift: f64) -> Result<crate::paw::confinement::Orbital> {
    let channel = &ds.channels[&l];
    let wave = channel.pseudo_waves.get(state)
        .ok_or_else(|| invalid("occupied state has no corresponding pseudo wave"))?;
    let rank = channel.projectors.len();
    let projectors: Vec<f64> = channel
        .projectors
        .iter()
        .flat_map(|p| p.as_slice().iter().copied())
        .collect();
    let data = ChannelData {
        table: ds.r.as_slice(),
        local: ds.v_local_screened.as_slice(),
        projectors: &projectors,
        rank,
        coupling: channel.coupling_screened.data(),
        overlap: channel.overlap_correction.data(),
        wave: wave.as_slice(),
        l: l as usize,
        r_cut: channel.r_cut,
    };
    let prepared = Prepared::new(ctx, data)?;
    let wall = crate::basis::confinement::Potential {
        amplitude: 12.,
        inner_fraction: 0.6,
    };
    let orb = orbital(ctx, &prepared, shift, &wall, state, None)?
        .map_err(|reach| invalid(format!("orbital does not fit table at reach {reach}")))?;
    Ok(orb)
}

/// Parse a JSON `"periodic-paw-input"` specification into [`SetupInput`].
pub fn parse_input_json(v: &Value) -> Result<SetupInput> {
    for name in ["hubbard", "ghosts", "diffuse", "energy_shift",
                 "confinement_potential", "polarization_method", "zeta_split", "size", "basis"] {
        if v.get(name).is_some() {
            return Err(crate::Error::Unsupported(format!(
                "native periodic setup does not yet support option {name:?}"
            )));
        }
    }
    if let Some(charge) = v.get("charge") {
        if charge.as_f64() != Some(0.) {
            return Err(crate::Error::Unsupported(
                "native periodic setup does not yet support charged crystals".into(),
            ));
        }
    }
    fn triple(value: &Value, name: &str) -> Result<[f64; 3]> {
        let array = value.as_array().ok_or_else(|| invalid(format!("{name} must contain three numbers")))?;
        if array.len() != 3 {
            return Err(invalid(format!("{name} must contain three numbers")));
        }
        let mut out = [0.; 3];
        for (to, value) in out.iter_mut().zip(array) {
            *to = value.as_f64().filter(|v| v.is_finite())
                .ok_or_else(|| invalid(format!("{name} must contain three finite numbers")))?;
        }
        Ok(out)
    }
    let lattice_vals = v["lattice"]
        .as_array()
        .ok_or_else(|| invalid("missing lattice in input"))?;
    if lattice_vals.len() != 9 {
        return Err(invalid("lattice must have 9 elements"));
    }
    let mut lattice = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            lattice[(i, j)] = lattice_vals[i * 3 + j].as_f64()
                .filter(|v| v.is_finite()).ok_or_else(|| invalid("lattice must contain finite numbers"))?;
        }
    }

    let shape_arr = v["grid_shape"]
        .as_array()
        .ok_or_else(|| invalid("missing grid_shape"))?;
    if shape_arr.len() != 3 {
        return Err(invalid("grid_shape must contain three positive integers"));
    }
    let mut grid_shape = [0usize; 3];
    for (to, value) in grid_shape.iter_mut().zip(shape_arr) {
        *to = value.as_u64().and_then(|v| usize::try_from(v).ok()).filter(|v| *v > 0)
            .ok_or_else(|| invalid("grid_shape must contain three positive integers"))?;
    }

    let mut atoms = Vec::new();
    if let Some(arr) = v["atoms"].as_array() {
        for item in arr {
            let species = match item.get("species") {
                None => 0,
                Some(value) => value.as_u64().and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| invalid("atom species must be a nonnegative dataset index"))?,
            };
            let symbol = match item.get("symbol") {
                None => String::new(),
                Some(value) => value.as_str().ok_or_else(|| invalid("atom symbol must be a string"))?.to_string(),
            };
            let fractional = triple(&item["fractional"], "atom fractional")?;
            let moment = match item.get("moment") {
                None => 0.,
                Some(value) => value.as_f64().filter(|v| v.is_finite())
                    .ok_or_else(|| invalid("atom moment must be finite"))?,
            };
            atoms.push(AtomSpec {
                species,
                symbol,
                fractional,
                moment,
            });
        }
    }

    let mut kpoints_frac = Vec::new();
    let mut weights = Vec::new();
    if let Some(value) = v.get("kpoints") {
        let k_arr = value.as_array().filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("kpoints must be a nonempty array"))?;
        for item in k_arr {
            let frac = triple(&item["fractional"], "kpoint fractional")?;
            let w = match item.get("weight") {
                None => 1.,
                Some(value) => value.as_f64().filter(|v| v.is_finite())
                    .ok_or_else(|| invalid("k-point weight must be finite"))?,
            };
            kpoints_frac.push(frac);
            weights.push(w);
        }
    } else {
        kpoints_frac.push([0., 0., 0.]);
        weights.push(1.0);
    }

    let string_option = |name: &str, default: &str| -> Result<String> {
        match v.get(name) {
            None => Ok(default.to_string()),
            Some(value) => value.as_str().map(str::to_string)
                .ok_or_else(|| invalid(format!("{name} must be a string"))),
        }
    };
    let bool_option = |name: &str, default: bool| -> Result<bool> {
        match v.get(name) {
            None => Ok(default),
            Some(value) => value.as_bool().ok_or_else(|| invalid(format!("{name} must be a boolean"))),
        }
    };
    let number_option = |name: &str| -> Result<Option<f64>> {
        match v.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_f64().filter(|x| x.is_finite()).map(Some)
                .ok_or_else(|| invalid(format!("{name} must be a finite number"))),
        }
    };
    let functional = string_option("functional", "lda")?;
    let precision = string_option("precision", "float64")?;
    let basis_size = string_option("basis_size", "SZ")?;
    let spinor = bool_option("spinor", false)?;
    let spin = bool_option("spin", false)?;
    let symmetry = bool_option("symmetry", true)?;
    let filter_cutoff = number_option("filter_cutoff")?;
    let sigma = number_option("sigma")?;
    let dataset_files = if let Some(arr) = v["dataset_files"].as_array().or_else(|| v["sites"]["datasets"].as_array()) {
        let mut files = Vec::new();
        for item in arr {
            let file = item["file"].as_str().unwrap_or("").to_string();
            let sha = item["sha256"].as_str().unwrap_or("").to_string();
            files.push((file, sha));
        }
        Some(files)
    } else {
        None
    };

    Ok(SetupInput {
        lattice,
        grid_shape,
        atoms,
        kpoints_frac,
        weights,
        functional,
        precision,
        basis_size,
        spinor,
        spin,
        symmetry,
        filter_cutoff,
        sigma,
        dataset_files,
        hubbard_u: v.get("hubbard_u").filter(|v| !v.is_null())
            .map(|v| crate::models::hubbard_request::Request::parse(&v.to_string()))
            .transpose()?,
    })
}
