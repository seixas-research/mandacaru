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
    pub spinor: bool,
    pub symmetry: bool,
    pub filter_cutoff: Option<f64>,
    pub sigma: Option<f64>,
    pub dataset_files: Option<Vec<(String, String)>>,
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

    let sigma = input.sigma.unwrap_or(0.7 * spacing);

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

    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let center = centers[atom_idx];

        for (&l, channel) in &ds.channels {
            if channel.occupation <= 0. {
                continue;
            }
            let l_usize = l as usize;
            let conf_radial = confined_radial(ctx, ds, l)?;
            let (filtered, _) = filter::filter(
                ctx,
                ds.r.as_slice(),
                &conf_radial,
                l_usize,
                k_c,
                &filter::Options::default(),
            )?;

            let table_idx = tables.len();
            tables.push(crate::periodic::description::Table {
                l: l_usize,
                r: ds.r.as_slice().to_vec(),
                values: filtered.clone(),
                confinement: None,
            });

            let ms: Vec<i32> = (-(l as i32)..=l as i32).collect();
            let n_m = ms.len();
            let rows: Vec<usize> = (row_idx..row_idx + n_m).collect();
            row_idx += n_m;

            for &m in &ms {
                function_atom.push(atom_idx);
                function_m.push(m);
                function_table.push(table_idx);
            }

            let radial = Arc::new(Radial::new(ctx, ds.r.as_slice(), &filtered, None)?);
            shells_vec.push(Shell {
                center,
                l: l_usize,
                ms,
                rows,
                radial,
            });
        }
    }
    let shells = Shells::new(shells_vec)?;

    // Build Projectors list
    let mut projectors = Vec::new();
    let mut proj_tuples = Vec::new();
    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let center = centers[atom_idx];
        for (&l, channel) in &ds.channels {
            let l_usize = l as usize;
            let radials: Vec<_> = channel
                .projectors
                .iter()
                .map(|t| Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).map(Arc::new))
                .collect::<Result<Vec<_>>>()?;

            for m in -(l as i32)..=l as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    proj_tuples.push((atom_idx, l_usize, m, index));
                    projectors.push(Projector {
                        atom: atom_idx,
                        l: l_usize,
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

    // Build LocalPotentials list (one per atom)
    let mut potentials = Vec::with_capacity(n_atoms);
    for (_atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let pot = LocalPotential::new(
            ctx,
            ds.r.as_slice(),
            ds.v_local.as_slice(),
            ds.valence_charge,
            ds.r_cut_local,
        )?;
        potentials.push(pot);
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
    let symmetry = if input.symmetry && !input.spinor {
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
            max_l: 4,
        };
        Symmetry::new(&geom).ok()
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
    let alias_ops = crate::math::reciprocal::operation_count(ctx, &grid).unwrap_or(1);
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
        sym.to_point_group(n_atoms).ok()
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

    let description = crate::periodic::description::Description::from_parts(
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
    ).ok();

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
        relativistic: false,
        spinor: input.spinor,
        time_reversal: true,
        pins,
        constant: 0.0,
        sigma,
        description,
    })
}

fn confined_radial(ctx: &Context, ds: &PawDataset, l: u32) -> Result<Vec<f64>> {
    let channel = &ds.channels[&l];
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
        wave: channel.pseudo_waves[0].as_slice(),
        l: l as usize,
        r_cut: channel.r_cut,
    };
    let prepared = Prepared::new(ctx, data)?;
    let wall = crate::basis::confinement::Potential {
        amplitude: 12.,
        inner_fraction: 0.6,
    };
    let orb = orbital(ctx, &prepared, 0.1, &wall, 0, None)?
        .map_err(|reach| invalid(format!("orbital does not fit table at reach {reach}")))?;
    Ok(orb.radial)
}

/// Parse a JSON `"periodic-paw-input"` specification into [`SetupInput`].
pub fn parse_input_json(v: &Value) -> Result<SetupInput> {
    let lattice_vals = v["lattice"]
        .as_array()
        .ok_or_else(|| invalid("missing lattice in input"))?;
    if lattice_vals.len() != 9 {
        return Err(invalid("lattice must have 9 elements"));
    }
    let mut lattice = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            lattice[(i, j)] = lattice_vals[i * 3 + j].as_f64().unwrap_or(0.);
        }
    }

    let shape_arr = v["grid_shape"]
        .as_array()
        .ok_or_else(|| invalid("missing grid_shape"))?;
    let grid_shape = [
        shape_arr[0].as_u64().unwrap_or(10) as usize,
        shape_arr[1].as_u64().unwrap_or(10) as usize,
        shape_arr[2].as_u64().unwrap_or(10) as usize,
    ];

    let mut atoms = Vec::new();
    if let Some(arr) = v["atoms"].as_array() {
        for item in arr {
            let species = item["species"].as_u64().unwrap_or(0) as usize;
            let symbol = item["symbol"].as_str().unwrap_or("").to_string();
            let frac_arr = item["fractional"].as_array().ok_or_else(|| invalid("atom fractional"))?;
            let fractional = [
                frac_arr[0].as_f64().unwrap_or(0.),
                frac_arr[1].as_f64().unwrap_or(0.),
                frac_arr[2].as_f64().unwrap_or(0.),
            ];
            let moment = item["moment"].as_f64().unwrap_or(0.);
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
    if let Some(k_arr) = v["kpoints"].as_array() {
        for item in k_arr {
            let frac_arr = item["fractional"].as_array().ok_or_else(|| invalid("kpoint fractional"))?;
            let frac = [
                frac_arr[0].as_f64().unwrap_or(0.),
                frac_arr[1].as_f64().unwrap_or(0.),
                frac_arr[2].as_f64().unwrap_or(0.),
            ];
            let w = item["weight"].as_f64().unwrap_or(1.0);
            kpoints_frac.push(frac);
            weights.push(w);
        }
    }
    if kpoints_frac.is_empty() {
        kpoints_frac.push([0., 0., 0.]);
        weights.push(1.0);
    }

    let functional = v["functional"].as_str().unwrap_or("lda").to_string();
    let precision = v["precision"].as_str().unwrap_or("float64").to_string();
    let spinor = v["spinor"].as_bool().unwrap_or(false);
    let symmetry = v["symmetry"].as_bool().unwrap_or(true);
    let filter_cutoff = v["filter_cutoff"].as_f64();
    let sigma = v["sigma"].as_f64();
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
        spinor,
        symmetry,
        filter_cutoff,
        sigma,
        dataset_files,
    })
}
