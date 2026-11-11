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
    math::SpatialGrid,
    paw::record::PawDataset,
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
    /// A Monkhorst-Pack mesh `(size, gamma)` the setup reduces itself (by the operations that keep
    /// it, then time reversal; whole for a magnetic crystal with `symmetry` off, a non-time-reversal spinor crystal and
    /// a hybrid, whose exact exchange sums over the listed k-points). When
    /// set, `kpoints_frac` and `weights` are its output and must be empty on input.
    pub mesh: Option<([usize; 3], bool)>,
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
    pub charge: Option<f64>,
    /// Each element's basis recipe (`paw::basis_recipe::Recipe::from_json`; a `"cutoff"` of
    /// `"auto"` is the grid's Nyquist wave-vector); `None`: the default recipe of `basis_size`.
    pub recipes: Option<std::collections::BTreeMap<String, Value>>,
    /// The projector basis, `"raw"` (the default) or `"dual"`: which projectors the projections
    /// are built from, and the basis of every operator block.
    pub projector_basis: String,
}

/// Construct a [`PeriodicSetup`] from structured inputs and loaded datasets.
pub fn create_setup(
    ctx: &Context,
    mut input: SetupInput,
    datasets: Vec<Arc<PawDataset>>,
) -> Result<PeriodicSetup> {
    if input.atoms.is_empty() {
        return Err(invalid("periodic setup needs at least one atom"));
    }
    if input.mesh.is_some() != input.kpoints_frac.is_empty() {
        return Err(invalid("periodic setup needs either a k mesh or explicit k-points"));
    }
    let hybrid = crate::models::functional(&input.functional)?.screened_exchange.is_some();
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
    if input.mesh.is_none() && (input.kpoints_frac.len() != input.weights.len()
        || input.kpoints_frac.iter().flatten().any(|x| !x.is_finite())
        || input.weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        || (input.weights.iter().sum::<f64>() - 1.).abs() > 1e-10)
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
    use crate::paw::basis_recipe::{self, Recipe};
    let recipe_of = |symbol: &str| -> Result<Recipe> {
        match &input.recipes {
            None => Ok(Recipe::new(zeta_count, polarization_count, Some(k_c))),
            Some(table) => Recipe::from_json(table.get(symbol).ok_or_else(|| invalid(format!(
                "the basis recipes name no recipe for {symbol}")))?, Some(k_c)),
        }
    };
    let mut species_shells: Vec<Option<Vec<(usize, usize, Arc<Radial>)>>> =
        (0..datasets.len()).map(|_| None).collect();

    for (atom_idx, &species_idx) in atoms_species.iter().enumerate() {
        let ds = &datasets[species_idx];
        let center = centers[atom_idx];

        if species_shells[species_idx].is_none() {
            let (shells, _report) = basis_recipe::species(ctx, ds, &recipe_of(&ds.symbol)?)?;
            let mut cached = Vec::with_capacity(shells.len());
            for shell in shells {
                let table_idx = tables.len();
                let radial = Arc::new(Radial::new(ctx, ds.r.as_slice(), &shell.values, None)?);
                tables.push(crate::periodic::description::Table {
                    l: shell.l, r: ds.r.as_slice().to_vec(), values: shell.values, confinement: None,
                });
                cached.push((shell.l, table_idx, radial));
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
                spin_orbit_couplings.insert((atom, l as usize), crate::models::spin_orbit::Coupling {
                    radial: entry.coupling.shape()[0], values: entry.coupling.data().to_vec(),
                });
            }
        }
    }
    let shells = Shells::new(shells_vec)?;

    let dual = match input.projector_basis.as_str() {
        "raw" => false,
        "dual" => true,
        other => return Err(invalid(format!("projector_basis must be raw or dual, got {other:?}"))),
    };

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
                let tables = if dual { &channel.projectors } else { &channel.raw_projectors };
                let radials = tables.iter()
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

    // Symmetry operations
    let magnetic = input.atoms.iter().any(|a| a.moment != 0.);
    // A collinear moment pattern keeps the operations that map every atom onto an atom of the SAME
    // element and the SAME signed moment (the density of each spin channel is then symmetrized
    // channel by channel). An operation that would send a moment to its opposite needs a spin flip
    // of the channels: it is not kept (the moment-preserving subgroup of the magnetic group).
    let symmetry = if input.symmetry && !input.spinor && !hybrid {
        let mut keys: Vec<(u32, i64)> = Vec::new();
        let labels: Vec<u32> = atomic_numbers
            .iter()
            .zip(&moments)
            .map(|(&z, &m)| {
                let key = (z, (m * 1e4).round() as i64);
                match keys.iter().position(|k| *k == key) {
                    Some(i) => i as u32,
                    None => {
                        keys.push(key);
                        (keys.len() - 1) as u32
                    }
                }
            })
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

    // The k mesh, reduced here when the request names one.
    let mut symmetry = symmetry;
    if let Some((size, gamma)) = input.mesh {
        let reduced = if let Some(sym) = symmetry.take() {
            let kept = sym.keeping_mesh(size, gamma)?;
            let reduced = kept.mesh(ctx, size, gamma)?;
            symmetry = Some(kept);
            reduced
        } else {
            let whole = magnetic || (input.spinor && input.spin) || hybrid;
            crate::periodic::spectrum::mesh(ctx, size, gamma, &[], !whole)?
        };
        input.kpoints_frac = reduced.points.clone();
        input.weights = reduced.weights.clone();
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
    // The same operations in the form a snapshot's `setup` block carries (`description::parse`):
    // integer rotations row by row, fractional translations and the atom map of each.
    let operations = symmetry.as_ref().filter(|sym| sym.len() > 0).map(|sym| {
        let mut map = vec![0; n_atoms];
        let mut out = crate::periodic::description::Operations {
            rotations: vec![], translations: vec![], atom_maps: vec![],
        };
        for k in 0..sym.len() {
            let w = sym.rotation(k);
            let mut r = [0i32; 9];
            for i in 0..3 {
                for j in 0..3 {
                    r[3 * i + j] = w[(i, j)];
                }
            }
            out.rotations.push(r);
            out.translations.push(sym.translation(k));
            sym.atom_map(k, &mut map);
            out.atom_maps.push(map.clone());
        }
        out
    });

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
            symbol: datasets[atoms_species[p.atom]].symbol.clone(), basis: input.projector_basis.clone(),
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
        input.projector_basis.clone(),
        channels.clone(),
        operations,
    )?);

    Ok(PeriodicSetup {
        dual,
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
        charge: input.charge,
    })
}


/// Parse a JSON `"periodic-paw-input"` specification into [`SetupInput`].
pub fn parse_input_json(v: &Value) -> Result<SetupInput> {
    if v.get("diffuse").is_some_and(|d| !d.is_null() && d != false) {
        return Err(invalid(
            "the 'diffuse' basis option is for molecules: in a crystal a diffuse \
             shell is nearly linearly dependent on the neighbors' valence \
             functions, and its Bloch sum runs over several times the images of \
             a confined orbital. Drop 'diffuse' for a periodic geometry",
        ));
    }
    for name in ["hubbard", "ghosts", "size", "basis"] {
        if v.get(name).is_some() {
            return Err(crate::Error::Unsupported(format!(
                "native periodic setup does not yet support option {name:?}"
            )));
        }
    }
    // A basis option belongs to an element's recipe (`recipes`), never to the request itself:
    // refused rather than ignored.
    for name in ["energy_shift", "confinement_potential", "polarization_method", "zeta_split"] {
        if v.get(name).is_some() {
            return Err(crate::Error::Unsupported(format!(
                "the basis option {name:?} is an entry of each element's recipe (\"recipes\"), \
                 not of the request"
            )));
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
    let mesh = match v.get("mesh") {
        None => None,
        Some(value) => {
            if v.get("kpoints").is_some() {
                return Err(invalid("give either 'mesh' or 'kpoints', not both"));
            }
            let size = value["size"].as_array().filter(|a| a.len() == 3)
                .and_then(|a| a.iter().map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n > 0)).collect::<Option<Vec<_>>>())
                .ok_or_else(|| invalid("mesh size must be three positive integers"))?;
            let gamma = value["gamma"].as_bool()
                .ok_or_else(|| invalid("mesh gamma must be a boolean"))?;
            Some(([size[0], size[1], size[2]], gamma))
        }
    };
    if mesh.is_some() {
    } else if let Some(value) = v.get("kpoints") {
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
    let charge = number_option("charge")?;
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

    let recipes = match v.get("recipes") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_object()
            .ok_or_else(|| invalid("recipes must map element symbols to basis recipes"))?
            .iter().map(|(symbol, recipe)| (symbol.clone(), recipe.clone())).collect()),
    };
    let projector_basis = string_option("projector_basis", "raw")?;
    Ok(SetupInput {
        mesh,
        recipes,
        projector_basis,
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
        charge,
    })
}
