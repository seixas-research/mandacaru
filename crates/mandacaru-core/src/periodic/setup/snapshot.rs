//! Reconstruct a [`PeriodicSetup`] from a converged or loaded [`Crystal`] carrying a
//! `setup`/`sites` [`Description`](crate::periodic::description::Description).

use super::{
    ions::LocalPotential,
    shapes::COMPENSATION_CUTOFF,
    shells::{Radial, Shell, Shells},
    spheres::Projector,
    PeriodicSetup, Pins,
};
use crate::{
    paw::record::PawDataset,
    periodic::{description::Description, invalid, Crystal},
    Error, Result,
};
use nalgebra::Matrix3;
use std::{collections::BTreeMap, path::Path, sync::Arc};

/// Reconstruct unique datasets and per-atom dataset indices from `description.datasets`.
pub(in crate::periodic) fn load_snapshot_datasets(
    crystal: &Crystal,
    description: &Description,
) -> Result<(Vec<Arc<PawDataset>>, Vec<usize>)> {
    let ctx = &crystal.context;
    let mut files: Vec<(&str, &str)> = Vec::new();
    let mut kind = Vec::with_capacity(description.datasets.len());
    for (file, digest) in &description.datasets {
        let key = (file.as_str(), digest.as_str());
        kind.push(match files.iter().position(|f| *f == key) {
            Some(at) => at,
            None => {
                files.push(key);
                files.len() - 1
            }
        });
    }
    let mut records = Vec::with_capacity(files.len());
    for (file, digest) in files {
        if file.is_empty() || digest.is_empty() {
            return Err(Error::Unsupported(
                "a dataset of this crystal snapshot was not recorded with a file path and \
                 content SHA-256 digest in its 'sites' block, so the periodic setup cannot be \
                 rebuilt from the snapshot"
                    .into(),
            ));
        }
        let record = PawDataset::read(ctx, Path::new(file))?;
        if record.content_sha256(ctx)? != digest {
            return Err(invalid(format!(
                "{file} is not the dataset file the snapshot was built from (content SHA-256 differs)"
            )));
        }
        records.push(Arc::new(record));
    }
    Ok((records, kind))
}

/// Reconstruct atomic basis [`Shells`] from a crystal [`Description`].
pub(in crate::periodic) fn shells_from_description(
    crystal: &Crystal,
    description: &Description,
) -> Result<Shells> {
    let ctx = &crystal.context;
    let tables: Vec<Arc<Radial>> = description
        .tables
        .iter()
        .map(|t| Radial::new(ctx, &t.r, &t.values, t.confinement).map(Arc::new))
        .collect::<Result<_>>()?;
    let mut shells: Vec<Shell> = Vec::new();
    let mut keys: Vec<(usize, usize)> = Vec::new();
    for (row, ((&atom, &m), &table)) in description
        .function_atom
        .iter()
        .zip(&description.function_m)
        .zip(&description.function_table)
        .enumerate()
    {
        if keys.last() == Some(&(atom, table)) {
            let last = shells.last_mut().expect("a shell per key");
            last.ms.push(m);
            last.rows.push(row);
        } else {
            keys.push((atom, table));
            shells.push(Shell {
                center: description.centers[atom],
                l: description.tables[table].l,
                ms: vec![m],
                rows: vec![row],
                radial: tables[table].clone(),
            });
        }
    }
    Shells::new(shells)
}

/// Reconstruct spin-orbit union projectors and couplings for spinor snapshots.
fn spin_orbit_from_datasets(
    crystal: &Crystal,
    datasets: &[Arc<PawDataset>],
    kind: &[usize],
    centers: &[[f64; 3]],
) -> Result<(
    Vec<Projector>,
    BTreeMap<(usize, usize), crate::models::spin_orbit::Coupling>,
)> {
    let mut projectors = Vec::new();
    let mut couplings = BTreeMap::new();
    if !crystal.spinor {
        return Ok((projectors, couplings));
    }
    for (atom, &species) in kind.iter().enumerate() {
        let ds = &datasets[species];
        let center = centers[atom];
        for (&l, entry) in &ds.spin_orbit {
            let channel = ds.channels.get(&l).ok_or_else(|| {
                invalid("a spin-orbit channel of the dataset has no scalar channel")
            })?;
            let r_cut = channel.r_cut.max(ds.r_cut_local);
            let radials = entry
                .projectors
                .iter()
                .map(|t| {
                    Radial::new(&crystal.context, ds.r.as_slice(), t.as_slice(), None).map(Arc::new)
                })
                .collect::<Result<Vec<_>>>()?;
            for m in -(l as i32)..=l as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    projectors.push(Projector {
                        atom,
                        l: l as usize,
                        m,
                        index,
                        center,
                        r_cut,
                        radial: radial.clone(),
                    });
                }
            }
            couplings.insert(
                (atom, l as usize),
                crate::models::spin_orbit::Coupling {
                    radial: entry.coupling.shape()[0],
                    values: entry.coupling.data().to_vec(),
                },
            );
        }
    }
    if projectors.is_empty() {
        return Err(Error::Unsupported(
            "a spin-orbit crystal whose datasets carry no spin-orbit projectors".into(),
        ));
    }
    Ok((projectors, couplings))
}

impl PeriodicSetup {
    /// Rebuild a [`PeriodicSetup`] from a snapshot-loaded [`Crystal`] carrying a `setup`/`sites`
    /// [`Description`], enabling off-mesh k-point evaluations (`Wannier`, `cRPA` long-wavelength
    /// heads, band paths, Berry quantities) without needing Python setup objects.
    pub fn from_crystal(crystal: &Crystal) -> Result<Self> {
        if let Some(cause) = &crystal.description_unavailable {
            return Err(Error::Unsupported(format!(
                "rebuilding the periodic setup from this crystal snapshot is unavailable: {cause}"
            )));
        }
        let d = crystal.description.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "this crystal snapshot has no 'setup' block (the radial tables of its basis, its \
                 function and projector lists, and dataset paths), so a periodic setup cannot be \
                 rebuilt from the snapshot alone"
                    .into(),
            )
        })?;
        let dual = match d.projector_basis.as_str() {
            "raw" => false,
            "dual" => true,
            other => {
                return Err(Error::Unsupported(format!(
                    "unsupported projector basis {other:?} in crystal snapshot"
                )));
            }
        };
        let (datasets, atoms_species) = load_snapshot_datasets(crystal, d)?;
        let shells = shells_from_description(crystal, d)?;
        let ctx = &crystal.context;
        let mut projectors = Vec::with_capacity(crystal.p);
        for (atom, &species) in atoms_species.iter().enumerate() {
            let ds = &datasets[species];
            let center = d.centers[atom];
            for (&l, channel) in &ds.channels {
                let tables = if dual {
                    &channel.projectors
                } else {
                    &channel.raw_projectors
                };
                let radials = tables
                    .iter()
                    .map(|t| Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).map(Arc::new))
                    .collect::<Result<Vec<_>>>()?;
                for m in -(l as i32)..=l as i32 {
                    for (index, radial) in radials.iter().enumerate() {
                        projectors.push(Projector {
                            atom,
                            l: l as usize,
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
        let listed: Vec<(usize, usize, i32, usize)> = projectors
            .iter()
            .map(|q| (q.atom, q.l, q.m, q.index))
            .collect();
        if listed != d.projectors || projectors.len() != crystal.p {
            return Err(invalid(
                "the dataset projectors do not match the crystal snapshot's projector table",
            ));
        }
        let potentials = atoms_species
            .iter()
            .map(|&species| LocalPotential::from_dataset(ctx, &datasets[species]))
            .collect::<Result<Vec<_>>>()?;
        let (spin_orbit_projectors, spin_orbit_couplings) =
            spin_orbit_from_datasets(crystal, &datasets, &atoms_species, &d.centers)?;
        let shape = crystal.grid.index.shape();
        let lattice = Matrix3::from_fn(|i, j| shape[j] as f64 * crystal.grid.step[(i, j)]);
        let valence_sum: f64 = atoms_species
            .iter()
            .map(|&s| datasets[s].valence_charge)
            .sum();
        let diff = valence_sum - crystal.electrons;
        let charge = (diff.abs() > 1e-12).then_some(diff);
        let relativistic = crystal.relativistic;
        let alias_operations = crate::math::reciprocal::operation_count(ctx, &crystal.grid)?;
        Ok(Self {
            dual,
            grid: crystal.grid.clone(),
            lattice,
            shells,
            projectors,
            spin_orbit_projectors,
            spin_orbit_couplings,
            potentials,
            kpoints_cart: crystal.points.iter().map(|p| p.k).collect(),
            kpoints_frac: crystal.points.iter().map(|p| p.fractional).collect(),
            weights: crystal.points.iter().map(|p| p.weight).collect(),
            atoms_species,
            centers: d.centers.clone(),
            channels: d.channels.clone(),
            symmetry: None,
            functional: crystal.functional.clone(),
            precision: crystal.precision,
            relativistic,
            spins: crystal.spins,
            moments: vec![0.; d.centers.len()],
            spinor: crystal.spinor,
            time_reversal: crystal.time_reversal,
            pins: Pins {
                sigma: d.sigma,
                filter_cutoff: 0.0,
                grid_shape: shape,
                alias_operations,
                dense_cutoff: COMPENSATION_CUTOFF,
                compensation_cutoff: COMPENSATION_CUTOFF,
            },
            constant: crystal.constant,
            sigma: d.sigma,
            description: Some(d.try_clone(ctx)?),
            hubbard: crystal.hubbard.clone(),
            datasets,
            charge,
            recipe: None,
        })
    }
}
