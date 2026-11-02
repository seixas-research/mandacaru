//! The PAW spheres of a hybrid crystal: where the atoms are, the datasets their projectors were
//! built from, the compensation shape tables and the one-center screened terms.
//!
//! A hybrid snapshot carries a `sites` block, the one place the engine learns the geometry of
//! the spheres (the moment blocks, projections and Hamiltonian pieces of the snapshot do not
//! hold it):
//!
//! ```text
//! "sites": { "centers": [[x, y, z], ...]          (Bohr, one per atom, in the grid's frame)
//!            "projector_basis": "raw" | "dual",   (the basis the projections were built in)
//!            "datasets": [{"symbol", "file", "sha256"}, ...] }   (one per atom)
//! ```
//!
//! The engine reads each dataset file through [`PawDataset`], refuses it if its SHA-256 is not
//! the recorded one (the projections were built from that exact file) and builds the species'
//! screened one-center terms once, here, so a hybrid that cannot be completed fails at
//! construction and never runs without its spheres. A hybrid crystal without the block raises.
use super::{
    invalid,
    screened_augmentation::Augmentation,
    screened_exchange::{
        exchange_operator_at_k_with, exchange_operators_with, Exchange, ExxDivTreatment, Screening,
    },
    screened_onecenter::{Species, Sphere, SphereModel},
    setup::{
        electrostatics::max_wavenumber,
        shapes::{Atom, Channel, Shapes, Sites},
    },
    Crystal, KPoint,
};
use crate::{
    math::{storage::MatrixSource, SpatialGrid},
    paw::{
        digest::sha256_file, onecenter::Basis, pair_tables::layout, record::PawDataset,
        screened_atom::screened, species_tables::SpeciesTables,
    },
    quantum::C64,
    Context, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use serde_json::Value;
use std::path::Path;

/// The spheres of a hybrid crystal, built once from its `sites` block.
pub struct Hybrid {
    atoms: Vec<Atom>,
    channels: Vec<Channel>,
    lattice: Matrix3<f64>,
    shapes: Shapes,
    model: SphereModel,
}

fn text<'a>(value: &'a Value, what: &str) -> Result<&'a str> {
    value
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(format!("the sites block needs {what}")))
}

impl Hybrid {
    /// Builds the spheres. `labels` are the snapshot's moment channels `(atom, L, M)` and
    /// `blocks` its moment matrices (`channels x p x p`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn parse(
        ctx: &Context,
        grid: &SpatialGrid,
        p: usize,
        labels: &[Channel],
        blocks: &[C64],
        sites: &Value,
        (omega, fraction): (f64, f64),
        spinor: bool,
        time_reversal: bool,
    ) -> Result<Self> {
        if sites.is_null() {
            return Err(invalid(
                "a hybrid crystal needs the PAW spheres its projections were built from: the \
                 snapshot has no 'sites' block (atom centers, projector basis, dataset files)",
            ));
        }
        if spinor && !time_reversal {
            return Err(invalid(
                "a magnetic spinor hybrid needs the spin-density functional of the spheres, \
                 which is not implemented; only a time-reversal-symmetric spinor crystal is",
            ));
        }
        let centers = sites["centers"]
            .as_array()
            .ok_or_else(|| invalid("the sites block needs the atom centers"))?;
        let datasets = sites["datasets"]
            .as_array()
            .filter(|d| d.len() == centers.len() && !d.is_empty())
            .ok_or_else(|| invalid("the sites block needs one dataset per atom"))?;
        let basis = Basis::parse(text(&sites["projector_basis"], "the projector basis")?)?;
        let mut files: Vec<(&str, &str)> = Vec::new();
        let mut kind = Vec::new();
        for entry in datasets {
            let key = (
                text(&entry["file"], "the dataset file of every atom (a dataset not read from a file cannot be handed over)")?,
                text(&entry["sha256"], "the SHA-256 of every dataset file")?,
            );
            kind.push(match files.iter().position(|f| *f == key) {
                Some(at) => at,
                None => {
                    files.push(key);
                    files.len() - 1
                }
            });
        }
        let mut species = Vec::new();
        let mut charges = Vec::new();
        let mut radii = Vec::new();
        for (file, digest) in files {
            let path = Path::new(file);
            if sha256_file(path)? != digest {
                return Err(invalid(format!(
                    "{file} is not the dataset file the snapshot was built from (SHA-256 differs)"
                )));
            }
            let record = PawDataset::read(ctx, path)?;
            let tables = SpeciesTables::from_record(ctx, &record, basis)?;
            let view = tables.view();
            let projectors = layout(&view);
            species.push(Species::new(
                screened(ctx, &view, &projectors, omega)?,
                &projectors,
            ));
            charges.push(record.valence_charge);
            radii.push(record.compensation_radius);
        }
        let mut atoms = Vec::new();
        let mut spheres = Vec::new();
        let mut offset = 0;
        for (center, &which) in centers.iter().zip(&kind) {
            let c = center
                .as_array()
                .filter(|c| c.len() == 3)
                .and_then(|c| c.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
                .ok_or_else(|| invalid("an atom center needs three numbers"))?;
            atoms.push(Atom {
                center: [c[0], c[1], c[2]],
                charge: charges[which],
                compensation_radius: radii[which],
            });
            spheres.push(Sphere {
                offset,
                species: which,
            });
            offset += species[which].projectors();
        }
        if offset != p {
            return Err(invalid(format!(
                "the datasets give {offset} projectors, the crystal has {p}"
            )));
        }
        for (a, &(atom, _, _)) in labels.iter().enumerate() {
            let sphere = spheres
                .get(atom)
                .ok_or_else(|| invalid("a moment channel names an atom the sites block lacks"))?;
            let own = sphere.offset..sphere.offset + species[sphere.species].projectors();
            let block = &blocks[a * p * p..(a + 1) * p * p];
            if (0..p * p).any(|ij| {
                !(own.contains(&(ij / p)) && own.contains(&(ij % p))) && block[ij].norm() > 1e-12
            }) {
                return Err(invalid(
                    "the datasets' projector layout is not the layout of the moment blocks",
                ));
            }
        }
        let shape = grid.index.shape();
        let lattice = grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
        let shapes = {
            let view = Sites {
                atoms: &atoms,
                channels: labels,
                lattice,
                sigma: 1.0,
            };
            Shapes::new(ctx, &view, max_wavenumber(ctx, grid)?)?
        };
        Ok(Self {
            atoms,
            channels: labels.to_vec(),
            lattice,
            shapes,
            model: SphereModel::new(species, spheres, p, fraction)?,
        })
    }

    /// The augmentation of the crystal's pair densities with the one-center terms attached.
    pub fn augmentation<'a>(&'a self, c: &Crystal) -> Result<Augmentation<'a>> {
        let sites = Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            // The ion width is read by the lattice sums of the setup, not by the shape transforms.
            sigma: 1.0,
        };
        Ok(Augmentation::new(c, sites, &self.shapes)?.with_one_center(&self.model))
    }
}

impl Crystal {
    fn spheres(&self) -> Result<&Hybrid> {
        self.hybrid.as_ref().ok_or_else(|| {
            invalid(format!(
                "exact exchange needs a hybrid functional, not '{}'",
                self.functional
            ))
        })
    }

    /// The screened exact exchange of every (spin channel, k-point) of the mesh with the
    /// compensation charges and the one-center terms of the spheres, and its energy.
    pub fn hybrid_exchange<M: MatrixSource>(
        &self,
        matrices: &[M],
        exxdiv: ExxDivTreatment,
    ) -> Result<Exchange> {
        let augmentation = self.spheres()?.augmentation(self)?;
        exchange_operators_with(
            self,
            matrices,
            Screening::of(self)?,
            exxdiv,
            Some(&augmentation),
        )
    }

    /// The same operator `K(k)` at an arbitrary (off-mesh) point, from the converged density.
    pub fn hybrid_exchange_at_k<M: MatrixSource>(
        &self,
        matrices: &[M],
        point: &KPoint,
        spin: usize,
        exxdiv: ExxDivTreatment,
    ) -> Result<DMatrix<C64>> {
        let augmentation = self.spheres()?.augmentation(self)?;
        exchange_operator_at_k_with(
            self,
            matrices,
            point,
            spin,
            Screening::of(self)?,
            exxdiv,
            Some(&augmentation),
        )
    }
}
