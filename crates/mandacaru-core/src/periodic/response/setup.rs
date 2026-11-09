//! The geometry of a crystal as the engines read it, rebuilt from the snapshot's `setup` block
//! and the dataset files it names (see the parent module).
use super::super::{
    gradient::{scf, Layout},
    invalid,
    setup::{
        electrostatics,
        fields::core_table,
        ions::LocalPotential,
        provider::{Core, Inputs, Operators},
        pulay::Matrix,
        shapes::{Atom, Channel, Shapes, Sites, COMPENSATION_CUTOFF},
        shells::{Radial, Shell, Shells},
        spheres::Projector,
    },
    Crystal,
};
use crate::{
    paw::{digest::sha256_file, record::PawDataset},
    quantum::C64,
    Error, Result,
};
use nalgebra::Matrix3;
use std::{collections::BTreeMap, path::Path, sync::Arc};

/// The geometry of a crystal as the engines read it, owned. The borrowed views (`cores`, `inputs`)
/// are made on demand.
pub(super) struct Setup {
    records: Vec<PawDataset>,
    /// The record of every atom.
    kind: Vec<usize>,
    atoms: Vec<Atom>,
    channels: Vec<Channel>,
    lattice: Matrix3<f64>,
    sigma: f64,
    shells: Shells,
    owner: Vec<usize>,
    projectors: Vec<Projector>,
    potentials: Vec<LocalPotential>,
    shapes: Shapes,
    kpoints: Vec<[f64; 3]>,
    augmentation: Matrix,
    ion_operator: Matrix,
    blocks: Vec<Matrix>,
}

impl Setup {
    pub(super) fn new(c: &Crystal) -> Result<Self> {
        let d = super::description(c)?;
        let ctx = &c.context;
        if d.projector_basis != "raw" {
            return Err(Error::Unsupported(format!(
                "crystal forces and stress differentiate the raw projector basis, the snapshot \
                 was built with {:?}",
                d.projector_basis
            )));
        }
        // One record per distinct dataset file; each atom names its file and the digest it was read with.
        let mut files: Vec<(&str, &str)> = Vec::new();
        let mut kind = Vec::new();
        for (file, digest) in &d.datasets {
            let key = (file.as_str(), digest.as_str());
            kind.push(match files.iter().position(|f| *f == key) {
                Some(at) => at,
                None => {
                    files.push(key);
                    files.len() - 1
                }
            });
        }
        let mut records = Vec::new();
        for (file, digest) in files {
            if file.is_empty() || digest.is_empty() {
                return Err(Error::Unsupported(
                    "a dataset of this crystal was not read from a file (the sites block has no \
                     file name or digest), which forces and stress read again"
                        .into(),
                ));
            }
            let path = Path::new(file);
            if sha256_file(path)? != digest {
                return Err(invalid(format!(
                    "{file} is not the dataset file the snapshot was built from (SHA-256 differs)"
                )));
            }
            records.push(PawDataset::read(ctx, path)?);
        }
        let atoms: Vec<Atom> = d
            .centers
            .iter()
            .zip(&kind)
            .map(|(center, &k)| Atom {
                center: *center,
                charge: records[k].valence_charge,
                compensation_radius: records[k].compensation_radius,
            })
            .collect();
        let channels: Vec<Channel> = d.channels.clone();
        if channels.len() != c.channels {
            return Err(invalid(
                "the snapshot's channel labels disagree with its channel count",
            ));
        }
        let shape = c.grid.index.shape();
        let lattice = Matrix3::from_fn(|i, j| shape[j] as f64 * c.grid.step[(i, j)]);
        // The basis: one shell per run of functions of one atom and one table.
        let tables: Vec<Arc<Radial>> = d
            .tables
            .iter()
            .map(|t| Radial::new(ctx, &t.r, &t.values, t.confinement).map(Arc::new))
            .collect::<Result<_>>()?;
        let mut shells: Vec<Shell> = Vec::new();
        let mut shell_key: Vec<(usize, usize)> = Vec::new();
        for (row, ((&atom, &m), &table)) in d
            .function_atom
            .iter()
            .zip(&d.function_m)
            .zip(&d.function_table)
            .enumerate()
        {
            if shell_key.last() == Some(&(atom, table)) {
                let s = shells.last_mut().expect("a shell per key");
                s.ms.push(m);
                s.rows.push(row);
            } else {
                shell_key.push((atom, table));
                shells.push(Shell {
                    center: d.centers[atom],
                    l: d.tables[table].l,
                    ms: vec![m],
                    rows: vec![row],
                    radial: tables[table].clone(),
                });
            }
        }
        let shells = Shells::new(shells)?;
        let owner = d.function_atom.clone();
        // The projectors of every atom in the order of the Python crystal; the block must say the same.
        let mut projectors = Vec::new();
        for (atom, &k) in kind.iter().enumerate() {
            projectors.extend(projector_list(c, &records[k], atom, d.centers[atom])?);
        }
        let listed: Vec<(usize, usize, i32, usize)> = projectors
            .iter()
            .map(|q| (q.atom, q.l, q.m, q.index))
            .collect();
        if listed != d.projectors {
            return Err(invalid(
                "the projectors the datasets give are not the projector list of the setup block",
            ));
        }
        if projectors.len() != c.p {
            return Err(invalid(format!(
                "the datasets give {} projectors, the crystal has {}",
                projectors.len(),
                c.p
            )));
        }
        let potentials = kind
            .iter()
            .map(|&k| {
                let ds = &records[k];
                LocalPotential::new(
                    ctx,
                    ds.r.as_slice(),
                    ds.v_local.as_slice(),
                    ds.valence_charge,
                    ds.r_cut_local,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let q_max = electrostatics::max_wavenumber(ctx, &c.grid)?;
        let shapes = {
            let sites = Sites {
                atoms: &atoms,
                channels: &channels,
                lattice,
                sigma: d.sigma,
            };
            Shapes::new(ctx, &sites, q_max)?
        };
        let p = c.p;
        let blocks = (0..c.channels)
            .map(|a| Matrix::from_fn(p, p, |i, j| c.blocks.as_slice()[(a * p + i) * p + j]))
            .collect();
        let dataset_of = |atom: usize| &records[kind[atom]];
        let augmentation =
            block_diagonal(&projectors, &dataset_of, |ch| ch.overlap_correction.data())?;
        let ion_operator = block_diagonal(&projectors, &dataset_of, |ch| ch.coupling.data())?;
        Ok(Self {
            records,
            kind,
            atoms,
            channels,
            lattice,
            sigma: d.sigma,
            shells,
            owner,
            projectors,
            potentials,
            shapes,
            kpoints: c.points.iter().map(|p| p.k).collect(),
            augmentation,
            ion_operator,
            blocks,
        })
    }

    /// The smooth core density of every atom's dataset, where its unscreening used one.
    pub(super) fn cores(&self) -> Result<Vec<Option<Core<'_>>>> {
        self.kind
            .iter()
            .map(|&k| {
                let ds = &self.records[k];
                Ok(core_table(
                    ds.nlcc.applied,
                    ds.nlcc.source.as_deref().unwrap_or(""),
                    ds.smooth_core_density.as_slice(),
                )?
                .map(|values| Core {
                    radius: ds.r.as_slice(),
                    values,
                }))
            })
            .collect()
    }

    pub(super) fn inputs<'a>(
        &'a self,
        c: &'a Crystal,
        cores: &'a [Option<Core<'a>>],
    ) -> Inputs<'a> {
        Inputs {
            grid: &c.grid,
            shells: &self.shells,
            projectors: &self.projectors,
            kpoints: &self.kpoints,
            sites: Sites {
                atoms: &self.atoms,
                channels: &self.channels,
                lattice: self.lattice,
                sigma: self.sigma,
            },
            shapes: &self.shapes,
            potentials: &self.potentials,
            cores,
            dense_cutoff: COMPENSATION_CUTOFF,
        }
    }

    pub(super) fn layout(&self, c: &Crystal) -> Layout {
        Layout {
            atoms: self.atoms.len(),
            m: c.m,
            p: c.p,
            spins: c.spins,
            degeneracy: c.degeneracy(),
            weights: c.points.iter().map(|p| p.weight).collect(),
            row_owner: self.owner.clone(),
            projector_owner: self.projectors.iter().map(|p| p.atom).collect(),
            reference: c.reference,
            request: scf::request(c),
        }
    }

    pub(super) fn operators(&self) -> Operators {
        Operators {
            blocks: self.blocks.clone(),
            augmentation: self.augmentation.clone(),
            ion_operator: self.ion_operator.clone(),
        }
    }

    /// The spin-orbit union projectors of every atom (the dataset's dual tables) and the radial
    /// couplings per `(atom, l)`; empty for a scalar crystal.
    pub(super) fn spin_orbit(
        &self,
        c: &Crystal,
    ) -> Result<(Vec<Projector>, BTreeMap<(usize, usize), crate::models::spin_orbit::Coupling>)> {
        let (mut projectors, mut couplings) = (Vec::new(), BTreeMap::new());
        if !c.spinor {
            return Ok((projectors, couplings));
        }
        for (atom, &k) in self.kind.iter().enumerate() {
            let ds = &self.records[k];
            let center = self.atoms[atom].center;
            for (&l, entry) in &ds.spin_orbit {
                let channel = ds.channels.get(&l).ok_or_else(|| {
                    invalid("a spin-orbit channel of the dataset has no scalar channel")
                })?;
                let r_cut = channel.r_cut.max(ds.r_cut_local);
                let radials = entry
                    .projectors
                    .iter()
                    .map(|t| {
                        Radial::new(&c.context, ds.r.as_slice(), t.as_slice(), None).map(Arc::new)
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
}

/// The raw projectors of one atom: `l`, then `m`, then the radial index.
fn projector_list(
    c: &Crystal,
    ds: &PawDataset,
    atom: usize,
    center: [f64; 3],
) -> Result<Vec<Projector>> {
    let mut out = Vec::new();
    for (&l, channel) in &ds.channels {
        let radials = channel
            .raw_projectors
            .iter()
            .map(|t| Radial::new(&c.context, ds.r.as_slice(), t.as_slice(), None).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        for m in -(l as i32)..=l as i32 {
            for (index, radial) in radials.iter().enumerate() {
                out.push(Projector {
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
    Ok(out)
}

/// The block-diagonal `p x p` matrix of a dataset block (`coupling`, `overlap_correction`) in the
/// raw projector basis: `B^-1 X B^-T` on every `(atom, l, m)` group.
fn block_diagonal<'a>(
    projectors: &[Projector],
    dataset: &impl Fn(usize) -> &'a PawDataset,
    pick: impl Fn(&crate::paw::record::Channel) -> &[f64],
) -> Result<Matrix> {
    let p = projectors.len();
    let mut out = Matrix::zeros(p, p);
    let mut start = 0;
    while start < p {
        let head = &projectors[start];
        let group: Vec<usize> = (start..p)
            .take_while(|&i| {
                (projectors[i].atom, projectors[i].l, projectors[i].m)
                    == (head.atom, head.l, head.m)
            })
            .collect();
        let channel = &dataset(head.atom).channels[&(head.l as u32)];
        let n = group.len();
        let x = nalgebra::DMatrix::from_row_slice(n, n, pick(channel));
        let b = nalgebra::DMatrix::from_row_slice(n, n, channel.vanderbilt.data());
        let b_inverse = b
            .try_inverse()
            .ok_or_else(|| invalid("the projector transform of a dataset is singular"))?;
        let block = &b_inverse * x * b_inverse.transpose();
        for (i, &a) in group.iter().enumerate() {
            for (j, &k) in group.iter().enumerate() {
                out[(a, k)] = C64::new(block[(i, j)], 0.);
            }
        }
        start += n;
    }
    Ok(out)
}
