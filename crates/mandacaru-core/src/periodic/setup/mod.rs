//! Setup of the periodic PAW problem: the compensation shapes, the lattice sums
//! of the compact charges and the Gaussian ions, and the density-independent
//! fields. Atomic units; the inputs are plain structs, not datasets.

pub mod basis_gradient;
pub mod blocks;
pub mod dense;
pub mod electrostatics;
pub mod fields;
pub mod gradients;
pub mod images;
pub mod ions;
pub mod matrices;
pub mod parse;
pub mod provider;
pub mod pulay;
pub mod remainder;
pub mod shapes;
pub mod shells;
pub mod spheres;
pub mod spin_orbit;
pub mod spinor_tau;
pub mod strain_grid;
pub mod strain_provider;
pub mod strain_pulay;
pub mod strain_samples;
pub mod strain_spheres;
pub mod strain_sums;
pub mod symmetry;

// Re-exports
pub use blocks::{ion_operator, multipole_blocks, overlap_operator};
pub use matrices::{kpoint_matrices, MatrixAssembly};
pub use parse::{create_setup, parse_input_json, AtomSpec, SetupInput};
pub use provider::{Core, Inputs, Operators, SetupDerivatives};
pub use symmetry::Symmetry;

use crate::{
    math::{
        storage::{ComplexSamples, Precision},
        SpatialGrid,
    },
    paw::record::PawDataset,
    periodic::{
        input::{Crystal, CrystalComponents, KPoint},
        invalid,
    },
    product, Context, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::sync::Arc;

/// Retained discretizations held constant under strain differencing and analytic stress.
#[derive(Clone, Debug)]
pub struct Pins {
    pub sigma: f64,
    pub filter_cutoff: f64,
    pub grid_shape: [usize; 3],
    pub alias_operations: usize,
    pub dense_cutoff: f64,
    pub compensation_cutoff: f64,
}

/// The complete recipe and geometry of a periodic PAW problem.
pub struct PeriodicSetup {
    pub grid: SpatialGrid,
    pub lattice: Matrix3<f64>,
    pub shells: shells::Shells,
    pub projectors: Vec<spheres::Projector>,
    pub spin_orbit_projectors: Vec<spheres::Projector>,
    pub spin_orbit_couplings: std::collections::BTreeMap<(usize, usize), spin_orbit::Coupling>,
    pub potentials: Vec<ions::LocalPotential>,
    pub datasets: Vec<Arc<PawDataset>>,
    pub kpoints_cart: Vec<[f64; 3]>,
    pub kpoints_frac: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub atoms_species: Vec<usize>,
    pub centers: Vec<[f64; 3]>,
    pub channels: Vec<shapes::Channel>,
    pub symmetry: Option<Symmetry>,
    pub functional: String,
    pub precision: Precision,
    pub relativistic: bool,
    pub spins: usize,
    pub moments: Vec<f64>,
    pub spinor: bool,
    pub time_reversal: bool,
    pub pins: Pins,
    pub constant: f64,
    pub sigma: f64,
    pub description: Option<crate::periodic::description::Description>,
    pub hubbard: Option<crate::periodic::hubbard::Spec>,
    pub charge: Option<f64>,
}

impl PeriodicSetup {
    /// Dataset for each atom, in atom order. Setup stores datasets by species,
    /// whereas the projector and multipole assemblers index them by atom.
    fn atom_datasets(&self) -> Result<Vec<Arc<PawDataset>>> {
        self.atoms_species.iter().map(|&species| {
            self.datasets.get(species).cloned()
                .ok_or_else(|| invalid("atom dataset index is out of range"))
        }).collect()
    }

    /// Build the nonlocal and multipole operators for the setup's atoms.
    pub fn operators(&self) -> Result<Operators> {
        let datasets = self.atom_datasets()?;
        blocks::raw_operators(&self.projectors, &datasets, &self.channels)
    }

    /// Build the full [`Crystal`] structure ready for the Kohn-Sham SCF solver.
    pub fn build(&self, ctx: &Context) -> Result<Crystal> {
        let m = self.shells.functions();
        let p = self.projectors.len();
        let n_ch = self.channels.len();

        // 1. Build state-independent operators
        let datasets_by_atom = self.atom_datasets()?;
        let ops = blocks::raw_operators(&self.projectors, &datasets_by_atom, &self.channels)?;

        // 2. Build k-point matrices
        let assembly = MatrixAssembly {
            grid: &self.grid,
            shells: &self.shells,
            projectors: &self.projectors,
            potentials: &self.potentials,
            centers: &self.centers,
            lattice: &self.lattice,
            kpoints_cart: &self.kpoints_cart,
            kpoints_frac: &self.kpoints_frac,
            weights: &self.weights,
            operators: &ops,
            spin_orbit_couplings: self.spinor.then_some(&self.spin_orbit_couplings),
            spin_orbit_projectors: self.spinor.then_some(self.spin_orbit_projectors.as_slice()),
            sigma: self.sigma,
            precision: self.precision,
        };
        let points = matrices::kpoint_matrices(ctx, &assembly)?;

        // 3. Compensation multipole blocks
        // Reuse the operators already assembled for the k-point matrices.
        let blocks = blocks::store_multipole_blocks(ctx, &ops.blocks, p)?;
        drop(ops);

        // 4. Sites and Shapes
        let atoms: Vec<shapes::Atom> = self
            .centers
            .iter()
            .zip(&self.atoms_species)
            .map(|(&center, &sp)| {
                let ds = &self.datasets[sp];
                shapes::Atom {
                    center,
                    charge: ds.valence_charge,
                    compensation_radius: ds.compensation_radius,
                }
            })
            .collect();

        let sites = shapes::Sites {
            atoms: &atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: self.sigma,
        };

        let q_max = electrostatics::max_wavenumber(ctx, &self.grid)?;
        let shapes = shapes::Shapes::new(ctx, &sites, q_max)?;

        // 5. Electrostatic fields
        let kernel = electrostatics::kernel(ctx, &self.grid)?;
        let ion = electrostatics::ion_charge(ctx, &self.grid, &sites)?;
        let raw_hats = electrostatics::hats(ctx, &self.grid, &shapes, &sites)?;
        let hats = ComplexSamples::Float64(raw_hats);
        let couplings = electrostatics::couplings(ctx, &shapes, &sites, self.pins.dense_cutoff)?;
        let u = couplings.u;
        let uion = couplings.u_ion;
        let eion = couplings.e_ion;

        // 6. Ions terms
        let ion_constant = ions::ion_constants(&sites)?;
        let linear_vec = ions::linear(ctx, &sites, &self.potentials)?;
        let mut linear = ctx.allocate::<Complex64>(linear_vec.len())?;
        linear.as_mut_slice().copy_from_slice(&linear_vec);

        // 7. Density-independent fields
        let species_tables: Vec<fields::Tables<'_>> = self
            .datasets
            .iter()
            .map(|ds| Ok(fields::Tables {
                radius: ds.r.as_slice(),
                valence_density: ds.valence_density.as_slice(),
                core: fields::core_table(
                    ds.nlcc.applied, ds.nlcc.source.as_deref().unwrap_or(""),
                    ds.smooth_core_density.as_slice(),
                )?,
            }))
            .collect::<Result<Vec<_>>>()?;

        let placed_atoms: Vec<fields::Placed> = self
            .centers
            .iter()
            .zip(&self.atoms_species)
            .map(|(&center, &species)| fields::Placed { species, center })
            .collect();

        let core = fields::core(ctx, &self.grid, &species_tables, &placed_atoms, false)?;
        let definition = crate::models::functional(&self.functional)?;
        let core_tau = if definition.requires_tau {
            Some(fields::core(ctx, &self.grid, &species_tables, &placed_atoms, true)?)
        } else {
            None
        };

        let initial = fields::initial_density(ctx, &self.grid, &species_tables, &placed_atoms)?;
        let charges: Vec<f64> = self
            .atoms_species
            .iter()
            .map(|&s| self.datasets[s].valence_charge)
            .collect();
        let scalar_m = fields::initial_magnetization(
            ctx,
            &self.grid,
            &species_tables,
            &placed_atoms,
            &charges,
            &self.moments,
        )?;
        let initial_m = if self.spinor {
            let n = scalar_m.as_slice().len();
            let mut vector = ctx.allocate::<f64>(product(&[3, n])?)?;
            vector.as_mut_slice()[2 * n..].copy_from_slice(scalar_m.as_slice());
            vector
        } else {
            scalar_m
        };

        let comp_charges: Vec<f64> = self
            .atoms_species
            .iter()
            .map(|&s| self.datasets[s].compensation_charge)
            .collect();
        let init_q_vec = fields::initial_charges(&sites, &comp_charges)?;
        let mut initial_q = ctx.allocate::<Complex64>(init_q_vec.len())?;
        initial_q.as_mut_slice().copy_from_slice(&init_q_vec);

        let kerker = fields::kerker(ctx, &self.grid)?;
        let phase = fields::phase(ctx, &self.grid)?;
        let reference = fields::reference(&sites);
        let xc_wave = crate::math::reciprocal::symmetric_wavevectors(ctx, &self.grid)?;

        // The core offset belongs to the dataset's own functional and
        // exchange convention, which may differ from the requested functional.
        let mut constants = vec![None; self.datasets.len()];
        let mut total_constant = 0.;
        for &species in &self.atoms_species {
            if constants[species].is_none() {
                let ds = &self.datasets[species];
                let table = &species_tables[species];
                let xc = if ds.xc.is_empty() { "lda" } else { &ds.xc };
                constants[species] = Some(fields::species_constant(
                    ctx, ds.one_center_energy, table.radius, table.valence_density,
                    table.core, xc, ds.relativistic_exchange,
                )?);
            }
            total_constant += constants[species].unwrap();
        }

        // 8. Symmetry
        let qsym = if let Some(ref sym) = self.symmetry {
            sym.qsym(ctx, &self.channels)?
        } else {
            let mut eye = ctx.allocate::<Complex64>(product(&[n_ch, n_ch])?)?;
            let slice = eye.as_mut_slice();
            slice.fill(Complex64::new(0., 0.));
            for i in 0..n_ch {
                slice[i * n_ch + i] = Complex64::new(1., 0.);
            }
            eye
        };

        let mut permutations = Vec::new();
        if let Some(ref sym) = self.symmetry {
            let n_pts = self.grid.index.points();
            for op in 0..sym.len() {
                let mut perm = vec![0; n_pts];
                sym.permutation(op, &mut perm);
                permutations.push(perm);
            }
        }

        let valence_sum: f64 = self
            .atoms_species
            .iter()
            .map(|&s| self.datasets[s].valence_charge)
            .sum();
        let electrons = valence_sum - self.charge.unwrap_or(0.0);
        if electrons <= 0.0 {
            return Err(invalid("charged crystal must have positive electron count"));
        }

        let components = CrystalComponents {
            grid: self.grid.clone(),
            m,
            p,
            channels: n_ch,
            points,
            blocks,
            core,
            core_tau,
            precision: self.precision,
            initial,
            initial_m,
            initial_q,
            kernel,
            xc_wave,
            phase,
            kerker,
            ion,
            hats,
            u,
            uion,
            linear,
            eion,
            ion_constant,
            constant: total_constant,
            reference,
            electrons,
            spins: self.spins,
            spinor: self.spinor,
            time_reversal: self.time_reversal,
            functional: self.functional.clone(),
            relativistic: self.relativistic,
            permutations,
            qsym,
            hubbard: self.hubbard.clone(),
            hybrid: None,
            description: if let Some(ref d) = self.description {
                Some(d.try_clone(ctx)?)
            } else {
                None
            },
            description_unavailable: Some(
                "native comparison setup has not passed force and stress parity validation".into(),
            ),
        };

        Crystal::from_components(ctx, components)
    }

    /// Compute k-point matrices at arbitrary fractional k-points (off-mesh band structures).
    pub fn matrices_at(
        &self,
        ctx: &Context,
        fractional_k: &[[f64; 3]],
        weights: &[f64],
    ) -> Result<Vec<KPoint>> {
        let inv_lattice = self.lattice.try_inverse().ok_or_else(|| {
            invalid("singular crystal lattice matrix")
        })?;
        let rec_lattice = 2.0 * std::f64::consts::PI * inv_lattice.transpose();

        let mut cart = Vec::with_capacity(fractional_k.len());
        for f in fractional_k {
            let f_vec = nalgebra::Vector3::new(f[0], f[1], f[2]);
            let k = rec_lattice * f_vec;
            cart.push([k[0], k[1], k[2]]);
        }

        let ops = self.operators()?;

        let assembly = MatrixAssembly {
            grid: &self.grid,
            shells: &self.shells,
            projectors: &self.projectors,
            potentials: &self.potentials,
            centers: &self.centers,
            lattice: &self.lattice,
            kpoints_cart: &cart,
            kpoints_frac: fractional_k,
            weights,
            operators: &ops,
            spin_orbit_couplings: self.spinor.then_some(&self.spin_orbit_couplings),
            spin_orbit_projectors: self.spinor.then_some(self.spin_orbit_projectors.as_slice()),
            sigma: self.sigma,
            precision: self.precision,
        };

        matrices::kpoint_matrices(ctx, &assembly)
    }

    /// Strain the crystal cell with pinned discretizations held constant ([`Pins`]).
    pub fn with_cell(&self, _ctx: &Context, new_lattice: &Matrix3<f64>) -> Result<Self> {
        let shape = self.pins.grid_shape;
        let mut step = [0.; 9];
        for row in 0..3 {
            for col in 0..3 {
                step[row * 3 + col] = new_lattice[(row, col)] / shape[col] as f64;
            }
        }
        let grid = SpatialGrid::new(shape, [0.; 3], step, true)?;

        // Recompute Cartesian centers from fractional positions
        let inv_old = self.lattice.try_inverse().ok_or_else(|| invalid("singular lattice"))?;
        let mut new_centers = Vec::with_capacity(self.centers.len());
        for c in &self.centers {
            let old_v = nalgebra::Vector3::new(c[0], c[1], c[2]);
            let frac = inv_old * old_v;
            let new_v = new_lattice * frac;
            new_centers.push([new_v[0], new_v[1], new_v[2]]);
        }

        // Recompute Cartesian k-points from fractional k-points
        let inv_new = new_lattice.try_inverse().ok_or_else(|| invalid("singular new lattice"))?;
        let rec_new = 2.0 * std::f64::consts::PI * inv_new.transpose();
        let mut new_kpoints_cart = Vec::with_capacity(self.kpoints_frac.len());
        for f in &self.kpoints_frac {
            let f_vec = nalgebra::Vector3::new(f[0], f[1], f[2]);
            let k = rec_new * f_vec;
            new_kpoints_cart.push([k[0], k[1], k[2]]);
        }

        // Update shells with new centers
        let mut new_shells_vec = Vec::new();
        for s in self.shells.shells() {
            let atom_idx = self
                .centers
                .iter()
                .position(|&c| (nalgebra::Vector3::from(c) - nalgebra::Vector3::from(s.center)).norm() < 1e-6)
                .unwrap_or(0);
            new_shells_vec.push(shells::Shell {
                center: new_centers[atom_idx],
                l: s.l,
                ms: s.ms.clone(),
                rows: s.rows.clone(),
                radial: s.radial.clone(),
            });
        }
        let shells = shells::Shells::new(new_shells_vec)?;

        // Update projectors with new centers
        let mut projectors = Vec::with_capacity(self.projectors.len());
        for p in &self.projectors {
            let mut p_new = p.clone();
            p_new.center = new_centers[p.atom];
            projectors.push(p_new);
        }
        let spin_orbit_projectors = self.spin_orbit_projectors.iter().map(|p| {
            let mut moved = p.clone();
            moved.center = new_centers[p.atom];
            moved
        }).collect();

        let potentials = self
            .potentials
            .iter()
            .map(|p| p.clone_with(_ctx))
            .collect::<Result<Vec<_>>>()?;

        let description = if let Some(ref d) = self.description {
            let mut d_new = d.try_clone(_ctx)?;
            d_new.centers = new_centers.clone();
            Some(d_new)
        } else {
            None
        };

        Ok(Self {
            grid,
            lattice: *new_lattice,
            shells,
            projectors,
            spin_orbit_projectors,
            spin_orbit_couplings: self.spin_orbit_couplings.clone(),
            potentials,
            datasets: self.datasets.clone(),
            kpoints_cart: new_kpoints_cart,
            kpoints_frac: self.kpoints_frac.clone(),
            weights: self.weights.clone(),
            atoms_species: self.atoms_species.clone(),
            centers: new_centers,
            channels: self.channels.clone(),
            symmetry: self.symmetry.clone(),
            functional: self.functional.clone(),
            precision: self.precision,
            relativistic: self.relativistic,
            spins: self.spins,
            moments: self.moments.clone(),
            spinor: self.spinor,
            time_reversal: self.time_reversal,
            pins: self.pins.clone(),
            constant: self.constant,
            sigma: self.pins.sigma,
            description,
            hubbard: self.hubbard.clone(),
            charge: self.charge,
        })
    }
}
