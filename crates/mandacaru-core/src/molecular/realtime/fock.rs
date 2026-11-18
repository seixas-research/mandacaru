//! The Fock matrix of a molecular reference as a function of complex densities, for the
//! propagator of `math::propagator`.
//!
//! It is the builder the SCF and the linear response use, so the propagated matrix is the first
//! derivative of the same energy whose second derivative T1 assembles: Kohn-Sham through
//! `dft::fock_and_energy` (the grid terms read `Re D`, exact in the real basis; the PAW one-center
//! terms are linearized at the reference atom and live in the constant core matrix, so nothing in
//! the spheres is recomputed), Hartree-Fock as `h + J[D] - K[D]` with the SCF's pair factors and
//! the whole complex `D` in the exchange. The Poisson solver, the screening of a hybrid, the
//! orbital gradients of a meta-GGA and the grid scratch are built once. A meta-GGA sees the
//! gauge-invariant kinetic density `tau~ = tau - |j|^2 / (2 n)` of the complex orbitals
//! (`dft::evaluate_with`; TDDFT_PLAN.md 13.6.1, owner decision Q19a).
//!
//! A uniform vector potential `A` along `e` (the velocity gauge) adds `A e.P + (A^2 / 2) 1` to
//! every channel, `P = -i <a|grad|b>` the Hermitian momentum matrix (the T2 operator with its
//! one-center part), and the matching terms to the energy; a meta-GGA's `tau~` reads the kinetic
//! `tau` and current in it (`dft::fock_and_energy`), so with a partial core it keeps the core's
//! share as the length kick's does, and the energy's slope in `A` gains the exchange-correlation
//! current `dE_xc / dA` ([`MolecularMap::xc_current`]).
use crate::math::fft::Poisson;
use crate::math::propagator::{Evaluation, FockMap};
use crate::molecular::{dft, integrals::Eri, spinor_dft, MolecularIntegrals};
use crate::{quantum::C64, Buffer, Result};
use nalgebra::DMatrix;
use std::sync::Arc;

type Matrix = DMatrix<C64>;

enum Theory {
    KohnSham {
        name: String,
        core: Matrix,
        poisson: Poisson,
        screening: Option<dft::Screening>,
        scratch: (Buffer<C64>, Buffer<C64>),
        /// The orbital gradients of a meta-GGA (its kinetic density and current), built once.
        gradients: Option<crate::math::storage::RealSamples>,
    },
    SpinorKohnSham {
        name: String,
        spin_orbit: Option<Matrix>,
        shared: spinor_dft::Shared,
    },
    HartreeFock {
        core: Matrix,
        eri: Arc<Eri>,
    },
}

/// The Fock map of a reference (`functional` `None` is Hartree-Fock), with an optional constant
/// vector potential `strength` along the unit `direction` and its momentum matrix `momentum`
/// (already `e . P`).
pub(super) struct MolecularMap<'a> {
    integrals: &'a MolecularIntegrals,
    theory: Theory,
    field: Option<(Matrix, f64)>,
    /// The vector potential `A e` (Bohr^-1) a meta-GGA's kinetic fields carry.
    potential: Option<[f64; 3]>,
    /// Evaluations so far (the record of the run).
    pub(super) evaluations: usize,
    /// `dE_xc / dA` at fixed `D` per spin (`[spin][axis]`) of the last [`MolecularMap::bare`]
    /// build: a meta-GGA with a partial core; zero otherwise.
    pub(super) xc_current: [[f64; 3]; 2],
}

impl<'a> MolecularMap<'a> {
    pub(super) fn new_spinor(integrals: &'a MolecularIntegrals, name: &str) -> Result<Self> {
        let spin_orbit = spinor_dft::spin_orbit_term(integrals)?;
        let shared = spinor_dft::Shared::new(integrals, name)?;
        Ok(Self {
            integrals,
            theory: Theory::SpinorKohnSham {
                name: name.to_string(),
                spin_orbit,
                shared,
            },
            field: None,
            potential: None,
            evaluations: 0,
            xc_current: [[0.; 3]; 2],
        })
    }

    pub(super) fn new(integrals: &'a MolecularIntegrals, functional: Option<&str>) -> Result<Self> {
        let m = integrals.m();
        let theory = match functional {
            Some(name) => {
                let n = integrals.grid.index.points();
                let ctx = &integrals.context;
                Theory::KohnSham {
                    name: name.to_string(),
                    core: integrals.kohn_sham_core(),
                    poisson: Poisson::isolated(ctx, &integrals.grid)?,
                    screening: integrals.screening(name)?,
                    scratch: (ctx.allocate::<C64>(n)?, ctx.allocate::<C64>(n)?),
                    gradients: if crate::models::functional(name)?.requires_tau {
                        Some(dft::orbital_gradients(integrals)?)
                    } else {
                        None
                    },
                }
            }
            None => {
                // Real-time TDHF is a Hartree-Fock entry point: the shared guard owns the
                // frozen-core exchange refusal.
                integrals.require_hartree_fock_core()?;
                Theory::HartreeFock {
                    core: DMatrix::from_fn(m, m, |a, b| {
                        C64::new(integrals.one.as_slice()[a * m + b], 0.)
                    }),
                    eri: integrals.eri()?,
                }
            }
        };
        Ok(Self {
            integrals,
            theory,
            field: None,
            potential: None,
            evaluations: 0,
            xc_current: [[0.; 3]; 2],
        })
    }

    /// Switches on the constant vector potential `strength` along the unit `direction`, whose
    /// momentum matrix is `momentum` (`e . P`).
    pub(super) fn set_field(&mut self, momentum: Matrix, strength: f64, direction: [f64; 3]) {
        self.field = Some((momentum, strength));
        self.potential = Some(direction.map(|e| strength * e));
    }
    /// The vector potential alone, without its one-body terms (the oracle of [`super::fock_matrices`]).
    pub(super) fn set_potential(&mut self, potential: Option<[f64; 3]>) {
        self.potential = potential;
    }

    /// The matrices and the energy without the field's one-body terms, at `densities` (one
    /// per-spin matrix of a restricted state, or both spin matrices); a meta-GGA's kinetic fields
    /// carry the field.
    pub(super) fn bare(&mut self, densities: &[Matrix]) -> Result<(Vec<Matrix>, f64)> {
        self.evaluations += 1;
        let integrals = self.integrals;
        let m = integrals.m();
        let field = self.potential;
        match &mut self.theory {
            Theory::KohnSham {
                name,
                core,
                poisson,
                screening,
                scratch,
                gradients,
            } => {
                let (focks, energy, current) = dft::fock_and_energy(
                    integrals,
                    core,
                    densities,
                    name,
                    poisson,
                    gradients.as_ref(),
                    screening.as_ref(),
                    (scratch.0.as_mut_slice(), scratch.1.as_mut_slice()),
                    field,
                )?;
                self.xc_current = current;
                Ok((focks, energy))
            }
            Theory::SpinorKohnSham {
                name,
                spin_orbit,
                shared,
            } => {
                let eval =
                    spinor_dft::spinor_parts(integrals, &densities[0], name, spin_orbit.as_ref(), shared)?.0;
                self.xc_current = [[0.; 3]; 2];
                Ok((vec![eval.fock], eval.energy))
            }
            Theory::HartreeFock { core, eri } => {
                let parts: Vec<_> = densities
                    .iter()
                    .map(|d| crate::molecular::scf::jk(eri, m, d))
                    .collect();
                let focks: Vec<Matrix> = if densities.len() == 1 {
                    let (j, k) = &parts[0];
                    vec![&*core + j * C64::new(2., 0.) - k]
                } else {
                    let total = &parts[0].0 + &parts[1].0;
                    parts.iter().map(|(_, k)| &*core + &total - k).collect()
                };
                // E = (g / 2) sum_s tr[D_s (h + F_s)], g = 2 for one restricted channel.
                let g = if densities.len() == 1 { 2. } else { 1. };
                let energy = 0.5
                    * g
                    * densities
                        .iter()
                        .zip(&focks)
                        .map(|(d, f)| (d * (&*core + f)).trace().re)
                        .sum::<f64>();
                Ok((focks, energy))
            }
        }
    }
}

impl FockMap for MolecularMap<'_> {
    fn evaluate(&mut self, _time: f64, densities: &[Matrix]) -> Result<Evaluation> {
        let spinor = matches!(self.theory, Theory::SpinorKohnSham { .. });
        let (mut matrices, mut energy) = self.bare(densities)?;
        if let Some((momentum, strength)) = &self.field {
            let g = if spinor || densities.len() > 1 { 1. } else { 2. };
            let m = momentum.nrows();
            let shift = Matrix::identity(m, m) * C64::new(0.5 * strength * strength, 0.);
            for (f, d) in matrices.iter_mut().zip(densities) {
                *f += momentum * C64::new(*strength, 0.) + &shift;
                energy += g
                    * (strength * (d * momentum).trace().re
                        + 0.5 * strength * strength * d.trace().re);
            }
        }
        Ok(Evaluation { matrices, energy })
    }
}
