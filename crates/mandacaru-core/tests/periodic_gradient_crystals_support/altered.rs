//! A provider with ONE thing wrong, for the negative controls: every method delegates to the
//! real [`SetupDerivatives`] but the named piece (the kinetic matrix, the sign of the core, the
//! reference of `W`, the ownership of the rows) and, for a hybrid crystal, the exact exchange left
//! out of the Hamiltonian the bands come from (all of it, or only in the eigenvalues `W` weighs).
#![allow(dead_code)]

use mandacaru_core::{
    periodic::{
        gradient::{scf, spin_orbit::SpinOrbit, Bands, Dense, Derivatives, Layout, State, Term},
        hybrid_gradient,
        setup::{
            provider::{Fields, SetupDerivatives},
            pulay::Matrix,
        },
        solver, spectrum, Crystal, Options,
    },
    quantum::C64,
    Result,
};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wrong {
    /// The overlap matrices in place of the kinetic ones.
    Kinetic,
    /// The opposite sign of the partial-core force.
    CoreSign,
    /// `W` built with the eigenvalues still shifted by the reference.
    ReferenceKept,
    /// The owners of the first and the last basis function exchanged.
    Ownership,
    /// The projector-moving columns charged to the owner of the rows (no recoil on the projector).
    ProjectorOwner,
    /// Hybrid: the bands of the semilocal Hamiltonian of the output density, `-a K` left out of
    /// the diagonalization (vectors, eigenvalues and occupations all change).
    NoExchangeInBands,
    /// Hybrid: the bands of the hybrid Hamiltonian, but with the eigenvalues of the semilocal one
    /// (band by band, ascending): `W` is built from the PBE-like spectrum.
    SemilocalEigenvalues,
    /// Hybrid: an exchange derivative without the basis-function terms (a provider with no `Motion`).
    NoFunctions,
    /// Hybrid: an exchange derivative with one response matrix too few.
    ShortResponse,
    /// Spin-polarized: the partial core differentiated against `v_up` alone, not `(v_up + v_dn) / 2`.
    CoreSpinUp,
    /// Spin-polarized: the spin-down channel contracted with the spin-up local potential.
    LocalSpinUp,
    /// Spin-polarized: both channels counted with the restricted degeneracy 2.
    DegeneracyTwo,
    /// Spin-polarized: the spin-down bands replaced by the spin-up ones.
    BandsSpinUp,
    /// Spin-polarized hybrid: the exchange derivative of the spin-down channel taken from the spin-UP
    /// density (the exchange of the wrong spin; swapping the two channels would change nothing, the
    /// explicit terms being spin sums and the projection chain symmetric in the spin label).
    ExchangeSpinUp,
    /// Spin-polarized hybrid: the spin densities summed BEFORE the exchange contraction (both channels
    /// contracted with `D_up + D_dn`, as if the exchange acted across spins).
    ExchangeSummedDensity,
}

/// The bands of both spins joined into one set per k-point: the density matrix `C f C^+` of the
/// result is `D_up + D_dn`.
fn summed(bands: &[Vec<Bands>]) -> Vec<Vec<Bands>> {
    let joined: Vec<Bands> = (0..bands[0].len())
        .map(|k| {
            let parts: Vec<&Bands> = bands.iter().map(|s| &s[k]).collect();
            let m = parts[0].coefficients.nrows();
            let columns: Vec<_> = parts
                .iter()
                .flat_map(|b| (0..b.coefficients.ncols()).map(move |j| (b, j)))
                .collect();
            Bands {
                coefficients: Matrix::from_fn(m, columns.len(), |i, j| {
                    let (b, c) = columns[j];
                    b.coefficients[(i, c)]
                }),
                eigenvalues: columns.iter().map(|(b, j)| b.eigenvalues[*j]).collect(),
                occupations: columns.iter().map(|(b, j)| b.occupations[*j]).collect(),
            }
        })
        .collect();
    bands.iter().map(|_| joined.clone()).collect()
}

pub struct Altered<'a> {
    pub inner: SetupDerivatives<'a>,
    layout: Layout,
    wrong: Wrong,
    /// The crystal and the SCF options a hybrid variant re-diagonalizes with.
    hybrid: Option<(&'a Crystal, Options)>,
    /// The semilocal bands of the output density, formed by `fields`.
    semilocal: RefCell<Vec<Vec<Bands>>>,
}

impl<'a> Altered<'a> {
    pub fn new(inner: SetupDerivatives<'a>, wrong: Wrong) -> Self {
        let mut layout = inner.layout().clone();
        match wrong {
            Wrong::ReferenceKept => layout.reference = 0.,
            Wrong::DegeneracyTwo => layout.degeneracy = 2.,
            Wrong::Ownership => {
                let last = layout.row_owner.len() - 1;
                layout.row_owner.swap(0, last);
            }
            Wrong::ProjectorOwner => {
                // Every projector owned by atom 0: the columns are charged to the wrong atom.
                layout.projector_owner.iter_mut().for_each(|o| *o = 0);
            }
            _ => {}
        }
        Self {
            inner,
            layout,
            wrong,
            hybrid: None,
            semilocal: RefCell::new(Vec::new()),
        }
    }

    /// A hybrid variant: the crystal and the options the semilocal bands are formed with.
    pub fn hybrid(
        inner: SetupDerivatives<'a>,
        wrong: Wrong,
        crystal: &'a Crystal,
        options: Options,
    ) -> Self {
        let mut altered = Self::new(inner, wrong);
        altered.hybrid = Some((crystal, options));
        altered
    }
}

/// The bands of `H_semilocal[D_out]` (no exchange), the Fermi level over the whole mesh, as
/// [`scf::rediagonalize`] forms the hybrid ones.
fn semilocal_bands(
    crystal: &Crystal,
    state: &State,
    q: &[C64],
    options: &Options,
) -> Vec<Vec<Bands>> {
    let densities = scf::densities(crystal, &state.bands).unwrap();
    let fields = scf::output_fields(crystal, &densities, q).unwrap();
    let n = crystal.grid.index.points();
    let mut eigenvalues = Vec::new();
    let mut coefficients = Vec::new();
    for s in 0..crystal.spins {
        for data in &crystal.points {
            let (eps, vectors, _) = solver::diagonalize_with_exchange(
                crystal,
                data,
                &fields.local.as_slice()[s * n..(s + 1) * n],
                fields.moments.as_slice(),
                None,
                None,
                None,
            )
            .unwrap();
            eigenvalues.push(eps);
            coefficients.push(vectors);
        }
    }
    let weights: Vec<f64> = (0..crystal.spins)
        .flat_map(|_| crystal.points.iter().map(|p| p.weight))
        .collect();
    let mu = spectrum::fermi(
        &eigenvalues,
        &weights,
        crystal.electrons,
        options.smearing,
        options.width,
        crystal.degeneracy(),
    )
    .unwrap();
    let mut flat = eigenvalues.into_iter().zip(coefficients);
    (0..crystal.spins)
        .map(|_| {
            (0..crystal.points.len())
                .map(|_| {
                    let (eps, coefficients) = flat.next().unwrap();
                    Bands {
                        occupations: eps
                            .iter()
                            .map(|e| options.smearing.occupation((e - mu) / options.width))
                            .collect(),
                        eigenvalues: eps.iter().map(|e| e + crystal.reference).collect(),
                        coefficients,
                    }
                })
                .collect()
        })
        .collect()
}

impl Derivatives for Altered<'_> {
    type Fields = Fields;

    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn blocks(&self) -> &[Matrix] {
        self.inner.blocks()
    }
    fn augmentation(&self) -> &Matrix {
        self.inner.augmentation()
    }
    fn ion_operator(&self) -> &Matrix {
        self.inner.ion_operator()
    }
    fn projection(&self, k: usize) -> Result<Matrix> {
        self.inner.projection(k)
    }
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        self.inner.projection_gradient(k)
    }
    fn cross(&self, spin: usize, k: usize, term: Term, fields: &Fields) -> Result<[Matrix; 3]> {
        let term = if self.wrong == Wrong::Kinetic && term == Term::Kinetic {
            Term::Overlap
        } else {
            term
        };
        let spin = if self.wrong == Wrong::LocalSpinUp && term == Term::Local {
            0
        } else {
            spin
        };
        self.inner.cross(spin, k, term, fields)
    }
    fn sphere_cross(&self, spin: usize, k: usize, fields: &Fields) -> Result<Vec<[Matrix; 3]>> {
        self.inner.sphere_cross(spin, k, fields)
    }
    fn fields(&self, state: &State, q: &[C64]) -> Result<Fields> {
        if matches!(
            self.wrong,
            Wrong::NoExchangeInBands | Wrong::SemilocalEigenvalues
        ) {
            let (crystal, options) = self
                .hybrid
                .as_ref()
                .expect("a hybrid variant needs Altered::hybrid");
            *self.semilocal.borrow_mut() = semilocal_bands(crystal, state, q, options);
        }
        self.inner.fields(state, q)
    }
    fn moments<'f>(&self, fields: &'f Fields) -> &'f [C64] {
        self.inner.moments(fields)
    }
    fn bands(
        &self,
        spin: usize,
        k: usize,
        fields: &Fields,
        hubbard: Option<&Matrix>,
    ) -> Result<Bands> {
        let spin = if self.wrong == Wrong::BandsSpinUp {
            0
        } else {
            spin
        };
        let hybrid = self.inner.bands(spin, k, fields, hubbard)?;
        let semilocal = |cache: &Vec<Vec<Bands>>| cache[spin][k].clone();
        Ok(match self.wrong {
            Wrong::NoExchangeInBands => semilocal(&self.semilocal.borrow()),
            Wrong::SemilocalEigenvalues => Bands {
                eigenvalues: semilocal(&self.semilocal.borrow()).eigenvalues,
                ..hybrid
            },
            _ => hybrid,
        })
    }
    fn exchange_gradient(&self, state: &State) -> Result<hybrid_gradient::Gradient> {
        let state = match self.wrong {
            Wrong::ExchangeSpinUp => State {
                bands: state.bands.iter().map(|_| state.bands[0].clone()).collect(),
            },
            Wrong::ExchangeSummedDensity => State {
                bands: summed(&state.bands),
            },
            _ => state.clone(),
        };
        let mut gradient = self.inner.exchange_gradient(&state)?;
        match self.wrong {
            Wrong::NoFunctions => gradient.functions = None,
            Wrong::ShortResponse => {
                gradient.compensation.pop();
            }
            _ => {}
        }
        Ok(gradient)
    }
    fn hat_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<Vec<C64>> {
        self.inner.hat_overlap(fields, atom, axis)
    }
    fn ion_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        self.inner.ion_overlap(fields, atom, axis)
    }
    fn ion_energy(&self, atom: usize, axis: usize) -> Result<f64> {
        self.inner.ion_energy(atom, axis)
    }
    fn dense(&self, atom: usize, axis: usize) -> Result<Dense> {
        self.inner.dense(atom, axis)
    }
    fn constant(&self, atom: usize, axis: usize) -> Result<f64> {
        self.inner.constant(atom, axis)
    }
    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        self.inner.spin_orbit()
    }
    fn spin_orbit_projection(&self, k: usize) -> Result<Matrix> {
        self.inner.spin_orbit_projection(k)
    }
    fn spin_orbit_projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        self.inner.spin_orbit_projection_gradient(k)
    }
    fn core_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        if self.wrong == Wrong::CoreSpinUp {
            // `xc` is the spin average; the two channels differ by `local[0] - local[1]` (the
            // electrostatic part is common), so `v_up = xc + (local_up - local_dn) / 2`.
            let up = fields
                .xc
                .iter()
                .zip(fields.local[0].iter().zip(&fields.local[1]))
                .map(|(x, (u, d))| x + 0.5 * (u - d))
                .collect();
            let only_up = Fields::new(
                fields.local.clone(),
                up,
                fields.weights.clone(),
                fields.moments.clone(),
            );
            return self.inner.core_overlap(&only_up, atom, axis);
        }
        let value = self.inner.core_overlap(fields, atom, axis)?;
        Ok(if self.wrong == Wrong::CoreSign {
            -value
        } else {
            value
        })
    }
    fn hubbard(&self) -> Option<&dyn mandacaru_core::periodic::gradient::Hubbard> {
        self.inner.hubbard()
    }
}
