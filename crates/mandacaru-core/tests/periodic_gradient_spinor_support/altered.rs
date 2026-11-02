//! A spinor provider with ONE thing wrong in its spin structure, for the negative controls of
//! the four-channel local term and of the doubled frame. Everything delegates to the real
//! [`SetupDerivatives`]; the named piece is damaged on the way to the engine.
#![allow(dead_code)]

use mandacaru_core::{
    periodic::{
        gradient::{scf, spin_orbit::SpinOrbit, Bands, Dense, Derivatives, Layout, State, Term},
        setup::{
            provider::{Fields, SetupDerivatives},
            pulay::Matrix,
        },
        Crystal, Options,
    },
    quantum::C64,
    Result,
};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wrong {
    /// The transverse spin blocks (`sigma_x`, `sigma_y` channels) of the local term dropped.
    NoTransverse,
    /// The `m_x` and `m_y` fields exchanged (`sigma_x` and `sigma_y` swapped): a wrong Pauli channel.
    XySwapped,
    /// The diagonal spin blocks of the local term exchanged (`m_z` of the other sign).
    FlippedMz,
    /// Only the charge channel of the local term (the magnetization fields dropped).
    ChargeChannelOnly,
    /// Every matrix term in the `alpha` block only (the `beta` block zeroed): a wrong spin factor.
    OneSpinBlock,
    /// `W` weighted by the eigenvalues of the Hamiltonian WITHOUT the spin-orbit matrix (the
    /// scalar spectrum, band by band), the spinor vectors and occupations kept.
    ScalarEigenvalues,
}

pub struct SpinorAltered<'a> {
    pub inner: SetupDerivatives<'a>,
    wrong: Wrong,
    /// The crystal, its copy without the spin-orbit matrix, and the SCF options.
    scalar: Option<(&'a Crystal, &'a Crystal, Options)>,
    /// The bands of the output density for the Hamiltonian without spin-orbit, formed by `fields`.
    spectrum: RefCell<Vec<Vec<Bands>>>,
}

impl<'a> SpinorAltered<'a> {
    pub fn new(inner: SetupDerivatives<'a>, wrong: Wrong) -> Self {
        Self {
            inner,
            wrong,
            scalar: None,
            spectrum: RefCell::new(Vec::new()),
        }
    }

    /// `W` from the eigenvalues of `without` (the same crystal with a zero spin-orbit matrix).
    pub fn scalar_eigenvalues(
        inner: SetupDerivatives<'a>,
        crystal: &'a Crystal,
        without: &'a Crystal,
        options: Options,
    ) -> Self {
        let mut altered = Self::new(inner, Wrong::ScalarEigenvalues);
        altered.scalar = Some((crystal, without, options));
        altered
    }

    fn m(&self) -> usize {
        self.inner.layout().m / 2
    }

    /// The `beta` rows and columns zeroed.
    fn alpha_only(&self, a: Matrix) -> Matrix {
        let m = self.m();
        let mut out = a;
        let (r, c) = out.shape();
        out.view_mut((m.min(r), 0), (r - m.min(r), c))
            .fill(C64::default());
        out.view_mut((0, m.min(c)), (r, c - m.min(c)))
            .fill(C64::default());
        out
    }

    fn damaged_local(&self, g: Matrix) -> Matrix {
        let m = self.m();
        let block = |i: usize, j: usize| g.view((i * m, j * m), (m, m)).into_owned();
        let mut out = g.clone();
        match self.wrong {
            Wrong::NoTransverse => {
                out.view_mut((0, m), (m, m)).fill(C64::default());
                out.view_mut((m, 0), (m, m)).fill(C64::default());
            }
            Wrong::XySwapped => {
                // ab = G1 - i G2, ba = G1 + i G2: exchange the roles of G1 and G2.
                let i = C64::i();
                let (ab, ba) = (block(0, 1), block(1, 0));
                let g1 = (&ab + &ba) * C64::new(0.5, 0.);
                let g2 = (&ab - &ba) * (i * 0.5);
                out.view_mut((0, m), (m, m)).copy_from(&(&g2 - &g1 * i));
                out.view_mut((m, 0), (m, m)).copy_from(&(&g2 + &g1 * i));
            }
            Wrong::FlippedMz => {
                out.view_mut((0, 0), (m, m)).copy_from(&block(1, 1));
                out.view_mut((m, m), (m, m)).copy_from(&block(0, 0));
            }
            Wrong::ChargeChannelOnly => {
                // The charge channel is the mean of the two diagonal blocks.
                let charge = (block(0, 0) + block(1, 1)) * C64::new(0.5, 0.);
                out = Matrix::zeros(2 * m, 2 * m);
                out.view_mut((0, 0), (m, m)).copy_from(&charge);
                out.view_mut((m, m), (m, m)).copy_from(&charge);
            }
            Wrong::OneSpinBlock | Wrong::ScalarEigenvalues => {}
        }
        out
    }
}

impl Derivatives for SpinorAltered<'_> {
    type Fields = Fields;

    fn layout(&self) -> &Layout {
        self.inner.layout()
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
        let c = self.inner.projection(k)?;
        Ok(if self.wrong == Wrong::OneSpinBlock {
            self.alpha_only(c)
        } else {
            c
        })
    }
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        let g = self.inner.projection_gradient(k)?;
        Ok(if self.wrong == Wrong::OneSpinBlock {
            g.map(|g| self.alpha_only(g))
        } else {
            g
        })
    }
    fn cross(&self, spin: usize, k: usize, term: Term, fields: &Fields) -> Result<[Matrix; 3]> {
        let g = self.inner.cross(spin, k, term, fields)?;
        Ok(match (self.wrong, term) {
            (Wrong::OneSpinBlock, _) => g.map(|g| self.alpha_only(g)),
            (_, Term::Local) => g.map(|g| self.damaged_local(g)),
            _ => g,
        })
    }
    fn sphere_cross(&self, spin: usize, k: usize, fields: &Fields) -> Result<Vec<[Matrix; 3]>> {
        self.inner.sphere_cross(spin, k, fields)
    }
    fn fields(&self, state: &State, q: &[C64]) -> Result<Fields> {
        if let Some((crystal, without, options)) = &self.scalar {
            let densities = scf::densities(crystal, &state.bands)?;
            let output = scf::output_fields(crystal, &densities, q)?;
            *self.spectrum.borrow_mut() = scf::rediagonalize(without, &output, options, &[])?;
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
        let bands = self.inner.bands(spin, k, fields, hubbard)?;
        Ok(if self.wrong == Wrong::ScalarEigenvalues {
            Bands {
                eigenvalues: self.spectrum.borrow()[spin][k].eigenvalues.clone(),
                ..bands
            }
        } else {
            bands
        })
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
    fn core_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        self.inner.core_overlap(fields, atom, axis)
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
}
