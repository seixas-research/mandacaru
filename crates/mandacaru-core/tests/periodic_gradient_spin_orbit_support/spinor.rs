//! A spinor geometry provider on the toy, in the doubled frame the spinor engine will use: the
//! frame has `2m` functions (spin major, `alpha` block first), so the spinor density of a
//! frozen state is the frame's `D`; the spin-orbit projections stay in the scalar frame (`m x p`).
//! The scalar part is a crystal with no projectors, channels or matrix terms (every scalar
//! derivative is zero), so the whole engine total is the spin-orbit term and its oracle is
//! `oracle()`.
#![allow(dead_code)]

use super::{density, row_derivative};
use crate::hubbard_gradient_support::{Kind, Matrix, Toy};
use mandacaru_core::{
    periodic::gradient::{
        spin_orbit::SpinOrbit, Bands, Dense, Derivatives, Frozen, Layout, Request, State, Term,
    },
    quantum::C64,
    Error, Result,
};

pub struct SpinorToy {
    pub toy: Toy,
    pub layout: Layout,
    pub so: SpinOrbit,
    /// False: the provider carries no spin-orbit data and raises.
    pub data: bool,
    unit: Matrix,
}

impl SpinorToy {
    pub fn new(toy: &Toy, so: SpinOrbit, spinor: bool, data: bool) -> Self {
        let owner = toy.basis_owner();
        let layout = Layout {
            atoms: toy.atoms(),
            m: 2 * toy.basis(),
            p: 1,
            spins: 1,
            degeneracy: 1.,
            weights: toy.weights.clone(),
            row_owner: owner.iter().chain(&owner).copied().collect(),
            projector_owner: vec![0],
            reference: 0.,
            request: Request {
                functional: "lda".into(),
                spinor,
                single_precision: false,
            },
        };
        Self {
            toy: toy.clone(),
            layout,
            so,
            data,
            unit: Matrix::zeros(1, 1),
        }
    }

    fn zeros(&self) -> Matrix {
        Matrix::zeros(self.layout.m, self.layout.m)
    }
}

fn outside<T>() -> Result<T> {
    Err(Error::Unsupported(
        "the spin-orbit toy has no density fields or bands".into(),
    ))
}

impl Derivatives for SpinorToy {
    type Fields = ();

    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn blocks(&self) -> &[Matrix] {
        &[]
    }
    fn augmentation(&self) -> &Matrix {
        &self.unit
    }
    fn ion_operator(&self) -> &Matrix {
        &self.unit
    }
    fn projection(&self, _k: usize) -> Result<Matrix> {
        Ok(Matrix::zeros(self.layout.m, 1))
    }
    fn projection_gradient(&self, _k: usize) -> Result<[Matrix; 3]> {
        Ok(std::array::from_fn(|_| Matrix::zeros(self.layout.m, 1)))
    }
    fn cross(&self, _spin: usize, _k: usize, _term: Term, _f: &()) -> Result<[Matrix; 3]> {
        Ok(std::array::from_fn(|_| self.zeros()))
    }
    fn sphere_cross(&self, _spin: usize, _k: usize, _f: &()) -> Result<Vec<[Matrix; 3]>> {
        Ok(Vec::new())
    }
    fn fields(&self, _state: &State, _q: &[C64]) -> Result<()> {
        outside()
    }
    fn moments<'a>(&self, _fields: &'a ()) -> &'a [C64] {
        &[]
    }
    fn bands(&self, _s: usize, _k: usize, _f: &(), _h: Option<&Matrix>) -> Result<Bands> {
        outside()
    }
    fn hat_overlap(&self, _f: &(), _atom: usize, _axis: usize) -> Result<Vec<C64>> {
        Ok(Vec::new())
    }
    fn ion_overlap(&self, _f: &(), _atom: usize, _axis: usize) -> Result<f64> {
        Ok(0.)
    }
    fn ion_energy(&self, _atom: usize, _axis: usize) -> Result<f64> {
        Ok(0.)
    }
    fn dense(&self, _atom: usize, _axis: usize) -> Result<Dense> {
        Ok(Dense {
            u: Matrix::zeros(0, 0),
            uion: Vec::new(),
            linear: Vec::new(),
        })
    }
    fn constant(&self, _atom: usize, _axis: usize) -> Result<f64> {
        Ok(0.)
    }
    fn core_overlap(&self, _f: &(), _atom: usize, _axis: usize) -> Result<f64> {
        Ok(0.)
    }

    fn spin_orbit(&self) -> Result<&SpinOrbit> {
        if self.data {
            Ok(&self.so)
        } else {
            outside()
        }
    }
    fn spin_orbit_projection(&self, k: usize) -> Result<Matrix> {
        Ok(self.toy.bloch(Kind::Projection, self.toy.kpoints()[k]))
    }
    fn spin_orbit_projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        Ok(row_derivative(&self.toy, self.toy.kpoints()[k]))
    }
}

/// A frozen spinor state: at every k-point the bands `R` (`2m x 2m`, fixed random, all occupied),
/// so the density is `R R^+`, Hermitian and positive with every spin block populated.
pub fn frozen(toy: &Toy, seed: u64) -> Frozen<()> {
    let n = 2 * toy.basis();
    let coefficients = density(toy.basis(), seed) + Matrix::identity(n, n) * C64::new(1.5, 0.);
    let bands = Bands {
        coefficients,
        eigenvalues: vec![0.; n],
        occupations: vec![1.; n],
    };
    Frozen {
        state: State {
            bands: vec![vec![bands; toy.kpoints().len()]],
        },
        fields: (),
        q: Vec::new(),
        hubbard: None,
    }
}
