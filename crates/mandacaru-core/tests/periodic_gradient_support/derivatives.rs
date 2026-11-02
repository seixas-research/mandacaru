//! The toy as a geometry provider: `Derivatives` and `Hubbard` for [`Prepared`].
//!
//! The row-center derivative `G` of a Bloch block is read off the toy's real image blocks
//! (`s * dB/dd`, the phases and the sign convention of each kind live in `hubbard_gradient_support`);
//! nothing here applies an ownership mask, that is the engine's.
use super::{
    c,
    scf::{self, observe, Fields},
    Geometry, Model, Prepared,
};
use crate::hubbard_gradient_support::{Kind, Matrix};
use mandacaru_core::{
    periodic::gradient::{Bands, Dense, Derivatives, Hubbard, Layout, Request, State, Term},
    quantum::C64,
    Result,
};

pub struct ToyDerivatives {
    pub prep: Prepared,
    pub layout: Layout,
}

impl ToyDerivatives {
    pub fn new(model: &Model) -> Self {
        let toy = &model.toy;
        let layout = Layout {
            atoms: toy.atoms(),
            m: toy.basis(),
            p: toy.projectors(),
            spins: model.spins(),
            degeneracy: model.degeneracy,
            weights: toy.weights.clone(),
            row_owner: toy.basis_owner(),
            projector_owner: toy.projector_owner(),
            reference: model.reference,
            request: Request {
                functional: "lda".to_string(),
                spinor: false,
                single_precision: false,
            },
        };
        ToyDerivatives {
            prep: Prepared::new(model),
            layout,
        }
    }

    fn geometry(&self) -> Geometry<'_> {
        Geometry {
            toy: &self.prep.model.toy,
        }
    }

    /// `G^a(k) = sum_R phase * s * dB_R/dd^a`: the derivative with respect to the row center.
    fn row_derivative(&self, kind: Kind, k: usize) -> [Matrix; 3] {
        let toy = &self.prep.model.toy;
        let kpoint = toy.kpoints()[k];
        let s = toy.frame(kind).4;
        let blocks = toy.blocks(kind);
        std::array::from_fn(|axis| {
            toy.sum(kind, kpoint, |image, i, j| {
                s * blocks.gradients[image][axis][(i, j)]
            })
        })
    }
}

impl Derivatives for ToyDerivatives {
    type Fields = Fields;

    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn blocks(&self) -> &[Matrix] {
        &self.prep.blocks
    }
    fn augmentation(&self) -> &Matrix {
        &self.prep.q_matrix
    }
    fn ion_operator(&self) -> &Matrix {
        &self.prep.d_ion
    }
    fn projection(&self, k: usize) -> Result<Matrix> {
        Ok(self.prep.projection[k].clone())
    }
    fn projection_gradient(&self, k: usize) -> Result<[Matrix; 3]> {
        Ok(self.row_derivative(Kind::Projection, k))
    }
    fn cross(&self, _spin: usize, k: usize, term: Term, fields: &Fields) -> Result<[Matrix; 3]> {
        let (kind, scale) = match term {
            Term::Overlap => (Kind::Overlap, 1.0),
            Term::Kinetic => (Kind::Hopping, 1.0),
            Term::Local => (Kind::Local, fields.v),
            Term::ShortRange => (Kind::Short, 1.0),
        };
        Ok(self.row_derivative(kind, k).map(|g| g * c(scale)))
    }

    fn sphere_cross(&self, _spin: usize, _k: usize, _fields: &Fields) -> Result<Vec<[Matrix; 3]>> {
        // The toy's short-range kind is a pair function: no third center.
        Ok(Vec::new())
    }

    fn fields(&self, state: &State, q: &[C64]) -> Result<Fields> {
        let r = observe(&self.prep, state).r;
        let real: Vec<f64> = q.iter().map(|x| x.re).collect();
        Ok(scf::fields(&self.prep, r, &real))
    }
    fn moments<'a>(&self, fields: &'a Fields) -> &'a [C64] {
        &fields.moments
    }
    fn bands(
        &self,
        spin: usize,
        k: usize,
        fields: &Fields,
        hubbard: Option<&Matrix>,
    ) -> Result<Bands> {
        Ok(scf::bands(&self.prep, spin, k, fields, hubbard))
    }

    fn hat_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<Vec<C64>> {
        let g = self.geometry();
        Ok((0..self.prep.blocks.len())
            .map(|a| c(fields.r * g.d_hat(a, atom, axis)))
            .collect())
    }
    fn ion_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        Ok(fields.r * self.geometry().d_ion(atom, axis))
    }
    fn ion_energy(&self, atom: usize, axis: usize) -> Result<f64> {
        Ok(self.geometry().d_ion_dense(atom, axis))
    }
    fn dense(&self, atom: usize, axis: usize) -> Result<Dense> {
        let g = self.geometry();
        let h = self.prep.blocks.len();
        Ok(Dense {
            u: g.d_u(atom, axis).map(c),
            uion: (0..h).map(|a| c(g.d_uion(a, atom, axis))).collect(),
            linear: (0..h).map(|a| c(g.d_linear(a, atom, axis))).collect(),
        })
    }
    fn constant(&self, atom: usize, axis: usize) -> Result<f64> {
        Ok(self.geometry().d_constant(atom, axis))
    }
    fn core_overlap(&self, fields: &Fields, atom: usize, axis: usize) -> Result<f64> {
        Ok(fields.vxc * self.geometry().d_core(atom, axis))
    }
}

/// The +U hook of the toy: `O(n)` of the density of the state, from the model's correction.
pub struct ToyHubbard<'a> {
    pub prep: &'a Prepared,
}

impl Hubbard for ToyHubbard<'_> {
    fn operators(&self, state: &State) -> Result<Vec<Matrix>> {
        Ok(scf::hubbard(self.prep, &observe(self.prep, state).n).1)
    }
}
