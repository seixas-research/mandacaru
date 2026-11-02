//! The one-center screened terms of the periodic hybrid: what the PAW spheres add to the exact
//! exchange of the grid pair densities and their compensation charges.
//!
//! In a sphere the screened exchange of the grid is replaced by three things, all of them built
//! per species by [`crate::paw::screened_atom::screened`] (the code the molecular hybrid uses):
//!
//! ```text
//! E = -(a/2) sum_s tr(Q_s K[Q_s]) - a V . (Q_up + Q_dn) - a P(Q_up, Q_dn) - a (X0 - V . D0)
//! K[Q] = sum_k lambda_k F_k Q F_k
//! ```
//!
//! `Q_s = <p_i| rho_s |p_j>` is the projected density of one spin on the atom's projectors,
//! `Delta W^SR = sum_k lambda_k F_k (x) F_k` the screened one-center tensor, `V` the frozen
//! short-range semilocal exchange potential of the reference atom, `P` the spin-scaling remainder
//! (zero for equal spin densities, so absent from a restricted or a time-reversal-symmetric
//! crystal) and the last term a constant counted once per atom, never per k-point.
//!
//! The species code works in real spherical harmonics; a crystal's projectors carry complex ones
//! (`math::angular::complex`). `Y_lm = sum_m' S[m, m'] R_lm'` rotates the factors once,
//! `F^c = S^* F S^T`, and `V` and `P` (both diagonal in `m`, equal for every `m`) are invariant.
use super::screened_augmentation::{OneCenter, OneCenterTerms};
use crate::{
    paw::{pair_tables::Projector, screened_atom::ScreenedAtom},
    periodic::invalid,
    quantum::C64,
    Context, Result,
};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// `S[m, m']` of the degree `l`: `Y_lm = sum_m' S[m, m'] R_lm'`, with `R` the real harmonics
/// `math::angular::cartesian` and `Y` the complex ones with the Condon-Shortley phase of
/// `math::angular::complex`. Rows and columns run `m = -l..=l`.
pub fn harmonic_rotation(l: usize) -> Matrix {
    let n = 2 * l + 1;
    let at = |m: i32| (m + l as i32) as usize;
    let mut s = Matrix::zeros(n, n);
    s[(at(0), at(0))] = C64::new(1., 0.);
    let root = std::f64::consts::FRAC_1_SQRT_2;
    for order in 1..=l as i32 {
        let sign = if order % 2 == 0 { 1. } else { -1. };
        s[(at(order), at(order))] = C64::new(sign * root, 0.);
        s[(at(order), at(-order))] = C64::new(0., sign * root);
        s[(at(-order), at(order))] = C64::new(root, 0.);
        s[(at(-order), at(-order))] = C64::new(0., -root);
    }
    s
}

/// One species: the screened terms in the complex-harmonic layout of its projectors.
pub struct Species {
    /// `F^c_k` and their weights.
    factors: Vec<Matrix>,
    weights: Vec<f64>,
    v: Matrix,
    constant: f64,
    atom: ScreenedAtom,
}

impl Species {
    /// `layout` is the projector layout `atom` was built for (real harmonics).
    pub fn new(atom: ScreenedAtom, layout: &[Projector]) -> Self {
        let n = layout.len();
        let mut s = Matrix::zeros(n, n);
        for (i, a) in layout.iter().enumerate() {
            for (j, b) in layout.iter().enumerate() {
                if a.channel == b.channel && a.radial == b.radial {
                    let l = a.l;
                    s[(i, j)] = harmonic_rotation(l)
                        [((a.m + l as i32) as usize, (b.m + l as i32) as usize)];
                }
            }
        }
        let (conjugate, transpose) = (s.conjugate(), s.transpose());
        let factors = atom
            .factors
            .iter()
            .map(|f| &conjugate * f.map(|x| C64::new(x, 0.)) * &transpose)
            .collect();
        Self {
            factors,
            weights: atom.weights.clone(),
            v: atom.v.map(|x| C64::new(x, 0.)),
            constant: atom.constant,
            atom,
        }
    }
    pub fn projectors(&self) -> usize {
        self.v.nrows()
    }
    /// `K[Q] = sum_k lambda_k F_k Q F_k`.
    fn exchange(&self, q: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(q.nrows(), q.ncols());
        for (f, w) in self.factors.iter().zip(&self.weights) {
            out += (f * q * f) * C64::new(*w, 0.);
        }
        out
    }
}

/// Where one atom's projectors sit in the crystal's `p` and which species it is.
#[derive(Clone, Copy)]
pub struct Sphere {
    pub offset: usize,
    pub species: usize,
}

/// The one-center model of a crystal, every sphere of it.
pub struct SphereModel {
    species: Vec<Species>,
    spheres: Vec<Sphere>,
    p: usize,
    fraction: f64,
}

fn block(m: &Matrix, row: usize, col: usize, n: usize) -> Matrix {
    m.view((row, col), (n, n)).into_owned()
}
fn real(m: &Matrix) -> DMatrix<f64> {
    m.map(|z| z.re)
}

impl SphereModel {
    /// The spheres must tile `p` projectors in order, atom after atom.
    pub fn new(
        species: Vec<Species>,
        spheres: Vec<Sphere>,
        p: usize,
        fraction: f64,
    ) -> Result<Self> {
        let mut next = 0;
        for s in &spheres {
            let n = species
                .get(s.species)
                .ok_or_else(|| invalid("a sphere names an unknown species"))?
                .projectors();
            if s.offset != next {
                return Err(invalid("the spheres' projectors must follow each other"));
            }
            next += n;
        }
        if next != p || !fraction.is_finite() {
            return Err(invalid(format!(
                "the datasets give {next} projectors, the crystal has {p}"
            )));
        }
        Ok(Self {
            species,
            spheres,
            p,
            fraction,
        })
    }
    fn size(&self, sphere: &Sphere) -> usize {
        self.species[sphere.species].projectors()
    }
}

impl OneCenter for SphereModel {
    fn collinear(&self, ctx: &Context, projected: &[Matrix]) -> Result<OneCenterTerms> {
        let restricted = match projected.len() {
            1 => true,
            2 => false,
            _ => return Err(invalid("one or two spin densities are needed")),
        };
        if projected.iter().any(|m| m.shape() != (self.p, self.p)) {
            return Err(invalid("a projected density must be p x p"));
        }
        let a = self.fraction;
        // The projected matrix of a restricted crystal holds both spins.
        let (share, copies) = if restricted { (0.5, 2.) } else { (1., 1.) };
        let mut operators = vec![Matrix::zeros(self.p, self.p); projected.len()];
        let mut energy = 0.;
        for sphere in &self.spheres {
            let (species, n) = (&self.species[sphere.species], self.size(sphere));
            let at = sphere.offset;
            let q: Vec<Matrix> = projected
                .iter()
                .map(|m| block(m, at, at, n) * C64::new(share, 0.))
                .collect();
            for (s, q) in q.iter().enumerate() {
                let k = species.exchange(q);
                energy -= copies * a * (0.5 * (q * &k).trace().re + (&species.v * q).trace().re);
                let mut view = operators[s].view_mut((at, at), (n, n));
                view += k + &species.v;
            }
            energy -= a * species.constant;
            if !restricted {
                let terms = species
                    .atom
                    .polarization
                    .evaluate(ctx, &real(&q[0]), &real(&q[1]))?;
                energy -= a * terms.energy;
                for (s, d) in [terms.up, terms.down].into_iter().enumerate() {
                    let mut view = operators[s].view_mut((at, at), (n, n));
                    view += d.map(|x| C64::new(x, 0.));
                }
            }
        }
        Ok(OneCenterTerms { energy, operators })
    }

    fn spinor(&self, _ctx: &Context, projected: &Matrix) -> Result<OneCenterTerms> {
        let p = self.p;
        if projected.shape() != (2 * p, 2 * p) {
            return Err(invalid("a projected spinor density must be 2p x 2p"));
        }
        let a = self.fraction;
        let mut operator = Matrix::zeros(2 * p, 2 * p);
        let mut energy = 0.;
        for sphere in &self.spheres {
            let (species, n) = (&self.species[sphere.species], self.size(sphere));
            let at = sphere.offset;
            let q = |s: usize, t: usize| block(projected, s * p + at, t * p + at, n);
            for s in 0..2 {
                for t in 0..2 {
                    let k = species.exchange(&q(s, t));
                    energy -= 0.5 * a * (q(t, s) * &k).trace().re;
                    let mut view = operator.view_mut((s * p + at, t * p + at), (n, n));
                    view += k;
                    if s == t {
                        energy -= a * (&species.v * q(s, s)).trace().re;
                        view += &species.v;
                    }
                }
            }
            energy -= a * species.constant;
        }
        Ok(OneCenterTerms {
            energy,
            operators: vec![operator],
        })
    }
}
