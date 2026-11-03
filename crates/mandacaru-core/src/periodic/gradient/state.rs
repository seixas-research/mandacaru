//! The converged-state data of the force: `D = C f C^+`, `W = C f (eps - reference) C^+`, the
//! channel moments `q`, and the output-density freeze that closes the traps of `solve()`.
//!
//! `solve()` diagonalizes with the operator of the INPUT density of its last iteration, returns
//! the potentials of that input density, and shifts the eigenvalues by the reference. A force
//! built from those is wrong at first order in the SCF residual (and by `reference * tr[D dS]`
//! in `W`). [`freeze`] recomputes `q`, the potentials and the +U operator from the density the
//! last diagonalization produced, re-diagonalizes with them, and keeps the new bands.
use super::{invalid, Derivatives, Hubbard, Matrix};
use crate::{quantum::C64, Result};

/// The occupied (and partly occupied) orbitals of one spin and k-point.
#[derive(Clone, Debug)]
pub struct Bands {
    /// `m x nb`, `c^+ S c = 1`.
    pub coefficients: Matrix,
    /// As the solver returns them: shifted by the reference.
    pub eigenvalues: Vec<f64>,
    /// Occupation of each band, `0..=1` per spin channel.
    pub occupations: Vec<f64>,
}

impl Bands {
    fn check(&self, m: usize) -> Result<()> {
        let nb = self.coefficients.ncols();
        if self.coefficients.nrows() != m
            || self.eigenvalues.len() != nb
            || self.occupations.len() != nb
            || self.eigenvalues.iter().any(|x| !x.is_finite())
            || self.occupations.iter().any(|f| !f.is_finite() || *f < 0.0)
        {
            return Err(invalid("crystal gradient: inconsistent bands"));
        }
        Ok(())
    }

    /// `sum_i f_i s_i c_i c_i^+`.
    fn weighted_sum(&self, scale: impl Fn(usize) -> f64) -> Matrix {
        let c = &self.coefficients;
        let scaled = Matrix::from_fn(c.nrows(), c.ncols(), |i, j| {
            c[(i, j)] * (self.occupations[j] * scale(j))
        });
        &scaled * c.adjoint()
    }

    /// `D = C f C^+`.
    pub fn density(&self) -> Matrix {
        self.weighted_sum(|_| 1.0)
    }

    /// `W = C f (eps - reference) C^+`: the eigenvalues are the UNSHIFTED ones.
    pub fn weighted(&self, reference: f64) -> Matrix {
        self.weighted_sum(|j| self.eigenvalues[j] - reference)
    }
}

/// `bands[spin][k]`.
#[derive(Clone, Debug)]
pub struct State {
    pub bands: Vec<Vec<Bands>>,
}

impl State {
    pub fn check<D: Derivatives>(&self, d: &D) -> Result<()> {
        let l = d.layout();
        if self.bands.len() != l.spins || self.bands.iter().any(|s| s.len() != l.weights.len()) {
            return Err(invalid(
                "crystal gradient: one set of bands per spin and k-point",
            ));
        }
        self.bands.iter().flatten().try_for_each(|b| b.check(l.m))
    }
}

/// `q_a = deg sum_{s,k} w_k tr[B_a C^+ D C]`, the channel moments of the density of `state`.
pub fn moments<D: Derivatives>(d: &D, state: &State) -> Result<Vec<C64>> {
    let l = d.layout();
    let mut q = vec![C64::new(0.0, 0.0); d.blocks().len()];
    for k in 0..l.weights.len() {
        let c = d.projection(k)?;
        for s in 0..l.spins {
            let r = c.adjoint() * state.bands[s][k].density() * &c;
            for (qa, block) in q.iter_mut().zip(d.blocks()) {
                let trace: C64 = (0..l.p)
                    .flat_map(|i| (0..l.p).map(move |j| (i, j)))
                    .map(|(i, j)| block[(i, j)] * r[(j, i)])
                    .sum();
                *qa += trace * (l.degeneracy * l.weights[k]);
            }
        }
    }
    // The k-sum visited one point per star: the moments the potentials are made of are the group average.
    d.symmetrize_moments(&mut q)?;
    Ok(q)
}

/// A consistent state for the contraction: bands, the fields and `q` of the density the bands
/// were re-diagonalized for, and the +U operator of the same density.
pub struct Frozen<F> {
    pub state: State,
    pub fields: F,
    pub q: Vec<C64>,
    /// `O_s` per spin channel, `None` without +U.
    pub hubbard: Option<Vec<Matrix>>,
}

/// Recomputes everything from the OUTPUT density of `input`, then re-diagonalizes.
pub fn freeze<D: Derivatives>(
    d: &D,
    input: &State,
    hubbard: Option<&dyn Hubbard>,
) -> Result<Frozen<D::Fields>> {
    input.check(d)?;
    let l = d.layout();
    let q = moments(d, input)?;
    let fields = d.fields(input, &q)?;
    let hubbard = hubbard.map(|h| h.operators(input)).transpose()?;
    if let Some(o) = &hubbard {
        if o.len() != l.spins || o.iter().any(|m| m.shape() != (l.p, l.p)) {
            return Err(invalid("crystal gradient: one p x p +U operator per spin"));
        }
    }
    let bands = (0..l.spins)
        .map(|s| {
            (0..l.weights.len())
                .map(|k| d.bands(s, k, &fields, hubbard.as_ref().map(|o| &o[s])))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let state = State { bands };
    state.check(d)?;
    Ok(Frozen {
        state,
        fields,
        q,
        hubbard,
    })
}
