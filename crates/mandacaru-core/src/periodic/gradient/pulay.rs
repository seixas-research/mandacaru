//! The matrix-derivative terms: `sum_{s,k} deg w_k Re tr[D dH_X - W dS_X]` with the basis
//! functions and the projectors moving rigidly with their atoms.
//!
//! A cross term `dM_X = P_X G + G^+ P_X` contracts as `2 Re sum_{mu in X} (G D)_{mumu}`, so one
//! pass over the elements scatters the force of every atom at once. The projections use the
//! ownership identity `dC_X = P_X G_C - G_C Q_X`: with `Z = H_C C^+ D - q C^+ W` (`p x m`),
//! `H_C = D_ion + sum_a m_a B_a (+ O_s)`, the contribution is `2 Re sum G_C[mu,p] Z[p,mu]`,
//! added to the owner of `mu` (the rows) and subtracted from the owner of `p` (the columns).
//! The +U operator joins `D_ion` here and nowhere else.
use super::{invalid, Derivatives, Frozen, Matrix, Term};
use crate::{quantum::C64, Result};

/// `dE/dtau` per atom, matrix terms only.
pub struct Pulay {
    pub overlap: Vec<[f64; 3]>,
    pub kinetic: Vec<[f64; 3]>,
    pub local: Vec<[f64; 3]>,
    pub short_range: Vec<[f64; 3]>,
    pub rows: Vec<[f64; 3]>,
    pub columns: Vec<[f64; 3]>,
}

impl Pulay {
    fn new(atoms: usize) -> Self {
        let zero = || vec![[0.0; 3]; atoms];
        Pulay {
            overlap: zero(),
            kinetic: zero(),
            local: zero(),
            short_range: zero(),
            rows: zero(),
            columns: zero(),
        }
    }

    fn term(&mut self, term: Term) -> &mut Vec<[f64; 3]> {
        match term {
            Term::Overlap => &mut self.overlap,
            Term::Kinetic => &mut self.kinetic,
            Term::Local => &mut self.local,
            Term::ShortRange => &mut self.short_range,
        }
    }
}

/// `D_ion + sum_a m_a B_a (+ O)`: the projector-space operator of one spin channel.
pub fn projector_operator<D: Derivatives>(
    d: &D,
    fields: &D::Fields,
    hubbard: Option<&Matrix>,
) -> Matrix {
    let mut operator = d.ion_operator().clone();
    for (block, &m) in d.blocks().iter().zip(d.moments(fields)) {
        operator += block * m;
    }
    if let Some(o) = hubbard {
        operator += o;
    }
    operator
}

/// `2 Re sum_nu G[mu,nu] M[nu,mu]` for every row `mu`, scattered to the owner of `mu`.
fn scatter_rows(
    out: &mut [[f64; 3]],
    owner: &[usize],
    axis: usize,
    weight: f64,
    g: &Matrix,
    m: &Matrix,
) {
    for (mu, &atom) in owner.iter().enumerate() {
        let row: C64 = (0..g.ncols()).map(|nu| g[(mu, nu)] * m[(nu, mu)]).sum();
        out[atom][axis] += 2.0 * weight * row.re;
    }
}

pub fn contract<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Pulay> {
    let l = d.layout();
    let mut out = Pulay::new(l.atoms);
    for k in 0..l.weights.len() {
        let c = d.projection(k)?;
        let gc = d.projection_gradient(k)?;
        for s in 0..l.spins {
            let crosses = Term::ALL
                .iter()
                .map(|&t| d.cross(s, k, t, &frozen.fields).map(|g| (t, g)))
                .collect::<Result<Vec<_>>>()?;
            let bands = &frozen.state.bands[s][k];
            let (density, weighted) = (bands.density(), bands.weighted(l.reference));
            let w = l.degeneracy * l.weights[k];
            for (term, g) in &crosses {
                // P1 enters the energy through -W dS, every other term through D dH.
                let (m, sign) = match term {
                    Term::Overlap => (&weighted, -1.0),
                    _ => (&density, 1.0),
                };
                for (axis, ga) in g.iter().enumerate() {
                    scatter_rows(out.term(*term), &l.row_owner, axis, sign * w, ga, m);
                }
            }
            // P4's moving spheres: each sphere atom pays for the functions' rigid translation.
            let spheres = d.sphere_cross(s, k, &frozen.fields)?;
            if !spheres.is_empty() && spheres.len() != l.atoms {
                return Err(invalid("crystal gradient: one sphere derivative per atom"));
            }
            for (atom, g) in spheres.iter().enumerate() {
                for (axis, ga) in g.iter().enumerate() {
                    let t = 2.0 * w * (ga * &density).trace().re;
                    out.short_range[atom][axis] -= t;
                }
            }
            let hubbard = frozen.hubbard.as_ref().map(|o| &o[s]);
            let operator = projector_operator(d, &frozen.fields, hubbard);
            let cd = c.adjoint() * &density;
            let z = &operator * &cd - d.augmentation() * c.adjoint() * &weighted;
            for (axis, ga) in gc.iter().enumerate() {
                for (mu, &row) in l.row_owner.iter().enumerate() {
                    for (p, &column) in l.projector_owner.iter().enumerate() {
                        let t = 2.0 * w * (ga[(mu, p)] * z[(p, mu)]).re;
                        out.rows[row][axis] += t;
                        out.columns[column][axis] -= t;
                    }
                }
            }
        }
    }
    Ok(out)
}
