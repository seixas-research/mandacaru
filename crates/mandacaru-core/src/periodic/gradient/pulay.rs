//! The matrix-derivative terms: `sum_{s,k} deg w_k Re tr[D dH_X - W dS_X]` with the basis
//! functions and the projectors moving rigidly with their atoms.
//!
//! A cross term `dM_X = P_X G + G^+ P_X` contracts as `2 Re sum_{mu in X} (G D)_{mumu}`, so one
//! pass over the elements scatters the force of every atom at once. The projections use the
//! ownership identity `dC_X = P_X G_C - G_C Q_X`: with `Z = H_C C^+ D - q C^+ W` (`p x m`),
//! `H_C = D_ion + sum_a m_a B_a (+ O_s)`, the contribution is `2 Re sum G_C[mu,p] Z[p,mu]`,
//! added to the owner of `mu` (the rows) and subtracted from the owner of `p` (the columns).
//! The +U operator joins `D_ion` here and nowhere else.
use super::{invalid, spin_orbit::Gradient, Derivatives, Frozen, Matrix, Term};
use crate::{quantum::C64, Result};

/// `dE/dtau` per atom, matrix terms only.
pub struct Pulay {
    pub overlap: Vec<[f64; 3]>,
    pub kinetic: Vec<[f64; 3]>,
    pub local: Vec<[f64; 3]>,
    pub short_range: Vec<[f64; 3]>,
    pub kinetic_density: Vec<[f64; 3]>,
    pub rows: Vec<[f64; 3]>,
    pub columns: Vec<[f64; 3]>,
    /// The spin-orbit union-projector term (spinor requests; zero otherwise).
    pub spin_orbit: Gradient,
}

impl Pulay {
    fn new(atoms: usize) -> Self {
        let zero = || vec![[0.0; 3]; atoms];
        Pulay {
            overlap: zero(),
            kinetic: zero(),
            local: zero(),
            short_range: zero(),
            kinetic_density: zero(),
            rows: zero(),
            columns: zero(),
            spin_orbit: Gradient::new(atoms),
        }
    }

    fn term(&mut self, term: Term) -> &mut Vec<[f64; 3]> {
        match term {
            Term::Overlap => &mut self.overlap,
            Term::Kinetic => &mut self.kinetic,
            Term::Local => &mut self.local,
            Term::ShortRange => &mut self.short_range,
            Term::KineticDensity => &mut self.kinetic_density,
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

/// `Re sum_{ij} left[i, j] right[j, i]` (`Re tr[left right]`), without forming the product.
fn trace_re(left: &Matrix, right: &Matrix) -> f64 {
    let (n, m) = left.shape();
    let mut sum = C64::new(0., 0.);
    for j in 0..m {
        for i in 0..n {
            sum += left[(i, j)] * right[(j, i)];
        }
    }
    sum.re
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

/// The derivative data of one k-point: what [`contract`] contracts, produced on a k-point lane.
struct KData {
    c: Matrix,
    gc: [Matrix; 3],
    /// Per spin channel: the cross matrices of the terms, in [`Term::of`] order, and the
    /// moving spheres.
    spins: Vec<(Vec<(Term, [Matrix; 3])>, Vec<[Matrix; 3]>)>,
    spin_orbit: Option<(Matrix, [Matrix; 3])>,
}

pub fn contract<D: Derivatives>(d: &D, frozen: &Frozen<D::Fields>) -> Result<Pulay> {
    let reference = d.layout().reference;
    contract_matrices(d, &frozen.fields, frozen.hubbard.as_deref(), |s, k| {
        let bands = &frozen.state.bands[s][k];
        (bands.density(), bands.weighted(reference))
    })
}

/// [`contract`] at any density `D` and energy-weighted density `W` of each `(spin, k)`
/// (`matrices(spin, k)`, Hermitian `m x m`, in the provider's rows; not necessarily those of
/// bands: the relaxed densities of a correlated state are not positive), with `extra[spin]`
/// added to the projector operator (the +U operator of the crystal; the Hartree-Fock one-center
/// operator of the PAW-LCAO supercell).
pub fn contract_matrices<D: Derivatives>(
    d: &D,
    fields: &D::Fields,
    extra: Option<&[Matrix]>,
    matrices: impl Fn(usize, usize) -> (Matrix, Matrix) + Sync,
) -> Result<Pulay> {
    let l = d.layout();
    let mut out = Pulay::new(l.atoms);
    let terms = Term::of(&l.request.functional)?;
    // What one point holds: C, G_C (m x p), per spin the term crosses and the spheres (m x m).
    let per_lane = super::matrix_bytes(4, l.m, l.p)?
        .checked_add(super::matrix_bytes(
            l.spins * 3 * (terms.len() + l.atoms),
            l.m,
            l.m,
        )?)
        .ok_or(crate::Error::Overflow)?;
    let produce = |k: usize| -> Result<KData> {
        let c = d.projection(k)?;
        let gc = d.projection_gradient(k)?;
        let mut spins = Vec::with_capacity(l.spins);
        for s in 0..l.spins {
            let crosses = terms
                .iter()
                .map(|&t| d.cross(s, k, t, fields).map(|g| (t, g)))
                .collect::<Result<Vec<_>>>()?;
            let spheres = d.sphere_cross(s, k, fields)?;
            spins.push((crosses, spheres));
        }
        let spin_orbit = if l.request.spinor {
            Some((
                d.spin_orbit_projection(k)?,
                d.spin_orbit_projection_gradient(k)?,
            ))
        } else {
            None
        };
        Ok(KData {
            c,
            gc,
            spins,
            spin_orbit,
        })
    };
    // The contraction of each point in k order, as the serial loop ran it.
    let consume = |k: usize, data: KData| -> Result<()> {
        let KData {
            c,
            gc,
            spins,
            spin_orbit,
        } = data;
        for (s, (crosses, spheres)) in spins.into_iter().enumerate() {
            let (density, weighted) = matrices(s, k);
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
            if !spheres.is_empty() && spheres.len() != l.atoms {
                return Err(invalid("crystal gradient: one sphere derivative per atom"));
            }
            for (atom, g) in spheres.iter().enumerate() {
                for (axis, ga) in g.iter().enumerate() {
                    // `Re tr[G D]` as the elementwise sum `Re sum_ij G[i, j] D[j, i]`.
                    let t = 2.0 * w * trace_re(ga, &density);
                    out.short_range[atom][axis] -= t;
                }
            }
            let operator = projector_operator(d, fields, extra.map(|o| &o[s]));
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
        if let Some((c_so, g_so)) = spin_orbit {
            if l.spins != 1 || l.degeneracy != 1.0 {
                return Err(invalid(
                    "crystal gradient: a spinor layout has one spin channel of degeneracy 1",
                ));
            }
            // The spin-orbit matrix moves with the atoms only through its own projections: no
            // `dS`, no hat, no `W`. `D` is the spinor density in the provider's frame.
            let so = d.spin_orbit()?;
            for s in 0..l.spins {
                so.add_k(
                    &mut out.spin_orbit,
                    l.degeneracy * l.weights[k],
                    &c_so,
                    &g_so,
                    &matrices(s, k).0,
                )?;
            }
        }
        Ok(())
    };
    super::kpoint_windows(d.context(), l.weights.len(), per_lane, produce, consume)?;
    Ok(out)
}
