//! The spin-orbit union-projector term of the spinor crystal force (CS-2).
//!
//! The spin-orbit matrix of a k-point is the molecule's operator
//! ([`crate::models::spin_orbit::pair`], the single source of the coupling) between the
//! projections `C` of the Bloch sums on the dataset's spin-orbit union projectors:
//!
//! ```text
//! h^SO(k) = (I2 x C) B (I2 x C)^+,   B = pair(I_p, I_p)   (2p x 2p, constant, Hermitian)
//! ```
//!
//! `B` collects the angular `<l m s| L.S |l m' t>` and the radial `D^SO` blocks and does not
//! depend on any position, so the spin-orbit energy `E_SO = sum_k w_k Re tr[D_k h^SO_k]` moves
//! with the atoms only through `C`: there is no `dD_SO` and no hat. With the frozen spinor
//! density `D` (`2m x 2m`, the Lowdin frame carried) and the ownership identity of the scalar
//! projections, `dC_X = P_X G - G Q_X` (`G = dC/dc` the derivative with respect to the center of
//! the ROW function, `P_X` / `Q_X` the owner masks of the rows / of the projector columns),
//!
//! ```text
//! dE_SO/dtau_X = sum_k w_k 2 Re tr[D (I2 x dC_X) B (I2 x C)^+]
//!              = sum_k w_k 2 Re sum_{mu p} dC_X[mu, p] Z[p, mu],
//! Z = sum_s [B (I2 x C)^+ D]_{(s,p),(s,mu)}                      (p x m)
//! ```
//!
//! `Z` is formed once per k-point and axis-independent; the rows of `G` are charged to the
//! owner of `mu` and the columns to the owner of `p` with the opposite sign, so the sum over the
//! atoms vanishes by construction of the scatter (a structural property of the ownership
//! identity; the physical translation invariance is the invariance of the energy under a rigid
//! shift of every center, which the tests check on the energy itself). The term is linear in
//! the coupling strength: every entry of `B` scales with it.
//!
//! Hook. `pulay::contract` calls [`SpinOrbit::add_k`] once per k-point for a spinor request,
//! with the spinor `D_k` of the frozen state and the provider's spin-orbit projections
//! ([`Derivatives::spin_orbit_projection`](super::Derivatives::spin_orbit_projection) and its
//! gradient: the same `projection(_gradient)` code with the SO projector list). The term needs no
//! `dS`, hat or `W`. `D_k` and the owners of the basis functions are in the provider's frame.
use crate::{
    models::spin_orbit::{pair, Channel, Projectors},
    periodic::invalid,
    quantum::C64,
    Context, Result,
};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// The constant `B` of the spin-orbit operator over `count` union projectors: `(2 count, 2 count)`,
/// spin major (`alpha` block first), through [`pair`] with the identity as both projections.
pub fn operator(ctx: &Context, count: usize, channels: &[Channel]) -> Result<Matrix> {
    let mut identity = ctx.allocate::<C64>(crate::product(&[count, count])?)?;
    for i in 0..count {
        identity.as_mut_slice()[i * count + i] = C64::new(1., 0.);
    }
    let b = pair(
        ctx,
        Projectors {
            left: identity.as_slice(),
            right: identity.as_slice(),
            rows: count,
            columns: count,
            count,
        },
        channels,
        true,
    )?;
    Ok(Matrix::from_row_slice(2 * count, 2 * count, b.as_slice()))
}

/// `dE_SO/dtau` per atom, split by who moves: the basis functions (rows of `C`) or the
/// projectors (its columns).
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub rows: Vec<[f64; 3]>,
    pub columns: Vec<[f64; 3]>,
}

impl Gradient {
    pub fn new(atoms: usize) -> Self {
        Gradient {
            rows: vec![[0.; 3]; atoms],
            columns: vec![[0.; 3]; atoms],
        }
    }
    /// `dE_SO/dtau` per atom.
    pub fn total(&self) -> Vec<[f64; 3]> {
        self.rows
            .iter()
            .zip(&self.columns)
            .map(|(r, c)| std::array::from_fn(|a| r[a] + c[a]))
            .collect()
    }
}

/// The spin-orbit operator of a crystal: `B` and who owns the basis functions and the union
/// projectors.
#[derive(Clone, Debug)]
pub struct SpinOrbit {
    operator: Matrix,
    row_owner: Vec<usize>,
    projector_owner: Vec<usize>,
    atoms: usize,
}

impl SpinOrbit {
    /// `operator` is `B` (`2p x 2p`); `row_owner` / `projector_owner` give the atom of each
    /// basis function / union projector.
    pub fn new(
        operator: Matrix,
        row_owner: Vec<usize>,
        projector_owner: Vec<usize>,
        atoms: usize,
    ) -> Result<Self> {
        let p = projector_owner.len();
        if operator.shape() != (2 * p, 2 * p)
            || operator
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
            || row_owner
                .iter()
                .chain(&projector_owner)
                .any(|&x| x >= atoms)
        {
            return Err(invalid(
                "spin-orbit force: B must be 2p x 2p and finite, owners must be atoms",
            ));
        }
        Ok(SpinOrbit {
            operator,
            row_owner,
            projector_owner,
            atoms,
        })
    }

    pub fn operator(&self) -> &Matrix {
        &self.operator
    }

    fn check(&self, c: &Matrix, density: &Matrix) -> Result<(usize, usize)> {
        let (m, p) = (self.row_owner.len(), self.projector_owner.len());
        if c.shape() != (m, p) || density.shape() != (2 * m, 2 * m) {
            return Err(invalid(
                "spin-orbit force: C must be m x p and the density 2m x 2m",
            ));
        }
        Ok((m, p))
    }

    /// `h^SO = (I2 x C) B (I2 x C)^+` for the projections `c` (`m x p`).
    pub fn matrix(&self, c: &Matrix) -> Matrix {
        let (m, p) = (c.nrows(), c.ncols());
        let wide = Matrix::from_fn(2 * m, 2 * p, |i, j| {
            if i / m == j / p {
                c[(i % m, j % p)]
            } else {
                C64::default()
            }
        });
        &wide * &self.operator * wide.adjoint()
    }

    /// `w Re tr[D h^SO]`: this k-point's share of `E_SO`.
    pub fn energy(&self, weight: f64, c: &Matrix, density: &Matrix) -> Result<f64> {
        self.check(c, density)?;
        Ok(weight * (density * self.matrix(c)).trace().re)
    }

    /// `Z = sum_s [B (I2 x C)^+ D]_{(s,p),(s,mu)}`, `p x m`.
    fn z(&self, c: &Matrix, density: &Matrix) -> Matrix {
        let (m, p) = (c.nrows(), c.ncols());
        let adjoint = c.adjoint();
        // (I2 x C)^+ D: block (t, s) is C^+ D_{ts}.
        let projected = Matrix::from_fn(2 * p, 2 * m, |i, j| {
            (0..m)
                .map(|mu| adjoint[(i % p, mu)] * density[((i / p) * m + mu, j)])
                .sum()
        });
        let b = &self.operator * projected;
        Matrix::from_fn(p, m, |q, mu| b[(q, mu)] + b[(p + q, m + mu)])
    }

    /// Adds `w 2 Re sum_{mu p} dC_X[mu, p] Z[p, mu]` of one k-point to every atom. `gc[axis]` is
    /// `G` (`m x p`, the derivative of `C` with respect to the ROW center), `density` the spinor
    /// `D_k` (`2m x 2m`); a non-Hermitian `density` contributes its Hermitian part, the only one
    /// that `Re tr[D h]` sees.
    pub fn add_k(
        &self,
        out: &mut Gradient,
        weight: f64,
        c: &Matrix,
        gc: &[Matrix; 3],
        density: &Matrix,
    ) -> Result<()> {
        let (m, p) = self.check(c, density)?;
        if gc.iter().any(|g| g.shape() != (m, p))
            || out.rows.len() != self.atoms
            || out.columns.len() != self.atoms
        {
            return Err(invalid(
                "spin-orbit force: G must be m x p per axis and the output one entry per atom \
                 in both shares",
            ));
        }
        let hermitian = (density + density.adjoint()) * C64::new(0.5, 0.);
        let z = self.z(c, &hermitian);
        for (axis, g) in gc.iter().enumerate() {
            for mu in 0..m {
                for q in 0..p {
                    let t = 2. * weight * (g[(mu, q)] * z[(q, mu)]).re;
                    out.rows[self.row_owner[mu]][axis] += t;
                    out.columns[self.projector_owner[q]][axis] -= t;
                }
            }
        }
        Ok(())
    }

    /// The strain derivative of this k-point's share of `E_SO`, `w 2 Re sum_{mu p} dC_ab[mu, p]
    /// Z[p, mu]` for the nine entries `[a][b]` (AS-8): the same `Z` as [`add_k`](Self::add_k),
    /// contracted with the strain tangent of the projections `dc[a][b]` (`m x p`, every function
    /// and projector moves: nothing is owner-masked). `B` is constant, so there is no `dB`, no hat
    /// and no `W`.
    pub fn strain_k(
        &self,
        weight: f64,
        c: &Matrix,
        dc: &[[Matrix; 3]; 3],
        density: &Matrix,
    ) -> Result<[[f64; 3]; 3]> {
        let (m, p) = self.check(c, density)?;
        if dc.iter().flatten().any(|g| g.shape() != (m, p)) {
            return Err(invalid(
                "spin-orbit stress: dC must be m x p for each of the nine strain entries",
            ));
        }
        let hermitian = (density + density.adjoint()) * C64::new(0.5, 0.);
        let z = self.z(c, &hermitian);
        Ok(std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let mut sum = 0.;
                for mu in 0..m {
                    for q in 0..p {
                        sum += (dc[a][b][(mu, q)] * z[(q, mu)]).re;
                    }
                }
                2. * weight * sum
            })
        }))
    }
}
