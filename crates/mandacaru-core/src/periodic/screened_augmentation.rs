//! The compensation charges of the pair densities of the periodic screened exact exchange.
//!
//! In PAW a pair density of two states is the smooth pair on the FFT grid plus the compensation
//! charges, one shape `g_a` per multipole channel `a = (atom, L, M)`:
//!
//! ```text
//! rho_{mu n}(r) = u_mu(k)^* u_n(q)(r) + sum_a Q^a_{mu n} g_a(r - R_atom)
//! Q^a_{mu n}   = sum_{ij} C(k)_{mu i} B_a[i, j] (C(q)^+ c_n)_j
//! ```
//!
//! `C(k)_{mu i} = <phi_mu(k) | p_i>` is the stored `projections` matrix, `B_a` the stored
//! compensation moments `Delta Q^{LM}_{ij}` (`blocks`, the matrices the Hartree term contracts
//! with the density: `q_a = tr(P B_a)`), and `c_n` the coefficients of the occupied state with the
//! square root of its occupation folded in. The Fourier component at `Q = q - k + G` is
//!
//! ```text
//! rho(Q) = e^{-i G.r0} DFT(G) / N  +  (1/Omega) sum_a Q^a hat_a(Q),   hat_a(Q) = 4 pi (-i)^L Y_LM F_L(|Q|) e^{-i Q.R}
//! ```
//!
//! and the grid code works with `N rho = DFT + sum_a Q^a T_a`, so [`Augmentation::pair_shapes`]
//! returns `T_a(G) = hat_a(Q) e^{+i G.r0} / dv`: the phase at `G` (the origin of the grid), the
//! transform at `Q` (the position of the atom). Its `Q = 0` entry is the unit-charge part:
//! the grid sum plus `sqrt(4 pi) Q^{00}` of a pair of normalized states integrates to their
//! overlap, which is the sum rule the tests gate.
//!
//! The one-center terms (`Delta W^SR`, `V`, the frozen semilocal exchange) are NOT here. They
//! plug in through [`OneCenter`].
//!
//! Two approximations are measured, not hidden (log, section W-E4P / P1). The shapes are
//! evaluated at the one wrapped wave vector `Q` of each grid index, not averaged over the
//! aliases the grid's lattice operations pick as the setup's `hats` are: the averaged form
//! breaks the exact translation invariance of `K` (the origin phase is at one alias, the atoms
//! at the others) and moves `K` by 2e-8 (Si) of itself. The tail of the shapes beyond the box
//! (the on-site `tail` of the Hartree `U`) is not added: 4e-9 of `K`, 4e-7 of the
//! compensation's own effect.
use super::{invalid, Crystal, KPoint};
use crate::{
    math::{fft::frequency, reciprocal},
    periodic::setup::shapes::{Shapes, Sites, Transient},
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::{DMatrix, Matrix3, Vector3};

type Matrix = DMatrix<C64>;

/// What a one-center model adds to the exchange of the crystal, in the convention of the
/// grid part: every `K_s(k)` gains `C(k) O_s C(k)^+` and `energy` (Hartree, the fraction and the
/// constant included) is added to the exchange energy, so the Hamiltonian takes `-fraction K`.
pub struct OneCenterTerms {
    pub energy: f64,
    /// `O_s`: `p x p` per spin channel, `2p x 2p` for the one block of a spinor.
    pub operators: Vec<Matrix>,
}

/// What completes the exact exchange inside the PAW spheres, beyond the grid and the
/// compensation charges of [`Augmentation`]. A one-center model is a function of the projected
/// density, `P_s = sum_k w_k C(k)^+ D_s(k) C(k)` (`p x p`, the matrix `density` of the solver
/// contracts with `B_a`), and of nothing else: it sees all spin channels at once, since the
/// spheres' semilocal exchange couples them.
pub trait OneCenter {
    /// `projected[s]`: one matrix per spin channel as the solver stores it (a restricted
    /// crystal has one, holding both spins).
    fn collinear(&self, ctx: &Context, projected: &[Matrix]) -> Result<OneCenterTerms>;
    /// `projected`: the `2p x 2p` density of a spinor, the four spin blocks of `p x p`.
    fn spinor(&self, ctx: &Context, projected: &Matrix) -> Result<OneCenterTerms>;
}

/// The compensation charges of the pair densities of one crystal: where the atoms are (the
/// snapshot does not hold them) and the shape tables of its compensation radii.
pub struct Augmentation<'a> {
    sites: Sites<'a>,
    shapes: &'a Shapes,
    one_center: Option<&'a dyn OneCenter>,
}

impl<'a> Augmentation<'a> {
    /// `sites` must describe `crystal`: its channels are the stored moment blocks, in order,
    /// and its lattice is the cell of the grid.
    pub fn new(crystal: &Crystal, sites: Sites<'a>, shapes: &'a Shapes) -> Result<Self> {
        sites.check()?;
        if sites.channels.len() != crystal.channels {
            return Err(invalid(format!(
                "the augmentation has {} channels, the crystal {}",
                sites.channels.len(),
                crystal.channels
            )));
        }
        let shape = crystal.grid.index.shape();
        let cell =
            crystal.grid.step * Matrix3::from_diagonal(&Vector3::from_fn(|i, _| shape[i] as f64));
        if (cell - sites.lattice).norm() > 1e-8 * cell.norm() {
            return Err(invalid(
                "the augmentation lattice is not the cell of the grid",
            ));
        }
        Ok(Self {
            sites,
            shapes,
            one_center: None,
        })
    }
    /// The same augmentation with a one-center model attached.
    pub fn with_one_center(mut self, model: &'a dyn OneCenter) -> Self {
        self.one_center = Some(model);
        self
    }
    pub(super) fn one_center(&self) -> Option<&'a dyn OneCenter> {
        self.one_center
    }
    /// The atoms, channels and cell the shapes are evaluated for.
    pub(super) fn sites(&self) -> &Sites<'a> {
        &self.sites
    }

    /// `T_a(G) = hat_a(Q) e^{+i G.r0} / dv`, `(channels, n)` row-major, for the pairs of two
    /// points whose wave vectors differ by `k_diff = q - k`. `Q` is the box vector of the grid
    /// index shifted by `k_diff` and wrapped into the FFT box, as the kernel's `Q` is; `G` is
    /// that `Q` minus `k_diff`. Rows are built in blocks reserved from the context.
    pub fn pair_shapes(&self, c: &Crystal, k_diff: [f64; 3]) -> Result<Buffer<C64>> {
        self.shape_rows(c, k_diff, None)
    }

    /// `dT_a/dR_{atom(a)}^axis = -i Q_axis T_a`: the derivative of [`Augmentation::pair_shapes`]
    /// with respect to the center of the channel's atom, at the same wrapped `Q`.
    pub fn pair_shape_slopes(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        axis: usize,
    ) -> Result<Buffer<C64>> {
        if axis > 2 {
            return Err(invalid("a Cartesian axis is 0, 1 or 2"));
        }
        self.shape_rows(c, k_diff, Some(axis))
    }

    /// The rows of [`Augmentation::pair_shapes`], times `-i Q_axis` when `slope` names an axis.
    fn shape_rows(
        &self,
        c: &Crystal,
        k_diff: [f64; 3],
        slope: Option<usize>,
    ) -> Result<Buffer<C64>> {
        let rows = self.sites.channels.len();
        let n = c.grid.index.points();
        let shape = c.grid.index.shape();
        let dv = c.grid.index.dv();
        let mut out = c.context.allocate::<C64>(rows * n)?;
        if rows == 0 {
            return Ok(out);
        }
        let b = reciprocal::reciprocal_cell(&c.grid)?;
        let kappa = b
            .try_inverse()
            .ok_or_else(|| invalid("singular reciprocal cell"))?
            * Vector3::from(k_diff);
        let block = (Transient::of(&c.context).dense_bytes / (rows * 16 + 64 + 56)).max(1);
        let _scratch = c
            .context
            .reserve(crate::product(&[block.min(n), rows * 16 + 64 + 56])?)?;
        let mut vectors: Vec<[f64; 3]> = Vec::with_capacity(block.min(n));
        let mut phases: Vec<C64> = Vec::with_capacity(block.min(n));
        for start in (0..n).step_by(block) {
            vectors.clear();
            phases.clear();
            for at in start..(start + block).min(n) {
                let index = [
                    at / (shape[1] * shape[2]),
                    at / shape[2] % shape[1],
                    at % shape[2],
                ];
                let wrapped = Vector3::from_fn(|a, _| {
                    let size = shape[a] as f64;
                    let x = frequency(index[a], shape[a]) + kappa[a];
                    x - size * ((x + 0.5 * size + 1e-9) / size).floor()
                });
                let q = b * wrapped;
                let g = q - Vector3::from(k_diff);
                let angle =
                    g[0] * c.grid.origin[0] + g[1] * c.grid.origin[1] + g[2] * c.grid.origin[2];
                vectors.push(q.into());
                phases.push(C64::from_polar(1. / dv, angle));
            }
            self.shapes
                .transforms(&self.sites, &vectors, &mut out.as_mut_slice()[start..], n)?;
            for row in 0..rows {
                for (v, phase) in phases.iter().enumerate() {
                    out.as_mut_slice()[row * n + start + v] *= phase;
                    if let Some(axis) = slope {
                        out.as_mut_slice()[row * n + start + v] *=
                            C64::new(0., -vectors[v][axis]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// `a_j = <p_j | psi>`, `C(q)^+ c` for every spin component of the coefficients `c`
    /// (`components * m`, the components of a spinor one after the other): `components * p`.
    pub fn project(&self, c: &Crystal, source: &KPoint, coefficients: &[C64]) -> Vec<C64> {
        let (m, p) = (c.m, c.p);
        let mut out = vec![C64::default(); (coefficients.len() / m) * p];
        for (component, chunk) in coefficients.chunks(m).enumerate() {
            for i in 0..p {
                out[component * p + i] = (0..m)
                    .map(|mu| source.projections[(mu, i)].conj() * chunk[mu])
                    .sum();
            }
        }
        out
    }

    /// `Q^a_{mu n}` of one occupied state against every basis function of the target point:
    /// `(components * m) x channels`, row `component * m + mu`. `projected` is the output of
    /// [`Augmentation::project`] at the source point.
    pub fn moments(&self, c: &Crystal, target: &KPoint, projected: &[C64]) -> Matrix {
        let (m, p, channels) = (c.m, c.p, self.sites.channels.len());
        let components = projected.len() / p;
        let mut out = Matrix::zeros(components * m, channels);
        let mut by_channel = Matrix::zeros(p, channels);
        for component in 0..components {
            let a = &projected[component * p..(component + 1) * p];
            for channel in 0..channels {
                let block = &c.blocks.as_slice()[channel * p * p..(channel + 1) * p * p];
                for j in 0..p {
                    by_channel[(j, channel)] = (0..p).map(|i| block[j * p + i] * a[i]).sum();
                }
            }
            let rows = &target.projections * &by_channel;
            out.view_mut((component * m, 0), (m, channels))
                .copy_from(&rows);
        }
        out
    }
}

/// `row += sum_a moments[a] * shapes[a, :]`: the compensation charges of one pair, added to its
/// grid transform (`row` has `n` entries, `shapes` is [`Augmentation::pair_shapes`]).
pub fn add_compensation(row: &mut [C64], moments: &[C64], shapes: &[C64]) {
    let n = row.len();
    for (a, &moment) in moments.iter().enumerate() {
        if moment == C64::default() {
            continue;
        }
        for (value, shape) in row.iter_mut().zip(&shapes[a * n..(a + 1) * n]) {
            *value += moment * shape;
        }
    }
}
