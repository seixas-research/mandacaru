//! The Fu-Kane parity criterion for the Z2 index of an inversion-symmetric, time-reversal-invariant
//! insulator (Fu and Kane, PRB 76, 045302 (2007)): the product of the inversion eigenvalues
//! `xi = +-1` of the occupied Kramers pairs at the eight time-reversal-invariant points
//! `Gamma_i = (n_1 G_1 + n_2 G_2 + n_3 G_3)/2` gives `delta_i`, and
//! `(-1)^{nu_0} = prod_i delta_i`, `(-1)^{nu_j} = prod_{n_j = 1} delta_i`, the product over the four
//! points of the plane `k_j = 1/2`.
//!
//! The inversion is read off the PAW-LCAO basis. About the center `O`, a function `phi_lm` of an
//! atom at `c` goes to `(-1)^l phi_lm` of the atom at `c'` with `c + c' = 2 O - T`, `T` a lattice
//! vector, and the Bloch sums `chi_k = sum_R e^{ik.R} phi(r - c - R)` of the setup (the lattice
//! phase only) go to
//!
//! ```text
//! P chi_{c,lm,k} = (-1)^l e^{ik.T} chi_{c',lm,-k},     chi_{-k} = chi_k at a TRIM (-k = k - 2k, 2k in G)
//! ```
//!
//! so on the coefficients `(P c)_{c'} = (-1)^l e^{ik.T} c_c`, the same on both spinor components (the
//! inversion leaves the spin alone). The PAW partial waves and projectors of equivalent atoms are
//! equal, so the same map holds for the augmented state. At a TRIM the occupied block of
//! `M = C^+ S P C` (`S` the PAW overlap) is Hermitian with eigenvalues `+-1`; anything else raises
//! (the crystal is not inversion-symmetric about `O`, or a band group is split).
use super::{invalid, Bands, Metal, Sampler};
use crate::{math::linalg::eigen, periodic::wannier::sampling::Frame, quantum::C64, Error, Result};
use nalgebra::{DMatrix, Vector3};
use std::sync::Arc;

type Matrix = DMatrix<C64>;

/// The anti-Hermitian part of `M` and the deviation of its eigenvalues from +-1 above this raise.
pub const PARITY_TOLERANCE: f64 = 1e-6;
/// Two levels closer than this (Hartree) are one degenerate level.
const DEGENERACY: f64 = 1e-7;
/// Empty levels whose parity is listed beyond the occupied ones, per TRIM.
const EMPTY_LEVELS: usize = 6;

/// The inversion of the basis: where each function goes and with which phase.
#[derive(Clone, Debug)]
pub struct Inversion {
    /// The inversion center `O` (fractional).
    pub center: [f64; 3],
    /// `image[a]`: the function `a` goes to `image[a]`.
    pub image: Vec<usize>,
    /// `(-1)^l` of each function.
    pub sign: Vec<f64>,
    /// The lattice vector `T_a = 2 O - c_a - c_image` (Cartesian, Bohr) of each function.
    pub shift: Vec<Vector3<f64>>,
}

impl Inversion {
    /// The map of `sampler`'s basis about an inversion center of its atoms (the first found, an
    /// atom's site if there is one; the others differ by half lattice vectors `t`, which multiply the
    /// eigenvalue at a TRIM `k` by `e^{ik.t} = +-1` and leave the product over the eight points
    /// alone).
    pub fn find(sampler: &Sampler) -> Result<Self> {
        let setup = sampler.setup;
        let lattice = sampler.lattice;
        let inverse = lattice
            .try_inverse()
            .ok_or_else(|| invalid("singular crystal lattice"))?;
        let atoms: Vec<Vector3<f64>> = setup.centers.iter().map(|c| Vector3::from(*c)).collect();
        let lattice_vector = |d: Vector3<f64>| -> bool {
            let f = inverse * d;
            f.iter().all(|x| (x - x.round()).abs() < 1e-6)
        };
        let partner = |i: usize, o: &Vector3<f64>| -> Option<usize> {
            let target = 2. * o - atoms[i];
            (0..atoms.len()).find(|j| {
                setup.atoms_species[*j] == setup.atoms_species[i]
                    && lattice_vector(target - atoms[*j])
            })
        };
        // An atom's own site first (the conventional origin of a parity table), then the midpoint
        // of atom 0 and another.
        let center = (0..atoms.len())
            .map(|j| atoms[j])
            .chain((1..atoms.len()).map(|j| 0.5 * (atoms[0] + atoms[j])))
            .find(|o| (0..atoms.len()).all(|i| partner(i, o).is_some()))
            .ok_or_else(|| {
                invalid(
                    "the crystal has no inversion center: the Fu-Kane parity criterion needs an \
                     inversion-symmetric structure",
                )
            })?;
        let functions = sampler.crystal.m;
        let mut image = vec![usize::MAX; functions];
        let mut sign = vec![0.; functions];
        let mut shift = vec![Vector3::zeros(); functions];
        let shells = setup.shells.shells();
        // The shells of each atom, in basis order, with the atom their center is.
        let atom_of = |c: [f64; 3]| -> Result<usize> {
            let c = Vector3::from(c);
            (0..atoms.len())
                .find(|a| (atoms[*a] - c).norm() < 1e-8)
                .ok_or_else(|| invalid("a basis shell is not centered on an atom"))
        };
        let mut owners = Vec::with_capacity(shells.len());
        for shell in shells {
            owners.push(atom_of(shell.center)?);
        }
        for (s, shell) in shells.iter().enumerate() {
            let atom = owners[s];
            let other = partner(atom, &center).ok_or_else(|| invalid("no inversion partner"))?;
            // The same-l shells of an atom come in the same order for atoms of one species.
            let rank = (0..s)
                .filter(|t| owners[*t] == atom && shells[*t].l == shell.l)
                .count();
            let twin = (0..shells.len())
                .filter(|t| owners[*t] == other && shells[*t].l == shell.l)
                .nth(rank)
                .ok_or_else(|| invalid("an atom's inversion partner lacks one of its shells"))?;
            let twin_shell = &shells[twin];
            if twin_shell.ms != shell.ms
                || (!Arc::ptr_eq(&twin_shell.radial, &shell.radial)
                    && (twin_shell.radial.support() - shell.radial.support()).abs() > 1e-9)
            {
                return Err(invalid(
                    "inversion-related shells carry different radial functions or components",
                ));
            }
            let t = 2. * center - atoms[atom] - atoms[other];
            for (row, to) in shell.rows.iter().zip(&twin_shell.rows) {
                image[*row] = *to;
                sign[*row] = if shell.l % 2 == 0 { 1. } else { -1. };
                shift[*row] = t;
            }
        }
        if image.iter().any(|v| *v == usize::MAX) {
            return Err(invalid("a basis function has no inversion image"));
        }
        let fractional = inverse * center;
        Ok(Self {
            center: [fractional[0], fractional[1], fractional[2]],
            image,
            sign,
            shift,
        })
    }

    /// `P c` for the coefficient columns `c` (`components * m` rows) at the Cartesian `k`.
    fn apply(&self, c: &Matrix, k: &Vector3<f64>, components: usize) -> Matrix {
        let m = self.image.len();
        let mut out = Matrix::zeros(c.nrows(), c.ncols());
        for component in 0..components {
            for a in 0..m {
                let angle = k.dot(&self.shift[a]);
                let factor = C64::from_polar(self.sign[a], angle);
                for col in 0..c.ncols() {
                    out[(component * m + self.image[a], col)] =
                        factor * c[(component * m + a, col)];
                }
            }
        }
        out
    }
}

/// One parity eigenvalue of a level (a degenerate multiplet gives one entry per state).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    /// Hartree, on the scale of the SCF Fermi level.
    pub energy: f64,
    /// `+1` or `-1`.
    pub parity: i8,
}

/// The inversion eigenvalues at one time-reversal-invariant point.
#[derive(Clone, Debug)]
pub struct Trim {
    /// `n_j in {0, 1}`: the point is `sum_j n_j G_j / 2`.
    pub index: [u8; 3],
    /// Ascending levels: the occupied ones first, then up to six empty ones.
    pub levels: Vec<Level>,
    /// The occupied bands.
    pub occupied: usize,
    /// Occupied states of parity `-1`.
    pub minus: usize,
    /// The product of the parities over the occupied Kramers pairs (`+1` or `-1`).
    pub delta: i8,
    /// The largest deviation of the eigenvalues of `M` from +-1, and the anti-Hermitian part of `M`.
    pub residual: f64,
}

/// The Fu-Kane parities of every TRIM and the Z2 indices they give.
#[derive(Clone, Debug)]
pub struct Parity {
    pub inversion: Inversion,
    /// The eight points, `index` in C order (`n_3` fastest).
    pub trims: Vec<Trim>,
    /// `nu_0` of `(-1)^{nu_0} = prod delta_i`.
    pub strong: u8,
    /// `nu_j`: the product over the plane `k_j = 1/2`.
    pub weak: [u8; 3],
    /// The Z2 index of the plane `k_c = offset`, `planes[c][0]` for 0 and `planes[c][1]` for 1/2.
    pub planes: [[u8; 2]; 3],
    /// Occupied states per Kramers multiplet unit: 2 for a spin-degenerate crystal, 1 for a spinor.
    pub spin_degenerate: bool,
}

fn exponent(delta: i8) -> u8 {
    (delta < 0) as u8
}

/// The parity criterion of the converged crystal's occupied bands (an insulator, checked).
pub fn parity(sampler: &Sampler) -> Result<Parity> {
    let c = sampler.crystal;
    if c.spins != 1 || !super::flow::time_reversal_invariant(c) {
        return Err(Error::Unsupported(
            "the Fu-Kane parity criterion needs a time-reversal-invariant crystal in one spin \
             channel (a spin-degenerate crystal or a spinor with a time-reversal-invariant \
             density); a collinear spin-polarized or magnetic crystal has no Z2 index"
                .into(),
        ));
    }
    let manifold = sampler.manifold(Bands::Occupied, Metal::Undefined)?;
    let occupied = manifold.ranges[0].1;
    let components = if c.spinor { 2 } else { 1 };
    let inversion = Inversion::find(sampler)?;
    let ctx = &c.context;
    let mut trims = Vec::with_capacity(8);
    for index in 0..8usize {
        let n = [(index >> 2) & 1, (index >> 1) & 1, index & 1];
        let fractional = [0.5 * n[0] as f64, 0.5 * n[1] as f64, 0.5 * n[2] as f64];
        let point = sampler
            .setup
            .matrices_at(ctx, &[fractional], &[1.])?
            .pop()
            .ok_or_else(|| invalid("the setup made no k-point"))?;
        // The occupied bands and the lowest empty ones, cut where no multiplet is split.
        let all = c.bands();
        let mut selected = (occupied + EMPTY_LEVELS).min(all);
        let bands: Vec<usize> = (0..selected).collect();
        let frame = Frame::from_point(c, sampler.state, &point, 0, &bands)?;
        let energies = frame.levels.as_slice();
        while selected > occupied
            && selected < energies.len()
            && energies[selected] - energies[selected - 1] < DEGENERACY
        {
            selected -= 1;
        }
        let k = Vector3::from(point.k);
        let coefficients = frame.coefficients.columns(0, selected).into_owned();
        let pc = inversion.apply(&coefficients, &k, components);
        let mut m_matrix = Matrix::zeros(selected, selected);
        for component in 0..components {
            let left = coefficients.rows(component * c.m, c.m);
            let right = pc.rows(component * c.m, c.m);
            m_matrix += left.adjoint() * &point.overlap * right;
        }
        // Cluster the degenerate levels and diagonalize the parity inside each. The occupied
        // clusters must be exact (+-1 and closed under P); an empty cluster that is not ends the
        // list of empty levels.
        let mut list = Vec::with_capacity(selected);
        let mut residual = 0f64;
        let mut start = 0;
        while start < selected {
            let mut end = start + 1;
            while end < selected && energies[end] - energies[end - 1] < DEGENERACY {
                end += 1;
            }
            let block = m_matrix
                .view((start, start), (end - start, end - start))
                .into_owned();
            // The columns of the cluster against every selected row: a state P maps out of the
            // cluster is a symmetry that is not there.
            let leak = (1.
                - (m_matrix.columns(start, end - start).norm_squared() / (end - start) as f64)
                    .sqrt())
            .abs();
            let hermitian = (&block + block.adjoint()) * C64::new(0.5, 0.);
            let (values, _) = eigen(hermitian)?;
            let deviation = values
                .iter()
                .map(|v| (v.abs() - 1.).abs())
                .fold((&block - block.adjoint()).norm().max(leak), f64::max);
            if start >= occupied && deviation > PARITY_TOLERANCE {
                break;
            }
            residual = residual.max(deviation);
            for v in values {
                list.push((energies[start], if v > 0. { 1i8 } else { -1i8 }));
            }
            start = end;
        }
        if residual > PARITY_TOLERANCE || list.len() < occupied {
            return Err(invalid(format!(
                "the inversion eigenvalues at the TRIM {fractional:?} are not +-1 (deviation \
                 {residual:.2e}): the crystal is not inversion-symmetric about {:?}, or its \
                 occupied states are not those of a symmetric Hamiltonian",
                inversion.center
            )));
        }
        let minus = list[..occupied].iter().filter(|(_, p)| *p < 0).count();
        let delta = if c.spinor {
            if minus % 2 == 1 {
                return Err(invalid(format!(
                    "{minus} occupied states of parity -1 at the TRIM {fractional:?}: Kramers \
                     pairs share their parity, so the count is even for a time-reversal-invariant \
                     spinor"
                )));
            }
            if minus / 2 % 2 == 0 {
                1
            } else {
                -1
            }
        } else if minus % 2 == 0 {
            // A spin-degenerate band is its own Kramers pair (spin up and down).
            1
        } else {
            -1
        };
        trims.push(Trim {
            index: [n[0] as u8, n[1] as u8, n[2] as u8],
            levels: list
                .iter()
                .map(|(energy, parity)| Level {
                    energy: *energy,
                    parity: *parity,
                })
                .collect(),
            occupied,
            minus,
            delta,
            residual,
        });
    }
    let plane = |axis: usize, half: usize| -> u8 {
        let product: i8 = trims
            .iter()
            .filter(|t| t.index[axis] as usize == half)
            .map(|t| t.delta)
            .product();
        exponent(product)
    };
    let all: i8 = trims.iter().map(|t| t.delta).product();
    Ok(Parity {
        inversion,
        strong: exponent(all),
        weak: [plane(0, 1), plane(1, 1), plane(2, 1)],
        planes: [
            [plane(0, 0), plane(0, 1)],
            [plane(1, 0), plane(1, 1)],
            [plane(2, 0), plane(2, 1)],
        ],
        trims,
        spin_degenerate: !c.spinor,
    })
}
