//! Strain derivatives of the dense reciprocal sums `U`, `U_ion` and `E_ion` of
//! [`super::electrostatics::couplings`].
//!
//! Convention (see [`super::strain_grid`]): `x -> (1 + eps) x` on the lattice and the atoms, the
//! integer triples of the sum, the Gaussian width, the shape tables and the on-site tail all
//! pinned at the reference cell. A pinned triple `n` has `G = B n -> (1 - eps^T) G`, the atom
//! phases `G.R` do not move, `Omega -> Omega (1 + tr eps)`. The sum is the one `couplings` runs
//! over the reference cell's `|G| <= cutoff`: re-enumerating the list at a strained cell would
//! add a jump (vectors entering and leaving the ball) that is not a derivative, so the list is
//! the reference's by construction here, and the tail beyond the cutoff, a function of the pinned
//! cutoff radius alone, has zero derivative.
use super::{
    electrostatics::{coulomb, ion_transform},
    shapes::{visit_lattice, Shapes, Sites, Transient},
};
use crate::{Buffer, Context, Result};
use num_complex::Complex64;

/// `d / d eps_pq` of [`super::electrostatics::Couplings`] for the nine strain components `pq`
/// (`p * 3 + q`; the shapes are not symmetric in `pq`).
pub struct CouplingsStrain {
    /// `dU[a, b]`, `(9, channels * channels)` row-major per component.
    pub u: Buffer<Complex64>,
    /// `dU_ion[a]`, `(9, channels)`.
    pub u_ion: Buffer<Complex64>,
    /// `dE_ion`, symmetric.
    pub e_ion: [[f64; 3]; 3],
    channels: usize,
}
impl CouplingsStrain {
    pub fn u(&self, p: usize, q: usize) -> &[Complex64] {
        let size = self.channels * self.channels;
        &self.u.as_slice()[(p * 3 + q) * size..(p * 3 + q + 1) * size]
    }
    pub fn u_ion(&self, p: usize, q: usize) -> &[Complex64] {
        &self.u_ion.as_slice()[(p * 3 + q) * self.channels..(p * 3 + q + 1) * self.channels]
    }
}

/// The running sums of [`couplings_strain`]: blocks of reciprocal vectors are added one at a time
/// (`add`) and the Jacobian of `1 / Omega` and the volume division follow in `finish`. It takes
/// any set of vectors, which is how a test sums the vectors that a finite difference of the
/// piecewise-linear shape tables does not straddle; the production list is the ball of
/// [`couplings_strain`].
pub struct CouplingsStrainSum<'a> {
    shapes: &'a Shapes,
    sites: &'a Sites<'a>,
    rows: usize,
    block: usize,
    u: Buffer<Complex64>,
    u_ion: Buffer<Complex64>,
    raw_u: Vec<Complex64>,
    raw_ion: Vec<Complex64>,
    e_sum: f64,
    e_raw: [[f64; 3]; 3],
    f: Vec<Complex64>,
    gradient: Vec<Complex64>,
    dg: Vec<Complex64>,
    _scratch: crate::Reservation,
}
impl<'a> CouplingsStrainSum<'a> {
    pub fn new(ctx: &Context, shapes: &'a Shapes, sites: &'a Sites<'a>) -> Result<Self> {
        sites.check()?;
        let rows = sites.channels.len();
        let size = rows * rows;
        let per_vector = (rows * 13 + 6) * 16 + 64;
        let block = (Transient::of(ctx).dense_bytes / per_vector).max(1);
        Ok(Self {
            shapes,
            sites,
            rows,
            block,
            u: ctx.allocate(9 * size)?,
            u_ion: ctx.allocate(9 * rows)?,
            raw_u: vec![Complex64::default(); size],
            raw_ion: vec![Complex64::default(); rows],
            e_sum: 0.,
            e_raw: [[0.; 3]; 3],
            f: vec![Complex64::default(); rows * block],
            gradient: vec![Complex64::default(); rows * 3 * block],
            dg: vec![Complex64::default(); 9 * rows * block],
            _scratch: ctx.reserve(crate::product(&[block, per_vector])?)?,
        })
    }
    /// The most vectors one `add` takes.
    pub fn block(&self) -> usize {
        self.block
    }
    /// Adds the terms of `vectors` (at most [`CouplingsStrainSum::block`], none zero).
    pub fn add(&mut self, vectors: &[[f64; 3]]) -> Result<()> {
        let (sites, rows, m) = (self.sites, self.rows, vectors.len());
        if m > self.block || vectors.iter().any(|v| *v == [0.; 3]) {
            return Err(crate::periodic::invalid(
                "a block of at most `block` nonzero wave vectors",
            ));
        }
        let sigma2 = sites.sigma * sites.sigma;
        let (f, gradient, dg) = (&mut self.f, &mut self.gradient, &mut self.dg);
        self.shapes
            .transforms(sites, vectors, &mut f[..rows * m], m)?;
        self.shapes
            .transform_gradients(sites, vectors, &mut gradient[..rows * 3 * m], m)?;
        // dg[(pq * rows + a) * m + g] = -G_p d_q g_a
        for (g, v) in vectors.iter().enumerate() {
            for a in 0..rows {
                for p in 0..3 {
                    for q in 0..3 {
                        dg[((p * 3 + q) * rows + a) * m + g] =
                            gradient[(a * 3 + q) * m + g] * -v[p];
                    }
                }
            }
        }
        let k: Vec<f64> = vectors.iter().map(|v| coulomb(*v)).collect();
        let ion: Vec<Complex64> = vectors.iter().map(|v| ion_transform(sites, *v)).collect();
        let g2: Vec<f64> = vectors
            .iter()
            .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            .collect();
        for (g, v) in vectors.iter().enumerate() {
            let energy = 0.5 * k[g] * ion[g].norm_sqr();
            self.e_sum += energy;
            for p in 0..3 {
                for q in 0..3 {
                    self.e_raw[p][q] +=
                        energy * (2. * v[p] * v[q] / g2[g] + 2. * sigma2 * v[p] * v[q]);
                }
            }
        }
        // d K / d eps_pq = 2 K G_p G_q / G^2 and d I / d eps_pq = sigma^2 G_p G_q I.
        let dk = |pq: usize, g: usize| {
            let (p, q) = (pq / 3, pq % 3);
            2. * k[g] * vectors[g][p] * vectors[g][q] / g2[g]
        };
        let dion = |pq: usize, g: usize| {
            let (p, q) = (pq / 3, pq % 3);
            ion[g] * (sigma2 * vectors[g][p] * vectors[g][q])
        };
        let size = rows * rows;
        for a in 0..rows {
            let fa = &f[a * m..(a + 1) * m];
            for b in a..rows {
                let fb = &f[b * m..(b + 1) * m];
                self.raw_u[a * rows + b] += (0..m)
                    .map(|g| fa[g].conj() * k[g] * fb[g])
                    .sum::<Complex64>();
                for pq in 0..9 {
                    let da = &dg[(pq * rows + a) * m..(pq * rows + a + 1) * m];
                    let db = &dg[(pq * rows + b) * m..(pq * rows + b + 1) * m];
                    let sum: Complex64 = (0..m)
                        .map(|g| {
                            da[g].conj() * k[g] * fb[g]
                                + fa[g].conj() * dk(pq, g) * fb[g]
                                + fa[g].conj() * k[g] * db[g]
                        })
                        .sum();
                    self.u.as_mut_slice()[pq * size + a * rows + b] += sum;
                }
            }
            self.raw_ion[a] += (0..m)
                .map(|g| fa[g].conj() * k[g] * ion[g])
                .sum::<Complex64>();
            for pq in 0..9 {
                let da = &dg[(pq * rows + a) * m..(pq * rows + a + 1) * m];
                let sum: Complex64 = (0..m)
                    .map(|g| {
                        da[g].conj() * k[g] * ion[g]
                            + fa[g].conj() * dk(pq, g) * ion[g]
                            + fa[g].conj() * k[g] * dion(pq, g)
                    })
                    .sum();
                self.u_ion.as_mut_slice()[pq * rows + a] += sum;
            }
        }
        Ok(())
    }
    /// The Jacobian `-delta_pq` of `1 / Omega`, the division by the volume and the Hermitian
    /// lower triangle.
    pub fn finish(mut self) -> CouplingsStrain {
        let (rows, volume) = (self.rows, self.sites.volume());
        let size = rows * rows;
        for pq in 0..9 {
            let jacobian = if pq % 4 == 0 { -1. } else { 0. };
            for a in 0..rows {
                for b in a..rows {
                    let value = (self.u.as_slice()[pq * size + a * rows + b]
                        + self.raw_u[a * rows + b] * jacobian)
                        / volume;
                    self.u.as_mut_slice()[pq * size + a * rows + b] = value;
                    self.u.as_mut_slice()[pq * size + b * rows + a] = value.conj();
                }
                let value =
                    (self.u_ion.as_slice()[pq * rows + a] + self.raw_ion[a] * jacobian) / volume;
                self.u_ion.as_mut_slice()[pq * rows + a] = value;
            }
        }
        let (e_sum, e_raw) = (self.e_sum, self.e_raw);
        CouplingsStrain {
            u: self.u,
            u_ion: self.u_ion,
            e_ion: std::array::from_fn(|p| {
                std::array::from_fn(|q| (e_raw[p][q] - if p == q { e_sum } else { 0. }) / volume)
            }),
            channels: rows,
        }
    }
}

/// The strain derivative of `couplings(ctx, shapes, sites, cutoff)` over its own `G` list:
///
/// * `dU[a, b] = -delta_pq (U - tail)[a, b] + (1/Omega) sum_G [ conj(dg_a) K g_b + conj(g_a) dK g_b
///   + conj(g_a) K dg_b ]` with `dg_a = -G_p d_q g_a` ([`Shapes::transform_gradients`], the
///   atom phases fixed) and `dK = 2 K G_p G_q / G^2`;
/// * `dU_ion[a]` the same against the ions, whose envelope gives `dI = sigma^2 G_p G_q I`;
/// * `dE_ion = -delta_pq E_ion + (1/2 Omega) sum_G K |I|^2 (2 G_p G_q / G^2 + 2 sigma^2 G_p G_q)`.
///
/// The erfc pair sum of the point ions (`ions::ion_constants`) is not here: its strain
/// derivative is [`super::gradients::ion_constants_strain`].
pub fn couplings_strain(
    ctx: &Context,
    shapes: &Shapes,
    sites: &Sites,
    cutoff: f64,
) -> Result<CouplingsStrain> {
    let mut sum = CouplingsStrainSum::new(ctx, shapes, sites)?;
    let block = sum.block();
    visit_lattice(&sites.reciprocal()?, cutoff, true, block, &mut |vectors| {
        sum.add(vectors)
    })?;
    Ok(sum.finish())
}
