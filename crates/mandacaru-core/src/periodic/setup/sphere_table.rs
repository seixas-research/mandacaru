//! The image-resolved short-range sphere of an atom: `V^sr_A = sum_x w v conj(chi_mu(x)) chi_nu(x)`
//! and its center and strain derivatives, for EVERY k-point, from one set of quadrature sweeps.
//!
//! A Bloch sum is `chi_{mu k} = sum_T e^{ik.T} phi_{mu T}` over the lattice images `T` of the
//! function `phi_mu`, so every quantity of the sphere is a sum of pair integrals of the images,
//!
//! ```text
//! P^a[(mu T), (nu T')] = sum_x w_x v_x conj(d_a phi_{mu T}(x)) phi_{nu T'}(x),
//! ```
//!
//! weighted by the phases `conj(e^{ik.T}) e^{ik.T'}`. The pair integrals do not depend on `k`: the
//! table holds the three of them, once, as products of two `N x n` arrays (`N` image functions that
//! reach the sphere, `n` points of a chunk), and a k-point only contracts them with its phases:
//!
//! * the row-center derivative `G^a(k)[mu, nu] = -<d_a chi_mu | V | chi_nu>` ([`SphereTable::gradient`]);
//! * the strain derivative with the sphere rigid about its atom, `d chi_mu(x) / d eps_ab =
//!   -Delta_b sum_T e^{ik.T} (d_a phi_{mu T})(x)` with `Delta = c_mu + T - R_A` constant per image,
//!   contracted with a Hermitian matrix ([`SphereTable::strain`]).
//!
//! The work on the points does not grow with the number of k-points, nor with the number of strain
//! directions.
use super::{
    basis_gradient::ShortRange,
    images::Images,
    pulay::Matrix,
    shells::Shells,
    spheres::{Sphere, CHUNK_POINTS},
};
use crate::{
    kernels::{self, ShellImages},
    math::gemm::{multiply, MatrixMut, MatrixRef},
    paw::local_split::short_range_potential,
    periodic::invalid,
    product, Buffer, Context, Result,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use std::sync::{Mutex, OnceLock};

/// Bytes of the four `N x n` arrays (value and three derivatives) a chunk of points may hold.
const CHUNK_BYTES: usize = 256 << 20;

/// One atom's short-range sphere, image resolved.
pub struct SphereTable {
    /// The function `mu` of each image function.
    function: Vec<usize>,
    shift: Vec<[f64; 3]>,
    /// `Delta = c_mu + T - R_A` of each image function.
    lever: Vec<[f64; 3]>,
    /// `P^a`, `N x N` row-major.
    pairs: [Buffer<Complex64>; 3],
}

impl SphereTable {
    pub fn new(
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        sphere: &ShortRange<'_>,
    ) -> Result<Self> {
        let rule =
            Sphere::short_range(ctx, lattice, sphere.center, sphere.potential, sphere.sigma)?;
        let all = 0..rule.len();
        let radii = rule.radii(all.clone());
        let sampled: Vec<f64> = radii.iter().map(|r| sphere.potential.value(*r)).collect();
        let potential =
            short_range_potential(&sampled, sphere.potential.charge, sphere.sigma, radii)?;
        let weights: Vec<f64> = rule
            .weights(all)
            .iter()
            .zip(&potential)
            .map(|(w, v)| w * v)
            .collect();
        let region = rule.region();
        let reached = shells.reaching(ctx, lattice, &region)?;
        // The image functions: every component of every shell at every image that reaches.
        struct Block {
            base: usize,
            images: Images,
        }
        let mut function = Vec::new();
        let mut shift = Vec::new();
        let mut lever = Vec::new();
        let mut blocks = Vec::new();
        for shell in reached.shells.shells() {
            let images =
                Images::around(ctx, lattice, &region, shell.center, shell.radial.support())?;
            let base = function.len();
            for image in images.vectors() {
                for &local in &shell.rows {
                    function.push(reached.rows[local]);
                    shift.push(*image);
                    lever.push(std::array::from_fn(|b| {
                        shell.center[b] + image[b] - sphere.center[b]
                    }));
                }
            }
            blocks.push(Block { base, images });
        }
        let count = function.len();
        let zero = |ctx: &Context| ctx.allocate::<Complex64>(product(&[count, count])?);
        let mut pairs = [zero(ctx)?, zero(ctx)?, zero(ctx)?];
        if count == 0 {
            return Ok(Self {
                function,
                shift,
                lever,
                pairs,
            });
        }
        // Real and imaginary planes of `P^a`, column-major, accumulated by the real products.
        let planes = || -> Result<[Buffer<f64>; 2]> {
            Ok([
                ctx.allocate::<f64>(product(&[count, count])?)?,
                ctx.allocate::<f64>(product(&[count, count])?)?,
            ])
        };
        let mut sums = [planes()?, planes()?, planes()?];
        let chunk = (CHUNK_BYTES / (4 * count * 16)).clamp(64, CHUNK_POINTS);
        for range in rule.chunks(chunk) {
            let points = rule.points(range.clone());
            let n = range.len();
            let w = &weights[range];
            // Value and derivative samples of every image function, `(function, point)`.
            let mut phi: Vec<Buffer<Complex64>> = (0..4)
                .map(|_| ctx.allocate::<Complex64>(product(&[count, n])?))
                .collect::<Result<_>>()?;
            for (shell, block) in reached.shells.shells().iter().zip(&blocks) {
                let spline = shell.radial.spline(shell.l)?;
                let components = shell.ms.len();
                let local: Vec<i64> = (0..components as i64).collect();
                for (r, image) in block.images.vectors().iter().enumerate() {
                    for (slot, buffer) in phi.iter_mut().enumerate() {
                        let start = (block.base + r * components) * n;
                        kernels::bloch_shell(
                            ctx,
                            &ShellImages {
                                xyz: points,
                                center: shell.center,
                                images: std::slice::from_ref(image),
                                phases: &[Complex64::new(1., 0.)],
                                support: shell.radial.support(),
                                spline: &spline,
                                ms: &shell.ms,
                                derivative: slot as i32 - 1,
                            },
                            &local,
                            components,
                            &mut buffer.as_mut_slice()[start..start + components * n],
                        )?;
                    }
                }
            }
            // Only the image functions that are nonzero somewhere on this chunk take part: the
            // chunk is a slab of the sphere and a function reaches a fraction of it, so the
            // products run on the `active` subset (`a x a` instead of `N x N`).
            let active: Vec<usize> = (0..count)
                .filter(|&i| {
                    phi.iter().any(|buffer| {
                        buffer.as_slice()[i * n..(i + 1) * n]
                            .iter()
                            .any(|v| v.re != 0. || v.im != 0.)
                    })
                })
                .collect();
            let live = active.len();
            if live == 0 {
                continue;
            }
            // The weighted ket `w phi` as two real `n x a` planes (column `c` is image function
            // `active[c]`).
            let mut ket = [
                ctx.allocate::<f64>(product(&[n, live])?)?,
                ctx.allocate::<f64>(product(&[n, live])?)?,
            ];
            for (c, &j) in active.iter().enumerate() {
                for (x, value) in phi[0].as_slice()[j * n..(j + 1) * n].iter().enumerate() {
                    ket[0].as_mut_slice()[c * n + x] = value.re * w[x];
                    ket[1].as_mut_slice()[c * n + x] = value.im * w[x];
                }
            }
            // The conjugate transpose of each derivative as two real `a x n` planes.
            let mut bra = [
                ctx.allocate::<f64>(product(&[live, n])?)?,
                ctx.allocate::<f64>(product(&[live, n])?)?,
            ];
            let mut product_planes = [
                ctx.allocate::<f64>(product(&[live, live])?)?,
                ctx.allocate::<f64>(product(&[live, live])?)?,
            ];
            for axis in 0..3 {
                for (c, &i) in active.iter().enumerate() {
                    for (x, value) in phi[axis + 1].as_slice()[i * n..(i + 1) * n]
                        .iter()
                        .enumerate()
                    {
                        bra[0].as_mut_slice()[c + x * live] = value.re;
                        bra[1].as_mut_slice()[c + x * live] = -value.im;
                    }
                }
                for plane in product_planes.iter_mut() {
                    plane.as_mut_slice().fill(0.);
                }
                // (Ar + i Ai)(Br + i Bi) with A = bra^+ and B = ket.
                for (plane, left, right, alpha) in
                    [(0, 0, 0, 1.), (0, 1, 1, -1.), (1, 0, 1, 1.), (1, 1, 0, 1.)]
                {
                    multiply(
                        MatrixRef {
                            values: bra[left].as_slice(),
                            rows: live,
                            columns: n,
                            leading: live,
                        },
                        MatrixRef {
                            values: ket[right].as_slice(),
                            rows: n,
                            columns: live,
                            leading: n,
                        },
                        MatrixMut {
                            values: product_planes[plane].as_mut_slice(),
                            rows: live,
                            columns: live,
                            leading: live,
                        },
                        alpha,
                        1.,
                    )?;
                }
                for (plane, from) in sums[axis].iter_mut().zip(&product_planes) {
                    for (cj, &j) in active.iter().enumerate() {
                        for (ci, &i) in active.iter().enumerate() {
                            plane.as_mut_slice()[i + j * count] += from.as_slice()[ci + cj * live];
                        }
                    }
                }
            }
        }
        for (pair, [real, imaginary]) in pairs.iter_mut().zip(&sums) {
            for i in 0..count {
                for j in 0..count {
                    pair.as_mut_slice()[i * count + j] = Complex64::new(
                        real.as_slice()[i + j * count],
                        imaginary.as_slice()[i + j * count],
                    );
                }
            }
        }
        Ok(Self {
            function,
            shift,
            lever,
            pairs,
        })
    }

    fn phases(&self, k: [f64; 3]) -> Vec<Complex64> {
        self.shift
            .iter()
            .map(|t| {
                let angle = k[0] * t[0] + k[1] * t[1] + k[2] * t[2];
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    /// `G^a(k)[mu, nu] = -sum_{T T'} e^{-ik.T} e^{ik.T'} P^a[(mu T), (nu T')]`, `m x m` per axis.
    pub fn gradient(&self, k: [f64; 3], functions: usize) -> [Matrix; 3] {
        let n = self.function.len();
        let s = self.phases(k);
        std::array::from_fn(|axis| {
            let mut out = Matrix::zeros(functions, functions);
            let pairs = self.pairs[axis].as_slice();
            for i in 0..n {
                let left = s[i].conj();
                for j in 0..n {
                    out[(self.function[i], self.function[j])] -= left * s[j] * pairs[i * n + j];
                }
            }
            out
        })
    }

    /// `Re tr[H dV_E]` of this sphere for each Hermitian matrix `H` of `matrices` and each
    /// direction `E`: `dV_E = T + T^+`, `T[mu, nu] = sum_x w v conj(d chi_E,mu) chi_nu`, so
    /// `Re tr[H dV_E] = 2 Re sum_{i j} c_E(i) e^{-ik.T_i} P[i, j] e^{ik.T_j} H[nu_j, mu_i]` with
    /// `c_E(i)` the coefficient of the derivative of image function `i` in `d chi_E`.
    pub fn strain(
        &self,
        k: [f64; 3],
        directions: &[[[f64; 3]; 3]],
        matrices: &[Matrix],
    ) -> Vec<Vec<f64>> {
        let n = self.function.len();
        let s = self.phases(k);
        let mut out = vec![vec![0.; directions.len()]; matrices.len()];
        if n == 0 {
            return out;
        }
        // `Y_a[i, mu'] = sum_j P^a[i, j] e^{ik.T_j} [nu_j == nu] H[nu, mu']`: the pair sums that
        // do not depend on the direction, per matrix and axis.
        for (m, h) in matrices.iter().enumerate() {
            for axis in 0..3 {
                let used = directions.iter().any(|e| e[axis].iter().any(|x| *x != 0.));
                if !used {
                    continue;
                }
                let pairs = self.pairs[axis].as_slice();
                // `x[i] = sum_j P[i, j] s_j H[nu_j, mu_i]`.
                let x: Vec<Complex64> = (0..n)
                    .map(|i| {
                        let column = self.function[i];
                        (0..n)
                            .map(|j| pairs[i * n + j] * s[j] * h[(self.function[j], column)])
                            .sum()
                    })
                    .collect();
                for (d, e) in directions.iter().enumerate() {
                    let sum: Complex64 = (0..n)
                        .map(|i| {
                            let c = -(0..3).map(|b| e[axis][b] * self.lever[i][b]).sum::<f64>();
                            s[i].conj() * x[i] * c
                        })
                        .sum();
                    out[m][d] += 2. * sum.re;
                }
            }
        }
        out
    }
}

/// The tables of every sphere of a setup, formed on first use and shared by every provider that
/// holds the cache.
pub struct SphereCache {
    tables: OnceLock<Vec<SphereTable>>,
    forming: Mutex<()>,
}

impl Default for SphereCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SphereCache {
    pub fn new() -> Self {
        Self {
            tables: OnceLock::new(),
            forming: Mutex::new(()),
        }
    }

    /// One table per sphere of `spheres`, in order.
    pub fn tables(
        &self,
        ctx: &Context,
        shells: &Shells,
        lattice: &Matrix3<f64>,
        spheres: &[ShortRange<'_>],
    ) -> Result<&[SphereTable]> {
        if let Some(tables) = self.tables.get() {
            return Ok(tables);
        }
        let _forming = self.forming.lock().unwrap_or_else(|e| e.into_inner());
        if self.tables.get().is_none() {
            let tables = spheres
                .iter()
                .map(|sphere| SphereTable::new(ctx, shells, lattice, sphere))
                .collect::<Result<Vec<_>>>()?;
            let _ = self.tables.set(tables);
        }
        self.tables
            .get()
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("the sphere tables were not formed"))
    }
}
