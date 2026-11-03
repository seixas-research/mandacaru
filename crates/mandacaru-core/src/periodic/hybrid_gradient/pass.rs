//! The one pass over (spin, target k, source q, occupied state) that accumulates every explicit
//! term. Per state the weighted pair rows `W` (`rows x N`, exactly the rows `target_matrix`
//! builds) and `V = D W` give `E = scale sum_i W_i . conj(V_i)`, and a tangent `W'` of the rows
//! gives `dE = scale 2 Re sum_i W'_i . conj(V_i)`, with `scale = -(a/2) mult w_k`.
//!
//! The compensation tangents are carried by `Phi[i,a] = sum_at omega T_a conj(V_i)`: with the
//! shape slopes `dT` it gives the phase term, with `T` the response `Z` of the projections.
//!
//! A spinor crystal (SH-F1) has `rows = 2m`, row `i = sigma m + mu`, `D_spin = D` (occupation 1,
//! `mult = 1`) and the same algebra. What the second component changes: a bra row `(sigma, mu)`
//! belongs to the owner of the function `mu` (`owner[i % m]`); the state moved by atom `X` is one
//! function per component; the projections `C(k)` act identically on both components, so the
//! response `Z` stays `p x m` and sums the two components (`dC` is spin diagonal).
use super::{Component, Matrix, Motion};
use crate::{
    math::{fft::Fft3, storage::MatrixSource},
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{
            kernel_weights, occupied_states, periodic_part, weighted_pairs, ExxDivTreatment,
            OccupiedStates, PairInputs, Part, Screening,
        },
        Crystal, KPoint,
    },
    quantum::C64,
    Buffer, Result,
};

pub(super) struct Pass {
    pub energy: f64,
    /// One entry per spin component of the rows (one, or two for a spinor): the terms that the
    /// rows of that component contribute. Their sum is the term.
    pub components: Vec<Component>,
}

/// What is fixed while one target point is swept against the sources.
struct Target<'a> {
    point: &'a KPoint,
    /// `D_spin(k)`.
    d: Matrix,
    scale: f64,
    /// `z` index of this point.
    at: usize,
    u: Buffer<C64>,
    /// Periodic parts of the center derivatives of its basis.
    du: Option<[Buffer<C64>; 3]>,
}

struct Work<'a> {
    c: &'a Crystal,
    aug: &'a Augmentation<'a>,
    motion: Option<&'a dyn Motion>,
    fft: Fft3,
    pair: Buffer<C64>,
    out: Pass,
    channel_owner: Vec<usize>,
}

/// Spin components of a state's wave function: 2 for a spinor, 1 otherwise.
fn components(c: &Crystal) -> usize {
    if c.spinor {
        2
    } else {
        1
    }
}

fn slopes(c: &Crystal, motion: &dyn Motion, k: usize) -> Result<[Buffer<C64>; 3]> {
    let derivative = motion.center_derivative(c, k)?;
    let n = c.grid.index.points();
    let mut out = [
        c.context.allocate::<C64>(c.m * n)?,
        c.context.allocate::<C64>(c.m * n)?,
        c.context.allocate::<C64>(c.m * n)?,
    ];
    for (to, from) in out.iter_mut().zip(&derivative) {
        if from.as_slice().len() != c.m * n {
            return Err(crate::periodic::invalid(
                "a center derivative must hold m x N samples",
            ));
        }
        periodic_part(c, c.points[k].k, from.as_slice(), to.as_mut_slice());
    }
    Ok(out)
}

/// `Phi[i, a] = sum_at kernel[at] shapes[a, at] conj(v[i, at])`, `m x channels`.
fn response_of(v: &[C64], kernel: &[f64], shapes: &[C64], m: usize, channels: usize) -> Matrix {
    let n = kernel.len();
    Matrix::from_fn(m, channels, |i, a| {
        let (v, s) = (&v[i * n..(i + 1) * n], &shapes[a * n..(a + 1) * n]);
        (0..n).map(|at| s[at] * v[at].conj() * kernel[at]).sum()
    })
}

/// `sum_{i, at} w[i, at] conj(v[i, at])` per row `i`.
fn rows_dot<'a>(w: &'a [C64], v: &'a [C64], n: usize) -> impl Iterator<Item = C64> + 'a {
    w.chunks(n)
        .zip(v.chunks(n))
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y.conj()).sum())
}

pub(super) fn run<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    aug: &Augmentation,
    motion: Option<&dyn Motion>,
) -> Result<Pass> {
    let (nk, n) = (c.points.len(), c.grid.index.points());
    let atoms = aug.sites().atoms.len();
    // The matrices this pass keeps outside the `Context`'s buffers: `Z` per (spin, k), the scaled
    // density matrix of the target and the per-state response blocks (`rows x channels`, `p x
    // channels`, a few of each).
    let rows = components(c) * c.m;
    let channels = aug.sites().channels.len();
    let _held = c.context.reserve(crate::product(&[
        16,
        components(c) * c.spins * nk * c.p * c.m
            + rows * rows
            + 6 * (rows + c.p) * channels
            + 4 * c.p,
    ])?)?;
    // A restricted scalar crystal carries half of the density per spin and counts it twice; a
    // spinor (one channel, occupation 1 per state) and an unrestricted crystal do neither.
    let (spin_scale, mult) = if c.spins == 1 && !c.spinor {
        (0.5, 2.0)
    } else {
        (1.0, 1.0)
    };
    let mut work = Work {
        c,
        aug,
        motion,
        fft: Fft3::new(&c.context, c.grid.index.shape())?,
        pair: c.context.allocate::<C64>(n)?,
        out: Pass {
            energy: 0.,
            components: (0..components(c))
                .map(|_| Component {
                    phase: vec![[0.; 3]; atoms],
                    bra: vec![[0.; 3]; atoms],
                    ket: vec![[0.; 3]; atoms],
                    compensation: vec![Matrix::zeros(c.p, c.m); c.spins * nk],
                })
                .collect(),
        },
        channel_owner: aug.sites().channels.iter().map(|ch| ch.0).collect(),
    };
    // A spinor has one channel (`spins = 1`) whose matrices are `2m x 2m`.
    for s in 0..c.spins {
        let occupied = occupied_states(c, matrices, s, spin_scale, Some((aug, Part::Both)))?;
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[s * nk + k].matrix(&c.context)?;
            let mut u = c.context.allocate::<C64>(c.m * n)?;
            periodic_part(c, point.k, &point.psi.read(&c.context)?, u.as_mut_slice());
            let target = Target {
                point,
                d: &*d * C64::new(spin_scale, 0.),
                scale: -0.5 * screening.fraction * mult * point.weight,
                at: s * nk + k,
                u,
                du: motion.map(|mo| slopes(c, mo, k)).transpose()?,
            };
            for (q, source) in occupied.iter().enumerate() {
                if !source.wavefunctions.is_empty() {
                    work.source(&target, s * nk + q, q, source, screening, exxdiv)?;
                }
            }
        }
    }
    Ok(work.out)
}

impl Work<'_> {
    fn source(
        &mut self,
        t: &Target,
        z_source: usize,
        q: usize,
        source: &OccupiedStates,
        screening: Screening,
        exxdiv: ExxDivTreatment,
    ) -> Result<()> {
        let c = self.c;
        let point = &c.points[q];
        let k_diff = std::array::from_fn(|a| point.k[a] - t.point.k[a]);
        let kernel = kernel_weights(c, k_diff, source.weight, screening.omega, exxdiv)?;
        let shapes = self.aug.pair_shapes(c, k_diff)?;
        let dshapes = [
            self.aug.pair_shape_slopes(c, k_diff, 0)?,
            self.aug.pair_shape_slopes(c, k_diff, 1)?,
            self.aug.pair_shape_slopes(c, k_diff, 2)?,
        ];
        let du_source = self.motion.map(|mo| slopes(c, mo, q)).transpose()?;
        for (state, psi) in source.wavefunctions.iter().enumerate() {
            let moments = self.aug.moments(c, t.point, &source.projected[state]);
            let n = c.grid.index.points();
            let rows = components(c) * c.m;
            debug_assert_eq!(moments.nrows(), rows);
            let mut w = c.context.allocate::<C64>(rows * n)?;
            let input = PairInputs {
                u_k: t.u.as_slice(),
                psi,
                moments: Some(&moments),
                shapes: Some(shapes.as_slice()),
                kernel: kernel.as_slice(),
                part: Part::Both,
            };
            weighted_pairs(c, &mut self.fft, &mut self.pair, &input, w.as_mut_slice())?;
            // V = D W
            let mut v = c.context.allocate::<C64>(rows * n)?;
            for i in 0..rows {
                for j in 0..rows {
                    let dij = t.d[(i, j)];
                    for at in 0..n {
                        v.as_mut_slice()[i * n + at] += dij * w.as_slice()[j * n + at];
                    }
                }
            }
            self.out.energy += t.scale
                * rows_dot(w.as_slice(), v.as_slice(), n)
                    .map(|x| x.re)
                    .sum::<f64>();
            let args = State {
                t,
                z_source,
                source,
                state,
                kernel: kernel.as_slice(),
                shapes: shapes.as_slice(),
                dshapes: &dshapes,
                v: v.as_slice(),
                psi,
            };
            self.compensation(&args, &moments)?;
            if let Some(du_source) = &du_source {
                self.functions(&args, du_source)?;
            }
        }
        Ok(())
    }

    /// The phase of the shapes and the response `Z` of the projections. The rows of `moments` and
    /// of `V` run over `(component, mu)`; `Z` sums the components.
    fn compensation(&mut self, a: &State, moments: &Matrix) -> Result<()> {
        let (c, t) = (self.c, a.t);
        let (m, p, channels) = (c.m, c.p, self.channel_owner.len());
        let rows = moments.nrows();
        let phi = response_of(a.v, a.kernel, a.shapes, rows, channels);
        for axis in 0..3 {
            let phi_axis = response_of(a.v, a.kernel, a.dshapes[axis].as_slice(), rows, channels);
            for (channel, &atom) in self.channel_owner.iter().enumerate() {
                for (component, part) in self.out.components.iter_mut().enumerate() {
                    let sum: C64 = (component * m..(component + 1) * m)
                        .map(|i| moments[(i, channel)] * phi_axis[(i, channel)])
                        .sum();
                    part.phase[atom][axis] += 2.0 * t.scale * sum.re;
                }
            }
        }
        let scale = C64::new(t.scale, 0.);
        let projected = &a.source.projected[a.state];
        let coefficients = &a.source.coefficients[a.state];
        for component in 0..rows / m {
            // The component's rows of `Phi` (`m x channels`) and its projections `a_{n, sigma}`.
            let phi = phi.rows(component * m, m).into_owned();
            // b_a = B_a a_n, as `Augmentation::moments` forms it.
            let mut b = Matrix::zeros(p, channels);
            for channel in 0..channels {
                let block = &c.blocks.as_slice()[channel * p * p..(channel + 1) * p * p];
                for j in 0..p {
                    b[(j, channel)] = (0..p)
                        .map(|i| block[j * p + i] * projected[component * p + i])
                        .sum();
                }
            }
            self.out.components[component].compensation[t.at] += (&b * phi.transpose()) * scale;
            // Gamma[p'] = sum_{a, p} (Phi^T C)[a, p] B_a[p, p']
            let weighted = phi.transpose() * &t.point.projections;
            let mut gamma = vec![C64::default(); p];
            for channel in 0..channels {
                let block = &c.blocks.as_slice()[channel * p * p..(channel + 1) * p * p];
                for (pp, g) in gamma.iter_mut().enumerate() {
                    *g += (0..p)
                        .map(|j| weighted[(channel, j)] * block[j * p + pp])
                        .sum::<C64>();
                }
            }
            // The source part of both components lands in the same `p x m` matrix.
            for nu in 0..m {
                let coefficient = coefficients[component * m + nu];
                for (pp, g) in gamma.iter().enumerate() {
                    self.out.components[component].compensation[a.z_source][(pp, nu)] +=
                        (coefficient * g).conj() * scale;
                }
            }
        }
        Ok(())
    }

    /// The basis functions of the grid pair moving with their atoms: the bra functions (rows of
    /// the target basis, `owner[i % m]` for the row `(component, mu)`) and the ket (the state's
    /// own functions, one moved function per component), each explicitly.
    fn functions(&mut self, a: &State, du_source: &[Buffer<C64>; 3]) -> Result<()> {
        let (c, t) = (self.c, a.t);
        let (m, n) = (c.m, c.grid.index.points());
        let owner = self.motion.map(|mo| mo.owner()).unwrap_or(&[]);
        let du = t.du.as_ref().unwrap();
        let coefficients = &a.source.coefficients[a.state];
        let components = components(c);
        let mut rows = c.context.allocate::<C64>(components * m * n)?;
        for axis in 0..3 {
            let bra = PairInputs {
                u_k: du[axis].as_slice(),
                psi: a.psi,
                moments: None,
                shapes: None,
                kernel: a.kernel,
                part: Part::Pseudo,
            };
            weighted_pairs(c, &mut self.fft, &mut self.pair, &bra, rows.as_mut_slice())?;
            let dots: Vec<C64> = rows_dot(rows.as_slice(), a.v, n).collect();
            for (i, dot) in dots.iter().enumerate() {
                self.out.components[i / m].bra[owner[i % m]][axis] += 2.0 * t.scale * dot.re;
            }
            for atom in 0..self.out.components[0].ket.len() {
                if !owner.contains(&atom) {
                    continue;
                }
                let mut moved = Vec::with_capacity(components);
                for component in 0..components {
                    let mut buffer = c.context.allocate::<C64>(n)?;
                    for mu in (0..m).filter(|&mu| owner[mu] == atom) {
                        let coefficient = coefficients[component * m + mu];
                        let row = &du_source[axis].as_slice()[mu * n..(mu + 1) * n];
                        for (x, y) in buffer.as_mut_slice().iter_mut().zip(row) {
                            *x += coefficient * y;
                        }
                    }
                    moved.push(buffer);
                }
                let ket = PairInputs {
                    u_k: t.u.as_slice(),
                    psi: &moved,
                    moments: None,
                    shapes: None,
                    kernel: a.kernel,
                    part: Part::Pseudo,
                };
                weighted_pairs(c, &mut self.fft, &mut self.pair, &ket, rows.as_mut_slice())?;
                // Rows of component `s` carry the moved component `s` alone.
                for (i, dot) in rows_dot(rows.as_slice(), a.v, n).enumerate() {
                    self.out.components[i / m].ket[atom][axis] += 2.0 * t.scale * dot.re;
                }
            }
        }
        Ok(())
    }
}

/// One occupied state against one target point.
struct State<'a> {
    t: &'a Target<'a>,
    z_source: usize,
    source: &'a OccupiedStates,
    state: usize,
    kernel: &'a [f64],
    shapes: &'a [C64],
    dshapes: &'a [Buffer<C64>; 3],
    v: &'a [C64],
    psi: &'a [Buffer<C64>],
}
