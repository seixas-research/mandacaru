//! The pair-density pass of the exchange strain: one sweep over (spin, target `k`, source `q`,
//! occupied state) that accumulates H-1..H-4.
//!
//! Per state the RAW pair rows `F_i(G) = FFT[conj(u_i(k)) psi_n(q)](G) + sum_a Q^a_i T_a(G)`
//! (unit kernel, the assembly of [`weighted_pairs`] the energy itself runs) and `V' = D F` give
//! `E = scale sum_G kw(G) sum_i Re F_i conj(V'_i)`, with `kw` the square of the kernel weight and
//! `scale = -(a/2) mult w_k`. A tangent `dF` of the rows (the same assembly with tangent inputs)
//! gives `dE = scale 2 Re sum_G kw sum_i dF_i conj(V'_i)`, and the weight itself
//! `kw = (w_q dv / N) v(G)` gives
//!
//! * H-1, `delta_ab E` (`dv` strains; `N` does not);
//! * H-2, `scale sum_G (w_q dv / N) dv_ab(G) rho_D(G)` (the kernel, with the head);
//! * H-3, the shapes: `-delta_ab` of the compensation rows (the `1 / dv` of `T_a`) and the
//!   form-factor part `-Q_a d_b f_a`;
//! * H-4, the grid pair, BOTH slots explicitly: the bra with `dU(k)`, the ket with
//!   `dpsi_n = sum_mu c_mu dU_mu(q)`.
//!
//! H-5 and H-6 (the projections) are the response `Z` contracted with `dC_ab`: [`super::chain`].
//!
//! A spinor crystal (AS-10) runs the same algebra on `rows = 2m` rows `(sigma, mu)` with
//! `D_spin = D` (`2m x 2m`, occupation 1 per state, `mult = 1`), as the force pass does: the
//! spin sum of the pair density is inside `weighted_pairs` (a row reads the component `sigma` of
//! the state), the tangents of the basis are the scalar `m x N` ones (the strain acts on space
//! only, `T (x) I_2`; the bra row `(sigma, mu)` reads the tangent of the function `mu`), the moved
//! ket is one function per component, and the shapes and kernel are spin independent.
use super::{Sources, Tangents, Terms};
use crate::{
    math::{fft::Fft3, storage::MatrixSource},
    periodic::{
        invalid,
        screened_augmentation::Augmentation,
        screened_exchange::{
            kernel_weights, occupied_states, periodic_part, weighted_pairs, ExxDivTreatment,
            PairInputs, Part, Screening,
        },
        setup::strain_grid::VOIGT,
        Crystal,
    },
    product,
    quantum::C64,
    Buffer, Result,
};

/// The Voigt row (xx, yy, zz, yz, xz, xy) of the symmetric pair `(a, b)`.
fn row(a: usize, b: usize) -> usize {
    VOIGT
        .iter()
        .position(|&(p, q)| (p, q) == (a, b) || (q, p) == (a, b))
        .expect("an axis pair")
}

/// `sum_i sum_G kw(G) Re rows_i(G) conj(v_i(G))`.
fn dot(rows: &[C64], v: &[C64], kw: &[f64]) -> f64 {
    let n = kw.len();
    rows.chunks(n)
        .zip(v.chunks(n))
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .zip(kw)
                .map(|((x, y), w)| w * (x * y.conj()).re)
                .sum::<f64>()
        })
        .sum()
}

const PAIRS: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

pub(super) fn run<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    aug: &Augmentation,
    tangents: &Tangents,
    sources: &dyn Sources,
) -> Result<Terms> {
    let (nk, n, m) = (c.points.len(), c.grid.index.points(), c.m);
    // Rows of the pair buffers and size of the density matrices: `(sigma, mu)` for a spinor.
    let components = if c.spinor { 2 } else { 1 };
    let rows = components * m;
    let dv = c.grid.index.dv();
    // The periodic parts of the strain tangents of every k-point and every `(a, b)`: nine
    // `m x N` arrays per point, charged to the context.
    let mut tangent: Vec<Vec<Buffer<C64>>> = Vec::with_capacity(nk);
    for (k, point) in c.points.iter().enumerate() {
        let mut per_pair = Vec::with_capacity(9);
        for &(a, b) in &PAIRS {
            let samples = tangents.samples(k, a, b)?;
            if samples.as_slice().len() != m * n {
                return Err(invalid("a strain tangent must hold m x N samples"));
            }
            let mut u = c.context.allocate::<C64>(product(&[m, n])?)?;
            periodic_part(c, point.k, samples.as_slice(), u.as_mut_slice());
            per_pair.push(u);
        }
        tangent.push(per_pair);
    }
    // A restricted scalar crystal carries half of the density per spin and counts it twice; a
    // spinor (one channel, occupation 1 per state) and an unrestricted crystal do neither.
    let (spin_scale, mult) = if c.spins == 1 && !c.spinor {
        (0.5, 2.0)
    } else {
        (1.0, 1.0)
    };
    let mut out = Terms::default();
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    let mut pair = c.context.allocate::<C64>(n)?;
    let mut unit = c.context.allocate::<f64>(n)?;
    unit.as_mut_slice().fill(1.);
    let mut raw = c.context.allocate::<C64>(rows * n)?;
    let mut v = c.context.allocate::<C64>(rows * n)?;
    let mut work = c.context.allocate::<C64>(rows * n)?;
    let mut density = vec![0.; n];
    for s in 0..c.spins {
        let occupied = occupied_states(c, matrices, s, spin_scale, Some((aug, Part::Both)))?;
        for (k, point) in c.points.iter().enumerate() {
            let d = matrices[s * nk + k].matrix(&c.context)?;
            let d = &*d * C64::new(spin_scale, 0.);
            let mut u = c.context.allocate::<C64>(m * n)?;
            periodic_part(c, point.k, &point.psi.read(&c.context)?, u.as_mut_slice());
            let scale = -0.5 * screening.fraction * mult * point.weight;
            for (q, source) in occupied.iter().enumerate() {
                if source.wavefunctions.is_empty() {
                    continue;
                }
                let k_diff = std::array::from_fn(|a| c.points[q].k[a] - point.k[a]);
                let root = kernel_weights(c, k_diff, source.weight, screening.omega, exxdiv)?;
                let kw: Vec<f64> = root.as_slice().iter().map(|x| x * x).collect();
                let shapes = aug.pair_shapes(c, k_diff)?;
                let dkernel = sources.kernel(c, k_diff, source.weight, screening.omega, exxdiv)?;
                if dkernel.as_slice().len() != 6 * n {
                    return Err(invalid("the kernel strain must hold six rows of N values"));
                }
                let strain = sources.shapes(aug, c, k_diff)?;
                let forms = PAIRS
                    .iter()
                    .map(|&(a, b)| strain.form(&c.context, a, b))
                    .collect::<Result<Vec<_>>>()?;
                let prefactor = source.weight * dv / n as f64;
                for (state, psi) in source.wavefunctions.iter().enumerate() {
                    let moments = aug.moments(c, point, &source.projected[state]);
                    let both = PairInputs {
                        u_k: u.as_slice(),
                        psi,
                        moments: Some(&moments),
                        shapes: Some(shapes.as_slice()),
                        kernel: unit.as_slice(),
                        part: Part::Both,
                    };
                    weighted_pairs(c, &mut fft, &mut pair, &both, raw.as_mut_slice())?;
                    v.as_mut_slice().fill(C64::default());
                    for i in 0..rows {
                        for j in 0..rows {
                            let dij = d[(i, j)];
                            for at in 0..n {
                                v.as_mut_slice()[i * n + at] += dij * raw.as_slice()[j * n + at];
                            }
                        }
                    }
                    density.fill(0.);
                    for i in 0..rows {
                        for at in 0..n {
                            density[at] +=
                                (raw.as_slice()[i * n + at] * v.as_slice()[i * n + at].conj()).re;
                        }
                    }
                    let energy: f64 =
                        scale * kw.iter().zip(&density).map(|(w, r)| w * r).sum::<f64>();
                    out.energy += energy;
                    for a in 0..3 {
                        out.jacobian[a][a] += energy;
                    }
                    for &(a, b) in &PAIRS {
                        let line = &dkernel.as_slice()[row(a, b) * n..(row(a, b) + 1) * n];
                        out.kernel[a][b] += scale
                            * prefactor
                            * line.iter().zip(&density).map(|(x, r)| x * r).sum::<f64>();
                    }
                    // H-3: the compensation rows alone, the `1 / dv` of `T_a` ...
                    let comp = PairInputs {
                        u_k: u.as_slice(),
                        psi,
                        moments: Some(&moments),
                        shapes: Some(shapes.as_slice()),
                        kernel: unit.as_slice(),
                        part: Part::Compensation,
                    };
                    weighted_pairs(c, &mut fft, &mut pair, &comp, work.as_mut_slice())?;
                    let volume = 2. * scale * dot(work.as_slice(), v.as_slice(), &kw);
                    for a in 0..3 {
                        out.shape_volume[a][a] -= volume;
                    }
                    // ... and the form factors.
                    for (&(a, b), form) in PAIRS.iter().zip(&forms) {
                        let input = PairInputs {
                            u_k: u.as_slice(),
                            psi,
                            moments: Some(&moments),
                            shapes: Some(form.as_slice()),
                            kernel: unit.as_slice(),
                            part: Part::Compensation,
                        };
                        weighted_pairs(c, &mut fft, &mut pair, &input, work.as_mut_slice())?;
                        out.shape_form[a][b] +=
                            2. * scale * dot(work.as_slice(), v.as_slice(), &kw);
                    }
                    // H-4: the bra slot (the target's functions) and the ket slot (the state).
                    for (index, &(a, b)) in PAIRS.iter().enumerate() {
                        let bra = PairInputs {
                            u_k: tangent[k][index].as_slice(),
                            psi,
                            moments: None,
                            shapes: None,
                            kernel: unit.as_slice(),
                            part: Part::Pseudo,
                        };
                        weighted_pairs(c, &mut fft, &mut pair, &bra, work.as_mut_slice())?;
                        out.bra[a][b] += 2. * scale * dot(work.as_slice(), v.as_slice(), &kw);
                        // The moved state: one function per spin component.
                        let mut moved = Vec::with_capacity(components);
                        for component in 0..components {
                            let mut buffer = c.context.allocate::<C64>(n)?;
                            for mu in 0..m {
                                let coefficient = source.coefficients[state][component * m + mu];
                                let slice = &tangent[q][index].as_slice()[mu * n..(mu + 1) * n];
                                for (x, y) in buffer.as_mut_slice().iter_mut().zip(slice) {
                                    *x += coefficient * y;
                                }
                            }
                            moved.push(buffer);
                        }
                        let ket = PairInputs {
                            u_k: u.as_slice(),
                            psi: &moved,
                            moments: None,
                            shapes: None,
                            kernel: unit.as_slice(),
                            part: Part::Pseudo,
                        };
                        weighted_pairs(c, &mut fft, &mut pair, &ket, work.as_mut_slice())?;
                        out.ket[a][b] += 2. * scale * dot(work.as_slice(), v.as_slice(), &kw);
                    }
                }
            }
        }
    }
    Ok(out)
}
