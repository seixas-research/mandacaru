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
use crate::periodic::stress_terms::Directions;
use crate::{
    math::{fft::Fft3, storage::MatrixSource},
    periodic::{
        input::has_partner,
        invalid,
        screened_augmentation::Augmentation,
        screened_exchange::{
            kernel_weights, occupied_states, periodic_part, weighted_pairs, ExxDivTreatment,
            OccupiedStates, PairInputs, Part, Screening,
        },
        setup::strain_grid::VOIGT,
        star::{self, Reduction},
        Crystal,
    },
    product,
    quantum::C64,
    Buffer, Level, Result,
};
use nalgebra::DMatrix;

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

/// The terms of one share or of the sum, one number per strain direction.
#[derive(Clone, Default)]
struct Shares {
    energy: f64,
    jacobian: Vec<f64>,
    kernel: Vec<f64>,
    shape_volume: Vec<f64>,
    shape_form: Vec<f64>,
    bra: Vec<f64>,
    ket: Vec<f64>,
}

impl Shares {
    fn new(directions: usize) -> Self {
        let zero = || vec![0.; directions];
        Self {
            energy: 0.,
            jacobian: zero(),
            kernel: zero(),
            shape_volume: zero(),
            shape_form: zero(),
            bra: zero(),
            ket: zero(),
        }
    }
}

fn trace(e: &[[f64; 3]; 3]) -> f64 {
    e[0][0] + e[1][1] + e[2][2]
}

pub(super) fn run<M: MatrixSource + Sync>(
    c: &Crystal,
    matrices: &[M],
    screening: Screening,
    exxdiv: ExxDivTreatment,
    aug: &Augmentation,
    tangents: &Tangents,
    sources: &dyn Sources,
    directions: &Directions,
) -> Result<Terms> {
    let dirs = directions.as_slice();
    let (nk, n, m) = (c.points.len(), c.grid.index.points(), c.m);
    // Rows of the pair buffers and size of the density matrices: `(sigma, mu)` for a spinor.
    let components = if c.spinor { 2 } else { 1 };
    let rows = components * m;
    let dv = c.grid.index.dv();
    // The periodic parts of the strain tangents of every k-point and every direction: one
    // `m x N` array per direction and point, charged to the context.
    let mut tangent: Vec<Vec<Buffer<C64>>> = Vec::with_capacity(nk);
    for (k, point) in c.points.iter().enumerate() {
        let mut per_pair = Vec::with_capacity(dirs.len());
        for e in dirs {
            let samples = tangents.samples_along(k, e)?;
            if samples.as_slice().len() != m * n {
                return Err(invalid("a strain tangent must hold m x N samples"));
            }
            let mut u = c.context.allocate::<C64>(product(&[m, n])?)?;
            periodic_part(c, point.k, samples.as_slice(), u.as_mut_slice());
            per_pair.push(u);
        }
        tangent.push(per_pair);
    }
    // On a time-reversal-halved mesh the partner at `-k` of a listed point moves with the
    // conjugates of its tangents.
    // On a mesh reduced by space-group operations the sources are the images of the listed points
    // (`star`): the ket of an image is the listed point's moved state rotated with it, along the
    // same strain direction (the directions of a reduced mesh are invariant under its operations, or
    // averaged over them afterwards), so no partner tangents are needed.
    let reduction = star::reduced(c).then(|| Reduction::new(c)).transpose()?;
    let reversed: Vec<Option<Vec<Buffer<C64>>>> = (0..nk)
        .map(|k| {
            if reduction.is_some() || has_partner(&c.points, k) {
                return Ok(None);
            }
            tangent[k]
                .iter()
                .map(|u| {
                    let mut out = c.context.allocate::<C64>(u.as_slice().len())?;
                    for (o, v) in out.as_mut_slice().iter_mut().zip(u.as_slice()) {
                        *o = v.conj();
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<_>>>()
                .map(Some)
        })
        .collect::<Result<_>>()?;
    // A restricted scalar crystal carries half of the density per spin and counts it twice; a
    // spinor (one channel, occupation 1 per state) and an unrestricted crystal do neither.
    let (spin_scale, mult) = if c.spins == 1 && !c.spinor {
        (0.5, 2.0)
    } else {
        (1.0, 1.0)
    };
    // The target points of a spin channel run on the `kpts` lanes, the states of each source on
    // the `bands` lanes of its target. Every state adds exactly one value to each entry of the
    // terms it touches, so its share is a delta of the terms, and the deltas are added in the
    // order of the serial sweep (spin, target, source, state): the same sums on every layout.
    let ctx = &c.context;
    let k_lanes = ctx.lanes(Level::Kpts, nk);
    let b_lanes = ctx.lanes(Level::Bands, usize::MAX);
    let lane_bytes = product(&[3 * rows + 1, n, 16])?
        .checked_add(product(&[
            *c.grid.index.shape().iter().max().unwrap_or(&1),
            4096,
        ])?)
        .ok_or(crate::Error::Overflow)?;
    ctx.check_lanes(Level::Bands, b_lanes, lane_bytes)?;
    let k_lanes = ctx.fit_lanes(
        Level::Kpts,
        k_lanes,
        product(&[m, n, 16])?
            .checked_add(product(&[b_lanes, lane_bytes])?)
            .ok_or(crate::Error::Overflow)?,
    )?;
    let mut targets: Vec<Vec<StateLane>> = (0..k_lanes)
        .map(|_| (0..b_lanes).map(|_| StateLane::default()).collect())
        .collect();
    let mut out = Shares::new(dirs.len());
    for s in 0..c.spins {
        let occupied = occupied_states(c, matrices, s, spin_scale, Some((aug, Part::Both)))?;
        let deltas = ctx.map_tasks(Level::Kpts, &mut targets, nk, |lanes, k| {
            let point = &c.points[k];
            let d = matrices[s * nk + k].matrix(&c.context)?;
            let d = &*d * C64::new(spin_scale, 0.);
            let mut u = c.context.allocate::<C64>(m * n)?;
            periodic_part(c, point.k, &point.psi.read(&c.context)?, u.as_mut_slice());
            let scale = -0.5 * screening.fraction * mult * point.weight;
            let mut shares = Vec::new();
            for source in &occupied {
                if source.wavefunctions.is_empty() {
                    continue;
                }
                let k_diff = std::array::from_fn(|a| source.k[a] - point.k[a]);
                let root = kernel_weights(c, k_diff, source.weight, screening.omega, exxdiv)?;
                let kw: Vec<f64> = root.as_slice().iter().map(|x| x * x).collect();
                let shapes = aug.pair_shapes(c, k_diff)?;
                let dkernel = sources.kernel(c, k_diff, source.weight, screening.omega, exxdiv)?;
                if dkernel.as_slice().len() != 6 * n {
                    return Err(invalid("the kernel strain must hold six rows of N values"));
                }
                let strain = sources.shapes(aug, c, k_diff)?;
                let forms = dirs
                    .iter()
                    .map(|e| {
                        // The form factor part is linear in the strain: `sum_ab E_ab form(a, b)`.
                        let mut combined: Option<Buffer<C64>> = None;
                        for a in 0..3 {
                            for b in 0..3 {
                                if e[a][b] == 0. {
                                    continue;
                                }
                                let form = strain.form(&c.context, a, b)?;
                                let sum = match combined.as_mut() {
                                    Some(sum) => sum,
                                    None => combined
                                        .insert(c.context.allocate::<C64>(form.as_slice().len())?),
                                };
                                for (to, from) in sum.as_mut_slice().iter_mut().zip(form.as_slice())
                                {
                                    *to += from * e[a][b];
                                }
                            }
                        }
                        combined.ok_or_else(|| invalid("a strain direction without entries"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let pair = PairData {
                    aug,
                    point,
                    u: u.as_slice(),
                    d: &d,
                    scale,
                    kw: &kw,
                    shapes: shapes.as_slice(),
                    dkernel: dkernel.as_slice(),
                    forms: &forms,
                    prefactor: source.weight * dv / n as f64,
                    source,
                    reduction: reduction.as_ref(),
                    bra_tangents: &tangent[k],
                    ket_tangents: if source.partner {
                        reversed[source.point]
                            .as_ref()
                            .ok_or_else(|| invalid("a partner of a point with its own -k"))?
                    } else {
                        &tangent[source.point]
                    },
                    directions: dirs,
                };
                let states = source.wavefunctions.len();
                let window = lanes.len().max(1);
                for first in (0..states).step_by(window) {
                    let count = window.min(states - first);
                    shares.extend(c.context.map_tasks(
                        Level::Bands,
                        &mut lanes[..count],
                        count,
                        |lane, offset| state_terms(c, lane, &pair, first + offset),
                    )?);
                }
            }
            Ok(shares)
        })?;
        for share in deltas.into_iter().flatten() {
            add(&mut out, &share);
        }
    }
    Ok(Terms {
        energy: out.energy,
        jacobian: directions.assemble(&out.jacobian),
        kernel: directions.assemble(&out.kernel),
        shape_volume: directions.assemble(&out.shape_volume),
        shape_form: directions.assemble(&out.shape_form),
        bra: directions.assemble(&out.bra),
        ket: directions.assemble(&out.ket),
        ..Terms::default()
    })
}

/// What a state lane keeps from one state to the next: its transform and its pair buffers, made on
/// the lane's own thread the first time (so the transform takes the lane's share of the workers).
#[derive(Default)]
struct StateLane {
    fft: Option<Fft3>,
    buffers: Option<[Buffer<C64>; 4]>,
}

/// What the states of one (target, source) pair share.
struct PairData<'a> {
    aug: &'a Augmentation<'a>,
    point: &'a crate::periodic::KPoint,
    u: &'a [C64],
    d: &'a DMatrix<C64>,
    scale: f64,
    kw: &'a [f64],
    shapes: &'a [C64],
    dkernel: &'a [f64],
    forms: &'a [Buffer<C64>],
    prefactor: f64,
    source: &'a OccupiedStates,
    /// The rotation of the states of a reduced mesh, `None` on a whole or halved one.
    reduction: Option<&'a Reduction>,
    bra_tangents: &'a [Buffer<C64>],
    ket_tangents: &'a [Buffer<C64>],
    directions: &'a [[[f64; 3]; 3]],
}

/// `out += share`, every entry: a share holds one value per entry the state touched and zeros
/// elsewhere (`out` is never a negative zero, so adding `+0` leaves it as it is).
fn add(out: &mut Shares, share: &Shares) {
    out.energy += share.energy;
    for (to, from) in [
        (&mut out.jacobian, &share.jacobian),
        (&mut out.kernel, &share.kernel),
        (&mut out.shape_volume, &share.shape_volume),
        (&mut out.shape_form, &share.shape_form),
        (&mut out.bra, &share.bra),
        (&mut out.ket, &share.ket),
    ] {
        for (t, f) in to.iter_mut().zip(from) {
            *t += f;
        }
    }
}

/// H-1..H-4 of one occupied state of a (target, source) pair, as a delta of the terms.
fn state_terms(c: &Crystal, lane: &mut StateLane, pd: &PairData, state: usize) -> Result<Shares> {
    let (n, m) = (c.grid.index.points(), c.m);
    let components = if c.spinor { 2 } else { 1 };
    let rows = components * m;
    if lane.fft.is_none() {
        lane.fft = Some(Fft3::new(&c.context, c.grid.index.shape())?);
    }
    if lane.buffers.is_none() {
        lane.buffers = Some([
            c.context.allocate::<C64>(n)?,
            c.context.allocate::<C64>(rows * n)?,
            c.context.allocate::<C64>(rows * n)?,
            c.context.allocate::<C64>(rows * n)?,
        ]);
    }
    let (Some(fft), Some([pair, raw, v, work])) = (lane.fft.as_mut(), lane.buffers.as_mut()) else {
        unreachable!()
    };
    let (point, source, scale, kw) = (pd.point, pd.source, pd.scale, pd.kw);
    let psi = &source.wavefunctions[state];
    let mut out = Shares::new(pd.directions.len());
    let mut density = vec![0.; n];
    let unit = vec![1.; n];
    let moments = pd.aug.moments(c, point, &source.projected[state]);
    let both = PairInputs {
        u_k: pd.u,
        psi,
        moments: Some(&moments),
        shapes: Some(pd.shapes),
        kernel: &unit,
        part: Part::Both,
    };
    weighted_pairs(c, fft, pair, &both, raw.as_mut_slice())?;
    v.as_mut_slice().fill(C64::default());
    for i in 0..rows {
        for j in 0..rows {
            let dij = pd.d[(i, j)];
            for at in 0..n {
                v.as_mut_slice()[i * n + at] += dij * raw.as_slice()[j * n + at];
            }
        }
    }
    for i in 0..rows {
        for at in 0..n {
            density[at] += (raw.as_slice()[i * n + at] * v.as_slice()[i * n + at].conj()).re;
        }
    }
    let energy: f64 = scale * kw.iter().zip(&density).map(|(w, r)| w * r).sum::<f64>();
    out.energy += energy;
    // The kernel's derivative against the pair density, for the six symmetric pairs.
    let kernel_rows: Vec<f64> = (0..6)
        .map(|r| {
            let line = &pd.dkernel[r * n..(r + 1) * n];
            scale * pd.prefactor * line.iter().zip(&density).map(|(x, r)| x * r).sum::<f64>()
        })
        .collect();
    for (d, e) in pd.directions.iter().enumerate() {
        out.jacobian[d] += trace(e) * energy;
        for a in 0..3 {
            for b in 0..3 {
                out.kernel[d] += e[a][b] * kernel_rows[row(a, b)];
            }
        }
    }
    // H-3: the compensation rows alone, the `1 / dv` of `T_a` ...
    let comp = PairInputs {
        u_k: pd.u,
        psi,
        moments: Some(&moments),
        shapes: Some(pd.shapes),
        kernel: &unit,
        part: Part::Compensation,
    };
    weighted_pairs(c, fft, pair, &comp, work.as_mut_slice())?;
    let volume = 2. * scale * dot(work.as_slice(), v.as_slice(), kw);
    for (d, e) in pd.directions.iter().enumerate() {
        out.shape_volume[d] -= trace(e) * volume;
    }
    // ... and the form factors.
    for (d, form) in pd.forms.iter().enumerate() {
        let input = PairInputs {
            u_k: pd.u,
            psi,
            moments: Some(&moments),
            shapes: Some(form.as_slice()),
            kernel: &unit,
            part: Part::Compensation,
        };
        weighted_pairs(c, fft, pair, &input, work.as_mut_slice())?;
        out.shape_form[d] += 2. * scale * dot(work.as_slice(), v.as_slice(), kw);
    }
    // H-4: the bra slot (the target's functions) and the ket slot (the state).
    for index in 0..pd.directions.len() {
        let bra = PairInputs {
            u_k: pd.bra_tangents[index].as_slice(),
            psi,
            moments: None,
            shapes: None,
            kernel: &unit,
            part: Part::Pseudo,
        };
        weighted_pairs(c, fft, pair, &bra, work.as_mut_slice())?;
        out.bra[index] += 2. * scale * dot(work.as_slice(), v.as_slice(), kw);
        // The moved state: one function per spin component.
        let mut moved = Vec::with_capacity(components);
        for component in 0..components {
            let mut buffer = c.context.allocate::<C64>(n)?;
            for mu in 0..m {
                let coefficient = source.coefficients[state][component * m + mu];
                let slice = &pd.ket_tangents[index].as_slice()[mu * n..(mu + 1) * n];
                for (x, y) in buffer.as_mut_slice().iter_mut().zip(slice) {
                    *x += coefficient * y;
                }
            }
            moved.push(buffer);
        }
        if let (Some(image), Some(reduction)) = (source.rotation.as_ref(), pd.reduction) {
            let input: Vec<&[C64]> = moved.iter().map(|b| b.as_slice()).collect();
            moved = reduction.grid_states(c, c.points[source.point].k, image, &input)?;
        }
        let ket = PairInputs {
            u_k: pd.u,
            psi: &moved,
            moments: None,
            shapes: None,
            kernel: &unit,
            part: Part::Pseudo,
        };
        weighted_pairs(c, fft, pair, &ket, work.as_mut_slice())?;
        out.ket[index] += 2. * scale * dot(work.as_slice(), v.as_slice(), kw);
    }
    Ok(out)
}
