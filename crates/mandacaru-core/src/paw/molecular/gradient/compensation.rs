//! The compensation multipoles in the gradient: their moments in the occupied
//! orbitals, the field of their potential on the grid, the pair quadratures
//! `u` (Coulomb coupling of two multipoles) and `ionic` (the local potentials
//! of the other atoms in a compensation sphere), and the derivatives of all of
//! them with respect to the atom positions.
//!
//! The energy is the bilinear form `1/2 q.u.q + q.w` of the moments `q` of the
//! projections and `w` of the grid density with the multipole potentials, plus
//! the same form in the occupied pairs for exchange. The eigen-decomposition
//! of the coupled matrix that the setup uses to build its pair factors is a
//! rewriting of this form and is never differentiated.
use super::{
    super::{
        distance, harmonic, onecenter::Multipole, potential, shape, sphere, Setup, ELECTROSTATIC,
        IONIC,
    },
    radial::{local_slope, potential_slope, radial_gradient},
    Occupied,
};
use crate::{
    math::spline::Cubic, molecular::MolecularIntegrals,
    paw::onecenter_radial::long_range_shape_spline, product, Result,
};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

/// One compensation multipole `(l, m)` of an atom and its `p x p` block. A
/// screened channel carries the spline of the potential's long-range part and
/// has the short-range potential (bare minus long range) of a screened hybrid.
pub(super) struct Channel<'a> {
    pub atom: usize,
    pub l: usize,
    pub m: i32,
    pub radius: f64,
    pub block: &'a DMatrix<f64>,
    pub long_range: Option<Arc<Cubic>>,
}
/// The channels in the order the setup assembles its multipoles.
pub(super) fn channels(setup: &Setup) -> Vec<Channel<'_>> {
    let mut out = Vec::new();
    for (atom, &s) in setup.atoms.iter().enumerate() {
        for Multipole { l, m, block } in &setup.species_terms[s].multipoles {
            out.push(Channel {
                atom,
                l: *l,
                m: *m,
                radius: setup.species[s].compensation_radius,
                block,
                long_range: None,
            });
        }
    }
    out
}
/// The channels of a screened hybrid: the shape potentials under `erfc(omega r) / r`.
pub(super) fn screened_channels<'a>(
    setup: &'a Setup,
    integrals: &MolecularIntegrals,
    omega: f64,
) -> Result<Vec<Channel<'a>>> {
    let mut out = channels(setup);
    let positions = integrals.geometry.positions();
    // The table reaches the farthest grid point of any atom, as the setup's own
    // (per atom) does; one spline per distinct shape. Convexity of Euclidean distance
    // guarantees the maximum occurs at one of the eight grid corners.
    let mut largest = 0f64;
    let [nx, ny, nz] = integrals.grid.index.shape();
    for ix in [0, nx.saturating_sub(1)] {
        for iy in [0, ny.saturating_sub(1)] {
            for iz in [0, nz.saturating_sub(1)] {
                let k = (ix * ny + iy) * nz + iz;
                let point = integrals.grid.point(k);
                for center in positions {
                    largest = largest.max(distance(
                        std::array::from_fn(|i| point[i] - center[i]),
                        [0.; 3],
                    ));
                }
            }
        }
    }
    let mut cache: Vec<((u64, usize), Arc<Cubic>)> = Vec::new();
    for channel in &mut out {
        let key = (channel.radius.to_bits(), channel.l);
        let spline = match cache.iter().find(|(k, _)| *k == key) {
            Some((_, spline)) => spline.clone(),
            None => {
                let spline = Arc::new(long_range_shape_spline(
                    &integrals.context,
                    largest,
                    channel.radius,
                    channel.l,
                    omega,
                )?);
                cache.push((key, spline.clone()));
                spline
            }
        };
        channel.long_range = Some(spline);
    }
    Ok(out)
}
impl Channel<'_> {
    /// `(value, slope)` of the radial potential at `r`.
    fn radial(&self, r: f64) -> (f64, f64) {
        let (value, slope) = potential_slope(r, self.radius, self.l);
        match &self.long_range {
            Some(long) => {
                let (lv, ls) = long.value_slope(r);
                (value - lv, slope - ls)
            }
            None => (value, slope),
        }
    }
    /// The potential of the multipole at the offset `delta` from its atom.
    pub fn field(&self, delta: [f64; 3]) -> f64 {
        let r = distance(delta, [0.; 3]);
        let value = match &self.long_range {
            Some(long) => potential(r, self.radius, self.l) - long.value_slope(r).0,
            None => potential(r, self.radius, self.l),
        };
        value * harmonic(self.l, self.m, delta)
    }
    /// Its gradient with respect to `delta`.
    pub fn gradient(&self, delta: [f64; 3]) -> [f64; 3] {
        let (value, slope) = self.radial(distance(delta, [0.; 3]));
        radial_gradient(value, slope, self.l, self.m, delta)
    }
}

/// Moments and pair quadratures of the occupied orbitals.
pub(super) struct Moments {
    /// `q[A]`: the moment of multipole `A` in the projections of all orbitals.
    pub q: Vec<f64>,
    /// `w[A]`: the moment of the grid density with the potential of `A`.
    pub w: Vec<f64>,
    /// `xq[(A * n + i) * n + j]`: the matrix element of `A` between the
    /// projections of orbitals `i` and `j`.
    pub xq: Vec<f64>,
    /// The same element of the grid pair density (exchange only, else empty).
    pub xw: Vec<f64>,
    /// The symmetrized Coulomb coupling of two multipoles.
    pub usym: DMatrix<f64>,
    pub ionic: Vec<f64>,
}
impl Moments {
    pub fn n(&self) -> usize {
        self.q.len()
    }
}

/// Orbital `i`'s projection onto the projectors of the atom of multipole `A`.
pub(super) fn matrix_element(
    channel: &Channel<'_>,
    left: &DVector<f64>,
    right: &DVector<f64>,
) -> f64 {
    left.dot(&(channel.block * right))
}

/// The field of multipole `A` on the grid, without the volume element.
pub(super) fn grid_field(integrals: &MolecularIntegrals, channel: &Channel<'_>, out: &mut [f64]) {
    let center = integrals.geometry.positions()[channel.atom];
    for (k, value) in out.iter_mut().enumerate() {
        let point = integrals.grid.point(k);
        *value = channel.field(std::array::from_fn(|i| point[i] - center[i]));
    }
}

/// `electrostatics` is the setup's own `(U, ionic)` for these `channels`, when the caller has it:
/// the bare channels may pass the calculation's cached matrices, while screened channels
/// (`screened_channels`) pass `None`, because a cached matrix holds the bare `1/r` coupling,
/// not the `erfc(omega r)/r` one.
#[allow(clippy::too_many_arguments)]
pub(super) fn moments(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    channels: &[Channel<'_>],
    electrostatics: Option<(&DMatrix<f64>, &[f64])>,
    occupied: &[Occupied],
    projected: &[Vec<DVector<f64>>],
    psi: &[f64],
    rho: &[f64],
    hf: bool,
) -> Result<Moments> {
    let ctx = &integrals.context;
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let nocc = occupied.len();
    let nch = channels.len();
    let _memory = ctx.reserve(product(&[nch.max(1), nocc.max(1), nocc.max(1), 8, 2])?)?;
    let mut q = vec![0.; nch];
    let mut w = vec![0.; nch];
    let mut xq = vec![0.; nch * nocc * nocc];
    let mut xw = if hf {
        vec![0.; nch * nocc * nocc]
    } else {
        Vec::new()
    };
    let mut field = ctx.allocate::<f64>(n)?;
    for (a, channel) in channels.iter().enumerate() {
        for i in 0..nocc {
            for j in 0..nocc {
                xq[(a * nocc + i) * nocc + j] = matrix_element(
                    channel,
                    &projected[channel.atom][i],
                    &projected[channel.atom][j],
                );
            }
            q[a] += occupied[i].weight * xq[(a * nocc + i) * nocc + i];
        }
        grid_field(integrals, channel, field.as_mut_slice());
        w[a] = dv
            * rho
                .iter()
                .zip(field.as_slice())
                .map(|(r, f)| r * f)
                .sum::<f64>();
        if hf {
            for i in 0..nocc {
                for j in 0..=i {
                    if occupied[i].spin != occupied[j].spin {
                        continue;
                    }
                    let value = dv
                        * (0..n)
                            .map(|k| psi[i * n + k] * psi[j * n + k] * field.as_slice()[k])
                            .sum::<f64>();
                    xw[(a * nocc + i) * nocc + j] = value;
                    xw[(a * nocc + j) * nocc + i] = value;
                }
            }
        }
    }
    // Coulomb coupling of two multipoles (the setup's electrostatic matrix) and
    // the local potentials of the other atoms in a compensation sphere.
    let positions = integrals.geometry.positions();
    let (u, ionic) = if let Some((u, ionic)) = electrostatics {
        (u.clone(), ionic.to_vec())
    } else {
        let mut u = DMatrix::<f64>::zeros(nch, nch);
        let mut ionic = vec![0.; nch];
        for (b, &s) in setup.atoms.iter().enumerate() {
            let radius = setup.species[s].compensation_radius;
            sphere(
                radius,
                0.,
                ELECTROSTATIC[0],
                ELECTROSTATIC[1],
                ELECTROSTATIC[2],
                |offset, r, weight| {
                    let point: [f64; 3] = std::array::from_fn(|i| positions[b][i] + offset[i]);
                    for (j, cj) in channels.iter().enumerate().filter(|(_, c)| c.atom == b) {
                        let density = shape(r, radius, cj.l) * harmonic(cj.l, cj.m, offset) * weight;
                        for (i, ci) in channels.iter().enumerate() {
                            let delta = std::array::from_fn(|k| point[k] - positions[ci.atom][k]);
                            u[(i, j)] += ci.field(delta) * density;
                        }
                    }
                    Ok(())
                },
            )?;
            sphere(
                radius,
                0.,
                IONIC[0],
                IONIC[1],
                IONIC[2],
                |offset, r, weight| {
                    let point: [f64; 3] = std::array::from_fn(|i| positions[b][i] + offset[i]);
                    let value: f64 = setup
                        .atoms
                        .iter()
                        .enumerate()
                        .filter(|(a, _)| *a != b)
                        .map(|(a, &sa)| setup.species[sa].local(distance(point, positions[a])))
                        .sum();
                    for (i, ch) in channels.iter().enumerate().filter(|(_, c)| c.atom == b) {
                        ionic[i] +=
                            weight * value * shape(r, radius, ch.l) * harmonic(ch.l, ch.m, offset);
                    }
                    Ok(())
                },
            )?;
        }
        (u, ionic)
    };
    let usym = (&u + u.transpose()) * 0.5;
    Ok(Moments {
        q,
        w,
        xq,
        xw,
        usym,
        ionic,
    })
}

/// Weights of the pair quadratures in the gradient: `energy[A][B] =
/// d E / d u_AB` of the Coulomb form (`coulomb`) and of the exchange form.
pub(super) struct PairWeights {
    pub coulomb: DMatrix<f64>,
    pub exchange: DMatrix<f64>,
}
pub(super) fn pair_weights(moments: &Moments, occupied: &[Occupied], hf: bool) -> PairWeights {
    let nch = moments.n();
    let nocc = occupied.len();
    let coulomb = DMatrix::from_fn(nch, nch, |a, b| 0.5 * moments.q[a] * moments.q[b]);
    let exchange = if hf {
        DMatrix::from_fn(nch, nch, |a, b| {
            let mut sum = 0.;
            for i in 0..nocc {
                for j in 0..nocc {
                    if occupied[i].spin == occupied[j].spin {
                        sum += occupied[i].weight
                            * moments.xq[(a * nocc + i) * nocc + j]
                            * moments.xq[(b * nocc + i) * nocc + j];
                    }
                }
            }
            -0.5 * sum
        })
    } else {
        DMatrix::zeros(nch, nch)
    };
    PairWeights { coulomb, exchange }
}

/// `d E / d u` contracted with the derivative of the quadrature: the force on
/// every atom from one weight matrix. Points of the sphere of atom `b` move
/// with `b`, so a multipole on `b` does not move relative to its own sphere.
pub(super) fn electrostatic_force(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    channels: &[Channel<'_>],
    weights: &DMatrix<f64>,
) -> Result<Vec<f64>> {
    let positions = integrals.geometry.positions();
    let n_atoms = setup.atoms.len();

    let compute_atom = |b: usize| -> Result<Vec<f64>> {
        let s = setup.atoms[b];
        let radius = setup.species[s].compensation_radius;
        let mut force = vec![0.; 3 * positions.len()];
        let b_channels: Vec<(usize, usize, i32)> = channels
            .iter()
            .enumerate()
            .filter(|(_, c)| c.atom == b)
            .map(|(j, c)| (j, c.l, c.m))
            .collect();
        if b_channels.is_empty() {
            return Ok(force);
        }
        let other_channels: Vec<usize> = channels
            .iter()
            .enumerate()
            .filter(|(_, c)| c.atom != b)
            .map(|(i, _)| i)
            .collect();
        if other_channels.is_empty() {
            return Ok(force);
        }
        let mut a_j = vec![0.; b_channels.len()];
        let mut source = vec![0.; channels.len()];

        sphere(
            radius,
            0.,
            ELECTROSTATIC[0],
            ELECTROSTATIC[1],
            ELECTROSTATIC[2],
            |offset, r, weight| {
                let mut any_non_zero = false;
                for (idx, &(_, l, m)) in b_channels.iter().enumerate() {
                    let val = shape(r, radius, l) * harmonic(l, m, offset) * weight;
                    a_j[idx] = val;
                    if val != 0. {
                        any_non_zero = true;
                    }
                }
                if !any_non_zero {
                    return Ok(());
                }
                let point: [f64; 3] = std::array::from_fn(|i| positions[b][i] + offset[i]);
                for &i in &other_channels {
                    let mut s = 0.;
                    for (idx, &(j, _, _)) in b_channels.iter().enumerate() {
                        let w = weights[(i, j)];
                        if w != 0. {
                            s += w * a_j[idx];
                        }
                    }
                    source[i] = s;
                }
                for &i in &other_channels {
                    let src = source[i];
                    if src == 0. {
                        continue;
                    }
                    let ci = &channels[i];
                    let delta = std::array::from_fn(|k| point[k] - positions[ci.atom][k]);
                    let gradient = ci.gradient(delta);
                    for axis in 0..3 {
                        let value = gradient[axis] * src;
                        force[3 * ci.atom + axis] -= value;
                        force[3 * b + axis] += value;
                    }
                }
                Ok(())
            },
        )?;
        Ok(force)
    };

    let workers = integrals.context.threads().max(1).min(n_atoms.max(1));
    let mut total_force = vec![0.; 3 * positions.len()];
    if workers <= 1 {
        for b in 0..n_atoms {
            let part = compute_atom(b)?;
            for (tot, val) in total_force.iter_mut().zip(part) {
                *tot += val;
            }
        }
    } else {
        integrals.context.run(|| {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n_atoms)
                    .map(|b| mandacaru_kernels::runtime::spawn(scope, move || compute_atom(b)))
                    .collect();
                for handle in handles {
                    let part = handle.join().unwrap().unwrap();
                    for (tot, val) in total_force.iter_mut().zip(part) {
                        *tot += val;
                    }
                }
            })
        });
    }
    Ok(total_force)
}

/// The force on the atoms from `sum_A coefficient[A] ionic[A]`.
pub(super) fn ionic_force(
    setup: &Setup,
    integrals: &MolecularIntegrals,
    channels: &[Channel<'_>],
    coefficient: &[f64],
) -> Result<Vec<f64>> {
    let positions = integrals.geometry.positions();
    let mut force = vec![0.; 3 * positions.len()];
    for (b, &s) in setup.atoms.iter().enumerate() {
        let radius = setup.species[s].compensation_radius;
        sphere(
            radius,
            0.,
            IONIC[0],
            IONIC[1],
            IONIC[2],
            |offset, r, weight| {
                let point: [f64; 3] = std::array::from_fn(|i| positions[b][i] + offset[i]);
                let source: f64 = channels
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.atom == b)
                    .map(|(i, c)| {
                        coefficient[i] * shape(r, radius, c.l) * harmonic(c.l, c.m, offset) * weight
                    })
                    .sum();
                if source == 0. {
                    return Ok(());
                }
                for (a, &sa) in setup.atoms.iter().enumerate().filter(|(a, _)| *a != b) {
                    let delta: [f64; 3] = std::array::from_fn(|k| point[k] - positions[a][k]);
                    let d = distance(delta, [0.; 3]);
                    let slope = local_slope(&setup.species[sa], d);
                    for axis in 0..3 {
                        let value = source * slope * delta[axis] / d;
                        force[3 * a + axis] -= value;
                        force[3 * b + axis] += value;
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(force)
}

/// The force on the multipole atoms from `sum_k weight[A][k] dphi_A(k)/dR`, the
/// field of multipole `A` summed against a grid weight per multipole.
pub(super) fn field_force(
    integrals: &MolecularIntegrals,
    channels: &[Channel<'_>],
    weight: impl Fn(usize, usize) -> f64 + Sync,
) -> Result<Vec<f64>> {
    let positions = integrals.geometry.positions();
    let n = integrals.grid.index.points();
    let dv = integrals.grid.index.dv();
    let nch = channels.len();

    let compute_channel = |a: usize| -> [f64; 3] {
        let channel = &channels[a];
        let center = positions[channel.atom];
        let mut sum = [0.; 3];
        for k in 0..n {
            let w = weight(a, k);
            if w == 0. {
                continue;
            }
            let point = integrals.grid.point(k);
            let delta = std::array::from_fn(|i| point[i] - center[i]);
            let gradient = channel.gradient(delta);
            for axis in 0..3 {
                sum[axis] -= w * gradient[axis];
            }
        }
        sum
    };

    let workers = integrals.context.threads().max(1).min(nch.max(1));
    let mut force = vec![0.; 3 * positions.len()];
    if workers <= 1 {
        for (a, channel) in channels.iter().enumerate() {
            let sum = compute_channel(a);
            for axis in 0..3 {
                force[3 * channel.atom + axis] += dv * sum[axis];
            }
        }
    } else {
        let sums: Vec<[f64; 3]> = integrals.context.run(|| {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..nch)
                    .map(|a| {
                        mandacaru_kernels::runtime::spawn(scope, move || compute_channel(a))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        });
        for (a, channel) in channels.iter().enumerate() {
            let sum = sums[a];
            for axis in 0..3 {
                force[3 * channel.atom + axis] += dv * sum[axis];
            }
        }
    }
    Ok(force)
}

/// Fill `out` (`n` values per row) with `-sum_A phi_A(k) sum_j xq[A][i][j] psi_j(k)`,
/// the compensation part of the exchange action on every occupied orbital, and
/// the Coulomb potential `sum_A q[A] phi_A(k)` of the compensation charges.
pub(super) fn grid_actions(
    integrals: &MolecularIntegrals,
    channels: &[Channel<'_>],
    moments: &Moments,
    occupied: &[Occupied],
    psi: &[f64],
    exchange: Option<&mut [f64]>,
    coulomb: &mut [f64],
) -> Result<()> {
    let n = integrals.grid.index.points();
    let nocc = occupied.len();
    let mut field = integrals.context.allocate::<f64>(n)?;
    let mut exchange = exchange;
    for (a, channel) in channels.iter().enumerate() {
        grid_field(integrals, channel, field.as_mut_slice());
        for (c, f) in coulomb.iter_mut().zip(field.as_slice()) {
            *c += moments.q[a] * f;
        }
        if let Some(action) = exchange.as_deref_mut() {
            for i in 0..nocc {
                for j in 0..nocc {
                    if occupied[i].spin != occupied[j].spin {
                        continue;
                    }
                    let element = moments.xq[(a * nocc + i) * nocc + j];
                    if element == 0. {
                        continue;
                    }
                    for k in 0..n {
                        action[i * n + k] -= element * field.as_slice()[k] * psi[j * n + k];
                    }
                }
            }
        }
    }
    Ok(())
}
