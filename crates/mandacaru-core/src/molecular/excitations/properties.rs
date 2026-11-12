//! Transition moments, oscillator strengths, dominant transitions, natural transition orbitals
//! and the one-center diagnostic of the solved roots.
use super::{
    invalid, response::Response, unrestricted::Unrestricted, Channel, Excitations, Request,
    Solved, Spin, Transition,
};
use crate::{
    math::linalg::real_eigen,
    molecular::{
        multipoles::{gradient_integrals, position_integrals},
        MeanField, MolecularIntegrals,
    },
    Result,
};
use nalgebra::DMatrix;

/// Transitions below this weight are left out of a root's list (the largest is always kept).
const SMALLEST_WEIGHT: f64 = 0.01;
/// At most this many transitions are listed per root.
const MOST_TRANSITIONS: usize = 10;

pub(super) fn assemble(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    response: &Response<'_>,
    request: &Request,
    solved: Solved,
    count: usize,
) -> Result<Excitations> {
    let (o, v) = (response.occupied(), response.virtual_orbitals());
    let orbitals = response.orbitals();
    let (co, cv) = (orbitals.columns(0, o), orbitals.columns(o, v));
    let singlet = response.spin() == Spin::Singlet;
    let dipoles = if singlet {
        let r = position_integrals(integrals, [0.; 3])?;
        Some(r.map(|d| co.transpose() * d * cv))
    } else {
        None
    };
    let velocities = if singlet {
        let g = gradient_integrals(integrals)?;
        Some(g.map(|d| co.transpose() * d * cv))
    } else {
        None
    };
    // The one-center Coulomb factors the Kohn-Sham energy linearizes (PAW-LCAO only).
    let one_center = match (
        &integrals.paw_setup,
        reference.functional.is_some(),
        &integrals.corrections,
    ) {
        (Some(setup), true, Some(corrections)) => {
            let count = setup.one_center_factor_count();
            let pairs = &corrections.pairs;
            if count > pairs.factors.len() {
                return Err(invalid(
                    "PAW one-center factors disagree with the pair corrections",
                ));
            }
            Some(
                pairs.factors[..count]
                    .iter()
                    .zip(&pairs.weights[..count])
                    .map(|(b, &w)| (w, co.transpose() * b * cv))
                    .collect::<Vec<_>>(),
            )
        }
        _ => None,
    };
    let mut out = Excitations {
        approximation: request.approximation,
        spin: response.spin(),
        coupling: request.coupling,
        functional: reference.functional.clone(),
        energies: solved.values[..count].to_vec(),
        oscillator_strengths: Vec::with_capacity(count),
        transition_dipoles: Vec::with_capacity(count),
        transition_velocities: Vec::with_capacity(count),
        velocity_oscillator_strengths: Vec::with_capacity(count),
        plus: Vec::with_capacity(count),
        minus: Vec::with_capacity(count),
        transitions: Vec::with_capacity(count),
        nto_weights: Vec::with_capacity(count),
        one_center_hartree: one_center.as_ref().map(|_| Vec::with_capacity(count)),
        channels: vec![Channel {
            occupied: o,
            virtual_orbitals: v,
            orbital_energies: response.orbital_energies().to_vec(),
            atomic_orbitals: &integrals.lowdin * orbitals,
        }],
        spin_squared: None,
        reference_spin_squared: None,
        requested: request.num_states,
        added: count - request.num_states,
        residuals: solved.residuals[..count].to_vec(),
        iterations: solved.iterations,
        applications: solved.applications,
    };
    for root in 0..count {
        let plus = response.matrix(solved.plus.column(root).as_slice());
        let minus = response.matrix(solved.minus.column(root).as_slice());
        let omega = solved.values[root];
        let mu = match &dipoles {
            Some(d) => d.clone().map(|d| std::f64::consts::SQRT_2 * d.dot(&plus)),
            None => [0.; 3],
        };
        out.oscillator_strengths
            .push(2. / 3. * omega * mu.iter().map(|x| x * x).sum::<f64>());
        out.transition_dipoles.push(mu);
        let p = match &velocities {
            Some(g) => g.clone().map(|g| std::f64::consts::SQRT_2 * g.dot(&minus)),
            None => [0.; 3],
        };
        out.velocity_oscillator_strengths
            .push(2. / 3. * p.iter().map(|x| x * x).sum::<f64>() / omega);
        out.transition_velocities.push(p);
        let mut transitions: Vec<Transition> = (0..o)
            .flat_map(|i| (0..v).map(move |a| (i, a)))
            .map(|(i, a)| Transition {
                channel: 0,
                occupied: i,
                virtual_orbital: o + a,
                weight: plus[(i, a)] * minus[(i, a)],
            })
            .collect();
        transitions.sort_by(|x, y| {
            y.weight
                .total_cmp(&x.weight)
                .then((x.occupied, x.virtual_orbital).cmp(&(y.occupied, y.virtual_orbital)))
        });
        let keep = transitions
            .iter()
            .take(MOST_TRANSITIONS)
            .take_while(|t| t.weight >= SMALLEST_WEIGHT)
            .count()
            .max(1);
        transitions.truncate(keep);
        out.transitions.push(transitions);
        out.nto_weights.push(singular_weights(&plus)?);
        if let (Some(factors), Some(shifts)) = (&one_center, out.one_center_hartree.as_mut()) {
            // First order in the missing kernel: 2 sum_P w_P (L^P . (X + Y))^2 for a singlet
            // (the Coulomb-type kernel cancels between the spins of a triplet).
            let shift = if singlet {
                2. * factors
                    .iter()
                    .map(|(w, l)| w * l.dot(&plus).powi(2))
                    .sum::<f64>()
            } else {
                0.
            };
            shifts.push(shift);
        }
        out.plus.push(vec![plus]);
        out.minus.push(vec![minus]);
    }
    Ok(out)
}

/// `sigma_k^2 / sum sigma^2` of `t`, descending, from the eigenvalues of `t t^T`.
fn singular_weights(t: &DMatrix<f64>) -> Result<Vec<f64>> {
    let (values, _) = real_eigen(t * t.transpose())?;
    let total: f64 = values.iter().map(|v| v.max(0.)).sum();
    if !(total > 0.) {
        return Err(invalid("an excitation amplitude vanishes"));
    }
    Ok(values.iter().rev().map(|v| v.max(0.) / total).collect())
}

/// Weights and orbital pairs of the natural transition orbitals of `t` (occupied x virtual):
/// `t = U diag(sigma) V^T`, occupied orbitals `C_o U`, empty ones `C_v V`, descending weight,
/// each pair signed so the occupied orbital's largest atomic coefficient is positive.
pub(super) fn natural_transition_orbitals(
    t: &DMatrix<f64>,
    atomic: &DMatrix<f64>,
    o: usize,
) -> Result<(Vec<f64>, DMatrix<f64>, DMatrix<f64>)> {
    let v = atomic.ncols() - o;
    if t.shape() != (o, v) {
        return Err(invalid(
            "an excitation amplitude does not match the orbitals",
        ));
    }
    let (values, u) = real_eigen(t * t.transpose())?;
    let total: f64 = values.iter().map(|x| x.max(0.)).sum();
    if !(total > 0.) {
        return Err(invalid("an excitation amplitude vanishes"));
    }
    let mut weights = Vec::with_capacity(o);
    let mut occupied = DMatrix::zeros(atomic.nrows(), o);
    let mut empty = DMatrix::zeros(atomic.nrows(), o);
    for (column, k) in (0..o).rev().enumerate() {
        let sigma = values[k].max(0.).sqrt();
        weights.push(values[k].max(0.) / total);
        let uk = u.column(k).into_owned();
        let mut hole = atomic.columns(0, o) * &uk;
        let mut particle = if sigma > 1e-12 {
            atomic.columns(o, v) * (t.transpose() * &uk) / sigma
        } else {
            nalgebra::DVector::zeros(atomic.nrows())
        };
        let mut pivot = 0;
        for i in 1..hole.len() {
            if hole[i].abs() > hole[pivot].abs() {
                pivot = i;
            }
        }
        if hole[pivot] < 0. {
            hole = -hole;
            particle = -particle;
        }
        occupied.set_column(column, &hole);
        empty.set_column(column, &particle);
    }
    Ok((weights, occupied, empty))
}

/// The properties of the roots of an unrestricted reference: transition moments summed over both
/// spins (spin orbitals: no spin-adaptation factor), the dominant transitions with their spin,
/// the natural-transition-orbital weights over both spins, the one-center diagnostic, and `<S^2>`.
pub(super) fn assemble_unrestricted(
    integrals: &MolecularIntegrals,
    reference: &MeanField,
    operator: &Unrestricted<'_>,
    request: &Request,
    solved: Solved,
    count: usize,
) -> Result<Excitations> {
    let blocks = |m: &DMatrix<f64>, spin: usize| {
        let (o, v) = operator.channel(spin);
        let c = operator.orbitals(spin);
        c.columns(0, o).transpose() * m * c.columns(o, v)
    };
    let r = position_integrals(integrals, [0.; 3])?;
    let g = gradient_integrals(integrals)?;
    let dipoles: [[DMatrix<f64>; 3]; 2] =
        std::array::from_fn(|s| std::array::from_fn(|axis| blocks(&r[axis], s)));
    let velocities: [[DMatrix<f64>; 3]; 2] =
        std::array::from_fn(|s| std::array::from_fn(|axis| blocks(&g[axis], s)));
    let one_center = match (&integrals.paw_setup, reference.functional.is_some(), &integrals.corrections) {
        (Some(setup), true, Some(corrections)) => {
            let count = setup.one_center_factor_count();
            let pairs = &corrections.pairs;
            if count > pairs.factors.len() {
                return Err(invalid("PAW one-center factors disagree with the pair corrections"));
            }
            Some(
                pairs.factors[..count]
                    .iter()
                    .zip(&pairs.weights[..count])
                    .map(|(b, &w)| (w, [blocks(b, 0), blocks(b, 1)]))
                    .collect::<Vec<_>>(),
            )
        }
        _ => None,
    };
    let overlap = operator.orbitals(0).transpose() * operator.orbitals(1);
    let particles = reference.particles;
    let s_z = 0.5 * (particles[0] as f64 - particles[1] as f64);
    let (oa, ob) = (particles[0], particles[1]);
    let reference_spin_squared = s_z * (s_z + 1.) + ob as f64
        - overlap.view((0, 0), (oa, ob)).iter().map(|x| x * x).sum::<f64>();
    let channels = (0..2)
        .map(|s| {
            let (o, v) = operator.channel(s);
            Channel {
                occupied: o,
                virtual_orbitals: v,
                orbital_energies: operator.orbital_energies(s).to_vec(),
                atomic_orbitals: &integrals.lowdin * operator.orbitals(s),
            }
        })
        .collect();
    let mut out = Excitations {
        approximation: request.approximation,
        spin: Spin::Unrestricted,
        coupling: request.coupling,
        functional: reference.functional.clone(),
        energies: solved.values[..count].to_vec(),
        oscillator_strengths: Vec::with_capacity(count),
        transition_dipoles: Vec::with_capacity(count),
        transition_velocities: Vec::with_capacity(count),
        velocity_oscillator_strengths: Vec::with_capacity(count),
        plus: Vec::with_capacity(count),
        minus: Vec::with_capacity(count),
        transitions: Vec::with_capacity(count),
        nto_weights: Vec::with_capacity(count),
        one_center_hartree: one_center.as_ref().map(|_| Vec::with_capacity(count)),
        channels,
        spin_squared: Some(Vec::with_capacity(count)),
        reference_spin_squared: Some(reference_spin_squared),
        requested: request.num_states,
        added: count - request.num_states,
        residuals: solved.residuals[..count].to_vec(),
        iterations: solved.iterations,
        applications: solved.applications,
    };
    for root in 0..count {
        let plus = operator.matrices(solved.plus.column(root).as_slice());
        let minus = operator.matrices(solved.minus.column(root).as_slice());
        let omega = solved.values[root];
        let mu: [f64; 3] =
            std::array::from_fn(|axis| (0..2).map(|s| dipoles[s][axis].dot(&plus[s])).sum());
        let p: [f64; 3] =
            std::array::from_fn(|axis| (0..2).map(|s| velocities[s][axis].dot(&minus[s])).sum());
        out.oscillator_strengths.push(2. / 3. * omega * mu.iter().map(|x| x * x).sum::<f64>());
        out.transition_dipoles.push(mu);
        out.velocity_oscillator_strengths
            .push(2. / 3. * p.iter().map(|x| x * x).sum::<f64>() / omega);
        out.transition_velocities.push(p);
        let mut transitions: Vec<Transition> = Vec::new();
        for s in 0..2 {
            let (o, v) = operator.channel(s);
            for i in 0..o {
                for a in 0..v {
                    transitions.push(Transition {
                        channel: s,
                        occupied: i,
                        virtual_orbital: o + a,
                        weight: plus[s][(i, a)] * minus[s][(i, a)],
                    });
                }
            }
        }
        transitions.sort_by(|x, y| {
            y.weight.total_cmp(&x.weight).then(
                (x.channel, x.occupied, x.virtual_orbital).cmp(&(y.channel, y.occupied, y.virtual_orbital)),
            )
        });
        let keep = transitions
            .iter()
            .take(MOST_TRANSITIONS)
            .take_while(|t| t.weight >= SMALLEST_WEIGHT)
            .count()
            .max(1);
        transitions.truncate(keep);
        out.transitions.push(transitions);
        // Singular values of both spin blocks together.
        let mut values = Vec::new();
        for block in &plus {
            if block.nrows() > 0 && block.ncols() > 0 {
                values.extend(real_eigen(block * block.transpose())?.0.into_iter().map(|v| v.max(0.)));
            }
        }
        let total: f64 = values.iter().sum();
        if !(total > 0.) {
            return Err(invalid("an excitation amplitude vanishes"));
        }
        values.sort_by(|a, b| b.total_cmp(a));
        out.nto_weights.push(values.iter().map(|v| v / total).collect());
        if let (Some(factors), Some(shifts)) = (&one_center, out.one_center_hartree.as_mut()) {
            // First order in the missing Coulomb-type kernel, 2 sum_P w_P (sum_s L^P_s . T_s)^2 / 2.
            shifts.push(
                factors
                    .iter()
                    .map(|(w, l)| w * (l[0].dot(&plus[0]) + l[1].dot(&plus[1])).powi(2))
                    .sum::<f64>(),
            );
        }
        let squared = spin_squared(&overlap, particles, &plus, &minus)?;
        if let Some(values) = out.spin_squared.as_mut() {
            values.push(squared);
        }
        out.plus.push(plus.to_vec());
        out.minus.push(minus.to_vec());
    }
    Ok(out)
}

/// `<S^2>` of the singles combination `sum_s sum_ia X_s,ia |Phi_i^a(s)>` of the reference's
/// determinants, with `X = (plus + minus) / 2` normalized (the Tamm-Dancoff `X` itself), and
/// `overlap` the alpha-beta orbital overlap `C_a^T C_b` (orthogonal: both are complete in the
/// basis). With `Gamma_s` the singles part of each spin's density matrix,
///
/// ```text
/// <S^2> = S_z (S_z + 1) + N_b - tr(Gamma_a M P_b M^T) - tr(P_a M Gamma_b M^T)
///         - 2 sum_ij M_ij (X_a M_vv X_b^T)_ij,
/// ```
///
/// (`M` the overlap, `P_s` the occupied projector of spin `s`, `M_vv` its empty-empty block).
/// It is 0 for a singlet and 2 for the `M_s = 0` triplet of a closed shell, exactly. An
/// approximation of the interacting state's value, as the reference determinant is.
fn spin_squared(
    overlap: &DMatrix<f64>,
    particles: [usize; 2],
    plus: &[DMatrix<f64>; 2],
    minus: &[DMatrix<f64>; 2],
) -> Result<f64> {
    let mut x: Vec<DMatrix<f64>> = (0..2).map(|s| (&plus[s] + &minus[s]) * 0.5).collect();
    let norm = (x[0].norm_squared() + x[1].norm_squared()).sqrt();
    if !(norm > 0.) {
        return Err(invalid("an excitation amplitude vanishes"));
    }
    for block in &mut x {
        *block /= norm;
    }
    let (oa, ob) = (particles[0], particles[1]);
    let m = overlap.nrows();
    let (va, vb) = (m - oa, m - ob);
    let s_z = 0.5 * (oa as f64 - ob as f64);
    // tr(Gamma_a Q) with Q = M P_b M^T in the alpha basis; Gamma_a = [n_a I - X X^T, 0; 0, X^T X].
    let mut traces = 0.;
    for (s, (o, v)) in [(oa, va), (ob, vb)].into_iter().enumerate() {
        let weight = x[s].norm_squared();
        // The other spin's occupied projector, carried into this spin's orbitals.
        let q = if s == 0 {
            let mb = overlap.columns(0, ob);
            &mb * mb.transpose()
        } else {
            let ma = overlap.rows(0, oa);
            ma.transpose() * ma
        };
        let q_oo = q.view((0, 0), (o, o));
        let q_vv = q.view((o, o), (v, v));
        let xx = &x[s] * x[s].transpose();
        let xtx = x[s].transpose() * &x[s];
        traces += weight * q_oo.trace() - (xx.component_mul(&q_oo.transpose())).sum()
            + (xtx.component_mul(&q_vv.transpose())).sum();
    }
    // The two singles of opposite spin coupled by the spin-flip operator.
    let m_oo = overlap.view((0, 0), (oa, ob));
    let m_vv = overlap.view((oa, ob), (va, vb));
    let cross = 2. * m_oo.component_mul(&(&x[0] * m_vv * x[1].transpose())).sum();
    Ok(s_z * (s_z + 1.) + ob as f64 - traces - cross)
}
