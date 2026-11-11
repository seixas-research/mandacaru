//! Open-shell MP2 natural orbitals (`molecular::correlation::open_shell_natural_orbitals`)
//! against an independent route: second-order Rayleigh-Schroedinger perturbation theory solved
//! exactly in the space of determinants, with the same partition (`H0` the semicanonical spin
//! orbital energies of the reference's spin Fock operators, `V = H - H0`), and against the
//! closed-shell MP2 in its closed-shell limit.
use mandacaru_core::{
    math::SpatialGrid,
    molecular::{correlation, MolecularIntegrals, ScfOptions},
    quantum::C64,
    Context, Geometry, LengthUnit,
};
use nalgebra::{DMatrix, SymmetricEigen};

fn integrals(ctx: &Context, numbers: Vec<u8>, positions: Vec<[f64; 3]>) -> MolecularIntegrals {
    let geometry = Geometry::new(
        numbers,
        positions,
        [[6., 0., 0.], [0., 6., 0.], [0., 0., 7.]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.4 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO","virtual_orbitals":1}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}

/// Ascending eigenpairs of a symmetric matrix.
fn eigen(matrix: DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let e = SymmetricEigen::new(matrix);
    let n = e.eigenvalues.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
    (
        order.iter().map(|&j| e.eigenvalues[j]).collect(),
        DMatrix::from_fn(n, n, |i, j| e.eigenvectors[(i, order[j])]),
    )
}

/// `a_q` then `a_p^+` on a determinant (bit `s` = spin orbital `s`), with its fermionic sign.
fn hop(det: u64, p: usize, q: usize) -> Option<(u64, f64)> {
    if det & (1 << q) == 0 {
        return None;
    }
    let sign_q = if (det & ((1u64 << q) - 1)).count_ones() % 2 == 0 {
        1.
    } else {
        -1.
    };
    let after = det & !(1 << q);
    if after & (1 << p) != 0 {
        return None;
    }
    let sign_p = if (after & ((1u64 << p) - 1)).count_ones() % 2 == 0 {
        1.
    } else {
        -1.
    };
    Some((after | (1 << p), sign_p * sign_q))
}

/// The reference partition, solved exactly: `(E2, second-order spin-summed density in the
/// model orbitals)`.
fn rayleigh_schroedinger(
    m: usize,
    one: &[C64],
    two: &[C64],
    particles: [usize; 2],
) -> (f64, DMatrix<f64>) {
    let h = DMatrix::from_fn(m, m, |p, q| one[p * m + q].re);
    let g = |p: usize, q: usize, r: usize, s: usize| two[((p * m + q) * m + r) * m + s].re;
    // Spin Fock operators of the reference determinant and their semicanonical rotations.
    let mut rotations = Vec::new();
    let mut energies = Vec::new();
    for spin in 0..2 {
        let n = particles[spin];
        let fock = DMatrix::from_fn(m, m, |p, q| {
            h[(p, q)]
                + (0..particles[0]).map(|i| g(p, i, q, i)).sum::<f64>()
                + (0..particles[1]).map(|i| g(p, i, q, i)).sum::<f64>()
                - (0..n).map(|i| g(p, i, i, q)).sum::<f64>()
        });
        let fock = (&fock + fock.transpose()) * 0.5;
        let mut u = DMatrix::<f64>::zeros(m, m);
        let mut e = vec![0.; m];
        for (start, size) in [(0, n), (n, m - n)] {
            if size == 0 {
                continue;
            }
            let (values, vectors) = eigen(fock.view((start, start), (size, size)).into_owned());
            for j in 0..size {
                e[start + j] = values[j];
                for i in 0..size {
                    u[(start + i, start + j)] = vectors[(i, j)];
                }
            }
        }
        rotations.push(u);
        energies.push(e);
    }
    // Integrals over the spin orbitals of the rotated bases: one-body per spin, two-body per
    // spin pair (electron 1 in spin `s`, electron 2 in spin `t`).
    let h_s: Vec<DMatrix<f64>> = rotations.iter().map(|u| u.transpose() * &h * u).collect();
    let mut g_st = vec![vec![0.; m * m * m * m]; 4];
    for s in 0..2 {
        for t in 0..2 {
            let (us, ut) = (&rotations[s], &rotations[t]);
            let mut current: Vec<f64> = (0..m * m * m * m)
                .map(|f| {
                    let (p, q, r, x) = (f / (m * m * m), (f / (m * m)) % m, (f / m) % m, f % m);
                    g(p, q, r, x)
                })
                .collect();
            for axis in 0..4 {
                let u = if axis % 2 == 0 { us } else { ut };
                let mut next = vec![0.; current.len()];
                for (f, out) in next.iter_mut().enumerate() {
                    let mut index = [f / (m * m * m), (f / (m * m)) % m, (f / m) % m, f % m];
                    let target = index[axis];
                    for source in 0..m {
                        index[axis] = source;
                        let flat = ((index[0] * m + index[1]) * m + index[2]) * m + index[3];
                        *out += current[flat] * u[(source, target)];
                    }
                }
                current = next;
            }
            g_st[s * 2 + t] = current;
        }
    }
    // Determinants: spin orbital `s * m + p`.
    let modes = 2 * m;
    let dets: Vec<u64> = (0u64..1 << modes)
        .filter(|d| {
            (d & ((1 << m) - 1)).count_ones() as usize == particles[0]
                && (d >> m).count_ones() as usize == particles[1]
        })
        .collect();
    let index = |d: u64| dets.binary_search(&d).unwrap();
    let reference: u64 = ((1 << particles[0]) - 1) | (((1 << particles[1]) - 1) << m);
    let zeroth = |d: u64| -> f64 {
        (0..modes)
            .filter(|&s| d & (1 << s) != 0)
            .map(|s| energies[s / m][s % m])
            .sum()
    };
    // H applied to the reference.
    let mut h_ref = vec![0.; dets.len()];
    for s1 in 0..modes {
        for s2 in 0..modes {
            let (a, b) = (s1 / m, s2 / m);
            if a != b {
                continue;
            }
            if let Some((d, sign)) = hop(reference, s1, s2) {
                h_ref[index(d)] += sign * h_s[a][(s1 % m, s2 % m)];
            }
        }
    }
    for p in 0..modes {
        for q in 0..modes {
            for r in 0..modes {
                for x in 0..modes {
                    // 1/2 <pq|rx> a_p^+ a_q^+ a_x a_r, spin of p = r, q = x.
                    if p / m != r / m || q / m != x / m {
                        continue;
                    }
                    let value = 0.5
                        * g_st[(p / m) * 2 + q / m]
                            [(((p % m) * m + q % m) * m + r % m) * m + x % m];
                    if value == 0. {
                        continue;
                    }
                    // Right to left: a_r, a_x, a_q^+, a_p^+.
                    let applied = annihilate(reference, r)
                        .and_then(|(d, a)| annihilate(d, x).map(|(d, b)| (d, a * b)))
                        .and_then(|(d, a)| create(d, q).map(|(d, b)| (d, a * b)))
                        .and_then(|(d, a)| create(d, p).map(|(d, b)| (d, a * b)));
                    if let Some((d, sign)) = applied {
                        h_ref[index(d)] += value * sign;
                    }
                }
            }
        }
    }
    let e0 = zeroth(reference);
    let psi1: Vec<f64> = dets
        .iter()
        .enumerate()
        .map(|(k, &d)| {
            if d == reference {
                0.
            } else {
                -h_ref[k] / (zeroth(d) - e0)
            }
        })
        .collect();
    let e2: f64 = psi1.iter().zip(&h_ref).map(|(c, v)| c * v).sum();
    let norm: f64 = psi1.iter().map(|c| c * c).sum();
    // Second-order density per spin in its rotated basis, then back to the model orbitals.
    let mut gamma = DMatrix::<f64>::zeros(m, m);
    for spin in 0..2 {
        let mut d = DMatrix::<f64>::zeros(m, m);
        for p in 0..m {
            for q in 0..m {
                let (sp, sq) = (spin * m + p, spin * m + q);
                let mut value = 0.;
                for (k, &det) in dets.iter().enumerate() {
                    if psi1[k] == 0. {
                        continue;
                    }
                    if let Some((out, sign)) = hop(det, sp, sq) {
                        value += psi1[index(out)] * sign * psi1[k];
                    }
                }
                if p == q && p < particles[spin] {
                    value -= norm;
                }
                d[(p, q)] = value;
            }
        }
        gamma += &rotations[spin] * d * rotations[spin].transpose();
    }
    (e2, gamma)
}

/// `a_s` on a determinant, with the sign of the occupied spin orbitals before `s`.
fn annihilate(det: u64, s: usize) -> Option<(u64, f64)> {
    if det & (1 << s) == 0 {
        return None;
    }
    let sign = if (det & ((1u64 << s) - 1)).count_ones() % 2 == 0 {
        1.
    } else {
        -1.
    };
    Some((det & !(1 << s), sign))
}

/// `a_s^+` on a determinant, with the sign of the occupied spin orbitals before `s`.
fn create(det: u64, s: usize) -> Option<(u64, f64)> {
    if det & (1 << s) != 0 {
        return None;
    }
    let sign = if (det & ((1u64 << s) - 1)).count_ones() % 2 == 0 {
        1.
    } else {
        -1.
    };
    Some((det | (1 << s), sign))
}

#[test]
fn open_shell_mp2_is_second_order_perturbation_theory_in_its_partition() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    {
        // Linear H3: three electrons, the doublet (2, 1).
        let integrals = integrals(
            &ctx,
            vec![1, 1, 1],
            vec![[3., 3., 2.6], [3., 3., 3.5], [3., 3., 4.4]],
        );
        let state = integrals
            .mean_field("uhf", [2, 1], &ScfOptions::default())
            .unwrap();
        assert!(state.converged);
        let model = state.model(&integrals).unwrap();
        let m = model.m;
        let mp2 = correlation::open_shell_natural_orbitals(&integrals, &state.model_orbitals, 2, 1)
            .unwrap();
        let (e2, gamma) =
            rayleigh_schroedinger(m, model.one.as_slice(), model.two.as_slice(), [2, 1]);
        eprintln!(
            "m {m}: E2 {:.12} (reference {e2:.12})",
            mp2.correlation_energy
        );
        assert!((mp2.correlation_energy - e2).abs() < 1e-10);
        // The occupied block, reference plus second order.
        let high = 2;
        for p in 0..high {
            for q in 0..high {
                let reference = if p == q {
                    if p < 1 {
                        2.
                    } else {
                        1.
                    }
                } else {
                    0.
                };
                let got = mp2.occupied_density.as_slice()[p * high + q];
                assert!((got - reference - gamma[(p, q)]).abs() < 1e-10, "({p},{q})");
            }
        }
        // The true-virtual natural occupations, descending.
        let (values, _) = eigen(gamma.view((high, high), (m - high, m - high)).into_owned());
        for (got, want) in mp2
            .virtual_occupations
            .as_slice()
            .iter()
            .zip(values.iter().rev())
        {
            assert!((got - want).abs() < 1e-10);
        }
        // Every occupation together is the electron count to second order.
        let total: f64 = mp2.occupied_occupations.as_slice().iter().sum::<f64>()
            + mp2.virtual_occupations.as_slice().iter().sum::<f64>();
        assert!((total - 3.).abs() < 1e-10);
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_closed_shell_reference_gives_the_closed_shell_mp2() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    {
        let integrals = integrals(&ctx, vec![1, 1], vec![[3., 3., 3.13], [3., 3., 3.87]]);
        let state = integrals
            .mean_field("rhf", [1, 1], &ScfOptions::default())
            .unwrap();
        let model = state.model(&integrals).unwrap();
        let closed = correlation::natural_orbitals(
            &ctx,
            model.m,
            model.one.as_slice(),
            model.two.as_slice(),
            1,
        )
        .unwrap();
        let open =
            correlation::open_shell_natural_orbitals(&integrals, &state.model_orbitals, 1, 1)
                .unwrap();
        eprintln!(
            "closed {:.12}, open-shell route {:.12}",
            closed.correlation_energy, open.correlation_energy
        );
        assert!((closed.correlation_energy - open.correlation_energy).abs() < 1e-10);
        for (a, b) in closed
            .virtual_occupations
            .as_slice()
            .iter()
            .zip(open.virtual_occupations.as_slice())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}
