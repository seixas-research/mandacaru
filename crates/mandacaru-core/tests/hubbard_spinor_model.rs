//! Noncollinear (spin-orbit) Dudarev model on the full 2p x 2p occupation matrix:
//! the closed forms, the freeze in `test/migration/reference/hubbard/spinor.json`,
//! rotation invariance, the orientation of V (the energy cannot see it), the
//! collinear embedding, the j-resolved atomic limit built from
//! `models::spin_orbit::angular`, the time-reversal operator and its completion.
use mandacaru_core::math::angular;
use mandacaru_core::models::hubbard::{self, Hubbard, Spin};
use mandacaru_core::models::hubbard_spinor::{
    self as spinor, Evaluation, Layout, Occupation, Potential,
};
use mandacaru_core::models::spin_orbit;
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;
use serde_json::Value;

const ONE_MIB: usize = 1 << 20;

fn context() -> Context {
    Context::new(Some(1), 64 * ONE_MIB).unwrap()
}

fn freeze() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/hubbard/spinor.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn fraction(v: &Value) -> f64 {
    let text = v.as_str().unwrap();
    let (a, b) = text.split_once('/').unwrap();
    a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap()
}

fn json_matrix(v: &Value) -> DMatrix<C64> {
    let (re, im) = (v["re"].as_array().unwrap(), v["im"].as_array().unwrap());
    DMatrix::from_fn(re.len(), re.len(), |i, j| {
        C64::new(re[i][j].as_f64().unwrap(), im[i][j].as_f64().unwrap())
    })
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
    fn unit(&mut self) -> f64 {
        0.5 * (self.next() + 1.)
    }
}

fn unitary(dim: usize, rng: &mut Lcg) -> DMatrix<C64> {
    DMatrix::from_fn(dim, dim, |_, _| C64::new(rng.next(), rng.next()))
        .qr()
        .q()
}

/// `Q diag(lambda) Q^dagger` with a random Q and eigenvalues in [0, 1].
fn random_state(dim: usize, rng: &mut Lcg) -> (DMatrix<C64>, DMatrix<C64>, Vec<f64>) {
    let q = unitary(dim, rng);
    let lambda: Vec<f64> = (0..dim).map(|_| rng.unit()).collect();
    let d = DMatrix::from_fn(dim, dim, |i, j| {
        C64::new(if i == j { lambda[i] } else { 0. }, 0.)
    });
    (&q * d * q.adjoint(), q, lambda)
}

fn hermitian_direction(dim: usize, rng: &mut Lcg) -> DMatrix<C64> {
    let a = DMatrix::from_fn(dim, dim, |_, _| C64::new(rng.next(), rng.next()));
    (&a + a.adjoint()) * C64::new(0.5, 0.)
}

fn flat(m: &DMatrix<C64>) -> Vec<C64> {
    let d = m.nrows();
    (0..d * d).map(|k| m[(k / d, k % d)]).collect()
}

fn dm(d: usize, v: &[C64]) -> DMatrix<C64> {
    DMatrix::from_fn(d, d, |i, j| v[i * d + j])
}

fn max_abs(m: &DMatrix<C64>) -> f64 {
    m.iter().map(|x| x.norm()).fold(0., f64::max)
}

fn hub(u: f64, j: f64) -> Hubbard {
    Hubbard::new(u, j).unwrap()
}

fn layout(l: usize, real: bool) -> Layout {
    Layout::new(l, real).unwrap()
}

fn occ(layout: Layout, m: &DMatrix<C64>) -> Occupation {
    Occupation::new(&context(), layout, &flat(m)).unwrap()
}

fn eval(h: &Hubbard, layout: Layout, m: &DMatrix<C64>) -> Evaluation {
    spinor::evaluate(&context(), h, &occ(layout, m)).unwrap()
}

fn potential(e: &Evaluation) -> DMatrix<C64> {
    let d = e.potential.layout().size();
    dm(d, e.potential.as_slice())
}

/// `Re tr[V H] = Re sum_ab V_ab H_ba`, computed here and not by the model.
fn tr_vh(v: &DMatrix<C64>, h: &DMatrix<C64>) -> f64 {
    let d = v.nrows();
    (0..d)
        .flat_map(|a| (0..d).map(move |b| (a, b)))
        .map(|(a, b)| (v[(a, b)] * h[(b, a)]).re)
        .sum()
}

/// Richardson derivative of the model's own energy along `h`.
fn finite_difference(hb: &Hubbard, lay: Layout, n: &DMatrix<C64>, h: &DMatrix<C64>) -> f64 {
    let f = |t: f64| eval(hb, lay, &(n + h * C64::new(t, 0.))).energy;
    let d = |s: f64| (f(s) - f(-s)) / (2. * s);
    (4. * d(0.04) - d(0.08)) / 3.
}

fn ls_matrix(l: usize, real: bool) -> DMatrix<C64> {
    let a = spin_orbit::angular(&context(), l, real).unwrap();
    dm(2 * (2 * l + 1), a.as_slice())
}

fn theta_matrix(lay: Layout) -> DMatrix<C64> {
    let u = spinor::time_reversal(&context(), lay).unwrap();
    dm(lay.size(), u.as_slice())
}

/// Pauli vector `tr[sigma n]`, test-local.
fn pauli(p: usize, n: &DMatrix<C64>) -> [f64; 3] {
    let mut m = [0.; 3];
    for a in 0..p {
        let (uu, ud, du, dd) = (n[(a, a)], n[(a, p + a)], n[(p + a, a)], n[(p + a, p + a)]);
        m[0] += (ud + du).re;
        m[1] += (C64::new(0., 1.) * (ud - du)).re;
        m[2] += (uu - dd).re;
    }
    m
}

fn spin_rotation(p: usize, rng: &mut Lcg) -> DMatrix<C64> {
    // exp(-i theta n.sigma / 2) = cos(theta/2) - i sin(theta/2) n.sigma
    let mut axis = [rng.next(), rng.next(), rng.next()];
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    axis.iter_mut().for_each(|x| *x /= norm);
    let theta = 0.3 + 2.7 * rng.unit();
    let (c, s) = ((0.5 * theta).cos(), (0.5 * theta).sin());
    let u = [
        [
            C64::new(c, -s * axis[2]),
            C64::new(-s * axis[1], -s * axis[0]),
        ],
        [
            C64::new(s * axis[1], -s * axis[0]),
            C64::new(c, s * axis[2]),
        ],
    ];
    DMatrix::from_fn(2 * p, 2 * p, |i, j| {
        if i % p == j % p {
            u[i / p][j / p]
        } else {
            C64::default()
        }
    })
}

const CASES: [(usize, bool); 4] = [(2, false), (3, false), (2, true), (3, true)];

// ---------------------------------------------------------------- the functional

#[test]
fn the_freeze_reproduces_energy_and_potential_for_d_and_f() {
    let doc = freeze();
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let d = &doc[name]["dudarev"];
        let u_eff = doc[name]["U_eff"].as_f64().unwrap();
        let n = json_matrix(&d["n"]);
        let e = eval(&hub(u_eff, 0.), layout(l, false), &n);
        let want = d["E"].as_f64().unwrap();
        assert!(
            (e.energy - want).abs() < 1e-12 * want.abs().max(1.),
            "{name}"
        );
        let v = json_matrix(&d["V"]);
        assert!(max_abs(&(potential(&e) - v)) < 1e-13, "{name}");
        assert!((e.per_u_eff * u_eff - e.energy).abs() < 1e-14);
    }
}

#[test]
fn the_energy_and_potential_follow_from_the_spectrum() {
    let mut rng = Lcg(11);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, q, lam) = random_state(lay.size(), &mut rng);
        let (u, j) = (3.7, 0.5);
        let u_eff = u - j;
        let e = eval(&hub(u, j), lay, &n);
        let want: f64 = lam.iter().map(|x| 0.5 * u_eff * x * (1. - x)).sum();
        assert!((e.energy - want).abs() < 1e-13, "l={l} real={real}");
        let vd = DMatrix::from_fn(lay.size(), lay.size(), |a, b| {
            C64::new(
                if a == b {
                    0.5 * u_eff * (1. - 2. * lam[a])
                } else {
                    0.
                },
                0.,
            )
        });
        assert!(max_abs(&(potential(&e) - &q * vd * q.adjoint())) < 1e-13);
        let (lo, hi) = spinor::eigenvalue_range(&context(), &occ(lay, &n)).unwrap();
        let (wlo, whi) = (
            lam.iter().cloned().fold(f64::INFINITY, f64::min),
            lam.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!((lo - wlo).abs() < 1e-12 && (hi - whi).abs() < 1e-12);
    }
}

#[test]
fn the_energy_is_invariant_and_the_potential_covariant_under_u_2p() {
    let mut rng = Lcg(23);
    let hb = hub(3.7, 0.);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let base = eval(&hb, lay, &n);
        let q = unitary(lay.size(), &mut rng);
        let rotated = eval(&hb, lay, &(&q * &n * q.adjoint()));
        assert!((rotated.energy - base.energy).abs() < 1e-13);
        let expected = &q * potential(&base) * q.adjoint();
        assert!(max_abs(&(potential(&rotated) - expected)) < 1e-13);
        // The SU(2) spin rotation is one such unitary; the stored one is a second witness.
        let w = spin_rotation(lay.width(), &mut rng);
        let spin = eval(&hb, lay, &(&w * &n * w.adjoint()));
        assert!((spin.energy - base.energy).abs() < 1e-13);
        // Negative control: a non-unitary change of basis is not a symmetry.
        let s = DMatrix::from_fn(lay.size(), lay.size(), |i, j| {
            C64::new(
                if i == j {
                    1. + 0.1 * (i as f64 + 1.) / lay.size() as f64
                } else {
                    0.
                },
                0.,
            )
        });
        let scaled = eval(&hb, lay, &(&s * &n * s.adjoint()));
        assert!((scaled.energy - base.energy).abs() > 1e-3);
    }
}

// ---------------------------------------------------------------- orientation

#[test]
fn the_potential_is_the_derivative_in_the_stated_orientation() {
    // dE = tr[V dn] with V_ab = dE/dn_ba. The energy is blind to n <-> n^T, V is not.
    let doc = freeze();
    let mut rng = Lcg(31);
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let d = &doc[name]["dudarev"];
        let u_eff = doc[name]["U_eff"].as_f64().unwrap();
        let (hb, lay) = (hub(u_eff, 0.), layout(l, false));
        let n = json_matrix(&d["n"]);
        let stored_h = json_matrix(&d["direction"]);
        let random_h = hermitian_direction(lay.size(), &mut rng);
        // purely imaginary Hermitian direction: V^T and conj(V) flip its sign
        let a = DMatrix::from_fn(lay.size(), lay.size(), |_, _| C64::new(rng.next(), 0.));
        let imaginary = (&a - a.transpose()) * C64::new(0., 1.);
        for (label, h) in [
            ("stored", stored_h),
            ("random", random_h),
            ("imaginary", imaginary),
        ] {
            let v = potential(&eval(&hb, lay, &n));
            let fd = finite_difference(&hb, lay, &n, &h);
            let tr = tr_vh(&v, &h);
            assert!((fd - tr).abs() < 1e-9, "{name} {label}: fd {fd} tr {tr}");
            // The model's own first-order helper agrees with the test-local trace.
            let ev = eval(&hb, lay, &n);
            let first = ev.potential.first_order(&flat(&h)).unwrap();
            assert!((first - tr).abs() < 1e-13);
            // Negative controls must fail by far more than the agreement above.
            assert!(
                (fd - tr_vh(&v.transpose(), &h)).abs() > 1e-3,
                "{name} {label}: V^T"
            );
            assert!(
                (fd - tr_vh(&v.conjugate(), &h)).abs() > 1e-3,
                "{name} {label}: conj V"
            );
            assert!((fd - 1.01 * tr).abs() > 1e-4, "{name} {label}: 1.01 V");
        }
        let stored_fd = d["fd_richardson"].as_f64().unwrap();
        let stored_dir = json_matrix(&d["direction"]);
        let model_fd = finite_difference(&hb, lay, &n, &stored_dir);
        assert!(
            (model_fd - stored_fd).abs() < 1e-9,
            "stored Richardson anchor"
        );
    }
}

#[test]
fn a_pure_state_is_an_eigenvector_of_v_and_not_of_its_transpose() {
    // n_ij = c_i conj(c_j) = <i|rho|j> for rho = |c><c|; V = U/2 (1 - 2 |c><c|), so V c = -U/2 c.
    let mut rng = Lcg(37);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let d = lay.size();
        let c: Vec<C64> = (0..d).map(|_| C64::new(rng.next(), rng.next())).collect();
        let norm = c.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let c = nalgebra::DVector::from_iterator(d, c.into_iter().map(|x| x / norm));
        let n = DMatrix::from_fn(d, d, |i, j| c[i] * c[j].conj());
        let u = 3.0;
        let e = eval(&hub(u, 0.), lay, &n);
        let v = potential(&e);
        assert!(
            e.energy.abs() < 1e-13,
            "a pure state is idempotent: E_U = 0"
        );
        assert!((&v * &c + &c * C64::new(0.5 * u, 0.)).norm() < 1e-13);
        assert!((v.transpose() * &c + &c * C64::new(0.5 * u, 0.)).norm() > 1e-2);
        // The block constructor uses the same orientation: block (s, s') holds <a s|rho|b s'>.
        let p = lay.width();
        let block = |s: usize, t: usize| -> Vec<C64> {
            (0..p * p)
                .map(|k| n[(s * p + k / p, t * p + k % p)])
                .collect()
        };
        let (uu, ud, du, dd) = (block(0, 0), block(0, 1), block(1, 0), block(1, 1));
        let built = Occupation::from_blocks(&context(), lay, [&uu, &ud, &du, &dd]).unwrap();
        assert_eq!(built.as_slice(), &flat(&n)[..]);
        assert_eq!(built.entry(1, p + 2), n[(1, p + 2)]);
    }
}

// ---------------------------------------------------------------- the collinear model

#[test]
fn a_block_diagonal_occupation_reproduces_the_collinear_model() {
    let mut rng = Lcg(41);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let p = lay.width();
        for (u, j) in [(3.7, 0.), (4.2, 0.9)] {
            let hb = hub(u, j);
            let (up, _, _) = random_state(p, &mut rng);
            let (dn, _, _) = random_state(p, &mut rng);
            let mut n = DMatrix::<C64>::zeros(2 * p, 2 * p);
            n.view_mut((0, 0), (p, p)).copy_from(&up);
            n.view_mut((p, p), (p, p)).copy_from(&dn);
            let e = eval(&hb, lay, &n);
            let (fu, fd) = (flat(&up), flat(&dn));
            let c = hubbard::evaluate(
                &context(),
                &hb,
                l,
                Spin::Unrestricted,
                &[fu.as_slice(), fd.as_slice()],
            )
            .unwrap();
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.);
            // The spinor sums the same two blocks in the same order and adds exact zeros.
            assert_eq!(e.energy.to_bits(), c.energy.to_bits());
            assert!(rel(e.interaction, c.interaction) < 1e-13);
            assert!(rel(e.double_counting, c.double_counting) < 1e-13);
            let v = potential(&e);
            for (s, block) in c.potentials.iter().enumerate() {
                for a in 0..p {
                    for b in 0..p {
                        // Bit for bit: the same expression on the same Hermitian part.
                        assert_eq!(v[(s * p + a, s * p + b)], block.as_slice()[a * p + b]);
                    }
                }
            }
            for a in 0..p {
                for b in 0..p {
                    assert_eq!(v[(a, p + b)], C64::default());
                    assert_eq!(v[(p + a, b)], C64::default());
                }
            }
            // A spin-flip block is not decoration: the collinear model cannot see it.
            let mut flipped = n.clone();
            flipped[(0, p)] = C64::new(0.1, 0.05);
            flipped[(p, 0)] = C64::new(0.1, -0.05);
            assert!((eval(&hb, lay, &flipped).energy - e.energy).abs() > 1e-4);
        }
    }
}

#[test]
fn the_double_counting_uses_the_spin_vector_and_is_rotation_invariant() {
    let mut rng = Lcg(43);
    let mut control_shift = 0f64;
    for (l, real) in CASES {
        let lay = layout(l, real);
        let p = lay.width();
        let (hb, (u, j)) = (hub(4.2, 0.9), (4.2, 0.9));
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let e = eval(&hb, lay, &n);
        let big_n: f64 = (0..2 * p).map(|a| n[(a, a)].re).sum();
        let m = pauli(p, &n);
        let m2: f64 = m.iter().map(|x| x * x).sum();
        let want_dc =
            0.5 * u * big_n * (big_n - 1.) - 0.5 * j * (0.5 * (big_n * big_n + m2) - big_n);
        assert!((e.double_counting - want_dc).abs() < 1e-12);
        assert!((e.interaction - e.double_counting - e.energy).abs() < 1e-12);
        for (got, want) in e.spin_vector.iter().zip(m) {
            assert!((got - want).abs() < 1e-13);
        }
        let w = spin_rotation(p, &mut rng);
        let r = eval(&hb, lay, &(&w * &n * w.adjoint()));
        assert!((r.double_counting - e.double_counting).abs() < 1e-12);
        assert!((r.interaction - e.interaction).abs() < 1e-12);
        // Control: a z-only moment, the collinear reading, changes under the rotation.
        let mz = pauli(p, &(&w * &n * w.adjoint()))[2];
        control_shift = control_shift.max((mz * mz - m[2] * m[2]).abs());
    }
    assert!(
        control_shift > 1e-3,
        "m_z^2 must not be rotation invariant: {control_shift}"
    );
}

// ---------------------------------------------------------------- the j-resolved atomic limit

/// `(P_{l+1/2}, P_{l-1/2})` from L.S: eigenvalues l/2 and -(l+1)/2.
fn j_projectors(l: usize, real: bool) -> (DMatrix<C64>, DMatrix<C64>) {
    let ls = ls_matrix(l, real);
    let d = ls.nrows();
    let one = DMatrix::<C64>::identity(d, d);
    let lf = l as f64;
    let hi = (&ls + &one * C64::new(0.5 * (lf + 1.), 0.)) * C64::new(1. / (lf + 0.5), 0.);
    let lo = &one - &hi;
    (hi, lo)
}

#[test]
fn the_j_projectors_from_the_angular_operator_are_the_frozen_ones() {
    let doc = freeze();
    for (name, l) in [("d", 2usize), ("f", 3)] {
        let (hi, lo) = j_projectors(l, false);
        let d = hi.nrows();
        assert!(max_abs(&(&hi * &hi - &hi)) < 1e-14 && max_abs(&(&lo * &lo - &lo)) < 1e-14);
        assert!((hi.trace().re - (2 * l + 2) as f64).abs() < 1e-13);
        assert!((lo.trace().re - (2 * l) as f64).abs() < 1e-13);
        let frozen = &doc[name]["so_atom"]["projectors"];
        for (key, mine) in [("j=l+1/2", &hi), ("j=l-1/2", &lo)] {
            let rows = frozen[key]["P_real_part"].as_array().unwrap();
            let stored = DMatrix::from_fn(d, d, |i, j| C64::new(rows[i][j].as_f64().unwrap(), 0.));
            // The frozen Clebsch-Gordan layout is the layout of angular().
            assert!(max_abs(&(mine - stored)) < 1e-14, "{name} {key}");
        }
    }
}

#[test]
fn the_atomic_limit_reproduces_the_exact_fractions_in_both_bases() {
    let doc = freeze();
    let hb = hub(1.0, 0.);
    for (name, l) in [("d", 2usize), ("f", 3)] {
        for real in [false, true] {
            let (hi, lo) = j_projectors(l, real);
            let lay = layout(l, real);
            let p = lay.width();
            let mut control_moved = false;
            for case in doc[name]["so_atom"]["cases"].as_array().unwrap() {
                let occupations = case["occupation_per_state"].as_object().unwrap();
                let f = |key: &str| occupations.get(key).map_or(0., fraction);
                let (f_hi, f_lo) = (f("j=l+1/2"), f("j=l-1/2"));
                let n = &hi * C64::new(f_hi, 0.) + &lo * C64::new(f_lo, 0.);
                let e = eval(&hb, lay, &n);
                let exact = fraction(&case["E_U_over_U_eff_exact"]);
                assert!(
                    (e.energy - exact).abs() < 1e-13,
                    "{name} real={real} {case}"
                );
                let v = &hi * C64::new(0.5 - f_hi, 0.) + &lo * C64::new(0.5 - f_lo, 0.);
                assert!(max_abs(&(potential(&e) - v)) < 1e-13);
                let mut dropped = n.clone();
                for a in 0..p {
                    for b in 0..p {
                        dropped[(a, p + b)] = C64::default();
                        dropped[(p + a, b)] = C64::default();
                    }
                }
                if !real {
                    let stored = case["control_spin_flip_blocks_dropped_E_U_over_U_eff"]
                        .as_f64()
                        .unwrap();
                    assert!((eval(&hb, lay, &dropped).energy - stored).abs() < 1e-12);
                }
                control_moved |= (eval(&hb, lay, &dropped).energy - exact).abs() > 0.05;
            }
            assert!(
                control_moved,
                "a collinear model must fail some atomic case"
            );
        }
    }
}

#[test]
fn a_pure_single_state_has_no_u_energy_and_a_half_gap_in_v() {
    for (l, real) in CASES {
        let lay = layout(l, real);
        let d = lay.size();
        // The stretched state |m = l, up> is a j = l + 1/2 state in the complex basis; any
        // basis vector is a rank-1 idempotent, which is all this test needs.
        let mut n = DMatrix::<C64>::zeros(d, d);
        let k = lay.index(0, 2 * l).unwrap();
        n[(k, k)] = C64::new(1., 0.);
        let e = eval(&hub(2.0, 0.), lay, &n);
        assert!(e.energy.abs() < 1e-15);
        let v = potential(&e);
        assert!((v[(k, k)].re + 1.0).abs() < 1e-15);
        let other = lay.index(1, 0).unwrap();
        assert!((v[(other, other)].re - 1.0).abs() < 1e-15);
    }
}

// ---------------------------------------------------------------- time reversal

#[test]
fn the_time_reversal_operator_squares_to_minus_one_and_commutes_with_l_dot_s() {
    for l in 1..=3usize {
        for real in [false, true] {
            let lay = layout(l, real);
            let u = theta_matrix(lay);
            let d = lay.size();
            let id = DMatrix::<C64>::identity(d, d);
            assert!(max_abs(&(&u * u.adjoint() - &id)) < 1e-15, "unitary");
            // Theta = U K: Theta^2 = U conj(U).
            assert!(max_abs(&(&u * u.conjugate() + &id)) < 1e-15, "Theta^2 = -1");
            let ls = ls_matrix(l, real);
            let transformed = &u * ls.conjugate() * u.adjoint();
            assert!(max_abs(&(transformed - &ls)) < 1e-14, "l={l} real={real}");
            // Control: drop the orbital (-1)^m in the complex basis, the spin sign in the real
            // one. Each is the wrong antiunitary and must break the L.S symmetry.
            let p = lay.width();
            let wrong = DMatrix::from_fn(d, d, |i, j| {
                let kept = if real {
                    1.
                } else if j / p == 0 {
                    1.
                } else {
                    -1.
                };
                C64::new(kept * u[(i, j)].norm(), 0.)
            });
            let bad = &wrong * ls.conjugate() * wrong.adjoint();
            assert!(max_abs(&(bad - &ls)) > 0.1, "control l={l} real={real}");
        }
    }
}

#[test]
fn the_energy_is_time_reversal_invariant() {
    let mut rng = Lcg(53);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let o = occ(lay, &n);
        let image = spinor::time_reversed(&context(), &o).unwrap();
        let hb = hub(3.7, 0.);
        let a = spinor::evaluate(&context(), &hb, &o).unwrap();
        let b = spinor::evaluate(&context(), &hb, &image).unwrap();
        assert!((a.energy - b.energy).abs() < 1e-13);
        // Theta n Theta^-1 = U conj(n) U^dagger, and V transforms the same way.
        let u = theta_matrix(lay);
        assert!(
            max_abs(&(dm(lay.size(), image.as_slice()) - &u * n.conjugate() * u.adjoint())) < 1e-15
        );
        let tv = &u * potential(&a).conjugate() * u.adjoint();
        assert!(max_abs(&(potential(&b) - tv)) < 1e-13);
    }
}

#[test]
fn the_completion_is_idempotent_trace_preserving_and_kills_the_spin_vector() {
    let mut rng = Lcg(59);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let ctx = context();
        let (n, _, _) = random_state(lay.size(), &mut rng);
        let o = occ(lay, &n);
        assert!(
            spinor::time_reversal_defect(&o) > 0.05,
            "control: a random n breaks it"
        );
        let c = spinor::complete(&ctx, &o).unwrap();
        let again = spinor::complete(&ctx, &c).unwrap();
        assert!(c
            .as_slice()
            .iter()
            .zip(again.as_slice())
            .all(|(x, y)| (x - y).norm() < 1e-15));
        assert!((c.trace() - o.trace()).abs() < 1e-14);
        assert!(c.spin_vector().iter().all(|x| x.abs() < 1e-14));
        assert!(spinor::time_reversal_defect(&c) < 1e-15);
        spinor::require_time_reversal_invariant(&c, 1e-12).unwrap();
        let msg = spinor::require_time_reversal_invariant(&o, 1e-12).unwrap_err();
        assert!(
            matches!(&msg, Error::Invalid(m) if m.contains("time-reversal")),
            "{msg}"
        );
        // The potential of a completed n is itself time-reversal invariant; an uncompleted one is not.
        let u = theta_matrix(lay);
        let hb = hub(3.7, 0.);
        let vc = potential(&spinor::evaluate(&ctx, &hb, &c).unwrap());
        assert!(max_abs(&(&vc - &u * vc.conjugate() * u.adjoint())) < 1e-14);
        let vo = potential(&spinor::evaluate(&ctx, &hb, &o).unwrap());
        assert!(max_abs(&(&vo - &u * vo.conjugate() * u.adjoint())) > 0.05);
    }
}

#[test]
fn a_kramers_pair_is_already_complete() {
    let mut rng = Lcg(61);
    for (l, real) in CASES {
        let lay = layout(l, real);
        let (d, ctx) = (lay.size(), context());
        let u = theta_matrix(lay);
        let c = nalgebra::DVector::from_fn(d, |_, _| C64::new(rng.next(), rng.next()));
        let c = &c / C64::new(c.norm(), 0.);
        let tc = &u * c.conjugate();
        let n = DMatrix::from_fn(d, d, |i, j| c[i] * c[j].conj() + tc[i] * tc[j].conj());
        let o = occ(lay, &n);
        assert!(spinor::time_reversal_defect(&o) < 1e-15);
        let done = spinor::complete(&ctx, &o).unwrap();
        assert!(done
            .as_slice()
            .iter()
            .zip(o.as_slice())
            .all(|(x, y)| (x - y).norm() < 1e-15));
        // Theta c is orthogonal to c (Theta^2 = -1), so the pair is a rank-2 projector: E_U = 0.
        let e = spinor::evaluate(&ctx, &hub(2.0, 0.), &o).unwrap();
        assert!(e.energy.abs() < 1e-13);
    }
}

// ---------------------------------------------------------------- refusals and memory

#[test]
fn a_bad_request_raises_and_nothing_is_kept() {
    let ctx = context();
    assert!(matches!(Layout::new(4, false), Err(Error::Invalid(_))));
    let lay = layout(2, false);
    let d = lay.size();
    let good = flat(&DMatrix::<C64>::identity(d, d));
    assert!(matches!(
        Occupation::new(&ctx, lay, &good[1..]),
        Err(Error::Invalid(_))
    ));
    let mut bad = good.clone();
    bad[3] = C64::new(f64::NAN, 0.);
    assert!(matches!(
        Occupation::new(&ctx, lay, &bad),
        Err(Error::Invalid(_))
    ));
    let mut skew = good.clone();
    skew[1] = C64::new(0.2, 0.1);
    skew[d] = C64::new(0.2, 0.1); // not the conjugate
    assert!(
        matches!(Occupation::new(&ctx, lay, &skew), Err(Error::Invalid(m)) if m.contains("Hermitian"))
    );
    let p = lay.width();
    let short = vec![C64::default(); p * p - 1];
    let ok = vec![C64::default(); p * p];
    let r = Occupation::from_blocks(&ctx, lay, [&ok, &ok, &ok, &short]);
    assert!(matches!(r, Err(Error::Invalid(_))));
    assert_eq!(ctx.memory_used(), 0, "a refused request keeps nothing");
    // Budget: the occupation fits, the potential does not, and the failure frees what it took.
    let small = Context::new(Some(1), d * d * 16 + 100).unwrap();
    let o = Occupation::new(&small, lay, &good).unwrap();
    let used = small.memory_used();
    assert!(matches!(
        spinor::evaluate(&small, &hub(2.0, 0.), &o),
        Err(Error::Budget { .. })
    ));
    assert_eq!(small.memory_used(), used);
    assert!(matches!(
        Occupation::new(&Context::new(Some(1), 64).unwrap(), lay, &good),
        Err(Error::Budget { .. })
    ));
}

#[test]
fn zero_u_is_the_identity_and_the_hellmann_feynman_factor_survives() {
    let mut rng = Lcg(67);
    let lay = layout(2, false);
    let (n, _, _) = random_state(lay.size(), &mut rng);
    let e = eval(&hub(2.0, 2.0), lay, &n);
    assert_eq!(e.energy, 0.);
    assert!(potential(&e).iter().all(|x| *x == C64::default()));
    let tr = (0..lay.size()).map(|a| n[(a, a)].re).sum::<f64>();
    let sq = (&n * &n).trace().re;
    assert!((e.per_u_eff - 0.5 * (tr - sq)).abs() < 1e-13);
    // dE/dU = E / U_eff at fixed n.
    let (a, b) = (eval(&hub(3.0, 0.), lay, &n), eval(&hub(3.001, 0.), lay, &n));
    assert!(((b.energy - a.energy) / 0.001 - a.per_u_eff).abs() < 1e-10);
    let _: &Potential = &a.potential;
}

// ---------------------------------------------------------------- the harmonics behind the layout

#[test]
fn the_real_basis_of_angular_is_the_real_harmonic_of_math_angular() {
    // `angular` writes S_k = sum u[k, k'] Y_k' (complex Y_lm, Condon-Shortley) and
    // L.S(real) = W L.S(complex) W^dagger with W = blockdiag(conj u). The time-reversal
    // operator of the real basis is only i sigma_y K if S_k is `math::angular::cartesian`.
    let mut rng = Lcg(71);
    for l in 1..=3usize {
        let p = 2 * l + 1;
        let root = std::f64::consts::FRAC_1_SQRT_2;
        let mut u = DMatrix::<C64>::zeros(p, p);
        u[(l, l)] = C64::new(1., 0.);
        for m in 1..=l {
            let sign = if m % 2 == 0 { 1. } else { -1. };
            u[(l + m, l + m)] = C64::new(sign * root, 0.);
            u[(l + m, l - m)] = C64::new(root, 0.);
            u[(l - m, l - m)] = C64::new(0., root);
            u[(l - m, l + m)] = C64::new(0., -sign * root);
        }
        let w = DMatrix::from_fn(2 * p, 2 * p, |i, j| {
            if i / p == j / p {
                u[(i % p, j % p)].conj()
            } else {
                C64::default()
            }
        });
        // This copy of u is the one angular() uses ...
        let mapped = &w * ls_matrix(l, false) * w.adjoint();
        assert!(max_abs(&(mapped - ls_matrix(l, true))) < 1e-14, "l={l}");
        // ... and it maps the complex harmonics of math::angular onto its real ones.
        for _ in 0..6 {
            let v = [rng.next(), rng.next(), rng.next()];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dir = [v[0] / r, v[1] / r, v[2] / r];
            for k in 0..p {
                let from_complex: C64 = (0..p)
                    .map(|kp| u[(k, kp)] * angular::complex(l, kp as i32 - l as i32, dir))
                    .sum();
                let real = angular::cartesian(l, k as i32 - l as i32, dir);
                assert!(
                    (from_complex - C64::new(real, 0.)).norm() < 1e-12,
                    "l={l} k={k}"
                );
            }
        }
    }
}
