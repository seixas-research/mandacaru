//! The strain derivatives of the dense reciprocal sums (`couplings_strain`: `U`, `U_ion`, `E_ion`)
//! against central differences of the same sums over the PINNED list of integer triples of the
//! reference cell, on Si, Al2 and the skew, non-symmetric LiH cell; the point-ion constant
//! `c_ion` against its pair-sum derivative; the size of the list jump and of the table-kink floor
//! of a difference over the whole list; and the negative controls. No SCF, no crystal.
//! Needs `MANDACARU_PAW_PATH`.
//!
//! The shape tables are piecewise linear (nodes 0.01 apart), so the sum is a function with a kink
//! at every vector that crosses a node, and a difference over the whole list of 10^4 to 10^5
//! vectors cannot reach the plan's 1e-7 for the dense sums (measured below: 1e-5 on LiH at any
//! step). The gate is therefore a difference over the vectors that do not straddle a node at
//! the step, fed through the SAME accumulator as the production sum; the whole-list difference
//! is reported as the kink floor.
mod strain_support;

use mandacaru_core::periodic::setup::{
    electrostatics::{couplings, max_wavenumber},
    gradients::ion_constants_strain,
    ions::ion_constants,
    shapes::{Shapes, COMPENSATION_CUTOFF},
    strain_sums::{couplings_strain, CouplingsStrain, CouplingsStrainSum},
};
use num_complex::Complex64;
use strain_support::{pinned, *};

/// The plan's gate for the dense sums, relative to the term's maximum.
const GATE: f64 = 1e-7;
/// The central-difference step of the whole-list difference that reports the kink floor.
const STEP: f64 = 5e-6;
/// The step of the sums that carry no table (`E_ion`): round-off, not a kink, limits it.
const SMOOTH_STEP: f64 = 1e-5;

struct Fixture {
    cell: Cell,
    shapes: Shapes,
    list: Vec<[i64; 3]>,
}
fn fixture(name: &'static str) -> Fixture {
    let ctx = context();
    let cell = Cell::load(&ctx, name);
    let q_max = max_wavenumber(&ctx, &cell.grid).unwrap();
    let shapes = Shapes::new(&ctx, &cell.sites(), q_max).unwrap();
    let list = pinned::list(&cell.sites(), COMPENSATION_CUTOFF);
    Fixture { cell, shapes, list }
}

/// The triples whose wave number does not straddle a table node under any of the nine strain
/// entries `+-h`.
fn without_straddles(f: &Fixture, h: f64) -> Vec<[i64; 3]> {
    let norm = |g: [f64; 3]| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let mut keep = vec![true; f.list.len()];
    for p in 0..3 {
        for q in 0..3 {
            let side = |delta: f64| {
                let strained = f.cell.strained(entry(p, q, delta));
                pinned::vectors(&strained.sites(), &f.list)
            };
            let (plus, minus) = (side(h), side(-h));
            for (i, flag) in keep.iter_mut().enumerate() {
                *flag &= !straddles(norm(plus[i]), norm(minus[i]));
            }
        }
    }
    f.list
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(n, _)| *n)
        .collect()
}

/// `couplings_strain` summed over exactly `list` at the reference cell.
fn analytic_over(f: &Fixture, list: &[[i64; 3]]) -> CouplingsStrain {
    let ctx = context();
    let sites = f.cell.sites();
    let mut sum = CouplingsStrainSum::new(&ctx, &f.shapes, &sites).unwrap();
    let vectors = pinned::vectors(&sites, list);
    for chunk in vectors.chunks(sum.block()) {
        sum.add(chunk).unwrap();
    }
    sum.finish()
}

/// Central difference over `list` along `eps_pq`, term by term.
fn difference(f: &Fixture, list: &[[i64; 3]], p: usize, q: usize, h: f64) -> pinned::Sums {
    let (plus, minus) = (
        f.cell.strained(entry(p, q, h)),
        f.cell.strained(entry(p, q, -h)),
    );
    pinned::derivative(&f.shapes, &plus.sites(), &minus.sites(), list, h)
}

/// The worst miss of `got` against the oracle's nine differences, relative to the largest
/// entry of the whole nine-component array (the plan's "term maximum"): `(dU, dU_ion, dE_ion,
/// max dU)`. The controls below make sure a defect in a small component is not hidden.
fn misses(got: &CouplingsStrain, oracle: &[pinned::Sums]) -> (f64, f64, f64, f64) {
    let big_u = oracle.iter().map(|s| maximum(&s.u)).fold(0., f64::max);
    let big_ion = oracle.iter().map(|s| maximum(&s.u_ion)).fold(0., f64::max);
    let big_e = oracle.iter().map(|s| s.e_ion.abs()).fold(0., f64::max);
    let (mut mu, mut mi, mut me) = (0f64, 0f64, 0f64);
    for (pq, s) in oracle.iter().enumerate() {
        let (p, q) = (pq / 3, pq % 3);
        mu = mu.max(worst(got.u(p, q), &s.u, &|_| false) / big_u);
        mi = mi.max(worst(got.u_ion(p, q), &s.u_ion, &|_| false) / big_ion);
        me = me.max((got.e_ion[p][q] - s.e_ion).abs() / big_e);
    }
    (mu, mi, me, big_u)
}

#[test]
fn the_pinned_oracle_is_the_plain_sum_at_the_reference_cell() {
    for name in ["si", "al2", "lih"] {
        let ctx = context();
        let f = fixture(name);
        let plain = couplings(&ctx, &f.shapes, &f.cell.sites(), COMPENSATION_CUTOFF).unwrap();
        let mine = pinned::sums(
            &f.shapes,
            &f.cell.sites(),
            &f.list,
            COMPENSATION_CUTOFF,
            true,
        );
        let miss = worst(&mine.u, plain.u.as_slice(), &|_| false) / maximum(plain.u.as_slice());
        let miss_ion = worst(&mine.u_ion, plain.u_ion.as_slice(), &|_| false)
            / maximum(plain.u_ion.as_slice());
        println!(
            "{name}: {} vectors, oracle - couplings: U {miss:.1e}, U_ion {miss_ion:.1e}",
            f.list.len()
        );
        assert!(miss < 1e-12 && miss_ion < 1e-12);
        assert!((mine.e_ion - plain.e_ion).abs() < 1e-12 * plain.e_ion);
    }
}

#[test]
fn the_dense_strain_derivative_is_the_difference_of_the_pinned_sums() {
    for name in ["si", "al2", "lih"] {
        let f = fixture(name);
        // The largest step that leaves 95% of the vectors unstraddled (round-off falls with it).
        let (step, kept) = [2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7]
            .iter()
            .map(|&h| (h, without_straddles(&f, h)))
            .find(|(_, kept)| kept.len() as f64 >= 0.95 * f.list.len() as f64)
            .expect("a step that straddles under 5% of the vectors");
        let share = kept.len() as f64 / f.list.len() as f64;
        let got = analytic_over(&f, &kept);
        let mut oracle: Vec<pinned::Sums> = (0..9)
            .map(|pq| difference(&f, &kept, pq / 3, pq % 3, step))
            .collect();
        // E_ion has no table: its own, larger step (the round-off of a smaller one dominates).
        for (pq, sums) in oracle.iter_mut().enumerate() {
            sums.e_ion = difference(&f, &kept, pq / 3, pq % 3, SMOOTH_STEP).e_ion;
        }
        let (mu, mi, me, big_u) = misses(&got, &oracle);
        println!(
            "{name}: step {step:.0e}, {} of {} vectors kept ({:.2}%); dU {mu:.2e} (max {big_u:.2e}), dU_ion {mi:.2e}, dE_ion {me:.2e}",
            kept.len(), f.list.len(), 100. * share
        );
        // The comparison covers the list: a localized defect cannot hide in the excluded few.
        assert!(share >= 0.95 && big_u > 1e-3, "{name}");
        assert!(mu < GATE && mi < GATE && me < GATE, "{name}");
        // The sums are Hermitian, so is their derivative; E_ion is symmetric in the strain.
        let rows = f.cell.channels.len();
        for pq in 0..9 {
            let u = got.u(pq / 3, pq % 3);
            for a in 0..rows {
                for b in 0..rows {
                    assert!((u[a * rows + b] - u[b * rows + a].conj()).norm() <= 1e-12 * big_u);
                }
            }
        }
        for p in 0..3 {
            for q in 0..3 {
                assert!((got.e_ion[p][q] - got.e_ion[q][p]).abs() < 1e-12 * got.e_ion[0][0].abs());
            }
        }
        // Controls, each a wrong variant of the analytic result against the same oracle.
        let bite = 100. * GATE;
        let raw = pinned::sums(
            &f.shapes,
            &f.cell.sites(),
            &kept,
            COMPENSATION_CUTOFF,
            false,
        );
        let volume_free = |pq: usize| -> Vec<Complex64> {
            // The Jacobian of 1 / Omega dropped: add back delta_pq (U - tail).
            let mut u = got.u(pq / 3, pq % 3).to_vec();
            if pq % 4 == 0 {
                for (x, r) in u.iter_mut().zip(&raw.u) {
                    *x += r;
                }
            }
            u
        };
        let (mut transposed, mut no_jacobian, mut flipped) = (0f64, 0f64, 0f64);
        for pq in 0..9 {
            let (p, q) = (pq / 3, pq % 3);
            let want = &oracle[pq].u;
            let scale = big_u;
            transposed = transposed.max(worst(got.u(q, p), want, &|_| false) / scale);
            no_jacobian = no_jacobian.max(worst(&volume_free(pq), want, &|_| false) / scale);
            let negated: Vec<Complex64> = got.u(p, q).iter().map(|x| -x).collect();
            flipped = flipped.max(worst(&negated, want, &|_| false) / scale);
        }
        println!("{name}: controls: transposed {transposed:.2e}, no Jacobian {no_jacobian:.2e}, sign {flipped:.2e}");
        assert!(
            transposed > bite && no_jacobian > bite && flipped > bite,
            "{name}"
        );
    }
}

#[test]
fn the_point_ion_energy_strain_equals_the_difference_of_the_pair_and_reciprocal_sums() {
    // E-7 and E-5 together: c_ion + e_ion, the Gaussian-split point-ion energy. The reciprocal
    // part has no table, so the whole list is used, at the step of the smooth sums.
    let h = 1e-5;
    for name in ["si", "al2", "lih"] {
        let ctx = context();
        let f = fixture(name);
        let got = couplings_strain(&ctx, &f.shapes, &f.cell.sites(), COMPENSATION_CUTOFF).unwrap();
        let pairs = ion_constants_strain(&f.cell.sites()).unwrap();
        let (mut miss, mut scale, mut flipped) = (0f64, 0f64, 0f64);
        for p in 0..3 {
            for q in 0..3 {
                let total = |delta: f64| {
                    let strained = f.cell.strained(entry(p, q, delta));
                    let sites = strained.sites();
                    let sum = pinned::sums(&f.shapes, &sites, &f.list, COMPENSATION_CUTOFF, true);
                    sum.e_ion + ion_constants(&sites).unwrap()
                };
                let numeric = (total(h) - total(-h)) / (2. * h);
                let analytic = got.e_ion[p][q] + pairs[p][q];
                miss = miss.max((analytic - numeric).abs());
                flipped = flipped.max((-analytic - numeric).abs());
                scale = scale.max(analytic.abs());
            }
        }
        println!("{name}: d(e_ion + c_ion) {miss:.2e} of {scale:.2e}");
        assert!(scale > 1e-4 && miss < 1e-8 * scale.max(1.), "{name}");
        assert!(flipped > 100. * 1e-8 * scale, "{name}");
    }
}

#[test]
fn the_whole_list_difference_reports_the_table_kink_floor_and_the_list_jump() {
    for name in ["si", "al2", "lih"] {
        let ctx = context();
        let f = fixture(name);
        let got = couplings_strain(&ctx, &f.shapes, &f.cell.sites(), COMPENSATION_CUTOFF).unwrap();
        // The kink floor: the same difference over the whole pinned list, at the three steps of
        // the plan's measurement and at the gated one.
        let mut floors = Vec::new();
        for h in [1e-5, 5e-5, 2e-4, STEP] {
            let oracle: Vec<pinned::Sums> = (0..9)
                .map(|pq| difference(&f, &f.list, pq / 3, pq % 3, h))
                .collect();
            let (mu, _, _, _) = misses(&got, &oracle);
            floors.push(mu);
            println!("{name}: whole-list difference at h = {h:.0e}: dU {mu:.2e}");
        }
        // Grossly right at every step (a wrong index or Jacobian is of order one), but not 1e-7.
        assert!(floors.iter().all(|m| *m < 1e-2), "{name}: {floors:?}");
        // The jump of re-enumerating the ball at a strained cell (the plain `couplings`): the
        // shells of a lattice are discrete, so the membership changes only at some strains.
        let mut changed = 0;
        let mut worst_jump = 0f64;
        for h in [2e-3, -2e-3, 5e-3, -5e-3, 2e-2, -2e-2] {
            let strained = f.cell.strained(entry(0, 0, h));
            let moved = pinned::list(&strained.sites(), COMPENSATION_CUTOFF);
            if moved == f.list {
                continue;
            }
            changed += 1;
            let fixed = pinned::sums(
                &f.shapes,
                &strained.sites(),
                &f.list,
                COMPENSATION_CUTOFF,
                true,
            );
            let plain = couplings(&ctx, &f.shapes, &strained.sites(), COMPENSATION_CUTOFF).unwrap();
            let jump = worst(plain.u.as_slice(), &fixed.u, &|_| false) / maximum(&fixed.u);
            worst_jump = worst_jump.max(jump / h.abs());
            println!(
                "{name}: ball re-enumerated at eps_xx = {h:+.0e}: {} vs {} vectors, U jumps by {jump:.2e} (relative), i.e. {:.2e} as a derivative",
                moved.len(), f.list.len(), jump / h.abs()
            );
        }
        assert!(
            changed > 0,
            "{name}: no strain changed the membership, the control would be vacuous"
        );
    }
}
