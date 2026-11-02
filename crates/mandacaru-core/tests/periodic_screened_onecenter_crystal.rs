//! The one-center screened terms inside the crystal's exchange (W-E4P / P2): the operators and
//! the energy `Crystal::hybrid_exchange` adds to the grid pair and its compensation charges,
//! against the older `onecenter_hybrid::Hybrid` evaluator fed with the projected density the
//! test forms itself from the stored projections, on the phase-5 fixtures (restricted, unrestricted
//! with the spin-scaling remainder, spinor), the limits of the model, and the negative controls.
//!
//! The states are the fixtures' mechanics densities (`periodic_screened_augmentation_support`).
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_screened_augmentation_support;
mod periodic_screened_onecenter_support;
mod setup_support;

use mandacaru_core::{
    paw::{
        onecenter::Basis,
        onecenter_hybrid::{Hybrid, HybridAtom},
        pair_tables::{layout, Projector},
        screened_atom::ScreenedAtom,
    },
    periodic::{
        screened_augmentation::Augmentation, screened_exchange::exchange_operators_with, Crystal,
    },
    quantum::C64,
    Context,
};
use periodic_screened_augmentation_support::{Rig, CASES, GB};
use periodic_screened_onecenter_support::*;
use setup_support::{library_of, symbol};

/// Every atom of the fixture: its screened species and projector layout.
fn species(ctx: &Context, name: &str) -> Vec<(ScreenedAtom, Vec<Projector>)> {
    let fixture = periodic_screened_augmentation_support::fixture(name);
    fixture["provenance"]["geometry"]["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|z| {
            terms(
                ctx,
                library_of(name),
                symbol(z.as_u64().unwrap()),
                Basis::Raw,
            )
        })
        .collect()
}

/// The older evaluator over every atom of the fixture; `zero_v` drops the frozen potential.
fn older(atoms: &[(ScreenedAtom, Vec<Projector>)], p: usize, zero_v: bool) -> Hybrid {
    let mut offset = 0;
    let mut spheres = Vec::new();
    for (atom, projectors) in atoms {
        let n = projectors.len();
        spheres.push(HybridAtom {
            members: (offset..offset + n).collect(),
            tensor: rotated_tensor(atom, projectors),
            v: if zero_v {
                atom.v.map(|_| 0.)
            } else {
                atom.v.clone()
            },
        });
        offset += n;
    }
    let frozen = atoms.iter().map(|(a, _)| a.constant).sum();
    Hybrid::new(spheres, p, FRACTION, frozen).unwrap()
}

/// `P_s = sum_k w_k C(k)^+ D_s(k) C(k)` of a collinear crystal, one matrix per spin channel.
fn projected(c: &Crystal, d: &[Matrix]) -> Vec<Matrix> {
    let nk = c.points.len();
    (0..c.spins)
        .map(|s| {
            let mut total = Matrix::zeros(c.p, c.p);
            for (k, point) in c.points.iter().enumerate() {
                total += point.projections.adjoint()
                    * &d[s * nk + k]
                    * &point.projections
                    * C64::new(point.weight, 0.);
            }
            total
        })
        .collect()
}

/// The spin-scaling remainder of every sphere for the spin densities `up` and `down`: its energy
/// (the fraction not included) and the `p x p` derivative of each spin.
fn polarization(
    ctx: &Context,
    atoms: &[(ScreenedAtom, Vec<Projector>)],
    up: &Matrix,
    down: &Matrix,
) -> (f64, [Matrix; 2]) {
    let p = up.nrows();
    let (mut energy, mut ops) = (0., [Matrix::zeros(p, p), Matrix::zeros(p, p)]);
    let mut at = 0;
    for (atom, projectors) in atoms {
        let n = projectors.len();
        let part = |m: &Matrix| m.view((at, at), (n, n)).map(|z| z.re).into_owned();
        let pol = atom
            .polarization
            .evaluate(ctx, &part(up), &part(down))
            .unwrap();
        energy += pol.energy;
        for (op, d) in ops.iter_mut().zip([pol.up, pol.down]) {
            let mut view = op.view_mut((at, at), (n, n));
            view += d.map(|x| C64::new(x, 0.));
        }
        at += n;
    }
    (energy, ops)
}

fn largest(m: &Matrix) -> f64 {
    m.iter().map(|v| v.norm()).fold(0., f64::max)
}

#[test]
fn the_crystals_one_center_exchange_is_the_older_evaluator_on_the_projected_density() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let nk = c.points.len();
        let aug = Augmentation::new(c, rig.geometry.sites(), &rig.shapes).unwrap();
        let base = exchange_operators_with(
            c,
            &rig.d,
            periodic_screened_augmentation_support::screening(0.11),
            GB,
            Some(&aug),
        )
        .unwrap();
        let full = c.hybrid_exchange(&rig.d, GB).unwrap();
        let atoms = species(&ctx, fixture);
        let older = older(&atoms, c.p, false);
        let p = projected(c, &rig.d);
        // The reference operators in the convention of K: O_h = -fraction O.
        let (mut energy, mut operators): (f64, Vec<Matrix>) = if c.spins == 1 {
            let t = older.evaluate(&ctx, &[&slice(&p[0])]).unwrap();
            (t.energy, vec![from_flat(c.p, t.operators[0].as_slice())])
        } else {
            let t = older
                .evaluate(&ctx, &[&slice(&p[0]), &slice(&p[1])])
                .unwrap();
            (
                t.energy,
                (0..2)
                    .map(|s| from_flat(c.p, t.operators[s].as_slice()))
                    .collect(),
            )
        };
        for o in &mut operators {
            *o *= C64::new(-1. / FRACTION, 0.);
        }
        let mut dropped_polarization = None;
        if c.spins == 2 {
            let (e, extra) = polarization(&ctx, &atoms, &p[0], &p[1]);
            dropped_polarization = Some(FRACTION * e);
            energy -= FRACTION * e;
            for (o, x) in operators.iter_mut().zip(extra) {
                *o += x;
            }
        }
        let mut worst = 0f64;
        for (s, operator) in operators.iter().enumerate() {
            for (k, point) in c.points.iter().enumerate() {
                let added = &full.operators[s * nk + k] - &base.operators[s * nk + k];
                let want = &point.projections * operator * point.projections.adjoint();
                worst = worst.max(largest(&(added - want)));
            }
        }
        let energy_gap = (full.energy - base.energy - energy).abs();
        println!(
            "{fixture}: one-center operator {worst:.2e}, energy {:.8} ({energy_gap:.2e})",
            energy
        );
        assert!(worst < 1e-11, "{fixture}");
        assert!(energy_gap < 1e-11 * energy.abs().max(1.), "{fixture}");

        // Negative controls: each wrong convention must miss the same gate by far more than 100x.
        let gate = 1e-11;
        let miss = |wrong_operators: &[Matrix], wrong_energy: f64| -> (f64, f64) {
            let mut worst = 0f64;
            for (s, operator) in wrong_operators.iter().enumerate() {
                for (k, point) in c.points.iter().enumerate() {
                    let added = &full.operators[s * nk + k] - &base.operators[s * nk + k];
                    let want = &point.projections * operator * point.projections.adjoint();
                    worst = worst.max(largest(&(added - want)));
                }
            }
            (worst, (full.energy - base.energy - wrong_energy).abs())
        };
        let constant: f64 = atoms.iter().map(|(a, _)| a.constant).sum();
        // The constant dropped.
        let (_, gap) = miss(&operators, energy + FRACTION * constant);
        println!("  control, constant dropped: energy off by {gap:.2e}");
        assert!(gap > 100. * gate);
        // The constant counted once per k-point.
        if nk > 1 {
            let (_, gap) = miss(&operators, energy - FRACTION * constant * (nk - 1) as f64);
            println!("  control, constant per k: energy off by {gap:.2e}");
            assert!(gap > 100. * gate);
        }
        // The spin-scaling remainder dropped (an open shell only).
        if let Some(piece) = dropped_polarization {
            let (_, gap) = miss(&operators, energy + piece);
            println!("  control, polarization dropped: energy off by {gap:.2e}");
            assert!(gap > 100. * gate);
        }
        // The frozen potential V dropped (restricted and unrestricted alike).
        let no_v = self::older(&atoms, c.p, true);
        let (e_no_v, ops_no_v): (f64, Vec<Matrix>) = if c.spins == 1 {
            let t = no_v.evaluate(&ctx, &[&slice(&p[0])]).unwrap();
            (
                t.energy,
                vec![from_flat(c.p, t.operators[0].as_slice()) * C64::new(-1. / FRACTION, 0.)],
            )
        } else {
            let t = no_v
                .evaluate(&ctx, &[&slice(&p[0]), &slice(&p[1])])
                .unwrap();
            (
                t.energy,
                (0..2)
                    .map(|s| {
                        from_flat(c.p, t.operators[s].as_slice()) * C64::new(-1. / FRACTION, 0.)
                    })
                    .collect(),
            )
        };
        let (worst_v, gap_v) = miss(&ops_no_v, e_no_v);
        println!("  control, V dropped: operator off by {worst_v:.2e}, energy by {gap_v:.2e}");
        assert!(worst_v > 100. * gate && gap_v > 100. * gate);
        // The spin factor: a restricted total density used as if it were one spin.
        if c.spins == 1 {
            let t = older
                .evaluate(&ctx, &[&slice(&p[0]), &slice(&p[0])])
                .unwrap();
            let wrong = from_flat(c.p, t.operators[0].as_slice()) * C64::new(-1. / FRACTION, 0.);
            let (worst_s, gap_s) = miss(&[wrong], t.energy);
            println!(
                "  control, spin factor: operator off by {worst_s:.2e}, energy by {gap_s:.2e}"
            );
            assert!(worst_s > 100. * gate && gap_s > 100. * gate);
        }
        // The projections transposed (conjugated), where they are complex.
        let complex = c
            .points
            .iter()
            .any(|pt| pt.projections.iter().any(|z| z.im.abs() > 1e-9));
        if complex {
            let mut worst_t = 0f64;
            for (s, operator) in operators.iter().enumerate() {
                for (k, point) in c.points.iter().enumerate() {
                    let added = &full.operators[s * nk + k] - &base.operators[s * nk + k];
                    let conj = point.projections.conjugate();
                    worst_t = worst_t.max(largest(&(added - &conj * operator * conj.adjoint())));
                }
            }
            println!("  control, conjugated projections: operator off by {worst_t:.2e}");
            assert!(worst_t > 100. * gate);
        }
    }
}

fn from_flat(n: usize, flat: &[C64]) -> Matrix {
    Matrix::from_fn(n, n, |i, j| flat[i * n + j])
}

#[test]
fn the_spinor_crystals_one_center_exchange_is_the_older_evaluator_on_the_projected_density() {
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let rig = Rig::new("si_so.json", true, &[8]);
    let c = &rig.crystal;
    assert!(c.spinor);
    let aug = Augmentation::new(c, rig.geometry.sites(), &rig.shapes).unwrap();
    let base = exchange_operators_with(
        c,
        &rig.d,
        periodic_screened_augmentation_support::screening(0.11),
        GB,
        Some(&aug),
    )
    .unwrap();
    let full = c.hybrid_exchange(&rig.d, GB).unwrap();
    let atoms = species(&ctx, "si_so.json");
    let older = older(&atoms, c.p, false);
    let (m, p) = (c.m, c.p);
    let mut block = vec![vec![Matrix::zeros(p, p); 2]; 2];
    for (k, point) in c.points.iter().enumerate() {
        for s in 0..2 {
            for t in 0..2 {
                block[s][t] += point.projections.adjoint()
                    * rig.d[k].view((s * m, t * m), (m, m))
                    * &point.projections
                    * C64::new(point.weight, 0.);
            }
        }
    }
    let flat: Vec<Vec<Vec<C64>>> = block
        .iter()
        .map(|r| r.iter().map(slice).collect())
        .collect();
    let t = older
        .evaluate_spinor(
            &ctx,
            [[&flat[0][0], &flat[0][1]], [&flat[1][0], &flat[1][1]]],
        )
        .unwrap();
    let operator = Matrix::from_fn(2 * p, 2 * p, |i, j| {
        t.operators[0].as_slice()[i * 2 * p + j]
    }) * C64::new(-1. / FRACTION, 0.);
    let mut worst = 0f64;
    for (k, point) in c.points.iter().enumerate() {
        let added = &full.operators[k] - &base.operators[k];
        let mut want = Matrix::zeros(2 * m, 2 * m);
        for s in 0..2 {
            for u in 0..2 {
                let part = &point.projections
                    * operator.view((s * p, u * p), (p, p))
                    * point.projections.adjoint();
                want.view_mut((s * m, u * m), (m, m)).copy_from(&part);
            }
        }
        worst = worst.max(largest(&(added - want)));
    }
    let gap = (full.energy - base.energy - t.energy).abs();
    println!(
        "si_so: one-center operator {worst:.2e}, energy {:.8} ({gap:.2e})",
        t.energy
    );
    assert!(worst < 1e-11 && gap < 1e-11 * t.energy.abs().max(1.));
}

#[test]
fn the_screened_tensor_tends_to_the_bare_one_as_the_screening_vanishes() {
    use mandacaru_core::paw::{
        pair_tables::{pair_tensor, Tables},
        screened_atom::screened,
        species_tables::SpeciesTables,
    };
    let ctx = Context::new(Some(2), 768 << 20).unwrap();
    let record = read(&ctx, "lda-sr", "Si");
    let tables = SpeciesTables::from_record(&ctx, &record, Basis::Raw).unwrap();
    let view = tables.view();
    let projectors = layout(&view);
    let bare = pair_tensor(&Tables::new(&view, &projectors).unwrap(), None).unwrap();
    let mut previous = f64::INFINITY;
    for omega in [1e-3, 1e-5, 1e-7] {
        let atom = screened(&ctx, &view, &projectors, omega).unwrap();
        let mut rebuilt = nalgebra::DMatrix::<f64>::zeros(bare.nrows(), bare.ncols());
        for (f, w) in atom.factors.iter().zip(&atom.weights) {
            let v = nalgebra::DVector::from_iterator(f.len(), f.iter().copied());
            rebuilt += &v * v.transpose() * *w;
        }
        let gap = (&rebuilt - &bare).norm() / bare.norm();
        println!("omega {omega:e}: |W^SR - W| / |W| = {gap:.2e}");
        assert!(
            gap < previous,
            "the screened tensor must approach the bare one"
        );
        previous = gap;
    }
    assert!(previous < 1e-4);
}
