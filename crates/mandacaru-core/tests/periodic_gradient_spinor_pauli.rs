//! CS-3: the Pauli algebra of the four-channel local term, at the matrix level (no SCF).
//!
//! The local potential of a non-collinear spinor is `sum_a sigma_a x V^a`; its force contracts the
//! row-center derivatives `G^a` with the spatial combinations `component_a(D)` of the spinor
//! density (`models::noncollinear::component`, the definition `spinor::density` scatters the four
//! fields with). `gradient::spinor::pauli_sum` is the block assembly the provider hands the
//! engine, so `Re tr[D pauli_sum(G)]` must equal `sum_a Re tr[component_a(D) G^a]` for ANY
//! Hermitian `D` and ANY `G^a`: here random ones with every spin block populated, so a wrong
//! channel, a transposed block or a dropped transverse block is a large miss.
use mandacaru_core::models::noncollinear::component;
use mandacaru_core::{
    periodic::{
        gradient::{
            spinor::{doubled, doubled_layout, pauli_sum},
            Layout, Matrix, Request,
        },
    },
    quantum::C64,
};

const M: usize = 5;
const GATE: f64 = 1e-11;

/// A congruential stream in `[-0.5, 0.5)`.
fn stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }
}

fn general(next: &mut impl FnMut() -> f64, n: usize) -> Matrix {
    Matrix::from_fn(n, n, |_, _| C64::new(next(), next()))
}

fn density(next: &mut impl FnMut() -> f64) -> Matrix {
    let raw = general(next, 2 * M);
    (&raw + raw.adjoint()) * C64::new(0.5, 0.)
}

/// `Re tr[D H]`.
fn trace(d: &Matrix, h: &Matrix) -> f64 {
    (d * h).trace().re
}

/// `sum_a Re tr[component_a(D) G^a]`, the definition side.
fn by_components(d: &Matrix, g: &[Matrix]) -> f64 {
    g.iter()
        .enumerate()
        .map(|(a, g)| (component(d, M, a) * g).trace().re)
        .sum()
}

fn fixture() -> (Matrix, Vec<Matrix>) {
    let mut next = stream(0x9E3779B97F4A7C15);
    let d = density(&mut next);
    let g = (0..4).map(|_| general(&mut next, M)).collect();
    (d, g)
}

#[test]
fn the_pauli_sum_contracts_as_the_component_definition() {
    let (d, g) = fixture();
    let engine = trace(&d, &pauli_sum(&g).unwrap());
    let definition = by_components(&d, &g);
    eprintln!("pauli_sum {engine:+.12e}, components {definition:+.12e}");
    assert!(
        (engine - definition).abs() < GATE,
        "{:e}",
        engine - definition
    );
    // Every channel carries signal: the populated spin blocks are not vacuous.
    for a in 0..4 {
        let part = (component(&d, M, a) * &g[a]).trace().re;
        assert!(part.abs() > 1e-2, "channel {a} carries {part:e}");
    }
}

/// Each wrong variant of the block assembly misses the definition by more than 100 gates.
#[test]
fn the_wrong_pauli_channels_are_far_from_the_definition() {
    let (d, g) = fixture();
    let definition = by_components(&d, &g);
    let bite = |variant: Matrix| (trace(&d, &variant) - definition).abs() / GATE;
    let swapped = {
        let mut s = g.clone();
        s.swap(1, 2);
        pauli_sum(&s).unwrap()
    };
    let no_transverse = {
        let mut s = g.clone();
        s[1] = Matrix::zeros(M, M);
        s[2] = Matrix::zeros(M, M);
        pauli_sum(&s).unwrap()
    };
    let sign_of_y = {
        let mut s = g.clone();
        s[2] = -&s[2];
        pauli_sum(&s).unwrap()
    };
    let charge_only = pauli_sum(&g[..1]).unwrap();
    let right = pauli_sum(&g).unwrap();
    // The transposed off-diagonal blocks (sigma_x, sigma_y act on the other index order).
    let transposed = {
        let mut t = right.clone();
        let (ab, ba) = (
            right.view((0, M), (M, M)).into_owned(),
            right.view((M, 0), (M, M)).into_owned(),
        );
        t.view_mut((0, M), (M, M)).copy_from(&ba);
        t.view_mut((M, 0), (M, M)).copy_from(&ab);
        t
    };
    let rows = [
        ("x and y channels swapped", bite(swapped)),
        ("transverse channels dropped", bite(no_transverse)),
        ("sign of the y channel", bite(sign_of_y)),
        ("charge channel only", bite(charge_only)),
        ("off-diagonal spin blocks exchanged", bite(transposed)),
    ];
    for (name, b) in rows {
        eprintln!("{name}: bites {b:.2e} gates");
        assert!(b > 100., "{name}: {b:e}");
    }
}

#[test]
fn a_time_reversal_crystal_has_the_charge_channel_only() {
    let (d, g) = fixture();
    let one = pauli_sum(&g[..1]).unwrap();
    assert_eq!(one, doubled(&g[0]));
    let want = (component(&d, M, 0) * &g[0]).trace().re;
    assert!((trace(&d, &one) - want).abs() < GATE);
    for count in [0, 2, 3, 5] {
        let parts: Vec<Matrix> = (0..count).map(|_| Matrix::zeros(M, M)).collect();
        assert!(pauli_sum(&parts).is_err(), "{count} channels");
    }
    let mut odd = g.clone();
    odd[3] = Matrix::zeros(M + 1, M + 1);
    assert!(pauli_sum(&odd).is_err(), "channels of different shapes");
}

#[test]
fn the_doubled_frame_repeats_the_scalar_blocks() {
    let mut next = stream(7);
    let a = general(&mut next, M);
    let d = density(&mut next);
    // tr[(I2 x A) D] = tr[A (D_aa + D_bb)].
    let spin_traced = component(&d, M, 0);
    assert!((trace(&d, &doubled(&a)) - (&spin_traced * &a).trace().re).abs() < GATE);
    let layout = Layout {
        atoms: 2,
        m: 3,
        p: 4,
        spins: 1,
        degeneracy: 1.,
        weights: vec![1.],
        row_owner: vec![0, 1, 1],
        projector_owner: vec![0, 0, 1, 1],
        reference: 0.,
        request: Request {
            functional: "lda".into(),
            spinor: true,
            single_precision: false,
        },
    };
    let wide = doubled_layout(&layout).unwrap();
    assert_eq!((wide.m, wide.p), (6, 8));
    assert_eq!(wide.row_owner, [0, 1, 1, 0, 1, 1]);
    assert_eq!(wide.projector_owner, [0, 0, 1, 1, 0, 0, 1, 1]);
    // A scalar request, two spin channels or degeneracy 2 are not a spinor layout.
    let mut scalar = layout.clone();
    scalar.request.spinor = false;
    assert!(doubled_layout(&scalar).is_err());
    let mut polarized = layout.clone();
    polarized.spins = 2;
    assert!(doubled_layout(&polarized).is_err());
    let mut restricted = layout;
    restricted.degeneracy = 2.;
    assert!(doubled_layout(&restricted).is_err());
}
