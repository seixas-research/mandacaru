//! The augmentation remainder of a crystal atom (`periodic::setup::remainder`): the transforms
//! of the remainder densities of its projector pairs and the on-site Coulomb remainder, against
//! the frozen Python (Si and Al, raw projectors; gate `max(1e-12, 100 x` the measured floor`)`),
//! and the identities that need no frozen number (Hermiticity, translation with the atom, the
//! pair symmetry of the quadrature). The frozen transform is the LINEAR TABLE of the Python era
//! (step 0.025, end 12, clamp 1.000001), so the twin interpolates the same table. The one frozen
//! vector past the end of the table (12.717 > 12.025) is where the Python reads the last entry
//! (a wrong number: 8.5e-3 of the largest entry against a table that reaches it); the Rust
//! raises there, so that vector is excluded from the comparison and asserted to raise. Needs
//! `MANDACARU_PAW_PATH`.
mod setup_gradients_support;

use mandacaru_core::{
    paw::onecenter::{Basis, Projector},
    periodic::setup::{
        remainder::{onsite_remainder, projector_columns, Remainder},
        shapes::Shapes,
    },
};
use num_complex::Complex64;
use setup_gradients_support::*;

/// The end of the table, `(points - 1) * step`.
const TABLE_END: f64 = 12.025;

fn vectors(case: &Case) -> Vec<[f64; 3]> {
    let a = Array::read(&case.inputs()["remainder_vectors_bohr_inv"]);
    let n = a.shape[1];
    (0..n)
        .map(|v| [a.re[v], a.re[n + v], a.re[2 * n + v]])
        .collect()
}

/// `(owners, layout)` of every projector in crystal order, from the frozen `projector_lm`.
fn layout(case: &Case) -> (Vec<usize>, Vec<Projector>) {
    let lm = case.quantity("projector_lm");
    let rows: Vec<&[f64]> = lm.re.chunks(4).collect();
    let owners = rows.iter().map(|r| r[0] as usize).collect();
    let projectors = rows
        .iter()
        .map(|r| Projector {
            l: r[1] as usize,
            m: r[2] as i32,
            index: r[3] as usize,
            real: false,
        })
        .collect();
    (owners, projectors)
}

fn of_atom(case: &Case, atom: usize) -> Vec<Projector> {
    let (owners, all) = layout(case);
    projector_columns(&owners, atom)
        .into_iter()
        .map(|p| all[p])
        .collect()
}

#[test]
fn the_projector_columns_are_the_frozen_lists() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let (owners, _) = layout(&case);
        for atom in 0..case.atoms() {
            let want: Vec<usize> = case
                .quantity(&format!("projector_columns.atom{atom}"))
                .re
                .iter()
                .map(|x| *x as usize)
                .collect();
            let got = projector_columns(&owners, atom);
            assert_eq!(got, want, "{name} atom {atom}");
            assert!(!got.is_empty());
        }
        assert!(projector_columns(&owners, 99).is_empty());
    }
}

#[test]
fn the_remainder_transforms_equal_the_frozen_python_and_their_identities() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        let cell = case.cell();
        let sites = cell.sites();
        let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
        let all = vectors(&case);
        let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let keep: Vec<usize> = (0..all.len())
            .filter(|&v| norm(&all[v]) <= TABLE_END)
            .collect();
        let g: Vec<[f64; 3]> = keep.iter().map(|&v| all[v]).collect();
        let past: Vec<[f64; 3]> = all
            .iter()
            .filter(|v| norm(v) > TABLE_END)
            .copied()
            .collect();
        assert!(
            !past.is_empty() && g.len() + past.len() == all.len(),
            "{name}: the frozen vectors no longer reach past the table"
        );
        let mut first: Option<Vec<Complex64>> = None;
        for atom in 0..case.atoms() {
            let ds = case.dataset(&ctx, atom);
            let projectors = of_atom(&case, atom);
            let n = projectors.len();
            let center = cell.atoms[atom].center;
            let remainder =
                Remainder::new(&ctx, &ds, &shapes, Basis::Raw, &projectors, center).unwrap();
            let mut got = vec![Complex64::default(); n * n * g.len()];
            remainder.transforms(&g, &mut got).unwrap();
            let key = format!("remainder_transforms.atom{atom}");
            let frozen = case.quantity(&key).complex();
            let want: Vec<Complex64> = (0..n * n)
                .flat_map(|ab| keep.iter().map(move |&v| (ab, v)))
                .map(|(ab, v)| frozen[ab * all.len() + v])
                .collect();
            let miss = relative(&got, &want);
            let mut beyond = vec![Complex64::default(); n * n * past.len()];
            assert!(remainder.transforms(&past, &mut beyond).is_err());
            let gate = case.gate(&key);
            println!("{name} atom {atom}: remainder transforms {miss:.2e} (gate {gate:.1e})");
            assert!(miss <= gate);
            // Hermiticity: E_ab(g) = conj(E_ba(-g)).
            let minus: Vec<[f64; 3]> = g.iter().map(|v| [-v[0], -v[1], -v[2]]).collect();
            let mut flipped = vec![Complex64::default(); got.len()];
            remainder.transforms(&minus, &mut flipped).unwrap();
            let count = g.len();
            let mut hermitian = 0.0f64;
            for a in 0..n {
                for b in 0..n {
                    for v in 0..count {
                        hermitian = hermitian.max(
                            (got[(a * n + b) * count + v]
                                - flipped[(b * n + a) * count + v].conj())
                            .norm(),
                        );
                    }
                }
            }
            assert!(hermitian < 1e-9, "{hermitian}");
            // The atom carries its transform: the same table times exp(-i g.(R1 - R0)).
            match &first {
                None => first = Some(got.clone()),
                Some(base) if projectors.len() * projectors.len() * count == base.len() => {
                    let shift = cell.atoms[atom].center;
                    let d = [
                        shift[0] - cell.atoms[0].center[0],
                        shift[1] - cell.atoms[0].center[1],
                        shift[2] - cell.atoms[0].center[2],
                    ];
                    let mut worst = 0.0f64;
                    for ab in 0..n * n {
                        for v in 0..count {
                            let angle = -(g[v][0] * d[0] + g[v][1] * d[1] + g[v][2] * d[2]);
                            let phase = Complex64::new(angle.cos(), angle.sin());
                            worst = worst
                                .max((got[ab * count + v] - base[ab * count + v] * phase).norm());
                        }
                    }
                    assert!(worst < 1e-9, "{worst}");
                }
                Some(_) => {}
            }
            // Controls: the phase of the atom dropped, and the unit coefficients of the dual basis.
            let bare =
                Remainder::new(&ctx, &ds, &shapes, Basis::Raw, &projectors, [0.; 3]).unwrap();
            let mut dropped = vec![Complex64::default(); got.len()];
            bare.transforms(&g, &mut dropped).unwrap();
            let moved = center.iter().any(|x| x.abs() > 1e-6);
            if moved {
                assert!(relative(&dropped, &want) > 1e3 * gate);
            }
            let dual =
                Remainder::new(&ctx, &ds, &shapes, Basis::Dual, &projectors, center).unwrap();
            let mut other = vec![Complex64::default(); got.len()];
            dual.transforms(&g, &mut other).unwrap();
            assert!(relative(&other, &want) > 1e3 * gate);
        }
    }
}

/// A vector past the end of the table raises instead of reading the last entry; the last point of
/// the table is still read.
#[test]
fn a_vector_past_the_table_end_raises() {
    let case = Case::load("si");
    let ctx = context();
    let cell = case.cell();
    let sites = cell.sites();
    let shapes = Shapes::new(&ctx, &sites, 20.).unwrap();
    let ds = case.dataset(&ctx, 0);
    let projectors = of_atom(&case, 0);
    let n = projectors.len();
    let center = cell.atoms[0].center;
    let remainder = Remainder::new(&ctx, &ds, &shapes, Basis::Raw, &projectors, center).unwrap();
    let mut out = vec![Complex64::default(); n * n];
    // The table ends at 12.025: 12.02 reads, 12.03 and 40 raise.
    remainder.transforms(&[[12.02, 0., 0.]], &mut out).unwrap();
    for far in [12.03, 40.] {
        let err = remainder.transforms(&[[0., far, 0.]], &mut out);
        assert!(err.is_err(), "|g| = {far}");
    }
}

#[test]
fn the_onsite_remainder_equals_the_frozen_python_and_is_a_hermitian_pair_tensor() {
    for name in ["si", "al2", "al"] {
        let case = Case::load(name);
        let ctx = context();
        for atom in 0..case.atoms() {
            let ds = case.dataset(&ctx, atom);
            let projectors = of_atom(&case, atom);
            let n = projectors.len();
            let tensor = onsite_remainder(&ctx, &ds, Basis::Raw, &projectors).unwrap();
            let key = format!("onsite_remainder.atom{atom}");
            let want = case.quantity(&key).complex();
            let miss = relative(&tensor.data, &want);
            let gate = case.gate(&key);
            let index = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
            let (mut hermitian, mut exchange, mut scale) = (0.0f64, 0.0f64, 0.0f64);
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            let w = tensor.data[index(a, b, c, d)];
                            hermitian =
                                hermitian.max((w - tensor.data[index(b, a, d, c)].conj()).norm());
                            exchange = exchange.max((w - tensor.data[index(c, d, a, b)]).norm());
                            scale = scale.max(w.norm());
                        }
                    }
                }
            }
            println!("{name} atom {atom}: onsite remainder {miss:.2e} (gate {gate:.1e}), hermitian {hermitian:.1e}, pair exchange {:.1e} of max {scale:.1e}", exchange / scale);
            assert!(miss <= gate && hermitian < 1e-10 && exchange <= 5e-5 * scale);
            // Control: the pair convention (a, c, b, d) is another tensor.
            let wrong: Vec<Complex64> = (0..n * n * n * n)
                .map(|i| {
                    let (a, b, c, d) = (i / (n * n * n), i / (n * n) % n, i / n % n, i % n);
                    tensor.data[index(a, c, b, d)]
                })
                .collect();
            assert!(relative(&wrong, &want) > 1e3 * gate);
        }
    }
}
