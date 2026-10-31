//! PAW shell weights of the DFT+U occupation matrix `n = W^T R W` (HU-2): the
//! partial-wave projection of the deleted Python `shell_weights`, reproduced
//! from the shipped datasets against `test/migration/reference/hubbard/frozen.json`.
//!
//! The datasets are read from `MANDACARU_PAW_PATH/lda-sr` (the freeze's set; the
//! scalar-occupation channels also from `lda-fr`). A gate that needs them FAILS when
//! the variable is unset: the sphere fractions are acceptance gates and must not pass
//! by not running.
use mandacaru_core::parquet::read_table;
use mandacaru_core::paw::onecenter::{Basis, Channel};
use mandacaru_core::paw::shell::{occupation, reference_electrons, sphere_fraction, weights};
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde_json::Value;
use std::path::PathBuf;

/// Symbol, shell `l` and the owner's four-digit sphere fraction of the filled shell.
const SHELLS: [(&str, usize, f64); 11] = [
    ("H", 0, 0.2805),
    ("O", 1, 0.6798),
    ("Ti", 2, 0.695),
    ("Mn", 2, 0.905),
    ("Fe", 2, 0.925),
    ("Co", 2, 0.940),
    ("Ni", 2, 0.9511),
    ("Cu", 2, 0.960),
    ("Zn", 2, 0.967),
    ("Ce", 3, 0.425),
    ("Gd", 3, 0.4579),
];

fn context() -> Context {
    Context::new(Some(1), 256 << 20).unwrap()
}

fn frozen() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/migration/reference/hubbard/frozen.json"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let all: Value = serde_json::from_str(&text).unwrap();
    all["cases"]["weights"]["rows"].clone()
}

/// One channel of a shipped library file, as the arrays `shell::weights` takes.
struct Shipped {
    r: Vec<f64>,
    ae: Vec<f64>,
    vanderbilt: Vec<f64>,
    waves: usize,
    r_cut: f64,
    occupations: Vec<f64>,
    l: usize,
}

impl Shipped {
    fn read(symbol: &str, l: usize) -> Self {
        Self::read_from("lda-sr", symbol, l)
    }

    /// A library entry that stores only the scalar channel `occupation` has the list
    /// `[occupation, 0, ...]`, one number per partial wave: this is the padding the
    /// library reader applies before it hands a channel to the weights.
    fn read_from(library: &str, symbol: &str, l: usize) -> Self {
        let root = std::env::var_os("MANDACARU_PAW_PATH")
            .filter(|v| !v.is_empty())
            .expect("MANDACARU_PAW_PATH is unset: the sphere-fraction gates need the lda-sr set");
        let path = PathBuf::from(root).join(format!("{library}/{symbol}.parquet"));
        let ctx = context();
        let table = read_table(&ctx, &path).unwrap_or_else(|e| panic!("{symbol}: {e}"));
        let meta: Value =
            serde_json::from_str(&table.metadata["mandacaru.pseudopotential"]).unwrap();
        let entry = &meta["channels"][l.to_string()];
        let floats = |v: &Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let column = |name: &str| -> Vec<f64> {
            table
                .column(name)
                .and_then(|c| c.as_f64())
                .unwrap_or_else(|| panic!("{symbol}: no column {name}"))
                .to_vec()
        };
        let waves = entry["vanderbilt"].as_array().unwrap().len();
        let occupations = match entry.get("occupations").filter(|v| !v.is_null()) {
            Some(listed) => floats(listed),
            None => std::iter::once(entry["occupation"].as_f64().unwrap())
                .chain(std::iter::repeat(0.))
                .take(waves)
                .collect(),
        };
        assert_eq!(occupations.len(), waves, "{symbol}");
        let ae = (0..waves)
            .flat_map(|i| column(&format!("ae_wave_l{l}_{i}")))
            .collect();
        let vanderbilt = entry["vanderbilt"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| floats(row))
            .collect::<Vec<_>>();
        assert_eq!(vanderbilt.len(), waves * waves, "{symbol}");
        Self {
            r: column("r"),
            ae,
            vanderbilt,
            waves,
            r_cut: entry["r_cut"].as_f64().unwrap(),
            occupations,
            l,
        }
    }

    fn channel(&self) -> Channel<'_> {
        Channel {
            l: self.l,
            waves: self.waves,
            ae: &self.ae,
            ps: &[],
            vanderbilt: &self.vanderbilt,
            r_cut: self.r_cut,
        }
    }

    fn weights(&self, basis: Basis) -> Vec<f64> {
        weights(
            &context(),
            &self.channel(),
            &self.r,
            &self.occupations,
            basis,
        )
        .unwrap()
    }

    fn fraction(&self) -> f64 {
        sphere_fraction(&context(), &self.channel(), &self.r, &self.occupations).unwrap()
    }

    fn b(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.waves, self.waves, &self.vanderbilt)
    }
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.),
        "{what}: got {got:e}, want {want:e}, diff {:e}",
        got - want
    );
}

fn array(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn weights_reproduce_the_frozen_old_python_in_both_bases() {
    let rows = frozen();
    let mut worst = 0f64;
    for (symbol, l, _) in SHELLS {
        let row = &rows[symbol];
        assert_eq!(row["l"].as_u64().unwrap() as usize, l, "{symbol}");
        let shipped = Shipped::read(symbol, l);
        assert_eq!(
            shipped.r.len() as u64,
            row["r_grid_points"].as_u64().unwrap(),
            "{symbol}: grid"
        );
        close(shipped.r_cut, row["r_cut"].as_f64().unwrap(), 1e-14, symbol);
        for (basis, key) in [(Basis::Raw, "w_raw"), (Basis::Dual, "w_dual")] {
            let got = shipped.weights(basis);
            let want = array(&row[key]);
            assert_eq!(got.len(), want.len(), "{symbol} {key}");
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((g - w).abs() / w.abs().max(1.));
                close(*g, *w, 1e-10, &format!("{symbol} {key}"));
            }
        }
    }
    eprintln!("largest weight deviation from the freeze: {worst:e}");
}

#[test]
fn the_isolated_atom_sphere_fractions_are_the_frozen_ones() {
    let rows = frozen();
    let mut deficit = 0f64;
    for (symbol, l, target) in SHELLS {
        let row = &rows[symbol];
        let shipped = Shipped::read(symbol, l);
        let fraction = shipped.fraction();
        close(
            fraction,
            row["fraction_in_sphere"].as_f64().unwrap(),
            1e-10,
            symbol,
        );
        assert!(
            (fraction - target).abs() < 6e-4,
            "{symbol}: fraction {fraction} vs headline {target}"
        );
        // n of the filled reference shell is w_dual[0]^2: the sphere share of the
        // orbital normalized on the whole grid, so it is fraction * norm.
        let n_dual = shipped.weights(Basis::Dual)[0].powi(2);
        close(
            n_dual,
            row["n_reference_dual"].as_f64().unwrap(),
            1e-10,
            symbol,
        );
        let norm = row["reference_orbital_norm"].as_f64().unwrap();
        close(n_dual, fraction * norm, 1e-10, symbol);
        deficit = deficit.max((n_dual - fraction).abs());
    }
    eprintln!("largest |w_dual[0]^2 - fraction|: {deficit:e}");
    assert!(deficit < 1e-5, "{deficit:e}");
}

/// Library, symbol, `l`, the scalar `occupation`, the sphere fraction and the raw and dual
/// weights of the channels that the library stores with a scalar occupation and no list (lda-sr
/// Nd and Eu 4f; lda-fr Ac d, Dy f, Hf f, La d, Nd f, Pr f, U f). The numbers are those of the
/// deleted Python `shell_weights` on the restored v26.10.38 tree (`get_paw(symbol, library)`,
/// the reference wave being the first occupied one, wave 0 after the padding), printed with
/// `repr`. The lda-fr sets still refuse DFT+U at the crystal level (spin-orbit); the weights
/// are defined there and reproduce the old ones all the same.
#[allow(clippy::type_complexity)]
const SCALAR_OCCUPATION: [(&str, &str, usize, f64, f64, [f64; 2], [f64; 2]); 9] = [
    (
        "lda-sr",
        "Nd",
        3,
        4.0,
        0.445240210252258,
        [-0.031004879688911538, -0.013816750502668788],
        [0.6672632241102008, 0.9960229059183964],
    ),
    (
        "lda-sr",
        "Eu",
        3,
        7.0,
        0.45673788380615726,
        [-0.015877026991284314, -0.007898387374659484],
        [0.6758238556095828, 0.9973647472844684],
    ),
    (
        "lda-fr",
        "Ac",
        2,
        1.0,
        0.5515414262650841,
        [-1.068134470592229, 0.45370719967775697],
        [0.7426583509680675, 0.29965590252867014],
    ),
    (
        "lda-fr",
        "Dy",
        3,
        10.0,
        0.4595385778450077,
        [-0.010701471952472508, -0.0035750261137755222],
        [0.6778927480429358, 0.998035360462351],
    ),
    (
        "lda-fr",
        "Hf",
        3,
        14.0,
        0.9998932926941636,
        [-0.19310835400473308, 0.35606623529187625],
        [0.9999466449232016, 0.0106787260551436],
    ),
    (
        "lda-fr",
        "La",
        2,
        1.0,
        0.5513206333411417,
        [-0.6847190490997407, 0.22075553715276816],
        [0.7425096856859462, 0.5781359404038975],
    ),
    (
        "lda-fr",
        "Nd",
        3,
        4.0,
        0.4429929155131378,
        [-0.031077941287982542, -0.0139380394673469],
        [0.6655771296472612, 0.995799242035353],
    ),
    (
        "lda-fr",
        "Pr",
        3,
        3.0,
        0.4354506695177409,
        [-0.018836389655727693, -0.010733732596392253],
        [0.65988686115346, 0.9950343734522782],
    ),
    (
        "lda-fr",
        "U",
        3,
        4.0,
        0.4842272368121803,
        [-0.036899890117425116, -0.02060251704114081],
        [0.6958643810481425, 0.9803471476835294],
    ),
];

#[test]
fn channels_stored_with_a_scalar_occupation_reproduce_the_old_python() {
    let mut worst = 0f64;
    for (library, symbol, l, scalar, fraction, raw, dual) in SCALAR_OCCUPATION {
        let shipped = Shipped::read_from(library, symbol, l);
        assert_eq!(shipped.occupations, vec![scalar, 0.], "{library} {symbol}");
        let what = format!("{library} {symbol} l = {l}");
        close(shipped.fraction(), fraction, 1e-10, &what);
        for (basis, want) in [(Basis::Raw, raw), (Basis::Dual, dual)] {
            let got = shipped.weights(basis);
            assert_eq!(got.len(), 2, "{what}");
            for (g, w) in got.iter().zip(want) {
                worst = worst.max((g - w).abs() / w.abs().max(1.));
                close(*g, w, 1e-10, &what);
            }
        }
    }
    eprintln!("largest scalar-occupation weight deviation from the old Python: {worst:e}");
}

#[test]
fn an_occupation_list_of_the_wrong_length_names_the_library_convention() {
    let shipped = Shipped::read("Nd", 3);
    let message = invalid_message(weights(
        &context(),
        &shipped.channel(),
        &shipped.r,
        &shipped.occupations[..1],
        Basis::Raw,
    ));
    assert!(message.contains("[occupation, 0, ...]"), "{message}");
}

/// `R_dual = c c^dagger`, a shell of `2l+1` blocks of `waves` radial indices (m
/// then radial); the raw-basis projections are `B^T c` in each block.
fn kron_raw(b: &DMatrix<f64>, l: usize) -> DMatrix<Complex64> {
    let waves = b.nrows();
    let size = (2 * l + 1) * waves;
    let mut out = DMatrix::<Complex64>::zeros(size, size);
    for m in 0..2 * l + 1 {
        for i in 0..waves {
            for j in 0..waves {
                out[(m * waves + i, m * waves + j)] = Complex64::new(b[(j, i)], 0.);
            }
        }
    }
    out
}

/// A deterministic complex vector: no external random-number crate is needed.
fn pseudo_random(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    (0..len).map(|_| Complex64::new(next(), next())).collect()
}

#[test]
fn n_is_the_same_in_the_raw_and_the_dual_projector_basis() {
    let mut worst = 0f64;
    for (symbol, l, _) in SHELLS {
        let shipped = Shipped::read(symbol, l);
        let b = shipped.b();
        let (w_dual, w_raw) = (
            DMatrix::from_column_slice(shipped.waves, 1, &shipped.weights(Basis::Dual)),
            DMatrix::from_column_slice(shipped.waves, 1, &shipped.weights(Basis::Raw)),
        );
        // The weights carry the basis change: w_raw = B^-1 w_dual.
        let back = b.clone().try_inverse().unwrap() * &w_dual;
        for i in 0..shipped.waves {
            close(
                w_raw[(i, 0)],
                back[(i, 0)],
                1e-11,
                &format!("{symbol} B^-1 w"),
            );
        }
        // The isolated reference orbital: dual coefficients (1, 0, ...), raw B^T e_0.
        let mut e0 = DMatrix::<f64>::zeros(shipped.waves, 1);
        e0[(0, 0)] = 1.;
        let c_raw = b.transpose() * &e0;
        let n_dual = (w_dual.transpose() * &e0)[(0, 0)].powi(2);
        let n_raw = (w_raw.transpose() * &c_raw)[(0, 0)].powi(2);
        close(n_raw, n_dual, 1e-12, &format!("{symbol} isolated n"));

        // A full complex density matrix of the shell: n agrees, matrix element by element.
        let size = (2 * l + 1) * shipped.waves;
        let mut r_dual = DMatrix::<Complex64>::zeros(size, size);
        for seed in 1..=3u64 {
            let c = pseudo_random(size, seed + 17 * l as u64);
            for i in 0..size {
                for j in 0..size {
                    r_dual[(i, j)] += c[i] * c[j].conj();
                }
            }
        }
        let t = kron_raw(&b, l);
        let r_raw = &t * &r_dual * t.adjoint();
        let n_d = occupation(l, w_dual.as_slice(), &r_dual).unwrap();
        let n_r = occupation(l, w_raw.as_slice(), &r_raw).unwrap();
        let scale = n_d.iter().map(|x| x.norm()).fold(1., f64::max);
        let gap = (&n_d - &n_r).iter().map(|x| x.norm()).fold(0., f64::max);
        worst = worst.max(gap / scale);
        assert!(
            gap < 1e-11 * scale,
            "{symbol}: raw vs dual n differ by {gap:e}"
        );
        // n is Hermitian.
        let skew = (&n_d - n_d.adjoint())
            .iter()
            .map(|x| x.norm())
            .fold(0., f64::max);
        assert!(skew < 1e-12 * scale, "{symbol}: n not Hermitian ({skew:e})");
    }
    eprintln!("largest raw-vs-dual n difference (relative): {worst:e}");
}

#[test]
fn a_wrong_basis_change_breaks_the_invariance() {
    // Negative control: raw weights on unchanged (dual) projections give a different n. (The Ni
    // B is symmetric to rounding, so B c versus B^T c is not a control.)
    let shipped = Shipped::read("Ni", 2);
    let b = shipped.b();
    let w_dual = DMatrix::from_column_slice(shipped.waves, 1, &shipped.weights(Basis::Dual));
    let w_raw = DMatrix::from_column_slice(shipped.waves, 1, &shipped.weights(Basis::Raw));
    let c_dual = DMatrix::from_column_slice(shipped.waves, 1, &[0.3, 0.8]);
    let n_dual = (w_dual.transpose() * &c_dual)[(0, 0)];
    let right = (w_raw.transpose() * (b.transpose() * &c_dual))[(0, 0)];
    let wrong = (w_raw.transpose() * &c_dual)[(0, 0)];
    assert!((right - n_dual).abs() < 1e-12);
    assert!(
        (wrong - n_dual).abs() > 1e-3,
        "control is blind: {wrong} vs {n_dual}"
    );
}

/// Constant and linear waves on a uniform grid: Simpson is exact on `r^2` and `r^3`.
fn polynomial_channel() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r = (0..=200).map(|i| i as f64 * 0.01).collect::<Vec<_>>();
    let mut ae = vec![1.; r.len()];
    ae.extend(r.iter().copied());
    (r, ae, vec![2., 0., 1., 1.])
}

#[test]
fn weights_of_polynomial_waves_are_exact() {
    let (r, ae, b) = polynomial_channel();
    let channel = Channel {
        l: 1,
        waves: 2,
        ae: &ae,
        ps: &[],
        vanderbilt: &b,
        r_cut: 1.,
    };
    let ctx = context();
    let occupations = [1., 0.];
    // chi = 1: norm int_0^1 r^2 = 1/3; w_0 = (1/3)/sqrt(1/3); w_1 = (1/4)/sqrt(1/3).
    let root = (1. / 3f64).sqrt();
    let dual = weights(&ctx, &channel, &r, &occupations, Basis::Dual).unwrap();
    close(dual[0], root, 1e-14, "w0");
    close(dual[1], 0.25 / root, 1e-14, "w1");
    // B = [[2, 0], [1, 1]]: B^-1 = [[1/2, 0], [-1/2, 1]].
    let raw = weights(&ctx, &channel, &r, &occupations, Basis::Raw).unwrap();
    close(raw[0], 0.5 * dual[0], 1e-14, "raw 0");
    close(raw[1], -0.5 * dual[0] + dual[1], 1e-14, "raw 1");
    // The sphere share of chi = 1 on [0, 2] is (1/3)/(8/3).
    let fraction = sphere_fraction(&ctx, &channel, &r, &occupations).unwrap();
    close(fraction, 1. / 8., 1e-14, "fraction");
    // The first occupied wave is the reference, not the first wave.
    let second = weights(&ctx, &channel, &r, &[0., 1.], Basis::Dual).unwrap();
    // int_0^1 r^4 = 1/5; Simpson is not exact for r^4 (error about 1e-8 at h = 0.01).
    let norm = 0.2f64;
    close(second[0], 0.25 / norm.sqrt(), 1e-7, "second ref 0");
    close(second[1], norm.sqrt(), 1e-7, "second ref 1");
    assert_eq!(ctx.memory_used(), 0);
}

fn invalid_message(result: Result<Vec<f64>, Error>) -> String {
    match result {
        Err(Error::Invalid(message)) => message,
        other => panic!("expected Error::Invalid, got {other:?}"),
    }
}

#[test]
fn unsupported_input_raises() {
    let (r, ae, b) = polynomial_channel();
    let ctx = context();
    let good = Channel {
        l: 1,
        waves: 2,
        ae: &ae,
        ps: &[],
        vanderbilt: &b,
        r_cut: 1.,
    };
    let run = |channel: &Channel<'_>, r: &[f64], occupations: &[f64]| {
        weights(&ctx, channel, r, occupations, Basis::Raw)
    };
    // No occupied reference: the old code silently took wave 0.
    let message = invalid_message(run(&good, &r, &[0., 0.]));
    assert!(message.contains("occupied"), "{message}");
    // Occupations of the wrong length, negative or not finite.
    assert!(run(&good, &r, &[1.]).is_err());
    assert!(run(&good, &r, &[-1., 0.]).is_err());
    assert!(run(&good, &r, &[f64::NAN, 0.]).is_err());
    // Array sizes that disagree with the grid.
    assert!(run(&good, &r[..100], &[1., 0.]).is_err());
    let short = Channel {
        vanderbilt: &b[..3],
        ..good
    };
    assert!(run(&short, &r, &[1., 0.]).is_err());
    // A singular Vanderbilt matrix cannot be inverted for the raw basis.
    let singular = [1., 2., 2., 4.];
    let flat = Channel {
        vanderbilt: &singular,
        ..good
    };
    assert!(run(&flat, &r, &[1., 0.]).is_err());
    assert!(weights(&ctx, &flat, &r, &[1., 0.], Basis::Dual).is_ok());
    // A sphere that holds fewer than two grid points, or a radius that is not a number.
    for r_cut in [0., -1., f64::NAN, f64::INFINITY] {
        let c = Channel { r_cut, ..good };
        assert!(run(&c, &r, &[1., 0.]).is_err(), "r_cut {r_cut}");
    }
    // A reference wave that vanishes inside the sphere has no norm.
    let mut dead = ae.clone();
    dead[..r.len()].iter_mut().for_each(|x| *x = 0.);
    let c = Channel { ae: &dead, ..good };
    assert!(invalid_message(run(&c, &r, &[1., 0.])).contains("bound"));
    // Non-finite samples.
    let mut bad = ae.clone();
    bad[3] = f64::NAN;
    let c = Channel { ae: &bad, ..good };
    assert!(run(&c, &r, &[1., 0.]).is_err());
    // A grid that is not increasing.
    let mut unsorted = r.clone();
    unsorted.swap(5, 6);
    assert!(run(&good, &unsorted, &[1., 0.]).is_err());
}

#[test]
fn the_reference_electrons_are_the_channel_occupations() {
    // The DFT+U seed starts from the electrons of the reference shell; a library entry that
    // stores only the scalar occupation arrives as [occupation, 0, ...].
    assert_eq!(reference_electrons(&[8., 0.]).unwrap(), 8.);
    assert_eq!(reference_electrons(&[4., 2.5]).unwrap(), 6.5);
    assert_eq!(reference_electrons(&[]).unwrap(), 0.);
    for bad in [&[-1., 0.][..], &[f64::NAN][..], &[f64::INFINITY, 1.][..]] {
        assert!(reference_electrons(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_working_memory_comes_from_the_context() {
    let (r, ae, b) = polynomial_channel();
    let channel = Channel {
        l: 1,
        waves: 2,
        ae: &ae,
        ps: &[],
        vanderbilt: &b,
        r_cut: 1.,
    };
    let tiny = Context::new(Some(1), 64).unwrap();
    match weights(&tiny, &channel, &r, &[1., 0.], Basis::Raw) {
        Err(Error::Budget { .. }) => {}
        other => panic!("expected a budget error, got {other:?}"),
    }
    assert!(matches!(
        sphere_fraction(&tiny, &channel, &r, &[1., 0.]),
        Err(Error::Budget { .. })
    ));
    assert_eq!(tiny.memory_used(), 0);
}

#[test]
fn the_occupation_matrix_is_w_transpose_r_w() {
    // p shell (l = 1), two radial indices: R is 6 x 6, ordered m then radial.
    let weights = [0.6, -0.2];
    let size = 6;
    let c = pseudo_random(size, 5);
    let mut r = DMatrix::<Complex64>::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            r[(i, j)] = c[i] * c[j].conj();
        }
    }
    let n = occupation(1, &weights, &r).unwrap();
    assert_eq!((n.nrows(), n.ncols()), (3, 3));
    for m in 0..3 {
        for mp in 0..3 {
            let mut want = Complex64::default();
            for i in 0..2 {
                for j in 0..2 {
                    want += weights[i] * r[(m * 2 + i, mp * 2 + j)] * weights[j];
                }
            }
            assert!((n[(m, mp)] - want).norm() < 1e-14, "({m}, {mp})");
        }
    }
    // A pure state in the shell gives an idempotent-rank-one n: Tr n = |w . c_m|^2 summed.
    let trace: f64 = (0..3)
        .map(|m| (weights[0] * c[2 * m] + weights[1] * c[2 * m + 1]).norm_sqr())
        .sum();
    assert!((n.trace().re - trace).abs() < 1e-14);
    // Refusals.
    assert!(occupation(1, &weights, &DMatrix::zeros(5, 5)).is_err());
    assert!(occupation(1, &[], &DMatrix::zeros(0, 0)).is_err());
    assert!(occupation(1, &[f64::NAN, 1.], &r).is_err());
}
