//! The Bethe-Salpeter excitons of a crystal (`periodic::optics::bse`, TDDFT_PLAN.md T5b), checked
//! by the gates of section 4.4:
//!
//! - E0: without kernel the excitonic tensor is T5a's independent-particle tensor;
//! - E1: the full (coupling) exchange-only equation is T5a's Dyson equation with local fields;
//! - E3: a two-band Wannier-Mott model through the same head average, contraction and solvers
//!   gives the hydrogenic 1s binding `mu / (2 eps^2)` and a 2s near a quarter of it;
//! - E4, E5: a bound singlet below the continuum's onset in LiH and LiF on two meshes, the triplet
//!   at or below it, the Wannier-Mott estimate beside it (recorded, `#[ignore]` for LiF);
//! - E6: `H` Hermitian, Davidson equal to the dense roots, Haydock equal to the eigenvector sum;
//! - E7: the operator does not depend on the layout.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod berry_support;
mod setup_shells_support;

use berry_support::*;
use mandacaru_core::{
    periodic::{
        bloch_mesh::MeshIndex,
        optics::{
            bse::{
                self, contract, coupled_tensor, operator, solve_davidson, solve_dense, Request,
                Spin,
            },
            dielectric::{dielectric, pairs, Kernel, Request as Optical},
        },
        screening::interaction::{cell_average, folded, head_average, vector},
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::f64::consts::PI;

const HARTREE_EV: f64 = 27.211386245988;

fn worst(a: &[[[C64; 3]; 3]], b: &[[[C64; 3]; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            (0..3)
                .flat_map(move |i| (0..3).map(move |j| (x[i][j] - y[i][j]).norm() / y[0][0].norm()))
        })
        .fold(0f64, f64::max)
}

#[test]
fn without_a_kernel_the_excitons_are_the_independent_transitions() {
    // E0: H = D, so every exciton is one transition and the tensor is T5a's independent one.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let op = operator(
        &sampler,
        &mesh,
        &Request::new(Kernel::Independent, Spin::Singlet),
    )
    .unwrap();
    let all = op.dense().unwrap();
    let frequencies = [0., 0.1, 0.25, 0.4];
    let reference = dielectric(
        &sampler,
        &mesh,
        &Optical::new(frequencies.to_vec(), 0.05, Kernel::Independent),
    )
    .unwrap();
    let ours: Vec<_> = frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let difference = worst(&ours, &reference.tensor);
    let moment = bse::first_moment(&op);
    eprintln!(
        "E0 LiH: {} transitions, tensor difference {difference:.2e}, first moment {:?} against {:?}",
        op.energies.len(),
        [0, 1, 2].map(|a| moment[a][a]),
        [0, 1, 2].map(|a| reference.sum_rule[a][a])
    );
    assert!(difference < 1e-12, "{difference:e}");
    assert_eq!(op.hermiticity, 0.);
    for a in 0..3 {
        assert!((moment[a][a] - reference.sum_rule[a][a]).abs() < 1e-12 * reference.sum_rule[0][0]);
    }
}

#[test]
fn the_coupled_exchange_equation_is_the_local_field_dyson_equation() {
    // E1: the full (resonant and antiresonant, coupled) equation with the exchange alone is T5a's
    // Dyson equation on the same columns, by the Woodbury identity; the Tamm-Dancoff tensor of the
    // operator differs from it by the coupling only (recorded).
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut optical = Optical::new(vec![0., 0.1, 0.25, 0.4], 0.05, Kernel::Hartree);
    optical.cutoff = 2.;
    let columns = pairs(&sampler, &mesh, &optical).unwrap();
    let reference = dielectric(&sampler, &mesh, &optical).unwrap();
    let coupled: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| coupled_tensor(&columns, C64::new(*w, 0.05)).unwrap())
        .collect();
    let difference = worst(&coupled, &reference.tensor);
    let mut request = Request::new(Kernel::Hartree, Spin::Singlet);
    request.cutoff = 2.;
    let op = operator(&sampler, &mesh, &request).unwrap();
    let all = op.dense().unwrap();
    let tda: Vec<_> = optical
        .frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    eprintln!(
        "E1 LiH ({} transitions, {} vectors): coupled against Dyson {difference:.2e}; Tamm-Dancoff \
         against Dyson {:.2e}; eps_xx(0.05 i) Dyson {:.6}, coupled {:.6}, TDA {:.6}; independent \
         {:.6}",
        op.energies.len(),
        op.vectors,
        worst(&tda, &reference.tensor),
        reference.tensor[0][0][0],
        coupled[0][0][0],
        tda[0][0][0],
        dielectric(&sampler, &mesh, &Optical::new(vec![0.], 0.05, Kernel::Independent))
            .unwrap()
            .tensor[0][0][0]
    );
    assert!(difference < 1e-10, "{difference:e}");
}

/// The model's direct element of every mesh vector `p` through the crate's head average and
/// contraction (one pair density, `G = 0` only), and the band energies `D_k = 1 + k^2`.
fn wannier_mott_terms(n: usize, a: f64, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let reciprocal = Matrix3::identity() * (2. * PI / a);
    let nk = mesh.points();
    let one = DMatrix::from_element(1, 1, C64::new(1., 0.));
    let dielectric = Matrix3::identity() * eps;
    let direct = (0..nk)
        .map(|p| {
            // The mean of 4 pi / (eps q^2) over each mesh vector's mini-zone cell, as the crystal's
            // W takes it (the self-similar head at p = 0).
            let w = if p == 0 {
                head_average(&reciprocal, &mesh, &dielectric, 8).unwrap()
            } else {
                cell_average(&reciprocal, &mesh, folded(&mesh, p), &dielectric, 8).unwrap()
            };
            let screened = DMatrix::from_element(1, 1, C64::new(w, 0.));
            contract(&one, &screened, &one, 1, 1).unwrap()[(0, 0)].re / (nk as f64 * a * a * a)
        })
        .collect();
    let energies = (0..nk)
        .map(|k| 1. + Vector3::from(vector(&reciprocal, &mesh, folded(&mesh, k))).norm_squared())
        .collect();
    (energies, direct)
}

/// The explicit model operator `H_{kk'} = D_k delta - w(k' - k)`.
fn wannier_mott_matrix(n: usize, a: f64, eps: f64) -> DMatrix<C64> {
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let (energies, direct) = wannier_mott_terms(n, a, eps);
    let nk = mesh.points();
    DMatrix::from_fn(nk, nk, |i, j| {
        let (at, bt) = (mesh.indices(i), mesh.indices(j));
        let p = mesh.index(std::array::from_fn(|x| (bt[x] + n - at[x]) % n));
        C64::new(if i == j { energies[i] } else { 0. } - direct[p], 0.)
    })
}

/// The same operator applied by a circular convolution on the mesh (`w` is even in `p`).
struct WannierMott {
    energies: Vec<f64>,
    transform: Vec<C64>,
    fft: std::sync::Mutex<mandacaru_core::math::fft::Fft3>,
}

impl mandacaru_core::math::davidson::Symmetric for WannierMott {
    fn dimension(&self) -> usize {
        self.energies.len()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.energies.clone()
    }
    fn apply(&self, block: &DMatrix<f64>) -> mandacaru_core::Result<DMatrix<f64>> {
        let mut out = DMatrix::zeros(block.nrows(), block.ncols());
        let mut fft = self.fft.lock().unwrap();
        for j in 0..block.ncols() {
            let mut work: Vec<C64> = block.column(j).iter().map(|v| C64::new(*v, 0.)).collect();
            fft.transform(&mut work, false)?;
            for (x, w) in work.iter_mut().zip(&self.transform) {
                *x *= w;
            }
            fft.transform(&mut work, true)?;
            for i in 0..block.nrows() {
                out[(i, j)] = self.energies[i] * block[(i, j)] - work[i].re;
            }
        }
        Ok(out)
    }
}

/// The lowest roots `(binding 1 - E, |sum_k A_k|^2)` of the matrix-free model on an `n^3` mesh.
fn wannier_mott(n: usize, a: f64, eps: f64, roots: usize) -> Vec<(f64, f64)> {
    let ctx = Context::new(Some(1), 1 << 31).unwrap();
    let (energies, direct) = wannier_mott_terms(n, a, eps);
    let mut fft = mandacaru_core::math::fft::Fft3::new(&ctx, [n; 3]).unwrap();
    let mut transform: Vec<C64> = direct.iter().map(|w| C64::new(*w, 0.)).collect();
    fft.transform(&mut transform, false).unwrap();
    let op = WannierMott {
        energies,
        transform,
        fft: std::sync::Mutex::new(fft),
    };
    let mut options = mandacaru_core::math::davidson::Options::new(roots);
    options.tolerance = 1e-8;
    options.max_iterations = 400;
    let solved = mandacaru_core::math::davidson::lowest(&op, &options).unwrap();
    assert!(solved.converged);
    (0..roots)
        .map(|l| {
            let sum: f64 = solved.vectors.column(l).iter().sum();
            (1. - solved.values[l], sum * sum)
        })
        .collect()
}

#[test]
fn the_wannier_mott_model_binds_a_hydrogenic_series() {
    // E3: masses 1 and 1 (mu = 1/2) and eps = 2: the 1s binding mu / (2 eps^2) = 1/16 Ha, the 2s
    // at a quarter of it and its weight |sum_k A_k|^2 (the s amplitude at r = 0) at an eighth;
    // Bohr radius eps / mu = 4, lattice constant 3 (the zone reaches k a_B = 4 pi / 3; the
    // supercell n a reaches 30 a_B at n = 40).
    let exact = 0.5 / (2. * 2. * 2.);
    // The explicit operator through the crate's dense and Davidson solvers, and the matrix-free
    // one, on the smallest mesh.
    let h = wannier_mott_matrix(8, 3., 2.);
    let dense = solve_dense(&h).unwrap();
    let iterative = solve_davidson(&h, 6, 1e-9, 400).unwrap();
    let free = wannier_mott(8, 3., 2., 6);
    for l in 0..6 {
        assert!((dense.energies[l] - iterative.energies[l]).abs() < 1e-9);
        assert!((1. - dense.energies[l] - free[l].0).abs() < 1e-8);
    }
    let mut errors = vec![];
    for n in [16usize, 24, 32, 40] {
        let roots = wannier_mott(n, 3., 2., 6);
        let first = roots[0];
        let second = roots[1..]
            .iter()
            .copied()
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap();
        eprintln!(
            "E3 n = {n}: 1s binding {:.6} Ha ({:+.3} % of mu / 2 eps^2 = {exact:.6}), 2s {:.6} Ha \
             (ratio {:.4} against 0.25), s weights ratio {:.4} against 0.125",
            first.0,
            100. * (first.0 / exact - 1.),
            second.0,
            second.0 / first.0,
            second.1 / first.1,
        );
        errors.push((
            (first.0 / exact - 1.).abs(),
            (second.0 / first.0 / 0.25 - 1.).abs(),
        ));
    }
    // The zone's truncation: the same n with a smaller lattice constant (the zone reaches
    // k a_B = 2 pi instead of 4 pi / 3; the supercell is 20 a_B instead of 30).
    let finer = wannier_mott(40, 2., 2., 1)[0].0;
    eprintln!(
        "E3 n = 40, a = 2: 1s binding {finer:.6} Ha ({:+.3} %)",
        100. * (finer / exact - 1.)
    );
    let last = errors[errors.len() - 1];
    assert!(last.0 < errors[0].0 && last.0 < 0.03, "{errors:?}");
    assert!(last.1 < errors[0].1 && last.1 < 0.1, "{errors:?}");
}

#[test]
fn the_operator_is_hermitian_and_its_solvers_agree() {
    // E6 on LiH 2 x 1 x 1 with every band and the direct term: H Hermitian to round-off,
    // Davidson on the real form equal to the dense roots, the Haydock spectrum (the whole Krylov
    // space) equal to the eigenvector sum.
    let solved = solve(&lih("lda-sr"));
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.cutoff = 2.;
    let op = operator(&sampler, &mesh, &request).unwrap();
    let n = op.energies.len();
    let all = op.dense().unwrap();
    let roots = 3.min(n);
    let iterative = solve_davidson(&op.hamiltonian, roots, 1e-9, 400).unwrap();
    let parity = (0..roots)
        .map(|l| (iterative.energies[l] - all.energies[l]).abs())
        .fold(0f64, f64::max);
    let haydock = op.haydock(&solved.ctx, n).unwrap();
    let frequencies = [0., 0.1, 0.2, 0.3];
    let exact: Vec<_> = frequencies
        .iter()
        .map(|w| op.tensor(&all, C64::new(*w, 0.05)))
        .collect();
    let lanczos: Vec<_> = frequencies
        .iter()
        .map(|w| haydock.tensor(C64::new(*w, 0.05)))
        .collect();
    let spectra = worst(&lanczos, &exact);
    eprintln!(
        "E6 LiH ({n} transitions): Hermiticity {:.2e}; Davidson against dense {parity:.2e}; \
         Haydock against the eigenvector sum {spectra:.2e}; head {:.4} Ha Bohr^3 (eps_M {:.4}); \
         lowest {:?} Ha against onset {:.6}",
        op.hermiticity,
        op.head,
        op.static_tensor[0][0],
        &all.energies[..roots],
        op.onset
    );
    assert!(op.hermiticity < 1e-10, "{:e}", op.hermiticity);
    assert!(parity < 1e-8, "{parity:e}");
    assert!(spectra < 1e-9, "{spectra:e}");
}

#[test]
fn the_operator_does_not_depend_on_the_layout() {
    // E7: threads and kpts lanes change no bit of H.
    let cell = lih("lda-sr");
    let mut operators = vec![];
    for (threads, kpts) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let ctx = Context::new(Some(threads), 1 << 30)
            .unwrap()
            .with_layout(kpts, 1)
            .unwrap();
        let solved = solve_on(&cell, "raw", "SZ", ctx);
        let sampler = solved.sampler();
        let mesh = MeshIndex::new([2, 1, 1], [0.; 3]).unwrap();
        let mut request = Request::new(Kernel::Bse, Spin::Singlet);
        request.cutoff = 2.;
        let op = operator(&sampler, &mesh, &request).unwrap();
        eprintln!(
            "E7 ({threads}, {kpts}): head {:e}, eps_M {:e}",
            op.head, op.static_tensor[0][0]
        );
        let exchange = operator(&sampler, &mesh, &{
            let mut r = request.clone();
            r.kernel = Kernel::Hartree;
            r
        })
        .unwrap()
        .hamiltonian;
        operators.push((op.hamiltonian, exchange));
    }
    for (h, x) in &operators[1..] {
        let direct = (h - &operators[0].0)
            .iter()
            .map(|v| v.norm())
            .fold(0f64, f64::max);
        let exchange = (x - &operators[0].1)
            .iter()
            .map(|v| v.norm())
            .fold(0f64, f64::max);
        eprintln!("E7: largest change of H {direct:e}, of the exchange-only H {exchange:e}");
    }
    for (h, x) in &operators[1..] {
        assert_eq!(x, &operators[0].1);
        assert_eq!(h, &operators[0].0);
    }
}

/// A rock-salt crystal (fcc, cubic constant `a` Bohr) with the cation at the origin.
fn rocksalt(symbols: [&'static str; 2], a: f64, grid: usize, mesh: usize) -> Cell {
    let h = a / 2.;
    Cell {
        library: "lda-sr",
        symbols: symbols.to_vec(),
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., h, h),
            Vector3::new(h, 0., h),
            Vector3::new(h, h, 0.),
        ]),
        grid: [grid; 3],
        fractional: vec![[0.; 3], [0.5; 3]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [mesh; 3],
        hubbard: None,
        width: 1e-3,
    }
}

/// The lowest singlet and triplet of `solved` on an `n^3` response mesh: binding against the
/// onset, participation ratio, and the Wannier-Mott estimate from `eps_M`.
fn excitons(
    solved: &Solved,
    n: usize,
    window: Option<(usize, usize)>,
    label: &str,
) -> (f64, f64, f64) {
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([n; 3], [0.; 3]).unwrap();
    let started = std::time::Instant::now();
    let mut request = Request::new(Kernel::Bse, Spin::Singlet);
    request.window = window;
    let singlet = operator(&sampler, &mesh, &request).unwrap();
    let lowest = singlet.lowest(4, 1e-7, 400).unwrap();
    request.spin = Spin::Triplet;
    let triplet = operator(&sampler, &mesh, &request).unwrap();
    let lowest_triplet = triplet.lowest(1, 1e-7, 400).unwrap();
    // E6 on a window with a degenerate root and a complex H: Davidson on the real form at the
    // binding's default tolerance against the dense roots.
    let iterative = solve_davidson(&singlet.hamiltonian, 4, 1e-5, 400).unwrap();
    let parity = (0..4)
        .map(|l| (iterative.energies[l] - lowest.energies[l]).abs())
        .fold(0f64, f64::max);
    eprintln!("{label} {n}^3: Davidson (1e-5) against dense {parity:.2e} Ha");
    assert!(parity < 1e-8, "{parity:e}");
    assert!(singlet.hermiticity < 1e-10 && triplet.hermiticity < 1e-10);
    let a = lowest.vectors.column(0);
    let participation = 1. / a.iter().map(|x| x.norm_sqr().powi(2)).sum::<f64>();
    let binding = singlet.onset - lowest.energies[0];
    let eps = singlet.static_tensor[0][0];
    eprintln!(
        "{label} {n}^3 ({} transitions, window {window:?}): onset {:.4} eV; singlets {:?} eV; \
         binding {:.4} eV; triplet {:.4} eV (binding {:.4}); participation {:.1} of {}; eps_M \
         {eps:.4}; head {:.3}; Hermiticity {:.1e}; {:.0} s",
        singlet.energies.len(),
        singlet.onset * HARTREE_EV,
        lowest
            .energies
            .iter()
            .map(|e| (e * HARTREE_EV * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        binding * HARTREE_EV,
        lowest_triplet.energies[0] * HARTREE_EV,
        (triplet.onset - lowest_triplet.energies[0]) * HARTREE_EV,
        participation,
        singlet.energies.len(),
        singlet.head,
        singlet.hermiticity,
        started.elapsed().as_secs_f64()
    );
    (
        binding,
        lowest_triplet.energies[0] - lowest.energies[0],
        participation / singlet.energies.len() as f64,
    )
}

#[test]
#[ignore = "E4/E5 on LiH DZP, recorded: the 4^3 and 6^3 meshes (both hold X; about 10 minutes)"]
fn lithium_hydride_binds_an_exciton_below_the_continuum() {
    // E4 on LiH (rock salt, 4.084 A): the lowest singlet below the onset on two meshes, by more
    // than the change between them; the triplet at or below it; the weight concentrated.
    // DZP: an SZ basis gives LiH one conduction band, and an eps_M of 1.2.
    let solved = solve_sized(&rocksalt(["Li", "H"], 7.718, 24, 3), "raw", "DZP");
    let (b3, t3, r3) = excitons(&solved, 4, Some((1, 4)), "E4 LiH DZP");
    let (b4, t4, r4) = excitons(&solved, 6, Some((1, 4)), "E4 LiH DZP");
    eprintln!("E4 LiH: bindings {:.4} / {:.4} eV, triplet offsets {t3:.2e} / {t4:.2e}, participation fractions {r3:.3} / {r4:.3}", b3 * HARTREE_EV, b4 * HARTREE_EV);
    assert!(b4 > 0. && b4 > (b4 - b3).abs(), "{b3} {b4}");
    assert!(t3 <= 1e-10 && t4 <= 1e-10);
}

#[test]
#[ignore = "E4/E5 on LiF, recorded: two meshes (several minutes)"]
fn lithium_fluoride_binds_a_strong_exciton() {
    let size = std::env::var("SIZES").unwrap_or("SZ".into());
    let meshes: Vec<usize> = std::env::var("MESHES")
        .map(|v| v.split(',').map(|m| m.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![3, 4]);
    let solved = solve_sized(&rocksalt(["Li", "F"], 7.608, 28, 4), "raw", &size);
    let mut out = vec![];
    for n in meshes {
        out.push(excitons(
            &solved,
            n,
            Some((3, 3)),
            &format!("E4 LiF {size}"),
        ));
    }
    let b: Vec<f64> = out.iter().map(|o| o.0 * HARTREE_EV).collect();
    eprintln!("E4 LiF: bindings {b:?} eV");
    assert!(out.iter().all(|o| o.0 > 0.));
}

#[test]
#[ignore = "T5c against T5b, recorded: LiH DZP, LiF DZP and Si SZ on 4^3 (several minutes)"]
fn the_long_range_kernel_against_the_bethe_salpeter_equation() {
    // eps_M(0) and the first peak of Im eps (0 to 15 eV, damping 0.1 eV) for independent particles,
    // local fields, the long-range kernel with its fitted alpha, and the Bethe-Salpeter equation,
    // each on the same window and mesh.
    let cases = [
        (
            "LiH DZP",
            rocksalt(["Li", "H"], 7.718, 24, 4),
            "DZP",
            (1, 4),
        ),
        (
            "LiF DZP",
            rocksalt(["Li", "F"], 7.608, 28, 4),
            "DZP",
            (3, 3),
        ),
        (
            "Si SZ",
            {
                let mut si = rocksalt(["Si", "Si"], 10.26, 20, 4);
                si.fractional = vec![[0.; 3], [0.25; 3]];
                si
            },
            "SZ",
            (4, 4),
        ),
    ];
    let only = std::env::var("CASES").unwrap_or_default();
    let energies: Vec<f64> = (0..=300).map(|i| 0.05 * i as f64 / HARTREE_EV).collect();
    for (label, cell, size, window) in cases {
        if !only.is_empty() && !only.split(',').any(|c| label.starts_with(c)) {
            continue;
        }
        let started = std::time::Instant::now();
        let solved = solve_sized(&cell, "raw", size);
        let sampler = solved.sampler();
        let mesh = MeshIndex::new([4; 3], [0.; 3]).unwrap();
        let mut line = format!("T5c {label} 4^3 window {window:?}:");
        for kernel in [
            Kernel::Independent,
            Kernel::Hartree,
            Kernel::LongRange,
            Kernel::Bse,
        ] {
            let mut request = Optical::new(energies.clone(), 0.1 / HARTREE_EV, kernel);
            request.window = Some(window);
            let out = dielectric(&sampler, &mesh, &request).unwrap();
            let along = out.along(None).unwrap();
            let (peak, height) = energies
                .iter()
                .zip(&along)
                .map(|(e, v)| (e * HARTREE_EV, v.im))
                .fold(
                    (0., f64::NEG_INFINITY),
                    |best, x| if x.1 > best.1 { x } else { best },
                );
            let first = energies
                .iter()
                .zip(&along)
                .position(|(_, v)| v.im > 0.5 * height)
                .map(|i| energies[i] * HARTREE_EV)
                .unwrap_or(f64::NAN);
            line += &format!(
                " {}: eps_M(0) {:.4}, peak {peak:.2} eV ({height:.2}), half-height onset {first:.2} eV{};",
                kernel.name(),
                (out.static_tensor[0][0] + out.static_tensor[1][1] + out.static_tensor[2][2]) / 3.,
                out.alpha.map(|a| format!(", alpha {a:.4}")).unwrap_or_default()
            );
        }
        // The Tamm-Dancoff exchange-only tensor: separates the excitonic effect from the TDA's.
        let mut request = Request::new(Kernel::Hartree, Spin::Singlet);
        request.window = Some(window);
        let op = operator(&sampler, &mesh, &request).unwrap();
        let all = op.dense().unwrap();
        let tda: Vec<C64> = energies
            .iter()
            .map(|e| {
                let t = op.tensor(&all, C64::new(*e, 0.1 / HARTREE_EV));
                (t[0][0] + t[1][1] + t[2][2]) / 3.
            })
            .collect();
        let (peak, height) = energies
            .iter()
            .zip(&tda)
            .map(|(e, v)| (e * HARTREE_EV, v.im))
            .fold(
                (0., f64::NEG_INFINITY),
                |best, x| if x.1 > best.1 { x } else { best },
            );
        let t0 = op.tensor(&all, C64::new(0., 0.));
        line += &format!(
            " hartree (TDA): eps_M(0) {:.4}, peak {peak:.2} eV ({height:.2});",
            (t0[0][0].re + t0[1][1].re + t0[2][2].re) / 3.
        );
        eprintln!("{line} {:.0} s", started.elapsed().as_secs_f64());
    }
}

#[test]
fn the_lithium_hydride_sz_momentum_is_hermitian_where_it_vanishes_by_symmetry() {
    use mandacaru_core::periodic::optics::momentum::{one_center_momentum, transitions};
    // Rock-salt LiH in SZ has two s-like bands: at the inversion-invariant points of the mesh
    // `P` vanishes (largest element 4e-16) and was refused for 5.6e-2 of that, rounding, though
    // its grid part (A^dagger diag(w) A, real symbols) and its sphere part are Hermitian.
    let solved = solve_sized(&rocksalt(["Li", "H"], 7.718, 16, 4), "raw", "SZ");
    let sampler = solved.sampler();
    let mesh = MeshIndex::new([4; 3], [0.; 3]).unwrap();
    let one_center = one_center_momentum(&solved.ctx, &solved.setup).unwrap();
    let block = one_center
        .iter()
        .map(|b| (b - b.adjoint()).camax() / b.camax().max(1e-300))
        .fold(0f64, f64::max);
    let all: Vec<Vec<usize>> = (0..sampler.channels())
        .map(|_| (0..solved.crystal.bands()).collect())
        .collect();
    let mut sphere = 0f64;
    sampler
        .visit(&mesh, &all, |_, frame| {
            for b in &one_center {
                let part = &frame.projections * b * frame.projections.adjoint();
                sphere = sphere.max((&part - part.adjoint()).camax());
            }
            Ok(())
        })
        .unwrap();
    let residual = transitions(&sampler, &mesh, [1., 0., 0.]).unwrap().residual;
    eprintln!(
        "LiH SZ grid 16^3, 4^3 mesh: Delta p anti-Hermitian {block:.2e} (relative), sphere part \
         {sphere:.2e} (absolute), P residual {residual:.2e}"
    );
    assert!(
        block < 1e-10 && sphere < 1e-12 && residual < 1e-6,
        "{residual:e}"
    );
}
