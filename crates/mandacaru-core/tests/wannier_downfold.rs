//! The bare downfolded problem of a Wannier fragment (`periodic::wannier::downfold`).
//!
//! The system is an H2 molecule in a chain cell (a mechanics state: the density is the Gamma-only
//! SCF one; the Wannier functions are built on the full mesh by the production path). Gates:
//!
//! * the Fourier grid assembly against a brute-force real-space double sum of the smooth pair
//!   densities with the real-space truncated kernel `theta(R_c - r) / r` (an independent
//!   discretization of the same integral: agreement to the voxel quadrature, 1e-2 relative);
//! * the truncated kernel and the supercell frequencies against the closed form of two Gaussian
//!   charges (`erf(d / 2 sigma) / d`), which needs no crystal;
//! * the sum rules of the pair densities: `<w_p|S|w_q> = delta` from the grid overlap plus the
//!   overlap correction, and the same from the zeroth Fourier component with the compensation
//!   charges;
//! * translation invariance (the fragment moved by a lattice vector is the same problem), the
//!   pair symmetries of `<pq|rs>`, the Fock identity `F[gamma] = h^KS` against an independent
//!   Fock build, and `tr(gamma h)` against the k-space sum, which a transposed `gamma` fails on
//!   a complex mesh.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;
mod wannier_downfold_support;

use mandacaru_core::{
    math::fft::Fft3,
    periodic::wannier::{
        downfold::{
            coulomb::{truncated, Cell},
            CoulombBox, Request, Screening,
        },
        setup::Wannier,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use std::f64::consts::PI;
use wannier_downfold_support::{chain, rephase, request};

fn at(n: usize, p: usize, q: usize, r: usize, s: usize) -> usize {
    ((p * n + q) * n + r) * n + s
}

#[test]
fn the_pair_densities_carry_the_overlap_and_the_fragment_is_a_neutral_two_electron_problem() {
    let (_ctx, wannier) = chain([2, 1, 1]);
    let result = wannier.downfold(&request(None)).unwrap();
    let identity = DMatrix::<C64>::identity(2, 2);
    for (metric, charge) in &result.norms {
        assert!((metric - &identity).norm() < 1e-8, "overlap {metric}");
        assert!((charge - &identity).norm() < 1e-8, "charge {charge}");
    }
    let problem = &result.problem;
    assert_eq!(problem.num_particles, [1, 1]);
    assert!((problem.density_matrix[0].trace().re - 2.).abs() < 1e-3);
    // the pair symmetries of <pq|rs>: electron exchange and Hermiticity
    let g = &problem.wannier.two_body[0];
    let n = problem.n;
    let scale = g.iter().map(|v| v.norm()).fold(0., f64::max);
    let (mut swap, mut hermitian) = (0f64, 0f64);
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let a = g[at(n, p, q, r, s)];
                    swap = swap.max((a - g[at(n, q, p, s, r)]).norm());
                    hermitian = hermitian.max((a - g[at(n, r, s, p, q)].conj()).norm());
                }
            }
        }
    }
    println!(
        "pair symmetries: electron swap {swap:.2e}, Hermiticity {hermitian:.2e} of {scale:.3}"
    );
    assert!(swap < 1e-12 * scale && hermitian < 1e-12 * scale);
    // an on-site repulsion of a hydrogen-like function, in eV, is positive and of order 10 eV
    let u = g[at(n, 0, 0, 0, 0)].re * 27.211386;
    assert!(u > 3. && u < 30., "U = {u} eV");
    println!("H2 chain, bare U = {u:.4} eV, radius {:.3}", result.radius);
}

#[test]
fn the_fragment_is_translation_invariant_and_the_fock_identity_holds() {
    let (_ctx, mut wannier) = chain([3, 1, 1]);
    rephase(&mut wannier, [0.0, 0.8]);
    let home = wannier.downfold(&request(None)).unwrap().problem;
    let moved = wannier
        .downfold(&request(Some(vec![(0, [1, 0, 0]), (1, [1, 0, 0])])))
        .unwrap()
        .problem;
    let n = home.n;
    assert!(
        (&home.kohn_sham[0] - &moved.kohn_sham[0]).norm() < 1e-9,
        "h moves with the fragment"
    );
    assert!((&home.density_matrix[0] - &moved.density_matrix[0]).norm() < 1e-9);
    let scale = home.wannier.two_body[0]
        .iter()
        .map(|v| v.norm())
        .fold(0., f64::max);
    let miss = home.wannier.two_body[0]
        .iter()
        .zip(&moved.wannier.two_body[0])
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    assert!(miss < 1e-8 * scale, "translation moved <pq|rs> by {miss}");
    // F[gamma] = h^KS from an independent Fock build, on a complex mesh
    let g = &home.wannier.two_body[0];
    let gamma = &home.density_matrix[0];
    let mut fock = home.wannier.one_body[0].clone();
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    // gamma_rs = <c_r^+ c_s>: F_pq = h_pq + sum gamma_rs (<pr|qs> - 1/2 <pr|sq>)
                    fock[(p, q)] +=
                        gamma[(r, s)] * (g[at(n, p, r, q, s)] - 0.5 * g[at(n, p, r, s, q)]);
                }
            }
        }
    }
    assert!((&fock - &home.kohn_sham[0]).norm() < 1e-10, "Fock identity");
    // sum_pq gamma_pq h_pq over every function of every cell is the band energy
    // sum_k sum_b g f eps, from the frames alone. gamma_pq = <c_p^+ c_q> = rho_qp, so the sum is
    // tr(rho h); the untransposed density matrix gives tr(rho^T h), which differs when h is
    // complex (here: a gauge with a relative phase between the two functions).
    let everything: Vec<(usize, [i64; 3])> = (0..3)
        .flat_map(|c| (0..2).map(move |f| (f, [c as i64, 0, 0])))
        .collect();
    let full = wannier
        .downfold(&request(Some(everything)))
        .unwrap()
        .problem;
    assert_eq!(full.num_particles, [3, 3]);
    let channel = &wannier.channels[0];
    let mut band_energy = 0.;
    for frame in &channel.frames {
        for e in frame.energies.as_slice() {
            band_energy += 2.
                * wannier
                    .state
                    .options
                    .smearing
                    .occupation((e - wannier.state.fermi) / wannier.state.options.width)
                * e;
        }
    }
    let trace = |a: &DMatrix<C64>, b: &DMatrix<C64>| (a * b).trace();
    // sum_pq gamma_pq h_pq (elementwise) = tr(gamma^T h)
    let right = trace(&full.density_matrix[0].transpose(), &full.kohn_sham[0]);
    let wrong = trace(&full.density_matrix[0], &full.kohn_sham[0]);
    assert!(
        (right.re - band_energy).abs() < 1e-9 && right.im.abs() < 1e-9,
        "tr(gamma h) = {right} against the band energy {band_energy}"
    );
    assert!(
        (wrong.re - band_energy).abs() > 1e-7,
        "the mesh is too real to tell gamma from its transpose: {wrong} against {band_energy}"
    );
}

/// The smooth pair densities `w_p^* w_q` on the supercell grid, from the frames and the gauge
/// alone: `w_m(r + T) = (1/N) sum_k e^{ik.T} sum_b U_bm psi_bk(r)` written out directly, a mode
/// `(m, R)` read from the home function at `X - R * shape` (wrapped).
fn smooth_densities(wannier: &Wannier, modes: &[(usize, [i64; 3])]) -> (Cell, Vec<Vec<C64>>) {
    let crystal = &wannier.crystal;
    let ctx = &crystal.context;
    let cell = Cell::of(crystal, wannier.size).unwrap();
    let n = crystal.grid.index.points();
    let channel = &wannier.channels[0];
    let mut home = vec![vec![C64::default(); cell.points()]; 2];
    for (k, frame) in channel.frames.iter().enumerate() {
        let gauge = &channel.localized.gauge[k];
        for t in 0..cell.cells {
            let tv = cell.cell_vector(t);
            let phase = C64::from_polar(
                1. / cell.cells as f64,
                2. * PI
                    * (0..3)
                        .map(|a| frame.fractional[a] * tv[a] as f64)
                        .sum::<f64>(),
            );
            for m in 0..2 {
                for b in 0..2 {
                    let sample = frame.samples.read_range(ctx, b * n, n).unwrap();
                    for j in 0..n {
                        let local = [
                            j / (cell.shape[1] * cell.shape[2]),
                            (j / cell.shape[2]) % cell.shape[1],
                            j % cell.shape[2],
                        ];
                        let x =
                            cell.point_of([tv[0] as usize, tv[1] as usize, tv[2] as usize], local);
                        home[m][x] += phase * gauge[(b, m)] * sample[j];
                    }
                }
            }
        }
    }
    let total = cell.total;
    let moved = |(m, r): (usize, [i64; 3]), i: usize| -> C64 {
        let x = [
            i / (total[1] * total[2]),
            (i / total[2]) % total[1],
            i % total[2],
        ];
        let source: [usize; 3] = std::array::from_fn(|a| {
            (x[a] as i64 - r[a] * cell.shape[a] as i64).rem_euclid(total[a] as i64) as usize
        });
        home[m][(source[0] * total[1] + source[1]) * total[2] + source[2]]
    };
    let mut out = vec![];
    for p in 0..modes.len() {
        for q in 0..modes.len() {
            out.push(
                (0..cell.points())
                    .map(|i| moved(modes[p], i).conj() * moved(modes[q], i))
                    .collect(),
            );
        }
    }
    (cell, out)
}

/// A real-space double sum of the smooth pair densities with `theta(R_c - r) / r` on the minimum
/// image and the cubic voxel's self term `2.3800 / h` at `r = 0`.
fn real_space(cell: &Cell, grid_step: f64, left: &[C64], right: &[C64]) -> f64 {
    let h = grid_step;
    let total = cell.total;
    let index = |i: usize| {
        [
            i / (total[1] * total[2]),
            (i / total[2]) % total[1],
            i % total[2],
        ]
    };
    let extent: [f64; 3] = std::array::from_fn(|a| total[a] as f64 * h);
    let keep = |v: &[C64]| -> Vec<usize> {
        let top = v.iter().map(|z| z.norm()).fold(0., f64::max);
        (0..v.len()).filter(|i| v[*i].norm() > 1e-9 * top).collect()
    };
    let (ia, ib) = (keep(left), keep(right));
    let place = |i: usize| -> [f64; 3] {
        let x = index(i);
        std::array::from_fn(|a| x[a] as f64 * h)
    };
    let positions: Vec<[f64; 3]> = ib.iter().map(|&j| place(j)).collect();
    let mut sum = C64::default();
    for &i in &ia {
        let pi = place(i);
        for (&j, pj) in ib.iter().zip(&positions) {
            let mut d2 = 0.;
            for c in 0..3 {
                let mut d = pi[c] - pj[c];
                d -= extent[c] * (d / extent[c]).round();
                d2 += d * d;
            }
            let r = d2.sqrt();
            let kernel = if r < 1e-12 {
                2.3800 / h
            } else if r <= cell.half_shortest {
                1. / r
            } else {
                0.
            };
            sum += left[i] * right[j] * kernel;
        }
    }
    (sum * cell.dv * cell.dv).re
}

/// `dv e^{-i G.r0} sum_x rho(x) e^{-2 pi i m.x / N}` by one naive transform per axis (no FFT), with
/// the signed frequencies of the grid (the Nyquist one negative), and the sum of
/// `conj(rho_a) v_c rho_b / Omega`.
fn direct_fourier(cell: &Cell, left: &[C64], right: &[C64]) -> f64 {
    let total = cell.total;
    let transform = |rho: &[C64]| -> Vec<C64> {
        let mut data = rho.to_vec();
        for axis in 0..3 {
            let n = total[axis];
            let stride: usize = total[axis + 1..].iter().product();
            let outer = data.len() / (n * stride);
            let mut next = vec![C64::default(); data.len()];
            for o in 0..outer {
                for s in 0..stride {
                    for m in 0..n {
                        let mut sum = C64::default();
                        for x in 0..n {
                            let angle = -2. * PI * ((m * x) % n) as f64 / n as f64;
                            sum += data[(o * n + x) * stride + s] * C64::from_polar(1., angle);
                        }
                        next[(o * n + m) * stride + s] = sum;
                    }
                }
            }
            data = next;
        }
        data
    };
    let (a, b) = (transform(left), transform(right));
    let mut energy = 0.;
    let freq = |i: usize, n: usize| {
        if i < n.div_ceil(2) {
            i as f64
        } else {
            i as f64 - n as f64
        }
    };
    for g in 0..a.len() {
        let m = [
            g / (total[1] * total[2]),
            (g / total[2]) % total[1],
            g % total[2],
        ];
        let k = cell.reciprocal
            * nalgebra::Vector3::new(
                freq(m[0], total[0]),
                freq(m[1], total[1]),
                freq(m[2], total[2]),
            );
        let phase = C64::from_polar(
            1.,
            -(k[0] * cell.origin[0] + k[1] * cell.origin[1] + k[2] * cell.origin[2]),
        );
        // the origin phase cancels in conj(rho_a) rho_b
        let _ = phase;
        // a Nyquist plane carries no interaction (it has no partner at -G)
        let nyquist = (0..3)
            .any(|a| total[a] % 2 == 0 && 2 * (freq(m[a], total[a]).abs() as usize) == total[a]);
        if nyquist {
            continue;
        }
        energy += (a[g].conj() * b[g]).re * truncated(k.norm_squared(), cell.half_shortest);
    }
    energy * cell.dv * cell.dv / cell.volume
}

#[test]
fn the_grid_integrals_equal_an_independent_fourier_sum_and_a_real_space_double_sum() {
    let (_ctx, wannier) = chain([2, 1, 1]);
    let modes = vec![(0usize, [0i64; 3]), (1, [0, 0, 0]), (0, [1, 0, 0])];
    let n = modes.len();
    let grid = wannier.grid_integrals(0, &modes).unwrap();
    let (cell, rho) = smooth_densities(&wannier, &modes);
    let h = wannier.crystal.grid.step.column(0).norm();
    let ev = 27.211386;
    let pairs = [(0, 0, 0, 0), (0, 1, 1, 0), (0, 0, 2, 2), (0, 2, 2, 0)];
    let mut largest = 0f64;
    let mut table = vec![];
    for &(p, q, r, s) in &pairs {
        let got = grid[at(n, p, q, r, s)];
        // (pq|rs) = sum_G conj(rho_qp) v rho_rs: the left density is the swapped pair
        let fourier = direct_fourier(&cell, &rho[q * n + p], &rho[r * n + s]);
        let real = real_space(&cell, h, &rho[p * n + q], &rho[r * n + s]);
        largest = largest.max(got.re.abs());
        println!(
            "({p}{q}|{r}{s}): production {:.6} eV, direct Fourier {:.6} eV, real space {:.6} eV",
            got.re * ev,
            fourier * ev,
            real * ev
        );
        assert!(
            (got.re - fourier).abs() < 1e-9,
            "({p}{q}|{r}{s}) {} against the direct transform {fourier}",
            got.re
        );
        assert!(got.im.abs() < 1e-9);
        table.push((got.re, real));
    }
    // The continuum integral differs by what the grid does to the pair density (its products
    // alias past the Nyquist wave number) and to the 1/r kernel (voxel quadrature): measured
    // 0.5 percent of the largest integral here.
    for (got, real) in table {
        assert!((got - real).abs() < 1e-2 * largest, "{got} against {real}");
    }
}

#[test]
fn two_gaussian_charges_see_the_truncated_kernel_at_their_closed_form() {
    let ctx = Context::new(Some(1), 1 << 28).unwrap();
    let lattice = Matrix3::from_diagonal(&nalgebra::Vector3::new(14., 14., 14.));
    let shape = [56, 56, 56];
    let dv = (14. / 56f64).powi(3);
    let cell = Cell::new(lattice, shape, [1, 1, 1], dv, [0.3, -0.2, 0.1]).unwrap();
    let sigma = 0.5;
    let gaussian = |center: [f64; 3]| -> Vec<C64> {
        let norm = (2. * PI * sigma * sigma).powf(-1.5);
        (0..cell.points())
            .map(|i| {
                let ix = [i / (56 * 56), (i / 56) % 56, i % 56];
                let r: Vec<f64> = (0..3)
                    .map(|a| {
                        let x = cell.origin[a] + ix[a] as f64 * 14. / 56.;
                        let mut d = x - center[a];
                        d -= 14. * (d / 14.).round();
                        d
                    })
                    .collect();
                C64::new(
                    norm * (-(r.iter().map(|v| v * v).sum::<f64>()) / (2. * sigma * sigma)).exp(),
                    0.,
                )
            })
            .collect()
    };
    let transform = |mut rho: Vec<C64>| -> Vec<C64> {
        Fft3::new(&ctx, shape)
            .unwrap()
            .transform(&mut rho, false)
            .unwrap();
        for (g, v) in rho.iter_mut().enumerate() {
            let k = cell.wavevector(g);
            let angle = -(k[0] * cell.origin[0] + k[1] * cell.origin[1] + k[2] * cell.origin[2]);
            *v *= C64::from_polar(cell.dv, angle);
        }
        rho
    };
    let radius = cell.half_shortest;
    for d in [0.0, 1.3, 2.4] {
        let a = transform(gaussian([7., 7., 7.]));
        let b = transform(gaussian([7. + d, 7., 7.]));
        let mut energy = 0.;
        for g in 0..cell.points() {
            let k = cell.wavevector(g);
            let g2 = k.iter().map(|v| v * v).sum::<f64>();
            energy += (a[g].conj() * b[g]).re * truncated(g2, radius);
        }
        energy /= cell.volume;
        let exact = if d == 0. {
            1. / (sigma * PI.sqrt())
        } else {
            statrs_erf(d / (2. * sigma)) / d
        };
        println!("d = {d}: {energy:.12} against {exact:.12}");
        // the truncation at R_c = 7 leaves erfc(R_c / 2 sigma)-sized charge outside: ~1e-9
        assert!((energy - exact).abs() < 5e-9, "d = {d}");
    }
}

/// erf by its Maclaurin series with the continued remainder, accurate to ~1e-14 for x < 3.
fn statrs_erf(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..80 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    2. / PI.sqrt() * sum
}

#[test]
fn incoherent_screening_options_are_refused_by_name() {
    let (_ctx, wannier) = chain([2, 1, 1]);
    for bad in [
        Request {
            cutoff: Some(2.),
            ..request(None)
        },
        Request {
            bands: Some(2),
            ..request(None)
        },
        Request {
            screening: Some(Screening::Rpa),
            cutoff: Some(-1.),
            ..request(None)
        },
        Request {
            screening: Some(Screening::Rpa),
            bands: Some(0),
            ..request(None)
        },
    ] {
        assert!(matches!(
            wannier.downfold(&bad),
            Err(mandacaru_core::Error::Invalid(_))
        ));
    }
    // more states than the Wannier calculation kept cannot screen
    let refused = wannier.downfold(&Request {
        screening: Some(Screening::Rpa),
        bands: Some(9),
        ..request(None)
    });
    assert!(matches!(
        refused,
        Err(mandacaru_core::Error::Unsupported(_))
    ));
}

#[test]
fn the_isolated_box_keeps_the_sum_rules_and_agrees_with_the_supercell_for_a_compact_fragment() {
    let (_ctx, wannier) = chain([3, 1, 1]);
    let alone = wannier
        .downfold(&Request {
            coulomb: CoulombBox::Isolated,
            ..request(Some(vec![(0, [0; 3])]))
        })
        .unwrap();
    // the function, cut at its supercell's Wigner-Seitz cell and set in the doubled box, is still
    // normalized to what the cut leaves (this 8 Bohr cell keeps 99.8 percent of the charge)
    let identity = DMatrix::<C64>::identity(1, 1);
    for (metric, charge) in &alone.norms {
        assert!((metric - &identity).norm() < 1e-2, "overlap {metric}");
        assert!(
            (metric - &identity).norm() > 1e-5,
            "the cut must remove something: {metric}"
        );
        assert!((charge - &identity).norm() < 1e-2, "charge {charge}");
    }
    // the box's radius is the supercell's shortest vector, twice the plain one
    let plain = wannier.downfold(&request(Some(vec![(0, [0; 3])]))).unwrap();
    assert!((alone.radius - 2. * plain.radius).abs() < 1e-12);
    // the repulsion of one hydrogen-like function: the isolated box has no periodic copies to
    // add to it, the supercell's truncation at 4 Bohr cuts the function's own tail: a few tenths
    // of an eV apart, the same sign and size
    let ev = 27.211386;
    let (a, b) = (
        alone.problem.wannier.two_body[0][0].re * ev,
        plain.problem.wannier.two_body[0][0].re * ev,
    );
    println!("U isolated {a:.4} eV, supercell {b:.4} eV");
    assert!((a - b).abs() < 0.8 && a > 14. && a < 22., "{a} against {b}");
}
