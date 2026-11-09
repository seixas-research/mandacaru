//! The screened exact exchange of a PAW atom in its own reference state, against
//! the all-electron value of the same valence orbitals.
//!
//! The basis is the dataset's own smooth bound partial waves, so the atom's
//! density matrix is the reference one exactly and the PAW exchange (grid pair
//! densities, compensation charges, one-center tensor) must reproduce the
//! all-electron short-range exchange of the valence shells, which is a radial
//! integral: `E_x = -1/2 sum_(l l') f_l f_l' (2l+1)(2l'+1) sum_L (l L l'; 000)^2
//! R^L_SR(l, l')` per spin, with `R^L_SR` the double radial integral of the
//! shell-pair densities under `erfc(omega r)/r`. The semilocal term the hybrid
//! replaces is not in this comparison, and the frozen linearization is exact at
//! the reference, which the last assertion checks. The difference of the two
//! exchange energies is the grid's and the partial waves' error alone, free of
//! bonding. `$MANDACARU_PAW_PATH` is required; a missing library FAILS the test.
mod paw_molecular_support;

use mandacaru_core::{
    math::spline::simpson,
    molecular::MolecularIntegrals,
    paw::{
        molecular::Arrays,
        onecenter_radial::{long_range_radial_kernel, radial_coulomb, three_j_squared},
        record::PawDataset,
    },
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use paw_molecular_support::{dataset, molecule, Shell};

const OMEGA: f64 = 0.11;
const FRACTION: f64 = 0.25;

fn bessel(l: usize, x: f64) -> f64 {
    if x < 1e-4 {
        // The leading series x^l / (2l+1)!!.
        return x.powi(l as i32) / (1..=l).map(|k| (2 * k + 1) as f64).product::<f64>();
    }
    match l {
        0 => x.sin() / x,
        1 => x.sin() / (x * x) - x.cos() / x,
        2 => (3. / (x * x) - 1.) * x.sin() / x - 3. * x.cos() / (x * x),
        _ => panic!("the test needs l up to 2"),
    }
}

/// The smooth bound partial wave of channel `l`: the Bessel expansion inside the
/// cutoff radius, the all-electron wave outside it.
fn smooth_wave(data: &PawDataset, l: u32) -> Vec<f64> {
    let channel = &data.channels[&l];
    let (q, c) = (
        channel.wavevectors[0].data(),
        channel.wave_coefficients[0].data(),
    );
    let ae = channel.ae_waves[0].as_slice();
    data.r
        .as_slice()
        .iter()
        .zip(ae)
        .map(|(&r, &ae)| {
            if r <= channel.r_cut {
                q.iter()
                    .zip(c)
                    .map(|(q, c)| c * bessel(l as usize, q * r))
                    .sum()
            } else {
                ae
            }
        })
        .collect()
}

/// The all-electron short-range exchange of the valence shells (both spins), Hartree.
fn all_electron_exchange(ctx: &Context, data: &PawDataset) -> f64 {
    let r = data.r.as_slice();
    let weights: Vec<f64> = (0..r.len())
        .map(|i| {
            let mut unit = vec![0.; r.len()];
            unit[i] = 1.;
            simpson(r, &unit).unwrap() * r[i] * r[i]
        })
        .collect();
    let levels: Vec<usize> = (0..=2 * data.channels.len().saturating_sub(1)).collect();
    let kernels = long_range_radial_kernel(ctx, r, r, &levels, OMEGA, None).unwrap();
    let mut per_spin = 0.;
    for (&l1, c1) in &data.channels {
        for (&l2, c2) in &data.channels {
            let f = |l: u32, c: &mandacaru_core::paw::record::Channel| {
                c.occupations.data()[0] / (2. * (2 * l + 1) as f64)
            };
            let (f1, f2) = (f(l1, c1), f(l2, c2));
            let rho: Vec<f64> = c1.ae_waves[0]
                .as_slice()
                .iter()
                .zip(c2.ae_waves[0].as_slice())
                .map(|(a, b)| a * b)
                .collect();
            for big in (l1.abs_diff(l2))..=(l1 + l2) {
                let three_j = three_j_squared(l1 as usize, big as usize, l2 as usize).unwrap();
                if three_j == 0. {
                    continue;
                }
                let bare = radial_coulomb(r, &rho, &rho, big as usize).unwrap();
                let kernel = &kernels[big as usize];
                let weighted: Vec<f64> = rho.iter().zip(&weights).map(|(a, w)| a * w).collect();
                let long: f64 = (0..r.len())
                    .map(|i| {
                        weighted[i]
                            * (0..r.len())
                                .map(|j| kernel[i * r.len() + j] * weighted[j])
                                .sum::<f64>()
                    })
                    .sum();
                per_spin -=
                    0.5 * f1 * f2 * ((2 * l1 + 1) * (2 * l2 + 1)) as f64 * three_j * (bare - long);
            }
        }
    }
    2. * per_spin
}

/// The atom on a grid of spacing `h` Bohr with the smooth partial waves as its
/// basis, and its reference density matrix (per spin) in the orthonormal frame.
fn reference_atom(
    ctx: &Context,
    data: &PawDataset,
    number: u8,
    h: f64,
) -> (MolecularIntegrals, DMatrix<C64>) {
    let mut arrays = Arrays::new();
    let species = dataset::species(ctx, &mut arrays, data);
    let shells: Vec<Shell> = data
        .channels
        .keys()
        .map(|&l| (l as usize, data.r.as_slice().to_vec(), smooth_wave(data, l)))
        .collect();
    let base = molecule(
        ctx,
        &[number],
        &[[0., 0., 0.]],
        if h < 0.2 { 12. } else { 16. },
        h,
        (vec![species], vec![0], arrays),
        "dual",
        &shells,
    );
    // The reference occupation per spin and per m of each function, in the order
    // of the basis (shell by shell, m = -l..l).
    let mut diagonal = Vec::new();
    for (&l, channel) in &data.channels {
        let f = channel.occupations.data()[0] / (2. * (2 * l + 1) as f64);
        diagonal.extend(std::iter::repeat_n(f, 2 * l as usize + 1));
    }
    let basis = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(diagonal));
    let inverse = base.lowdin.clone().try_inverse().unwrap();
    let orthonormal = &inverse * basis * inverse.transpose();
    (base, orthonormal.map(|v| C64::new(v, 0.)))
}

/// The PAW exchange pieces of the atom at the reference density, grid spacing
/// `h` in Bohr: `(a E_x, X0 linearization error)`.
fn paw_exchange(ctx: &Context, set: &str, element: &str, number: u8, h: f64) -> (f64, f64) {
    let data = dataset::read(ctx, set, element);
    let (base, d) = reference_atom(ctx, &data, number, h);
    let pieces = base
        .exchange_pieces("hse06", std::slice::from_ref(&d))
        .unwrap()
        .unwrap();
    println!("{element} h {h}: {pieces:?}");
    let exchange = pieces.grid + pieces.one_center + pieces.compensation;
    (exchange, pieces.linearization)
}

#[test]
#[ignore = "slow: 13 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_molecular_reference_atom -- --ignored the_oxygen_atom_exchange_at_its_reference_density_matches_the_all_electron_value"]
fn the_oxygen_atom_exchange_at_its_reference_density_matches_the_all_electron_value() {
    // Measured: PAW - AE = -1.4 mHa at h 0.55 and 0.45 Bohr (grid noise), then
    // +55 uHa (0.35), -21 uHa (0.30) and -3 uHa (0.25).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let data = dataset::read(&ctx, "pbe-sr", "O");
    let ae = FRACTION * all_electron_exchange(&ctx, &data);
    println!("O all-electron valence a E_x^SR = {ae:.9} Ha");
    for (h, tolerance) in [(0.35, 2e-4), (0.25, 3e-5)] {
        let (paw, linearization) = paw_exchange(&ctx, "pbe-sr", "O", 8, h);
        println!(
            "O h {h}: PAW a E_x^SR = {paw:.9}, difference PAW - AE = {:.3e} Ha, linearization at D0 = {linearization:.2e}",
            paw - ae
        );
        assert!((paw - ae).abs() < tolerance, "h {h}: {} Ha", paw - ae);
        // The frozen semilocal term is exact at the reference density.
        assert!(linearization.abs() < 1e-9);
    }
}

#[test]
#[ignore = "slow: 32 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_molecular_reference_atom -- --ignored the_neon_atom_exchange_at_its_reference_density_converges_to_the_all_electron_value"]
fn the_neon_atom_exchange_at_its_reference_density_converges_to_the_all_electron_value() {
    // Neon's smooth p waves are the hardest of the first row: the exchange of
    // their pair densities is grid-limited (measured PAW - AE: -138, -11, -4, -8,
    // -9 and +1 mHa at h 0.55, 0.45, 0.35, 0.30, 0.25 and 0.20 Bohr) and then
    // converges (-34, -61 and -8 uHa at h 0.17, 0.15 and 0.13 Bohr). HSE06 - PBE
    // on the same density is +20.35 mHa from h 0.17 on, against +25.36 mHa for the
    // all-electron valence-only value (valence-only exact and short-range
    // semilocal exchange, hole(0) full-range term on the valence density, no
    // core-valence exchange): the converged exchange is right, so the missing
    // 5.0 mHa is in the semilocal terms (see the PAW migration log).
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let data = dataset::read(&ctx, "pbe-sr", "Ne");
    let ae = FRACTION * all_electron_exchange(&ctx, &data);
    println!("Ne all-electron valence a E_x^SR = {ae:.9} Ha");
    for h in [0.35, 0.17] {
        let (base, d) = reference_atom(&ctx, &data, 10, h);
        let slice = std::slice::from_ref(&d);
        let pbe = base.kohn_sham_energy("pbe", slice).unwrap();
        let hse = base.kohn_sham_energy("hse06", slice).unwrap();
        let pieces = base.exchange_pieces("hse06", slice).unwrap().unwrap();
        let difference = pieces.grid + pieces.one_center + pieces.compensation - ae;
        println!(
            "Ne h {h}: HSE06 - PBE at D0 = {:.4} mHa; PAW a E_x^SR - AE = {difference:.3e} Ha",
            1e3 * (hse - pbe)
        );
        assert!(pieces.linearization.abs() < 1e-9);
        if h < 0.2 {
            assert!(difference.abs() < 2e-4, "h {h}: {difference:e} Ha");
        }
    }
}
