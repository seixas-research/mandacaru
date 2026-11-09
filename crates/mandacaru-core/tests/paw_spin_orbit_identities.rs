//! What the spin-orbit block must satisfy besides matching the freeze: the Dirac atom's levels
//! average to its scalar levels, each `j` is recovered exactly from the stored average and
//! `L.S` pair, the reference levels of the two branches split as the Dirac atom's do, and the
//! `(2j+1)`-weighted average of the branches reduces to the scalar channel. The shipped
//! `lda-fr` file is a read-only cross-check where its recipe is reproducible.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_spin_orbit_support;

use mandacaru_core::models::configuration::valence_subshells;
use mandacaru_core::models::relativity::{degeneracy, SPEED_OF_LIGHT};
use mandacaru_core::paw::generate::spin_orbit::{branches, kappas, lowest_level, recover, weights};
use mandacaru_core::paw::record::Array;
use nalgebra::DMatrix;
use paw_spin_orbit_support::items::Owned;
use paw_spin_orbit_support::{context, loaded, open, Case, LIGHT};

/// Tolerance of the first-order estimate, from the measured sodium, iron 3p and iron 3d values.
const FIRST_ORDER: f64 = 0.05;
/// Tolerance of the j average against the scalar channel, as a fraction of the j splitting.
const REDUCTION: f64 = 0.02;

fn matrix(a: &Array) -> DMatrix<f64> {
    DMatrix::from_row_slice(a.shape()[0], a.shape()[1], a.data())
}

fn scale(m: &DMatrix<f64>) -> f64 {
    m.iter().fold(0., |s, x| f64::max(s, x.abs())).max(1e-300)
}

/// The `(2j+1)`-weighted average of the Dirac levels of each `(n, l)` is the level of the
/// scalar atom the construction is built on.
#[test]
#[ignore = "slow: 17 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit_identities -- --ignored the_dirac_levels_of_the_atom_average_to_its_scalar_levels"]
fn the_dirac_levels_of_the_atom_average_to_its_scalar_levels() {
    let ctx = context();
    for name in LIGHT {
        let case = open(&ctx, name);
        let mut worst: f64 = 0.;
        for (i, &(n, l, _)) in case.atom.occupations.iter().enumerate() {
            let levels = case
                .atom
                .j_states
                .iter()
                .filter(|j| (j.n, j.l) == (n, l))
                .collect::<Vec<_>>();
            assert_eq!(levels.len(), if l == 0 { 1 } else { 2 }, "{name} {n}{l}");
            let total = levels
                .iter()
                .map(|j| degeneracy(j.kappa) as f64)
                .sum::<f64>();
            let mean = levels
                .iter()
                .map(|j| degeneracy(j.kappa) as f64 * j.eigenvalue)
                .sum::<f64>()
                / total;
            worst = worst.max((mean - case.atom.eigenvalues[i]).abs() / mean.abs().max(1.));
        }
        println!("{name}: j average against scalar level {worst:.2e}");
        assert!(worst < 1e-13, "{name}: {worst:e}");
    }
}

/// `V_j = V_avg + V_SO (L.S)_j`: each branch comes back from the stored pair on its own block,
/// and the blocks do not couple.
#[test]
fn each_j_is_recovered_exactly_from_the_stored_average_and_coupling() {
    let shared = loaded("lda-fr_Na");
    let (ctx, case, generated) = (&shared.ctx, &shared.case, &shared.generated);
    let owned = Owned::of(case, generated);
    let inputs = owned.inputs(case, generated);
    let mut worst: f64 = 0.;
    for (&l, so) in &generated.dataset.spin_orbit {
        let l = l as usize;
        let [down, up] = branches(ctx, &inputs, l).unwrap();
        let count = down.projectors.len();
        for (stored_average, stored_coupling, got) in [
            (&so.average, &so.coupling, (&down.ionic, &up.ionic)),
            (
                &so.average_screened,
                &so.coupling_screened,
                (&down.screened, &up.screened),
            ),
            (
                &so.overlap_average,
                &so.overlap_coupling,
                (&down.overlap, &up.overlap),
            ),
        ] {
            let (average, coupling) = (matrix(stored_average), matrix(stored_coupling));
            for (up_branch, expected) in [(false, got.0), (true, got.1)] {
                let back = recover(l, &average, &coupling, up_branch, count);
                worst = worst.max((back - expected).amax() / scale(expected));
            }
            // The off-diagonal blocks are exactly zero.
            for i in 0..count {
                for j in 0..count {
                    assert_eq!(average[(i, count + j)], 0.);
                    assert_eq!(coupling[(count + j, i)], 0.);
                }
            }
        }
    }
    println!("Na: branches recovered from (average, coupling) to {worst:.2e}");
    assert!(worst < 1e-14, "{worst:e}");
}

/// Whether the atom re-solved the `(n, l, kappa)` level with the Numerov shoot: only the occupied
/// states of its valence are (`models::atom`); the others keep the tridiagonal level, which sits
/// about 1e-3 Ha from the shot one (Na 2p, Fe 3p, the empty Si 3p3/2).
fn numerov_level(case: &Case, n: usize, l: usize, kappa: i32) -> Option<f64> {
    let configuration = case
        .atom
        .occupations
        .iter()
        .filter(|s| s.2 > 0.)
        .copied()
        .collect::<Vec<_>>();
    let j = case
        .atom
        .j_states
        .iter()
        .find(|j| (j.n, j.l, j.kappa) == (n, l, kappa))?;
    (j.occupation > 0. && valence_subshells(&configuration).contains(&(n, l)))
        .then_some(j.eigenvalue)
}

/// The first reference level of each branch against the Dirac atom's `(n, l, kappa)` level. Where
/// the atom's level is a Numerov one (bitwise the same equation and potential) the two are the
/// same number; elsewhere the atom's level is a grid-limited one and the difference is reported
/// (it is the discretization of the atom, not of the branch).
#[test]
#[ignore = "slow: 28 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit_identities -- --ignored the_branch_levels_are_the_dirac_atoms_where_the_atom_is_a_shot_level"]
fn the_branch_levels_are_the_dirac_atoms_where_the_atom_is_a_shot_level() {
    let mut gated = 0;
    for name in LIGHT {
        let shared = loaded(name);
        let (case, generated) = (&shared.case, &shared.generated);
        for (&l, so) in &generated.dataset.spin_orbit {
            let channel = &generated.dataset.channels[&l];
            let n = channel.n as usize;
            let count = so.reference_energies.shape()[1];
            let energies = so.reference_energies.data();
            for (row, kappa) in kappas(l as usize).into_iter().enumerate() {
                let Some(atom) = case
                    .atom
                    .j_states
                    .iter()
                    .find(|j| (j.n, j.l, j.kappa) == (n, l as usize, kappa))
                else {
                    println!("{name} l={l} kappa={kappa}: the atom has no occupied level (empty channel)");
                    continue;
                };
                let ours = energies[row * count];
                let shot = numerov_level(case, n, l as usize, kappa);
                println!(
                    "{name} l={l} (n={n}) kappa={kappa:+}: branch {ours:+.10} atom {:+.10} ({:+.2e}){}",
                    atom.eigenvalue,
                    ours - atom.eigenvalue,
                    if shot.is_some() { ", Numerov level" } else { ", tridiagonal level" }
                );
                if let Some(level) = shot {
                    assert!(
                        (ours - level).abs() < 1e-9,
                        "{name} l={l} kappa={kappa}: {:e}",
                        ours - level
                    );
                    gated += 1;
                }
            }
        }
    }
    assert!(gated >= 3, "only {gated} shot levels were compared");
}

/// The fine-structure splitting against first-order perturbation theory on the scalar
/// all-electron wave: `E(l+1/2) - E(l-1/2) = (2l+1)/2 <xi>`, `xi = (dV/dr) / (2 c^2 r)`. An
/// independent estimate (no Dirac solver involved), good to a few percent for the light shells it
/// is applied to; it is not applied where the splitting is a large relativistic effect.
#[test]
#[ignore = "slow: 28 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit_identities -- --ignored the_branch_splitting_agrees_with_first_order_perturbation_theory"]
fn the_branch_splitting_agrees_with_first_order_perturbation_theory() {
    let mut compared = 0;
    for (name, l) in [("lda-fr_Na", 1), ("lda-fr_Fe", 1), ("lda-fr_Fe", 2)] {
        let shared = loaded(name);
        let (case, generated) = (&shared.case, &shared.generated);
        let dataset = &generated.dataset;
        let so = &dataset.spin_orbit[&l];
        let r = dataset.r.as_slice();
        let v = case.atom.effective.as_slice();
        let wave = dataset.channels[&l].ae_waves[0].as_slice();
        let h = r[1] - r[0];
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        let mut mean = 0.;
        for i in 1..r.len() - 1 {
            let dv = (v[i + 1] - v[i - 1]) / (2. * h);
            mean += wave[i] * wave[i] * r[i] * r[i] * dv / (2. * c2 * r[i]) * h;
        }
        let estimate = (2 * l + 1) as f64 / 2. * mean;
        let count = so.reference_energies.shape()[1];
        let split = so.reference_energies.data()[count] - so.reference_energies.data()[0];
        println!(
            "{name} l={l}: splitting {split:.6e}, first order {estimate:.6e} ({:+.2}%)",
            100. * (estimate - split) / split
        );
        assert!(
            (estimate - split).abs() < FIRST_ORDER * split,
            "{name} l={l}"
        );
        compared += 1;
    }
    assert_eq!(compared, 3);
}

/// The lowest level of a record channel's generalized problem.
fn level(
    ctx: &mandacaru_core::Context,
    dataset: &mandacaru_core::paw::record::PawDataset,
    l: usize,
    projectors: &[mandacaru_core::Buffer<f64>],
    coupling: &Array,
    overlap: &Array,
) -> f64 {
    let tables = projectors.iter().map(|p| p.as_slice()).collect::<Vec<_>>();
    lowest_level(
        ctx,
        dataset.r.as_slice(),
        dataset.v_local_screened.as_slice(),
        l,
        &tables,
        coupling.data(),
        overlap.data(),
    )
    .unwrap()
}

/// The `(2j+1)`-weighted average of the branches against the scalar channel of the same
/// dataset: the reference energies, and the lowest generalized level of the average problem
/// (union projectors, `average_screened`, `overlap_average`) against the scalar channel's. The
/// scalar (Koelling-Harmon) channel is the j average only to the order the scalar equation drops
/// terms, so the gate is a fraction of the splitting between the two j: the measured worst is
/// 0.9% (iron 3p), the gate 2%.
#[test]
#[ignore = "slow: 28 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit_identities -- --ignored the_j_average_reduces_to_the_scalar_channel"]
fn the_j_average_reduces_to_the_scalar_channel() {
    let mut compared = 0;
    for name in LIGHT {
        let shared = loaded(name);
        let (ctx, generated) = (&shared.ctx, &shared.generated);
        let dataset = &generated.dataset;
        for (&l, so) in &dataset.spin_orbit {
            let channel = &dataset.channels[&l];
            let [w_down, w_up] = weights(l as usize);
            let count = so.reference_energies.shape()[1];
            let e = so.reference_energies.data();
            let split = e[count] - e[0];
            let mean = (w_down * e[0] + w_up * e[count]) / (w_down + w_up);
            let scalar = channel.reference_energies.data()[0];
            let average = level(
                ctx,
                dataset,
                l as usize,
                &so.projectors,
                &so.average_screened,
                &so.overlap_average,
            );
            let own = level(
                ctx,
                dataset,
                l as usize,
                &channel.projectors,
                &channel.coupling_screened,
                &channel.overlap_correction,
            );
            println!(
                "{name} l={l}: splitting {split:.3e}; reference energies, j average {mean:+.8} scalar {scalar:+.8} ({:.2}% of it); lowest level, average problem {average:+.8} scalar {own:+.8} ({:.2}%)",
                100. * (mean - scalar).abs() / split,
                100. * (average - own).abs() / split
            );
            assert!(
                (mean - scalar).abs() < REDUCTION * split,
                "{name} l={l}: reference energies"
            );
            assert!(
                (average - own).abs() < REDUCTION * split,
                "{name} l={l}: lowest level"
            );
            compared += 1;
        }
    }
    assert!(compared >= 4);
}

/// The spin-orbit block survives the record: written at stride 4 with ten significant digits and
/// read back, the matrices and level data are the generated ones to that precision and the
/// union projectors are the generated tables at every fourth node.
#[test]
#[ignore = "slow: 22 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_spin_orbit_identities -- --ignored a_dirac_dataset_survives_the_record_round_trip"]
fn a_dirac_dataset_survives_the_record_round_trip() {
    use mandacaru_core::paw::record::{PawDataset, WriteSpec};
    let shared = loaded("lda-fr_Na");
    let (ctx, generated) = (&shared.ctx, &shared.generated);
    let dir = std::env::temp_dir().join(format!("mandacaru_paw_dirac_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Na.parquet");
    let spec = WriteSpec {
        stride: 4,
        codec: mandacaru_core::parquet::Codec::Zstd,
    };
    generated.dataset.write(ctx, &path, &spec).unwrap();
    let back = PawDataset::read(ctx, &path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(back.relativity, "dirac");
    assert_eq!(
        back.spin_orbit.keys().collect::<Vec<_>>(),
        generated.dataset.spin_orbit.keys().collect::<Vec<_>>()
    );
    for (l, so) in &generated.dataset.spin_orbit {
        let other = &back.spin_orbit[l];
        for (name, a, b) in [
            ("average", &so.average, &other.average),
            (
                "coupling_screened",
                &so.coupling_screened,
                &other.coupling_screened,
            ),
            (
                "overlap_coupling",
                &so.overlap_coupling,
                &other.overlap_coupling,
            ),
            (
                "reference_energies",
                &so.reference_energies,
                &other.reference_energies,
            ),
            ("level_errors", &so.level_errors, &other.level_errors),
        ] {
            let scale = a.data().iter().fold(0., |m: f64, x| m.max(x.abs()));
            for (x, y) in a.data().iter().zip(b.data()) {
                assert!(
                    (x - y).abs() <= 1e-9 * scale,
                    "l={l} {name}: {x} against {y}"
                );
            }
        }
        assert_eq!(so.projectors.len(), other.projectors.len());
        for (p, q) in so.projectors.iter().zip(&other.projectors) {
            let scale = p.as_slice().iter().fold(0., |m: f64, x| m.max(x.abs()));
            for (k, y) in q.as_slice().iter().enumerate() {
                assert!(
                    (p.as_slice()[4 * k + 3] - y).abs() <= 1e-9 * scale,
                    "l={l} projector {k}"
                );
            }
        }
    }
}
