//! A stack on disk is streamed tile by tile into the projection and the density, and gives
//! the very numbers the stack in memory gives (bitwise), without a whole-stack read.
use mandacaru_core::{
    math::{
        projection::{hermitian_density, hermitian_density_tiles, ComplexProjector},
        storage::{ComplexSamples, ComplexTiles, Precision, TILE},
    },
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;

/// Deterministic values in (-1, 1) from a linear congruence: no dependency, repeatable.
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
        })
        .collect()
}
fn stack(rows: usize, points: usize, seed: u64, real_first_tile: bool) -> Vec<C64> {
    let re = values(rows * points, seed);
    let im = values(rows * points, seed + 1);
    (0..rows * points)
        .map(|i| {
            let k = i % points;
            C64::new(re[i], if real_first_tile && k < TILE { 0. } else { im[i] })
        })
        .collect()
}
fn hermitian(m: usize, imaginary: bool) -> DMatrix<C64> {
    let a = values(m * m, 7);
    let b = values(m * m, 8);
    DMatrix::from_fn(m, m, |i, j| {
        let (p, q) = (a[i * m + j] + a[j * m + i], b[i * m + j] - b[j * m + i]);
        C64::new(p, if imaginary { q } else { 0. })
    })
}

#[test]
fn a_stack_on_disk_projects_and_accumulates_exactly_as_the_stack_in_memory() {
    let (m, n) = (6, 2 * TILE + 904);
    let field = values(n, 99);
    for precision in [Precision::Float64, Precision::Float32] {
        for real_first_tile in [false, true] {
            let ctx = Context::new(Some(1), 64 << 20).unwrap();
            let original = stack(m, n, 3, real_first_tile);
            let resident = ComplexSamples::copy(&ctx, &original, precision).unwrap();
            let archived = ComplexSamples::archived_copy(&ctx, &original, precision).unwrap();
            assert!(!resident.archived() && archived.archived());
            // The oracle is the plain-slice path over the values as stored (float32 rounds once).
            let seen = resident.read(&ctx).unwrap().to_vec();
            let oracle = ComplexProjector::new(&ctx, &seen, m, n)
                .unwrap()
                .project(&field)
                .unwrap()
                .to_vec();
            for source in [&resident, &archived] {
                let got = ComplexProjector::streaming(&ctx, source, m, n)
                    .unwrap()
                    .project(&field)
                    .unwrap()
                    .to_vec();
                assert_eq!(got, oracle, "{precision:?} streamed projection");
            }
            for imaginary in [false, true] {
                let d = hermitian(m, imaginary);
                let oracle = hermitian_density(&ctx, &seen, &d, n).unwrap();
                for source in [&resident, &archived] {
                    let got = hermitian_density_tiles(&ctx, source, &d, n).unwrap();
                    assert_eq!(got.as_slice(), oracle.as_slice(), "{precision:?} density");
                }
            }
            // Negative control: a stack with two orbitals exchanged is another stack; the
            // projection of it differs from the oracle's by far more than the exact parity.
            let mut swapped = original.clone();
            for k in 0..n {
                swapped.swap(k, n + k);
            }
            let swapped = ComplexSamples::archived_copy(&ctx, &swapped, precision).unwrap();
            let wrong = ComplexProjector::streaming(&ctx, &swapped, m, n)
                .unwrap()
                .project(&field)
                .unwrap()
                .to_vec();
            drop(swapped);
            let bite = wrong
                .iter()
                .zip(&oracle)
                .map(|(a, b)| (a - b).norm())
                .fold(0f64, f64::max);
            assert!(bite > 100. * 1e-12, "the control moved the matrix by {bite}");
            drop((resident, archived));
            assert_eq!(ctx.memory_used(), 0);
        }
    }
}
#[test]
fn tiles_cover_the_stack_and_refuse_what_is_outside_it() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let (m, n) = (3, 700);
    let original = stack(m, n, 11, false);
    let archived = ComplexSamples::archived_copy(&ctx, &original, Precision::Float64).unwrap();
    let mut scratch = vec![C64::default(); archived.scratch_len(m, 300)];
    assert_eq!(scratch.len(), m * 300);
    for (start, count) in [(0, 300), (300, 300), (600, 100), (699, 1)] {
        let tile = archived.tile(m, n, start, count, &mut scratch).unwrap();
        for row in 0..m {
            assert_eq!(tile.row(row), &original[row * n + start..row * n + start + count]);
        }
    }
    assert!(archived.tile(m, n, 650, 100, &mut scratch).is_err());
    assert!(archived.tile(m + 1, n, 0, 10, &mut scratch).is_err());
    assert!(ComplexProjector::streaming(&ctx, &archived, m, n + 1).is_err());
    let mut short = vec![C64::default(); m * 299];
    assert!(archived.tile(m, n, 0, 300, &mut short).is_err());
}
