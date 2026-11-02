//! `mandacaru_bloch_project` under a thread policy: the workers it starts are counted and bounded
//! by the policy a `Context` installs, and the sum of their partial arrays does not depend on the
//! order in which they finish.
//!
//! The policy is the one `Context::run` installs (`runtime::with_policy` with the context's thread
//! count and its worker statistics). The Python setup reaches the same kernel through ctypes with
//! no policy at all, which a Rust test cannot bound.
use mandacaru_kernels::{
    bloch::Complex64,
    mandacaru_bloch_project,
    runtime::{with_policy, WorkerStats},
};
use std::sync::Arc;

struct Problem {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    translations: Vec<f64>,
    breaks: Vec<f64>,
    coeffs: Vec<f64>,
    columns: Vec<Complex64>,
    ms: Vec<i32>,
    p: usize,
}

fn problem() -> Problem {
    // Deterministic points in a 6 Bohr box: 40 blocks of 256, so several workers have work.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let npts = 40 * 256;
    let coordinates = |next: &mut dyn FnMut() -> f64| (0..npts).map(|_| 6. * next()).collect();
    let (x, y, z) = (
        coordinates(&mut next),
        coordinates(&mut next),
        coordinates(&mut next),
    );
    let p = 5;
    let columns = (0..p * npts)
        .map(|_| Complex64::new(next() - 0.5, next() - 0.5))
        .collect();
    Problem {
        x,
        y,
        z,
        translations: vec![0., 0., 0., 6., 0., 0., -6., 0., 0., 0., 6., 0.],
        breaks: (0..=6).map(f64::from).collect(),
        coeffs: (0..6 * 4).map(|_| next() - 0.5).collect(),
        columns,
        ms: vec![-1, 0, 1],
        p,
    }
}

fn project(problem: &Problem) -> Vec<Complex64> {
    let images = problem.translations.len() / 3;
    let mut out = vec![Complex64::default(); images * problem.ms.len() * problem.p];
    let center = [3., 3., 3.];
    // SAFETY: every pointer addresses a live slice of the length the kernel is told (SAFETY.md,
    // "raw ABI"); `out` holds `n_images * nm * p` elements.
    unsafe {
        mandacaru_bloch_project(
            problem.x.as_ptr(),
            problem.y.as_ptr(),
            problem.z.as_ptr(),
            problem.x.len() as i64,
            center.as_ptr(),
            problem.translations.as_ptr(),
            images as i32,
            4.5,
            problem.breaks.as_ptr(),
            problem.coeffs.as_ptr(),
            problem.breaks.len() as i32,
            6.,
            1,
            -1,
            problem.ms.as_ptr(),
            problem.ms.len() as i32,
            problem.columns.as_ptr(),
            problem.p as i32,
            out.as_mut_ptr(),
        );
    }
    out
}

#[test]
fn the_workers_of_a_projection_are_bounded_by_the_context_and_add_in_worker_order() {
    let problem = problem();
    let mut one = None;
    for threads in [1usize, 2, 4] {
        let stats = Arc::new(WorkerStats::default());
        let run = || with_policy(threads, stats.clone(), || project(&problem));
        let first = run();
        assert!(first.iter().any(|v| v.re != 0. || v.im != 0.));
        assert!(stats.peak() <= threads, "{} > {threads}", stats.peak());
        if threads > 1 {
            assert!(stats.peak() > 1, "the test must start several workers");
        }
        // Forty repeats: before the partial arrays were added in worker order, three or more
        // workers disagreed in the last bits in about one run out of six.
        for repeat in 0..40 {
            assert!(
                run() == first,
                "{threads} threads, repeat {repeat}: not bitwise equal"
            );
        }
        assert!(stats.peak() <= threads);
        match &one {
            None => one = Some(first),
            // A different worker count changes the grouping of the sum, nothing else.
            Some(reference) => {
                let worst = reference
                    .iter()
                    .zip(&first)
                    .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
                    .fold(0., f64::max);
                assert!(
                    worst < 1e-12,
                    "{threads} threads differ from one by {worst}"
                );
            }
        }
    }
}
