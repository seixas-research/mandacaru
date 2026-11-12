//! The processor layout of a context (`kpts x bands x threads`): the effective width outside and
//! inside a parallel region, the lanes of a region on the persistent pool, re-entry into the
//! context from a lane without a deadlock, the share of the threads a lane's kernels inherit, the
//! counted workers, the order of results and errors, and the refusals.

use mandacaru_core::{math::fft::Fft3, quantum::C64, Context, Error, Layout, Level};

#[test]
fn a_context_has_one_level_until_its_processors_are_arranged() {
    let ctx = Context::new(Some(3), 1 << 20).unwrap();
    assert_eq!(
        ctx.layout(),
        Layout {
            kpts: 1,
            bands: 1,
            threads: 3
        }
    );
    assert_eq!(ctx.width(), 3);
    let ctx = Context::new(Some(2), 1 << 20)
        .unwrap()
        .with_layout(3, 2)
        .unwrap();
    assert_eq!(
        ctx.layout(),
        Layout {
            kpts: 3,
            bands: 2,
            threads: 2
        }
    );
    assert_eq!(ctx.layout().workers(), 12);
    assert_eq!(
        ctx.width(),
        12,
        "outside a region a kernel takes every processor"
    );
    for (kpts, bands) in [(0, 1), (1, 0)] {
        let refused = Context::new(Some(1), 1 << 20)
            .unwrap()
            .with_layout(kpts, bands);
        assert!(
            matches!(refused, Err(Error::Invalid(_))),
            "kpts={kpts} bands={bands}"
        );
    }
    let message = Context::new(Some(2), 1 << 20)
        .unwrap()
        .with_layout(32, 32)
        .err()
        .expect("2048 processors exceed the limit")
        .to_string();
    assert!(message.contains("1024"), "{message}");
}

#[test]
fn k_point_tasks_run_on_their_lanes_with_their_share_and_return_in_index_order() {
    let ctx = Context::new(Some(2), 64 << 20)
        .unwrap()
        .with_layout(3, 1)
        .unwrap();
    assert_eq!(ctx.lanes(Level::Kpts, 10), 3);
    assert_eq!(ctx.lanes(Level::Kpts, 2), 2);
    let mut lanes = vec![0usize; ctx.lanes(Level::Kpts, 10)];
    let before = ctx.workers_total();
    let out = ctx
        .map_tasks(Level::Kpts, &mut lanes, 10, |done, i| {
            *done += 1;
            Ok((i, ctx.width(), mandacaru_kernels::runtime::thread_count()))
        })
        .unwrap();
    for (i, &(index, width, threads)) in out.iter().enumerate() {
        assert_eq!((index, width, threads), (i, 2, 2), "task {i}");
    }
    assert_eq!(lanes.iter().sum::<usize>(), 10, "every task ran once");
    assert_eq!(
        ctx.workers_total() - before,
        3,
        "each lane is one counted worker"
    );
    assert!(ctx.workers_peak() <= 3);
    assert_eq!(ctx.width(), 6, "the width is restored after the region");
}

#[test]
fn a_lane_reenters_its_context_and_sizes_its_kernels_by_its_share() {
    // 32^3 points: the transform splits its lines over its workers in each of its three passes.
    let shape = [32, 32, 32];
    let input = |i: usize| -> Vec<C64> {
        (0..32768)
            .map(|j| C64::new(((j * 7 + i) % 11) as f64, ((j + 3 * i) % 5) as f64))
            .collect()
    };
    let serial = Context::new(Some(1), 512 << 20).unwrap();
    let expected: Vec<Vec<C64>> = (0..2)
        .map(|i| {
            let mut values = input(i);
            Fft3::new(&serial, shape)
                .unwrap()
                .transform(&mut values, false)
                .unwrap();
            values
        })
        .collect();
    let ctx = Context::new(Some(2), 512 << 20)
        .unwrap()
        .with_layout(2, 1)
        .unwrap();
    let before = ctx.workers_total();
    let mut units = vec![(); 2];
    // `Fft3::transform` runs `Context::run` on the pool thread of its lane: without the lane
    // marked as running for the context it would wait for the calling thread's region forever.
    let out = ctx
        .map_tasks(Level::Kpts, &mut units, 2, |_, i| {
            let mut values = input(i);
            Fft3::new(&ctx, shape)?.transform(&mut values, false)?;
            Ok(values)
        })
        .unwrap();
    // Two lanes, each one counted worker whose transform started its share of two workers in
    // each of the three passes: 2 + 2 x 3 x 2. The whole width would have started 2 + 2 x 3 x 4.
    assert_eq!(ctx.workers_total() - before, 2 + 2 * 3 * 2);
    for (got, want) in out.iter().zip(&expected) {
        assert!(
            got.iter().zip(want).all(|(a, b)| a == b),
            "the transform depends on its lane"
        );
    }
}

#[test]
fn the_first_failing_task_by_index_is_reported() {
    let ctx = Context::new(Some(1), 1 << 20)
        .unwrap()
        .with_layout(4, 1)
        .unwrap();
    let mut units = vec![(); 4];
    for _ in 0..20 {
        let error = ctx
            .map_tasks(Level::Kpts, &mut units, 40, |_, i| {
                if i == 5 || i == 9 {
                    Err(Error::Invalid(format!("task {i}")))
                } else {
                    Ok(i)
                }
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "task 5");
    }
}

#[test]
fn lanes_that_do_not_fit_in_the_budget_are_refused_with_the_largest_that_does() {
    let ctx = Context::new(Some(1), 10_000)
        .unwrap()
        .with_layout(4, 1)
        .unwrap();
    assert!(ctx.check_lanes(Level::Kpts, 3, 3000).is_ok());
    let error = ctx.check_lanes(Level::Kpts, 4, 3000).unwrap_err();
    assert!(matches!(error, Error::LayoutBudget(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("kpts=4") && message.contains("the largest kpts that fits is 3"),
        "{message}"
    );
    let _held = ctx.reserve(4000).unwrap();
    let message = ctx
        .check_lanes(Level::Kpts, 3, 3000)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("the largest kpts that fits is 2"),
        "{message}"
    );
}

#[test]
fn a_one_processor_context_runs_its_lanes_on_the_calling_thread() {
    let ctx = Context::new(Some(1), 1 << 20).unwrap();
    assert_eq!(ctx.lanes(Level::Kpts, 8), 1);
    assert_eq!(ctx.lanes(Level::Threads, 8), 1);
    let caller = std::thread::current().id();
    let mut units = vec![(); 1];
    let out = ctx
        .map_tasks(Level::Kpts, &mut units, 3, |_, i| {
            Ok((i, std::thread::current().id()))
        })
        .unwrap();
    assert!(out
        .iter()
        .enumerate()
        .all(|(i, &(j, id))| i == j && id == caller));
    assert_eq!(ctx.workers_total(), 0);
}
