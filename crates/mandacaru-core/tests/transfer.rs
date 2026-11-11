use mandacaru_core::{
    math::SpatialGrid,
    molecular::MolecularIntegrals,
    quantum::{
        properties,
        transfer::{self, Frame},
        SolveOptions, C64,
    },
    Context, Geometry, LengthUnit,
};
use nalgebra::DMatrix;

#[test]
fn molecular_transfer_tracks_orbital_signs_and_preserves_the_physical_density() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[4., 4., 3.3], [4., 4., 4.7]],
            [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.6; 3], false, false).unwrap();
        let integrals = MolecularIntegrals::new(
            &context,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )
        .unwrap();
        let old = integrals
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        let mut new = integrals
            .mean_field("rhf", [1, 1], &Default::default())
            .unwrap();
        new.model_orbitals.column_mut(1).scale_mut(-1.);
        let old_model = old.model(&integrals).unwrap();
        let new_model = new.model(&integrals).unwrap();
        let old_problem = old_model
            .problem(&context, "jordan_wigner", true, false, true)
            .unwrap();
        let new_problem = new_model
            .problem(&context, "jordan_wigner", true, false, true)
            .unwrap();
        let index = old_problem
            .pool
            .iter()
            .position(|g| g.label == "S(0->1)")
            .unwrap();
        let parameters = [0.17];
        let indices = [index];
        let captured = transfer::capture(
            Frame {
                integrals: &integrals,
                reference: &old,
                model: &old_model,
                problem: &old_problem,
            },
            &indices,
            &parameters,
        )
        .unwrap();
        let seed = transfer::carry(
            &captured,
            Frame {
                integrals: &integrals,
                reference: &new,
                model: &new_model,
                problem: &new_problem,
            },
            0.9,
        )
        .unwrap();
        assert!(seed.accepted);
        assert!((seed.confidence - 1.).abs() < 1e-12);
        assert!((seed.parameters[0] + parameters[0]).abs() < 1e-14);
        let first = old_problem
            .prepare_state(&indices, &parameters, true)
            .unwrap();
        let second = new_problem
            .prepare_state(&seed.indices, &seed.parameters, true)
            .unwrap();
        let first = properties::spatial_rdms(
            &context,
            &old_problem.encoding,
            old_problem.register("test").unwrap(),
            first.complex128().unwrap(),
        )
        .unwrap();
        let second = properties::spatial_rdms(
            &context,
            &new_problem.encoding,
            new_problem.register("test").unwrap(),
            second.complex128().unwrap(),
        )
        .unwrap();
        let density = |reference: &DMatrix<C64>, one: &[C64]| {
            reference * DMatrix::from_row_slice(2, 2, one).transpose() * reference.adjoint()
        };
        assert!(
            (density(&old.model_orbitals, first.one.as_slice())
                - density(&new.model_orbitals, second.one.as_slice()))
            .norm()
                < 1e-12
        );
        let options = SolveOptions {
            adapt: true,
            ..Default::default()
        };
        let solved = new_problem
            .solve_sequence(&options, Some(&seed.parameters), Some(&seed.indices))
            .unwrap();
        assert!(solved.seed.as_ref().unwrap().success);
        assert!(solved.success);
        let independent = new_problem.solve(&options, None).unwrap();
        assert!((solved.energy - independent.energy).abs() < 1e-10);
    }
    assert_eq!(context.memory_used(), 0);
}


#[test]
fn a_coupled_exchange_ansatz_follows_orbital_sign_flips() {
    // The default ADAPT pool is `ceo`: its handover must survive the arbitrary signs of the
    // displaced geometry's orbitals (2026-11-10: every H2O relaxation step refused it).
    let context = Context::new(Some(1), 256 << 20).unwrap();
    {
        let geometry = Geometry::new(
            vec![1, 1, 1, 1],
            vec![[6., 6., 3.6], [6., 6., 5.0], [6., 6., 6.4], [6., 6., 7.8]],
            [[12., 0., 0.], [0., 12., 0.], [0., 0., 12.]],
            [false; 3],
            LengthUnit::Bohr,
        )
        .unwrap();
        let grid = SpatialGrid::from_geometry(&geometry, [0.6; 3], false, false).unwrap();
        let integrals = MolecularIntegrals::new(
            &context, geometry, grid, r#"{"name":"HAO"}"#, "spectral", None, Default::default(),
        )
        .unwrap();
        let old = integrals.mean_field("rhf", [2, 2], &Default::default()).unwrap();
        let mut new = integrals.mean_field("rhf", [2, 2], &Default::default()).unwrap();
        new.model_orbitals.column_mut(1).scale_mut(-1.);
        new.model_orbitals.column_mut(2).scale_mut(-1.);
        let old_model = old.model(&integrals).unwrap();
        let new_model = new.model(&integrals).unwrap();
        let mut old_problem = old_model.problem(&context, "jordan_wigner", true, false, true).unwrap();
        let mut new_problem = new_model.problem(&context, "jordan_wigner", true, false, true).unwrap();
        old_problem.use_pool("ceo", true).unwrap();
        new_problem.use_pool("ceo", true).unwrap();
        let labels: Vec<&str> = old_problem.pool.iter().map(|g| g.label.as_str()).collect();
        let pick = |pattern: &str, nth: usize| {
            labels.iter().enumerate().filter(|(_, l)| l.starts_with("CEO[") && l.contains(pattern)).nth(nth).unwrap().0
        };
        let indices = [pick("{S(", 1), pick(")+D(", 0), pick(")-D(", 3), pick("{S(", 4), pick(")+D(", 7)];
        let parameters = [0.21, -0.13, 0.08, 0.05, -0.11];
        let frame = |reference, model, problem| Frame { integrals: &integrals, reference, model, problem };
        // A previous Hessian approximation travels with the ansatz, conjugated by the sign each
        // angle took: the new problem is the old one in flipped orbitals, so its gradient is the
        // old one times those signs and its Hessian `S H S`.
        let hessian: Vec<f64> = (0..25).map(|k| 1. + (k / 5 + k % 5) as f64 * 0.1 + if k % 6 == 0 { 2. } else { 0. }).collect();
        let captured = transfer::capture(frame(&old, &old_model, &old_problem), &indices, &parameters)
            .unwrap()
            .with_hessian(hessian.clone())
            .unwrap();
        let seed = transfer::carry(&captured, frame(&new, &new_model, &new_problem), 0.9).unwrap();
        assert!(seed.accepted, "{}", seed.reason);
        assert_eq!(seed.indices, indices);
        assert!(seed.parameters.iter().zip(&parameters).any(|(a, b)| (a + b).abs() < 1e-14));
        let signs: Vec<f64> = seed.parameters.iter().zip(&parameters).map(|(a, b)| (a / b).round()).collect();
        assert!(signs.iter().all(|s| s.abs() == 1.) && signs.iter().any(|&s| s < 0.));
        let carried = seed.hessian.as_ref().expect("the previous Hessian is carried");
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(carried[i * 5 + j], hessian[i * 5 + j] * signs[i] * signs[j]);
            }
        }
        let (g_old, g_new) = (
            old_problem.evaluate(&indices, &parameters, true).unwrap().gradient,
            new_problem.evaluate(&seed.indices, &seed.parameters, true).unwrap().gradient,
        );
        for i in 0..5 {
            assert!((g_new[i] - signs[i] * g_old[i]).abs() < 1e-10, "{i}: {} vs {}", g_new[i], signs[i] * g_old[i]);
        }
        assert!(transfer::capture(frame(&old, &old_model, &old_problem), &indices, &parameters).unwrap().with_hessian(vec![0.; 24]).is_err());
        let first = old_problem.evaluate(&indices, &parameters, false).unwrap();
        let second = new_problem.evaluate(&seed.indices, &seed.parameters, false).unwrap();
        assert!((first.energy - second.energy).abs() < 1e-11, "{} vs {}", first.energy, second.energy);
        let rdm = |problem: &mandacaru_core::quantum::CachedProblem, state: &mandacaru_core::quantum::State| {
            properties::spatial_rdms(&context, &problem.encoding, problem.register("test").unwrap(), state.complex128().unwrap()).unwrap()
        };
        let (a, b) = (rdm(&old_problem, &first.state), rdm(&new_problem, &second.state));
        let density = |reference: &DMatrix<C64>, one: &[C64]| {
            reference * DMatrix::from_row_slice(4, 4, one).transpose() * reference.adjoint()
        };
        assert!((density(&old.model_orbitals, a.one.as_slice()) - density(&new.model_orbitals, b.one.as_slice())).norm() < 1e-11);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn captured_handover_releases_the_previous_frame_before_the_next_budgeted_build() {
    use mandacaru_core::{
        molecular::{transform::Model, MeanField},
        quantum::CachedProblem,
        Error, Result,
    };
    use std::sync::Arc;
    struct Calculation {
        integrals: MolecularIntegrals,
        reference: MeanField,
        model: Arc<Model>,
        problem: CachedProblem,
    }
    fn build(ctx: &Context) -> Result<Calculation> {
        let geometry = Geometry::new(
            vec![1, 1],
            vec![[4., 4., 3.3], [4., 4., 4.7]],
            [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
            [false; 3],
            LengthUnit::Bohr,
        )?;
        let grid = SpatialGrid::from_geometry(&geometry, [0.6; 3], false, false)?;
        let integrals = MolecularIntegrals::new(
            ctx,
            geometry,
            grid,
            r#"{"name":"HAO"}"#,
            "spectral",
            None,
            Default::default(),
        )?;
        let reference = integrals.mean_field("rhf", [1, 1], &Default::default())?;
        let model = reference.model(&integrals)?;
        let problem = model.problem(ctx, "jordan_wigner", true, false, true)?;
        Ok(Calculation {
            integrals,
            reference,
            model,
            problem,
        })
    }
    fn frame(calculation: &Calculation) -> Frame<'_> {
        Frame {
            integrals: &calculation.integrals,
            reference: &calculation.reference,
            model: &calculation.model,
            problem: &calculation.problem,
        }
    }
    let measure = Context::new(Some(1), 64 << 20).unwrap();
    let previous = build(&measure).unwrap();
    let retained = measure.memory_used();
    let handover = transfer::capture(frame(&previous), &[0], &[0.1]).unwrap();
    let peak = measure.memory_peak();
    drop(previous);
    let small = measure.memory_used();
    assert!(
        small * 4 < retained,
        "handover {small}, complete frame {retained}"
    );
    drop(handover);
    assert_eq!(measure.memory_used(), 0);
    let ctx = Context::new(Some(1), peak + small + 1024).unwrap();
    {
        let previous = build(&ctx).unwrap();
        let handover = transfer::capture(frame(&previous), &[0], &[0.1]).unwrap();
        let before = ctx.memory_used();
        assert!(matches!(build(&ctx), Err(Error::Budget { .. })));
        assert_eq!(ctx.memory_used(), before);
        drop(previous);
        let current = build(&ctx).unwrap();
        let seed = transfer::carry(&handover, frame(&current), 0.9).unwrap();
        assert!(seed.accepted);
        assert!((seed.confidence - 1.).abs() < 1e-12);
        assert_eq!(seed.indices, [0]);
        assert_eq!(seed.parameters, [0.1]);
    }
    assert_eq!(ctx.memory_used(), 0);
}
