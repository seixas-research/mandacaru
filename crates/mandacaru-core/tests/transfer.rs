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
            &old_problem.basis,
            first.as_slice(),
        )
        .unwrap();
        let second = properties::spatial_rdms(
            &context,
            &new_problem.encoding,
            &new_problem.basis,
            second.as_slice(),
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
        let integrals =
            MolecularIntegrals::new(ctx, geometry, grid, r#"{"name":"HAO"}"#, "spectral", None)?;
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
