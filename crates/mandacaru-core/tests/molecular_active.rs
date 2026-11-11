//! Active-space selection (`molecular::active`): correlating pairs, and MP2 ranking of an
//! open-shell reference.
use mandacaru_core::{
    math::SpatialGrid,
    molecular::{active::correlating_partners, MolecularIntegrals, ScfOptions},
    Context, Geometry, LengthUnit,
};

fn chain(ctx: &Context, atoms: usize) -> MolecularIntegrals {
    let positions = (0..atoms).map(|i| [3., 3., 2.2 + 0.9 * i as f64]).collect();
    let geometry = Geometry::new(
        vec![1; atoms],
        positions,
        [[6., 0., 0.], [0., 6., 0.], [0., 0., 7.5]],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();
    let grid =
        SpatialGrid::from_geometry(&geometry, [0.4 / 0.529177210903; 3], false, false).unwrap();
    MolecularIntegrals::new(
        ctx,
        geometry,
        grid,
        r#"{"name":"HAO","virtual_orbitals":1}"#,
        "spectral",
        None,
        Default::default(),
    )
    .unwrap()
}

#[test]
fn the_partners_are_the_maximum_weight_assignment() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    {
        let integrals = chain(&ctx, 4);
        let state = integrals
            .mean_field("rhf", [2, 2], &ScfOptions::default())
            .unwrap();
        let model = state.model(&integrals).unwrap();
        let m = model.m;
        let occupied = [0usize, 1];
        let virtuals: Vec<usize> = (2..m).collect();
        let pairs = correlating_partners(
            &integrals,
            &state.model_orbitals,
            &[2, 2],
            &occupied,
            &virtuals,
        )
        .unwrap();
        // Independent weights from the model's own integrals: F_pp of the closed shell and the
        // exchange integral <ii|aa>.
        let h = |p: usize, q: usize| model.one.as_slice()[p * m + q].re;
        let g = |p: usize, q: usize, r: usize, s: usize| {
            model.two.as_slice()[((p * m + q) * m + r) * m + s].re
        };
        let fock = |p: usize| {
            h(p, p)
                + (0..2)
                    .map(|q| 2. * g(p, q, p, q) - g(p, q, q, p))
                    .sum::<f64>()
        };
        let weight = |i: usize, a: usize| g(i, i, a, a).abs() / (fock(a) - fock(i)).max(1e-6);
        let mut best = (f64::NEG_INFINITY, (0, 0));
        for &a in &virtuals {
            for &b in &virtuals {
                if a != b {
                    let total = weight(0, a) + weight(1, b);
                    if total > best.0 {
                        best = (total, (a, b));
                    }
                }
            }
        }
        eprintln!(
            "pairs {pairs:?}; brute force {:?} (total {:.6})",
            best.1, best.0
        );
        assert_eq!(pairs, vec![(0, best.1 .0), (1, best.1 .1)]);
        let assigned: f64 = pairs.iter().map(|&(i, a)| weight(i, a)).sum();
        assert!((assigned - best.0).abs() < 1e-12);

        // The selector: a count keeps every active occupied orbital's partner.
        let selected = state
            .active_model(&integrals, r#"{"orbitals": 4, "correlating_pairs": true}"#)
            .unwrap();
        for &(_, a) in &pairs {
            assert!(
                selected.orbital_indices.contains(&a),
                "{:?}",
                selected.orbital_indices
            );
        }
        assert_eq!(selected.orbital_indices.len(), 4);
        // An explicit list without a partner is refused; a count too small for both is refused.
        let unpaired = (2..m).find(|a| !pairs.iter().any(|p| p.1 == *a)).unwrap();
        let list = format!(r#"{{"orbitals": [0, 1, {unpaired}], "correlating_pairs": true}}"#);
        assert!(state.active_model(&integrals, &list).is_err());
        assert!(state
            .active_model(&integrals, r#"{"orbitals": 3, "correlating_pairs": true}"#)
            .is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn an_open_shell_reference_is_ranked_by_open_shell_mp2() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    {
        let integrals = chain(&ctx, 3);
        let state = integrals
            .mean_field("uhf", [2, 1], &ScfOptions::default())
            .unwrap();
        let model = state
            .active_model(&integrals, r#"{"method": "mp2", "orbitals": 4}"#)
            .unwrap();
        let rotation = model
            .orbital_rotation
            .as_ref()
            .expect("an MP2 ranking rotates");
        // Occupied first, then true virtuals by descending natural occupation.
        let virtual_occupations = &rotation.occupations[2..];
        assert!(virtual_occupations.windows(2).all(|w| w[0] >= w[1]));
        assert!(rotation.correlation_energy < 0.);
        assert_eq!(model.particles, [2, 1]);
        assert!(state
            .active_model(&integrals, r#"{"method": "dlpno-mp2", "orbitals": 4}"#)
            .is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
