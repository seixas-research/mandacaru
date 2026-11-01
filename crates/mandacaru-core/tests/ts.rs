use mandacaru_core::{
    models::ts::{
        fermi_damping, free_atom_params, functional_s_r, tkatchenko_scheffler,
    },
    Geometry, LengthUnit,
};

#[test]
fn ts_parameters_and_damping() {
    let h = free_atom_params(1).unwrap();
    assert_eq!(h.z, 1);
    assert!((h.alpha_0 - 4.50).abs() < 1e-10);
    assert!((h.c6 - 6.50).abs() < 1e-10);

    let c = free_atom_params(6).unwrap();
    assert_eq!(c.z, 6);
    assert!((c.alpha_0 - 12.00).abs() < 1e-10);
    assert!((c.c6 - 46.60).abs() < 1e-10);

    // Unsupported element
    assert!(free_atom_params(119).is_err());

    // Functionals
    assert!((functional_s_r("PBE").unwrap() - 0.940).abs() < 1e-10);
    assert!((functional_s_r("pbe").unwrap() - 0.940).abs() < 1e-10);
    assert!((functional_s_r("lda").unwrap() - 1.000).abs() < 1e-10);
    assert!((functional_s_r("r2scan").unwrap() - 1.000).abs() < 1e-10);
    assert!(functional_s_r("unknown_functional").is_err());

    // Fermi damping derivatives against numerical finite differences
    let r0_sum = 6.0;
    let s_r = 0.94;
    let d = 20.0;
    for &r in &[3.0, 5.0, 5.64, 6.0, 8.0, 12.0] {
        let (f0, f_prime) = fermi_damping(r, r0_sum, s_r, d);
        let h_step = 1e-5;
        let (fp, _) = fermi_damping(r + h_step, r0_sum, s_r, d);
        let (fm, _) = fermi_damping(r - h_step, r0_sum, s_r, d);
        let num_prime = (fp - fm) / (2.0 * h_step);
        assert!(
            (f_prime - num_prime).abs() < 1e-7,
            "failed at r={r}: f_prime={f_prime}, num_prime={num_prime}"
        );
        assert!(f0 >= 0.0 && f0 <= 1.0);
    }
}

#[test]
fn ts_h2_dimer_reproduces_ase_reference() {
    let geom = Geometry::new(
        vec![1, 1],
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        [[0.0; 3]; 3],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();

    let res = tkatchenko_scheffler(&geom, "pbe", None, None, None, None).unwrap();

    // Exact ASE reference:
    // E = -0.00012690742651563235 eV
    // F_z = [-0.0003197439, 0.0003197439] eV/A
    let expected_e_ev = -0.00012690742651563235;
    assert!(
        (res.energy_ev - expected_e_ev).abs() < 1e-10,
        "energy difference: {} vs {}",
        res.energy_ev,
        expected_e_ev
    );

    let expected_fz = 0.00031974390474839387;
    assert!((res.forces_ev_angstrom[0][2] - (-expected_fz)).abs() < 1e-8);
    assert!((res.forces_ev_angstrom[1][2] - expected_fz).abs() < 1e-8);

    // Forces sum to zero
    for a in 0..3 {
        assert!(
            (res.forces_ev_angstrom[0][a] + res.forces_ev_angstrom[1][a]).abs() < 1e-12
        );
    }
}

#[test]
fn ts_h2o_reproduces_ase_reference_and_forces_match_numerical_gradient() {
    // Standard ASE molecule("H2O") positions in Angstrom:
    // O: [0, 0, 0.119262]
    // H1: [0, 0.763239, -0.477047]
    // H2: [0, -0.763239, -0.477047]
    let pos_angstrom = vec![
        [0.0, 0.0, 0.119262],
        [0.0, 0.763239, -0.477047],
        [0.0, -0.763239, -0.477047],
    ];

    let geom = Geometry::new(
        vec![8, 1, 1],
        pos_angstrom.clone(),
        [[0.0; 3]; 3],
        [false; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();

    let res = tkatchenko_scheffler(&geom, "pbe", None, None, None, None).unwrap();

    // ASE reference: E = -0.007061642667966006 eV
    let expected_e_ev = -0.007061642667966006;
    assert!(
        (res.energy_ev - expected_e_ev).abs() < 1e-9,
        "H2O energy mismatch: {} vs {}",
        res.energy_ev,
        expected_e_ev
    );

    // Net force is zero
    for a in 0..3 {
        let sum_f: f64 = res.forces_ev_angstrom.iter().map(|f| f[a]).sum();
        assert!(sum_f.abs() < 1e-10, "axis {a} net force non-zero: {sum_f}");
    }

    // Verify analytic forces in atomic units against central finite difference of energy in Hartree:
    // F_i,a = - dE / d(R_i,a)
    let h_bohr = 1e-5;
    for atom in 0..3 {
        for axis in 0..3 {
            let mut pos_p = geom.positions().to_vec();
            let mut pos_m = geom.positions().to_vec();
            pos_p[atom][axis] += h_bohr;
            pos_m[atom][axis] -= h_bohr;

            let g_p = Geometry::new(
                geom.numbers().to_vec(),
                pos_p,
                *geom.cell(),
                geom.pbc(),
                LengthUnit::Bohr,
            )
            .unwrap();
            let g_m = Geometry::new(
                geom.numbers().to_vec(),
                pos_m,
                *geom.cell(),
                geom.pbc(),
                LengthUnit::Bohr,
            )
            .unwrap();

            let ep = tkatchenko_scheffler(&g_p, "pbe", None, None, None, None)
                .unwrap()
                .energy_ha;
            let em = tkatchenko_scheffler(&g_m, "pbe", None, None, None, None)
                .unwrap()
                .energy_ha;

            let num_grad = (ep - em) / (2.0 * h_bohr);
            let num_force = -num_grad;
            let ana_force = res.forces_ha_bohr[atom][axis];

            assert!(
                (ana_force - num_force).abs() < 1e-7,
                "atom {atom} axis {axis}: ana {ana_force} vs num {num_force}"
            );
        }
    }
}

#[test]
fn ts_periodic_diamond_reproduces_ase_and_net_forces_vanish() {
    // Diamond FCC unit cell (2 atoms, a=3.567 A)
    let a = 3.567;
    let cell_angstrom = [
        [0.0, a / 2.0, a / 2.0],
        [a / 2.0, 0.0, a / 2.0],
        [a / 2.0, a / 2.0, 0.0],
    ];
    let pos_angstrom = [
        [0.0, 0.0, 0.0],
        [a / 4.0, a / 4.0, a / 4.0],
    ];

    let geom = Geometry::new(
        vec![6, 6],
        pos_angstrom.to_vec(),
        cell_angstrom,
        [true; 3],
        LengthUnit::Angstrom,
    )
    .unwrap();

    let res = tkatchenko_scheffler(&geom, "pbe", None, None, None, None).unwrap();

    // Check that diamond energy is around -1.126 eV (matches ASE periodic cutoff)
    assert!(
        (res.energy_ev - (-1.12619)).abs() < 1e-4,
        "diamond energy unexpected: {}",
        res.energy_ev
    );

    // By inversion/tetrahedral symmetry, forces on both C atoms must be zero
    for atom in 0..2 {
        for axis in 0..3 {
            assert!(
                res.forces_ev_angstrom[atom][axis].abs() < 1e-8,
                "atom {atom} axis {axis} force should be zero, got {}",
                res.forces_ev_angstrom[atom][axis]
            );
        }
    }
}
