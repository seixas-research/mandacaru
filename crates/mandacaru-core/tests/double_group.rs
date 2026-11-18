//! Verification gates for Wave DG-1: crystallographic & magnetic double groups, projective
//! little-group irreps (symmorphic & nonsymmorphic), spinor SU(2) classifications, selection rules,
//! and live PAW spinor crystal band classification.
mod periodic_reduced_mesh_support;

use mandacaru_core::{
    math::linalg::eigen,
    models::spin_orbit,
    molecular::transform::Model,
    periodic::{
        irreps::{
            classify_bloch_bands, classify_crystal_kpoint, classify_kpath,
            crystal_dipole_selection, crystal_hamiltonian_and_overlap, dipole_selection_mask,
            irrep_blocked_diagonalize, paw_ao_actions, track_band_connectivity,
        },
        optics::momentum::{band_momenta, one_center_momentum},
        setup::pulay::Mesh as PulayMesh,
        space_group::{
            double_group::{CorepKind, DoubleGroup},
            find, Cell,
        },
        wannier::sampling::Frame,
    },
    quantum::{
        search::{
            learning::{
                data::{EditDataset, EditKind},
                training::{fit_edit_model, Model as SearchProposalModel},
            },
            vasqa::SearchOptions,
        },
        CrystalPointSymmetry, SolveOptions, Space, C64,
    },
    Context,
};
use nalgebra::{DMatrix, Matrix3, Vector3};

fn check_double_group_orthogonality(dg: &DoubleGroup) {
    let n = dg.unitary.len();
    let dim_sq: usize = dg.irreps.iter().map(|r| r.dimension * r.dimension).sum();
    assert_eq!(
        dim_sq, n,
        "Burnside sum d_alpha^2 = {dim_sq} != |G_bar| = {n}"
    );

    // Group associativity and inverse check
    for a in 0..n {
        assert_eq!(dg.product[a][dg.inverse[a]], 0);
        for b in 0..n {
            assert_eq!(
                dg.product[a][dg.bar_e], dg.product[dg.bar_e][a],
                "bar_E must be central"
            );
            let ab = dg.product[a][b];
            for c in 0..n {
                let lhs = dg.product[ab][c];
                let bc = dg.product[b][c];
                let rhs = dg.product[a][bc];
                assert_eq!(lhs, rhs, "group product must be associative");
            }
        }
    }

    // Character orthogonality: (1/n) sum_g chi_a(g)* chi_b(g) = delta_{ab}
    for (i, a) in dg.irreps.iter().enumerate() {
        let expect = if a.spinor {
            -(a.dimension as f64)
        } else {
            a.dimension as f64
        };
        assert!(
            (a.characters[dg.bar_e] - C64::new(expect, 0.)).norm() < 1e-6,
            "irrep {i} bar_E character mismatch: {:?}",
            a.characters[dg.bar_e]
        );
        for (j, b) in dg.irreps.iter().enumerate() {
            let dot: C64 = a
                .characters
                .iter()
                .zip(&b.characters)
                .map(|(x, y)| x.conj() * y)
                .sum::<C64>()
                / (n as f64);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - C64::new(want, 0.)).norm() < 1e-6,
                "orthogonality failed for irreps ({i}, {j}): dot={dot}"
            );
        }
    }
}

#[test]
fn point_double_group_oh_and_c2h_dimensions_and_orthogonality() {
    // Simple cubic primitive cell -> full O_h point group (|G| = 48, |G_bar| = 96).
    // O_h has 10 single-valued irreps (dims: 4x1D, 2x2D, 4x3D => sum d^2 = 4+8+36 = 48) and
    // 6 double-valued spinor irreps (dims: 2x2D E_{1/2}+/-, 2x2D E_{5/2}+/-, 2x4D G_{3/2}+/- => 8+8+32 = 48).
    let cubic = Matrix3::identity() * 5.0;
    let ops = find(
        &Cell {
            lattice: cubic,
            fractional: &[[0., 0., 0.]],
            labels: &[14],
        },
        1e-4,
    )
    .unwrap();
    assert_eq!(ops.point_group.symbol, "m-3m");
    assert_eq!(ops.point_group.order, 48);

    let dg_oh = DoubleGroup::from_operations(&ops, &cubic, None, None).unwrap();
    assert_eq!(dg_oh.unitary.len(), 96);
    assert_eq!(dg_oh.irreps.len(), 16);
    check_double_group_orthogonality(&dg_oh);

    let mut single_dims: Vec<usize> = dg_oh
        .irreps
        .iter()
        .filter(|r| !r.spinor)
        .map(|r| r.dimension)
        .collect();
    single_dims.sort_unstable();
    assert_eq!(single_dims, vec![1, 1, 1, 1, 2, 2, 3, 3, 3, 3]);

    let mut spinor_dims: Vec<usize> = dg_oh
        .irreps
        .iter()
        .filter(|r| r.spinor)
        .map(|r| r.dimension)
        .collect();
    spinor_dims.sort_unstable();
    assert_eq!(spinor_dims, vec![2, 2, 2, 2, 4, 4]);

    // Monoclinic C_{2h} (|G| = 4, |G_bar| = 8): 4 single 1D irreps + 4 double-valued 1D irreps (+/- i).
    let mono = Matrix3::new(4.0, 0.0, 0.5, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0);
    let mono_ops = find(
        &Cell {
            lattice: mono,
            fractional: &[[0., 0., 0.]],
            labels: &[1],
        },
        1e-4,
    )
    .unwrap();
    assert_eq!(mono_ops.point_group.order, 4);
    let dg_c2h = DoubleGroup::from_operations(&mono_ops, &mono, None, None).unwrap();
    assert_eq!(dg_c2h.unitary.len(), 8);
    assert_eq!(dg_c2h.irreps.len(), 8);
    check_double_group_orthogonality(&dg_c2h);
    assert_eq!(dg_c2h.irreps.iter().filter(|r| r.spinor).count(), 4);
}

#[test]
fn nonsymmorphic_diamond_silicon_double_little_group_at_zone_boundary_x() {
    // Primitive diamond Si (Fd-3m, 2 atoms at (0,0,0) and (1/4, 1/4, 1/4) in FCC basis).
    let a = 5.13;
    let fcc = Matrix3::from_columns(&[
        Vector3::new(0.0, a, a),
        Vector3::new(a, 0.0, a),
        Vector3::new(a, a, 0.0),
    ]);
    let ops = find(
        &Cell {
            lattice: fcc,
            fractional: &[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            labels: &[14, 14],
        },
        1e-4,
    )
    .unwrap();
    assert_eq!(ops.point_group.order, 48);

    // At Gamma: 16 standard O_h irreps (10 single + 6 spinor of dims 2, 2, 2, 2, 4, 4).
    let dg_gamma = DoubleGroup::from_operations(&ops, &fcc, None, None).unwrap();
    assert_eq!(dg_gamma.irreps.len(), 16);
    check_double_group_orthogonality(&dg_gamma);

    // At FCC zone-boundary X = (0, 1/2, 1/2): little co-group D_{4h} (|G_X| = 16, |G_bar_X| = 32).
    // Nonsymmorphic glide/screw translations force the 4 single-valued bands at X into four 2D irreps
    // (4 * 2^2 = 16) and ALL spinor bands into ONE 4-dimensional projective Dirac irrep X_5bar (4^2 = 16).
    let dg_x = DoubleGroup::from_operations(&ops, &fcc, None, Some([0.0, 0.5, 0.5])).unwrap();
    assert_eq!(dg_x.unitary.len(), 32);
    check_double_group_orthogonality(&dg_x);
    let spinor_x_dims: Vec<usize> = dg_x
        .irreps
        .iter()
        .filter(|r| r.spinor)
        .map(|r| r.dimension)
        .collect();
    assert_eq!(
        spinor_x_dims,
        vec![4],
        "Fd-3m X-point projective spinor irrep must be one 4D irrep, got {spinor_x_dims:?}"
    );
}

#[test]
fn spinor_classification_and_dipole_selection_rules_on_s_and_p_orbitals() {
    let cubic = Matrix3::identity() * 5.0;
    let ops = find(
        &Cell {
            lattice: cubic,
            fractional: &[[0., 0., 0.]],
            labels: &[14],
        },
        1e-4,
    )
    .unwrap();
    let dg = DoubleGroup::from_operations(&ops, &cubic, None, None).unwrap();

    // 1. s_{1/2} representation (even parity, d=2): U(g) = su2(R)
    let rep_s_half: Vec<DMatrix<C64>> = dg
        .unitary
        .iter()
        .map(|el| {
            DMatrix::from_row_slice(
                2,
                2,
                &[el.su2[0][0], el.su2[0][1], el.su2[1][0], el.su2[1][1]],
            )
        })
        .collect();
    let mult_s = dg.classify_representation(&rep_s_half).unwrap();
    assert_eq!(mult_s.len(), 1);
    assert!((mult_s[0].1 - 1.0).abs() < 1e-6);
    let irrep_s_half = mult_s[0].0;
    assert!(dg.irreps[irrep_s_half].spinor);
    assert_eq!(dg.irreps[irrep_s_half].dimension, 2);

    // 2. Odd-parity p \otimes spin-1/2 representation (d = 6 = p_{1/2} (2D) + p_{3/2} (4D))
    let mut rep_p_spinor = Vec::with_capacity(dg.unitary.len());
    let mut rep_t1u: Vec<DMatrix<C64>> = Vec::with_capacity(dg.unitary.len());
    for el in &dg.unitary {
        let d1 = el.cartesian.map(|v| C64::new(v, 0.));
        rep_t1u.push(DMatrix::from_column_slice(3, 3, d1.as_slice()));
        // Kronecker product su2 x d1 (6 x 6):
        let mut mat6 = DMatrix::<C64>::zeros(6, 6);
        for sa in 0..2 {
            for sb in 0..2 {
                for ma in 0..3 {
                    for mb in 0..3 {
                        mat6[(sa * 3 + ma, sb * 3 + mb)] = el.su2[sa][sb] * d1[(ma, mb)];
                    }
                }
            }
        }
        rep_p_spinor.push(mat6);
    }
    let mult_p = dg.classify_representation(&rep_p_spinor).unwrap();
    assert_eq!(
        mult_p.len(),
        2,
        "p x 1/2 must split into 2D + 4D irreps: {mult_p:?}"
    );
    let mut p_dims: Vec<usize> = mult_p
        .iter()
        .map(|&(idx, _)| dg.irreps[idx].dimension)
        .collect();
    p_dims.sort_unstable();
    assert_eq!(p_dims, vec![2, 4]);
    assert!(mult_p
        .iter()
        .all(|&(idx, w)| dg.irreps[idx].spinor && (w - 1.).abs() < 1e-6));

    // 3. Electric dipole T_{1u} selection rule: allows s_{1/2} -> p_{1/2}, p_{3/2} and forbids s_{1/2} -> s_{1/2}
    let mult_dip = dg.classify_representation(&rep_t1u).unwrap();
    assert_eq!(mult_dip.len(), 1);
    let irrep_t1u = mult_dip[0].0;
    assert!(!dg.irreps[irrep_t1u].spinor);
    assert_eq!(dg.irreps[irrep_t1u].dimension, 3);
    assert!(!dg
        .allows_transition(irrep_s_half, irrep_t1u, irrep_s_half)
        .unwrap());
    for &(p_irrep, _) in &mult_p {
        assert!(dg
            .allows_transition(irrep_s_half, irrep_t1u, p_irrep)
            .unwrap());
    }
}

#[test]
fn magnetic_shubnikov_corepresentations_detect_kramers_doubling() {
    // Non-centrosymmetric orthorhombic C_{2v}-like chiral/polar group? Use monoclinic cell with
    // mirror/C2 + anti-unitary time reversal T * g on one half (Shubnikov Type II vs Type III).
    let cubic = Matrix3::identity() * 5.0;
    let ops = find(
        &Cell {
            lattice: cubic,
            fractional: &[[0., 0., 0.]],
            labels: &[14],
        },
        1e-4,
    )
    .unwrap();
    // Keep only C_4 along z and mirror/inversion halving: simplest Type III check is all proper
    // rotations unitary (`det == +1`, group O) and improper reflections (`det == -1`) anti-unitary!
    let anti: Vec<bool> = ops
        .rotations
        .iter()
        .map(|w| w.map(f64::from).determinant() < 0.)
        .collect();
    let dg_mag = DoubleGroup::from_operations(&ops, &cubic, Some(&anti), None).unwrap();
    assert_eq!(dg_mag.unitary.len(), 48); // 24 proper rotations doubled
    assert_eq!(dg_mag.anti_unitary.len(), 48);
    // In Type III m'-3m' (unitary group O, anti-unitary = T * I * O): check Herring classification ran!
    assert!(dg_mag
        .irreps
        .iter()
        .all(|r| !matches!(r.corep, CorepKind::UnitaryOnly)));
    // Compare spinor irreps of O with pure single-valued irreps under anti-unitary coset:
    // At Gamma, (T * I)^2 = T^2 = -I on spinors, whereas sum_a (T * R)^2 flips signs depending on class!
    let kramers_count = dg_mag
        .irreps
        .iter()
        .filter(|r| matches!(r.corep, CorepKind::TypeB))
        .count();
    let a_count = dg_mag
        .irreps
        .iter()
        .filter(|r| matches!(r.corep, CorepKind::TypeA))
        .count();
    assert!(kramers_count + a_count > 0);
}

#[test]
fn dg2_dg3_spinor_bloch_irrep_classification_blocked_diagonalization_and_optics_rules() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let cubic = Matrix3::identity() * 5.0;
    let ops = find(
        &Cell {
            lattice: cubic,
            fractional: &[[0., 0., 0.]],
            labels: &[14],
        },
        1e-4,
    )
    .unwrap();
    let dg = DoubleGroup::from_operations(&ops, &cubic, None, None).unwrap();

    // 8-component spinor basis: spin-major [s_up, px_up, py_up, pz_up, s_dn, px_dn, py_dn, pz_dn]
    // with isotropic diagonal energies e_s = -1.2, e_p = 0.6, overlap s_p = 1.0, and isotropic L.S.
    let mut h = DMatrix::<C64>::zeros(8, 8);
    let mut s = DMatrix::<C64>::identity(8, 8);
    for spin in 0..2 {
        h[(spin * 4, spin * 4)] = C64::new(-1.2, 0.);
        s[(spin * 4, spin * 4)] = C64::new(1.15, 0.);
        for p in 1..4 {
            h[(spin * 4 + p, spin * 4 + p)] = C64::new(0.6, 0.);
            s[(spin * 4 + p, spin * 4 + p)] = C64::new(1.25, 0.);
        }
    }
    // Note: `models::spin_orbit::angular(l=1, false)` uses Y_{1,-1..1}; use Cartesian Pauli spin-orbit
    // `(L_a S_a) = (i/2) sum_{abc} eps_{abc} |p_b><p_c| \sigma_a` directly in Cartesian (x, y, z) gauge:
    let xi_so = 0.36_f64;
    let l_mats = [
        // L_x: (y, z) -> -i, +i
        [(2usize, 3usize, -1.0_f64), (3, 2, 1.0)],
        // L_y: (z, x) -> -i, +i
        [(3usize, 1usize, -1.0_f64), (1, 3, 1.0)],
        // L_z: (x, y) -> -i, +i
        [(1usize, 2usize, -1.0_f64), (2, 1, 1.0)],
    ];
    let pauli: [[[C64; 2]; 2]; 3] = [
        [
            [C64::default(), C64::new(0.5, 0.)],
            [C64::new(0.5, 0.), C64::default()],
        ],
        [
            [C64::default(), C64::new(0., -0.5)],
            [C64::new(0., 0.5), C64::default()],
        ],
        [
            [C64::new(0.5, 0.), C64::default()],
            [C64::default(), C64::new(-0.5, 0.)],
        ],
    ];
    for axis in 0..3 {
        for &(r, c, sign) in &l_mats[axis] {
            let l_rc = C64::new(0., sign);
            for sa in 0..2 {
                for sb in 0..2 {
                    h[(sa * 4 + r, sb * 4 + c)] += C64::new(xi_so, 0.) * l_rc * pauli[axis][sa][sb];
                }
            }
        }
    }
    let _ = spin_orbit::angular(&ctx, 1, true).unwrap();

    // Build exact unitary double-group AO matrices `D_ao[g]` on [s, px, py, pz] x {up, dn}
    let d_ao: Vec<DMatrix<C64>> = dg
        .unitary
        .iter()
        .map(|el| {
            let mut m = DMatrix::<C64>::zeros(8, 8);
            for sa in 0..2 {
                for sb in 0..2 {
                    let u_s = el.su2[sa][sb];
                    m[(sa * 4, sb * 4)] = u_s;
                    for pa in 0..3 {
                        for pb in 0..3 {
                            m[(sa * 4 + 1 + pa, sb * 4 + 1 + pb)] =
                                u_s * C64::new(el.cartesian[(pa, pb)], 0.);
                        }
                    }
                }
            }
            m
        })
        .collect();

    // 1. DG-3: Irrep-blocked generalized diagonalization matches full unblocked solve to < 1e-11.
    let blocked = irrep_blocked_diagonalize(&dg, &h, &s, &d_ao).unwrap();
    let inv_s_diag = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        8,
        (0..8).map(|i| C64::new(1. / s[(i, i)].re.sqrt(), 0.)),
    ));
    let h_orth = &inv_s_diag * &h * &inv_s_diag;
    let (full_e, full_v_orth) = eigen(h_orth).unwrap();
    let full_v = &inv_s_diag * full_v_orth;
    for (eb, ef) in blocked.eigenvalues.iter().zip(&full_e) {
        assert!((eb - ef).abs() < 1e-11, "blocked {eb} vs full {ef}");
    }

    // 2. DG-2: Classify Bloch spinor band clusters and check [s1/2 (2D), p1/2 (2D), p3/2 (4D)].
    let band_irreps = classify_bloch_bands(&dg, &full_e, &s, &full_v, &d_ao, 1e-8).unwrap();
    assert!(band_irreps.max_unitarity_error < 1e-11);
    let cluster_dims: Vec<usize> = band_irreps.clusters.iter().map(|c| c.hi - c.lo).collect();
    assert_eq!(cluster_dims, vec![2, 2, 4]);
    assert!(band_irreps
        .clusters
        .iter()
        .all(|c| dg.irreps[c.dominant_irrep].spinor));
    // Verify cluster dominant irreps agree identically with blocked eigensolver labels!
    for cluster in &band_irreps.clusters {
        for b in cluster.lo..cluster.hi {
            assert_eq!(blocked.band_irreps[b], cluster.dominant_irrep);
        }
    }

    // Connectivity tracking across small Hermitian perturbation
    let band_irreps_2 = classify_bloch_bands(
        &dg,
        &blocked.eigenvalues,
        &s,
        &blocked.eigenvectors,
        &d_ao,
        1e-8,
    )
    .unwrap();
    let links = track_band_connectivity(
        &band_irreps,
        &full_v,
        &s,
        &band_irreps_2,
        &blocked.eigenvectors,
    );
    assert_eq!(links.len(), 3);
    for (idx, &(a, b, w)) in links.iter().enumerate() {
        assert_eq!((a, b), (idx, idx));
        assert!(w > 0.9999);
    }

    // 3. DG-3: Electric dipole T_{1u} selection rules across classified spinor clusters:
    // s1/2 (cluster 0) -> p1/2 (cluster 1) and p3/2 (cluster 2) allowed; p1/2 -> p3/2 forbidden!
    let rep_t1u: Vec<DMatrix<C64>> = dg
        .unitary
        .iter()
        .map(|el| {
            let d1 = el.cartesian.map(|v| C64::new(v, 0.));
            DMatrix::from_column_slice(3, 3, d1.as_slice())
        })
        .collect();
    let irrep_t1u = dg.classify_representation(&rep_t1u).unwrap()[0].0;
    let s_irr = band_irreps.clusters[0].dominant_irrep;
    let p12_irr = band_irreps.clusters[1].dominant_irrep;
    let p32_irr = band_irreps.clusters[2].dominant_irrep;
    let mask =
        dipole_selection_mask(&dg, &[s_irr, p12_irr], &[p12_irr, p32_irr], &[irrep_t1u]).unwrap();
    assert_eq!(mask, vec![true, true, false, false]);
}

#[test]
fn dg4_relativistic_spinor_ghf_model_runs_vqe_adapt_vasqa_valsqa_with_double_group_symmetry() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    // 2 spatial sites => 4 spin-orbitals [0up, 1up, 0dn, 1dn] in spinor model ordering (`2*m = 4`),
    // N_e = 2 electrons (`particles = [1, 1]`). Include spin-flipping Rashba/SOC hoppings so S_z is
    // NOT conserved, while mirror/inversion double-group parity M_z (eigvals [+i, +i, -i, -i]
    // for spin-up vs spin-down pairs? No: spatial mirror `0 <-> +1, 1 <-> -1` paired with spin `+-i`)
    // commutes with H and leaves total spinor parity conserved!
    let m = 2usize;
    let n_spinors = 2 * m;
    // Modes 0 (0up), 2 (0dn) = occupied Kramers pair; modes 1 (1up), 3 (1dn) = virtual pair:
    // Double-group parity lambda = [+1, -1, -1, +1] (site parity +-1 * spin sign +-1).
    let mut one_mat = DMatrix::<C64>::zeros(n_spinors, n_spinors);
    let e_vals = [-1.4_f64, 0.7, -1.0, 1.3];
    for i in 0..n_spinors {
        one_mat[(i, i)] = C64::new(e_vals[i], 0.);
    }
    // Add Hermitian spin-flip/SOC coupling ONLY between modes of SAME parity (0<->3 and 2<->1):
    one_mat[(0, 3)] = C64::new(0.25, -0.18);
    one_mat[(3, 0)] = C64::new(0.25, 0.18);
    one_mat[(2, 1)] = C64::new(-0.22, 0.15);
    one_mat[(1, 2)] = C64::new(-0.22, -0.15);

    let mut one_buf = ctx.allocate::<C64>(n_spinors * n_spinors).unwrap();
    for p in 0..n_spinors {
        for q in 0..n_spinors {
            one_buf.as_mut_slice()[p * n_spinors + q] = one_mat[(p, q)];
        }
    }
    // Antisymmetric-compatible pair interaction conserving parity lambda = [+1, -1, -1, +1]:
    let mut two_buf = ctx
        .allocate::<C64>(n_spinors * n_spinors * n_spinors * n_spinors)
        .unwrap();
    let idx4 = |p: usize, q: usize, r: usize, s: usize| {
        ((p * n_spinors + q) * n_spinors + r) * n_spinors + s
    };
    let lambda = [1.0_f64, -1.0, -1.0, 1.0];
    for p in 0..n_spinors {
        for q in 0..n_spinors {
            for r in 0..n_spinors {
                for s in 0..n_spinors {
                    if (lambda[p] * lambda[q] * lambda[r] * lambda[s] - 1.0).abs() < 1e-8
                        && p == r
                        && q == s
                    {
                        two_buf.as_mut_slice()[idx4(p, q, r, s)] = C64::new(0.45, 0.);
                    }
                }
            }
        }
    }
    // Pair-hopping preserving total parity ((0,2) <-> (1,3))
    let u_pair = C64::new(0.32, 0.);
    two_buf.as_mut_slice()[idx4(1, 3, 0, 2)] = u_pair;
    two_buf.as_mut_slice()[idx4(0, 2, 1, 3)] = u_pair;
    two_buf.as_mut_slice()[idx4(3, 1, 2, 0)] = u_pair;
    two_buf.as_mut_slice()[idx4(2, 0, 3, 1)] = u_pair;

    let mut model = Model::new(&ctx, m, [1, 1], one_buf, two_buf, 0.2, true).unwrap();
    let dg_chars = vec![
        C64::new(1., 0.),
        C64::new(-1., 0.),
        C64::new(-1., 0.),
        C64::new(1., 0.),
    ];
    assert!(model.conserves_point_characters(&dg_chars));
    model.crystal_point_symmetry =
        Some(CrystalPointSymmetry::new(vec!["dg_mirror".into()], vec![dg_chars]).unwrap());

    let sym_prob = model
        .problem_in(&ctx, "jordan_wigner", Space::Sector, true, true)
        .unwrap();
    // Total number sector C(4, 2) = 6 states (wider than S_z=0 C(2,1)^2 = 4!)
    assert_eq!(sym_prob.register("spinor").unwrap().len(), 6);

    // Verify full unfiltered spinor problem has strictly larger pool than double-group-filtered one!
    let mut no_sym_model = Model::new(
        &ctx,
        m,
        [1, 1],
        {
            let mut b = ctx.allocate::<C64>(n_spinors * n_spinors).unwrap();
            b.as_mut_slice().copy_from_slice(model.one.as_slice());
            b
        },
        {
            let mut b = ctx.allocate::<C64>(model.two.as_slice().len()).unwrap();
            b.as_mut_slice().copy_from_slice(model.two.as_slice());
            b
        },
        0.2,
        true,
    )
    .unwrap();
    no_sym_model.crystal_point_symmetry = None;
    let unfiltered = no_sym_model
        .problem_in(&ctx, "jordan_wigner", Space::Sector, true, true)
        .unwrap();
    assert!(
        sym_prob.pool.len() < unfiltered.pool.len(),
        "double-group symmetry must prune spin-orbit pool: {} vs {}",
        sym_prob.pool.len(),
        unfiltered.pool.len()
    );

    // Exact diagonalization of the 6-state total-number spinor Hamiltonian
    let basis = sym_prob.register("fci").unwrap();
    let dense_h = sym_prob.hamiltonian.dense(&ctx, basis).unwrap();
    let fci_mat = DMatrix::from_fn(6, 6, |r, c| dense_h.as_slice()[r * 6 + c]);
    let (e_fci, _) = eigen(fci_mat).unwrap();
    let e_exact = e_fci[0];

    // VQE, ADAPT-VQE, VASQA, VALSQA all match exact SOC FCI energy!
    let res_vqe = sym_prob
        .solve(
            &SolveOptions {
                adapt: false,
                trotter: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!((res_vqe.energy - e_exact).abs() < 1e-6);

    let res_adapt = sym_prob
        .solve(
            &SolveOptions {
                adapt: true,
                trotter: true,
                max_growth: 8,
                gradient_tolerance: Some(1e-10),
                ..Default::default()
            },
            None,
        )
        .unwrap();
    assert!((res_adapt.energy - e_exact).abs() < 1e-6);

    let search_opts = SearchOptions {
        max_steps: 60,
        max_length: 6,
        move_weights: [1., 0.25, 0.25],
        temperatures: [0.005, 0.0001],
        seed: 2,
        ..Default::default()
    };
    let res_vasqa = sym_prob.search(&search_opts, &[], &[]).unwrap();
    assert!(
        (res_vasqa.best_energy - e_exact).abs() < 1e-3,
        "Spinor-VASQA {} vs exact {e_exact}",
        res_vasqa.best_energy
    );

    let rec = sym_prob
        .search(
            &SearchOptions {
                record: true,
                record_curvatures: true,
                screen_insertions: 4,
                max_steps: 8,
                ..search_opts.clone()
            },
            &[],
            &[],
        )
        .unwrap();
    let records: Vec<_> = rec
        .records
        .into_iter()
        .map(|mut step| {
            step["problem_id"] = serde_json::json!("p0");
            step["format"] = serde_json::json!("mandacaru-search-edit");
            step["version"] = serde_json::json!(1);
            step
        })
        .collect();
    let meta = serde_json::json!({
        "format": "mandacaru-search-problem",
        "version": 1,
        "num_qubits": sym_prob.hamiltonian.n,
        "spatial_orbitals": sym_prob.encoding.modes / 2,
        "num_particles": sym_prob.encoding.particles,
        "mapping": "jordan_wigner",
        "hamiltonian": sym_prob.hamiltonian.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>(),
        "pool": sym_prob.pool.iter().map(|g| g.operator.labels().into_iter().map(|(l, c)| serde_json::json!([l, c.re, c.im])).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "composition": [[1, 1]],
    });
    let payload = serde_json::json!({ "problems": { "p0": meta }, "records": records }).to_string();
    let edit_ds = EditDataset::from_json(
        &ctx,
        &payload,
        "molecule",
        &[EditKind::Insert, EditKind::Delete, EditKind::Replace],
    )
    .unwrap();
    let trained =
        SearchProposalModel::from_edit(fit_edit_model(&edit_ds, 1e-2).unwrap(), 0.2).unwrap();
    let res_valsqa = sym_prob
        .search_with_model(&search_opts, &[], &[], Some(&trained))
        .unwrap();
    assert!(
        (res_valsqa.best_energy - e_exact).abs() < 1e-3,
        "Spinor-VALSQA {} vs exact {e_exact}",
        res_valsqa.best_energy
    );
}

#[test]
#[ignore = "requires MANDACARU_PAW_PATH; solves live PAW spinor diamond Si"]
fn live_paw_spinor_silicon_double_group_classification_and_block_solver() {
    if std::env::var_os("MANDACARU_PAW_PATH").is_none() {
        eprintln!("skipping live PAW spinor test: MANDACARU_PAW_PATH not set");
        return;
    }
    let ctx = periodic_reduced_mesh_support::context();
    let mut cell = periodic_reduced_mesh_support::silicon("pbe-fr", "pbe", true, 12, 0.0);
    cell.mesh = ([2, 2, 2], true);
    let (setup, crystal) = periodic_reduced_mesh_support::build_with_setup(
        &ctx,
        &cell.lattice,
        &cell,
        periodic_reduced_mesh_support::Points::Reduced { symmetry: true },
    );
    assert!(crystal.spinor && crystal.time_reversal);
    let result =
        periodic_reduced_mesh_support::solve(&crystal, &periodic_reduced_mesh_support::tight());
    assert!(
        crystal.points.len() >= 2,
        "reduced 2x2x2 Si mesh has Gamma/L/X"
    );

    // Verify all reduced IBZ k-points (including Gamma and nonsymmorphic X boundary points).
    let mut saw_gamma = false;
    let mut saw_x = false;
    for (k_idx, point) in crystal.points.iter().enumerate() {
        let classified = classify_crystal_kpoint(&crystal, &result, 0, k_idx, 1e-4).unwrap();
        assert!(
            classified.overlap_residual < 1e-7,
            "PAW overlap invariance failed at k={:?}: {}",
            point.fractional,
            classified.overlap_residual
        );
        assert!(
            classified.irreps.max_unitarity_error < 1e-6,
            "PAW band representation non-unitary at k={:?}: {}",
            point.fractional,
            classified.irreps.max_unitarity_error
        );
        for cl in &classified.irreps.clusters {
            assert!(
                classified.double_group.irreps[cl.dominant_irrep].spinor,
                "every spinor PAW Si band cluster at k={:?} must belong to a spinor double irrep",
                point.fractional
            );
        }

        // Check irrep-blocked diagonalization reproduces live SCF eigenvalues at each k-point.
        let (h, s) = crystal_hamiltonian_and_overlap(&crystal, &result, point, 0).unwrap();
        let ao = paw_ao_actions(&classified.double_group, &crystal).unwrap();
        let solved = irrep_blocked_diagonalize(&classified.double_group, &h, &s, &ao).unwrap();
        let scf_e = &result.eigenvalues[k_idx];
        let solved_e: Vec<f64> = solved
            .eigenvalues
            .iter()
            .map(|&e| e + crystal.reference)
            .collect();
        let max_diff = scf_e
            .iter()
            .zip(&solved_e)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "irrep_blocked_diagonalize max diff {max_diff} at k={:?}",
            point.fractional
        );

        if point.fractional.iter().all(|&x| x.abs() < 1e-6) {
            saw_gamma = true;
            // Diamond Si valence top at Gamma splits under SOC into G_{3/2} (4D) + E_{1/2} (2D) or E_{5/2}.
            let dims: Vec<usize> = classified
                .irreps
                .clusters
                .iter()
                .map(|c| c.hi - c.lo)
                .collect();
            assert!(
                dims.iter().all(|&d| d >= 2 && d % 2 == 0),
                "Gamma spinor cluster dims must be Kramers-even >= 2: {dims:?}"
            );
            assert!(
                dims.contains(&4),
                "SOC Gamma valence states of Si must include a 4D G_{{3/2}} quartet: {dims:?}"
            );
        }
        // Check any X-equivalent point (two components at +/-0.5, one at 0).
        let n_half = point
            .fractional
            .iter()
            .filter(|&&x| (x.abs() - 0.5).abs() < 1e-6)
            .count();
        let n_zero = point.fractional.iter().filter(|&&x| x.abs() < 1e-6).count();
        if n_half == 2 && n_zero == 1 {
            saw_x = true;
            for cl in &classified.irreps.clusters {
                let mult = cl.hi - cl.lo;
                assert!(
                    mult % 4 == 0,
                    "nonsymmorphic X spinor diamond Si bands must stick in multiples of 4, got {mult}"
                );
            }
        }
    }
    assert!(
        saw_gamma && saw_x,
        "expected both Gamma and X on reduced 2x2x2 Si mesh"
    );

    // Verify automatic double-group block-diagonalized SCF matches unblocked SCF to < 1e-11 Ha.
    let unblocked = crystal
        .solve_unblocked(&periodic_reduced_mesh_support::tight())
        .unwrap();
    assert!(unblocked.converged);
    assert!(
        (result.energy - unblocked.energy).abs() < 1e-12,
        "blocked vs unblocked SCF energy diff: {}",
        (result.energy - unblocked.energy).abs()
    );
    assert!(
        (result.fermi - unblocked.fermi).abs() < 1e-11,
        "blocked vs unblocked Fermi level diff: {}",
        (result.fermi - unblocked.fermi).abs()
    );
    for (k_idx, (eb, eu)) in result
        .eigenvalues
        .iter()
        .zip(&unblocked.eigenvalues)
        .enumerate()
    {
        let diff = eb
            .iter()
            .zip(eu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            diff < 1e-11,
            "SCF eigenvalue mismatch at k={k_idx}: diff={diff}"
        );
    }

    // Verify double-group optical dipole selection screening at Gamma (parity-forbidden & allowed pairs)
    // directly against live PAW spinor velocity matrix elements (band_momenta).
    let gamma_idx = crystal
        .points
        .iter()
        .position(|p| p.fractional.iter().all(|&x| x.abs() < 1e-6))
        .unwrap();
    let sel = crystal_dipole_selection(&crystal, &result, 0, gamma_idx, 1e-4).unwrap();
    let n_allowed = sel.allowed.iter().filter(|&&b| b).count();
    let n_total = sel.allowed.len();
    assert!(
        n_allowed > 0 && n_allowed < n_total,
        "Gamma Si dipole mask should screen forbidden pairs: {n_allowed}/{n_total} allowed"
    );
    let bands_all: Vec<usize> = (0..crystal.bands()).collect();
    let frame =
        Frame::from_point(&crystal, &result, &crystal.points[gamma_idx], 0, &bands_all).unwrap();
    let grid_mesh = PulayMesh::new(&ctx, &setup.grid).unwrap();
    let one_center = one_center_momentum(&ctx, &setup).unwrap();
    let momenta = band_momenta(
        &ctx,
        &setup,
        &grid_mesh,
        &frame,
        crystal.points[gamma_idx].k,
        &one_center,
    )
    .unwrap();
    let mut max_forbidden_p = 0.0_f64;
    let mut max_allowed_p = 0.0_f64;
    for v in 0..sel.n_valence {
        for c in 0..sel.n_conduction {
            let idx = v * sel.n_conduction + c;
            for a in 0..3 {
                let p_amp = momenta.matrices[a][(v, sel.n_valence + c)].norm();
                if sel.allowed_axes[a][idx] {
                    max_allowed_p = max_allowed_p.max(p_amp);
                } else {
                    max_forbidden_p = max_forbidden_p.max(p_amp);
                }
            }
        }
    }
    assert!(
        max_forbidden_p < 1e-8,
        "forbidden PAW optical velocity at Gamma too large: {max_forbidden_p}"
    );
    assert!(
        max_allowed_p > 1e-2,
        "allowed PAW optical velocity at Gamma unexpectedly tiny: {max_allowed_p}"
    );

    // Verify symmetry-labeled SOC band path Gamma -> Delta -> X (`classify_kpath`):
    // 4D Gamma_8^+ splits into 2D Delta doublets (max parent-to-split overlap ~0.5) and re-merges into 4D X_5 (> 0.85).
    let path = [
        [0.0, 0.0, 0.0],
        [0.002, 0.0, 0.002],
        [0.25, 0.0, 0.25],
        [0.498, 0.0, 0.498],
        [0.5, 0.0, 0.5],
    ];
    let kpath = classify_kpath(&crystal, &setup, &result, 0, &path, 1e-4).unwrap();
    assert_eq!(kpath.len(), 5);
    let gamma_dims: Vec<usize> = kpath[0]
        .classification
        .irreps
        .clusters
        .iter()
        .map(|c| c.hi - c.lo)
        .collect();
    let delta_dims: Vec<usize> = kpath[2]
        .classification
        .irreps
        .clusters
        .iter()
        .map(|c| c.hi - c.lo)
        .collect();
    let x_dims: Vec<usize> = kpath[4]
        .classification
        .irreps
        .clusters
        .iter()
        .map(|c| c.hi - c.lo)
        .collect();
    assert!(gamma_dims.contains(&4), "Gamma must contain 4D G_{{3/2}}");
    assert!(
        delta_dims.iter().all(|&d| d == 2),
        "Delta line little group splits 4D quartet into 2D Kramers doublets: {delta_dims:?}"
    );
    assert!(
        x_dims.iter().all(|&d| d % 4 == 0),
        "nonsymmorphic X must re-merge 2D Delta branches into 4-fold multiplets: {x_dims:?}"
    );
    for (prev_idx, _cur_idx, weight) in &kpath[1].links_from_prev {
        let prev_cl = &kpath[0].classification.irreps.clusters[*prev_idx];
        let expect_min = if prev_cl.hi - prev_cl.lo == 4 {
            0.45
        } else {
            0.85
        };
        assert!(
            *weight > expect_min,
            "Gamma->Delta(0.002) link ({prev_idx}) weight {weight} below {expect_min}"
        );
    }
    assert!(
        kpath[4].links_from_prev.iter().all(|l| l.2 > 0.85),
        "Delta(0.498)->X overlap connectivity should stay > 0.85: {:?}",
        kpath[4].links_from_prev
    );
}

#[test]
fn dipole_axis_weights_capture_inversion_and_uniaxial_c4v_selection_rules() {
    // Tetragonal polar crystal along z (C4v, point group order 8, double group order 16):
    // z transforms as 1D A1 while (x, y) transform as 2D E.
    let lattice = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 6.0));
    let ops = find(
        &Cell {
            lattice,
            fractional: &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.35]],
            labels: &[14, 8],
        },
        1e-4,
    )
    .unwrap();
    assert_eq!(ops.n_distinct_rotations, 8);
    let dg = DoubleGroup::from_operations(&ops, &lattice, None, None).unwrap();
    assert_eq!(dg.unitary.len(), 16);

    let spinor_irreps: Vec<usize> = dg
        .irreps
        .iter()
        .filter(|r| r.spinor)
        .map(|r| r.index)
        .collect();
    assert_eq!(
        spinor_irreps.len(),
        2,
        "C4v has two 2D spinor irreps (E_{{1/2}} and E_{{3/2}})"
    );
    let (e1, e2) = (spinor_irreps[0], spinor_irreps[1]);
    let w_same = dg.dipole_axis_weights(e1, e1).unwrap();
    let w_cross = dg.dipole_axis_weights(e1, e2).unwrap();
    // In C4v: E_{1/2}* x E_{1/2} = A1 (z) + A2 + E (xy), whereas E_{1/2}* x E_{3/2} = B1 + B2 + E (xy only).
    assert!(
        w_same[2] > 0.5 && w_same[0] > 0.25 && w_same[1] > 0.25,
        "C4v intra-doublet E1/2 -> E1/2 allows all axes: {w_same:?}"
    );
    assert!(
        w_cross[2].abs() < 1e-8 && w_cross[0] > 0.25 && w_cross[1] > 0.25,
        "C4v E1/2 -> E3/2 forbids z polarization while allowing x/y: {w_cross:?}"
    );
}
