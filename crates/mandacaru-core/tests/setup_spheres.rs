//! Projector spheres and the projections of the Bloch sums (`periodic::setup::spheres`):
//! the quadrature rule, `C(k) = <chi_k | p>` against a plain sum over the points, and the frozen
//! `C` of `si_crystal.json`, `local_split.json` and `kpoint_matrices.json` (Si and Al, SZ,
//! `lda-sr`, raw projectors). Needs `MANDACARU_PAW_PATH`.
mod setup_shells_support;

use mandacaru_core::{
    basis::table,
    paw::local_split::long_range_potential,
    periodic::setup::{
        ions::LocalPotential,
        spheres::{add_projections, chunk_size, projections, weighted_gram, Sphere},
    },
    Error,
};
use nalgebra::Vector3;
use num_complex::Complex64;
use setup_shells_support::*;
use std::f64::consts::PI;

const T2: f64 = 1e-9;

fn cartesian(v: &serde_json::Value) -> Vec<[f64; 3]> {
    triples(&real_array(v))
}

#[test]
fn the_rule_integrates_the_ball_and_keeps_its_bins_sorted() {
    let ctx = context();
    let sphere = Sphere::projector(&ctx, [0.3, -1.2, 2.5], 2.4).unwrap();
    assert_eq!(sphere.len(), 64 * 24 * 48);
    let all = 0..sphere.len();
    let volume: f64 = sphere.weights(all.clone()).iter().sum();
    let want = 4. * PI * 2.4f64.powi(3) / 3.;
    assert!((volume / want - 1.).abs() < 1e-13, "{volume} {want}");
    // Every point is on the shell of its radius, inside the ball.
    let [x, y, z] = sphere.points(all.clone());
    for (j, &r) in sphere.radii(all.clone()).iter().enumerate() {
        let d = Vector3::new(x[j] - 0.3, y[j] + 1.2, z[j] - 2.5).norm();
        assert!((d - r).abs() < 1e-13 && r < 2.4);
    }
    // 1-Bohr bins in order of x, then y, then z.
    let bin = |j: usize| {
        (
            x[j].floor() as i64,
            y[j].floor() as i64,
            z[j].floor() as i64,
        )
    };
    assert!((1..sphere.len()).all(|j| bin(j - 1) <= bin(j)));
    // A radial potential averages like the unit weights: int r^2 dr x 4 pi = the ball.
    let moment: f64 = sphere
        .weights(all.clone())
        .iter()
        .zip(sphere.radii(all))
        .map(|(w, r)| w * r * r)
        .sum();
    assert!((moment / (4. * PI * 2.4f64.powi(5) / 5.) - 1.).abs() < 1e-13);
    let chunks: Vec<_> = sphere.chunks(16384).collect();
    assert_eq!(chunks.len(), 5);
    assert_eq!(chunks.last().unwrap().end, sphere.len());
    assert!(chunks.windows(2).all(|w| w[0].end == w[1].start));
    // The chunk size follows the sizes alone, between 512 and 2048 points.
    assert_eq!(chunk_size(1, 8), 2048);
    assert_eq!(chunk_size(50, 2000), 512);
    assert!(matches!(
        Sphere::projector(&ctx, [0.; 3], 0.),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_chunk_is_the_plain_sum_over_the_points() {
    let ctx = context();
    let cell = Cell::read("si_sz");
    let ds = dataset(&ctx, "lda-sr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let projectors = projector_list(&ctx, &ds, &cell.centers, true);
    let first = &projectors[0];
    let sphere = Sphere::rule(&ctx, first.center, first.r_cut, 10, 6, 10, 0.).unwrap();
    let range = 0..sphere.len();
    let points = sphere.points(range.clone());
    let weights = sphere.weights(range);
    let count = projectors.len();
    let n = weights.len();
    // Columns p(x) w(x): the table's radial value and the harmonic, written here.
    let mut columns = vec![Complex64::new(0., 0.); count * n];
    let column_of = |p: &mandacaru_core::periodic::setup::spheres::Projector,
                     row: &mut [Complex64]| {
        let radius: Vec<f64> = (0..n)
            .map(|j| {
                (Vector3::new(points[0][j], points[1][j], points[2][j]) - Vector3::from(p.center))
                    .norm()
            })
            .collect();
        let tab = &projectors_table(&ctx, &ds, p);
        let radial =
            table::evaluate(&ctx, &ds.r.as_slice().to_vec(), tab, p.l, &radius, false).unwrap();
        for j in 0..n {
            let d = (Vector3::new(points[0][j], points[1][j], points[2][j])
                - Vector3::from(p.center))
                / radius[j];
            row[j] = mandacaru_core::math::angular::complex(p.l, p.m, [d[0], d[1], d[2]])
                * radial[j]
                * weights[j];
        }
    };
    for (row, p) in columns.chunks_mut(n).zip(&projectors) {
        column_of(p, row);
    }
    let kpoints = [[0.31, -0.27, 0.19], [-0.31, 0.27, -0.19], [0.1, 0.2, 0.3]];
    let region = sphere.region();
    let functions = basis.shells.functions();
    let mut got = vec![Complex64::new(0., 0.); kpoints.len() * functions * count];
    add_projections(
        &ctx,
        &basis.shells,
        &cell.lattice,
        &kpoints,
        &region,
        points,
        &columns,
        &mut got,
    )
    .unwrap();
    // The Bloch sums at the points by the plain image sum, contracted point by point.
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|j| [points[0][j], points[1][j], points[2][j]])
        .collect();
    let chi = image_sum(&ctx, &basis, &cell.lattice, &kpoints, &pts);
    let mut want = vec![Complex64::new(0., 0.); got.len()];
    for k in 0..kpoints.len() {
        for mu in 0..functions {
            for p in 0..count {
                let mut sum = Complex64::new(0., 0.);
                for j in 0..n {
                    sum += chi[(k * functions + mu) * n + j].conj() * columns[p * n + j];
                }
                want[(k * functions + mu) * count + p] = sum;
            }
        }
    }
    let error = relative(&got, &want);
    eprintln!("projections against the plain sum over the points: {error:e}");
    assert!(error < 1e-12, "{error:e}");
    // The sign of k.R is pinned: the opposite sign is far away.
    let flipped: Vec<[f64; 3]> = kpoints.iter().map(|k| k.map(|x| -x)).collect();
    let mut wrong = vec![Complex64::new(0., 0.); got.len()];
    add_projections(
        &ctx,
        &basis.shells,
        &cell.lattice,
        &flipped,
        &region,
        points,
        &columns,
        &mut wrong,
    )
    .unwrap();
    assert!(
        relative(&wrong, &want) > 1e-2,
        "{}",
        relative(&wrong, &want)
    );
}

/// The radial table a projector was built from (its channel's raw table).
fn projectors_table(
    _ctx: &mandacaru_core::Context,
    ds: &mandacaru_core::paw::record::PawDataset,
    p: &mandacaru_core::periodic::setup::spheres::Projector,
) -> Vec<f64> {
    ds.channels[&(p.l as u32)].raw_projectors[p.index]
        .as_slice()
        .to_vec()
}

fn frozen_c(snapshot: &serde_json::Value, k: usize) -> Vec<Complex64> {
    pairs(&snapshot["kpoints"][k]["projections"])
}

#[test]
fn the_projections_equal_the_frozen_c_of_the_boundary_snapshot() {
    let ctx = context();
    let cell = Cell::read("si_sz");
    let ds = dataset(&ctx, "lda-sr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let raw = projector_list(&ctx, &ds, &cell.centers, true);
    // The projector table agrees with the frozen cutoffs and labels.
    let table = reference("basis_table.json");
    let frozen = table["si_sz"]["projectors"].as_array().unwrap();
    assert_eq!(frozen.len(), raw.len());
    for (f, p) in frozen.iter().zip(&raw) {
        assert_eq!(f["atom"].as_u64().unwrap() as usize, p.atom);
        assert_eq!(
            (
                f["l"].as_u64().unwrap() as usize,
                f["m"].as_i64().unwrap() as i32
            ),
            (p.l, p.m)
        );
        assert!((f["r_cut"].as_f64().unwrap() - p.r_cut).abs() < 1e-14);
    }
    let snap = snapshot("si_crystal.json");
    let entries = snap["kpoints"].as_array().unwrap();
    let ks: Vec<[f64; 3]> = entries
        .iter()
        .map(|e| {
            let k = reals(&e["k"]);
            [k[0], k[1], k[2]]
        })
        .collect();
    let c = projections(&ctx, &basis.shells, &cell.lattice, &ks, &raw).unwrap();
    let size = basis.shells.functions() * raw.len();
    for k in 0..ks.len() {
        let error = relative(&c.as_slice()[k * size..(k + 1) * size], &frozen_c(&snap, k));
        eprintln!("si_crystal.json C at k{k}: {error:e}");
        assert!(error < T2, "{error:e}");
    }
    // The dual projectors give another C: the frozen file pins the raw basis.
    let dual = projector_list(&ctx, &ds, &cell.centers, false);
    let other = projections(&ctx, &basis.shells, &cell.lattice, &ks[..1], &dual).unwrap();
    assert!(relative(other.as_slice(), &frozen_c(&snap, 0)) > 1e-3);
}

#[test]
fn the_projections_equal_the_frozen_c_at_off_mesh_k() {
    let ctx = context();
    let all = reference("kpoint_matrices.json");
    for (name, cell_name, symbol) in [("si", "si_sz", "Si"), ("al", "al_sz", "Al")] {
        let cell = Cell::read(cell_name);
        let ds = dataset(&ctx, "lda-sr", symbol);
        let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
        let raw = projector_list(&ctx, &ds, &cell.centers, true);
        let ks = cartesian(&all[name]["k_cartesian"]);
        let c = projections(&ctx, &basis.shells, &cell.lattice, &ks, &raw).unwrap();
        let want = complex_array(&all[name]["projections"]);
        let error = relative(c.as_slice(), &want);
        eprintln!("kpoint_matrices.json {name} C at three irrational k: {error:e}");
        assert!(error < T2, "{error:e}");
    }
    // The two Gamma-centered half-mesh points of local_split.json.
    let ctx = context();
    let cell = Cell::read("si_sz");
    let ds = dataset(&ctx, "lda-sr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let raw = projector_list(&ctx, &ds, &cell.centers, true);
    let split = reference("local_split.json");
    for point in split["points"].as_array().unwrap() {
        let k = reals(&point["k_cartesian"]);
        let c = projections(
            &ctx,
            &basis.shells,
            &cell.lattice,
            &[[k[0], k[1], k[2]]],
            &raw,
        )
        .unwrap();
        let error = relative(c.as_slice(), &complex_array(&point["C"]));
        eprintln!("local_split.json C: {error:e}");
        assert!(error < T2, "{error:e}");
    }
}

#[test]
fn bad_projections_raise_and_a_small_budget_is_released() {
    let ctx = context();
    let cell = Cell::read("si_sz");
    let ds = dataset(&ctx, "lda-sr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let mut raw = projector_list(&ctx, &ds, &cell.centers, true);
    let k = [[0.1, 0.2, 0.3]];
    assert!(projections(&ctx, &basis.shells, &cell.lattice, &[], &raw).is_err());
    raw[3].r_cut = f64::NAN;
    assert!(matches!(
        projections(&ctx, &basis.shells, &cell.lattice, &k, &raw),
        Err(Error::Invalid(_))
    ));
    raw[3].r_cut = raw[2].r_cut;
    let small = mandacaru_core::Context::new(Some(1), 1 << 20).unwrap();
    let result = projections(&small, &basis.shells, &cell.lattice, &k, &raw);
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert_eq!(small.memory_used(), 0);
}

#[test]
fn the_short_range_local_matrix_equals_the_frozen_v_sr() {
    let ctx = context();
    let cell = Cell::read("si_sz");
    let ds = dataset(&ctx, "lda-sr", "Si");
    let basis = minimal_basis(&ctx, &ds, &cell.centers, cell.filter_wavevector);
    let sigma = reference("basis_table.json")["si_sz"]["sigma"]
        .as_f64()
        .unwrap();
    let potential = LocalPotential::new(
        &ctx,
        ds.r.as_slice(),
        ds.v_local.as_slice(),
        ds.valence_charge,
        ds.r_cut_local,
    )
    .unwrap();
    let radius = potential.reach(sigma).unwrap();
    let m = basis.shells.functions();
    let split = reference("local_split.json");
    let points = split["points"].as_array().unwrap();
    let ks: Vec<[f64; 3]> = points
        .iter()
        .map(|p| {
            let k = reals(&p["k_cartesian"]);
            [k[0], k[1], k[2]]
        })
        .collect();
    let mut total = vec![Complex64::new(0., 0.); ks.len() * m * m];
    for center in &cell.centers {
        // 32 x 16 x 36 points on two panels split at the local cutoff.
        let sphere = Sphere::rule(&ctx, *center, radius, 32, 16, 36, ds.r_cut_local).unwrap();
        let all = 0..sphere.len();
        let long = long_range_potential(ds.valence_charge, sigma, sphere.radii(all.clone()));
        let weights: Vec<f64> = sphere
            .weights(all.clone())
            .iter()
            .zip(sphere.radii(all))
            .zip(&long)
            .map(|((w, r), l)| w * (potential.value(*r) - l))
            .collect();
        let gram = weighted_gram(
            &ctx,
            &basis.shells,
            None,
            &cell.lattice,
            &ks,
            &sphere,
            &weights,
        )
        .unwrap();
        // The same set given as the ket is the same matrix.
        let cross = weighted_gram(
            &ctx,
            &basis.shells,
            Some(&basis.shells),
            &cell.lattice,
            &ks,
            &sphere,
            &weights,
        )
        .unwrap();
        assert_eq!(gram.rows, cross.rows);
        assert!(relative(gram.values.as_slice(), cross.values.as_slice()) < 1e-14);
        let (nr, nc) = (gram.rows.len(), gram.columns.len());
        for k in 0..ks.len() {
            for (i, &row) in gram.rows.iter().enumerate() {
                for (j, &column) in gram.columns.iter().enumerate() {
                    total[(k * m + row) * m + column] +=
                        gram.values.as_slice()[(k * nr + i) * nc + j];
                }
            }
        }
    }
    for (k, point) in points.iter().enumerate() {
        let error = relative(
            &total[k * m * m..(k + 1) * m * m],
            &complex_array(&point["V_sr"]),
        );
        eprintln!("local_split.json V_sr at k{k} (radius {radius:.4}): {error:e}");
        assert!(error < T2, "{error:e}");
    }
    // A weight per point is required.
    let sphere = Sphere::projector(&ctx, [0.; 3], 1.).unwrap();
    assert!(weighted_gram(
        &ctx,
        &basis.shells,
        None,
        &cell.lattice,
        &ks,
        &sphere,
        &[1.]
    )
    .is_err());
}
