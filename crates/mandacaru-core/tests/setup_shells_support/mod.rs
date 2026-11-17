//! Shared by the `setup_shells*` tests: the frozen readers, the Si and Al SZ bases rebuilt from
//! the shipped datasets (the filtered pseudo wave of each valence channel, as the Python basis
//! did), and the plain image sum the kernels are checked against.
#![allow(dead_code)]

use mandacaru_core::{
    basis::{confinement::Potential, filter, table},
    paw::{
        confinement::{orbital, ChannelData, Prepared},
        record::PawDataset,
    },
    periodic::setup::{
        images::Region,
        shells::{DenseRows, Radial, Sampling, Shell, Shells},
        spheres::Projector,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration")
}
/// `body` of a `reference/periodic/<name>` file.
pub fn reference(name: &str) -> Value {
    let text = std::fs::read_to_string(root().join("reference/periodic").join(name)).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["body"].clone()
}
/// A `phase5/<name>` boundary snapshot.
pub fn snapshot(name: &str) -> Value {
    let text = std::fs::read_to_string(root().join("phase5").join(name)).unwrap();
    serde_json::from_str(&text).unwrap()
}
pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
/// `{shape, data}` of the freeze scripts.
pub fn real_array(v: &Value) -> Vec<f64> {
    reals(&v["data"])
}
/// `{shape, re, im}` of the freeze scripts.
pub fn complex_array(v: &Value) -> Vec<Complex64> {
    reals(&v["re"])
        .into_iter()
        .zip(reals(&v["im"]))
        .map(|(re, im)| Complex64::new(re, im))
        .collect()
}
/// A list of `[re, im]` pairs (the snapshot format).
pub fn pairs(v: &Value) -> Vec<Complex64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| Complex64::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
        .collect()
}
pub fn triples(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}
/// Largest difference over the largest magnitude of the reference.
pub fn relative(got: &[Complex64], want: &[Complex64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().map(|w| w.norm()).fold(0., f64::max);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).norm())
        .fold(0., f64::max);
    worst / scale.max(1e-300)
}

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

pub fn dataset(ctx: &Context, library: &str, symbol: &str) -> PawDataset {
    let root = std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .expect("MANDACARU_PAW_PATH is unset: the setup tests read the shipped datasets");
    let path = PathBuf::from(root).join(format!("{library}/{symbol}.parquet"));
    PawDataset::read(ctx, &path).unwrap_or_else(|e| panic!("{symbol}: {e}"))
}

/// One cell of `reference/periodic/basis_table.json`.
pub struct Cell {
    pub lattice: Matrix3<f64>,
    pub centers: Vec<[f64; 3]>,
    pub filter_wavevector: f64,
    pub orbitals: Vec<(usize, usize, i32, f64)>,
    pub region: Region,
}
impl Cell {
    pub fn read(name: &str) -> Self {
        let table = reference("basis_table.json");
        let cell = &table[name];
        // The frozen cells are symmetric, so rows and columns of the lattice agree.
        let lattice = Matrix3::from_row_slice(&real_array(&cell["lattice_bohr"]));
        let centers = triples(&real_array(&cell["centers_bohr"]));
        let orbitals = cell["orbitals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                (
                    o["atom"].as_u64().unwrap() as usize,
                    o["l"].as_u64().unwrap() as usize,
                    o["m"].as_i64().unwrap() as i32,
                    o["support"].as_f64().unwrap(),
                )
            })
            .collect();
        Self {
            region: bounding_ball(&lattice, Vector3::zeros()),
            lattice,
            centers,
            filter_wavevector: cell["filter_wavevector"].as_f64().unwrap(),
            orbitals,
        }
    }
}

/// The ball about the cell's center that holds the cell (`PeriodicPAW._cell_region`).
pub fn bounding_ball(lattice: &Matrix3<f64>, origin: Vector3<f64>) -> Region {
    let center = lattice * Vector3::new(0.5, 0.5, 0.5) + origin;
    let mut radius = 0f64;
    for s1 in [-1., 1.] {
        for s2 in [-1., 1.] {
            for s3 in [-1., 1.] {
                radius = radius.max((lattice * Vector3::new(s1, s2, s3) * 0.5).norm());
            }
        }
    }
    Region {
        center: center.into(),
        radius,
    }
}

/// A shell with the table it was built from, for the independent sum.
pub struct Table {
    pub center: [f64; 3],
    pub l: usize,
    pub ms: Vec<i32>,
    pub rows: Vec<usize>,
    pub r: Vec<f64>,
    pub values: Vec<f64>,
    pub support: f64,
}
pub struct Basis {
    pub shells: Shells,
    pub tables: Vec<Table>,
}

/// The tables as shells: `(table, confinement)` per shell, rows assigned in order.
pub fn shells_of(ctx: &Context, tables: Vec<Table>, confinement: Option<f64>) -> Basis {
    let shells = tables
        .iter()
        .map(|t| Shell {
            center: t.center,
            l: t.l,
            ms: t.ms.clone(),
            rows: t.rows.clone(),
            radial: Arc::new(Radial::new(ctx, &t.r, &t.values, confinement).unwrap()),
        })
        .collect();
    Basis {
        shells: Shells::new(shells).unwrap(),
        tables,
    }
}

/// Where a table stops counting: the node after its last value above 1e-12 of the peak.
pub fn support_of(r: &[f64], values: &[f64]) -> f64 {
    let peak = values.iter().fold(0f64, |a, v| a.max(v.abs()));
    let last = (0..values.len())
        .rev()
        .find(|&i| values[i].abs() > 1e-12 * peak)
        .unwrap();
    r[(last + 1).min(r.len() - 1)]
}

/// The first zeta of channel `l`: the pseudo wave confined so that its level lies 0.1 eV above
/// the free one (the PAW-LCAO default `energy_shift`), with the default wall (12 Ha, from 0.6 of
/// the radius).
pub fn confined_radial(ctx: &Context, ds: &PawDataset, l: u32) -> Vec<f64> {
    let channel = &ds.channels[&l];
    let rank = channel.projectors.len();
    let projectors: Vec<f64> = channel
        .projectors
        .iter()
        .flat_map(|p| p.as_slice().iter().copied())
        .collect();
    let data = ChannelData {
        table: ds.r.as_slice(),
        local: ds.v_local_screened.as_slice(),
        projectors: &projectors,
        rank,
        coupling: channel.coupling_screened.data(),
        overlap: channel.overlap_correction.data(),
        wave: channel.pseudo_waves[0].as_slice(),
        l: l as usize,
        r_cut: channel.r_cut,
    };
    let prepared = Prepared::new(ctx, data).unwrap();
    let wall = Potential {
        amplitude: 12.,
        inner_fraction: 0.6,
    };
    orbital(ctx, &prepared, 0.1, &wall, 0, None)
        .unwrap()
        .unwrap_or_else(|reach| panic!("a shift of 0.1 eV does not fit the table (reach {reach})"))
        .radial
}

/// The minimal basis of a dataset on every atom: the confined, filtered first pseudo wave of
/// each channel with a valence state, `m = -l..l`, atom major. This is the SZ basis of the
/// default PAW-LCAO options at the filter cutoff `k_c`.
pub fn minimal_basis(ctx: &Context, ds: &PawDataset, centers: &[[f64; 3]], k_c: f64) -> Basis {
    let mut tables = Vec::new();
    let mut row = 0;
    let confined: Vec<(u32, Vec<f64>)> = ds
        .channels
        .iter()
        .filter(|(_, channel)| channel.occupation > 0.)
        .map(|(&l, _)| (l, confined_radial(ctx, ds, l)))
        .collect();
    for center in centers {
        for (l, radial) in &confined {
            let l = *l as usize;
            let (values, _) = filter::filter(
                ctx,
                ds.r.as_slice(),
                radial,
                l,
                k_c,
                &filter::Options::default(),
            )
            .unwrap();
            let ms: Vec<i32> = (-(l as i32)..=l as i32).collect();
            tables.push(Table {
                center: *center,
                l,
                rows: (row..row + ms.len()).collect(),
                ms,
                r: ds.r.as_slice().to_vec(),
                support: support_of(ds.r.as_slice(), &values),
                values,
            });
            row += 2 * l + 1;
        }
    }
    shells_of(ctx, tables, None)
}

/// `sum_R e^{i k.R} R_l(|r - c - R|) Y_lm`, written with no code from the crate under test but
/// the table evaluation and the harmonics: `(k, function, point)`. Images are every lattice vector
/// of the box that holds a ball of radius `reach`; a function is cut at its `support`.
pub fn image_sum(
    ctx: &Context,
    basis: &Basis,
    lattice: &Matrix3<f64>,
    kpoints: &[[f64; 3]],
    points: &[[f64; 3]],
) -> Vec<Complex64> {
    let functions: usize = basis.tables.iter().map(|t| t.ms.len()).sum();
    let mut out = vec![Complex64::new(0., 0.); kpoints.len() * functions * points.len()];
    let lengths: Vec<f64> = (0..3).map(|i| lattice.column(i).norm()).collect();
    let spread = points
        .iter()
        .map(|p| Vector3::from(*p).norm())
        .fold(0., f64::max);
    for t in &basis.tables {
        let support = &t.support;
        let reach = support + spread + Vector3::from(t.center).norm() + 1.;
        let n: Vec<i64> = lengths
            .iter()
            .map(|a| (reach / a).ceil() as i64 + 2)
            .collect();
        for i in -n[0]..=n[0] {
            for j in -n[1]..=n[1] {
                for k in -n[2]..=n[2] {
                    let shift = lattice * Vector3::new(i as f64, j as f64, k as f64);
                    let origin = Vector3::from(t.center) + shift;
                    let mut radius = Vec::with_capacity(points.len());
                    for p in points {
                        radius.push((Vector3::from(*p) - origin).norm());
                    }
                    let radial =
                        table::evaluate(ctx, &t.r, &t.values, t.l, &radius, false).unwrap();
                    for (kk, q) in kpoints.iter().enumerate() {
                        let angle = q[0] * shift[0] + q[1] * shift[1] + q[2] * shift[2];
                        let phase = Complex64::new(angle.cos(), angle.sin());
                        for (&m, &row) in t.ms.iter().zip(&t.rows) {
                            for (ip, p) in points.iter().enumerate() {
                                if radius[ip] >= *support || radius[ip] == 0. {
                                    continue;
                                }
                                let d = (Vector3::from(*p) - origin) / radius[ip];
                                let y = mandacaru_core::math::angular::complex(
                                    t.l,
                                    m,
                                    [d[0], d[1], d[2]],
                                );
                                out[(kk * functions + row) * points.len() + ip] +=
                                    phase * y * radial[ip];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub struct Frozen {
    pub points: Vec<[f64; 3]>,
    pub kpoints: Vec<[f64; 3]>,
    pub values: Vec<Complex64>,
    pub image_sum: Vec<Complex64>,
}
pub fn frozen_orbitals() -> Frozen {
    let body = reference("orbitals_si_sz.json");
    Frozen {
        points: triples(&real_array(&body["points"])),
        kpoints: triples(&real_array(&body["k_cartesian"])),
        values: complex_array(&body["values"]),
        image_sum: complex_array(&body["image_sum"]),
    }
}
pub fn silicon(ctx: &Context) -> (Cell, Basis) {
    let cell = Cell::read("si_sz");
    let ds = dataset(ctx, "lda-sr", "Si");
    let basis = minimal_basis(ctx, &ds, &cell.centers, cell.filter_wavevector);
    (cell, basis)
}
/// The ball of the cell, grown until it holds every point (the kernel culls images against it).
pub fn region_for(cell: &Cell, points: &[[f64; 3]]) -> Region {
    let farthest = points
        .iter()
        .map(|p| (Vector3::from(*p) - Vector3::from(cell.region.center)).norm())
        .fold(0., f64::max);
    Region {
        center: cell.region.center,
        radius: cell.region.radius.max(farthest + 1e-6),
    }
}
pub fn bloch(
    ctx: &Context,
    basis: &Basis,
    cell: &Cell,
    kpoints: &[[f64; 3]],
    points: &[[f64; 3]],
) -> Vec<Complex64> {
    let region = region_for(cell, points);
    let columns: Vec<Vec<f64>> = (0..3)
        .map(|axis| points.iter().map(|p| p[axis]).collect())
        .collect();
    let functions = basis.shells.functions();
    let mut out = vec![Complex64::new(0., 0.); kpoints.len() * functions * points.len()];
    basis
        .shells
        .values(
            ctx,
            &Sampling {
                lattice: &cell.lattice,
                region,
                kpoints,
            },
            [&columns[0], &columns[1], &columns[2]],
            &mut DenseRows::new(&mut out, functions, points.len()),
        )
        .unwrap();
    out
}

/// The projectors of every atom, `l` ascending, then `m`, then the radial index: the raw
/// (`raw`) or the dual tables, each with the radius `max(r_cut(l), r_cut_local)`.
pub fn projector_list(
    ctx: &Context,
    ds: &PawDataset,
    centers: &[[f64; 3]],
    raw: bool,
) -> Vec<Projector> {
    let mut out = Vec::new();
    for (atom, center) in centers.iter().enumerate() {
        for (&l, channel) in &ds.channels {
            let tables = if raw {
                &channel.raw_projectors
            } else {
                &channel.projectors
            };
            let radials: Vec<_> = tables
                .iter()
                .map(|t| Arc::new(Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).unwrap()))
                .collect();
            for m in -(l as i32)..=l as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    out.push(Projector {
                        atom,
                        l: l as usize,
                        m,
                        index,
                        center: *center,
                        r_cut: channel.r_cut.max(ds.r_cut_local),
                        radial: radial.clone(),
                    });
                }
            }
        }
    }
    out
}

/// The spin-orbit projectors (every `l` of the record, dual tables) and the `(atom, l)` couplings.
pub fn spin_orbit_list(
    ctx: &Context,
    ds: &PawDataset,
    centers: &[[f64; 3]],
) -> (
    Vec<Projector>,
    BTreeMap<(usize, usize), mandacaru_core::models::spin_orbit::Coupling>,
) {
    let (mut out, mut couplings) = (Vec::new(), BTreeMap::new());
    for (atom, center) in centers.iter().enumerate() {
        for (&l, entry) in &ds.spin_orbit {
            let r_cut = ds.channels[&l].r_cut.max(ds.r_cut_local);
            let radials: Vec<_> = entry
                .projectors
                .iter()
                .map(|t| Arc::new(Radial::new(ctx, ds.r.as_slice(), t.as_slice(), None).unwrap()))
                .collect();
            for m in -(l as i32)..=l as i32 {
                for (index, radial) in radials.iter().enumerate() {
                    out.push(Projector {
                        atom,
                        l: l as usize,
                        m,
                        index,
                        center: *center,
                        r_cut,
                        radial: radial.clone(),
                    });
                }
            }
            couplings.insert(
                (atom, l as usize),
                mandacaru_core::models::spin_orbit::Coupling {
                    radial: entry.coupling.shape()[0],
                    values: entry.coupling.data().to_vec(),
                },
            );
        }
    }
    (out, couplings)
}

/// How far the AXIS of the magnetization of a spinor density (`[n, m_x, m_y, m_z]`, one block of
/// points each) turns in space, in radians: the largest angle between the line of `m` at a point
/// where `|m|` is above a tenth of its peak and its line at the peak (a sign change of a collinear
/// antiferromagnet is a swap of the two spin channels, not a turn). A collinear state is at
/// rounding, and a state whose magnetization peaks below 1e-4 (a nonmagnetic or time-reversal-symmetric
/// one: its direction is noise) returns zero.
pub fn frame_turn(density: &[f64]) -> f64 {
    let n = density.len() / 4;
    assert_eq!(density.len(), 4 * n, "a spinor density has four blocks");
    let m = |a: usize, k: usize| density[(a + 1) * n + k];
    let norm = |k: usize| (0..3).map(|a| m(a, k).powi(2)).sum::<f64>().sqrt();
    let Some(peak) = (0..n).max_by(|&i, &j| norm(i).total_cmp(&norm(j))) else {
        return 0.;
    };
    // Below this the magnetization is numerical noise, whose direction means nothing.
    if norm(peak) < 1e-4 {
        return 0.;
    }
    (0..n)
        .filter(|&k| norm(k) > 0.1 * norm(peak))
        .map(|k| {
            ((0..3).map(|a| m(a, k) * m(a, peak)).sum::<f64>() / (norm(k) * norm(peak)))
                .abs()
                .clamp(0., 1.)
                .acos()
        })
        .fold(0., f64::max)
}

/// Three hydrogen atoms on an equilateral triangle of side 5 Bohr in a skew 10 x 10 x 8 Bohr cell
/// (one grid node per Bohr, `strain` applied as `x -> (1 + strain) x` with the ion width and filter
/// cutoff pinned to the reference cell), as a spinor crystal on the fully relativistic set. The SCF
/// started with the three moments at 120 degrees in the xy plane ([`textured_triangle`]) converges
/// in about 95 iterations to a noncollinear state of net moment 0.825 whose magnetization axis turns
/// in space by 1.556 rad, with no spin-orbit term (hydrogen has only s channels) and no partial
/// core. The atoms are weakly coupled, so the angles between the moments are a nearly flat
/// direction of the energy: every SCF of a finite difference restarts from the reference density
/// ([`restart`]). The cored variant of the fixture is [`triangle_species`]. It is the one such
/// state found cheap and stable. Gold hydride at a moment of 0.5
/// decays to the nonmagnetic state (its converged magnetization peaks at 1.5e-10, noise, so the axis
/// `frame_turn` it gives is meaningless; at the geometry of the hydrogen pair of `periodic_gradient_xc`
/// it converges in 276 iterations, in 300 not at all on the other cell), the spinor r2SCAN of
/// IrH, ReH, OsH, PtH, WH and TaH does not converge, a triangle of hydrogen at 2 Bohr
/// (a 120 degree state, but a saddle: the ferromagnetic state is 8 mHa lower and displaced SCFs
/// flow to it), of lithium, chromium and manganese does not converge or ends collinear, and a
/// lithium atom next to a triangle of hydrogen collapses it to a collinear or nonmagnetic state.
pub fn triangle_input(
    files: Vec<(String, String)>,
    strain: Matrix3<f64>,
) -> mandacaru_core::periodic::setup::parse::SetupInput {
    use mandacaru_core::periodic::setup::parse::AtomSpec;
    let lattice = Matrix3::from_row_slice(&[10., 1., 0.5, 0., 10., 1., 0., 0., 8.]);
    let reference = lattice;
    let spacing = (0..3)
        .map(|a| reference.column(a).norm() / [10., 10., 8.][a])
        .fold(0f64, f64::max);
    let center = reference * Vector3::new(0.5, 0.5, 0.5);
    let inverse = reference.try_inverse().unwrap();
    let radius = 5. / 3f64.sqrt();
    let atoms = (0..3)
        .map(|i| {
            let phi = 2. * std::f64::consts::PI * i as f64 / 3.;
            let f = inverse * (center + Vector3::new(radius * phi.cos(), radius * phi.sin(), 0.));
            AtomSpec {
                species: 0,
                symbol: "H".into(),
                fractional: [f[0], f[1], f[2]],
                moment: 1.,
            }
        })
        .collect();
    mandacaru_core::periodic::setup::parse::SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: (Matrix3::identity() + strain) * reference,
        grid_shape: [10, 10, 8],
        atoms,
        kpoints_frac: vec![[-0.25, 0., 0.], [0.25, 0., 0.]],
        weights: vec![0.5, 0.5],
        functional: "r2scan".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: true,
        spin: false,
        symmetry: false,
        filter_cutoff: Some(std::f64::consts::PI / spacing),
        sigma: Some(1.4 * spacing),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    }
}

/// The crystal of a three-atom `input` whose starting magnetization is the three atoms' moments
/// at 120 degrees in the xy plane (the setup itself starts every moment along z): the starting
/// field of each atom is the z block of the setup with only that atom magnetic, turned to its
/// direction.
pub fn textured_triangle(
    ctx: &Context,
    input: &mandacaru_core::periodic::setup::parse::SetupInput,
    datasets: &[Arc<PawDataset>],
) -> mandacaru_core::periodic::Crystal {
    use mandacaru_core::periodic::setup::parse::create_setup;
    let build = |input| {
        create_setup(ctx, input, datasets.to_vec())
            .unwrap()
            .build(ctx)
            .unwrap()
    };
    let mut crystal = build(input.clone());
    let n = crystal.grid.index.points();
    let mut texture = vec![0.; 3 * n];
    for i in 0..3 {
        let phi = 2. * std::f64::consts::PI * i as f64 / 3.;
        let direction = [phi.cos(), phi.sin(), 0.];
        let mut one = input.clone();
        for (j, atom) in one.atoms.iter_mut().enumerate() {
            atom.moment = if j == i { input.atoms[i].moment } else { 0. };
        }
        let single = build(one);
        for a in 0..3 {
            for k in 0..n {
                texture[a * n + k] += direction[a] * single.initial_m.as_slice()[2 * n + k];
            }
        }
    }
    crystal.initial_m.as_mut_slice().copy_from_slice(&texture);
    crystal
}

/// Starts the SCF of `crystal` (a spinor crystal) from `density`, the converged `[n, m_x, m_y, m_z]`
/// of a neighboring geometry: a noncollinear texture has several nearby solutions, and a
/// finite difference of its energy needs every displaced SCF on the reference's.
pub fn restart(crystal: &mut mandacaru_core::periodic::Crystal, density: &[f64]) {
    let n = crystal.grid.index.points();
    assert_eq!(density.len(), 4 * n, "a spinor density has four blocks");
    crystal
        .initial
        .as_mut_slice()
        .copy_from_slice(&density[..n]);
    crystal
        .initial_m
        .as_mut_slice()
        .copy_from_slice(&density[n..]);
}

/// Peak (electrons per Bohr cubed) and 1/e radius (Bohr) of the synthetic smooth core of the
/// hydrogen of [`triangle_species`]: `PEAK exp(-(r / WIDTH)^2)`, 0.22 electrons in all. Measured
/// 2026-11-16: with it the textured triangle converges in 281 iterations to net moment 0.838 and a
/// magnetization axis that turns by 1.535 rad, and the Pauli-charge-channel expression of the core
/// term (the one before MG-F8 5a) misses the reconverged finite difference by 5.7e-6, 3.6e-6 and
/// 2.7e-6 Ha/Bohr in the three gated force components (2.7 to 5.7 gates) and by 2.6e-6 to
/// 3.1e-5 Ha in the six stress entries, where the exact term misses by 6e-10 to 2.4e-7 and 6e-10 to
/// 1.5e-7. At a peak of 0.05 the displaced SCFs of the force gate no longer converge in 1000
/// iterations, at 0.1 the reference SCF does not converge in 400.
pub const TRIANGLE_CORE_PEAK: f64 = 0.04;
pub const TRIANGLE_CORE_WIDTH: f64 = 1.0;

/// The hydrogen of the triangle fixtures: the shipped record, or the same record with a synthetic
/// smooth partial core attached (it is not a pseudopotential: the library has no hydrogen with a
/// core, and none of the heavier atoms tried keeps a turning magnetization with one, `triangle_input`). The
/// forces and the stress read the records again from their files, so the synthetic record is
/// written to the temporary directory (removed on drop) and its digest taken from that file.
pub struct TriangleSpecies {
    pub files: Vec<(String, String)>,
    pub datasets: Vec<Arc<PawDataset>>,
    scratch: Option<PathBuf>,
}

impl Drop for TriangleSpecies {
    fn drop(&mut self) {
        if let Some(path) = &self.scratch {
            let _ = std::fs::remove_file(path);
        }
    }
}

pub fn triangle_species(ctx: &Context, core: bool) -> TriangleSpecies {
    use mandacaru_core::{
        parquet::Codec, paw::digest::dataset_content_sha256, paw::record::WriteSpec,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let shipped =
        PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH is unset"))
            .join("pbe-fr/H.parquet");
    if !core {
        let digest = dataset_content_sha256(ctx, &shipped).unwrap();
        return TriangleSpecies {
            files: vec![(shipped.to_string_lossy().into_owned(), digest)],
            datasets: vec![Arc::new(dataset(ctx, "pbe-fr", "H"))],
            scratch: None,
        };
    }
    let mut record = dataset(ctx, "pbe-fr", "H");
    let r = record.r.as_slice().to_vec();
    let table = |r: f64| TRIANGLE_CORE_PEAK * (-(r / TRIANGLE_CORE_WIDTH).powi(2)).exp();
    for (to, &r) in record.smooth_core_density.as_mut_slice().iter_mut().zip(&r) {
        *to = table(r);
    }
    // The charge of the table (trapezoid rule), as the record's own field.
    let charge: f64 = r
        .windows(2)
        .map(|w| 0.5 * (w[1] - w[0]) * (table(w[0]) * w[0] * w[0] + table(w[1]) * w[1] * w[1]))
        .sum::<f64>()
        * 4.
        * std::f64::consts::PI;
    record.nlcc.applied = true;
    record.nlcc.source = Some("smooth_core".into());
    record.nlcc.partial_core_electrons = Some(charge);
    let path = std::env::temp_dir().join(format!(
        "mandacaru_triangle_core_{}_{}.parquet",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    record
        .write(
            ctx,
            &path,
            &WriteSpec {
                stride: 1,
                codec: Codec::Zstd,
            },
        )
        .unwrap();
    let digest = dataset_content_sha256(ctx, &path).unwrap();
    TriangleSpecies {
        files: vec![(path.to_string_lossy().into_owned(), digest)],
        datasets: vec![Arc::new(PawDataset::read(ctx, &path).unwrap())],
        scratch: Some(path),
    }
}

/// A grid proxy of what the Pauli-charge-channel expression of the core term (the one before
/// MG-F8 5a) adds to the force of a rigid translation of the whole crystal, per axis (Ha/Bohr):
/// the converged state's `v_exact - v_pauli` and `v_tau,exact - v_tau,pauli` against the SPECTRAL
/// gradient of the core and of its kinetic density, not the slope of the cubic the engine places
/// (it is 3.0e-5 along x where the engine's per-atom differences sum to about 7e-6). It is zero for
/// a state whose frame does not turn or whose core is nowhere magnetic: the floor that makes the
/// cored gates count. The exact old-minus-new numbers are in `dev_plan/DECISIONS.md`.
pub fn pauli_core_force_gap(
    crystal: &mandacaru_core::periodic::Crystal,
    result: &mandacaru_core::periodic::PeriodicResult,
) -> [f64; 3] {
    use mandacaru_core::{math::fft::Fft3, periodic::spinor, quantum::C64};
    let ctx = &crystal.context;
    let n = crystal.grid.index.points();
    let tau = result
        .kinetic_density
        .as_ref()
        .expect("a meta-GGA state keeps its kinetic density");
    let mut fft = Fft3::new(ctx, crystal.grid.index.shape()).unwrap();
    let terms = spinor::exchange_correlation(
        crystal,
        &mut fft,
        result.density.as_slice(),
        Some(tau.as_slice()),
    )
    .unwrap();
    let sources = terms
        .core
        .as_ref()
        .expect("a spinor meta-GGA returns its core sources");
    let pauli_kinetic = terms
        .kinetic_potential
        .as_ref()
        .expect("a meta-GGA returns v_tau");
    let core_tau = crystal
        .core_tau
        .as_ref()
        .expect("a meta-GGA crystal has its core kinetic density");
    let mut gap = [0.; 3];
    for (axis, to) in gap.iter_mut().enumerate() {
        let mut slope = |field: &[f64]| -> Vec<f64> {
            let mut work: Vec<C64> = field.iter().map(|v| C64::new(*v, 0.)).collect();
            fft.transform(&mut work, false).unwrap();
            for (k, w) in work.iter_mut().enumerate() {
                *w *= C64::new(0., crystal.xc_wave.as_slice()[axis * n + k]);
            }
            fft.transform(&mut work, true).unwrap();
            work.iter().map(|v| v.re).collect()
        };
        let (d_core, d_tau) = (slope(crystal.core.as_slice()), slope(core_tau.as_slice()));
        let dv = crystal.grid.index.dv();
        *to = dv
            * (0..n)
                .map(|k| {
                    (sources.potential.as_slice()[k] - terms.potential.as_slice()[k]) * d_core[k]
                        + (sources.kinetic.as_slice()[k] - pauli_kinetic.as_slice()[k]) * d_tau[k]
                })
                .sum::<f64>();
    }
    gap
}
