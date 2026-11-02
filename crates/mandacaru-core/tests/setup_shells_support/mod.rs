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
        spin_orbit::Coupling,
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
) -> (Vec<Projector>, BTreeMap<(usize, usize), Coupling>) {
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
                Coupling {
                    radial: entry.coupling.shape()[0],
                    values: entry.coupling.data().to_vec(),
                },
            );
        }
    }
    (out, couplings)
}
