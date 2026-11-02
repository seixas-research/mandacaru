//! Shared by the `setup_gradients*`, `setup_basis_gradient` and `setup_remainder` tests: the
//! frozen replay files of the periodic WP-X block (`reference/periodic/wpx/<case>.json`, Python
//! outputs with the inputs that produced them and the floors measured per quantity), and the
//! cells, bases and datasets rebuilt from those inputs.
#![allow(dead_code)]

use mandacaru_core::{
    paw::record::PawDataset,
    periodic::setup::{
        ions::LocalPotential,
        shapes::{Atom, Channel, Sites},
        shells::{Radial, Shell, Shells},
    },
    Context,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use serde_json::Value;
use std::{path::PathBuf, sync::Arc};

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

pub fn triples(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// `{shape, dtype, re[, im]}` of the freeze scripts.
pub struct Array {
    pub shape: Vec<usize>,
    pub re: Vec<f64>,
    pub im: Option<Vec<f64>>,
}
impl Array {
    pub fn read(v: &Value) -> Self {
        Self {
            shape: v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect(),
            re: reals(&v["re"]),
            im: v.get("im").map(reals),
        }
    }
    pub fn complex(&self) -> Vec<Complex64> {
        let zero = vec![0.; self.re.len()];
        self.re
            .iter()
            .zip(self.im.as_ref().unwrap_or(&zero))
            .map(|(r, i)| Complex64::new(*r, *i))
            .collect()
    }
}

/// Largest difference over the largest magnitude of the reference (absolute when the reference
/// is zero).
pub fn relative(got: &[Complex64], want: &[Complex64]) -> f64 {
    assert_eq!(got.len(), want.len(), "lengths differ");
    let scale = want.iter().map(|w| w.norm()).fold(0., f64::max);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).norm())
        .fold(0., f64::max);
    if scale == 0. {
        worst
    } else {
        worst / scale
    }
}
pub fn relative_real(got: &[f64], want: &[f64]) -> f64 {
    let c = |v: &[f64]| v.iter().map(|x| Complex64::new(*x, 0.)).collect::<Vec<_>>();
    relative(&c(got), &c(want))
}

/// One frozen case.
pub struct Case {
    pub name: &'static str,
    json: Value,
}
impl Case {
    pub fn load(name: &'static str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/migration/reference/periodic/wpx")
            .join(format!("{name}.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        Self {
            name,
            json: serde_json::from_str(&text).unwrap(),
        }
    }
    pub fn inputs(&self) -> &Value {
        &self.json["inputs"]
    }
    pub fn quantity(&self, key: &str) -> Array {
        Array::read(&self.json["quantities"][key])
    }
    /// The gate of a quantity: `100 x` its measured 1-ulp-input floor, never below 1e-12
    /// (relative to the array maximum).
    pub fn gate(&self, key: &str) -> f64 {
        let floor = self.json["floors"][key]["ulp_rel"].as_f64().unwrap_or(0.);
        (100. * floor).max(1e-12)
    }
    pub fn scale(&self, key: &str) -> f64 {
        self.json["floors"][key]["scale"].as_f64().unwrap()
    }
    pub fn atoms(&self) -> usize {
        self.inputs()["lattice_sums"]["charges"]["shape"][0]
            .as_u64()
            .unwrap() as usize
    }
    pub fn dataset(&self, ctx: &Context, atom: usize) -> PawDataset {
        // The dead-block case carries symbols only; the eight carry the file names.
        let named = self.inputs()["replay"]["datasets"][atom].as_str();
        let symbol = self.inputs()["symbols"][atom].as_str().unwrap_or("");
        let owned = format!("lda-sr/{symbol}.parquet");
        let name = named.unwrap_or(&owned);
        let root = std::env::var_os("MANDACARU_PAW_PATH")
            .filter(|v| !v.is_empty())
            .expect("MANDACARU_PAW_PATH is unset: the gradient tests read the shipped datasets");
        PawDataset::read(ctx, &PathBuf::from(root).join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    }
    pub fn potentials(&self, ctx: &Context) -> Vec<LocalPotential> {
        (0..self.atoms())
            .map(|a| {
                let ds = self.dataset(ctx, a);
                LocalPotential::new(
                    ctx,
                    ds.r.as_slice(),
                    ds.v_local.as_slice(),
                    ds.valence_charge,
                    ds.r_cut_local,
                )
                .unwrap()
            })
            .collect()
    }
    /// The cell of the dead-block case `si_sym` (no `lattice_sums` record): lattice and centers
    /// from the inputs, the width from the frozen split width, charges and compensation radii
    /// from the datasets, channels from the density channels.
    fn cell_of_dead_block(&self) -> Cell {
        let ctx = context();
        let inputs = self.inputs();
        let centers = triples(&Array::read(&inputs["centers_bohr"]).re);
        let datasets: Vec<PawDataset> = (0..centers.len()).map(|a| self.dataset(&ctx, a)).collect();
        Cell {
            atoms: centers
                .iter()
                .zip(&datasets)
                .map(|(c, d)| Atom {
                    center: *c,
                    charge: d.valence_charge,
                    compensation_radius: d.compensation_radius,
                })
                .collect(),
            channels: inputs["density_channels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c[0].as_u64().unwrap() as usize,
                        c[1].as_u64().unwrap() as usize,
                        c[2].as_i64().unwrap() as i32,
                    )
                })
                .collect(),
            lattice: Matrix3::from_row_slice(&Array::read(&inputs["lattice_bohr"]).re),
            sigma: self.quantity("local_split_width").re[0],
        }
    }
    pub fn cell(&self) -> Cell {
        if self.inputs()["lattice_sums"].is_null() {
            return self.cell_of_dead_block();
        }
        let sums = &self.inputs()["lattice_sums"];
        let centers = triples(&Array::read(&sums["centers_bohr"]).re);
        let charges = Array::read(&sums["charges"]).re;
        let radii = reals(&self.inputs()["replay"]["compensation_radius_bohr"]);
        Cell {
            atoms: centers
                .iter()
                .zip(&charges)
                .zip(&radii)
                .map(|((c, z), r)| Atom {
                    center: *c,
                    charge: *z,
                    compensation_radius: *r,
                })
                .collect(),
            channels: sums["channels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c[0].as_u64().unwrap() as usize,
                        c[1].as_u64().unwrap() as usize,
                        c[2].as_i64().unwrap() as i32,
                    )
                })
                .collect(),
            lattice: Matrix3::from_row_slice(&Array::read(&sums["lattice_bohr"]).re),
            sigma: sums["sigma_bohr"].as_f64().unwrap(),
        }
    }
    /// The basis of the case as shells (one per atom and table) and the function list
    /// `(atom, l, m, center)` in crystal order.
    pub fn basis(&self, ctx: &Context) -> (Shells, Vec<(usize, usize, i32, [f64; 3])>) {
        let tables: Vec<(usize, Arc<Radial>)> = self.inputs()["basis_radial_tables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let r = Array::read(&t["r"]).re;
                let values = Array::read(&t["values"]).re;
                (
                    t["l"].as_u64().unwrap() as usize,
                    Arc::new(Radial::new(ctx, &r, &values, None).unwrap()),
                )
            })
            .collect();
        let functions: Vec<_> = self.inputs()["basis_functions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["atom"].as_u64().unwrap() as usize,
                    f["table"].as_u64().unwrap() as usize,
                    f["m"].as_i64().unwrap() as i32,
                    {
                        let c = reals(&f["center_bohr"]);
                        [c[0], c[1], c[2]]
                    },
                )
            })
            .collect();
        let mut shells: Vec<Shell> = Vec::new();
        for (row, &(_, table, m, center)) in functions.iter().enumerate() {
            match shells.last_mut() {
                Some(s) if s.center == center && Arc::ptr_eq(&s.radial, &tables[table].1) => {
                    s.ms.push(m);
                    s.rows.push(row);
                }
                _ => shells.push(Shell {
                    center,
                    l: tables[table].0,
                    ms: vec![m],
                    rows: vec![row],
                    radial: tables[table].1.clone(),
                }),
            }
        }
        let list = functions
            .iter()
            .map(|&(atom, table, m, center)| (atom, tables[table].0, m, center))
            .collect();
        (Shells::new(shells).unwrap(), list)
    }
}

/// The plain inputs of the lattice sums, owned so a test can move an atom.
#[derive(Clone)]
pub struct Cell {
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
    pub sigma: f64,
}
impl Cell {
    pub fn sites(&self) -> Sites<'_> {
        Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: self.sigma,
        }
    }
    pub fn moved(&self, atom: usize, axis: usize, delta: f64) -> Self {
        let mut out = self.clone();
        out.atoms[atom].center[axis] += delta;
        out
    }
    /// The cell and its atoms carried by `x -> (1 + eps) x`.
    pub fn strained(&self, eps: [[f64; 3]; 3]) -> Self {
        let map = Matrix3::identity() + Matrix3::from_fn(|a, b| eps[a][b]);
        let mut out = self.clone();
        out.lattice = map * self.lattice;
        for a in &mut out.atoms {
            a.center = (map * nalgebra::Vector3::from(a.center)).into();
        }
        out
    }
    pub fn translated(&self, by: [f64; 3]) -> Self {
        let mut out = self.clone();
        for a in &mut out.atoms {
            for k in 0..3 {
                a.center[k] += by[k];
            }
        }
        out
    }
}
