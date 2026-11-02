//! The compensation inputs of a spinor fixture and the independent direct Fourier sum of
//! `periodic_screened_augmentation_support::reference` (shared, not copied) built on them.
use super::{library, read, symbol, Matrix};
use crate::setup_support::Shipped;
use mandacaru_core::{
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{exchange_operators_with, ExxDivTreatment, Screening},
        setup::{
            electrostatics::max_wavenumber,
            shapes::{Atom, Channel, Shapes, Sites},
        },
        Crystal,
    },
    units::BOHR_TO_ANGSTROM,
    Context,
};
use nalgebra::{Matrix3, Vector3};

#[path = "../periodic_screened_augmentation_support/reference.rs"]
pub mod reference;

/// Atoms, channels and lattice of a fixture, the compensation radius read from the shipped table.
pub struct Geometry {
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
}

pub fn geometry(name: &str, crystal: &Crystal) -> Geometry {
    let json = read(name);
    let library = library(&json, name);
    let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
    let geometry = &json["provenance"]["geometry"];
    let atoms = geometry["positions_angstrom"]
        .as_array()
        .unwrap()
        .iter()
        .zip(geometry["numbers"].as_array().unwrap())
        .map(|(p, z)| {
            let p: Vec<f64> = p
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let shipped = Shipped::read(&library, symbol(z.as_u64().unwrap()));
            Atom {
                center: [p[0] * to_bohr, p[1] * to_bohr, p[2] * to_bohr],
                charge: shipped.charge,
                compensation_radius: shipped.compensation_radius,
            }
        })
        .collect();
    let channels = json["channel_labels"]
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
        .collect();
    let shape = crystal.grid.index.shape();
    let lattice = crystal.grid.step
        * Matrix3::from_diagonal(&Vector3::new(
            shape[0] as f64,
            shape[1] as f64,
            shape[2] as f64,
        ));
    Geometry {
        atoms,
        channels,
        lattice,
    }
}

impl Geometry {
    pub fn sites(&self) -> Sites<'_> {
        Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: 1.0,
        }
    }
}

/// A hybrid fixture with the shape tables of its spheres (no one-center terms attached).
pub struct Rig {
    pub shapes: Shapes,
    pub geometry: Geometry,
    _context: Context,
}

impl Rig {
    pub fn new(name: &str, crystal: &Crystal) -> Self {
        let geometry = geometry(name, crystal);
        let q_max = max_wavenumber(&crystal.context, &crystal.grid).unwrap();
        let context = Context::new(Some(1), 512 << 20).unwrap();
        let shapes = Shapes::new(&context, &geometry.sites(), q_max).unwrap();
        Self {
            shapes,
            geometry,
            _context: context,
        }
    }

    /// The grid pair plus the compensation charges alone (no sphere model): what the direct
    /// Fourier sum reproduces.
    pub fn grid_and_compensation(
        &self,
        c: &Crystal,
        d: &[Matrix],
        exxdiv: ExxDivTreatment,
    ) -> Vec<Matrix> {
        let sites = self.geometry.sites();
        let augmentation = Augmentation::new(c, sites, &self.shapes).unwrap();
        let screening = Screening {
            omega: 0.11,
            fraction: 0.25,
        };
        exchange_operators_with(c, d, screening, exxdiv, Some(&augmentation))
            .unwrap()
            .operators
    }

    pub fn direct(&self, c: &Crystal, d: &[Matrix], k: usize, exxdiv: ExxDivTreatment) -> Matrix {
        let sites = self.geometry.sites();
        let comp = reference::Compensation {
            sites: &sites,
            shapes: &self.shapes,
            alias_hats: None,
        };
        reference::exchange(c, d, 0, k, 0.11, exxdiv, Some(&comp))
    }
}
