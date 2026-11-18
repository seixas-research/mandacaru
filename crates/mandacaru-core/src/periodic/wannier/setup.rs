//! The Wannier calculation of a converged crystal: the request (mesh, trials, bands, windows),
//! the mesh frames made from the crystal's own setup (`bloch_mesh`), the localization of every
//! spin channel and its result. Everything numerical is native; the request carries only what
//! the user chose, never geometry the setup already owns.
use super::{
    interpolation, invalid,
    model::{self, Channel, Windows},
    populations::{AtomPartition, PopulationScheme},
    sampling::{self, Trial},
    Localization, Matrix, Options,
};
use crate::{
    periodic::{
        berry::hall::Hall,
        bloch_mesh::{check_isolated, MeshIndex, Sampler},
        setup::PeriodicSetup,
        Crystal, PeriodicResult,
    },
    product,
    quantum::C64,
    Reservation, Result,
};
use serde_json::Value;
use std::sync::Arc;

pub struct Wannier {
    pub crystal: Arc<Crystal>,
    pub state: Arc<PeriodicResult>,
    pub size: [usize; 3],
    pub bands: Vec<usize>,
    pub trials: Vec<Trial>,
    pub channels: Vec<Channel>,
    pub localization: Localization,
    /// The periodic setup the crystal was built from, which makes states at any k-point (the
    /// mesh frames, and the long-wavelength limit of the screening).
    pub setup: Arc<PeriodicSetup>,
    _memory: Reservation,
}

/// The keys a Wannier request may carry; any other key raises (the geometry is the setup's).
const REQUEST_KEYS: [&str; 8] = [
    "size",
    "trials",
    "n_functions",
    "windows",
    "bands",
    "localization",
    "populations",
    "exponent",
];
fn real(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid("Wannier inputs must contain finite numbers"))
}
fn vector(value: &Value) -> Result<[f64; 3]> {
    let values = value
        .as_array()
        .filter(|v| v.len() == 3)
        .ok_or_else(|| invalid("Wannier positions need three Cartesian components in Bohr"))?;
    Ok([real(&values[0])?, real(&values[1])?, real(&values[2])?])
}
fn pair(value: &Value) -> Result<[f64; 2]> {
    let values = value
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or_else(|| invalid("Wannier windows need two energies in Hartree"))?;
    Ok([real(&values[0])?, real(&values[1])?])
}
fn integer(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("Wannier dimensions and indices must be nonnegative integers"))
}

type AngularParts = Vec<(String, Vec<(usize, i32, f64)>)>;
fn named(name: &str) -> Result<AngularParts> {
    let orbitals = [
        ("s", 0, 0),
        ("py", 1, -1),
        ("pz", 1, 0),
        ("px", 1, 1),
        ("dxy", 2, -2),
        ("dyz", 2, -1),
        ("dz2", 2, 0),
        ("dxz", 2, 1),
        ("dx2-y2", 2, 2),
    ];
    if let Some((_, l, m)) = orbitals.iter().find(|(n, _, _)| *n == name) {
        return Ok(vec![(name.into(), vec![(*l, *m, 1.)])]);
    }
    if let Some(l) = ["s", "p", "d", "f"].iter().position(|n| *n == name) {
        return Ok((-(l as i32)..=l as i32)
            .map(|m| {
                (
                    orbitals
                        .iter()
                        .find(|(_, a, b)| *a == l && *b == m)
                        .map_or_else(|| format!("{name}:{m}"), |(n, _, _)| n.to_string()),
                    vec![(l, m, 1.)],
                )
            })
            .collect());
    }
    if name == "t2g" || name == "eg" {
        let subset: &[&str] = if name == "t2g" {
            &["dxy", "dyz", "dxz"]
        } else {
            &["dz2", "dx2-y2"]
        };
        return subset
            .iter()
            .map(|n| named(n))
            .collect::<Result<Vec<_>>>()
            .map(|v| v.into_iter().flatten().collect());
    }
    let rows: Vec<[f64; 4]> = match name {
        "sp" => vec![
            [1. / 2f64.sqrt(), 0., 0., 1. / 2f64.sqrt()],
            [1. / 2f64.sqrt(), 0., 0., -1. / 2f64.sqrt()],
        ],
        "sp2" => vec![
            [1. / 3f64.sqrt(), -1. / 6f64.sqrt(), 1. / 2f64.sqrt(), 0.],
            [1. / 3f64.sqrt(), -1. / 6f64.sqrt(), -1. / 2f64.sqrt(), 0.],
            [1. / 3f64.sqrt(), 2. / 6f64.sqrt(), 0., 0.],
        ],
        "sp3" => vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ],
        _ => return Err(invalid(format!("unknown Wannier trial orbital {name}"))),
    };
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| (format!("{name}-{}", i + 1), hybrid(row)))
        .collect())
}
fn hybrid(row: [f64; 4]) -> Vec<(usize, i32, f64)> {
    [(0, 0), (1, 1), (1, -1), (1, 0)]
        .into_iter()
        .zip(row)
        .filter(|(_, c)| *c != 0.)
        .map(|((l, m), c)| (l, m, c))
        .collect()
}
fn trials(value: &Value, cell: nalgebra::Matrix3<f64>) -> Result<Vec<Trial>> {
    let inverse = cell
        .try_inverse()
        .ok_or_else(|| invalid("singular Wannier trial cell"))?;
    let mut out = vec![];
    for raw in value
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("Wannier setup needs localized trial orbitals"))?
    {
        let center = vector(&raw["center"])?;
        let width = if raw["width"].is_null() {
            1.
        } else {
            real(&raw["width"])?
        };
        let label = raw["label"].as_str().unwrap_or("trial");
        if let Some(neighbors) = raw["bond_neighbors"].as_array() {
            let neighbors = neighbors.iter().map(vector).collect::<Result<Vec<_>>>()?;
            let norm = |v: &[f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nearest = neighbors
                .iter()
                .map(norm)
                .filter(|r| *r > 1e-8)
                .fold(f64::INFINITY, f64::min);
            for direction in neighbors
                .iter()
                .filter(|v| norm(v) <= 1.1 * nearest && norm(v) > 1e-8)
            {
                let midpoint =
                    nalgebra::Vector3::from(center) + 0.5 * nalgebra::Vector3::from(*direction);
                let fractional = (inverse * midpoint).map(|v| v.rem_euclid(1.));
                if out.iter().any(|t: &Trial| {
                    let delta = inverse * nalgebra::Vector3::from(t.center) - fractional;
                    delta.iter().all(|v| (v - v.round()).abs() < 1e-6)
                }) {
                    continue;
                }
                out.push(Trial {
                    center: (cell * fractional).into(),
                    angular: vec![(0, 0, 1.)],
                    width,
                    spin: None,
                    label: format!("bond:{}", out.len() + 1),
                });
            }
            continue;
        }
        if let Some(neighbors) = raw["neighbors"].as_array() {
            let neighbors = neighbors.iter().map(vector).collect::<Result<Vec<_>>>()?;
            let norm = |v: &[f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nearest = neighbors
                .iter()
                .map(norm)
                .filter(|r| *r > 1e-8)
                .fold(f64::INFINITY, f64::min);
            let directions = neighbors
                .iter()
                .filter(|v| norm(v) <= 1.1 * nearest && norm(v) > 1e-8)
                .collect::<Vec<_>>();
            if directions.len() != 4 {
                return Err(invalid(format!(
                    "{label} has {} nearest neighbors; an sp3 set needs four",
                    directions.len()
                )));
            }
            for (bond, direction) in directions.into_iter().enumerate() {
                let r = norm(direction);
                let trial = Trial {
                    center,
                    angular: hybrid([
                        0.5,
                        0.5 * 3f64.sqrt() * direction[0] / r,
                        0.5 * 3f64.sqrt() * direction[1] / r,
                        0.5 * 3f64.sqrt() * direction[2] / r,
                    ]),
                    width,
                    spin: None,
                    label: format!("{label}:sp3-{}", bond + 1),
                };
                trial.validate()?;
                out.push(trial);
            }
            continue;
        }
        let parts = if !raw["direction"].is_null() {
            let direction = vector(&raw["direction"])?;
            let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm <= 1e-12 {
                return Err(invalid("an sp3 trial needs a nonzero bond direction"));
            }
            vec![(
                "sp3".into(),
                hybrid([
                    0.5,
                    0.5 * 3f64.sqrt() * direction[0] / norm,
                    0.5 * 3f64.sqrt() * direction[1] / norm,
                    0.5 * 3f64.sqrt() * direction[2] / norm,
                ]),
            )]
        } else {
            named(
                raw["orbital"]
                    .as_str()
                    .ok_or_else(|| invalid("a Wannier trial needs an orbital name"))?,
            )?
        };
        for (name, angular) in parts {
            let trial = Trial {
                center,
                angular,
                width,
                spin: None,
                label: format!("{label}:{name}"),
            };
            trial.validate()?;
            out.push(trial);
        }
    }
    Ok(out)
}

impl Wannier {
    /// Localize the Bloch states of a converged crystal as `source` (JSON: `size`, `trials`,
    /// optional `n_functions`, `windows` in Hartree on the eigenvalue scale, `bands`) asks. The
    /// frames of the full Gamma-anchored mesh are made from `setup`; the PAW one-center geometry
    /// is the setup's. Without windows the bands must form a group isolated from every other band
    /// at every mesh point. A spinor crystal without time reversal is refused when its subspace
    /// carries a Chern number ([`super::chern_guard`]).
    pub fn build(
        crystal: Arc<Crystal>,
        state: Arc<PeriodicResult>,
        setup: Arc<PeriodicSetup>,
        source: &str,
        options: &Options,
    ) -> Result<Self> {
        let ctx = crystal.context.clone();
        ctx.run(|| Self::build_inner(crystal, state, setup, source, options))
    }
    fn build_inner(
        crystal: Arc<Crystal>,
        state: Arc<PeriodicResult>,
        setup: Arc<PeriodicSetup>,
        source: &str,
        options: &Options,
    ) -> Result<Self> {
        let ctx = &crystal.context;
        let memory = ctx.reserve(product(&[source.len(), 32])?)?;
        let raw: Value = serde_json::from_str(source).map_err(|e| invalid(e.to_string()))?;
        let object = raw
            .as_object()
            .ok_or_else(|| invalid("a Wannier request is a JSON object"))?;
        if let Some(key) = object.keys().find(|k| !REQUEST_KEYS.contains(&k.as_str())) {
            return Err(invalid(format!(
                "unknown Wannier request key {key:?}; the keys are {REQUEST_KEYS:?} (the atom \
                 centers and PAW channels are the crystal setup's)"
            )));
        }
        let values = raw["size"]
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("Wannier mesh needs three dimensions"))?;
        let size = [
            integer(&values[0])?,
            integer(&values[1])?,
            integer(&values[2])?,
        ];
        let mesh = MeshIndex::new(size, [0.; 3])?;
        let mut trials = trials(&raw["trials"], sampling::lattice(&crystal.grid))?;
        if crystal.spinor {
            trials = trials
                .into_iter()
                .flat_map(|t| {
                    [0, 1].map(|s| Trial {
                        center: t.center,
                        angular: t.angular.clone(),
                        width: t.width,
                        spin: Some(if s == 0 {
                            [C64::new(1., 0.), C64::default()]
                        } else {
                            [C64::default(), C64::new(1., 0.)]
                        }),
                        label: format!("{}:{}", t.label, if s == 0 { "up" } else { "down" }),
                    })
                })
                .collect();
        }
        if !raw["n_functions"].is_null() && integer(&raw["n_functions"])? != trials.len() {
            return Err(invalid(
                "give one localized trial per requested Wannier function",
            ));
        }
        let windows = if raw["windows"].is_null() {
            None
        } else {
            Some(Windows {
                outer: pair(&raw["windows"]["outer"])?,
                frozen: if raw["windows"]["frozen"].is_null() {
                    None
                } else {
                    Some(pair(&raw["windows"]["frozen"])?)
                },
            })
        };
        let bands = if let Some(bands) = raw["bands"].as_array() {
            bands.iter().map(integer).collect::<Result<Vec<_>>>()?
        } else {
            (0..if windows.is_some() {
                crystal.bands()
            } else {
                trials.len()
            })
                .collect()
        };
        if bands.is_empty()
            || bands.len() < trials.len()
            || bands.iter().any(|b| *b >= crystal.bands())
            || bands
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != bands.len()
        {
            return Err(invalid(
                "Wannier bands must be unique valid indices spanning the trial space",
            ));
        }
        let scheme_str = raw["populations"].as_str().unwrap_or("mulliken");
        if !raw["populations"].is_null() && raw["populations"].as_str().is_none() {
            return Err(invalid("populations scheme must be a string"));
        }
        let scheme = PopulationScheme::parse(scheme_str)?;
        let exponent = if raw["exponent"].is_null() {
            2usize
        } else {
            integer(&raw["exponent"])?
        };
        if ![2, 4].contains(&exponent) {
            return Err(invalid(format!(
                "Pipek-Mezey exponent must be 2 or 4; got {exponent}"
            )));
        }
        let loc_mode = match raw["localization"].as_str() {
            None if raw["localization"].is_null() => options.localization,
            Some("spread") => Localization::Spread,
            Some("populations") => Localization::Populations { scheme, exponent },
            Some(other) => {
                return Err(invalid(format!(
                    "Wannier localization must be 'spread' or 'populations'; got {other:?}"
                )));
            }
            None => return Err(invalid("Wannier localization must be a string")),
        };
        let channel_options = Options {
            localization: loc_mode,
            ..*options
        };

        let sampler = Sampler::new(&crystal, &state, &setup)?;
        let mut sorted = bands.clone();
        sorted.sort_unstable();
        let isolated = windows.is_none();
        let need_overlap = matches!(loc_mode, Localization::Populations { .. });
        let mut frames: Vec<Vec<_>> = (0..crystal.spins).map(|_| Vec::new()).collect();
        let mut ao_metrics: Vec<(Matrix, Matrix)> = Vec::new();
        sampler.visit_points(
            &mesh,
            &vec![bands.clone(); crystal.spins],
            |s, frame, kp| {
                if isolated {
                    // A windowless group must be a smooth subspace: one that touches an outside band
                    // localizes without an error into a gauge that is not smooth.
                    check_isolated(
                        &frame,
                        s,
                        &sorted,
                        "so its Wannier functions are not defined without disentanglement: give \
                     windows= (an outer and a frozen energy window)",
                    )?;
                }
                if need_overlap && s == 0 {
                    ao_metrics.push((kp.overlap.clone(), kp.lowdin.clone()));
                }
                frames[s].push(frame);
                Ok(())
            },
        )?;
        let partition = if need_overlap {
            Some(AtomPartition::from_setup(&setup)?)
        } else {
            None
        };
        let pop_arg = partition.as_ref().map(|p| (p, ao_metrics.as_slice()));

        // Only a Hamiltonian without time-reversal symmetry (a spinor crystal with a moment) can
        // give a subspace a Chern number; every collinear or time-reversal-invariant channel has
        // C = 0 by symmetry.
        let guard = crystal.spinor && !setup.time_reversal;
        let channels = frames
            .into_iter()
            .map(|frames| {
                model::localize_channel(
                    &crystal,
                    &state,
                    frames,
                    model::Localization {
                        size,
                        augmentation: sampler.augmentation(),
                        trials: &trials,
                        windows: windows.as_ref(),
                        reciprocal: sampler.reciprocal,
                        chern_guard: guard,
                        populations: pop_arg,
                    },
                    &channel_options,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        drop(sampler);
        Ok(Wannier {
            crystal,
            state,
            size,
            bands,
            trials,
            channels,
            localization: loc_mode,
            setup,
            _memory: memory,
        })
    }

    /// Interpolate the anomalous Hall conductivity onto `size` via Wannier interpolation (`W4`).
    pub fn anomalous_hall(&self, size: [usize; 3], refine: bool) -> Result<Hall> {
        let models: Vec<&interpolation::Model> =
            self.channels.iter().map(|c| &c.interpolation).collect();
        let total_subspace_e: f64 = self.channels.iter().map(|c| c.subspace_electrons).sum();
        let target_electrons = if self.bands.first() == Some(&0)
            && (total_subspace_e - self.crystal.electrons).abs() < 1e-2
        {
            self.crystal.electrons
        } else {
            total_subspace_e
        };
        interpolation::anomalous_hall(
            &self.crystal.context,
            &models,
            size,
            target_electrons,
            self.state.fermi,
            self.state.options.smearing,
            self.state.options.width,
            self.crystal.degeneracy(),
            refine,
        )
    }
}
