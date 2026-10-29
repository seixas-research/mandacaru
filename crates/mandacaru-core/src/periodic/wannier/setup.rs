//! Incremental full-mesh setup. Geometry adapters transfer one k-point at a
//! time; all eigensolves, trial models and localization remain native.
use super::{
    invalid,
    model::{self, Channel, Windows},
    sampling::{self, Augmentation, Frame, Trial},
    Options,
};
use crate::{
    periodic::{Crystal, KPoint, PeriodicResult},
    product,
    quantum::C64,
    Reservation, Result,
};
use serde_json::Value;
use std::sync::Arc;

pub struct Setup {
    pub crystal: Arc<Crystal>,
    pub state: Arc<PeriodicResult>,
    pub size: [usize; 3],
    pub bands: Vec<usize>,
    pub trials: Vec<Trial>,
    pub augmentation: Augmentation,
    pub windows: Option<Windows>,
    frames: Vec<Vec<Frame>>,
    _memory: Reservation,
}
pub struct Wannier {
    pub crystal: Arc<Crystal>,
    pub state: Arc<PeriodicResult>,
    pub size: [usize; 3],
    pub bands: Vec<usize>,
    pub trials: Vec<Trial>,
    pub augmentation: Augmentation,
    pub channels: Vec<Channel>,
    _memory: Reservation,
}
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

impl Setup {
    pub fn from_json(
        crystal: Arc<Crystal>,
        state: Arc<PeriodicResult>,
        source: &str,
    ) -> Result<Self> {
        let ctx = &crystal.context;
        let memory = ctx.reserve(product(&[source.len(), 32])?)?;
        let raw: Value = serde_json::from_str(source).map_err(|e| invalid(e.to_string()))?;
        let values = raw["size"]
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("Wannier mesh needs three dimensions"))?;
        let size = [
            integer(&values[0])?,
            integer(&values[1])?,
            integer(&values[2])?,
        ];
        if size.contains(&0) {
            return Err(invalid("Wannier mesh dimensions must be positive"));
        }
        product(&size)?;
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
        let centers = raw["centers"]
            .as_array()
            .ok_or_else(|| invalid("Wannier setup needs the atom centers in Bohr"))?
            .iter()
            .map(vector)
            .collect::<Result<Vec<_>>>()?;
        let labels = raw["channel_labels"]
            .as_array()
            .ok_or_else(|| invalid("Wannier setup needs PAW channel labels"))?
            .iter()
            .map(|v| {
                let row = v
                    .as_array()
                    .filter(|v| v.len() == 3)
                    .ok_or_else(|| invalid("invalid PAW channel label"))?;
                let get = |a: usize| {
                    row[a]
                        .as_i64()
                        .and_then(|v| i32::try_from(v).ok())
                        .ok_or_else(|| invalid("invalid PAW channel label"))
                };
                Ok([get(0)?, get(1)?, get(2)?])
            })
            .collect::<Result<Vec<_>>>()?;
        let augmentation = Augmentation { centers, labels };
        augmentation.validate(&crystal)?;
        let frames = (0..crystal.spins).map(|_| vec![]).collect();
        Ok(Self {
            crystal,
            state,
            size,
            bands,
            trials,
            augmentation,
            windows,
            frames,
            _memory: memory,
        })
    }
    pub fn next_point(&self) -> Result<[f64; 3]> {
        sampling::mesh_point(self.size, self.frames[0].len())
    }
    pub fn add_point(&mut self, point: KPoint) -> Result<()> {
        let ctx = self.crystal.context.clone();
        ctx.run(|| self.add_point_inner(point))
    }
    fn add_point_inner(&mut self, point: KPoint) -> Result<()> {
        let expected = self.next_point()?;
        if (0..3).any(|a| (point.fractional[a] - expected[a]).abs() > 1e-8) {
            return Err(invalid(
                "Wannier k-point transfer does not follow the full Gamma mesh",
            ));
        }
        let frames = (0..self.crystal.spins)
            .map(|s| Frame::from_point(&self.crystal, &self.state, &point, s, &self.bands))
            .collect::<Result<Vec<_>>>()?;
        for (channel, frame) in self.frames.iter_mut().zip(frames) {
            channel.push(frame);
        }
        Ok(())
    }
    pub fn finish(self, options: &Options) -> Result<Wannier> {
        let ctx = self.crystal.context.clone();
        ctx.run(|| self.finish_inner(options))
    }
    fn finish_inner(self, options: &Options) -> Result<Wannier> {
        if self
            .frames
            .iter()
            .any(|v| v.len() != product(&self.size).unwrap_or(0))
        {
            return Err(invalid(
                "Wannier localization needs every full-mesh k-point",
            ));
        }
        let channels = self
            .frames
            .into_iter()
            .map(|frames| {
                model::localize_channel(
                    &self.crystal,
                    &self.state,
                    frames,
                    model::Localization {
                        size: self.size,
                        augmentation: &self.augmentation,
                        trials: &self.trials,
                        windows: self.windows.as_ref(),
                    },
                    options,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Wannier {
            crystal: self.crystal,
            state: self.state,
            size: self.size,
            bands: self.bands,
            trials: self.trials,
            augmentation: self.augmentation,
            channels,
            _memory: self._memory,
        })
    }
}
