//! Typed radial tables decoded by the Python file-format adapter.
use super::*;
use crate::math::{
    radial::{integrate, validate},
    spline::Cubic,
};
use serde_json::Value;
use std::collections::BTreeMap;

pub type Arrays = BTreeMap<String, Buffer<f64>>;
pub fn table(arrays: &mut Arrays, value: &Value) -> Result<Buffer<f64>> {
    let key = value
        .as_str()
        .ok_or_else(|| invalid("PAW table must name a typed array"))?;
    let out = arrays
        .remove(key)
        .ok_or_else(|| invalid(format!("missing PAW array {key}")))?;
    crate::finite(out.as_slice(), key)?;
    Ok(out)
}
pub fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid("PAW metadata needs a finite number"))
}
pub fn count(v: &Value) -> Result<usize> {
    v.as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("PAW metadata needs a nonnegative integer"))
}
fn matrix(arrays: &mut Arrays, v: &Value, n: usize) -> Result<DMatrix<f64>> {
    let values = table(arrays, v)?;
    if values.as_slice().len() != product(&[n, n])? {
        return Err(invalid("PAW channel matrix dimensions disagree"));
    }
    Ok(DMatrix::from_row_slice(n, n, values.as_slice()))
}
pub struct Channel {
    pub l: usize,
    pub radius: f64,
    pub projectors: Vec<Arc<Cubic>>,
    pub ae: Vec<Vec<f64>>,
    pub ps: Vec<Vec<f64>>,
    pub coupling: DMatrix<f64>,
    pub overlap: DMatrix<f64>,
    pub reference: DMatrix<f64>,
    pub xc: DMatrix<f64>,
    pub core_exchange: DMatrix<f64>,
}
pub struct Species {
    pub atomic_number: u8,
    pub charge: f64,
    pub local_radius: f64,
    pub compensation_radius: f64,
    pub ionic: Cubic,
    pub channels: Vec<Channel>,
    pub xc_core: Cubic,
    pub xc_core_tau: Cubic,
    pub dft_constant: f64,
    pub relativistic_exchange: bool,
    _memory: crate::Reservation,
}
impl Species {
    pub fn local(&self, r: f64) -> f64 {
        let knots = self.ionic.knots.as_slice();
        if r > knots[knots.len() - 1] {
            -self.charge / r
        } else {
            self.ionic.value_slope(r.max(knots[0])).0
        }
    }
    pub fn parse(ctx: &Context, v: &Value, arrays: &mut Arrays, raw: bool) -> Result<Self> {
        if v["spin_orbit"].as_bool() != Some(false) {
            return Err(invalid(
                "native molecular PAW spin-orbit projector terms are not yet implemented",
            ));
        }
        let atomic_number = u8::try_from(count(&v["atomic_number"])?)
            .ok()
            .filter(|z| (1..=118).contains(z))
            .ok_or_else(|| invalid("invalid PAW atomic number"))?;
        let charge = number(&v["charge"])?;
        let r = table(arrays, &v["r"])?;
        let ionic = table(arrays, &v["ionic"])?;
        validate(r.as_slice(), ionic.as_slice())?;
        let core = table(arrays, &v["core"])?;
        validate(r.as_slice(), core.as_slice())?;
        let core_charge = integrate(
            r.as_slice(),
            &r.as_slice()
                .iter()
                .zip(core.as_slice())
                .map(|(r, n)| 4. * PI * r * r * n)
                .collect::<Vec<_>>(),
        )?;
        let entries = v["channels"]
            .as_array()
            .ok_or_else(|| invalid("PAW channels must be a list"))?;
        let occupied_1s = entries.iter().any(|ch| {
            ch["principal_n"].as_u64() == Some(1)
                && ch["occupations"].as_array().is_some_and(|o| {
                    o.first()
                        .and_then(serde_json::Value::as_f64)
                        .is_some_and(|n| n > 0.)
                })
        });
        let ordinary_core = v["frozen_subshells"].as_array().is_some_and(Vec::is_empty)
            && v["semicore_subshells"].as_array().is_some_and(|shells| {
                !shells
                    .iter()
                    .any(|s| s[0].as_u64() == Some(1) && s[1].as_u64() == Some(0))
            });
        let closed_1s = ordinary_core
            && (3..=10).contains(&atomic_number)
            && ((atomic_number as f64 - charge) - 2.).abs() < 1e-8
            && (core_charge - 2.).abs() < 0.02
            && !occupied_1s;
        if core_charge > 1e-6 && !closed_1s {
            return Err(invalid("native PAW frozen-core exchange currently requires an empty core or a single closed 1s shell"));
        }
        if core.as_slice().iter().any(|v| *v < 0.) {
            return Err(invalid("core density must be nonnegative"));
        }
        // For one closed, nodeless 1s shell: n_c = 2 R_1s^2/(4 pi).
        // Recover its tabulated orbital exactly, preserving the dataset's
        // relativistic reference and normalization without an atomic re-solve.
        let _core_memory = ctx.reserve(product(&[r.as_slice().len(), 8])?)?;
        let core_orbital: Vec<_> = core
            .as_slice()
            .iter()
            .map(|n| (2. * PI * n).sqrt())
            .collect();
        let mut core_extent = 0;
        if closed_1s {
            let total = core_charge;
            let mut integrated = 0.;
            for i in 1..r.as_slice().len() {
                let a = r.as_slice()[i - 1];
                let b = r.as_slice()[i];
                integrated += 0.5
                    * (b - a)
                    * 4.
                    * PI
                    * (a * a * core.as_slice()[i - 1] + b * b * core.as_slice()[i]);
                if integrated >= (1. - 1e-14) * total {
                    core_extent = (i + 2).min(r.as_slice().len());
                    break;
                }
            }
        }
        let valence = table(arrays, &v["valence"])?;
        validate(r.as_slice(), valence.as_slice())?;
        let xc_core = table(arrays, &v["xc_core"])?;
        validate(r.as_slice(), xc_core.as_slice())?;
        let total_refs: usize = entries
            .iter()
            .map(|e| e["occupations"].as_array().map_or(0, Vec::len))
            .sum();
        let memory = ctx.reserve(
            product(&[
                r.as_slice().len(),
                total_refs.checked_add(2).ok_or(crate::Error::Overflow)?,
                8,
                8,
            ])?
            .checked_add(product(&[total_refs, total_refs, 8, 12])?)
            .ok_or(crate::Error::Overflow)?,
        )?;
        let mut all_density = core.as_slice().to_vec();
        let mut channels = Vec::new();
        let n = r.as_slice().len();
        let mut transforms = Vec::new();
        for entry in entries {
            let l = count(&entry["l"])?;
            let principal_n = count(&entry["principal_n"])?;
            let radius = number(&entry["radius"])?;
            if l > 8 || principal_n <= l || radius <= 0. {
                return Err(invalid("invalid PAW angular momentum or radius"));
            }
            let occupations = entry["occupations"]
                .as_array()
                .ok_or_else(|| invalid("missing PAW occupations"))?
                .iter()
                .map(number)
                .collect::<Result<Vec<_>>>()?;
            let refs = occupations.len();
            if refs == 0 {
                return Err(invalid("a PAW channel needs reference waves"));
            }
            let ae = table(arrays, &entry["ae"])?;
            let projectors = table(
                arrays,
                &entry[if raw { "raw_projectors" } else { "projectors" }],
            )?;
            if ae.as_slice().len() != product(&[refs, n])?
                || projectors.as_slice().len() != product(&[refs, n])?
            {
                return Err(invalid("PAW wave table dimensions disagree"));
            }
            let coupling = matrix(arrays, &entry["coupling"], refs)?;
            let overlap = matrix(arrays, &entry["overlap"], refs)?;
            let b = matrix(arrays, &entry["vanderbilt"], refs)?;
            let transform = if raw {
                b.clone()
                    .try_inverse()
                    .ok_or_else(|| invalid("singular PAW projector transform"))?
            } else {
                DMatrix::identity(refs, refs)
            };
            let mut dual_ps = vec![vec![0.; n]; refs];
            for i in 0..refs {
                let q = table(arrays, &entry["wavevectors"][i])?;
                let c = table(arrays, &entry["wave_coefficients"][i])?;
                if q.as_slice().len() != c.as_slice().len() {
                    return Err(invalid("PAW Bessel expansion dimensions disagree"));
                }
                let j = crate::kernels::bessel(ctx, l as i32, r.as_slice(), q.as_slice())?;
                for k in 0..n {
                    dual_ps[i][k] = if r.as_slice()[k] <= radius {
                        (0..c.as_slice().len())
                            .map(|z| c.as_slice()[z] * j.as_slice()[k * c.as_slice().len() + z])
                            .sum()
                    } else {
                        ae.as_slice()[i * n + k]
                    };
                    all_density[k] += occupations[i] * ae.as_slice()[i * n + k].powi(2) / (4. * PI);
                }
            }
            let ae_waves: Vec<Vec<f64>> = (0..refs)
                .map(|i| ae.as_slice()[i * n..(i + 1) * n].to_vec())
                .collect();
            let projectors = (0..refs)
                .map(|i| {
                    Cubic::new(
                        ctx,
                        ctx.copy_slice(r.as_slice())?,
                        ctx.copy_slice(&projectors.as_slice()[i * n..(i + 1) * n])?,
                    )
                    .map(Arc::new)
                })
                .collect::<Result<Vec<_>>>()?;
            let reference = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                refs,
                occupations.iter().map(|o| o / (2 * l + 1) as f64),
            ));
            channels.push(Channel {
                l,
                radius,
                projectors,
                ae: ae_waves,
                ps: dual_ps,
                coupling: &transform * coupling * transform.transpose(),
                overlap: &transform * overlap * transform.transpose(),
                reference: if raw {
                    b.transpose() * reference * b
                } else {
                    reference
                },
                xc: DMatrix::zeros(refs, refs),
                core_exchange: DMatrix::zeros(refs, refs),
            });
            transforms.push(transform);
        }
        let functional = v["functional"]
            .as_str()
            .ok_or_else(|| invalid("missing reference functional"))?;
        let relativistic = v["relativistic_exchange"]
            .as_bool()
            .ok_or_else(|| invalid("missing reference exchange convention"))?;
        let ae_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r.as_slice(),
            &all_density,
            functional,
            relativistic,
        )?;
        let ps_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r.as_slice(),
            &valence
                .as_slice()
                .iter()
                .zip(xc_core.as_slice())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
            functional,
            relativistic,
        )?;
        let bare_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r.as_slice(),
            valence.as_slice(),
            functional,
            relativistic,
        )?;
        let valence_potential = |potential: &[f64]| -> Result<f64> {
            integrate(
                r.as_slice(),
                &(0..n)
                    .map(|k| {
                        4. * PI * r.as_slice()[k].powi(2) * valence.as_slice()[k] * potential[k]
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let core_offset =
            bare_xc.energy - valence_potential(bare_xc.potential.as_slice())? - ps_xc.energy
                + valence_potential(ps_xc.potential.as_slice())?;
        let dft_constant = number(&v["one_center_energy"])? + core_offset;
        let xc_core = Cubic::new(ctx, ctx.copy_slice(r.as_slice())?, xc_core)?;
        let xc_core_tau = crate::paw::fields::core_kinetic(ctx, &xc_core)?;
        let local_radius = number(&v["local_radius"])?;
        let sphere = channels
            .iter()
            .map(|c| c.radius)
            .fold(local_radius, f64::max);
        let inside = (r.as_slice().partition_point(|x| *x < sphere) + 1).min(n);
        for (ch, transform) in channels.iter_mut().zip(transforms) {
            let refs = ch.ae.len();
            let mut vx = DMatrix::zeros(refs, refs);
            for i in 0..refs {
                for j in 0..refs {
                    let values: Vec<_> = (0..inside)
                        .map(|k| {
                            (ae_xc.potential.as_slice()[k] * ch.ae[i][k] * ch.ae[j][k]
                                - ps_xc.potential.as_slice()[k] * ch.ps[i][k] * ch.ps[j][k])
                                * r.as_slice()[k].powi(2)
                        })
                        .collect();
                    vx[(i, j)] = integrate(&r.as_slice()[..inside], &values)?;
                }
            }
            ch.xc = &transform * vx * transform.transpose();
            if closed_1s {
                if core_extent < 4 {
                    return Err(invalid("invalid core radial extent"));
                }
                let mut exchange = DMatrix::zeros(refs, refs);
                for i in 0..refs {
                    for j in 0..refs {
                        let left: Vec<_> = (0..core_extent)
                            .map(|k| ch.ae[i][k] * core_orbital[k])
                            .collect();
                        let right: Vec<_> = (0..core_extent)
                            .map(|k| ch.ae[j][k] * core_orbital[k])
                            .collect();
                        exchange[(i, j)] =
                            -0.5 * (super::onecenter::coulomb(
                                &r.as_slice()[..core_extent],
                                &left,
                                &right,
                                ch.l,
                            ) + super::onecenter::coulomb(
                                &r.as_slice()[..core_extent],
                                &right,
                                &left,
                                ch.l,
                            )) / (2 * ch.l + 1) as f64;
                    }
                }
                ch.core_exchange = &transform * exchange * transform.transpose();
            }
            let mix = |waves: &Vec<Vec<f64>>| {
                (0..refs)
                    .map(|i| {
                        (0..n)
                            .map(|k| (0..refs).map(|j| transform[(i, j)] * waves[j][k]).sum())
                            .collect()
                    })
                    .collect()
            };
            ch.ae = mix(&ch.ae);
            ch.ps = mix(&ch.ps);
        }
        let compensation_radius = number(&v["compensation_radius"])?;
        if charge <= 0. || local_radius <= 0. || compensation_radius <= 0. {
            return Err(invalid("PAW charges and radii must be positive"));
        }
        Ok(Self {
            atomic_number,
            charge,
            local_radius,
            compensation_radius,
            ionic: Cubic::new(ctx, r, ionic)?,
            channels,
            xc_core,
            xc_core_tau,
            dft_constant,
            relativistic_exchange: relativistic,
            _memory: memory,
        })
    }
}
