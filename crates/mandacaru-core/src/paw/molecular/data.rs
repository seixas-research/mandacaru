//! Typed radial tables decoded by the Python file-format adapter.
use super::*;
use crate::math::{
    radial::{integrate, validate},
    spline::Cubic,
};
use crate::paw::core_exchange::{FrozenCore, StoredShell};
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
    pub weights: Option<Vec<f64>>,
}
pub struct Species {
    pub atomic_number: u8,
    pub charge: f64,
    pub local_radius: f64,
    pub compensation_radius: f64,
    /// The radius where the partial core joins the all-electron core, when it is applied (a
    /// payload of tables carries none).
    pub nlcc_radius: Option<f64>,
    pub ionic: Cubic,
    pub channels: Vec<Channel>,
    pub xc_core: Cubic,
    pub xc_core_tau: Cubic,
    pub dft_constant: f64,
    pub relativistic_exchange: bool,
    /// Why this species' frozen core has no Hartree-Fock exchange operator (its record carries no
    /// core states, or a subshell is partially filled, or the states do not reproduce the stored
    /// core density), naming the element; `None` for an empty core and for a core whose
    /// core-valence exchange is in `Channel::core_exchange`. Kohn-Sham never reads it.
    pub core_refusal: Option<String>,
    /// Whether a screened hybrid is available: the core is empty or a single closed 1s shell, the
    /// only cores whose hybrid the molecule has been validated for (the screened one-center
    /// terms carry no core-valence exchange of their own).
    pub hybrid_core_supported: bool,
    /// The spin-orbit channels of a fully relativistic dataset (empty unless the molecule was
    /// built with spin-orbit allowed): read as the crystal reads them
    /// (`periodic::response::setup::spin_orbit`).
    pub spin_orbit: Vec<SpinOrbitChannel>,
    /// The reference atom's spherical densities on the knots of `ionic` that its linearized
    /// one-center exchange-correlation (`Channel::xc`) is taken at.
    pub xc_reference: XcReference,
    _memory: crate::Reservation,
}
/// The densities, per volume, at which a species' one-center exchange-correlation is linearized:
/// the all-electron reference (valence partial waves and core) and the smooth valence density
/// with the smooth core, and the knots of the sphere the difference is integrated over.
pub struct XcReference {
    pub all_electron: Vec<f64>,
    pub smooth: Vec<f64>,
    pub inside: usize,
}
/// One `l` of a species' spin-orbit term: its projectors' radial tables, the sphere they live in
/// (`max(r_cut of the scalar channel, r_cut_local)`) and the radial coupling `D^SO`.
pub struct SpinOrbitChannel {
    pub l: usize,
    pub radius: f64,
    pub radials: Vec<Cubic>,
    pub coupling: crate::models::spin_orbit::Coupling,
}
/// What a species' frozen core allows: the core of its stored states (the exchange with every
/// channel's partial waves, [`FrozenCore::exchange`]) or the reason it has none, and whether
/// the core is one a screened hybrid supports.
struct CoreSupport {
    frozen: Option<FrozenCore>,
    refusal: Option<String>,
    hybrid: bool,
}

/// The [`CoreSupport`] of a species of `atomic_number` and valence `charge` whose stored core
/// `density` is `core_charge` electrons on `r`, from its stored `states` (`None` for a record
/// that carries none).
fn core_support(
    symbol: &str,
    atomic_number: u8,
    charge: f64,
    (r, density): (&[f64], &[f64]),
    core_charge: f64,
    states: Option<&[StoredShell<'_>]>,
) -> CoreSupport {
    if core_charge <= 1e-6 {
        return CoreSupport {
            frozen: None,
            refusal: None,
            hybrid: true,
        };
    }
    let hybrid = states.is_some_and(|s| {
        s.len() == 1 && (s[0].n, s[0].l) == (1, 0) && (s[0].occupation - 2.).abs() < 1e-9
    });
    let electrons = atomic_number as f64 - charge;
    match states.map(|s| FrozenCore::from_states(symbol, r, density, electrons, s)) {
        Some(Ok(core)) => CoreSupport {
            frozen: Some(core),
            refusal: None,
            hybrid,
        },
        Some(Err(reason)) => CoreSupport {
            frozen: None,
            refusal: Some(reason),
            hybrid,
        },
        None => CoreSupport {
            frozen: None,
            refusal: Some(format!(
                "{symbol}: the dataset carries no core states, which the core-valence exchange \
                 of Hartree-Fock needs"
            )),
            hybrid,
        },
    }
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
        Self::parse_with_options(ctx, v, arrays, raw, false)
    }

    pub fn from_dataset(
        ctx: &Context,
        ds: &crate::paw::record::PawDataset,
        raw: bool,
        allow_spin_orbit: bool,
    ) -> Result<Self> {
        if !allow_spin_orbit && !ds.spin_orbit.is_empty() {
            return Err(invalid(
                "native molecular PAW spin-orbit projector terms are not yet implemented",
            ));
        }
        let atomic_number = u8::try_from(ds.atomic_number)
            .ok()
            .filter(|z| (1..=118).contains(z))
            .ok_or_else(|| invalid("invalid PAW atomic number"))?;
        let charge = ds.valence_charge;
        let r_slice = ds.r.as_slice();
        let ionic_slice = ds.v_local.as_slice();
        validate(r_slice, ionic_slice)?;
        let core_slice = ds.core_density.as_slice();
        validate(r_slice, core_slice)?;
        let core_charge = integrate(
            r_slice,
            &r_slice
                .iter()
                .zip(core_slice)
                .map(|(r, n)| 4. * PI * r * r * n)
                .collect::<Vec<_>>(),
        )?;
        if core_slice.iter().any(|v| *v < 0.) {
            return Err(invalid("core density must be nonnegative"));
        }
        let stored: Option<Vec<StoredShell<'_>>> = ds.core_states.as_ref().map(|states| {
            states
                .iter()
                .map(|s| StoredShell {
                    n: s.n,
                    l: s.l,
                    occupation: s.occupation,
                    orbital: s.orbital.as_slice(),
                })
                .collect()
        });
        let _core_memory = ctx.reserve(product(&[
            r_slice.len(),
            stored.as_ref().map_or(0, Vec::len) + 3,
            8,
        ])?)?;
        let support = core_support(
            &ds.symbol,
            atomic_number,
            charge,
            (r_slice, core_slice),
            core_charge,
            stored.as_deref(),
        );
        if ds.nlcc.applied
            && ds.nlcc.source.as_deref() != Some("smooth_core")
            && ds.nlcc.r_nlcc.is_some()
        {
            return Err(invalid(
                "native PAW reference XC currently reads the dataset's stored smooth-core correction",
            ));
        }
        let xc_core_vec: Vec<f64>;
        let xc_core_slice: &[f64] = if ds.nlcc.applied {
            ds.smooth_core_density.as_slice()
        } else {
            xc_core_vec = vec![0.0; r_slice.len()];
            &xc_core_vec
        };
        validate(r_slice, xc_core_slice)?;
        validate(r_slice, ds.valence_density.as_slice())?;

        let total_refs: usize = ds.channels.values().map(|c| c.occupations.len()).sum();
        let memory = ctx.reserve(
            product(&[
                r_slice.len(),
                total_refs.checked_add(4).ok_or(crate::Error::Overflow)?,
                8,
                8,
            ])?
            .checked_add(product(&[total_refs, total_refs, 8, 12])?)
            .ok_or(crate::Error::Overflow)?,
        )?;

        let mut all_density = core_slice.to_vec();
        let mut channels = Vec::new();
        let n = r_slice.len();
        let mut transforms = Vec::new();

        for (&l_u32, ch) in &ds.channels {
            let l = l_u32 as usize;
            if l >= 8 {
                continue;
            }
            let principal_n = ch.n as usize;
            let radius = ch.r_cut;
            if principal_n <= l || radius <= 0. {
                return Err(invalid("invalid PAW angular momentum or radius"));
            }
            let occupations = ch.occupations.data();
            let refs = occupations.len();
            if refs == 0 {
                return Err(invalid("a PAW channel needs reference waves"));
            }
            if ch.ae_waves.len() != refs {
                return Err(invalid("PAW wave table dimensions disagree"));
            }
            let proj_buffers = if raw {
                &ch.raw_projectors
            } else {
                &ch.projectors
            };
            if proj_buffers.len() != refs {
                return Err(invalid("PAW wave table dimensions disagree"));
            }
            for i in 0..refs {
                if ch.ae_waves[i].as_slice().len() != n || proj_buffers[i].as_slice().len() != n {
                    return Err(invalid("PAW wave table dimensions disagree"));
                }
            }
            let n_sq = product(&[refs, refs])?;
            if ch.coupling.len() != n_sq
                || ch.overlap_correction.len() != n_sq
                || ch.vanderbilt.len() != n_sq
            {
                return Err(invalid("PAW channel matrix dimensions disagree"));
            }
            let coupling = DMatrix::from_row_slice(refs, refs, ch.coupling.data());
            let overlap = DMatrix::from_row_slice(refs, refs, ch.overlap_correction.data());
            let b = DMatrix::from_row_slice(refs, refs, ch.vanderbilt.data());
            let transform = if raw {
                b.clone()
                    .try_inverse()
                    .ok_or_else(|| invalid("singular PAW projector transform"))?
            } else {
                DMatrix::identity(refs, refs)
            };
            let ae_waves: Vec<Vec<f64>> = (0..refs)
                .map(|i| ch.ae_waves[i].as_slice().to_vec())
                .collect();
            let mut dual_ps = vec![vec![0.; n]; refs];
            for i in 0..refs {
                let q = ch.wavevectors[i].data();
                let c = ch.wave_coefficients[i].data();
                if q.len() != c.len() {
                    return Err(invalid("PAW Bessel expansion dimensions disagree"));
                }
                let j = crate::kernels::bessel(ctx, l as i32, r_slice, q)?;
                for k in 0..n {
                    dual_ps[i][k] = if r_slice[k] <= radius {
                        (0..c.len())
                            .map(|z| c[z] * j.as_slice()[k * c.len() + z])
                            .sum()
                    } else {
                        ae_waves[i][k]
                    };
                    all_density[k] += occupations[i] * ae_waves[i][k].powi(2) / (4. * PI);
                }
            }
            let projectors = (0..refs)
                .map(|i| {
                    Cubic::new(
                        ctx,
                        ctx.copy_slice(r_slice)?,
                        ctx.copy_slice(proj_buffers[i].as_slice())?,
                    )
                    .map(Arc::new)
                })
                .collect::<Result<Vec<_>>>()?;
            let reference = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                refs,
                occupations.iter().map(|o| o / (2 * l + 1) as f64),
            ));
            let shell_weights = crate::paw::shell::channel_weights(
                ctx,
                &ae_waves,
                b.as_slice(),
                radius,
                r_slice,
                occupations,
                if raw {
                    crate::paw::onecenter::Basis::Raw
                } else {
                    crate::paw::onecenter::Basis::Dual
                },
            )
            .ok();
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
                weights: shell_weights,
            });
            transforms.push(transform);
        }

        let ae_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r_slice,
            &all_density,
            &ds.xc,
            ds.relativistic_exchange,
        )?;
        let smooth_density = ds
            .valence_density
            .as_slice()
            .iter()
            .zip(xc_core_slice)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let ps_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r_slice,
            &smooth_density,
            &ds.xc,
            ds.relativistic_exchange,
        )?;
        let bare_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r_slice,
            ds.valence_density.as_slice(),
            &ds.xc,
            ds.relativistic_exchange,
        )?;
        let valence_potential = |potential: &[f64]| -> Result<f64> {
            integrate(
                r_slice,
                &(0..n)
                    .map(|k| {
                        4. * PI
                            * r_slice[k].powi(2)
                            * ds.valence_density.as_slice()[k]
                            * potential[k]
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let core_offset =
            bare_xc.energy - valence_potential(bare_xc.potential.as_slice())? - ps_xc.energy
                + valence_potential(ps_xc.potential.as_slice())?;
        let dft_constant = ds.one_center_energy + core_offset;
        let xc_core = Cubic::new(
            ctx,
            ctx.copy_slice(r_slice)?,
            ctx.copy_slice(xc_core_slice)?,
        )?;
        let xc_core_tau = crate::paw::fields::core_kinetic(ctx, &xc_core)?;
        let local_radius = ds.r_cut_local;
        let sphere = channels
            .iter()
            .map(|c| c.radius)
            .fold(local_radius, f64::max);
        let inside = (r_slice.partition_point(|x| *x < sphere) + 1).min(n);
        for (ch, transform) in channels.iter_mut().zip(transforms) {
            let refs = ch.ae.len();
            let mut vx = DMatrix::zeros(refs, refs);
            for i in 0..refs {
                for j in 0..refs {
                    let values: Vec<_> = (0..inside)
                        .map(|k| {
                            (ae_xc.potential.as_slice()[k] * ch.ae[i][k] * ch.ae[j][k]
                                - ps_xc.potential.as_slice()[k] * ch.ps[i][k] * ch.ps[j][k])
                                * r_slice[k].powi(2)
                        })
                        .collect();
                    vx[(i, j)] = integrate(&r_slice[..inside], &values)?;
                }
            }
            ch.xc = &transform * vx * transform.transpose();
            if let Some(core) = &support.frozen {
                let exchange = core.exchange(r_slice, ch.l, &ch.ae)?;
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
        let compensation_radius = ds.compensation_radius;
        if charge <= 0. || local_radius <= 0. || compensation_radius <= 0. {
            return Err(invalid("PAW charges and radii must be positive"));
        }
        let mut spin_orbit = Vec::new();
        if allow_spin_orbit {
            for (&l, entry) in &ds.spin_orbit {
                let scalar = ds.channels.get(&l).ok_or_else(|| {
                    invalid("a spin-orbit channel of the dataset has no scalar channel")
                })?;
                let radials = entry
                    .projectors
                    .iter()
                    .map(|t| {
                        Cubic::new(ctx, ctx.copy_slice(r_slice)?, ctx.copy_slice(t.as_slice())?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                spin_orbit.push(SpinOrbitChannel {
                    l: l as usize,
                    radius: scalar.r_cut.max(ds.r_cut_local),
                    radials,
                    coupling: crate::models::spin_orbit::Coupling {
                        radial: entry.coupling.shape()[0],
                        values: entry.coupling.data().to_vec(),
                    },
                });
            }
        }
        Ok(Self {
            atomic_number,
            charge,
            local_radius,
            compensation_radius,
            nlcc_radius: ds.nlcc.r_nlcc.filter(|_| ds.nlcc.applied),
            ionic: Cubic::new(ctx, ctx.copy_slice(r_slice)?, ctx.copy_slice(ionic_slice)?)?,
            channels,
            xc_core,
            xc_core_tau,
            dft_constant,
            relativistic_exchange: ds.relativistic_exchange,
            core_refusal: support.refusal,
            hybrid_core_supported: support.hybrid,
            spin_orbit,
            xc_reference: XcReference {
                all_electron: all_density,
                smooth: smooth_density,
                inside,
            },
            _memory: memory,
        })
    }

    pub fn parse_with_options(
        ctx: &Context,
        v: &Value,
        arrays: &mut Arrays,
        raw: bool,
        allow_spin_orbit: bool,
    ) -> Result<Self> {
        if !allow_spin_orbit && v["spin_orbit"].as_bool() != Some(false) {
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
        if core.as_slice().iter().any(|v| *v < 0.) {
            return Err(invalid("core density must be nonnegative"));
        }
        // The core states of a species given as tables, as a record stores them (`n`, `l`,
        // `occupation` and the table of `u = r R`); a species without the key has none.
        let states = match v.get("core_states") {
            None | Some(Value::Null) => None,
            Some(list) => Some(
                list.as_array()
                    .ok_or_else(|| invalid("PAW core states must be a list"))?
                    .iter()
                    .map(|e| {
                        Ok((
                            u32::try_from(count(&e["n"])?)
                                .map_err(|_| invalid("invalid core state n"))?,
                            u32::try_from(count(&e["l"])?)
                                .map_err(|_| invalid("invalid core state l"))?,
                            number(&e["occupation"])?,
                            table(arrays, &e["orbital"])?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        let stored: Option<Vec<StoredShell<'_>>> = states.as_ref().map(|states| {
            states
                .iter()
                .map(|(n, l, occupation, orbital)| StoredShell {
                    n: *n,
                    l: *l,
                    occupation: *occupation,
                    orbital: orbital.as_slice(),
                })
                .collect()
        });
        let _core_memory = ctx.reserve(product(&[
            r.as_slice().len(),
            stored.as_ref().map_or(0, Vec::len) + 3,
            8,
        ])?)?;
        let support = core_support(
            &format!("Z = {atomic_number}"),
            atomic_number,
            charge,
            (r.as_slice(), core.as_slice()),
            core_charge,
            stored.as_deref(),
        );
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
                total_refs.checked_add(4).ok_or(crate::Error::Overflow)?,
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
            let shell_weights = crate::paw::shell::channel_weights(
                ctx,
                &ae_waves,
                b.as_slice(),
                radius,
                r.as_slice(),
                &occupations,
                if raw {
                    crate::paw::onecenter::Basis::Raw
                } else {
                    crate::paw::onecenter::Basis::Dual
                },
            )
            .ok();
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
                weights: shell_weights,
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
        let smooth_density = valence
            .as_slice()
            .iter()
            .zip(xc_core.as_slice())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let ps_xc = crate::models::spherical::exchange_correlation(
            ctx,
            r.as_slice(),
            &smooth_density,
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
            if let Some(core) = &support.frozen {
                let exchange = core.exchange(r.as_slice(), ch.l, &ch.ae)?;
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
            nlcc_radius: None,
            ionic: Cubic::new(ctx, r, ionic)?,
            channels,
            xc_core,
            xc_core_tau,
            dft_constant,
            relativistic_exchange: relativistic,
            core_refusal: support.refusal,
            hybrid_core_supported: support.hybrid,
            spin_orbit: Vec::new(),
            xc_reference: XcReference {
                all_electron: all_density,
                smooth: smooth_density,
                inside,
            },
            _memory: memory,
        })
    }
}
