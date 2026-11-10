//! The SCF state a crystal force reads: the public API over `solver.rs` that the engine and the
//! strain differencing need.
//!
//! * [`state_energy`]: band + electrostatic + xc + constant at FIXED density matrices (the energy
//!   of `PeriodicResult::matrices`, without the entropy), the quantity CF-5 differences.
//! * [`output_fields`]: the potentials, channel potentials and reciprocal weights of the OUTPUT
//!   density of a set of density matrices (the traps of `solve()` are the engine's to close).
//! * [`rediagonalize`]: one more diagonalization with those fields, the Fermi level over the whole
//!   mesh and the occupations of the Mermin free energy, as the engine's [`Bands`]. A hybrid crystal
//!   is diagonalized with `-fraction K[D_out]` ([`exchange_operators`]): the eigenvalues, hence the
//!   energy-weighted density `W`, are those of the hybrid Hamiltonian.
//! * [`hybrid_state_energy`]: the same energy for a hybrid crystal, with its exact exchange.
//! * [`refuse`]: the crystals this engine does not differentiate.
//!
//! Eigenvalues are returned shifted by `Crystal::reference`, as `solve()` returns them; the
//! engine removes it again through `Layout::reference`. Occupations are per spin channel
//! (`0..=1`); the solver's own occupations carry the degeneracy.
use super::{Bands, Matrix, Request};
use crate::{
    math::{fft::Fft3, storage::StoredMatrix},
    models,
    periodic::{
        invalid,
        screened_exchange::ExxDivTreatment,
        solver::{self, Options},
        spectrum, Crystal,
    },
    quantum::C64,
    Buffer, Error, Result,
};

/// What this engine cannot differentiate, from the crystal itself: float32 data,
/// DFT+U, a hybrid without the PAW spheres its projections were built from, and a mesh reduced by
/// space-group symmetry whose operations the snapshot does not carry ([`reduced_mesh`]; with them
/// the engine averages the k-sum over the group, CF-6). A collinear
/// spin-polarized crystal (`spins = 2`) and a spinor, magnetic or not, hybrid or not, are
/// differentiated (the one-center operator the model returns carries the non-collinear remainder).
pub fn refuse(c: &Crystal) -> Result<()> {
    super::check_request(&request(c))?;
    let no = |what: &str| {
        Err(Error::Unsupported(format!(
            "crystal forces are not available for {what}"
        )))
    };
    if c.hubbard.as_ref().is_some_and(|h| h.is_active()) {
        if c.screened_exchange()?.is_some() {
            return no("a DFT+U hybrid crystal");
        }
        if c.spinor {
            return no("a DFT+U spinor crystal");
        }
    }
    if c.screened_exchange()?.is_some() && c.hybrid.is_none() {
        return no("a hybrid crystal without the PAW spheres of its sites block");
    }
    if c.spins != 1 && c.spins != 2 {
        return no("a crystal with more than two spin channels");
    }
    let identity = c.qsym.as_slice().iter().enumerate().all(|(i, v)| {
        let want = if i / c.channels == i % c.channels {
            1.
        } else {
            0.
        };
        (v - C64::new(want, 0.)).norm() < 1e-12
    });
    if !c.permutations.is_empty() || !identity {
        reduced_mesh(c).map_err(|cause| {
            Error::Unsupported(format!("crystal forces are not available for {cause}"))
        })?;
    }
    Ok(())
}

/// Whether a mesh reduced by space-group operations can be differentiated: the snapshot must
/// carry the operations (its `setup` block) to average the k-sum over. A spin-orbit crystal is
/// not (the moment would need the spin rotation), nor is a hybrid (it needs the whole mesh).
/// The cause names what to do.
pub fn reduced_mesh(c: &Crystal) -> std::result::Result<(), String> {
    if c.spinor {
        return Err(
            "a spin-orbit crystal on a k mesh reduced by space-group symmetry (the \
                    magnetization would need the spin rotation of every operation; use \
                    symmetry=False)"
                .into(),
        );
    }
    if c.screened_exchange().map_err(|e| e.to_string())?.is_some() {
        return Err(
            "a hybrid crystal on a k mesh reduced by space-group symmetry (a hybrid \
                    needs the whole mesh, which the Mandacaru setup keeps for it)"
                .into(),
        );
    }
    if let Some(note) = &c.description_unavailable {
        return Err(format!(
            "a k mesh reduced by space-group symmetry whose symmetry operations are unavailable ({note})"
        ));
    }
    match c.description.as_ref() {
        None => Err(
            "a k mesh reduced by space-group symmetry in a snapshot without the \
                     'setup' block (its symmetry operations): rebuild the crystal with this \
                     version, or use symmetry=False"
                .into(),
        ),
        Some(d) if d.point_group.is_none() => Err("a k mesh reduced by space-group symmetry \
                     whose 'setup' block lists no operations"
            .into()),
        Some(_) => Ok(()),
    }
}

/// The request the engine refuses by: what the crystal is.
pub fn request(c: &Crystal) -> Request {
    Request {
        functional: c.functional.clone(),
        spinor: c.spinor,
        single_precision: c.precision.name() != "float64",
    }
}

/// `E = band + electrostatic + xc + constant` (Hartree) of the density matrices `densities`
/// (`spin * nk + k`, `C f C^+` with the solver's occupations): `PeriodicResult::energy` for the
/// converged ones, with no entropy. Raises for a hybrid, a meta-GGA or a DFT+U crystal, whose
/// energies have more terms.
pub fn state_energy(c: &Crystal, densities: &[StoredMatrix]) -> Result<f64> {
    let definition = models::functional(&c.functional)?;
    if definition.screened_exchange.is_some() || definition.requires_tau {
        return Err(Error::Unsupported(
            "state_energy covers the semilocal crystal without exact exchange or meta-GGA".into(),
        ));
    }
    if densities.len() != c.spins * c.points.len() {
        return Err(invalid("one density matrix per spin and k-point"));
    }
    let (rho, q) = solver::density(c, densities)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    let terms = solver::energy(c, &mut fft, densities, rho.as_slice(), q.as_slice(), None)?;
    let hubbard_energy = match &c.hubbard {
        Some(h) if h.is_active() => {
            let n = h.occupations(c, densities)?;
            h.energy(c, &n)?
        }
        _ => 0.0,
    };
    Ok(terms.iter().sum::<f64>() + hubbard_energy)
}

/// `E = band + electrostatic + xc + exact exchange + constant` (Hartree) of a hybrid crystal, with
/// the divergence treatment of the SCF (`options.exxdiv_treatment`): the energy of the converged
/// `PeriodicResult::matrices`, with no entropy.
pub fn hybrid_state_energy(
    c: &Crystal,
    densities: &[StoredMatrix],
    exxdiv: ExxDivTreatment,
) -> Result<f64> {
    let definition = models::functional(&c.functional)?;
    if definition.screened_exchange.is_none()
        || definition.requires_tau
        || c.hubbard.as_ref().is_some_and(|h| h.is_active())
    {
        return Err(Error::Unsupported(
            "hybrid_state_energy covers the hybrid crystal without DFT+U".into(),
        ));
    }
    if densities.len() != c.spins * c.points.len() {
        return Err(invalid("one density matrix per spin and k-point"));
    }
    let (rho, q) = solver::density(c, densities)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    let terms = solver::energy(c, &mut fft, densities, rho.as_slice(), q.as_slice(), None)?;
    Ok(terms.iter().sum::<f64>() + c.hybrid_exchange(densities, exxdiv)?.energy)
}

/// The exact-exchange matrices `K_s(k)` (`spin * nk + k`) of the density matrices `densities`,
/// the operators `-fraction K` is subtracted with in the Hamiltonian; empty for a semilocal
/// crystal.
pub fn exchange_operators(
    c: &Crystal,
    densities: &[StoredMatrix],
    exxdiv: ExxDivTreatment,
) -> Result<Vec<Matrix>> {
    if c.screened_exchange()?.is_none() {
        return Ok(Vec::new());
    }
    Ok(c.hybrid_exchange(densities, exxdiv)?.operators)
}

/// The +U operators `O_s` (`p x p`, one per spin channel) of the density matrices `densities`;
/// empty when the crystal has no active +U shells.
pub fn hubbard_operators(c: &Crystal, densities: &[StoredMatrix]) -> Result<Vec<Matrix>> {
    match &c.hubbard {
        Some(h) if h.is_active() => {
            let n = h.occupations(c, densities)?;
            h.operators(c, &n)
        }
        _ => Ok(Vec::new()),
    }
}

/// The fields of an output density, on the mesh.
pub struct OutputFields {
    /// `v_ee + v_xc` the basis sees, one block of `n` per spin channel.
    pub local: Buffer<f64>,
    /// The channel potentials `m_a`.
    pub moments: Buffer<C64>,
    /// `v_xc` alone, spin-averaged (`(v_up + v_dn) / 2` of a polarized run): the potential the
    /// partial core is differentiated against. The core is split in halves between the spin
    /// densities, so `dE_xc / d core = 1/2 (v_up + v_dn)`; `v_up` alone would be wrong by the
    /// half difference of the two potentials.
    pub xc: Buffer<f64>,
    /// `K rho^* / V`, the reciprocal weights of the hats and ions.
    pub weights: Buffer<C64>,
    /// A meta-GGA's `v_tau = dE_xc / d tau`, one block of `n` per spin channel, evaluated on the
    /// output density's own `tau`; `None` for every other functional.
    pub kinetic: Option<Buffer<f64>>,
}

/// The potentials of the density of `densities` with the channel moments `q`.
pub fn output_fields(c: &Crystal, densities: &[StoredMatrix], q: &[C64]) -> Result<OutputFields> {
    let n = c.grid.index.points();
    let (rho, _) = solver::density(c, densities)?;
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;
    // A meta-GGA is evaluated on the output density's own kinetic density, as the SCF does,
    // in the local potential and in the xc terms alike.
    let tau = if crate::models::functional(&c.functional)?.requires_tau {
        Some(super::super::fields::kinetic_density(c, densities)?)
    } else {
        None
    };
    let potential = solver::local_potential(c, &mut fft, rho.as_slice(), q, tau.as_ref().map(|t| t.as_slice()))?;
    let terms = solver::exchange_correlation(c, &mut fft, rho.as_slice(), tau.as_ref().map(|t| t.as_slice()))?;
    // One block of `n` per field channel. For a spinor the first block is the charge channel,
    // `0.5 (v_up + v_dn)` of the collinear functional in the local eigenframe (a time-reversal
    // crystal has that block only): the potential the unpolarized partial core is differentiated
    // against. The magnetization blocks never reach it.
    if terms.potential.as_slice().len() != c.field_channels() * n {
        return Err(invalid("one xc potential block per field channel"));
    }
    let mut xc = c.context.allocate::<f64>(n)?;
    for (k, to) in xc.as_mut_slice().iter_mut().enumerate() {
        *to = (0..c.spins)
            .map(|s| terms.potential.as_slice()[s * n + k])
            .sum::<f64>()
            / c.spins as f64;
    }
    let mut total = c.context.allocate::<f64>(n)?;
    for (k, to) in total.as_mut_slice().iter_mut().enumerate() {
        *to = (0..c.spins).map(|s| rho.as_slice()[s * n + k]).sum();
    }
    let mut weights = solver::reciprocal(c, &mut fft, total.as_slice())?;
    let volume = n as f64 * c.grid.index.dv();
    for (w, kernel) in weights.as_mut_slice().iter_mut().zip(c.kernel.as_slice()) {
        *w = w.conj() * (kernel / volume);
    }
    Ok(OutputFields {
        local: potential.local,
        moments: potential.moments,
        xc,
        weights,
        kinetic: terms.kinetic_potential,
    })
}

/// Diagonalizes every `(spin, k)` with `fields`, finds the Fermi level over the whole mesh and
/// returns the bands with their Mermin occupations (`bands[spin][k]`, all `m` of them; `2m` for a
/// spinor, whose single block holds both spin components).
///
/// `exchange` is [`exchange_operators`] of the density the fields come from: one `m x m` matrix per
/// `(spin, k)` for a hybrid crystal (a hybrid refuses an empty list, never a silent semilocal
/// Hamiltonian), empty otherwise.
///
/// `hubbard` is [`hubbard_operators`] of the density the fields come from: one `p x p` matrix per
/// spin channel for a DFT+U crystal, empty otherwise.
///
/// The returned bands hold `spins * nk * (16 m^2 + 16 m)` bytes (coefficients, eigenvalues,
/// occupations) outside the `Context`'s buffers; the caller that keeps them reserves that size
/// from the `Context` first (`SetupDerivatives::fields` does, before this call).
pub fn rediagonalize(
    c: &Crystal,
    fields: &OutputFields,
    options: &Options,
    exchange: &[Matrix],
    hubbard: &[Matrix],
) -> Result<Vec<Vec<Bands>>> {
    let n = c.grid.index.points();
    let hybrid = c.screened_exchange()?.is_some();
    if exchange.len() != if hybrid { c.spins * c.points.len() } else { 0 } {
        return Err(invalid(
            "one exchange matrix per spin and k-point for a hybrid crystal, none otherwise",
        ));
    }
    let active_hubbard = c.hubbard.as_ref().is_some_and(|h| h.is_active());
    if hubbard.len() != if active_hubbard { c.spins } else { 0 } {
        return Err(invalid(
            "one Hubbard matrix per spin channel for a DFT+U crystal, none otherwise",
        ));
    }
    let mut eigenvalues = Vec::new();
    let mut coefficients = Vec::new();
    for s in 0..c.spins {
        for (k, data) in c.points.iter().enumerate() {
            // A spinor diagonalization takes every Pauli field channel at once, as `solve` does.
            let local = if c.spinor {
                fields.local.as_slice()
            } else {
                &fields.local.as_slice()[s * n..(s + 1) * n]
            };
            // A spinor's single spin block holds every Pauli channel of v_tau.
            let kinetic = fields.kinetic.as_ref().map(|v| {
                if c.spinor { v.as_slice() } else { &v.as_slice()[s * n..(s + 1) * n] }
            });
            let (eps, coefficient, _) = solver::diagonalize_with_exchange(
                c,
                data,
                local,
                fields.moments.as_slice(),
                kinetic,
                hubbard.get(s),
                exchange.get(s * c.points.len() + k),
            )?;
            eigenvalues.push(eps);
            coefficients.push(coefficient);
        }
    }
    let weights: Vec<f64> = (0..c.spins)
        .flat_map(|_| c.points.iter().map(|p| p.weight))
        .collect();
    let mu = spectrum::fermi(
        &eigenvalues,
        &weights,
        c.electrons,
        options.smearing,
        options.width,
        c.degeneracy(),
    )?;
    let mut flat = eigenvalues.into_iter().zip(coefficients);
    Ok((0..c.spins)
        .map(|_| {
            (0..c.points.len())
                .map(|_| {
                    let (eps, coefficients): (Vec<f64>, Matrix) =
                        flat.next().expect("one per (s, k)");
                    Bands {
                        occupations: eps
                            .iter()
                            .map(|e| options.smearing.occupation((e - mu) / options.width))
                            .collect(),
                        eigenvalues: eps.iter().map(|e| e + c.reference).collect(),
                        coefficients,
                    }
                })
                .collect()
        })
        .collect())
}

/// The density matrices `C (deg f) C^+` of engine bands, in the solver's layout and
/// normalization (`spin * nk + k`).
pub fn densities(c: &Crystal, bands: &[Vec<Bands>]) -> Result<Vec<StoredMatrix>> {
    if bands.len() != c.spins || bands.iter().any(|s| s.len() != c.points.len()) {
        return Err(invalid("one set of bands per spin and k-point"));
    }
    let degeneracy = C64::new(c.degeneracy(), 0.);
    bands
        .iter()
        .flatten()
        .map(|b| StoredMatrix::new(&c.context, b.density() * degeneracy, false))
        .collect()
}

impl super::Hubbard for Crystal {
    fn operators(&self, state: &super::State) -> Result<Vec<Matrix>> {
        let densities = densities(self, &state.bands)?;
        hubbard_operators(self, &densities)
    }
}

impl Crystal {
    /// The +U hook of this crystal, `None` when the crystal carries no active DFT+U.
    pub fn hubbard_hook(&self) -> Option<&dyn super::Hubbard> {
        if self.hubbard.as_ref().is_some_and(|h| h.is_active()) {
            Some(self)
        } else {
            None
        }
    }
}
