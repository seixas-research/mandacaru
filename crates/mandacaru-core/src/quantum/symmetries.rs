//! A quantum problem on a Z2-tapered register: the Hamiltonian, the pool, the reference and the
//! basis reduced together by one [`Taper`], and the state-vector route back to the fermionic
//! register that the density matrices read.
use super::{
    algebra::Generator,
    amplitude::Amplitude,
    pools::{self, label_conserves_momentum, CrystalMomentum},
    properties::{self, Rdms, SpatialRdms},
    solver::QuantumProblem,
    tapering::Taper,
    Basis, Encoding, Operator, C64,
};
use crate::{product, Buffer, Error, Result};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

enum ResolvedFermion<'a, A: Amplitude> {
    Wide(Basis, Buffer<C64>),
    Local(Buffer<A>),
    Borrowed(&'a [A]),
}

impl QuantumProblem {
    /// Filter `(pool, pool_screenable, growth_members, _pool_metadata)` via `map` (`Ok(None)`
    /// drops the generator, `Ok(Some(op))` keeps and transforms it), remapping CEO members
    /// consistently without leaving dangling indices.
    fn retain_and_map_pool(
        &mut self,
        mut map: impl FnMut(&Generator) -> Result<Option<Operator>>,
    ) -> Result<Vec<(usize, String)>> {
        let ctx = &self.context;
        let mut renumber = vec![None; self.pool.len()];
        let mut pool = Vec::with_capacity(self.pool.len());
        let mut screenable = Vec::with_capacity(self.pool.len());
        let mut dropped = Vec::new();
        for (index, generator) in self.pool.iter().enumerate() {
            let Some(operator) = map(generator)? else {
                dropped.push((index, generator.label.clone()));
                continue;
            };
            renumber[index] = Some(pool.len());
            pool.push(Generator {
                label: generator.label.clone(),
                operator,
            });
            screenable.push(self.pool_screenable[index]);
        }
        let mut growth_members = vec![Vec::new(); pool.len()];
        for (index, members) in self.growth_members.iter().enumerate() {
            if let Some(new) = renumber[index] {
                growth_members[new] = members.iter().filter_map(|&m| renumber[m]).collect();
            }
        }
        let memory = ctx.reserve(product(&[pool.len(), 256])?)?;
        self.pool = pool;
        self.pool_screenable = screenable;
        self.growth_members = growth_members;
        self._pool_metadata = memory;
        Ok(dropped)
    }

    /// Construct a derived problem with its operator pool restricted to crystal-momentum
    /// conserving generators (`Bloch-*` domain extension for all variational solvers and subspace
    /// methods).
    pub fn with_momentum(&self, momentum: &CrystalMomentum) -> Result<Self> {
        if self.taper.is_some() {
            return Err(invalid(
                "apply momentum pool filtering before Z2 tapering the register",
            ));
        }
        if self.hva.is_some() {
            return Err(invalid(
                "momentum pool restriction applies to excitation/Pauli pools, not HVA",
            ));
        }
        let ctx = &self.context;
        let encoding = Encoding::new(
            &self.encoding.mapping,
            self.encoding.modes,
            self.encoding.particles,
        )?;
        let sector = self
            .basis
            .as_ref()
            .is_some_and(|b| b.len() < (1usize << b.n));
        let mut out = if self.is_register_free() {
            Self::new_in(
                ctx,
                encoding,
                self.hamiltonian.copy_into(ctx)?,
                super::Space::Pauli {
                    number_conserving: !self.conserved.is_empty(),
                },
                false,
                false,
            )?
        } else {
            Self::new(
                ctx,
                encoding,
                self.hamiltonian.copy_into(ctx)?,
                sector,
                self.dense.is_some(),
                false,
            )?
        };
        let mut copied_pool = Vec::with_capacity(self.pool.len());
        for g in &self.pool {
            copied_pool.push(Generator {
                label: g.label.clone(),
                operator: g.operator.copy_into(ctx)?,
            });
        }
        out.pool = copied_pool;
        out.pool_screenable = self.pool_screenable.clone();
        out.growth_members = self.growth_members.clone();
        out.pool_name = self.pool_name.clone();
        out.precision = self.precision;
        out._pool_metadata = ctx.reserve(product(&[out.pool.len(), 256])?)?;
        out.restrict_momentum(momentum)?;
        Ok(out)
    }

    /// Filter the problem's operator pool to generators conserving crystal momentum `momentum`,
    /// decoupling periodic Bloch symmetry once so VQE, ADAPT-VQE, VASQA, VALSQA, and all subspace
    /// variants automatically search/screen only momentum-allowed excitations.
    pub fn restrict_momentum(
        &mut self,
        momentum: &CrystalMomentum,
    ) -> Result<Vec<(usize, String)>> {
        if self.hva.is_some() {
            return Err(invalid(
                "momentum pool restriction applies to excitation/Pauli pools, not HVA",
            ));
        }
        if self.encoding.modes % 2 != 0 || self.encoding.modes / 2 != momentum.orbital_k.len() {
            return Err(invalid(
                "crystal momentum orbital count must match the problem's spatial orbitals",
            ));
        }
        let m = momentum.orbital_k.len();
        let basis_k: Option<Vec<[i64; 3]>> = if self.taper.is_none() {
            self.basis.as_ref().map(|b| {
                b.indices
                    .as_slice()
                    .iter()
                    .map(|&enc| {
                        let occ = self.encoding.decode(enc);
                        let mut sum = [0i64; 3];
                        for bit in 0..self.encoding.modes {
                            if (occ >> (self.encoding.modes - 1 - bit)) & 1 != 0 {
                                let k = momentum.orbital_k[bit % m];
                                for (s, &v) in sum.iter_mut().zip(&k) {
                                    *s += v;
                                }
                            }
                        }
                        sum
                    })
                    .collect()
            })
        } else {
            None
        };
        let ctx = self.context.clone();
        let keep_flags: Vec<bool> = {
            let basis_ref = self.basis.as_ref().zip(basis_k.as_ref());
            self.pool
                .iter()
                .map(
                    |generator| match label_conserves_momentum(&generator.label, momentum) {
                        Some(flag) => flag,
                        None => {
                            if let Some((basis, k_rows)) = basis_ref {
                                k_rows.iter().enumerate().all(|(row, &k_in)| {
                                    generator
                                        .operator
                                        .apply_sparse_state(basis, &[(row, C64::new(1., 0.))])
                                        .into_iter()
                                        .filter(|&(_, amp)| amp.norm() > 1e-12)
                                        .all(|(out_row, _)| {
                                            let k_out = k_rows[out_row];
                                            (0..3).all(|a| {
                                                (k_out[a] - k_in[a])
                                                    .rem_euclid(momentum.mesh[a] as i64)
                                                    == 0
                                            })
                                        })
                                })
                            } else {
                                false
                            }
                        }
                    },
                )
                .collect()
        };
        let mut idx = 0usize;
        self.retain_and_map_pool(|generator| {
            let keep = keep_flags[idx];
            idx += 1;
            if keep {
                Ok(Some(generator.operator.copy_into(&ctx)?))
            } else {
                Ok(None)
            }
        })
    }

    /// Remove the Z2 symmetries of the encoded Hamiltonian from the register, in the sector of the
    /// reference determinant. Call it after the final pool is chosen: the symmetries depend on the
    /// mapping, so they are found on the Hamiltonian as the encoding wrote it.
    ///
    /// * The Hamiltonian is tapered; the reference is `Taper::bits` of the encoded one.
    /// * A pool generator is tapered with `Taper::operator`, which refuses a leaking term.
    ///   A generator that does not commute with the symmetries connects different sectors, so it
    ///   cannot contribute in the sector the reference fixes: it is dropped (and listed in
    ///   `dropped`), which is exact there. The pool metadata follows the renumbering.
    /// * The particle-number sector does not apply (the Clifford does not conserve particle number
    ///   in the tapered computational basis): the basis is the whole tapered register.
    ///
    /// Leaves the problem as it was when the Hamiltonian has no symmetry beyond the identity
    /// (`taper` stays `None`). A register-based pool, an HVA ansatz, a built sparse matrix and a
    /// second tapering are refused.
    pub fn taper(&mut self) -> Result<()> {
        if self.taper.is_some() {
            return Err(invalid("the register is already tapered"));
        }
        if pools::needs_untapered_register(&self.pool_name) {
            return Err(pools::untapered_register(&self.pool_name));
        }
        if self.factorized.is_some() {
            return Err(invalid(
                "the factorized measurement reads the untapered Jordan-Wigner register and cannot be tapered",
            ));
        }
        if self.hva.is_some() {
            return Err(invalid(
                "the Hamiltonian-variational ansatz is built from the untapered Hamiltonian groups and cannot be tapered",
            ));
        }
        if self.sparse.is_some() {
            return Err(invalid(
                "taper the register before building its sparse matrix",
            ));
        }
        let ctx = self.context.clone();
        let Some(taper) = Taper::find(&self.hamiltonian, self.encoding.reference()) else {
            return Ok(());
        };
        // Everything fallible first: a failure leaves the problem untouched.
        let hamiltonian = taper.operator(&ctx, &self.hamiltonian)?;
        let taper_ref = &taper;
        let ctx_ref = &ctx;
        let dropped = self.retain_and_map_pool(|generator| {
            if generator
                .operator
                .terms
                .iter()
                .any(|t| taper_ref.leaks(t.x))
            {
                Ok(None)
            } else {
                Ok(Some(taper_ref.operator(ctx_ref, &generator.operator)?))
            }
        })?;
        // The whole tapered register is a state-vector object: a register-free problem stays
        // register-free (no `2^(n-k)` basis), and the generators it kept are algebraically
        // exact in the reference's sector by construction (`Taper::operator` refuses a leak).
        let basis = if self.is_register_free() {
            None
        } else {
            Some(Basis::register_at(
                &ctx,
                taper.width(),
                taper.bits(self.encoding.reference()),
            )?)
        };
        let dense = match (&basis, self.dense.is_some()) {
            (Some(basis), true) => Some(hamiltonian.dense(&ctx, basis)?),
            _ => None,
        };
        self.hamiltonian = hamiltonian;
        self.basis = basis;
        self.conserved = Vec::new();
        self.dense = dense;
        self.dropped = dropped;
        self.taper = Some(taper);
        Ok(())
    }

    fn resolve_fermionic_state<'a, A: Amplitude>(
        &self,
        purpose: &str,
        state: &'a [A],
    ) -> Result<ResolvedFermion<'a, A>> {
        let ctx = &self.context;
        let basis = self.register(purpose)?;
        if let Some(taper) = &self.taper {
            let lifted = taper.lift(ctx, state)?;
            let sector = Basis::new(ctx, &self.encoding, true)?;
            let compressed = sector.compress_state(ctx, lifted.as_slice())?;
            return Ok(ResolvedFermion::Wide(sector, compressed));
        }
        if state.len() != basis.len() {
            Ok(ResolvedFermion::Local(basis.compress_state(ctx, state)?))
        } else {
            Ok(ResolvedFermion::Borrowed(state))
        }
    }

    /// Spin-orbital density matrices of `state`, a state of this problem's register: the sector or
    /// whole register of an untapered problem, or the tapered register. A tapered state is first
    /// lifted to the fermionic register (`Taper::lift`, `2^n` amplitudes, state-vector only) and
    /// read there in the particle-number sector; one that leaves the sector raises.
    pub fn rdms<A: Amplitude>(&self, state: &[A], two_body: bool) -> Result<Rdms> {
        let resolved = self.resolve_fermionic_state("density matrices", state)?;
        let basis = self.register("density matrices")?;
        match &resolved {
            ResolvedFermion::Wide(sector, buf) => properties::rdms(
                &self.context,
                &self.encoding,
                sector,
                buf.as_slice(),
                two_body,
            ),
            ResolvedFermion::Local(buf) => properties::rdms(
                &self.context,
                &self.encoding,
                basis,
                buf.as_slice(),
                two_body,
            ),
            ResolvedFermion::Borrowed(slice) => {
                properties::rdms(&self.context, &self.encoding, basis, slice, two_body)
            }
        }
    }

    /// Total spin expectation `<psi | S^2 | psi> / <psi | psi>` for a state in this problem's
    /// register (lifted to the fermionic particle sector when tapered).
    pub fn spin_squared<A: Amplitude>(&self, state: &[A]) -> Result<f64> {
        let resolved = self.resolve_fermionic_state("spin expectation", state)?;
        let basis = self.register("spin expectation")?;
        match &resolved {
            ResolvedFermion::Wide(sector, buf) => {
                properties::spin_squared(&self.context, &self.encoding, sector, buf.as_slice())
            }
            ResolvedFermion::Local(buf) => {
                properties::spin_squared(&self.context, &self.encoding, basis, buf.as_slice())
            }
            ResolvedFermion::Borrowed(slice) => {
                properties::spin_squared(&self.context, &self.encoding, basis, slice)
            }
        }
    }

    /// Spin-summed spatial transition density matrices `(<bra | E_pq | ket>, ...)` between two
    /// states in this problem's register (lifted when tapered).
    pub fn transition_rdms<A: Amplitude>(&self, bra: &[A], ket: &[A]) -> Result<SpatialRdms> {
        let resolved_b = self.resolve_fermionic_state("transition RDMs", bra)?;
        let resolved_k = self.resolve_fermionic_state("transition RDMs", ket)?;
        let basis = self.register("transition RDMs")?;
        let to_wide = |slice: &[A]| -> Result<Buffer<C64>> {
            let mut out = self.context.allocate::<C64>(slice.len())?;
            for (dst, &src) in out.as_mut_slice().iter_mut().zip(slice) {
                *dst = src.widen();
            }
            Ok(out)
        };
        let (sector_opt, bra_buf) = match resolved_b {
            ResolvedFermion::Wide(sector, buf) => (Some(sector), buf),
            ResolvedFermion::Local(buf) => (None, to_wide(buf.as_slice())?),
            ResolvedFermion::Borrowed(slice) => (None, to_wide(slice)?),
        };
        let ket_buf = match resolved_k {
            ResolvedFermion::Wide(_, buf) => buf,
            ResolvedFermion::Local(buf) => to_wide(buf.as_slice())?,
            ResolvedFermion::Borrowed(slice) => to_wide(slice)?,
        };
        properties::spatial_transition_rdms(
            &self.context,
            &self.encoding,
            sector_opt.as_ref().unwrap_or(basis),
            bra_buf.as_slice(),
            ket_buf.as_slice(),
        )
    }

    /// Rows of the lowest determinants of the particle sector, for the subspace methods: the
    /// determinants as `(excitation rank, occupation, row in the basis)` sorted, lowest first,
    /// restricted to the reference's symmetry sector when tapered.
    pub(super) fn sector_determinants(&self) -> Result<Vec<(u32, std::cmp::Reverse<u64>, usize)>> {
        let ctx = &self.context;
        let basis = self.register("subspace determinants")?;
        let sector = Basis::new(ctx, &self.encoding, true)?;
        let _sort = ctx.reserve(product(&[sector.len(), 32])?)?;
        let hf = self.encoding.decode(self.encoding.reference());
        let mut states = Vec::with_capacity(sector.len());
        for &encoded in sector.indices.as_slice() {
            let register = match &self.taper {
                Some(taper) if !taper.in_sector(encoded) => continue,
                Some(taper) => taper.bits(encoded),
                None => encoded,
            };
            let row = basis
                .indices
                .as_slice()
                .binary_search(&register)
                .map_err(|_| invalid("subspace determinant is absent from basis"))?;
            let occupation = self.encoding.decode(encoded);
            states.push((
                (occupation & !hf).count_ones(),
                std::cmp::Reverse(occupation),
                row,
            ));
        }
        states.sort_unstable();
        Ok(states)
    }
}
