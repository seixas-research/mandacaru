//! A quantum problem on a Z2-tapered register: the Hamiltonian, the pool, the reference and the
//! basis reduced together by one [`Taper`], and the state-vector route back to the fermionic
//! register that the density matrices read.
use super::{
    algebra::Generator,
    amplitude::Amplitude,
    pools,
    properties::{self, Rdms},
    solver::CachedProblem,
    tapering::Taper,
    Basis,
};
use crate::{product, Error, Result};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

impl CachedProblem {
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
        let mut renumber = vec![None; self.pool.len()];
        let mut pool = Vec::with_capacity(self.pool.len());
        let mut screenable = Vec::with_capacity(self.pool.len());
        let mut dropped = Vec::new();
        for (index, generator) in self.pool.iter().enumerate() {
            if generator.operator.terms.iter().any(|t| taper.leaks(t.x)) {
                dropped.push((index, generator.label.clone()));
                continue;
            }
            renumber[index] = Some(pool.len());
            pool.push(Generator {
                label: generator.label.clone(),
                operator: taper.operator(&ctx, &generator.operator)?,
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
        self.pool = pool;
        self.pool_screenable = screenable;
        self.growth_members = growth_members;
        self.basis = basis;
        self.conserved = Vec::new();
        self.dense = dense;
        self.dropped = dropped;
        self._pool_metadata = memory;
        self.taper = Some(taper);
        Ok(())
    }

    /// Spin-orbital density matrices of `state`, a state of this problem's register: the sector or
    /// whole register of an untapered problem, or the tapered register. A tapered state is first
    /// lifted to the fermionic register (`Taper::lift`, `2^n` amplitudes, state-vector only) and
    /// read there in the particle-number sector; one that leaves the sector raises.
    pub fn rdms<A: Amplitude>(&self, state: &[A], two_body: bool) -> Result<Rdms> {
        let ctx = &self.context;
        self.register("density matrices")?;
        if let Some(taper) = &self.taper {
            let lifted = taper.lift(ctx, state)?;
            let sector = Basis::new(ctx, &self.encoding, true)?;
            let compressed = sector.compress_state(ctx, lifted.as_slice())?;
            return properties::rdms(
                ctx,
                &self.encoding,
                &sector,
                compressed.as_slice(),
                two_body,
            );
        }
        let basis = self.register("density matrices")?;
        let compressed = if state.len() != basis.len() {
            Some(basis.compress_state(ctx, state)?)
        } else {
            None
        };
        properties::rdms(
            ctx,
            &self.encoding,
            basis,
            compressed.as_ref().map_or(state, |v| v.as_slice()),
            two_body,
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
