//! The ionic local potential is the bare Coulomb potential `-Zv/r` outside the augmentation, and
//! the dataset says so exactly.
//!
//! `ionic = v_screened - V_H[augmented valence] - v_xc[smooth valence + core]` is the bare ion only
//! where the smooth valence density is the all-electron one: for a fully relativistic atom the
//! all-electron potential comes from the Dirac atom and the smooth density from the scalar
//! reference waves, which agree to a part in 1e3 of the electrons beyond 10 Bohr and no better. The
//! difference is `-dq/r` with `dq` of order 1e-3 to 1e-2 electrons out to 20 Bohr, a charge that
//! is not the ion's: a crystal sums it over every atom as a constant energy per volume
//! (`N_valence * alpha / V`, `alpha = int (v + Zv/r) 4 pi r^2 dr`), measured on Bi2Se3 as 6 GPa and
//! 6 eV from `alpha = -1.4 Ha bohr^3` of Bi, which only a sum over the whole table shows (the first
//! 15 Bohr give +0.16). The engine adds the Hartree potential of its own smooth density to the
//! ion, so the ion must carry exactly `Zv`: beyond the local radius the stored potential is
//! `-Zv/r`, joined to the one inside by a smooth step over a fifth of a radius.
/// The step is complete this multiple of the local radius out.
pub const STEP_END: f64 = 1.2;

/// `-Zv/r` replaces `ionic` from `r_local * STEP_END` out, through a cubic smooth step from
/// `r_local` (all of `ionic`) to there (none of it).
pub fn enforce(r: &[f64], ionic: &mut [f64], zv: f64, r_local: f64) {
    for (&x, v) in r.iter().zip(ionic.iter_mut()) {
        let t = ((x - r_local) / ((STEP_END - 1.) * r_local)).clamp(0., 1.);
        let weight = 1. - t * t * (3. - 2. * t);
        *v = weight * *v + (1. - weight) * (-zv / x);
    }
}
