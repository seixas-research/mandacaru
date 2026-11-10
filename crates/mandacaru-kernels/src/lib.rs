//! Low-level grid sampling, radial equations and one-center kernels, called from
//! `mandacaru-core` only (there is no C ABI). Raw-pointer entry points require the contracts
//! documented in SAFETY.md.

pub mod bloch;
pub mod heap;
pub mod one_body;
pub mod onecenter;
pub mod radial;
pub mod runtime;
pub mod two_center;
