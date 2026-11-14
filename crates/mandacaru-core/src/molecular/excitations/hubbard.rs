//! The +U part of the response kernel: the energy is quadratic in the shell occupations
//! `n_s = Q^T D_s Q` (Dudarev and Liechtenstein), so its kernel is the constant linear map of the
//! occupation change to the potential change (`MolecularHubbard::shell_linear`), carried to the
//! occupied-empty blocks by the shell projections of each spin's canonical orbitals. The
//! antisymmetric (`A - B`) channel is the same map at an imaginary Hermitian occupation change.
use super::invalid;
use crate::{molecular::hubbard::MolecularHubbard, quantum::C64, Context, Result};
use nalgebra::DMatrix;

/// How the spin channels of a trial vector map onto the shell occupations.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// A closed shell, equal changes of both spins (one restricted occupation change).
    Singlet,
    /// A closed shell, opposite changes of the two spins.
    Triplet,
    /// Two independent spin channels.
    Unrestricted,
}

/// The shell projections of one spin channel: `C_o^T Q` and `C_v^T Q` per shell.
struct Projections {
    shells: Vec<(DMatrix<f64>, DMatrix<f64>)>,
}

pub(super) struct HubbardKernel {
    model: MolecularHubbard,
    /// One per channel of the trial vector (one for a closed shell, two unrestricted).
    channels: Vec<Projections>,
}

impl HubbardKernel {
    /// The kernel for the canonical orbitals `orbitals` (orthonormal frame) of each channel, with
    /// `o` occupied orbitals each.
    pub(super) fn new(model: &MolecularHubbard, channels: &[(&DMatrix<f64>, usize)]) -> Self {
        let channels = channels
            .iter()
            .map(|(c, o)| {
                let (co, cv) = (c.columns(0, *o), c.columns(*o, c.ncols() - *o));
                Projections {
                    shells: model
                        .shells
                        .iter()
                        .map(|shell| (co.transpose() * &shell.q, cv.transpose() * &shell.q))
                        .collect(),
                }
            })
            .collect();
        Self {
            model: model.clone(),
            channels,
        }
    }
    /// The +U kernel of the trial blocks `ys` (one per channel), symmetric (`antisymmetric =
    /// false`: the `A + B` channel) or antisymmetric (`A - B`), in `mode`.
    pub(super) fn apply(
        &self,
        ctx: &Context,
        mode: Mode,
        ys: &[DMatrix<f64>],
        antisymmetric: bool,
    ) -> Result<Vec<DMatrix<f64>>> {
        let wanted = if mode == Mode::Unrestricted { 2 } else { 1 };
        if ys.len() != wanted || self.channels.len() != wanted {
            return Err(invalid("the +U kernel and its trial blocks disagree"));
        }
        let mut out: Vec<DMatrix<f64>> = ys
            .iter()
            .map(|y| DMatrix::zeros(y.nrows(), y.ncols()))
            .collect();
        let sign = if antisymmetric { -1. } else { 1. };
        let lift = |x: &DMatrix<f64>| {
            x.map(|v| {
                if antisymmetric {
                    C64::new(0., v)
                } else {
                    C64::new(v, 0.)
                }
            })
        };
        for (index, shell) in self.model.shells.iter().enumerate() {
            let change = |c: usize| {
                let (po, pv) = &self.channels[c].shells[index];
                let d = po.transpose() * &ys[c] * pv;
                &d + d.transpose() * sign
            };
            let changes: Vec<DMatrix<C64>> = match mode {
                Mode::Singlet => vec![lift(&change(0))],
                Mode::Triplet => {
                    let d = change(0);
                    vec![lift(&d), lift(&(-&d))]
                }
                Mode::Unrestricted => vec![lift(&change(0)), lift(&change(1))],
            };
            let potentials = self.model.shell_linear(ctx, shell, &changes)?;
            for (c, block) in out.iter_mut().enumerate() {
                let (po, pv) = &self.channels[c].shells[index];
                let part = potentials[c].map(|z| if antisymmetric { z.im } else { z.re });
                *block += po * part * pv.transpose();
            }
        }
        Ok(out)
    }
}
