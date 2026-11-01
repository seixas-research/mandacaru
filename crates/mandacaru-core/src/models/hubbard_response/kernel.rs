//! Reductions of the full spin-resolved kernel and the per-key interaction.
use super::{checked_inverse, Response};
use crate::models::invalid;
use crate::Result;
use nalgebra::DMatrix;

/// How the two spin channels of a shell are reduced to one number. There is no default:
/// which element of the 2 x 2 kernel maps onto the Dudarev functional's own response is
/// undecided (TO-VERIFY), and they differ by `J / 2` in the Hartree-Hund toy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// `(F_uu + F_dd) / 2`.
    SameSpin,
    /// `(F_uu + F_ud + F_du + F_dd) / 4`: the spin-summed scalar when the bare response
    /// has `(1, 1)` as an eigenvector.
    Charge,
    /// `chi0^-1 - chi^-1` of the spin-summed matrices: the response to a spin-independent
    /// shift of the total occupation.
    SpinSummed,
}

/// The value of one key (for example an element) over its shells.
#[derive(Clone, Debug)]
pub struct Group {
    pub key: String,
    pub shells: Vec<usize>,
    /// Mean of the diagonal `U_II` over the shells of the key.
    pub u: f64,
    /// Largest `|U_II - u|` over the same shells.
    pub spread: f64,
}

impl Response {
    /// The `shells x shells` kernel under `kernel`. A response with one channel per
    /// shell is already spin-summed and answers only [`Kernel::SpinSummed`].
    pub fn kernel(&self, kernel: Kernel) -> Result<DMatrix<f64>> {
        let (s, c) = (self.shells, self.channels);
        if c == 1 {
            return match kernel {
                Kernel::SpinSummed => Ok(self.u.clone()),
                other => Err(invalid(format!(
                    "{other:?} reduces two spin channels; this response has one channel per shell"
                ))),
            };
        }
        let block = |m: &DMatrix<f64>, i: usize, j: usize| -> [f64; 4] {
            [
                m[(2 * i, 2 * j)],
                m[(2 * i, 2 * j + 1)],
                m[(2 * i + 1, 2 * j)],
                m[(2 * i + 1, 2 * j + 1)],
            ]
        };
        match kernel {
            Kernel::SameSpin => Ok(DMatrix::from_fn(s, s, |i, j| {
                let b = block(&self.u, i, j);
                0.5 * (b[0] + b[3])
            })),
            Kernel::Charge => Ok(DMatrix::from_fn(s, s, |i, j| {
                block(&self.u, i, j).iter().sum::<f64>() / 4.
            })),
            Kernel::SpinSummed => {
                let sum = |m: &DMatrix<f64>| {
                    DMatrix::from_fn(s, s, |i, j| block(m, i, j).iter().sum::<f64>())
                };
                let a = checked_inverse(&sum(&self.chi0), self.limit, "spin-summed chi0", "")?;
                let b = checked_inverse(&sum(&self.chi), self.limit, "spin-summed chi", "")?;
                Ok(a.matrix - b.matrix)
            }
        }
    }

    /// Per shell, `F_ud - F_uu`: the same-shell opposite-spin kernel minus the same-spin
    /// one. In the Hartree-Hund toy it is `J`. In LSDA it is the projected exchange-
    /// correlation spin kernel, not the Slater `J`: reported, fed to nothing.
    pub fn j_response(&self) -> Result<Vec<f64>> {
        if self.channels != 2 {
            return Err(invalid("j_response needs two spin channels per shell"));
        }
        Ok((0..self.shells)
            .map(|i| {
                let (uu, ud) = (self.u[(2 * i, 2 * i)], self.u[(2 * i, 2 * i + 1)]);
                let (du, dd) = (self.u[(2 * i + 1, 2 * i)], self.u[(2 * i + 1, 2 * i + 1)]);
                0.5 * (ud + du) - 0.5 * (uu + dd)
            })
            .collect())
    }

    /// Groups the shells by `keys[shell]` (in order of first appearance) and returns the
    /// mean diagonal of `kernel` per key. A key whose shells spread by more than
    /// `tolerance` (same unit as the kernel) refuses, naming the key.
    pub fn hubbard_u(&self, kernel: Kernel, keys: &[&str], tolerance: f64) -> Result<Vec<Group>> {
        if keys.len() != self.shells {
            return Err(invalid(format!(
                "{} keys for {} shells",
                keys.len(),
                self.shells
            )));
        }
        if !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid(
                "the spread tolerance must be finite and not negative",
            ));
        }
        let matrix = self.kernel(kernel)?;
        let mut groups: Vec<Group> = Vec::new();
        for (shell, key) in keys.iter().enumerate() {
            match groups.iter_mut().find(|g| g.key == *key) {
                Some(g) => g.shells.push(shell),
                None => groups.push(Group {
                    key: key.to_string(),
                    shells: vec![shell],
                    u: 0.,
                    spread: 0.,
                }),
            }
        }
        for g in &mut groups {
            let values: Vec<f64> = g.shells.iter().map(|&s| matrix[(s, s)]).collect();
            g.u = values.iter().sum::<f64>() / values.len() as f64;
            g.spread = values.iter().fold(0., |m, v| f64::max(m, (v - g.u).abs()));
            if g.spread > tolerance {
                return Err(invalid(format!(
                    "the shells of {:?} spread by {:.3e} in U, more than the tolerance {tolerance:e}",
                    g.key, g.spread
                )));
            }
        }
        Ok(groups)
    }
}
