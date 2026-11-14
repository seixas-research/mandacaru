//! The shell ladders of the named Gaussian basis families: how many valence functions, which
//! polarization and core-correlating shells a family stacks at each zeta level. The Python
//! recipe (`models/basis.py`) only parses the name; the rule is here.

/// Longest contraction the Slater-orbital fit is asked for. The published cc-pV5Z core runs to
/// 14 primitives, but past ten the primitive overlap matrix is numerically singular and the fit
/// gains nothing (`1 - <S|P|S>` is already 4e-9 at ten); the function count does not depend on it.
pub const MAX_CONTRACTION: usize = 10;

/// The shell structure of a correlation-consistent basis at one level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DunningLadder {
    /// The contraction of the core shell.
    pub core: usize,
    /// The contraction of every valence shell: four primitives, then one each.
    pub valence: Vec<usize>,
    /// `(l, count)` polarization shells on heavy atoms.
    pub polarization: Vec<(usize, usize)>,
    /// `(l, count)` polarization shells on hydrogen and helium.
    pub polarization_h: Vec<(usize, usize)>,
    /// `(l, count)` core-correlating tight shells on heavy atoms (`cc-pCVXZ`), none otherwise.
    pub tight: Vec<(usize, usize)>,
}

/// The ladder of `cc-pVXZ` for `zeta = X` (2 for D, 3 for T, ...), with the tight `s` and `p`
/// shells of the core-correlating `cc-pCVXZ` when `core_correlating`. A heavy atom carries
/// `X - 1` d, `X - 2` f, ...; hydrogen `X - 1` p, `X - 2` d, ...; the tight shells `X - 1` each.
pub fn dunning(zeta: usize, core_correlating: bool) -> Option<DunningLadder> {
    if !(2..=6).contains(&zeta) {
        return None;
    }
    Some(DunningLadder {
        core: (4 + 2 * zeta).min(MAX_CONTRACTION),
        valence: std::iter::once(4)
            .chain(std::iter::repeat_n(1, zeta - 1))
            .collect(),
        polarization: (0..zeta - 1).map(|k| (2 + k, zeta - 1 - k)).collect(),
        polarization_h: (0..zeta - 1).map(|k| (1 + k, zeta - 1 - k)).collect(),
        tight: if core_correlating {
            vec![(0, zeta - 1), (1, zeta - 1)]
        } else {
            Vec::new()
        },
    })
}
