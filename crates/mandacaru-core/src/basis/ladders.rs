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

/// The shell structure of a Pople split-valence name `K-NL[M][+|++]G[*|**|(...)]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopleShells {
    /// `(l, count)` polarization shells on heavy atoms (`Z >= polarization_min_z`).
    pub polarization: Vec<(usize, usize)>,
    /// `(l, count)` polarization shells on hydrogen and helium.
    pub polarization_h: Vec<(usize, usize)>,
    /// Angular momenta with one diffuse function on heavy atoms (`+`: s and p).
    pub diffuse: Vec<usize>,
    /// The same on hydrogen and helium (`++`: s).
    pub diffuse_h: Vec<usize>,
    /// The lowest atomic number that counts as heavy: 11 for `3-21G*` (d on the second row
    /// only), 3 otherwise.
    pub polarization_min_z: usize,
}

/// The polarization shells of `"3df"`, `"2p"`, `"d"`: `(l, count)` in the order written, a
/// missing count being 1. Spaces between shells are allowed; any other character, or a count
/// of zero, is refused rather than skipped.
pub fn polarization_spec(text: &str) -> crate::Result<Vec<(usize, usize)>> {
    const LETTERS: &str = "spdfghi";
    let refuse = || crate::Error::Invalid(format!("cannot read the polarization spec '{text}'"));
    let lower = text.trim().to_lowercase();
    let mut shells = Vec::new();
    let mut chars = lower.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut digits = String::new();
        while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
            digits.push(d);
            chars.next();
        }
        let l = chars
            .next()
            .and_then(|letter| LETTERS.find(letter))
            .ok_or_else(refuse)?;
        let count = if digits.is_empty() {
            1
        } else {
            digits.parse::<usize>().map_err(|_| refuse())?
        };
        if count == 0 {
            return Err(crate::Error::Invalid(format!(
                "a polarization shell needs a count of at least 1: '{text}'"
            )));
        }
        shells.push((l, count));
    }
    Ok(shells)
}

/// The rule of a Pople name with `core` primitives in the core contraction, the valence
/// contractions `valence` (`[3, 1, 1]` for 6-311G), `plus` diffuse marks and `stars`
/// polarization marks, or the polarization `explicit` in parentheses (`"2df,2p"`: the heavy
/// atoms' shells, then hydrogen's). `*` puts one d shell on heavy atoms, `**` adds one p on
/// hydrogen; `+` adds diffuse s and p on heavy atoms, `++` an s on hydrogen as well.
pub fn pople(
    core: usize,
    valence: &[usize],
    plus: usize,
    stars: usize,
    explicit: Option<&str>,
) -> crate::Result<PopleShells> {
    if plus > 2 || stars > 2 {
        return Err(crate::Error::Invalid(
            "a Pople name has at most two '+' and two '*'".into(),
        ));
    }
    let (polarization, polarization_h) = match explicit {
        Some(_) if stars > 0 => {
            return Err(crate::Error::Invalid(
                "give a Pople polarization as '*' or in parentheses, not both".into(),
            ))
        }
        Some(text) => {
            let groups: Vec<&str> = text.split(',').map(str::trim).collect();
            if groups.len() > 2 {
                return Err(crate::Error::Invalid(format!(
                    "a Pople polarization names the heavy atoms' shells and hydrogen's: \
                     '({text})' has {} groups",
                    groups.len()
                )));
            }
            let heavy = polarization_spec(groups[0])?;
            let light = match groups.get(1) {
                Some(group) => polarization_spec(group)?,
                None => Vec::new(),
            };
            (heavy, light)
        }
        None => (
            if stars >= 1 { vec![(2, 1)] } else { Vec::new() },
            if stars == 2 { vec![(1, 1)] } else { Vec::new() },
        ),
    };
    Ok(PopleShells {
        polarization,
        polarization_h,
        diffuse: if plus >= 1 { vec![0, 1] } else { Vec::new() },
        diffuse_h: if plus == 2 { vec![0] } else { Vec::new() },
        polarization_min_z: if core == 3 && valence == [2, 1] {
            11
        } else {
            3
        },
    })
}

/// The shell structure of a Karlsruhe `def2-` basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KarlsruheShells {
    /// The contraction of every core shell; `[6, 1]` adds an uncontracted tight primitive.
    pub core: Vec<usize>,
    /// The contraction of every valence shell.
    pub valence: Vec<usize>,
    /// `(l, count)` polarization shells on heavy atoms.
    pub polarization: Vec<(usize, usize)>,
    /// `(l, count)` polarization shells on hydrogen and helium.
    pub polarization_h: Vec<(usize, usize)>,
    /// One diffuse function for every angular momentum the atom carries (the `D` members).
    pub diffuse_present: bool,
    /// The highest angular momentum that receives a diffuse function.
    pub diffuse_l_max: usize,
}

/// The shells of `def2-<key>` (`key` upper case: `"SV(P)"`, `"SVP"`, `"TZVPPD"`, ...); `TZDP`
/// is a spelling of `TZVP`. `None` for a name outside the family.
pub fn karlsruhe(key: &str) -> Option<KarlsruheShells> {
    type Shells = &'static [(usize, usize)];
    const SVP: (&[usize], &[usize], Shells) = (&[5], &[3, 1], &[(2, 1)]);
    const TZVP: (&[usize], &[usize], Shells) = (&[6, 1], &[3, 1, 1], &[(2, 2), (3, 1)]);
    const QZVP: (&[usize], &[usize], Shells) =
        (&[7, 1, 1], &[3, 1, 1, 1], &[(2, 3), (3, 2), (4, 1)]);
    let ((core, valence, polarization), polarization_h, diffuse_present): (_, Shells, bool) =
        match key {
            "SV(P)" => (SVP, &[], false),
            "SVP" => (SVP, &[(1, 1)], false),
            "SVPD" => (SVP, &[(1, 1)], true),
            "TZVP" | "TZDP" => (TZVP, &[(1, 1)], false),
            "TZVPD" => (TZVP, &[(1, 1)], true),
            "TZVPP" => (TZVP, &[(1, 2), (2, 1)], false),
            "TZVPPD" => (TZVP, &[(1, 2), (2, 1)], true),
            "QZVP" => (QZVP, &[(1, 3), (2, 2)], false),
            "QZVPD" => (QZVP, &[(1, 3), (2, 2)], true),
            "QZVPP" => (QZVP, &[(1, 3), (2, 2), (3, 1)], false),
            "QZVPPD" => (QZVP, &[(1, 3), (2, 2), (3, 1)], true),
            _ => return None,
        };
    Some(KarlsruheShells {
        core: core.to_vec(),
        valence: valence.to_vec(),
        polarization: polarization.to_vec(),
        polarization_h: polarization_h.to_vec(),
        diffuse_present,
        diffuse_l_max: 2,
    })
}
