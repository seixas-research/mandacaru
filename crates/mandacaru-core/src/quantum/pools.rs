//! Fermionic, spin-orbit, qubit-excitation and coupled-exchange ADAPT pools, plus optional
//! Bloch crystal-momentum symmetry filtering.
use super::algebra::{excitation_pool_filtered, Generator, Term};
use super::{Encoding, Operator, C64};
use crate::{product, Context, Error, Reservation, Result};
use std::collections::BTreeSet;

/// Crystal-momentum specification on a periodic Born–von Karman k-mesh for Bloch pools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrystalMomentum {
    pub mesh: [usize; 3],
    pub orbital_k: Vec<[i64; 3]>,
}

impl CrystalMomentum {
    /// Validate positive k-mesh extents and per-spatial-orbital integer crystal momenta.
    pub fn new(mesh: [usize; 3], orbital_k: Vec<[i64; 3]>) -> Result<Self> {
        if mesh.contains(&0) || orbital_k.is_empty() {
            return Err(Error::Invalid(
                "crystal momentum needs positive k-mesh dimensions and at least one spatial orbital"
                    .into(),
            ));
        }
        Ok(Self { mesh, orbital_k })
    }

    /// Return `true` iff exciting `annihilate -> create` conserves total crystal momentum modulo `mesh`.
    pub fn conserves_modes(&self, annihilate: &[usize], create: &[usize]) -> bool {
        let m = self.orbital_k.len();
        let mut delta = [0i64; 3];
        for &q in create {
            let k = self.orbital_k[q % m];
            for (d, &v) in delta.iter_mut().zip(&k) {
                *d += v;
            }
        }
        for &p in annihilate {
            let k = self.orbital_k[p % m];
            for (d, &v) in delta.iter_mut().zip(&k) {
                *d -= v;
            }
        }
        delta
            .iter()
            .zip(&self.mesh)
            .all(|(&d, &m)| d.rem_euclid(m as i64) == 0)
    }
}

/// Crystal point-/space-group Abelian orbital characters (`characters[s][p]` is the unit-modulus
/// eigenvalue of spatial orbital `p` under symmetry `s`).
#[derive(Clone, Debug, PartialEq)]
pub struct CrystalPointSymmetry {
    pub labels: Vec<String>,
    pub characters: Vec<Vec<C64>>,
}

impl CrystalPointSymmetry {
    /// Validate symmetry labels and unit-modulus orbital characters of uniform spatial width `m > 0`.
    pub fn new(labels: Vec<String>, characters: Vec<Vec<C64>>) -> Result<Self> {
        if labels.len() != characters.len() || characters.is_empty() {
            return Err(Error::Invalid(
                "crystal point symmetry needs matching non-empty symmetry labels and characters"
                    .into(),
            ));
        }
        let m = characters[0].len();
        if m == 0
            || characters.iter().any(|row| {
                row.len() != m
                    || row.iter().any(|c| {
                        !c.re.is_finite() || !c.im.is_finite() || (c.norm() - 1.).abs() > 1e-4
                    })
            })
        {
            return Err(Error::Invalid(
                "crystal point symmetry characters must be unit-modulus numbers of uniform spatial orbital length".into(),
            ));
        }
        Ok(Self { labels, characters })
    }

    /// Spatial orbital count `m`.
    pub fn n_orbitals(&self) -> usize {
        self.characters.first().map_or(0, Vec::len)
    }

    /// Return `true` iff exciting `annihilate -> create` transforms trivially under every symmetry.
    pub fn conserves_modes(&self, annihilate: &[usize], create: &[usize]) -> bool {
        let m = self.n_orbitals();
        if m == 0 {
            return true;
        }
        self.characters.iter().all(|row| {
            let mut prod = C64::new(1., 0.);
            for &q in create {
                prod *= row[q % m];
            }
            for &p in annihilate {
                prod *= row[p % m].conj();
            }
            (prod - C64::new(1., 0.)).norm() <= 1e-4
        })
    }

    /// Total symmetry character tuple across all occupied spatial orbitals encoded in `occ`
    /// (`occ` bit `2*m - 1 - bit`).
    pub fn occupation_characters(&self, modes: usize, occ: u64) -> Vec<C64> {
        let m = self.n_orbitals();
        self.characters
            .iter()
            .map(|row| {
                let mut prod = C64::new(1., 0.);
                for bit in 0..modes {
                    if (occ >> (modes - 1 - bit)) & 1 != 0 {
                        prod *= row[bit % m];
                    }
                }
                prod
            })
            .collect()
    }
}

fn label_matches_excitation(
    label: &str,
    predicate: impl Fn(&[usize], &[usize]) -> bool,
) -> Option<bool> {
    let mut saw_excitation = false;
    let mut rest = label;
    while let Some(open) = rest.find('(') {
        let after_open = &rest[open + 1..];
        let close = after_open.find(')')?;
        let inner = &after_open[..close];
        if let Some((lhs, rhs)) = inner.split_once("->") {
            let ann: Vec<usize> = lhs
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let cre: Vec<usize> = rhs
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            if ann.is_empty() || cre.is_empty() {
                return None;
            }
            saw_excitation = true;
            if !predicate(&ann, &cre) {
                return Some(false);
            }
        }
        rest = &after_open[close + 1..];
    }
    saw_excitation.then_some(true)
}

/// Check crystal-momentum conservation for labeled excitation/CEO generators (`Some(bool)`),
/// returning `None` for unlabeled Pauli-string generators (`qubit`).
pub(crate) fn label_conserves_momentum(label: &str, momentum: &CrystalMomentum) -> Option<bool> {
    label_matches_excitation(label, |ann, cre| momentum.conserves_modes(ann, cre))
}

/// Check Abelian crystal point-/space-group character conservation for labeled excitation/CEO
/// generators (`Some(bool)`), returning `None` for unlabeled Pauli-string generators (`qubit`).
pub(crate) fn label_conserves_point_symmetry(
    label: &str,
    symmetry: &CrystalPointSymmetry,
) -> Option<bool> {
    label_matches_excitation(label, |ann, cre| symmetry.conserves_modes(ann, cre))
}

pub struct Pool {
    pub operators: Vec<Generator>,
    pub screenable: Vec<bool>,
    pub growth_members: Vec<Vec<usize>>,
    pub memory: Reservation,
}

fn qubit_excitation(
    ctx: &Context,
    encoding: &Encoding,
    ladders: &[(usize, bool)],
) -> Result<Operator> {
    let modes = encoding.modes;
    let mut value = Operator::from_terms(
        ctx,
        modes,
        vec![Term {
            x: 0,
            z: 0,
            coefficient: C64::new(1., 0.),
        }],
    )?;
    for &(mode, dagger) in ladders {
        let bit = 1u64 << (modes - 1 - mode);
        let ladder = Operator::from_terms(
            ctx,
            modes,
            vec![
                Term {
                    x: bit,
                    z: 0,
                    coefficient: C64::new(0.5, 0.),
                },
                Term {
                    x: bit,
                    z: bit,
                    coefficient: C64::new(0., if dagger { -0.5 } else { 0.5 }),
                },
            ],
        )?;
        value = value.compose(ctx, &ladder)?;
    }
    let anti = value.add(ctx, &value.dagger(ctx)?, C64::new(-1., 0.))?;
    encoding.map_operator(ctx, &anti)
}

fn spin_orbit_pair(
    ctx: &Context,
    encoding: &Encoding,
    ladders: &[(usize, bool)],
) -> Result<[Operator; 2]> {
    let excitation = encoding.monomial(ctx, ladders)?;
    let adjoint = excitation.dagger(ctx)?;
    let real = excitation.add(ctx, &adjoint, C64::new(-1., 0.))?;
    let sum = excitation.add(ctx, &adjoint, C64::new(1., 0.))?;
    let imaginary = Operator::from_terms(
        ctx,
        encoding.n_qubits(),
        sum.terms
            .iter()
            .map(|term| Term {
                coefficient: C64::new(0., 1.) * term.coefficient,
                ..*term
            })
            .collect(),
    )?;
    Ok([real, imaginary])
}

/// Every pool `build` accepts and what it generates, in one table.
const KINDS: [(&str, &str); 6] = [
    ("fermionic", "fermionic excitations"),
    ("spin-orbit", "spin-orbit generalized excitations"),
    ("qubit", "individual Pauli strings"),
    ("qeb", "qubit excitations"),
    ("ceo", "coupled exchange operators"),
    (
        "ceo-ovp",
        "coupled exchange operators, one parameter per growth step",
    ),
];
/// What the pool called `name` generates, for a run log. Raises for a name
/// `build` does not accept, so a log never describes a pool that was not built.
pub fn kind(name: &str) -> Result<&'static str> {
    KINDS
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, label)| *label)
        .ok_or_else(|| Error::Invalid("unknown ADAPT operator pool".into()))
}

/// The refusal of a pool that is register-based by construction, which only an untapered register
/// can hold: a parity-reduced one (two qubits tapered) and a Z2-tapered one (`QuantumProblem::taper`)
/// both raise it.
pub fn untapered_register(name: &str) -> Error {
    Error::Invalid(
        match name {
            "spin-orbit" => "the spin-orbit pool changes alpha/beta parity and requires an untapered register",
            _ => "the individual Pauli pool uses an untapered register; use qeb for a particle-conserving tapered pool",
        }
        .into(),
    )
}
/// Whether the pool called `name` is register-based and so needs an untapered register.
pub fn needs_untapered_register(name: &str) -> bool {
    matches!(name, "spin-orbit" | "qubit")
}

pub fn build(ctx: &Context, encoding: &Encoding, name: &str, singles: bool) -> Result<Pool> {
    build_with_momentum(ctx, encoding, name, singles, None)
}

/// Build an operator pool optionally restricted to excitations conserving crystal momentum
/// (`Bloch-*` domain extension, preserving spin-orbit/fermionic/qeb/ceo/qubit parity).
pub fn build_with_momentum(
    ctx: &Context,
    encoding: &Encoding,
    name: &str,
    singles: bool,
    momentum: Option<&CrystalMomentum>,
) -> Result<Pool> {
    build_with_crystal_symmetry(ctx, encoding, name, singles, momentum, None)
}

/// Build an operator pool optionally restricted to excitations conserving crystal momentum
/// and Abelian crystal point-/space-group characters.
pub fn build_with_crystal_symmetry(
    ctx: &Context,
    encoding: &Encoding,
    name: &str,
    singles: bool,
    momentum: Option<&CrystalMomentum>,
    point_symmetry: Option<&CrystalPointSymmetry>,
) -> Result<Pool> {
    let m = encoding.modes / 2;
    if let Some(mom) = momentum {
        if encoding.modes % 2 != 0 || mom.orbital_k.len() != m {
            return Err(Error::Invalid(
                "crystal momentum orbital count must match the spatial orbitals".into(),
            ));
        }
    }
    if let Some(sym) = point_symmetry {
        if encoding.modes % 2 != 0 || (sym.n_orbitals() != m && sym.n_orbitals() != encoding.modes)
        {
            return Err(Error::Invalid(
                "crystal point symmetry orbital count must match spatial or spin-orbital count"
                    .into(),
            ));
        }
    }
    let conserves = |ann: &[usize], cre: &[usize]| -> bool {
        momentum.is_none_or(|mom| mom.conserves_modes(ann, cre))
            && point_symmetry.is_none_or(|sym| sym.conserves_modes(ann, cre))
    };
    let mut operators = Vec::new();
    let mut screenable = Vec::new();
    let mut growth_members = Vec::new();
    let mut append = |generator: Generator, visible: bool, members: Vec<usize>| {
        let index = operators.len();
        operators.push(generator);
        screenable.push(visible);
        growth_members.push(members);
        index
    };
    match name {
        "fermionic" => {
            for generator in
                excitation_pool_filtered(ctx, encoding, singles, |a, c| conserves(a, c))?
            {
                append(generator, true, vec![]);
            }
        }
        "spin-orbit" => {
            if encoding.mapping == "parity_reduced" {
                return Err(untapered_register("spin-orbit"));
            }
            let occupied: Vec<_> = (0..encoding.particles[0])
                .chain(m..m + encoding.particles[1])
                .collect();
            let virtuals: Vec<_> = (0..encoding.modes)
                .filter(|i| !occupied.contains(i))
                .collect();
            if singles {
                for &i in &occupied {
                    for &a in &virtuals {
                        if !conserves(&[i], &[a]) {
                            continue;
                        }
                        let [real, imaginary] =
                            spin_orbit_pair(ctx, encoding, &[(a, true), (i, false)])?;
                        append(
                            Generator {
                                label: format!("S({i}->{a})"),
                                operator: real,
                            },
                            true,
                            vec![],
                        );
                        append(
                            Generator {
                                label: format!("iS({i}->{a})"),
                                operator: imaginary,
                            },
                            true,
                            vec![],
                        );
                    }
                }
            }
            for (ii, &i) in occupied.iter().enumerate() {
                for &j in &occupied[ii + 1..] {
                    for (aa, &a) in virtuals.iter().enumerate() {
                        for &b in &virtuals[aa + 1..] {
                            if !conserves(&[i, j], &[a, b]) {
                                continue;
                            }
                            let [real, imaginary] = spin_orbit_pair(
                                ctx,
                                encoding,
                                &[(a, true), (b, true), (j, false), (i, false)],
                            )?;
                            append(
                                Generator {
                                    label: format!("D({i},{j}->{a},{b})"),
                                    operator: real,
                                },
                                true,
                                vec![],
                            );
                            append(
                                Generator {
                                    label: format!("iD({i},{j}->{a},{b})"),
                                    operator: imaginary,
                                },
                                true,
                                vec![],
                            );
                        }
                    }
                }
            }
        }
        "qubit" => {
            if encoding.mapping == "parity_reduced" {
                return Err(untapered_register("qubit"));
            }
            let mut masks = BTreeSet::new();
            for generator in
                excitation_pool_filtered(ctx, encoding, singles, |a, c| conserves(a, c))?
            {
                for term in &generator.operator.terms {
                    if term.x != 0 {
                        masks.insert((term.x, term.z));
                    }
                }
            }
            for (x, z) in masks {
                let operator = Operator::from_terms(
                    ctx,
                    encoding.n_qubits(),
                    vec![Term {
                        x,
                        z,
                        coefficient: C64::new(0., 1.),
                    }],
                )?;
                let label = format!("iP[{}]", operator.labels()[0].0);
                append(Generator { label, operator }, true, vec![]);
            }
        }
        "qeb" => {
            let occupied: Vec<_> = (0..encoding.particles[0])
                .chain(m..m + encoding.particles[1])
                .collect();
            let virtuals: Vec<_> = (encoding.particles[0]..m)
                .chain(m + encoding.particles[1]..2 * m)
                .collect();
            if singles {
                for &i in &occupied {
                    for &a in &virtuals {
                        if i / m == a / m && conserves(&[i], &[a]) {
                            append(
                                Generator {
                                    label: format!("QS({i}->{a})"),
                                    operator: qubit_excitation(
                                        ctx,
                                        encoding,
                                        &[(a, true), (i, false)],
                                    )?,
                                },
                                true,
                                vec![],
                            );
                        }
                    }
                }
            }
            for (ii, &i) in occupied.iter().enumerate() {
                for &j in &occupied[ii + 1..] {
                    for (aa, &a) in virtuals.iter().enumerate() {
                        for &b in &virtuals[aa + 1..] {
                            if i / m + j / m == a / m + b / m && conserves(&[i, j], &[a, b]) {
                                append(
                                    Generator {
                                        label: format!("QD({i},{j}->{a},{b})"),
                                        operator: qubit_excitation(
                                            ctx,
                                            encoding,
                                            &[(a, true), (b, true), (j, false), (i, false)],
                                        )?,
                                    },
                                    true,
                                    vec![],
                                );
                            }
                        }
                    }
                }
            }
        }
        "ceo" | "ceo-ovp" => {
            if singles {
                for i in 0..2 * m {
                    for a in i + 1..2 * m {
                        if i / m == a / m && conserves(&[i], &[a]) {
                            append(
                                Generator {
                                    label: format!("CEO[o{i},o{a}]{{S({i}->{a})}}"),
                                    operator: qubit_excitation(
                                        ctx,
                                        encoding,
                                        &[(a, true), (i, false)],
                                    )?,
                                },
                                true,
                                vec![],
                            );
                        }
                    }
                }
            }
            for p in 0..2 * m {
                for q in p + 1..2 * m {
                    for r in q + 1..2 * m {
                        for s in r + 1..2 * m {
                            let modes = [p, q, r, s];
                            let mut members = vec![];
                            for (left, right) in
                                [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]
                            {
                                let [i, j] = left.map(|n| modes[n]);
                                let [a, b] = right.map(|n| modes[n]);
                                if i / m + j / m == a / m + b / m && conserves(&[i, j], &[a, b]) {
                                    members.push(Generator {
                                        label: format!("D({i},{j}->{a},{b})"),
                                        operator: qubit_excitation(
                                            ctx,
                                            encoding,
                                            &[(a, true), (b, true), (i, false), (j, false)],
                                        )?,
                                    });
                                }
                            }
                            if members.is_empty() {
                                continue;
                            }
                            if members.len() == 1 {
                                let mut only = members.swap_remove(0);
                                only.label = format!("CEO[o{p},o{q},o{r},o{s}]{{{}}}", only.label);
                                append(only, true, vec![]);
                                continue;
                            }
                            let mut indices = vec![];
                            for member in &members {
                                indices.push(append(
                                    Generator {
                                        label: member.label.clone(),
                                        operator: member.operator.copy_into(ctx)?,
                                    },
                                    false,
                                    vec![],
                                ));
                            }
                            for x in 0..members.len() {
                                for y in x + 1..members.len() {
                                    for (sign, mark) in [(1., "+"), (-1., "-")] {
                                        let operator = members[x].operator.add(
                                            ctx,
                                            &members[y].operator,
                                            C64::new(sign, 0.),
                                        )?;
                                        let label = format!(
                                            "CEO[o{p},o{q},o{r},o{s}]{{{}{mark}{}}}",
                                            members[x].label, members[y].label
                                        );
                                        append(
                                            Generator { label, operator },
                                            true,
                                            if name == "ceo" {
                                                vec![indices[x], indices[y]]
                                            } else {
                                                vec![]
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => return Err(Error::Invalid("unknown ADAPT operator pool".into())),
    }
    let memory = ctx.reserve(product(&[operators.len(), 256])?)?;
    Ok(Pool {
        operators,
        screenable,
        growth_members,
        memory,
    })
}
