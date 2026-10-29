//! Fermionic, spin-orbit, qubit-excitation and coupled-exchange ADAPT pools.
use super::algebra::{excitation_pool, Generator, Term};
use super::{Encoding, Operator, C64};
use crate::{product, Context, Error, Reservation, Result};
use std::collections::BTreeSet;

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

pub fn build(ctx: &Context, encoding: &Encoding, name: &str, singles: bool) -> Result<Pool> {
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
    let m = encoding.modes / 2;
    match name {
        "fermionic" => {
            for generator in excitation_pool(ctx, encoding, singles)? {
                append(generator, true, vec![]);
            }
        }
        "spin-orbit" => {
            if encoding.mapping == "parity_reduced" {
                return Err(Error::Invalid("the spin-orbit pool changes alpha/beta parity and requires an untapered register".into()));
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
                return Err(Error::Invalid(
                    "the individual Pauli pool uses an untapered register; use qeb for a particle-conserving tapered pool"
                        .into(),
                ));
            }
            let mut masks = BTreeSet::new();
            for generator in excitation_pool(ctx, encoding, singles)? {
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
                        if i / m == a / m {
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
                            if i / m + j / m == a / m + b / m {
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
                        if i / m == a / m {
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
                                if i / m + j / m == a / m + b / m {
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
                            if members.len() == 1 {
                                let mut only = members.pop().unwrap();
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
