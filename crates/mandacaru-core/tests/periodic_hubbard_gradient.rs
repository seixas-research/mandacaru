//! The +U force and strain layer (`periodic::hubbard_gradient`, HU-10a) on a toy with no engine:
//! a periodic chain with a non-orthogonal Gaussian basis whose `S(X)`, `H0(X)` and `C(X)` are
//! analytic, a spin-polarized SCF converged to a density residual below 1e-14, and exact
//! (two-level Richardson) finite differences of the converged energy as the reference. The
//! plain part of the derivative, `tr[D dH0] - tr[W dS]`, is written in the test support; the
//! layer supplies the +U increment. See the header of `hubbard_gradient.rs` for the formula.
//!
//! Gate: 1e-10 on every force and strain component, translation sum rule 1e-12. Five negative
//! controls must fail by orders of magnitude: the projector-position term omitted, `R` dropped
//! from `d_R`, `V` scaled by 1.01, `W` from the bare `H0`, and `V_dc` dropped from `O`.
mod hubbard_gradient_support;

use hubbard_gradient_support::{
    functional::Functional,
    reference::{finite_forces, finite_strain, plain, Packed},
    scf::{solve, State},
    Matrix, Toy,
};
use mandacaru_core::periodic::hubbard_gradient::{Geometry, Gradient};
use mandacaru_core::{quantum::C64, Error};
use std::sync::OnceLock;

const GATE: f64 = 1e-10;
const SUM_RULE: f64 = 1e-12;

struct Scenario {
    toy: Toy,
    state: State,
    forces: Vec<[f64; 3]>,
    strain: [[f64; 3]; 3],
}

const DUDAREV: Functional = Functional::Dudarev { u: 0.25 };
const QUARTIC: Functional = Functional::Quartic {
    u: 0.30,
    j: 0.04,
    a: 0.35,
    b: -0.12,
    q: 0.2,
};

/// 0 two-site Dudarev, 1 two-site quartic, 2 three-site Dudarev, 3 three-site quartic,
/// 4 two-site without +U. Each is converged and differenced once for the whole file.
fn scenario(index: usize) -> &'static Scenario {
    static CELLS: [OnceLock<Scenario>; 5] = [const { OnceLock::new() }; 5];
    CELLS[index].get_or_init(|| {
        let (toy, functional) = match index {
            0 => (Toy::two_site(), DUDAREV),
            1 => (Toy::two_site(), QUARTIC),
            2 => (Toy::three_site(), DUDAREV),
            3 => (Toy::three_site(), QUARTIC),
            _ => (Toy::two_site(), Functional::None),
        };
        let state = solve(&toy, &functional, None);
        let forces = finite_forces(&toy, &functional, &state);
        let strain = finite_strain(&toy, &functional, &state);
        Scenario {
            toy,
            state,
            forces,
            strain,
        }
    })
}

fn deviation(
    forces: &[[f64; 3]],
    strain: &[[f64; 3]; 3],
    s: &Scenario,
    layer: &Gradient,
    bare: bool,
) -> f64 {
    let (grad, plain_strain) = plain(&s.toy, &s.state, bare);
    let mut worst: f64 = 0.0;
    for x in 0..s.toy.atoms() {
        for a in 0..3 {
            // The layer returns a force; the finite difference and the plain part are gradients.
            worst = worst.max((grad[x][a] - layer.forces[x][a] - forces[x][a]).abs());
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((plain_strain[a][b] + layer.strain[a][b] - strain[a][b]).abs());
        }
    }
    worst
}

fn against_reference(index: usize) -> (f64, Gradient) {
    let s = scenario(index);
    let layer = Packed::new(&s.toy, &s.state).run(None).unwrap();
    (deviation(&s.forces, &s.strain, s, &layer, false), layer)
}

fn largest(layer: &Gradient) -> f64 {
    layer
        .forces
        .iter()
        .flatten()
        .chain(layer.strain.iter().flatten())
        .fold(0.0, |m, x| m.max(x.abs()))
}

/// Same deviation with one input of the layer damaged.
fn damaged(index: usize, change: impl Fn(&mut Packed)) -> f64 {
    let s = scenario(index);
    let mut packed = Packed::new(&s.toy, &s.state);
    change(&mut packed);
    let layer = packed.run(None).unwrap();
    // A phantom atom may have been added: only the real atoms are compared.
    let real = Gradient {
        forces: layer.forces[..s.toy.atoms()].to_vec(),
        strain: layer.strain,
    };
    deviation(&s.forces, &s.strain, s, &real, false)
}

#[test]
fn the_standin_potentials_are_the_derivatives_of_the_energies() {
    let n = |x: f64, y: f64| {
        Matrix::from_row_slice(
            2,
            2,
            &[
                C64::new(x, 0.0),
                C64::new(0.1, y),
                C64::new(0.1, -y),
                C64::new(0.6 * x, 0.0),
            ],
        )
    };
    let (nu, nd) = (n(0.7, 0.05), n(0.4, -0.08));
    let (du, dd) = (n(0.3, -0.2), n(-0.2, 0.1));
    for f in [DUDAREV, QUARTIC] {
        let energy = |t: f64| {
            f.evaluate([
                &(&nu + &du * C64::new(t, 0.0)),
                &(&nd + &dd * C64::new(t, 0.0)),
            ])
            .energy
        };
        let exact = f.evaluate([&nu, &nd]);
        let predicted =
            (&exact.potential[0] * &du).trace().re + (&exact.potential[1] * &dd).trace().re;
        let h = 1e-2;
        let central = |h: f64| (energy(h) - energy(-h)) / (2.0 * h);
        let richardson = (4.0 * central(0.5 * h) - central(h)) / 3.0;
        assert!(
            (richardson - predicted).abs() < 1e-12,
            "{richardson} {predicted}"
        );
    }
}

#[test]
fn without_u_the_plain_toy_derivative_matches_finite_differences_and_the_layer_adds_nothing() {
    let (worst, layer) = against_reference(4);
    assert_eq!(largest(&layer), 0.0, "V = 0 must give a zero increment");
    eprintln!("without +U: deviation {worst:e}");
    assert!(worst < GATE, "plain part vs finite differences: {worst:e}");
}

#[test]
fn forces_and_strain_match_finite_differences_for_every_toy_and_functional() {
    for index in 0..4 {
        let (worst, layer) = against_reference(index);
        let s = scenario(index);
        // Not vacuous: the +U increment is far above the gate and above the plain-part error.
        assert!(
            largest(&layer) > 1e-3,
            "scenario {index}: increment {:e}",
            largest(&layer)
        );
        eprintln!(
            "scenario {index}: deviation {worst:e}, largest +U term {:e}",
            largest(&layer)
        );
        assert!(worst < GATE, "scenario {index}: {worst:e}");
        let omitted = Gradient {
            forces: vec![[0.0; 3]; s.toy.atoms()],
            strain: [[0.0; 3]; 3],
        };
        assert!(deviation(&s.forces, &s.strain, s, &omitted, false) > 1e-4 * largest(&layer));
    }
}

#[test]
fn the_forces_sum_to_zero() {
    for index in 0..4 {
        let s = scenario(index);
        let layer = Packed::new(&s.toy, &s.state).run(None).unwrap();
        let (grad, _) = plain(&s.toy, &s.state, false);
        for a in 0..3 {
            let increment: f64 = layer.forces.iter().map(|f| f[a]).sum();
            let total: f64 = (0..s.toy.atoms())
                .map(|x| layer.forces[x][a] - grad[x][a])
                .sum();
            eprintln!("scenario {index} axis {a}: +U sum {increment:e}, total sum {total:e}");
            assert!(
                increment.abs() < SUM_RULE,
                "scenario {index}: +U sum {increment:e}"
            );
            assert!(
                total.abs() < SUM_RULE,
                "scenario {index}: total sum {total:e}"
            );
        }
    }
}

#[test]
fn control_omitting_the_projector_position_term_fails() {
    for index in [1, 3] {
        // The projectors are owned by a phantom atom, so they never move relative to the basis.
        let worst = damaged(index, |p| {
            p.projector_owner = vec![p.atoms; p.projector_owner.len()];
            p.atoms += 1;
        });
        eprintln!("control scenario {index}: deviation {worst:e}");
        assert!(worst > 1e-4, "scenario {index}: {worst:e}");
    }
}

fn without_image(g: &Geometry, _image: usize, mu: usize, p: usize) -> [f64; 3] {
    std::array::from_fn(|b| g.basis_centers[mu][b] - g.projector_centers[p][b])
}

#[test]
fn control_dropping_r_from_the_displacement_fails() {
    for index in [0, 3] {
        let s = scenario(index);
        let layer = Packed::new(&s.toy, &s.state)
            .run(Some(without_image))
            .unwrap();
        let worst = deviation(&s.forces, &s.strain, s, &layer, false);
        eprintln!("control scenario {index}: deviation {worst:e}");
        assert!(worst > 1e-4, "scenario {index}: {worst:e}");
    }
}

#[test]
fn control_scaling_v_by_one_percent_fails() {
    for index in [0, 3] {
        let worst = damaged(index, |p| {
            for v in p.potentials.iter_mut().flatten() {
                *v *= C64::new(1.01, 0.0);
            }
        });
        eprintln!("control scenario {index}: deviation {worst:e}");
        assert!(worst > 1e-5, "scenario {index}: {worst:e}");
    }
}

#[test]
fn control_building_w_from_the_bare_hamiltonian_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let layer = Packed::new(&s.toy, &s.state).run(None).unwrap();
        let worst = deviation(&s.forces, &s.strain, s, &layer, true);
        eprintln!("control scenario {index}: deviation {worst:e}");
        assert!(worst > 1e-4, "scenario {index}: {worst:e}");
    }
}

#[test]
fn control_dropping_the_double_counting_shift_from_the_operator_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let worst = damaged(index, |p| {
            for (v, dc) in p
                .potentials
                .iter_mut()
                .flatten()
                .zip(s.state.double_counting.iter().flatten())
            {
                *v += dc;
            }
        });
        eprintln!("control scenario {index}: deviation {worst:e}");
        assert!(worst > 1e-4, "scenario {index}: {worst:e}");
    }
}

#[test]
fn inconsistent_input_raises() {
    let s = scenario(0);
    let cases: Vec<(&str, Box<dyn Fn(&mut Packed)>)> = vec![
        (
            "gradient images",
            Box::new(|p| {
                p.gradients.pop();
            }),
        ),
        ("owner out of range", Box::new(|p| p.basis_owner[0] = 99)),
        (
            "projection shape",
            Box::new(|p| p.projections[0] = Matrix::zeros(1, 1)),
        ),
        (
            "shared projector",
            Box::new(|p| p.shells[1].projectors[0] = 0),
        ),
        (
            "one density per k",
            Box::new(|p| {
                p.density[0].pop();
            }),
        ),
        ("nonfinite weight", Box::new(|p| p.weights[0] = f64::NAN)),
    ];
    for (name, change) in cases {
        let mut packed = Packed::new(&s.toy, &s.state);
        change(&mut packed);
        match packed.run(None) {
            Err(Error::Invalid(_)) => {}
            other => panic!(
                "{name}: expected Invalid, got {:?}",
                other.map(|g| g.strain)
            ),
        }
    }
}
