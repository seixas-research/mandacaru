//! The plain-DFT force engine (`periodic::gradient`, CF-3) on a toy with no crystal: the HU-10a
//! toy (`hubbard_gradient_support`) extended into a self-consistent functional with every term
//! of the crystal force (`periodic_gradient_support`): the grid overlap, kinetic, local and
//! short-range matrices, the projections with `C q C^+` and `C (D_ion + sum m_a B_a + O) C^+`,
//! the hat, ion, dense and core explicit terms, +U through the `Option` hook, fractional
//! occupations, and a restricted channel with degeneracy 2.
//!
//! Gates: the engine force equals the exact derivative of the converged toy energy (a two-level
//! Richardson central difference of reconverged energies, error `O(h^6)`, tolerance 1e-14 of the
//! SCF) to 1e-10; the translation sum rule holds to 1e-12 term by term; the force rotates with
//! the crystal. Negative controls must each miss that reference by more than 100 times the gate.
mod hubbard_gradient_support;
mod periodic_gradient_support;

use hubbard_gradient_support::{functional::Functional, Kind, Matrix, Toy};
use mandacaru_core::periodic::gradient::{contract, freeze, gradient, Breakdown, Frozen};
use mandacaru_core::Error;
use periodic_gradient_support::{
    c,
    derivatives::{ToyDerivatives, ToyHubbard},
    reference::{
        deviation, finite_gradient, hubbard_layer, hubbard_without_double_counting, largest,
        rotated, turn,
    },
    scf::{self, Solution, TOLERANCE},
    Model,
};
use std::sync::OnceLock;

const GATE: f64 = 1e-10;
const SUM_RULE: f64 = 1e-12;
const CONTROL: f64 = 100.0 * GATE;

const PRODUCTION: Functional = Functional::Production {
    u: 0.35,
    j: 0.05,
    l: 1,
};
const QUARTIC: Functional = Functional::Quartic {
    u: 0.30,
    j: 0.04,
    a: 0.35,
    b: -0.12,
    q: 0.2,
};

struct Scenario {
    model: Model,
    derivatives: ToyDerivatives,
    solution: Solution,
    /// `dE/dtau` of the reconverged energy.
    reference: Vec<[f64; 3]>,
}

/// 0 polarized two-site without +U, 1 two-site production Dudarev, 2 three-site quartic (the
/// spin-coupled `J != 0` stand-in), 3 restricted two-site production Dudarev, 4 two-site quartic
/// with fractional occupations. Each is converged and differenced once for the whole file.
fn scenario(index: usize) -> &'static Scenario {
    static CELLS: [OnceLock<Scenario>; 5] = [const { OnceLock::new() }; 5];
    CELLS[index].get_or_init(|| {
        let model = match index {
            0 => Model::polarized(Toy::two_site(), Functional::None),
            1 => Model::polarized(Toy::two_site(), PRODUCTION),
            2 => Model::polarized(Toy::three_site(), QUARTIC),
            3 => Model::restricted(Toy::two_site(), PRODUCTION),
            _ => Model::polarized(Toy::two_site(), QUARTIC).fractional(vec![1.0, 0.6]),
        };
        let derivatives = ToyDerivatives::new(&model);
        let solution = scf::solve(&derivatives.prep, None, TOLERANCE);
        let reference = finite_gradient(&model, &solution);
        Scenario {
            model,
            derivatives,
            solution,
            reference,
        }
    })
}

fn engine(s: &Scenario) -> Breakdown {
    let hook = ToyHubbard {
        prep: &s.derivatives.prep,
    };
    gradient(&s.derivatives, &s.solution.state, Some(&hook)).unwrap()
}

fn frozen(s: &Scenario) -> Frozen<scf::Fields> {
    let hook = ToyHubbard {
        prep: &s.derivatives.prep,
    };
    freeze(&s.derivatives, &s.solution.state, Some(&hook)).unwrap()
}

fn miss(s: &Scenario, gradient: &[[f64; 3]]) -> f64 {
    deviation(gradient, &s.reference)
}

fn axis_sum(term: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|a| term.iter().map(|g| g[a]).sum::<f64>().abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_engine_force_is_the_derivative_of_the_converged_energy_for_every_scenario() {
    for index in 0..5 {
        let s = scenario(index);
        let breakdown = engine(s);
        let worst = miss(s, &breakdown.gradient());
        eprintln!(
            "scenario {index}: deviation {worst:e}, largest force {:e}",
            largest(&s.reference)
        );
        assert!(worst < GATE, "scenario {index}: {worst:e}");
        // Not vacuous: every term of the contraction is far above the gate.
        for (name, term) in breakdown.terms() {
            let size = largest(term);
            eprintln!("  {name}: {size:e}");
            if name.starts_with("spin_orbit") || name.starts_with("exchange") {
                // A scalar semilocal request has neither a spin-orbit term
                // (periodic_gradient_spin_orbit_hook.rs) nor an exchange term
                // (periodic_gradient_hybrid.rs).
                assert_eq!(size, 0., "scenario {index}: {name}");
            } else {
                assert!(size > 1e-4, "scenario {index}: {name} is {size:e}");
            }
        }
    }
}

#[test]
fn the_matrix_terms_equal_the_toy_block_derivatives() {
    // The engine applies the ownership identity to the row derivatives `G`; the toy writes the
    // derivative of the same Bloch block directly (`bloch_atom`), with no mask.
    for index in [1, 3] {
        let s = scenario(index);
        let f = frozen(s);
        let (model, prep) = (&s.model, &s.derivatives.prep);
        let toy = &model.toy;
        let breakdown = engine(s);
        let ks = toy.kpoints();
        for x in 0..toy.atoms() {
            for a in 0..3 {
                let (mut overlap, mut kinetic, mut local, mut short, mut projection) =
                    (0.0, 0.0, 0.0, 0.0, 0.0);
                for (k, &w) in toy.weights.iter().enumerate() {
                    for sp in 0..model.spins() {
                        let bands = &f.state.bands[sp][k];
                        let (d, wt) = (bands.density(), bands.weighted(model.reference));
                        let tr =
                            |m: &Matrix, dm: &Matrix| (m * dm).trace().re * w * model.degeneracy;
                        let atom = |kind| toy.bloch_atom(kind, ks[k], x, a);
                        overlap -= tr(&wt, &atom(Kind::Overlap));
                        kinetic += tr(&d, &atom(Kind::Hopping));
                        local += tr(&d, &(atom(Kind::Local) * c(f.fields.v)));
                        short += tr(&d, &atom(Kind::Short));
                        let dc = atom(Kind::Projection);
                        let cc = &prep.projection[k];
                        let o = f.hubbard.as_ref().map(|o| &o[sp]);
                        let h = hamiltonian_operator(&s.derivatives, &f, o);
                        let by_h = &dc * &h * cc.adjoint();
                        let by_s = &dc * &prep.q_matrix * cc.adjoint();
                        projection +=
                            tr(&d, &(&by_h + by_h.adjoint())) - tr(&wt, &(&by_s + by_s.adjoint()));
                    }
                }
                let got = |t: &Vec<[f64; 3]>| t[x][a];
                let pairs = [
                    ("overlap", overlap, got(&breakdown.overlap)),
                    ("kinetic", kinetic, got(&breakdown.kinetic)),
                    ("local", local, got(&breakdown.local)),
                    ("short_range", short, got(&breakdown.short_range)),
                    (
                        "projection",
                        projection,
                        got(&breakdown.projection_rows) + got(&breakdown.projection_columns),
                    ),
                ];
                for (name, want, have) in pairs {
                    assert!((want - have).abs() < 1e-12, "{name}: {want} {have}");
                }
            }
        }
    }
}

fn hamiltonian_operator(d: &ToyDerivatives, f: &Frozen<scf::Fields>, o: Option<&Matrix>) -> Matrix {
    mandacaru_core::periodic::gradient::pulay::projector_operator(d, &f.fields, o)
}

#[test]
fn the_translation_sum_rule_holds_term_by_term() {
    for index in 0..5 {
        let breakdown = engine(scenario(index));
        // The rows and the columns of the projections sum to zero only together.
        for (name, term) in breakdown.terms() {
            if name.starts_with("projection") {
                continue;
            }
            let sum = axis_sum(term);
            eprintln!("scenario {index} {name}: sum over atoms {sum:e}");
            assert!(sum < SUM_RULE, "scenario {index} {name}: {sum:e}");
        }
        let rows_and_columns: Vec<[f64; 3]> = breakdown
            .projection_rows
            .iter()
            .zip(&breakdown.projection_columns)
            .map(|(r, c)| std::array::from_fn(|a| r[a] + c[a]))
            .collect();
        let sum = axis_sum(&rows_and_columns);
        eprintln!("scenario {index} projections: sum over atoms {sum:e}");
        assert!(sum < SUM_RULE, "scenario {index} projections: {sum:e}");
        assert!(axis_sum(&breakdown.gradient()) < SUM_RULE);
    }
}

#[test]
fn the_force_rotates_with_the_crystal() {
    let (sa, ca, sb, cb) = (0.6f64.sin(), 0.6f64.cos(), 0.9f64.sin(), 0.9f64.cos());
    let rotation = [
        [ca * cb, -sa, ca * sb],
        [sa * cb, ca, sa * sb],
        [-sb, 0.0, cb],
    ];
    for index in [1, 2] {
        let s = scenario(index);
        let model = rotated(&s.model, rotation);
        let d = ToyDerivatives::new(&model);
        let solution = scf::solve(&d.prep, None, TOLERANCE);
        let hook = ToyHubbard { prep: &d.prep };
        let rotated_force = gradient(&d, &solution.state, Some(&hook))
            .unwrap()
            .gradient();
        let expected: Vec<[f64; 3]> = engine(s)
            .gradient()
            .iter()
            .map(|&g| turn(rotation, g))
            .collect();
        let worst = deviation(&rotated_force, &expected);
        eprintln!("scenario {index}: rotation covariance {worst:e}");
        assert!(worst < GATE, "scenario {index}: {worst:e}");
    }
}

/// The +U increment of the engine hook equals the HU-10a layer on the same frozen state.
#[test]
fn the_hook_equals_the_hubbard_layer() {
    for index in [1, 2, 3] {
        let s = scenario(index);
        let mut f = frozen(s);
        let with = contract(&s.derivatives, &f).unwrap().gradient();
        f.hubbard.take().unwrap();
        let without = contract(&s.derivatives, &f).unwrap().gradient();
        let hook: Vec<[f64; 3]> = with
            .iter()
            .zip(&without)
            .map(|(a, b)| std::array::from_fn(|x| a[x] - b[x]))
            .collect();
        assert!(largest(&hook) > 1e-3, "scenario {index}: no +U increment");

        let layer_gradient = hubbard_layer(&s.derivatives.prep, &f.state);
        let worst = deviation(&hook, &layer_gradient);
        eprintln!("scenario {index}: hook against the layer {worst:e}");
        assert!(worst < SUM_RULE, "scenario {index}: {worst:e}");
    }
}

fn report(name: &str, index: usize, worst: f64) {
    eprintln!(
        "control {name}, scenario {index}: deviation {worst:e} = {:.1e} x gate",
        worst / GATE
    );
    assert!(
        worst > CONTROL,
        "control {name} scenario {index}: {worst:e}"
    );
}

#[test]
fn control_omitting_the_projector_moving_term_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let b = engine(s);
        let broken: Vec<[f64; 3]> = b
            .gradient()
            .iter()
            .zip(&b.projection_columns)
            .map(|(g, p)| std::array::from_fn(|a| g[a] - p[a]))
            .collect();
        report("projector-moving term omitted", index, miss(s, &broken));
    }
}

#[test]
fn control_flipping_the_sign_of_each_explicit_term_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let b = engine(s);
        for (name, term) in [
            ("E1 hats", &b.hats),
            ("E2 ion", &b.ion),
            ("E3 dense", &b.dense),
            ("E4 constants", &b.constants),
            ("E5 core", &b.core),
        ] {
            let flipped: Vec<[f64; 3]> = b
                .gradient()
                .iter()
                .zip(term)
                .map(|(g, t)| std::array::from_fn(|a| g[a] - 2.0 * t[a]))
                .collect();
            report(name, index, miss(s, &flipped));
        }
    }
}

#[test]
fn control_building_w_from_the_shifted_eigenvalues_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let mut d = ToyDerivatives::new(&s.model);
        d.layout.reference = 0.0;
        let hook = ToyHubbard { prep: &d.prep };
        let g = gradient(&d, &s.solution.state, Some(&hook))
            .unwrap()
            .gradient();
        report("W with the reference not removed", index, miss(s, &g));
    }
}

#[test]
fn control_a_wrong_degeneracy_fails() {
    for (index, wrong) in [(1, 2.0), (3, 1.0)] {
        let s = scenario(index);
        let mut d = ToyDerivatives::new(&s.model);
        d.layout.degeneracy = wrong;
        let hook = ToyHubbard { prep: &d.prep };
        let g = gradient(&d, &s.solution.state, Some(&hook))
            .unwrap()
            .gradient();
        report("wrong spin factor", index, miss(s, &g));
    }
}

#[test]
fn control_leaving_the_hubbard_operator_out_of_the_projector_terms_fails() {
    for index in [1, 2, 3] {
        let s = scenario(index);
        let mut f = frozen(s);
        f.hubbard = None;
        let g = contract(&s.derivatives, &f).unwrap().gradient();
        report("O not added to D_ion", index, miss(s, &g));
    }
}

#[test]
fn control_dropping_the_double_counting_from_the_hook_fails() {
    for index in [1, 3] {
        let s = scenario(index);
        let mut f = frozen(s);
        // V_Hub alone: the shift that the complete V subtracts is added back.
        let dropped = hubbard_without_double_counting(&s.derivatives.prep, &s.solution.state);
        f.hubbard = Some(dropped);
        let g = contract(&s.derivatives, &f).unwrap().gradient();
        report("V_dc dropped from O", index, miss(s, &g));
    }
}

/// Potentials, `q` and the +U operator of the INPUT density of a loosely converged iteration.
fn stale(s: &Scenario, loose: &Solution) -> Frozen<scf::Fields> {
    let prep = &s.derivatives.prep;
    Frozen {
        state: loose.state.clone(),
        fields: scf::fields(prep, loose.input.r, &loose.input.q),
        q: loose.input.q.iter().map(|&q| c(q)).collect(),
        hubbard: Some(scf::hubbard(prep, &loose.input.n).1),
    }
}

#[test]
fn control_potentials_from_the_input_density_fail() {
    for index in [1, 4] {
        let s = scenario(index);
        let hook = ToyHubbard {
            prep: &s.derivatives.prep,
        };
        for tolerance in [1e-3, 1e-5] {
            let loose = scf::solve(&s.derivatives.prep, None, tolerance);
            let input_route = contract(&s.derivatives, &stale(s, &loose))
                .unwrap()
                .gradient();
            let output_route = gradient(&s.derivatives, &loose.state, Some(&hook))
                .unwrap()
                .gradient();
            let (bad, good) = (miss(s, &input_route), miss(s, &output_route));
            eprintln!(
                "scenario {index}, SCF tolerance {tolerance:e}: input density {bad:e}, output density {good:e}"
            );
            if tolerance == 1e-3 {
                report("input density", index, bad);
                // The engine route must close the trap: with a freeze that kept the input
                // fields this would pass the control above and fail here.
                assert!(
                    good * 5. < bad,
                    "scenario {index}: output-density route {good:e} is not 5x below the input-density route {bad:e}"
                );
            }
        }
    }
}

#[test]
fn unsupported_requests_raise() {
    let s = scenario(0);
    let refusals: Vec<(&str, Box<dyn Fn(&mut ToyDerivatives)>)> = vec![
        (
            "hse06",
            Box::new(|d| d.layout.request.functional = "hse06".into()),
        ),
        (
            "r2scan",
            Box::new(|d| d.layout.request.functional = "r2scan".into()),
        ),
        (
            "rvv10",
            Box::new(|d| d.layout.request.functional = "pbe+rvv10".into()),
        ),
        (
            "float32",
            Box::new(|d| d.layout.request.single_precision = true),
        ),
    ];
    for (name, change) in refusals {
        let mut d = ToyDerivatives::new(&s.model);
        change(&mut d);
        match gradient(&d, &s.solution.state, None) {
            Err(Error::Unsupported(message)) => {
                eprintln!("{name}: {message}");
                // A hybrid request reaches the provider, which has no exchange to differentiate:
                // it raises (never a zero term).
                if name == "hse06" {
                    assert!(message.contains("screened exact exchange"), "{message}");
                }
            }
            other => panic!(
                "{name}: expected Unsupported, got {:?}",
                other.map(|b| b.overlap)
            ),
        }
    }
    let mut d = ToyDerivatives::new(&s.model);
    d.layout.request.functional = "pbe".into();
    assert!(gradient(&d, &s.solution.state, None).is_ok());
    d.layout.request.functional = "not-a-functional".into();
    assert!(matches!(
        gradient(&d, &s.solution.state, None),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn inconsistent_input_raises() {
    let s = scenario(1);
    let mut state = s.solution.state.clone();
    state.bands[0].pop();
    assert!(matches!(
        gradient(&s.derivatives, &state, None),
        Err(Error::Invalid(_))
    ));
    let mut state = s.solution.state.clone();
    state.bands[1][0].occupations.pop();
    assert!(matches!(
        gradient(&s.derivatives, &state, None),
        Err(Error::Invalid(_))
    ));
    let mut d = ToyDerivatives::new(&s.model);
    d.layout.row_owner[0] = 99;
    assert!(matches!(
        gradient(&d, &s.solution.state, None),
        Err(Error::Invalid(_))
    ));
}
