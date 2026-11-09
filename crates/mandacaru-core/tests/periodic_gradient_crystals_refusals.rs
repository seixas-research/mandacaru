//! CF-4 refusals: what the crystal force engine does not differentiate raises with the cause,
//! taken from the crystal itself (a hybrid without its PAW spheres, meta-GGA, rVV10, a magnetic spinor hybrid,
//! float32, DFT+U, a k mesh reduced by space-group symmetry; collinear spin-polarized crystals are
//! differentiated since F1), the same
//! refusals on a real HYBRID crystal (HX-W lifts only the plain hybrid), and a geometry that
//! disagrees with the
//! SCF crystal it is tied to is refused (sizes, weights, reference, the cell, the atom centers,
//! the functional). Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::{
    math::storage::Precision,
    models::hubbard::Form,
    periodic::{
        gradient::{gradient, scf, Derivatives, Hubbard, State},
        hubbard::{Seed, Shell, Spec},
        setup::{provider::SetupDerivatives, pulay::Matrix, shapes::Sites},
    },
    Error,
};
use periodic_gradient_crystals_support::{altered::*, *};

fn cause(result: mandacaru_core::Result<()>) -> String {
    match result {
        Err(Error::Unsupported(message)) => message,
        other => panic!("expected an Unsupported refusal, got {other:?}"),
    }
}

#[test]
fn what_is_not_differentiated_raises_with_the_cause() {
    let mut data = Data::load("lih");
    scf::refuse(&data.crystal).expect("the plain LDA crystal is supported");
    let original = data.crystal.functional.clone();
    for (functional, word) in [("hse06", "PAW spheres")] {
        data.crystal.functional = functional.to_string();
        let message = cause(scf::refuse(&data.crystal));
        assert!(message.contains(word), "{functional}: {message}");
    }
    // A collinear meta-GGA (MG-F5) and rVV10 (MG-F6) are differentiated
    // (`periodic_gradient_xc.rs`): neither is a refusal.
    for functional in ["r2scan", "pbe+rvv10", "r2scan+rvv10"] {
        data.crystal.functional = functional.to_string();
        scf::refuse(&data.crystal).unwrap_or_else(|e| panic!("{functional}: {e}"));
    }
    data.crystal.functional = original;
    // A spinor LDA/PBE crystal is differentiated (CS-3): the flag alone is no longer a refusal.
    data.crystal.spinor = true;
    scf::refuse(&data.crystal).expect("a spinor LDA crystal is supported");
    data.crystal.spinor = false;
    data.crystal.precision = Precision::Float32;
    assert!(cause(scf::refuse(&data.crystal)).contains("float32"));
    data.crystal.precision = Precision::Float64;
    // Collinear spin polarization is differentiated (F1); a third spin channel does not exist.
    data.crystal.spins = 2;
    scf::refuse(&data.crystal).expect("a spin-polarized LDA crystal is supported");
    data.crystal.spins = 3;
    assert!(cause(scf::refuse(&data.crystal)).contains("spin channels"));
    data.crystal.spins = 1;
    // DFT+U: a shell with U - J != 0 makes the crystal refuse; U - J = 0 runs as plain DFT.
    let shell = |u: f64| Shell {
        atom: 0,
        symbol: "Li".into(),
        l: 0,
        u,
        j: 0.,
        projectors: vec![0],
        weights: vec![1.],
        basis: None,
        seed: Seed::Moment {
            n_electrons: 1.,
            moment: 0.,
        },
    };
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![shell(0.)],
    });
    scf::refuse(&data.crystal).expect("U - J = 0 on every shell is plain DFT");
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![shell(0.1)],
    });
    scf::refuse(&data.crystal).expect("DFT+U on a semilocal crystal is supported");
    data.crystal.spinor = true;
    assert!(cause(scf::refuse(&data.crystal)).contains("DFT+U"));
    data.crystal.spinor = false;
    data.crystal.hubbard = None;
    data.crystal.permutations = vec![vec![0]];
    assert!(cause(scf::refuse(&data.crystal)).contains("space-group"));
    data.crystal.permutations.clear();
    // A moment symmetrization that is not the identity is a reduced mesh as well.
    data.crystal.qsym.as_mut_slice()[1] = num_complex::Complex64::new(0.5, 0.);
    assert!(cause(scf::refuse(&data.crystal)).contains("space-group"));
    data.crystal.qsym.as_mut_slice()[1] = num_complex::Complex64::new(0., 0.);
    scf::refuse(&data.crystal).expect("restored");
}

/// A real hybrid crystal (HSE06 with its spheres) is differentiated, and the engine keeps
/// refusing everything else about it, each with its cause: a spinor, float32, DFT+U, a reduced k mesh; a meta-GGA or rVV10 functional.
#[test]
fn a_hybrid_crystal_is_differentiated_and_its_other_refusals_stand() {
    let mut data = Data::load("lih_hse06");
    scf::refuse(&data.crystal).expect("the hybrid crystal with its spheres is supported");
    // A spinor hybrid is differentiated, time-reversal symmetric (F2a) or magnetic (F2b).
    data.crystal.spinor = true;
    scf::refuse(&data.crystal).expect("a magnetic spinor hybrid is supported");
    data.crystal.time_reversal = true;
    scf::refuse(&data.crystal).expect("a time-reversal-symmetric spinor hybrid is supported");
    data.crystal.time_reversal = false;
    data.crystal.spinor = false;
    data.crystal.precision = Precision::Float32;
    assert!(cause(scf::refuse(&data.crystal)).contains("float32"));
    data.crystal.precision = Precision::Float64;
    // Collinear spin polarization of a hybrid is differentiated (F1); a third channel does not exist.
    data.crystal.spins = 2;
    scf::refuse(&data.crystal).expect("a spin-polarized hybrid crystal is supported");
    data.crystal.spins = 3;
    assert!(cause(scf::refuse(&data.crystal)).contains("spin channels"));
    data.crystal.spins = 1;
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![Shell {
            atom: 0,
            symbol: "Li".into(),
            l: 0,
            u: 0.1,
            j: 0.,
            projectors: vec![0],
            weights: vec![1.],
            basis: None,
            seed: Seed::Moment {
                n_electrons: 1.,
                moment: 0.,
            },
        }],
    });
    assert!(cause(scf::refuse(&data.crystal)).contains("DFT+U"));
    data.crystal.hubbard = None;
    data.crystal.permutations = vec![vec![0]];
    assert!(cause(scf::refuse(&data.crystal)).contains("space-group"));
    data.crystal.permutations.clear();
    for functional in ["r2scan", "pbe+rvv10"] {
        let original = std::mem::replace(&mut data.crystal.functional, functional.to_string());
        scf::refuse(&data.crystal).unwrap_or_else(|e| panic!("{functional}: {e}"));
        data.crystal.functional = original;
    }
    scf::refuse(&data.crystal).expect("restored");
}

/// The engine checks what the provider returns for a hybrid: an exchange derivative without the
/// basis-function terms, or with a response that does not match the layout, is refused; a
/// provider that carries no exchange at all raises (the default hook), never a zero.
#[test]
#[ignore = "slow: 12 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_crystals_refusals -- --ignored an_exchange_derivative_that_disagrees_with_the_layout_is_refused"]
fn an_exchange_derivative_that_disagrees_with_the_layout_is_refused() {
    let data = Data::load("lih_hse06");
    let cores = data.cores();
    let result = data.solve();
    let state = data.state(&result);
    for (wrong, word) in [
        (Wrong::NoFunctions, "basis-function"),
        (Wrong::ShortResponse, "disagrees"),
    ] {
        let altered = Altered::new(provider(&data, &cores), wrong);
        match gradient(&altered, &state, None) {
            Err(Error::Invalid(m)) => assert!(m.contains(word), "{wrong:?}: {m}"),
            other => panic!(
                "{wrong:?}: expected Invalid, got {:?}",
                other.map(|b| b.overlap)
            ),
        }
    }
    // A hybrid state asked of a provider tied to no crystal: no fields, no exchange.
    let bare = SetupDerivatives::new(
        &data.crystal.context,
        inputs(&data, &cores),
        data.layout(),
        data.operators(),
    )
    .unwrap();
    assert!(matches!(
        bare.exchange_gradient(&state),
        Err(Error::Unsupported(m)) if m.contains("with_scf")
    ));
}

#[test]
fn the_provider_refuses_a_foreign_spin_layout_and_a_foreign_crystal() {
    let data = Data::load("lih");
    let cores = data.cores();
    let ctx = &data.crystal.context;
    let mut layout = data.layout();
    layout.spins = 3;
    let refused = SetupDerivatives::new(ctx, inputs(&data, &cores), layout, data.operators());
    assert!(matches!(refused, Err(Error::Invalid(m)) if m.contains("spin channels")));
    // A polarized layout tied to a restricted crystal disagrees with it.
    let mut layout = data.layout();
    layout.spins = 2;
    layout.degeneracy = 1.;
    let polarized =
        SetupDerivatives::new(ctx, inputs(&data, &cores), layout, data.operators()).unwrap();
    assert!(matches!(
        polarized.with_scf(&data.crystal, data.options.clone()),
        Err(Error::Invalid(_))
    ));
    // Tied to a crystal whose weights differ, the geometry is refused.
    let mut layout = data.layout();
    layout.weights[0] += 0.125;
    let off = SetupDerivatives::new(ctx, inputs(&data, &cores), layout, data.operators()).unwrap();
    assert!(matches!(
        off.with_scf(&data.crystal, data.options.clone()),
        Err(Error::Invalid(_))
    ));
    let mut layout = data.layout();
    layout.reference += 0.01;
    let off = SetupDerivatives::new(ctx, inputs(&data, &cores), layout, data.operators()).unwrap();
    assert!(off.with_scf(&data.crystal, data.options.clone()).is_err());
    // Without the crystal the potentials of a density do not exist: fields raise, not guess.
    let bare =
        SetupDerivatives::new(ctx, inputs(&data, &cores), data.layout(), data.operators()).unwrap();
    let result = data.solve();
    let err = gradient(&bare, &data.state(&result), None).expect_err("no SCF crystal, no fields");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}

/// A +U hook handed to the engine reaches `bands(.., Some(operator))` of the provider, which
/// refuses: the provider differentiates no DFT+U crystal even when the caller supplies the hook.
struct ZeroHook(usize);

impl Hubbard for ZeroHook {
    fn operators(&self, _state: &State) -> mandacaru_core::Result<Vec<Matrix>> {
        Ok(vec![Matrix::zeros(self.0, self.0)])
    }
}

#[test]
fn a_hubbard_hook_is_refused_by_the_provider() {
    let data = Data::load("lih");
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let result = data.solve();
    let hook = ZeroHook(data.crystal.p);
    let err =
        gradient(&provider, &data.state(&result), Some(&hook)).expect_err("a +U hook is refused");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("DFT+U")),
        "{err:?}"
    );
}

fn tied(data: &Data, sites: Sites<'_>) -> mandacaru_core::Result<()> {
    let cores = data.cores();
    let mut inputs = inputs(data, &cores);
    inputs.sites = sites;
    SetupDerivatives::new(
        &data.crystal.context,
        inputs,
        data.layout(),
        data.operators(),
    )?
    .with_scf(&data.crystal, data.options.clone())
    .map(|_| ())
}

/// For a hybrid crystal the setup's moment channels must be the spheres' too: a channel with
/// another `(L, M)` label (still sorted and distinct) would put a compensation charge on another
/// harmonic, and nothing else in the tie notices (the ions do not depend on the channels).
#[test]
fn a_setup_with_other_moment_channels_is_refused_for_a_hybrid_crystal() {
    let data = Data::load("lih_hse06");
    tied(&data, data.sites()).expect("the crystal's own sites");
    let mut channels = data.channels.clone();
    let last = channels.len() - 1;
    channels[last].1 += 1;
    match tied(
        &data,
        Sites {
            channels: &channels,
            ..data.sites()
        },
    ) {
        Err(Error::Invalid(m)) => assert!(m.contains("PAW spheres"), "{m}"),
        other => panic!("expected Invalid (PAW spheres), got {other:?}"),
    }
}

/// The setup's cell and atoms must be the SCF crystal's: a transposed lattice (a cell that is
/// another lattice when transposed, hence LiH), a rescaled one, an atom moved by 0.01 Bohr, a
/// rigid translation of every atom (invisible to every lattice sum, visible to the grid) and an
/// ion with another charge are each refused, with the cause; the right sites are accepted.
#[test]
fn a_setup_of_another_cell_or_other_atoms_is_refused() {
    let data = Data::load("lih");
    tied(&data, data.sites()).expect("the crystal's own sites");
    let refused = |sites: Sites<'_>, word: &str| match tied(&data, sites) {
        Err(Error::Invalid(m)) => assert!(m.contains(word), "{m}"),
        other => panic!("expected Invalid ({word}), got {other:?}"),
    };
    refused(
        Sites {
            lattice: data.lattice.transpose(),
            ..data.sites()
        },
        "lattice",
    );
    refused(
        Sites {
            lattice: data.lattice * 1.001,
            ..data.sites()
        },
        "lattice",
    );
    let moved = |delta: [f64; 3], atoms: &[usize]| {
        let mut out = data.atoms.clone();
        for &a in atoms {
            for (x, d) in delta.iter().enumerate() {
                out[a].center[x] += d;
            }
        }
        out
    };
    let one = moved([0.01, 0., 0.], &[0]);
    refused(
        Sites {
            atoms: &one,
            ..data.sites()
        },
        "ion",
    );
    let shifted = moved([0.05, -0.03, 0.02], &[0, 1]);
    refused(
        Sites {
            atoms: &shifted,
            ..data.sites()
        },
        "ion",
    );
    let mut charged = data.atoms.clone();
    charged[1].charge += 0.01;
    refused(
        Sites {
            atoms: &charged,
            ..data.sites()
        },
        "ion",
    );
    // A functional that is not the crystal's.
    let mut layout = data.layout();
    layout.request.functional = "pbe".into();
    let cores = data.cores();
    let off = SetupDerivatives::new(
        &data.crystal.context,
        inputs(&data, &cores),
        layout,
        data.operators(),
    )
    .unwrap();
    assert!(matches!(
        off.with_scf(&data.crystal, data.options.clone()),
        Err(Error::Invalid(m)) if m.contains("functional")
    ));
}

/// The moment average of the SCF crystal acts on exactly its channels: a vector of another length
/// is a different crystal's moments and raises (it used to be left unaveraged without a word), and
/// the right length is averaged by the crystal's own `qsym`.
#[test]
fn moments_of_the_wrong_length_are_refused_by_the_symmetrization() {
    let data = Data::load("lih");
    let cores = data.cores();
    let provider = provider(&data, &cores);
    let channels = data.crystal.channels;
    assert!(channels >= 2);
    for length in [0, channels - 1, channels + 1] {
        let mut q = vec![num_complex::Complex64::new(1., 0.); length];
        match provider.symmetrize_moments(&mut q) {
            Err(Error::Invalid(m)) => assert!(
                m.contains("moment symmetrization") && m.contains(&channels.to_string()),
                "{m}"
            ),
            other => panic!("length {length}: expected Invalid, got {other:?}"),
        }
    }
    // The control: the right length is averaged by `qsym` (the identity of a whole mesh: unchanged).
    let mut q: Vec<_> = (0..channels)
        .map(|a| num_complex::Complex64::new(a as f64 + 1., 0.5))
        .collect();
    let before = q.clone();
    provider.symmetrize_moments(&mut q).unwrap();
    for (a, b) in q.iter().zip(&before) {
        assert!((a - b).norm() < 1e-12);
    }
}
