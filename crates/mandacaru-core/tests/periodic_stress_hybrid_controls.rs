//! AS-9 negative controls: every wrong variant of the exchange strain is built from the production
//! pieces (a term dropped or transposed, a force-style term added, a derivative of the data
//! replaced through the `Sources` protocol) and compared with the SAME central difference of the
//! production exchange energy as the right one; each must miss it by more than 100 gates, and
//! the branches a control relies on are asserted to have been taken. LiH at Gamma. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_hybrid_support;

use mandacaru_core::periodic::{
    hybrid_gradient::explicit,
    screened_exchange::ExxDivTreatment,
    stress_terms::hybrid::{Production, Sources, Terms},
};
use periodic_stress_hybrid_support::{wrong::*, *};

const STEP: f64 = 2.5e-6;
const GATE: f64 = 1e-8;

struct Oracle {
    rig: Rig,
    all: Tensor,
    gate: f64,
}

impl Oracle {
    fn new(exxdiv: ExxDivTreatment) -> Self {
        let mut rig = Rig::new("lih_hse06", exxdiv);
        let (all, floor) = rig.tensor(&ALL, STEP);
        let gate = GATE.max(100. * floor);
        Self { rig, all, gate }
    }

    /// The right variant meets the gate, a wrong one must miss it by more than 100 gates.
    fn bites(&self, name: &str, wrong: &Tensor) -> f64 {
        self.bites_over(name, wrong, 100.)
    }

    /// The same with a stated threshold (a control whose tensor is nearly symmetric bites less
    /// than 100 gates of the whole energy; it is named and bounded, not hidden).
    fn bites_over(&self, name: &str, wrong: &Tensor, over: f64) -> f64 {
        let bite = miss(wrong, &self.all) / self.gate;
        println!("  control {name}: bites {bite:.2e} gates");
        assert!(bite > over, "{name}: bites only {bite:.2} gates");
        bite
    }
}

fn without(terms: &Terms, drop: &Tensor) -> Tensor {
    std::array::from_fn(|a| std::array::from_fn(|b| terms.total()[a][b] - drop[a][b]))
}

fn plus(terms: &Terms, extra: &Tensor) -> Tensor {
    std::array::from_fn(|a| std::array::from_fn(|b| terms.total()[a][b] + extra[a][b]))
}

fn terms_with(oracle: &Oracle, sources: &dyn Sources) -> Terms {
    oracle.rig.analytic(sources)
}

fn controls(exxdiv: ExxDivTreatment, head: bool) {
    let oracle = Oracle::new(exxdiv);
    let terms = terms_with(&oracle, &Production);
    println!(
        "{}: the right variant misses by {:.2e} (gate {:.1e})",
        exxdiv.as_str(),
        miss(&terms.total(), &oracle.all),
        oracle.gate
    );
    assert!(miss(&terms.total(), &oracle.all) < oracle.gate);
    // Each term dropped.
    for (name, part) in terms.parts() {
        oracle.bites(&format!("without {name}"), &without(&terms, part));
    }
    // Index order: each term transposed. A tensor whose antisymmetric part is below the gate
    // cannot tell the order (the control is vacuous on this fixture): that is measured and
    // reported with the part's peak, never asserted as a bite. The order of the form factors is
    // bitten 1e8-fold by the array-level test of `periodic_stress_hybrid_shapes`, and that of the
    // Bloch tangents by `strain_samples`.
    for (name, part) in [
        ("shape_form", &terms.shape_form),
        ("bra", &terms.bra),
        ("ket", &terms.ket),
        ("compensation", &terms.compensation),
        ("sphere", &terms.sphere),
    ] {
        let wrong = plus(&terms, &{
            let t = transposed(part);
            std::array::from_fn(|a| std::array::from_fn(|b| t[a][b] - part[a][b]))
        });
        let bite = miss(&wrong, &oracle.all) / oracle.gate;
        println!(
            "  control {name} transposed: bites {bite:.2e} gates (peak {:.2e}, antisymmetric part {:.2e})",
            peak(part),
            miss(part, &transposed(part)) / 2.
        );
        if bite > 100. {
            continue;
        }
        assert!(
            miss(part, &transposed(part)) / 2. < 1e3 * oracle.gate || name == "shape_form",
            "{name}: neither biting nor symmetric"
        );
    }
    // Bra or ket slot alone.
    oracle.bites("the bra slot only", &without(&terms, &terms.ket));
    oracle.bites("the ket slot only", &without(&terms, &terms.bra));
    // The force-style phase: the atoms move with the strain and `Q` does not.
    {
        let data = &oracle.rig.data;
        let aug = data.crystal.hybrid_augmentation().unwrap();
        let gradient = explicit(
            &data.crystal,
            &oracle.rig.d,
            oracle.rig.screening,
            oracle.rig.exxdiv,
            &aug,
            None,
        )
        .unwrap();
        let mut phase = [[0.; 3]; 3];
        for (atom, derivative) in gradient.phase.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    phase[a][b] += derivative[a] * data.atoms[atom].center[b];
                }
            }
        }
        assert!(peak(&phase) > 0., "the phase term is empty");
        oracle.bites("the phase differentiated", &plus(&terms, &phase));
    }
    // Wrong derivatives of the data.
    let mean = terms_with(&oracle, &MeanVector);
    oracle.bites("the kernel from the mean vector", &mean.total());
    let zero = terms_with(&oracle, &ZeroPair);
    println!(
        "  (the kernel at k_diff = 0: Gamma-only, every pair has k_diff = 0, so it IS the right one: miss {:.2e})",
        miss(&zero.total(), &terms.total())
    );
    if head {
        let no_head = NoHead::default();
        let headless = terms_with(&oracle, &no_head);
        // The control is only a control if it removed a head entry.
        assert!(no_head.heads.get() > 0, "no head entry in any pair");
        let removed = miss(&headless.kernel, &terms.kernel);
        assert!(removed > 0., "the Gygi-Baldereschi head carries no strain");
        oracle.bites("the Gygi-Baldereschi head dropped", &headless.total());
    }
}

#[test]
fn every_wrong_exchange_strain_misses_the_difference_without_a_divergence_treatment() {
    controls(ExxDivTreatment::None, false);
}

#[test]
fn every_wrong_exchange_strain_misses_the_difference_with_gygi_baldereschi() {
    controls(ExxDivTreatment::GygiBaldereschi, true);
}
