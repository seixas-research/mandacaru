//! What the spin-orbit tests share: the Dirac sample of the A0 freeze (the Dirac freeze of the
//! plan, section 3.4) and the generation of a frozen dataset through the legacy seam.
#![allow(dead_code)]

pub mod items;

use crate::paw_assemble_support::Frozen;
use crate::paw_generate_support::recipe_of;
use mandacaru_core::models::atom::Atom;
use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::paw::generate::recipe::Recipe;
use mandacaru_core::paw::generate::reference::solve_reference_with;
use mandacaru_core::paw::generate::{generate_with_atom, Generated};
use mandacaru_core::Context;

/// The light Dirac datasets of the freeze (Na and Si through 16,500 and 12,000 points, Fe).
pub const LIGHT: [&str; 3] = ["lda-fr_Na", "lda-fr_Fe", "pbe-fr_Si"];
/// The heavy ones: a lanthanide with a flagged scattering defect, a frozen-4f gold and lead.
pub const HEAVY: [&str; 3] = ["lda-fr_Ce", "lda-fr_Au", "lda-fr_Pb"];

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

/// One frozen Dirac dataset: its freeze, recipe, solved atom.
pub struct Case {
    pub frozen: Frozen,
    pub recipe: Recipe,
    pub atom: Atom,
}

pub fn open(ctx: &Context, name: &str) -> Case {
    let frozen = Frozen::open(name);
    let recipe = recipe_of(&frozen);
    let atom = solve_reference_with(ctx, &recipe, &AtomSolver).unwrap();
    Case {
        frozen,
        recipe,
        atom,
    }
}

impl Case {
    /// The dataset through the atom-solver radial scheme of the freeze.
    pub fn generate(&self, ctx: &Context) -> Generated {
        generate_with_atom(ctx, &self.recipe, &self.atom, &AtomSolver).unwrap()
    }
}

/// A generated Dirac dataset kept for the tests of one process (the light sample is generated
/// once, not once per test).
pub struct Loaded {
    pub ctx: Context,
    pub case: Case,
    pub generated: Generated,
}

pub fn loaded(name: &str) -> std::sync::Arc<Loaded> {
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::BTreeMap<String, Arc<Loaded>>>> =
        OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry(name.to_string())
        .or_insert_with(|| {
            let ctx = context();
            let case = open(&ctx, name);
            let generated = case.generate(&ctx);
            Arc::new(Loaded {
                ctx,
                case,
                generated,
            })
        })
        .clone()
}
