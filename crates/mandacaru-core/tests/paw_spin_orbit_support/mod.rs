//! What the spin-orbit tests share: the Dirac sample, the production recipe of each member and
//! the generation of it.
#![allow(dead_code)]

pub mod items;

use mandacaru_core::models::atom::Atom;
use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::models::xc_scheme::Variational;
use mandacaru_core::paw::generate::recipe::Recipe;
use mandacaru_core::paw::generate::reference::solve_reference;
use mandacaru_core::paw::generate::search::{plan, radii_of, search_with_atom, Mode, Options};
use mandacaru_core::paw::generate::{generate_with_atom, Generated};
use mandacaru_core::Context;

/// The light Dirac datasets (Na and Si through 16,500 and 12,000 points, Fe), named
/// `<set>_<element>`.
pub const LIGHT: [&str; 3] = ["lda-fr_Na", "lda-fr_Fe", "pbe-fr_Si"];

pub fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

/// One Dirac dataset: its recipe and solved atom.
pub struct Case {
    pub recipe: Recipe,
    pub atom: Atom,
}

/// The production recipe of a Dirac dataset named `<functional>-fr_<element>`.
pub fn recipe_of(name: &str) -> Recipe {
    let (set, element) = name.split_once('_').expect("a name is <set>_<element>");
    let functional = set
        .strip_suffix("-fr")
        .expect("a Dirac set is <functional>-fr");
    Recipe::for_dataset(element, functional, Treatment::Dirac).unwrap()
}

/// The recipe the production search selects for a dataset: the first construction, or the
/// attempt of the production table the search settles on when the first one binds a ghost
/// (pbe-fr Si does).
pub fn open(ctx: &Context, name: &str) -> Case {
    let first = recipe_of(name);
    let atom = solve_reference(ctx, &first).unwrap();
    let options = Options::new(&first, Mode::Repair);
    let searched = search_with_atom(ctx, &first, &options, &atom, &Variational).unwrap();
    let recipe = match searched.index {
        0 => first,
        selected => {
            let built = generate_with_atom(ctx, &first, &atom, &Variational).unwrap();
            let attempts = plan(&first, &radii_of(&built.dataset));
            attempts[selected - 1].recipe.clone()
        }
    };
    Case { recipe, atom }
}

impl Case {
    /// The dataset through the production variational radial scheme.
    pub fn generate(&self, ctx: &Context) -> Generated {
        generate_with_atom(ctx, &self.recipe, &self.atom, &Variational).unwrap()
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
