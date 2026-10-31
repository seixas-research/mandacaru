//! The `hubbard_u` option: what the user asked for, parsed and checked once.
//!
//! The option is a mapping from an element symbol to its correlated shell, written in eV:
//!
//! ```json
//! {"Ni": {"l": "d", "U": 6.0, "J": 0.0, "form": "dudarev-fll"}}
//! ```
//!
//! `l` (`"d"` or `"f"`) and `U` are required; `J` defaults to 0 and `form` to the one form
//! the engine implements. [`Request::parse`] reads that mapping as JSON text and resolves it
//! into [`Entry`] values: the shell as an angular momentum, a [`Form`], and `U` and `J` in
//! Hartree (the one place the unit changes, by division with
//! [`HARTREE_TO_EV`](crate::units::HARTREE_TO_EV)). A new functional form is one new [`Form`]
//! variant; a new key is one new field of [`Entry`]; the reading of the mapping does not change
//! for either. Every refusal is an [`Error`](crate::Error) whose message names `hubbard_u`, the
//! element and the reason.
//!
//! The text is the JSON a Python caller writes with `json.dumps`, so the non-finite numbers
//! `NaN`, `Infinity` and `-Infinity` that encoder emits are read as values and then refused as
//! "not a finite number", exactly like any other value that is not one.
use super::{
    elements::atomic_number,
    functional,
    hubbard::{shell_l, shell_letter, Form},
    invalid,
};
use crate::{units::HARTREE_TO_EV, Error, Result};
use serde_json::Value;
use std::borrow::Cow;

/// The shape of the option, quoted in the message for a mapping that is not one.
const EXAMPLE: &str = "{'Ni': {'l': 'd', 'U': 6.0, 'J': 0.0}}";
/// The keys a shell may carry.
const REQUIRED: [&str; 2] = ["l", "U"];
const OPTIONAL: [&str; 2] = ["J", "form"];

/// One element's correlated shell.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub symbol: String,
    /// Angular momentum of the shell: 2 for `d`, 3 for `f`.
    pub l: usize,
    pub form: Form,
    /// Hartree.
    pub u: f64,
    /// Hartree.
    pub j: f64,
}

impl Entry {
    /// `U - J` in Hartree.
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
    /// The shell letter, `'d'` or `'f'`.
    pub fn letter(&self) -> char {
        shell_letter(self.l).unwrap_or('?')
    }
}

/// The validated option: one entry per element, ordered by symbol.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub entries: Vec<Entry>,
}

/// Quotes the bare `NaN`, `Infinity` and `-Infinity` tokens outside strings so the reader sees
/// them as (non-numeric) values.
fn quote_non_finite(text: &str) -> Cow<'_, str> {
    const TOKENS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    if !TOKENS.iter().any(|t| text.contains(t)) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 8);
    let (mut inside, mut escaped) = (false, false);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if inside {
            out.push(c);
            (inside, escaped) = match (escaped, c) {
                (true, _) => (true, false),
                (false, '\\') => (true, true),
                (false, '"') => (false, false),
                _ => (true, false),
            };
            rest = &rest[c.len_utf8()..];
        } else if let Some(token) = TOKENS.iter().find(|t| rest.starts_with(**t)) {
            out.push('"');
            out.push_str(token);
            out.push('"');
            rest = &rest[token.len()..];
        } else {
            out.push(c);
            inside = c == '"';
            rest = &rest[c.len_utf8()..];
        }
    }
    Cow::Owned(out)
}

/// A shell energy in eV: a finite JSON number >= 0 (not a boolean, not a string).
fn energy(symbol: &str, name: &str, value: &Value) -> Result<f64> {
    value
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.)
        .ok_or_else(|| {
            invalid(format!(
                "hubbard_u['{symbol}']: {name} must be a finite number >= 0 in eV, got {value}"
            ))
        })
}

impl Entry {
    fn parse(symbol: &str, value: &Value) -> Result<Self> {
        if atomic_number(symbol).is_none() {
            return Err(invalid(format!(
                "hubbard_u: '{symbol}' is not an element symbol"
            )));
        }
        let keys = value.as_object().filter(|shell| {
            REQUIRED.iter().all(|k| shell.contains_key(*k))
                && shell
                    .keys()
                    .all(|k| REQUIRED.contains(&k.as_str()) || OPTIONAL.contains(&k.as_str()))
        });
        let Some(shell) = keys else {
            return Err(invalid(format!(
                "hubbard_u['{symbol}'] needs the keys 'l' and 'U' and may have 'J' and 'form', \
                 got {value}"
            )));
        };
        let l = shell["l"].as_str().and_then(shell_l).ok_or_else(|| {
            invalid(format!(
                "hubbard_u['{symbol}']: the shell 'l' must be 'd' or 'f', got {}",
                shell["l"]
            ))
        })?;
        let form = match shell.get("form") {
            None => Form::ALL[0],
            Some(name) => name
                .as_str()
                .ok_or_else(|| {
                    invalid(format!(
                        "hubbard_u['{symbol}']: form must be a string, got {name}"
                    ))
                })
                .and_then(Form::parse)
                .map_err(|e| invalid(format!("hubbard_u['{symbol}']: {e}")))?,
        };
        let u = energy(symbol, "U", &shell["U"])?;
        let j = match shell.get("J") {
            None => 0.,
            Some(j) => energy(symbol, "J", j)?,
        };
        if j > u {
            return Err(invalid(format!(
                "hubbard_u['{symbol}']: J = {j} eV exceeds U = {u} eV (U_eff = U - J would be \
                 negative)"
            )));
        }
        Ok(Self {
            symbol: symbol.to_string(),
            l,
            form,
            u: u / HARTREE_TO_EV,
            j: j / HARTREE_TO_EV,
        })
    }
}

impl Request {
    /// Reads and validates the option from its JSON text.
    pub fn parse(text: &str) -> Result<Self> {
        let mapping = serde_json::from_str::<Value>(&quote_non_finite(text))
            .map_err(|e| invalid(format!("hubbard_u is not valid JSON ({e}): {text}")))?;
        let Some(elements) = mapping.as_object().filter(|m| !m.is_empty()) else {
            return Err(invalid(format!(
                "hubbard_u must map an element symbol to its shell, for example {EXAMPLE} (eV), \
                 got {mapping}"
            )));
        };
        let entries = elements
            .iter()
            .map(|(symbol, shell)| Entry::parse(symbol, shell))
            .collect::<Result<Vec<_>>>()?;
        if let Some(other) = entries.iter().find(|e| e.form != entries[0].form) {
            return Err(invalid(format!(
                "hubbard_u: {} uses the form '{}' and {} uses '{}'; one form applies to the \
                 whole calculation",
                entries[0].symbol,
                entries[0].form.name(),
                other.symbol,
                other.form.name()
            )));
        }
        Ok(Self { entries })
    }

    /// Refuses the combinations the engine does not implement: a screened hybrid functional
    /// (the on-site correction has no exchange operator to share with it) and a precision other
    /// than float64 (unvalidated).
    pub fn require_scope(&self, xc: &str, precision: &str) -> Result<()> {
        if functional(xc).is_ok_and(|f| f.screened_exchange.is_some()) {
            return Err(Error::Unsupported(format!(
                "hubbard_u with the hybrid functional '{xc}': DFT+U is implemented for \
                 semilocal functionals only"
            )));
        }
        if precision != "float64" {
            return Err(invalid(
                "hubbard_u needs precision='float64' (single precision is unvalidated)",
            ));
        }
        Ok(())
    }

    /// The form of the calculation (the entries agree).
    pub fn form(&self) -> Form {
        self.entries[0].form
    }

    /// The entry for the element `symbol`.
    pub fn entry(&self, symbol: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.symbol == symbol)
    }
}
