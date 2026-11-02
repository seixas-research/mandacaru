//! The comparison every stress gate uses: all nine `(a, b)` entries, unsymmetrized, against the
//! oracle, relative to the peak entry of the oracle (the "term maximum"); NaN fails; a term
//! whose peak is below `FLOOR` is vacuous and fails loudly; a negative control is a WRONG
//! variant compared with the same oracle and must miss by more than 100 gates.
#![allow(dead_code)]

pub type Tensor = [[f64; 3]; 3];

/// A term smaller than this (Hartree) cannot discriminate anything.
pub const FLOOR: f64 = 1e-6;

pub fn peak(t: &Tensor) -> f64 {
    t.iter().flatten().map(|x| x.abs()).fold(0., f64::max)
}

/// The worst `|got - want|` over the nine entries, relative to the peak of `want`.
pub fn miss(got: &Tensor, want: &Tensor) -> f64 {
    let scale = peak(want).max(FLOOR);
    let mut worst = 0f64;
    for a in 0..3 {
        for b in 0..3 {
            let d = (got[a][b] - want[a][b]).abs() / scale;
            if d.is_nan() {
                return f64::INFINITY;
            }
            worst = worst.max(d);
        }
    }
    worst
}

/// Asserts the analytic term against the oracle; returns the measured miss.
pub fn gate(name: &str, got: &Tensor, want: &Tensor, tolerance: f64) -> f64 {
    assert!(
        peak(want) > FLOOR,
        "{name}: the oracle term is vacuous (peak {:.2e})",
        peak(want)
    );
    let m = miss(got, want);
    println!(
        "  {name}: peak {:.3e}, miss {m:.2e} (gate {tolerance:.0e})",
        peak(want)
    );
    assert!(
        m < tolerance,
        "{name}: miss {m:.2e} exceeds {tolerance:.0e}\n got {got:?}\nwant {want:?}"
    );
    m
}

/// A wrong variant must miss the oracle by more than 100 gates; returns the bite in gates.
pub fn bites(name: &str, wrong: &Tensor, want: &Tensor, tolerance: f64) -> f64 {
    let bite = miss(wrong, want) / tolerance;
    println!("  control {name}: bites {bite:.2e} gates");
    assert!(bite > 100., "control {name} bites only {bite:.1} gates");
    bite
}

pub fn map(t: &Tensor, f: impl Fn(usize, usize, f64) -> f64) -> Tensor {
    std::array::from_fn(|a| std::array::from_fn(|b| f(a, b, t[a][b])))
}

pub fn transposed(t: &Tensor) -> Tensor {
    map(t, |a, b, _| t[b][a])
}

pub fn negated(t: &Tensor) -> Tensor {
    map(t, |_, _, x| -x)
}
