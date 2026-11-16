//! The gate report: one [`Row`] per measurement, each with its value, its threshold and a
//! [`Status`]. A report never says "ok" for a check that was not made: a check that cannot be
//! evaluated is [`Status::Unevaluated`], and one that has nothing to compare with is
//! [`Status::NotApplicable`] with the reason in its detail.

/// The gates of one dataset file and what each one asks, in report order. G12 (a crystal) is not
/// a property of one file and is not here.
pub const GATES: [(&str, &str); 13] = [
    ("G1", "reload"),
    ("G2", "ghosts"),
    ("G3", "scattering"),
    ("G4", "states"),
    ("G5", "core"),
    ("G6", "conditioning"),
    ("G7", "hardness and grid"),
    ("G8", "sphere and partition"),
    ("G9", "smooth core"),
    ("G10", "transferability"),
    ("G11", "grid uniformity"),
    ("G13", "spin-orbit splitting"),
    ("G14", "ionic tail"),
];

/// What a row says about the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The value is inside its threshold.
    Pass,
    /// The value is outside its threshold, or not a number: the file is refused.
    Fail,
    /// A measurement that decides nothing by itself (no threshold) or that the plan records
    /// instead of refusing (a sphere wider than a bond, a working spacing past 0.15 Angstrom).
    Reported,
    /// The row fails and the file records that defect itself (`ghosts="flag"`): shown, not refused.
    Waived,
    /// There is nothing to measure or to compare with; the detail says why.
    NotApplicable,
    /// The check could not be evaluated (a level that is not finite, a solve that failed): a
    /// failure of the file, never a pass.
    Unevaluated,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "FAIL",
            Self::Reported => "reported",
            Self::Waived => "waived",
            Self::NotApplicable => "n/a",
            Self::Unevaluated => "UNEVALUATED",
        }
    }

    /// Whether a file with this row is refused.
    pub fn refuses(self) -> bool {
        matches!(self, Self::Fail | Self::Unevaluated)
    }
}

/// How the value is compared with the threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// `value <= threshold`.
    AtMost,
    /// `value < threshold`.
    Below,
    /// `value >= threshold`.
    AtLeast,
    /// No comparison: the row is a flag or a measurement.
    None,
}

impl Relation {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::AtMost => "<=",
            Self::Below => "<",
            Self::AtLeast => ">=",
            Self::None => "",
        }
    }
}

/// One measurement of one gate.
#[derive(Clone, Debug)]
pub struct Row {
    /// `"G1"` ... `"G13"`.
    pub gate: &'static str,
    /// What is measured, e.g. `"core density deviation"`.
    pub check: String,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
    pub relation: Relation,
    pub status: Status,
    /// The unit, the channel that sets the worst value, or why there is no comparison.
    pub detail: String,
}

impl Row {
    fn new(gate: &'static str, check: impl Into<String>, status: Status) -> Self {
        Self {
            gate,
            check: check.into(),
            value: None,
            threshold: None,
            relation: Relation::None,
            status,
            detail: String::new(),
        }
    }

    /// A value that must not exceed `threshold` (a NaN fails).
    pub fn at_most(
        gate: &'static str,
        check: impl Into<String>,
        value: f64,
        threshold: f64,
    ) -> Self {
        Self::compared(
            gate,
            check,
            value,
            threshold,
            Relation::AtMost,
            value <= threshold,
        )
    }

    /// A value that must stay strictly below `threshold` (a NaN fails).
    pub fn below(gate: &'static str, check: impl Into<String>, value: f64, threshold: f64) -> Self {
        Self::compared(
            gate,
            check,
            value,
            threshold,
            Relation::Below,
            value < threshold,
        )
    }

    /// A value that must reach `threshold` (a NaN fails).
    pub fn at_least(
        gate: &'static str,
        check: impl Into<String>,
        value: f64,
        threshold: f64,
    ) -> Self {
        Self::compared(
            gate,
            check,
            value,
            threshold,
            Relation::AtLeast,
            value >= threshold,
        )
    }

    fn compared(
        gate: &'static str,
        check: impl Into<String>,
        value: f64,
        threshold: f64,
        relation: Relation,
        inside: bool,
    ) -> Self {
        let mut row = Self::new(
            gate,
            check,
            if inside { Status::Pass } else { Status::Fail },
        );
        row.value = Some(value);
        row.threshold = Some(threshold);
        row.relation = relation;
        row
    }

    /// A yes/no fact with no number.
    pub fn flag(
        gate: &'static str,
        check: impl Into<String>,
        holds: bool,
        detail: impl Into<String>,
    ) -> Self {
        let mut row = Self::new(gate, check, if holds { Status::Pass } else { Status::Fail });
        row.detail = detail.into();
        row
    }

    /// A check that could not be measured and is recorded as such: [`Status::Reported`], with the
    /// reason, no number and no threshold.
    pub fn measured_without_value(
        gate: &'static str,
        check: impl Into<String>,
        why: impl Into<String>,
    ) -> Self {
        let mut row = Self::new(gate, check, Status::Reported);
        row.detail = why.into();
        row
    }

    /// A number with no threshold.
    pub fn measured(
        gate: &'static str,
        check: impl Into<String>,
        value: f64,
        detail: impl Into<String>,
    ) -> Self {
        let mut row = Self::new(gate, check, Status::Reported);
        row.value = Some(value);
        row.detail = detail.into();
        row
    }

    pub fn not_applicable(
        gate: &'static str,
        check: impl Into<String>,
        why: impl Into<String>,
    ) -> Self {
        let mut row = Self::new(gate, check, Status::NotApplicable);
        row.detail = why.into();
        row
    }

    pub fn unevaluated(
        gate: &'static str,
        check: impl Into<String>,
        why: impl Into<String>,
    ) -> Self {
        let mut row = Self::new(gate, check, Status::Unevaluated);
        row.detail = why.into();
        row
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// A failure that the plan records instead of refusing becomes [`Status::Reported`].
    pub fn advisory(mut self) -> Self {
        if self.status == Status::Fail {
            self.status = Status::Reported;
        }
        self
    }

    /// A failure the file records itself becomes [`Status::Waived`] when `recorded`.
    pub fn waived_if(mut self, recorded: bool) -> Self {
        if recorded && self.status == Status::Fail {
            self.status = Status::Waived;
        }
        self
    }

    /// The row as one line: `G5  pass         core density ...: 1.200e-9 <= 1.000e-8  detail`.
    pub fn line(&self) -> String {
        let number = |v: f64| format!("{v:.3e}");
        let comparison = match (self.value, self.threshold) {
            (Some(v), Some(t)) => {
                format!(": {} {} {}", number(v), self.relation.symbol(), number(t))
            }
            (Some(v), None) => format!(": {}", number(v)),
            _ => String::new(),
        };
        let mut line = format!(
            "{:<4}{:<13}{}{}",
            self.gate,
            self.status.name(),
            self.check,
            comparison
        );
        if !self.detail.is_empty() {
            line.push_str("  (");
            line.push_str(&self.detail);
            line.push(')');
        }
        line
    }
}

/// Every gate row of one dataset file.
#[derive(Clone, Debug, Default)]
pub struct GateReport {
    pub symbol: String,
    pub xc: String,
    pub relativity: String,
    pub rows: Vec<Row>,
}

impl GateReport {
    /// The rows that refuse the file.
    pub fn failures(&self) -> Vec<&Row> {
        self.rows.iter().filter(|r| r.status.refuses()).collect()
    }

    /// Whether no row refuses the file.
    pub fn passed(&self) -> bool {
        self.rows.iter().all(|r| !r.status.refuses())
    }

    /// The rows of one gate.
    pub fn gate(&self, gate: &str) -> Vec<&Row> {
        self.rows.iter().filter(|r| r.gate == gate).collect()
    }

    /// The worst status of a gate: a refusing row, else a reported or waived one, else a pass; a gate
    /// with only inapplicable rows is not applicable.
    pub fn gate_status(&self, gate: &str) -> Status {
        let rows = self.gate(gate);
        let any = |s: Status| rows.iter().any(|r| r.status == s);
        if any(Status::Unevaluated) {
            Status::Unevaluated
        } else if any(Status::Fail) {
            Status::Fail
        } else if any(Status::Waived) {
            Status::Waived
        } else if any(Status::Reported) {
            Status::Reported
        } else if any(Status::Pass) {
            Status::Pass
        } else {
            Status::NotApplicable
        }
    }

    /// One line per refusing row: the text of the error a refused build returns.
    pub fn refusal(&self) -> String {
        let mut text = format!(
            "{} ({} {}) fails {} gate row(s); not written:",
            self.symbol,
            self.xc,
            self.relativity,
            self.failures().len()
        );
        for row in self.failures() {
            text.push_str("\n  ");
            text.push_str(&row.line());
        }
        text
    }

    /// Every row, one per line.
    pub fn lines(&self) -> Vec<String> {
        self.rows.iter().map(Row::line).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_not_a_number_never_passes() {
        for row in [
            Row::at_most("G5", "x", f64::NAN, 1.),
            Row::below("G5", "x", f64::NAN, 1.),
            Row::at_least("G5", "x", f64::NAN, 1.),
        ] {
            assert_eq!(row.status, Status::Fail);
        }
        assert_eq!(Row::at_most("G5", "x", 1., 1.).status, Status::Pass);
        assert_eq!(Row::below("G5", "x", 1., 1.).status, Status::Fail);
        assert_eq!(Row::at_least("G5", "x", 1., 1.).status, Status::Pass);
    }

    #[test]
    fn only_a_failure_is_turned_into_a_report_or_a_waiver() {
        assert_eq!(
            Row::at_most("G8", "x", 2., 1.).advisory().status,
            Status::Reported
        );
        assert_eq!(
            Row::at_most("G8", "x", 0., 1.).advisory().status,
            Status::Pass
        );
        assert_eq!(
            Row::at_most("G2", "x", 2., 1.).waived_if(true).status,
            Status::Waived
        );
        assert_eq!(
            Row::at_most("G2", "x", 2., 1.).waived_if(false).status,
            Status::Fail
        );
        assert_eq!(
            Row::unevaluated("G2", "x", "why").waived_if(true).status,
            Status::Unevaluated
        );
    }
}
