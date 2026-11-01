//! Stand-in +U functionals for the toy: they take `(n_up, n_down)` of one shell and return the
//! energy and the potential `V = dE/dn = V_Hub - V_dc` per spin. They are not the production
//! forms: the force layer takes `V` as a matrix, so any smooth functional exercises it.
use super::{Matrix, NM};
use mandacaru_core::quantum::C64;

#[derive(Clone, Copy)]
pub enum Functional {
    /// No +U: `E = 0`, `V = 0`.
    None,
    /// Dudarev: `E = U/2 sum_s tr[n_s - n_s n_s]`.
    Dudarev { u: f64 },
    /// A spin-coupled quartic stand-in with `J != 0` and a double-counting shift:
    /// `E_Hub = A N_u N_d + B tr[n_u n_d] + (Q/4) sum tr[n^4]`,
    /// `E_dc = U/2 N (N - 1) - J/2 sum_s N_s (N_s - 1)`.
    Quartic {
        u: f64,
        j: f64,
        a: f64,
        b: f64,
        q: f64,
    },
}

pub struct Evaluation {
    pub energy: f64,
    /// The complete `V = V_Hub - V_dc`.
    pub potential: [Matrix; 2],
    /// The scalar shift `V_dc` (identity times a number) that `potential` already subtracts.
    pub double_counting: [Matrix; 2],
}

fn identity(scale: f64) -> Matrix {
    Matrix::identity(NM, NM) * C64::new(scale, 0.0)
}

fn real_trace(m: &Matrix) -> f64 {
    m.trace().re
}

impl Functional {
    pub fn evaluate(&self, n: [&Matrix; 2]) -> Evaluation {
        let zero = || [identity(0.0), identity(0.0)];
        match *self {
            Functional::None => Evaluation {
                energy: 0.0,
                potential: zero(),
                double_counting: zero(),
            },
            Functional::Dudarev { u } => {
                let mut energy = 0.0;
                for s in 0..2 {
                    energy += 0.5 * u * (real_trace(n[s]) - real_trace(&(n[s] * n[s])));
                }
                let v = |s: usize| identity(0.5 * u) - n[s] * C64::new(u, 0.0);
                Evaluation {
                    energy,
                    potential: [v(0), v(1)],
                    double_counting: [identity(-0.5 * u), identity(-0.5 * u)],
                }
            }
            Functional::Quartic { u, j, a, b, q } => {
                let (nu, nd) = (real_trace(n[0]), real_trace(n[1]));
                let nt = nu + nd;
                let cube = |m: &Matrix| m * m * m;
                let quartic = |m: &Matrix| real_trace(&(cube(m) * m));
                let hub = a * nu * nd
                    + b * real_trace(&(n[0] * n[1]))
                    + 0.25 * q * (quartic(n[0]) + quartic(n[1]));
                let dc = 0.5 * u * nt * (nt - 1.0) - 0.5 * j * (nu * (nu - 1.0) + nd * (nd - 1.0));
                let shift = |ns: f64| u * (nt - 0.5) - j * (ns - 0.5);
                let hub_v = [
                    identity(a * nd) + n[1] * C64::new(b, 0.0) + cube(n[0]) * C64::new(q, 0.0),
                    identity(a * nu) + n[0] * C64::new(b, 0.0) + cube(n[1]) * C64::new(q, 0.0),
                ];
                let dc_v = [identity(shift(nu)), identity(shift(nd))];
                Evaluation {
                    energy: hub - dc,
                    potential: [&hub_v[0] - &dc_v[0], &hub_v[1] - &dc_v[1]],
                    double_counting: dc_v,
                }
            }
        }
    }
}
