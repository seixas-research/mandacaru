//! Tkatchenko-Scheffler (TS) van der Waals dispersion correction.
//!
//! Evaluates the pair-wise dispersion energy and analytic gradients according to
//! A. Tkatchenko and M. Scheffler, Phys. Rev. Lett. 102, 073005 (2009).
//! All internal calculations are performed in atomic units (Hartree, Bohr).

use crate::{
    finite,
    units::{BOHR_TO_ANGSTROM, HARTREE_TO_EV},
    Error, Geometry, Result,
};
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Atomic-unit conversion helper: 1 Angstrom in Bohr.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// Free-atom reference parameters for Tkatchenko-Scheffler dispersion.
#[derive(Debug, Clone, Copy)]
pub struct FreeAtomParams {
    pub z: u8,
    /// Dipole polarizability in a.u. (Bohr^3).
    pub alpha_0: f64,
    /// Homonuclear C6 dispersion coefficient in a.u. (Hartree * Bohr^6).
    pub c6: f64,
    /// Free-atom vdW radius in Bohr.
    pub r0_bohr: f64,
}

/// Retrieve free-atom reference parameters (polarizability, C6, and vdW radius).
///
/// Polarizabilities and C6 coefficients from X. Chu and A. Dalgarno,
/// J. Chem. Phys. 121, 4083 (2004) and V. G. Ruiz et al., Phys. Rev. Lett.
/// 108, 146103 (2012). vdW radii from S. Grimme, J. Comput. Chem. 27, 1787 (2006).
pub fn free_atom_params(z: u8) -> Result<FreeAtomParams> {
    let (alpha_0, c6, r0_angstrom) = match z {
        1 => (4.50, 6.50, 1.001),     // H
        2 => (1.38, 1.42, 1.012),     // He
        3 => (164.0, 1392.0, 0.825),  // Li
        4 => (38.0, 227.0, 1.408),    // Be
        5 => (21.0, 99.50, 1.485),    // B
        6 => (12.0, 46.60, 1.452),    // C
        7 => (7.40, 24.20, 1.397),    // N
        8 => (5.40, 15.60, 1.342),    // O
        9 => (3.80, 9.52, 1.287),     // F
        10 => (2.67, 6.20, 1.243),    // Ne
        11 => (163.0, 1518.0, 1.144), // Na
        12 => (71.0, 626.0, 1.364),   // Mg
        13 => (60.0, 528.0, 1.639),   // Al
        14 => (37.0, 305.0, 1.716),   // Si
        15 => (25.0, 185.0, 1.705),   // P
        16 => (19.6, 134.0, 1.683),   // S
        17 => (15.0, 94.60, 1.639),   // Cl
        18 => (11.1, 64.20, 1.595),   // Ar
        20 => (160.0, 2163.0, 1.474), // Ca
        21 => (120.0, 1383.0, 1.562), // Sc
        22 => (98.0, 1044.0, 1.562),  // Ti
        23 => (84.0, 832.0, 1.562),   // V
        24 => (78.0, 602.0, 1.562),   // Cr
        25 => (63.0, 552.0, 1.562),   // Mn
        26 => (56.0, 482.0, 1.562),   // Fe
        27 => (50.0, 408.0, 1.562),   // Co
        28 => (48.0, 373.0, 1.562),   // Ni
        29 => (42.0, 253.0, 1.562),   // Cu
        30 => (40.0, 284.0, 1.562),   // Zn
        33 => (29.0, 246.0, 1.760),   // As
        34 => (25.0, 210.0, 1.771),   // Se
        35 => (20.0, 162.0, 1.749),   // Br
        36 => (16.7, 130.0, 1.727),   // Kr
        38 => (199.0, 3175.0, 1.606), // Sr
        46 => (23.7, 158.0, 1.639),   // Pd
        47 => (50.6, 339.0, 1.639),   // Ag
        52 => (40.0, 445.0, 1.892),   // Te
        53 => (35.0, 385.0, 1.892),   // I
        78 => (39.7, 347.0, 1.750),   // Pt
        79 => (36.5, 298.0, 1.760),   // Au
        _ => {
            return Err(Error::Unsupported(format!(
                "element with atomic number Z={z} is not supported by the Tkatchenko-Scheffler reference database"
            )))
        }
    };
    Ok(FreeAtomParams {
        z,
        alpha_0,
        c6,
        r0_bohr: r0_angstrom * ANGSTROM_TO_BOHR,
    })
}

/// Optimal range parameter s_R for a given XC functional.
pub fn functional_s_r(xc: &str) -> Result<f64> {
    let name = xc.to_ascii_lowercase();
    match name.as_str() {
        "pbe" => Ok(0.940),
        "rpbe" => Ok(0.590),
        "revpbe" => Ok(0.585),
        "pbesol" => Ok(1.055),
        "blyp" => Ok(0.625),
        "am05" => Ok(0.840),
        "pw91" => Ok(0.965),
        "lda" => Ok(1.000),
        "r2scan" | "scan" => Ok(1.000),
        _ => Err(Error::Unsupported(format!(
            "Tkatchenko-Scheffler dispersion correction is not parametrized for XC functional {xc:?}"
        ))),
    }
}

/// Fermi-type damping factor and its derivative with respect to distance R (in Bohr).
///
/// f_damp(R) = 1 / (1 + exp(-d * (R / (sR * R0) - 1)))
#[inline]
pub fn fermi_damping(r: f64, r0_sum: f64, s_r: f64, d: f64) -> (f64, f64) {
    let scale = 1.0 / (s_r * r0_sum);
    let x = r * scale;
    let arg = -d * (x - 1.0);
    if arg > 80.0 {
        (0.0, 0.0)
    } else if arg < -80.0 {
        (1.0, 0.0)
    } else {
        let chi = arg.exp();
        let denom = 1.0 + chi;
        let f = 1.0 / denom;
        let f_prime = d * scale * chi / (denom * denom);
        (f, f_prime)
    }
}

/// Dispersion calculation result containing energy and forces in atomic units and eV/Angstrom.
#[derive(Debug, Clone)]
pub struct DispersionResult {
    /// Dispersion energy in Hartree.
    pub energy_ha: f64,
    /// Dispersion energy in eV.
    pub energy_ev: f64,
    /// Atomic forces in Hartree/Bohr (shape N x 3).
    pub forces_ha_bohr: Vec<[f64; 3]>,
    /// Atomic forces in eV/Angstrom (shape N x 3).
    pub forces_ev_angstrom: Vec<[f64; 3]>,
}

/// Evaluate Tkatchenko-Scheffler dispersion energy and analytic forces.
///
/// # Arguments
/// * `geometry` - Atomic coordinates and cell in Bohr.
/// * `functional` - XC functional name used for s_R parameter lookup.
/// * `volume_ratios` - Optional relative Hirshfeld volume ratios (v_i = V_i / V_i^free).
///   Defaults to 1.0 for free-atom limit.
/// * `s_r` - Optional explicit range parameter override.
/// * `r_max_angstrom` - Maximal cutoff radius for periodic calculations in Angstrom (default 10.0).
/// * `l_decay_angstrom` - Decay length for periodic cutoff smoothing in Angstrom (default 1.0).
pub fn tkatchenko_scheffler(
    geometry: &Geometry,
    functional: &str,
    volume_ratios: Option<&[f64]>,
    s_r: Option<f64>,
    r_max_angstrom: Option<f64>,
    l_decay_angstrom: Option<f64>,
) -> Result<DispersionResult> {
    let numbers = geometry.numbers();
    let n = numbers.len();
    if n == 0 {
        return Ok(DispersionResult {
            energy_ha: 0.0,
            energy_ev: 0.0,
            forces_ha_bohr: Vec::new(),
            forces_ev_angstrom: Vec::new(),
        });
    }

    if let Some(ratios) = volume_ratios {
        if ratios.len() != n {
            return Err(Error::Invalid(format!(
                "volume ratios length ({}) must match number of atoms ({n})",
                ratios.len()
            )));
        }
        finite(ratios, "volume ratios")?;
        if ratios.iter().any(|&v| v <= 0.0) {
            return Err(Error::Invalid("volume ratios must be positive".into()));
        }
    }

    let s_r = match s_r {
        Some(val) => {
            if !val.is_finite() || val <= 0.0 {
                return Err(Error::Invalid("s_r range parameter must be positive".into()));
            }
            val
        }
        None => functional_s_r(functional)?,
    };

    let d = 20.0;
    let r_max = r_max_angstrom.unwrap_or(10.0) * ANGSTROM_TO_BOHR;
    let l_decay = l_decay_angstrom.unwrap_or(1.0) * ANGSTROM_TO_BOHR;

    // Load free atom reference data and compute effective atom parameters
    let mut alpha_eff = Vec::with_capacity(n);
    let mut c6_eff = Vec::with_capacity(n);
    let mut r0_eff = Vec::with_capacity(n);

    for (i, &z) in numbers.iter().enumerate() {
        let p = free_atom_params(z)?;
        let v = volume_ratios.map_or(1.0, |r| r[i]);
        alpha_eff.push(p.alpha_0 * v);
        c6_eff.push(p.c6 * v * v);
        r0_eff.push(p.r0_bohr * v.cbrt());
    }

    // Pair coefficients matrix C6_AB
    let mut c6_pair = vec![vec![0.0; n]; n];
    for i in 0..n {
        c6_pair[i][i] = c6_eff[i];
        for j in (i + 1)..n {
            let num = 2.0 * c6_eff[i] * c6_eff[j];
            let denom = (alpha_eff[j] / alpha_eff[i]) * c6_eff[i]
                + (alpha_eff[i] / alpha_eff[j]) * c6_eff[j];
            let c = num / denom;
            c6_pair[i][j] = c;
            c6_pair[j][i] = c;
        }
    }

    let positions = geometry.positions();
    let pbc = geometry.pbc();
    let is_periodic = pbc.iter().any(|&p| p);

    let mut energy_ha = 0.0;
    let mut forces_ha_bohr = vec![[0.0; 3]; n];

    if !is_periodic {
        // Isolated molecule summation
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let dz = positions[j][2] - positions[i][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                if r <= 1e-10 {
                    return Err(Error::Invalid(
                        "coincident atom positions in dispersion calculation".into(),
                    ));
                }

                let r6 = r2 * r2 * r2;
                let c6 = c6_pair[i][j];
                let r0_sum = r0_eff[i] + r0_eff[j];
                let (f_damp, f_prime) = fermi_damping(r, r0_sum, s_r, d);

                // Energy: - f_damp * C6 / r^6
                energy_ha -= f_damp * c6 / r6;

                // dE/dr = - (f'_damp - 6 * f_damp / r) * C6 / r^6
                let de_dr = -(f_prime - 6.0 * f_damp / r) * c6 / r6;

                // Force on i: F_i = - dE/d(R_i) = de_dr * (r_j - r_i) / r
                let f_mag = de_dr / r;
                let fx = f_mag * dx;
                let fy = f_mag * dy;
                let fz = f_mag * dz;

                forces_ha_bohr[i][0] += fx;
                forces_ha_bohr[i][1] += fy;
                forces_ha_bohr[i][2] += fz;

                forces_ha_bohr[j][0] -= fx;
                forces_ha_bohr[j][1] -= fy;
                forces_ha_bohr[j][2] -= fz;
            }
        }
    } else {
        // Periodic lattice summation with smooth cutoff
        let cell = geometry.cell();
        let a1 = Vector3::new(cell[0][0], cell[0][1], cell[0][2]);
        let a2 = Vector3::new(cell[1][0], cell[1][1], cell[1][2]);
        let a3 = Vector3::new(cell[2][0], cell[2][1], cell[2][2]);

        let volume = (a1.cross(&a2)).dot(&a3).abs();
        if volume <= 1e-12 {
            return Err(Error::Invalid(
                "periodic cell volume must be positive in dispersion calculation".into(),
            ));
        }

        // Reciprocal normal vectors for bounding box calculation
        let b1 = a2.cross(&a3) / volume;
        let b2 = a3.cross(&a1) / volume;
        let b3 = a1.cross(&a2) / volume;

        let r_cut = r_max + 4.0 * l_decay;
        let n1_max = if pbc[0] {
            (r_cut * b1.norm()).ceil() as i32
        } else {
            0
        };
        let n2_max = if pbc[1] {
            (r_cut * b2.norm()).ceil() as i32
        } else {
            0
        };
        let n3_max = if pbc[2] {
            (r_cut * b3.norm()).ceil() as i32
        } else {
            0
        };

        for n1 in -n1_max..=n1_max {
            for n2 in -n2_max..=n2_max {
                for n3 in -n3_max..=n3_max {
                    let is_origin = n1 == 0 && n2 == 0 && n3 == 0;
                    let t = a1 * (n1 as f64) + a2 * (n2 as f64) + a3 * (n3 as f64);

                    for i in 0..n {
                        for j in 0..n {
                            if is_origin && i == j {
                                continue;
                            }

                            let dx = positions[j][0] + t.x - positions[i][0];
                            let dy = positions[j][1] + t.y - positions[i][1];
                            let dz = positions[j][2] + t.z - positions[i][2];
                            let r2 = dx * dx + dy * dy + dz * dz;
                            let r = r2.sqrt();

                            if r > r_cut || r <= 1e-10 {
                                continue;
                            }

                            let r6 = r2 * r2 * r2;
                            let c6 = c6_pair[i][j];
                            let r0_sum = r0_eff[i] + r0_eff[j];
                            let (f_damp, f_prime) = fermi_damping(r, r0_sum, s_r, d);

                            // Smooth cutoff factor
                            let y = (r - r_max) / l_decay;
                            let smooth = 0.5 * libm::erfc(y);
                            let smooth_der =
                                -1.0 / (PI.sqrt() * l_decay) * (-y * y).exp();

                            // Full pair contribution (counted twice in symmetric cell loop, so halved)
                            let pair_e = (f_damp * c6 / r6) * smooth;
                            energy_ha -= 0.5 * pair_e;

                            if i != j {
                                let de_dr = -(f_prime - 6.0 * f_damp / r) * (c6 / r6) * smooth
                                    - (f_damp * c6 / r6) * smooth_der;
                                let f_mag = de_dr / r;
                                let fx = 0.5 * f_mag * dx;
                                let fy = 0.5 * f_mag * dy;
                                let fz = 0.5 * f_mag * dz;

                                forces_ha_bohr[i][0] += fx;
                                forces_ha_bohr[i][1] += fy;
                                forces_ha_bohr[i][2] += fz;

                                forces_ha_bohr[j][0] -= fx;
                                forces_ha_bohr[j][1] -= fy;
                                forces_ha_bohr[j][2] -= fz;
                            }
                        }
                    }
                }
            }
        }
    }

    let energy_ev = energy_ha * HARTREE_TO_EV;
    let force_scale = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
    let mut forces_ev_angstrom = vec![[0.0; 3]; n];
    for i in 0..n {
        for a in 0..3 {
            forces_ev_angstrom[i][a] = forces_ha_bohr[i][a] * force_scale;
        }
    }

    Ok(DispersionResult {
        energy_ha,
        energy_ev,
        forces_ha_bohr,
        forces_ev_angstrom,
    })
}
