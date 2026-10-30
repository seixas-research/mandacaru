//! Bracketed scalar root finding (Brent's method).
use crate::{Error, Result};

/// Root of `f` in `[a, b]` by Brent's method with scipy's stopping rule,
/// `|x - x*| <= (xtol + rtol |x|) / 2`. The endpoints must bracket a sign change.
pub fn brent(
    mut f: impl FnMut(f64) -> Result<f64>,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    max_iterations: usize,
) -> Result<f64> {
    let (mut xpre, mut xcur) = (a, b);
    let (mut xblk, mut fblk) = (0., 0.);
    let (mut spre, mut scur) = (0., 0.);
    let mut fpre = f(xpre)?;
    let mut fcur = f(xcur)?;
    if fpre == 0. {
        return Ok(xpre);
    }
    if fcur == 0. {
        return Ok(xcur);
    }
    if fpre * fcur > 0. {
        return Err(Error::Invalid("the root is not bracketed".into()));
    }
    for _ in 0..max_iterations {
        if fpre != 0. && fcur != 0. && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.;
        let sbis = (xblk - xcur) / 2.;
        if fcur == 0. || sbis.abs() < delta {
            return Ok(xcur);
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2. * stry.abs() < spre.abs().min(3. * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        xcur += if scur.abs() > delta {
            scur
        } else if sbis > 0. {
            delta
        } else {
            -delta
        };
        fcur = f(xcur)?;
    }
    Err(Error::Invalid("Brent iteration did not converge".into()))
}
