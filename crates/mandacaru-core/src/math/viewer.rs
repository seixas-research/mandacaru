//! What the 3D viewer computes before it plots: the weights and the isovalue of a field, the
//! brightness of a cloud, the normals and lighting of a surface, the bonds and marker sizes of the
//! nuclei, the extent of the view and the camera path of an animation (`utils/viewer.py`).
//!
//! Free of any plotting and of any physical model. Every operation keeps the order of the
//! expression it replaced, so the numbers the plot is built from do not change.
use super::{
    invalid, isosurface,
    numpy_rules::{linspace, maximum, percentile},
};
use crate::Result;
use std::f64::consts::PI;
use std::hint::black_box;

/// `|f|^power` of every value: the probability weight of an amplitude (`power = 2`) or of a
/// density (`power = 1`).
pub fn weights(data: &[f64], power: u32) -> Result<Vec<f64>> {
    match power {
        1 => Ok(data.iter().map(|v| v.abs()).collect()),
        2 => Ok(data.iter().map(|v| v.abs() * v.abs()).collect()),
        _ => Err(invalid("the weight exponent of a field is 1 or 2")),
    }
}

/// The isovalue whose inside holds `fraction` of the probability: the enclosing weight level of
/// the field's weights, taken back to the field's units as `level ** (1 / power)` (the C
/// library's `pow`, which is not always `sqrt`).
pub fn isovalue(data: &[f64], power: u32, fraction: f64) -> Result<f64> {
    let level = isosurface::enclosing_level(&weights(data, power)?, fraction)?;
    Ok(level.powf(black_box(1.0 / power as f64)))
}

/// `(positive lobes, negative lobes)` triangles of the surfaces `f = level` and (when `signed`)
/// `-f = level`, and the lobe sign (`1` or `-1`) of every triangle.
pub fn signed_surface(
    shape: [usize; 3],
    values: &[f64],
    coordinates: [&[f64]; 3],
    level: f64,
    signed: bool,
) -> Result<(Vec<isosurface::Triangle>, Vec<i64>)> {
    let mut triangles = isosurface::marching_tetrahedra(shape, values, coordinates, level)?;
    let mut signs = vec![1; triangles.len()];
    if signed {
        let negated: Vec<f64> = values.iter().map(|v| -v).collect();
        let lower = isosurface::marching_tetrahedra(shape, &negated, coordinates, level)?;
        signs.extend(std::iter::repeat_n(-1, lower.len()));
        triangles.extend(lower);
    }
    Ok((triangles, signs))
}

/// The brightness of every cloud point, `sqrt(w / max w)` for the weights `|f|^power`, and whether
/// its field value is non-negative (the phase that picks the color map).
pub fn cloud_shading(values: &[f64], power: u32) -> Result<(Vec<f64>, Vec<bool>)> {
    let w = weights(values, power)?;
    let largest = maximum(&w).ok_or_else(|| invalid("a cloud has at least one point"))?;
    Ok((
        w.iter().map(|v| (v / largest).sqrt()).collect(),
        values.iter().map(|&v| v >= 0.0).collect(),
    ))
}

/// The unit normal of every triangle, `(v1 - v0) x (v2 - v0) / max(|n|, 1e-300)`.
pub fn surface_normals(triangles: &[isosurface::Triangle]) -> Vec<[f64; 3]> {
    triangles
        .iter()
        .map(|t| {
            let a = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
            let b = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
            let n = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-300);
            [n[0] / norm, n[1] / norm, n[2] / norm]
        })
        .collect()
}

/// The flat color of every triangle: `positive` for a positive lobe (`sign > 0`), else `negative`.
pub fn surface_colors(signs: &[i64], positive: [f64; 3], negative: [f64; 3]) -> Vec<[f64; 3]> {
    signs.iter().map(|&s| if s > 0 { positive } else { negative }).collect()
}

/// The face colors of a surface lit by a headlight above and to the left of the eye:
/// `rgba = (base * shade, alpha)` with `shade = 0.35 + 0.65 |n . light|`, both sides of a face lit.
pub fn surface_rgba(
    normals: &[[f64; 3]],
    base: &[[f64; 3]],
    elevation: f64,
    azimuth: f64,
    alpha: f64,
) -> Result<Vec<[f64; 4]>> {
    if normals.len() != base.len() {
        return Err(invalid("a surface has one base color per normal"));
    }
    let (e, a) = ((elevation + 25.0).to_radians(), (azimuth - 30.0).to_radians());
    // The sine and the cosine of one angle are two calls of the C library: the compiler must not
    // fuse them into `sincos`, whose last bit differs.
    let (cos_e, cos_a) = (e.cos(), a.cos());
    let (sin_e, sin_a) = (black_box(e).sin(), black_box(a).sin());
    let light = [cos_e * cos_a, cos_e * sin_a, sin_e];
    Ok(normals
        .iter()
        .zip(base)
        .map(|(n, c)| {
            let shade = 0.35 + 0.65 * (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).abs();
            [c[0] * shade, c[1] * shade, c[2] * shade, alpha]
        })
        .collect())
}

/// The nuclei pairs `a < b` closer than `1.2 (r_a + r_b)`: the bonds.
pub fn bonded_pairs(nuclei: &[[f64; 3]], radii: &[f64]) -> Result<Vec<(usize, usize)>> {
    if nuclei.len() != radii.len() {
        return Err(invalid("a nucleus has one covalent radius"));
    }
    let mut pairs = Vec::new();
    for a in 0..nuclei.len() {
        for b in a + 1..nuclei.len() {
            let d: [f64; 3] = std::array::from_fn(|i| nuclei[a][i] - nuclei[b][i]);
            if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < 1.2 * (radii[a] + radii[b]) {
                pairs.push((a, b));
            }
        }
    }
    Ok(pairs)
}

/// The marker area of every nucleus, `60 (r / max r)^2 + 20`.
pub fn marker_sizes(radii: &[f64]) -> Result<Vec<f64>> {
    let largest = maximum(radii).ok_or_else(|| invalid("a picture has at least one nucleus"))?;
    Ok(radii
        .iter()
        .map(|r| 60.0 * ((r / largest) * (r / largest)) + 20.0)
        .collect())
}

/// `points.mean(axis=0)`: the rows added in order, then divided by their count.
pub fn mean_point(points: &[[f64; 3]]) -> Result<[f64; 3]> {
    if points.is_empty() {
        return Err(invalid("the mean of no points"));
    }
    let sum = points
        .iter()
        .fold([0.0; 3], |s, p| [s[0] + p[0], s[1] + p[1], s[2] + p[2]]);
    let n = points.len() as f64;
    Ok([sum[0] / n, sum[1] / n, sum[2] / n])
}

/// The half-width of the view at zoom one: `1.05` times the larger of the farthest nucleus from
/// `center` plus `0.5` and the `percentile` (in percent) of the shown points' distances along the
/// worst axis (a percentile, not the maximum, so a few far points do not shrink the picture).
pub fn view_extent(nuclei: &[[f64; 3]], center: [f64; 3], shown: &[[f64; 3]], percentile_q: f64) -> Result<f64> {
    let apart: Vec<f64> = nuclei
        .iter()
        .flat_map(|n| (0..3).map(move |i| (n[i] - center[i]).abs()))
        .collect();
    let mut reach = maximum(&apart).ok_or_else(|| invalid("a picture has at least one nucleus"))? + 0.5;
    if !shown.is_empty() {
        let worst: Vec<f64> = shown
            .iter()
            .map(|p| {
                maximum(&[
                    (p[0] - center[0]).abs(),
                    (p[1] - center[1]).abs(),
                    (p[2] - center[2]).abs(),
                ])
                .unwrap_or(0.0)
            })
            .collect();
        let tail = percentile(&worst, percentile_q)?;
        if tail > reach {
            reach = tail;
        }
    }
    Ok(1.05 * reach)
}

/// The axis limits `(center - half, center + half)` of the view at `zoom`, `half = extent / zoom`.
pub fn view_limits(center: [f64; 3], extent: f64, zoom: f64) -> [[f64; 2]; 3] {
    let half = extent / zoom;
    center.map(|c| [c - half, c + half])
}

/// The delay of one frame of an animation played at `fps`, in milliseconds: `1000 / fps` rounded
/// to the nearest integer (ties to even), at least 20 (the shortest delay a GIF viewer honors).
pub fn frame_duration(fps: f64) -> Result<u64> {
    if !(fps.is_finite() && fps > 0.) {
        return Err(invalid("fps must be finite and positive"));
    }
    Ok(((1000.0 / fps).round_ties_even() as u64).max(20))
}

/// Cosine ease-in-out on `[0, 1]`.
pub fn ease(t: f64) -> f64 {
    0.5 - 0.5 * (PI * t).cos()
}

/// Evenly spaced `keys` (`dim` values each) interpolated at `times` in `[0, 1]`, each segment
/// eased; `log` interpolates geometrically (the keys' logarithms). Row-major `(times, dim)`.
pub fn keyframe_values(keys: &[f64], dim: usize, times: &[f64], log: bool) -> Result<Vec<f64>> {
    if dim == 0 || keys.is_empty() || keys.len() % dim != 0 {
        return Err(invalid("keyframes are rows of equal length"));
    }
    let count = keys.len() / dim;
    let keys: Vec<f64> = if log {
        keys.iter().map(|k| k.ln()).collect()
    } else {
        keys.to_vec()
    };
    let mut out = Vec::with_capacity(times.len() * dim);
    for &t in times {
        let t = t.clamp(0.0, 1.0);
        if count == 1 {
            out.extend_from_slice(&keys);
            continue;
        }
        let position = t * (count - 1) as f64;
        let segment = (position as usize).min(count - 2);
        let local = ease(position - segment as f64);
        for d in 0..dim {
            let (from, to) = (keys[segment * dim + d], keys[(segment + 1) * dim + d]);
            out.push(from + local * (to - from));
        }
    }
    if log {
        out.iter_mut().for_each(|v| *v = v.exp());
    }
    Ok(out)
}

/// One camera pose: elevation and azimuth in degrees, the zoom and the point looked at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub elevation: f64,
    pub azimuth: f64,
    pub zoom: f64,
    pub focus: [f64; 3],
}

/// The camera spin superimposed on a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    None,
    Azimuth,
    Elevation,
}

/// The pose of every frame of an animation: the keyframes `poses` joined evenly and eased (zoom
/// geometrically), the zoom keyframes `zoom` (when given) multiplied into the zoom, and a spin of
/// `turns` turns on top, `360 turns i / frames` degrees at frame `i` (periodic, so a loop has no
/// seam).
pub fn camera_path(
    poses: &[Pose],
    frames: usize,
    zoom: Option<&[f64]>,
    rotation: Rotation,
    turns: f64,
) -> Result<Vec<Pose>> {
    if frames == 0 || poses.is_empty() {
        return Err(invalid("a camera path needs frames and at least one pose"));
    }
    let times = if frames > 1 {
        linspace(0.0, 1.0, frames)
    } else {
        vec![0.0]
    };
    let angles: Vec<f64> = poses.iter().flat_map(|p| [p.elevation, p.azimuth]).collect();
    let angles = keyframe_values(&angles, 2, &times, false)?;
    let zooms: Vec<f64> = poses.iter().map(|p| p.zoom).collect();
    let mut zooms = keyframe_values(&zooms, 1, &times, true)?;
    let focus: Vec<f64> = poses.iter().flat_map(|p| p.focus).collect();
    let focus = keyframe_values(&focus, 3, &times, false)?;
    if let Some(factors) = zoom {
        let factors = keyframe_values(factors, 1, &times, true)?;
        zooms.iter_mut().zip(&factors).for_each(|(z, f)| *z *= f);
    }
    Ok((0..frames)
        .map(|i| {
            let sweep = 360.0 * turns * i as f64 / frames as f64;
            let (mut elevation, mut azimuth) = (angles[2 * i], angles[2 * i + 1]);
            match rotation {
                Rotation::Elevation => elevation += sweep,
                Rotation::Azimuth => azimuth += sweep,
                Rotation::None => {}
            }
            Pose {
                elevation,
                azimuth,
                zoom: zooms[i],
                focus: [focus[3 * i], focus[3 * i + 1], focus[3 * i + 2]],
            }
        })
        .collect())
}
