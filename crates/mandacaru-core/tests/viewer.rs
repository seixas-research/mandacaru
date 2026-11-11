//! What the 3D viewer computes before it plots (`math::viewer`): weights, isovalue, shading,
//! bonds, extent and camera path, each against a value worked out by hand.
use mandacaru_core::math::viewer::{self, Pose, Rotation};

#[test]
fn weights_are_the_amplitude_or_the_density_and_the_isovalue_takes_them_back() {
    assert_eq!(viewer::weights(&[-2., 3.], 2).unwrap(), vec![4., 9.]);
    assert_eq!(viewer::weights(&[-2., 3.], 1).unwrap(), vec![2., 3.]);
    assert!(viewer::weights(&[1.], 3).is_err());
    // Weights 4, 1, 1 (total 6): the level holding 2/3 of it is 4, so the isovalue of an
    // amplitude is 4 ** 0.5 = 2 and of a density 4.
    assert_eq!(viewer::isovalue(&[2., 1., -1.], 2, 0.6).unwrap(), 2.);
    assert_eq!(viewer::isovalue(&[4., 1., 1.], 1, 0.6).unwrap(), 4.);
}

#[test]
fn a_signed_surface_lists_the_positive_lobe_then_the_negative_one() {
    // A single cell whose corner 0 is +1 and corner 7 is -1 cuts one lobe of each sign.
    let mut values = vec![0.; 8];
    values[0] = 1.;
    values[7] = -1.;
    let axis = [0., 1.];
    let x: Vec<f64> = (0..8).map(|c| axis[c >> 2]).collect();
    let y: Vec<f64> = (0..8).map(|c| axis[(c >> 1) & 1]).collect();
    let z: Vec<f64> = (0..8).map(|c| axis[c & 1]).collect();
    let (triangles, signs) = viewer::signed_surface([2, 2, 2], &values, [&x, &y, &z], 0.5, true).unwrap();
    assert!(!triangles.is_empty());
    let positive = signs.iter().filter(|&&s| s == 1).count();
    assert_eq!(signs[..positive], vec![1; positive]);
    assert!(signs[positive..].iter().all(|&s| s == -1) && positive < signs.len());
    let (one_sided, signs) = viewer::signed_surface([2, 2, 2], &values, [&x, &y, &z], 0.5, false).unwrap();
    assert_eq!(one_sided.len(), positive);
    assert!(signs.iter().all(|&s| s == 1));
}

#[test]
fn the_cloud_is_brightest_where_the_weight_is_largest_and_colored_by_phase() {
    let (brightness, positive) = viewer::cloud_shading(&[-4., 1., 0.], 2).unwrap();
    assert_eq!(brightness, vec![1., (1f64 / 16.).sqrt(), 0.]);
    assert_eq!(positive, vec![false, true, true]);
    assert!(viewer::cloud_shading(&[], 1).is_err());
}

#[test]
fn a_surface_has_unit_normals_and_is_lit_on_both_sides() {
    let triangle = [[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]];
    assert_eq!(viewer::surface_normals(&[triangle]), vec![[0., 0., 1.]]);
    // Reversed winding flips the normal.
    let flipped = [triangle[0], triangle[2], triangle[1]];
    assert_eq!(viewer::surface_normals(&[flipped]), vec![[0., 0., -1.]]);
    // A degenerate triangle has the zero normal, not NaN.
    assert_eq!(viewer::surface_normals(&[[[1.; 3]; 3]]), vec![[0., 0., 0.]]);
    let base = viewer::surface_colors(&[1, -1], [0., 0.5, 1.], [1., 0.5, 0.]);
    assert_eq!(base, vec![[0., 0.5, 1.], [1., 0.5, 0.]]);
    // The headlight points along +x at elevation -25 and azimuth 30: a face turned to it is fully
    // lit, a face turned away from it keeps the 0.35 ambient share.
    let rgba = viewer::surface_rgba(
        &[[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.]],
        &[[1., 1., 1.]; 3],
        -25.,
        30.,
        0.8,
    )
    .unwrap();
    assert_eq!(rgba[0], [1., 1., 1., 0.8]);
    assert_eq!(rgba[1], rgba[0]);
    assert_eq!(rgba[2], [0.35, 0.35, 0.35, 0.8]);
    assert!(viewer::surface_rgba(&[[1., 0., 0.]], &[], 0., 0., 1.).is_err());
}

#[test]
fn nuclei_closer_than_one_and_a_fifth_covalent_radii_are_bonded() {
    let nuclei = [[0., 0., 0.], [0.74, 0., 0.], [5., 0., 0.]];
    assert_eq!(
        viewer::bonded_pairs(&nuclei, &[0.31, 0.31, 0.31]).unwrap(),
        vec![(0, 1)]
    );
    assert!(viewer::bonded_pairs(&nuclei, &[0.31]).is_err());
    assert_eq!(viewer::marker_sizes(&[1., 0.5]).unwrap(), vec![80., 35.]);
}

#[test]
fn the_view_is_fitted_to_the_nuclei_and_the_tail_of_what_is_shown() {
    assert_eq!(viewer::mean_point(&[[0., 0., 0.], [2., 4., 6.]]).unwrap(), [1., 2., 3.]);
    let nuclei = [[0., 0., 0.], [2., 0., 0.]];
    let shown = [[0., 0., 0.], [1., 3., 0.]];
    // The nuclei reach 1 + 0.5; the shown points' worst-axis distances are 1 and 3.
    assert_eq!(
        viewer::view_extent(&nuclei, [1., 0., 0.], &shown, 100.).unwrap(),
        1.05 * 3.
    );
    assert_eq!(
        viewer::view_extent(&nuclei, [1., 0., 0.], &shown, 0.).unwrap(),
        1.05 * 1.5
    );
    assert_eq!(
        viewer::view_extent(&nuclei, [1., 0., 0.], &[], 99.5).unwrap(),
        1.05 * 1.5
    );
    assert!(viewer::view_extent(&[], [0.; 3], &shown, 100.).is_err());
}

#[test]
fn keyframes_are_eased_between_and_geometric_for_a_zoom() {
    assert!((viewer::ease(0.) - 0.).abs() < 1e-16 && viewer::ease(1.) == 1.);
    // The ends of a segment are the keys and the midpoint is halfway.
    let v = viewer::keyframe_values(&[0., 10.], 1, &[0., 0.5, 1.], false).unwrap();
    assert_eq!(v[0], 0.);
    assert!((v[1] - 5.).abs() < 1e-12);
    assert_eq!(v[2], 10.);
    // A zoom from 1 to 4 meets 2 halfway, geometrically.
    let z = viewer::keyframe_values(&[1., 4.], 1, &[0.5], true).unwrap();
    assert!((z[0] - 2.).abs() < 1e-12);
    // Three keys cover two segments; one key repeats.
    let w = viewer::keyframe_values(&[0., 1., 0.], 1, &[0., 0.5, 1.], false).unwrap();
    assert!((w[0]).abs() < 1e-16 && (w[1] - 1.).abs() < 1e-12 && (w[2]).abs() < 1e-12);
    assert_eq!(
        viewer::keyframe_values(&[3., 4.], 2, &[0., 1.], false).unwrap(),
        vec![3., 4., 3., 4.]
    );
    assert!(viewer::keyframe_values(&[1., 2., 3.], 2, &[0.], false).is_err());
}

#[test]
fn a_camera_path_spins_by_whole_turns_and_multiplies_the_extra_zoom() {
    let pose = Pose {
        elevation: 15.,
        azimuth: -60.,
        zoom: 1.,
        focus: [0.; 3],
    };
    let path = viewer::camera_path(&[pose], 4, None, Rotation::Azimuth, 1.).unwrap();
    let azimuth: Vec<f64> = path.iter().map(|p| p.azimuth).collect();
    assert_eq!(azimuth, vec![-60., 30., 120., 210.]);
    assert!(path.iter().all(|p| p.elevation == 15. && p.zoom == 1.));
    let up = viewer::camera_path(&[pose], 2, None, Rotation::Elevation, 0.5).unwrap();
    assert_eq!((up[0].elevation, up[1].elevation), (15., 15. + 90.));
    // One frame is the first pose; the extra zoom (1, 4) ends at 4.
    let one = viewer::camera_path(&[pose], 1, None, Rotation::None, 1.).unwrap();
    assert_eq!(one, vec![pose]);
    let zoomed = viewer::camera_path(&[pose], 3, Some(&[1., 4.]), Rotation::None, 1.).unwrap();
    assert_eq!(zoomed[0].zoom, 1.);
    assert!((zoomed[1].zoom - 2.).abs() < 1e-12 && (zoomed[2].zoom - 4.).abs() < 1e-12);
    assert!(viewer::camera_path(&[], 3, None, Rotation::None, 1.).is_err());
    assert!(viewer::camera_path(&[pose], 0, None, Rotation::None, 1.).is_err());
}

#[test]
fn the_view_limits_and_the_frame_delay_follow_their_definitions() {
    assert_eq!(
        viewer::view_limits([1., 2., 3.], 4., 2.),
        [[-1., 3.], [0., 4.], [1., 5.]]
    );
    assert_eq!(viewer::frame_duration(12.).unwrap(), 83);
    // 1000 / 16 = 62.5 rounds to the even 62, not up; a fast playback is held at 20 ms.
    assert_eq!(viewer::frame_duration(16.).unwrap(), 62);
    assert_eq!(viewer::frame_duration(1000.).unwrap(), 20);
    assert!(viewer::frame_duration(0.).is_err() && viewer::frame_duration(f64::NAN).is_err());
}
