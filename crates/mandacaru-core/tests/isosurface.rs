//! MG-V1: the viewer's geometry kernels (`math::isosurface`) against the frozen outputs of the
//! Python kernels they replace (`test/migration/reference/viewer.json`, `freeze_viewer.py`): the
//! triangles of a signed field on a skewed grid, a sphere and an empty level, vertex by vertex in
//! the same order, and the enclosing levels exactly.
use mandacaru_core::math::isosurface::{enclosing_level, marching_tetrahedra};
use serde_json::Value;

fn frozen() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/viewer.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn the_triangles_are_the_frozen_python_ones() {
    let reference = frozen();
    for case in reference["surfaces"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let shape: Vec<usize> = case["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let (x, y, z) = (floats(&case["x"]), floats(&case["y"]), floats(&case["z"]));
        let triangles = marching_tetrahedra(
            [shape[0], shape[1], shape[2]],
            &floats(&case["values"]),
            [&x, &y, &z],
            case["level"].as_f64().unwrap(),
        )
        .unwrap();
        let want = floats(&case["triangles"]);
        assert_eq!(triangles.len(), case["count"].as_u64().unwrap() as usize, "{name}");
        let got: Vec<f64> = triangles.iter().flatten().flatten().copied().collect();
        let miss = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        println!("{name}: {} triangles, worst vertex miss {miss:.1e}", triangles.len());
        assert!(miss <= 1e-12, "{name}: {miss:e}");
    }
}

#[test]
fn the_enclosing_levels_are_the_frozen_python_ones() {
    let reference = frozen();
    let weights = floats(&reference["weights"]);
    for case in reference["levels"].as_array().unwrap() {
        let fraction = case["fraction"].as_f64().unwrap();
        assert_eq!(enclosing_level(&weights, fraction).unwrap(), case["level"].as_f64().unwrap(), "{fraction}");
    }
    for fraction in [0., 1., 1.5] {
        assert!(enclosing_level(&weights, fraction).unwrap_err().to_string().contains("enclosed"));
    }
    assert!(enclosing_level(&[0., 0.], 0.5).unwrap_err().to_string().contains("zero everywhere"));
}
