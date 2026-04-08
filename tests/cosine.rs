use approx::assert_abs_diff_eq;
use proxima_ml::{Cosine, DistanceExt, SimilarityExt};

#[test]
fn cosine_identical_vectors() {
    let a = [1.0, 0.0];
    let b = [1.0, 0.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0);
    assert_abs_diff_eq!(Cosine::distance(&a, &b), 0.0);
}

#[test]
fn cosine_perpendicular_vectors() {
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.0);
    assert_abs_diff_eq!(Cosine::distance(&a, &b), 1.0);
}

#[test]
fn cosine_same_direction_different_magnitude() {
    let a = [3.0, 1.0, 0.0];
    let b = [6.0, 2.0, 0.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0, epsilon = 1e-10);
}

#[test]
fn cosine_three_dimensions() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.974631, epsilon = 1e-6);
}

#[test]
fn cosine_zero_vector() {
    let a = [0.0, 0.0];
    let b = [1.0, 2.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.0);
}

#[test]
fn cosine_opposite_vectors() {
    let a = [1.0, 0.0];
    let b = [-1.0, 0.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), -1.0);
    assert_abs_diff_eq!(Cosine::distance(&a, &b), 2.0);
}

#[test]
#[should_panic(expected = "proxima: slice lengths must match")]
fn cosine_mismatched_lengths() {
    let a = [1.0, 2.0];
    let b = [1.0, 2.0, 3.0];
    Cosine::similarity(&a, &b);
}

#[test]
fn cosine_works_with_f32() {
    let a: [f32; 2] = [1.0, 0.0];
    let b: [f32; 2] = [1.0, 0.0];
    assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0_f32);
}

#[test]
fn cosine_batch_similarity() {
    let query = [1.0, 0.0];
    let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let similarities = Cosine::batch_similarity(&query, &targets);
    assert_abs_diff_eq!(similarities[0], 1.0);
    assert_abs_diff_eq!(similarities[1], 0.0);
    assert_abs_diff_eq!(similarities[2], 0.707107, epsilon = 1e-6);
}

#[test]
fn cosine_batch_distance() {
    let query = [1.0, 0.0];
    let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let distances = Cosine::batch_distance(&query, &targets);
    assert_abs_diff_eq!(distances[0], 0.0);
    assert_abs_diff_eq!(distances[1], 1.0);
}
