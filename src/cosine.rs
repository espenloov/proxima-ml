use crate::validate_lengths;
use crate::{Distance, Similarity};
use num_traits::Float;
use std::iter::Sum;

pub struct Cosine;

impl<F: Float + Sum> Similarity<F> for Cosine {
    fn compute_similarity(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);

        let dot: F = a.iter().zip(b.iter()).map(|(x, y)| (*x) * (*y)).sum();
        let mag_a: F = a.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        let mag_b: F = b.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();

        if mag_a == F::zero() || mag_b == F::zero() {
            return F::zero();
        }

        dot / (mag_a * mag_b)
    }
}

impl<F: Float + Sum> Distance<F> for Cosine {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        F::one() - Self::compute_similarity(a, b)
    }
}
