use crate::Similarity;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Dot;

impl<F: Float + Sum> Similarity<F> for Dot {
    fn compute_similarity(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter().zip(b.iter()).map(|(x, y)| (*x) * (*y)).sum::<F>()
    }
}
