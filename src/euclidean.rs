use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Euclidean {}

pub struct SqEuclidean {}

impl<F: Float + Sum> Distance<F> for Euclidean {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
            .sqrt()
    }
}

impl<F: Float + Sum> Distance<F> for SqEuclidean {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
    }
}
