use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Manhattan;

impl<F: Float + Sum> Distance<F> for Manhattan {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .sum::<F>()
    }
}
