use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Chebyshev {}

impl<F: Float + Sum> Distance<F> for Chebyshev {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .fold(F::zero(), F::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceExt;
    use approx::assert_abs_diff_eq;

    #[test]
    fn chebyshev_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 1.0, 7.0];
        assert_abs_diff_eq!(Chebyshev::distance(&a, &b), 4.0);
    }

    #[test]
    fn chebyshev_identical() {
        let a = [1.0, 2.0, 3.0];
        assert_abs_diff_eq!(Chebyshev::distance(&a, &a), 0.0);
    }

    #[test]
    fn chebyshev_single_element() {
        let a = [5.0];
        let b = [3.0];
        assert_abs_diff_eq!(Chebyshev::distance(&a, &b), 2.0);
    }

    #[test]
    fn chebyshev_negative_values() {
        let a = [-1.0, -5.0];
        let b = [1.0, 2.0];
        assert_abs_diff_eq!(Chebyshev::distance(&a, &b), 7.0);
    }

    #[test]
    fn chebyshev_always_lte_manhattan() {
        use crate::Manhattan;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let chebyshev = Chebyshev::distance(&a, &b);
        let manhattan = Manhattan::distance(&a, &b);
        assert!(chebyshev <= manhattan);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn chebyshev_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Chebyshev::distance(&a, &b);
    }

    #[test]
    fn chebyshev_works_with_f32() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 1.0, 7.0];
        assert_abs_diff_eq!(Chebyshev::distance(&a, &b), 4.0_f32);
    }
}
