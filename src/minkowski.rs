pub struct Minkowski<F> {
    pub p: F,
}

impl<F: Float + Sum> Minkowski<F> {
    pub fn new(p: F) -> Self {
        Minkowski { p }
    }

    pub fn distance(&self, a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs().powf(self.p))
            .sum::<F>()
            .powf(F::one() / self.p)
    }
}
