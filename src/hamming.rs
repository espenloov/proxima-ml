pub struct Hamming;

impl Hamming {
    pub fn distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
        assert_eq!(
            a.len(),
            b.len(),
            "proxima: slice lengths must match, got {} and {}",
            a.len(),
            b.len()
        );
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }
}
