# proxima-ml

Distance and similarity metrics for Rust.

Generic over `f32` and `f64`, with batch operations out of the box.

## Quick start
```toml
[dependencies]
proxima-ml = "0.3"
```
```rust
use proxima_ml::{Distance, Similarity, Euclidean, Cosine};

let a = &[1.0, 2.0, 3.0];
let b = &[4.0, 5.0, 6.0];

let dist = Euclidean::distance(a, b);
let sim = Cosine::similarity(a, b);
```

## Metrics

### Vector metrics

These work on ordered numerical vectors and support `f32`, `f64`, batch operations, and ndarray.

| Metric | Type | Description |
|--------|------|-------------|
| `Euclidean` | Distance | Straight-line distance between two points |
| `SqEuclidean` | Distance | Euclidean without the square root, faster when you only need to compare |
| `Manhattan` | Distance | Sum of absolute differences, robust in high dimensions |
| `Chebyshev` | Distance | Maximum difference in any single dimension |
| `Cosine` | Both | Measures direction similarity, ignores magnitude |
| `Dot` | Similarity | Dot product, core operation in neural networks |
| `Canberra` | Distance | Weighted Manhattan, sensitive to small changes near zero |

### Set metrics

These work on categorical or discrete data. Any type that supports equality comparison.

| Metric | Type | Description |
|--------|------|-------------|
| `Hamming` | Distance | Counts positions where two sequences differ |
| `Jaccard` | Both | Measures overlap between two sets |

## ndarray support

Enable the `ndarray` feature to pass `Array1` and `ArrayView1` directly to any vector metric:
```toml
[dependencies]
proxima-ml = { version = "0.3", features = ["ndarray"] }
```
```rust
use ndarray::Array1;
use proxima_ml::{Distance, Euclidean};

let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

let dist = Euclidean::distance(&a, &b);
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENCE-APACHE).
