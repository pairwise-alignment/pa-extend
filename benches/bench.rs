#![feature(portable_simd)]
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use pa_extend::*;
use pa_types::I;

fn bench(c: &mut Criterion) {
    let len = 3000;
    for e in [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002] {
        let c = &mut c.benchmark_group(format!("{e:.3}", e = e));

        let setup = |i| {
            if e == 1. {
                pa_generate::independent_seeded(len, i)
            } else {
                pa_generate::uniform_seeded(len, e, i)
            }
        };
        // Make alternating offsets 0 and 1 to simulate unaligned reads.
        let pairs: Vec<_> = (0..100).map(|i| (setup(i), i as usize % 2)).collect();

        let mut test2 = |name: &str, f: ExtendFn2| {
            c.bench_function(name, |bb| {
                bb.iter(|| {
                    pairs
                        .iter()
                        .map(|((a, b), i)| f(&a[*i..], &b[*i..]))
                        .sum::<I>()
                });
                // bb.iter_batched(setup, |(a, b)| f(&a, &b), criterion::BatchSize::SmallInput);
            });
        };

        test2("zip_safe", zip);
        // test2("naive", naive);
        // test2("naive_fast", naive_fast);
        // test2("naive_unsafe", naive_unsafe);
        // test2("scalar", scalar);
        // test2("scalar_fast", scalar_fast);
        test2("scalar", scalar_unsafe);
        // test2("u64", u64);
        test2("u64_xor", u64_unsafe);
        test2("u64_eq", u64_unsafe_eq);
        // test2("s64", s64_unsafe);
        // test2("s64_nz", s64_unsafe_nz);
        // test2("s128", s128_unsafe);
        // test2("s128_nz", s128_unsafe_nz);
        test2("s256_xor", s256_unsafe);
        test2("s256_eq", s256_unsafe_eq);
        // test2("s256_unsafe_nz", s256_unsafe_nz);
        // test2("edk", |a, b| editdistancek::mismatch(a, b) as I);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_millis(1000)).warm_up_time(Duration::from_millis(100));
    targets = bench
);
criterion_main!(benches);
