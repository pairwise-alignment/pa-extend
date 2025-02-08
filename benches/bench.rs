#![feature(portable_simd, array_chunks)]
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use std::{
    array,
    simd::{LaneCount, Simd, SimdInt, SupportedLaneCount},
    time::Duration,
};

use pa_extend::{parallel::ExtendFnParallel, *};
use pa_types::{Seq, I};

fn bench(c: &mut Criterion) {
    let len = 3000;
    for e in [1.0, 0.30, 0.10, 0.03, 0.01, 0.003, 0.001] {
        let c = &mut c.benchmark_group(format!("{e:.3}", e = e));

        let setup = |i| {
            if e == 1. {
                pa_generate::independent_seeded(len, i)
            } else {
                pa_generate::uniform_seeded(len, e, i)
            }
        };
        // // Make alternating offsets 0 and 1 to simulate unaligned reads.
        // let pairs: Vec<_> = (0..128).map(|i| (setup(i), i as usize % 2)).collect();

        let (a, b) = &setup(0);
        let indices = (0..128)
            .map(|_| {
                let i = 1 + rand::random::<u16>() as I % 10;
                let j = i - 1 + rand::random::<u16>() as I % 3;
                assert!(i >= 0);
                assert!(j >= 0);
                (i, j)
            })
            .collect::<Vec<_>>();

        let mut test = |name: &str, f: ExtendFn| {
            c.bench_function(name, |bb| {
                bb.iter(|| {
                    indices
                        .iter()
                        .map(|&(i, j)| f(&a[i as usize..], &b[j as usize..]))
                        .sum::<I>()
                });
            });
        };

        // test("zip_safe", zip);
        // test2("naive", naive);
        // test2("naive_fast", naive_fast);
        // test2("naive_unsafe", naive_unsafe);
        // test2("scalar", scalar);
        // test2("scalar_fast", scalar_fast);
        test("scalar", scalar_unsafe);
        // test2("u64", u64);
        // test("u64_xor", u64_unsafe);
        test("u64_eq", u64_unsafe_eq);
        // test("u64_eq_if0", u64_unsafe_eq_if0);
        // test("u64_eq_if1", u64_unsafe_eq_if1);
        // test2("s64", s64_unsafe);
        // test2("s64_nz", s64_unsafe_nz);
        // test2("s128", s128_unsafe);
        // test2("s128_nz", s128_unsafe_nz);
        test("s256_xor", s256_unsafe);
        test("s256_eq", s256_unsafe_eq);
        test("u64_then_s256", u64_then_s256);
        // test2("s256_unsafe_nz", s256_unsafe_nz);
        // test("edk", |a, b| editdistancek::mismatch(a, b) as I);

        fn testp<const K: usize>(
            c: &mut BenchmarkGroup<'_, WallTime>,
            a: Seq,
            b: Seq,
            indices: &Vec<(I, I)>,
            name: &str,
            f: ExtendFnParallel<K>,
        ) where
            LaneCount<K>: SupportedLaneCount,
        {
            c.bench_function(name, |bb| {
                bb.iter(|| {
                    indices
                        .array_chunks()
                        .map(|array: &[(I, I); K]| {
                            f(
                                a,
                                b,
                                Simd::from(array::from_fn(|k| array[k].0)),
                                Simd::from(array::from_fn(|k| array[k].1)),
                            )
                            .reduce_sum()
                        })
                        .sum::<I>()
                });
            });
        }

        testp(c, a, b, &indices, "u32_once", parallel::u32_unsafe_once);

        testp(c, a, b, &indices, "u16_once", parallel::u16_unsafe_once);

        testp(c, a, b, &indices, "u8_once", parallel::u8_unsafe_once);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_millis(1000)).warm_up_time(Duration::from_millis(100));
    targets = bench
);
criterion_main!(benches);
