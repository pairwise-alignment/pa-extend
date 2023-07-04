#![feature(portable_simd, split_array)]
use std::{
    cmp::min,
    num::{NonZeroU16, NonZeroU32, NonZeroU8},
    simd::{LaneCount, Simd, SimdPartialEq, SupportedLaneCount, ToBitMask},
};

use pa_types::{Seq, I};

pub type ExtendFn = fn(Seq, Seq) -> I;

pub fn zip(a: Seq, b: Seq) -> I {
    a.iter().zip(b).take_while(|(ca, cb)| ca == cb).count() as _
}

fn eq(a: Seq, b: Seq, cnt: usize) -> bool {
    unsafe { a.get_unchecked(cnt) == b.get_unchecked(cnt) }
}

fn index_u64(a: Seq, cnt: usize) -> u64 {
    unsafe { *(a.as_ptr().offset(cnt as _) as *const u64) }
}

fn index_s<const L: usize>(a: Seq, cnt: usize) -> Simd<u8, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    unsafe { Simd::from_array(*(a.as_ptr().offset(cnt as _) as *const [u8; L])) }
}

pub fn naive(mut a: Seq, mut b: Seq) -> I {
    let mut cnt = 0;
    while !a.is_empty() && !b.is_empty() && eq(a, b, 0) {
        cnt += 1;
        a = &a[1..];
        b = &b[1..];
    }
    cnt
}

pub fn naive_fast(mut a: Seq, mut b: Seq) -> I {
    let mut cnt = 0;
    let len = min(a.len(), b.len());
    while cnt < len && eq(a, b, 0) {
        cnt += 1;
        a = &a[1..];
        b = &b[1..];
    }
    cnt as I
}

pub fn naive_unsafe(mut a: Seq, mut b: Seq) -> I {
    let mut cnt = 0;
    while eq(a, b, 0) {
        cnt += 1;
        a = &a[1..];
        b = &b[1..];
    }
    cnt
}

pub fn scalar(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    while cnt < a.len() && cnt < b.len() && eq(a, b, cnt) {
        cnt += 1;
    }
    cnt as I
}

pub fn scalar_fast(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    let len = min(a.len(), b.len());
    while cnt < len && eq(a, b, cnt) {
        cnt += 1;
    }
    cnt as I
}

pub fn scalar_unsafe(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    while eq(a, b, cnt) {
        cnt += 1;
    }
    cnt as I
}

pub fn u64(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    while cnt + 8 <= a.len() && cnt + 8 <= b.len() {
        let a = index_u64(a, cnt);
        let b = index_u64(b, cnt);
        let cmp = a ^ b;
        if cmp == 0 {
            cnt += 8;
        } else {
            return cnt as I + (cmp.leading_zeros() / u8::BITS) as I;
        };
    }
    cnt as I + zip(&a[cnt..], &b[cnt..])
}

pub fn u64_unsafe(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    loop {
        let a = index_u64(a, cnt);
        let b = index_u64(b, cnt);
        let cmp = a ^ b;
        if cmp == 0 {
            cnt += 8;
        } else {
            return cnt as I + (cmp.leading_zeros() / u8::BITS) as I;
        };
    }
}

pub fn u64_unsafe_eq(a: Seq, b: Seq) -> I {
    let mut cnt = 0;
    loop {
        let a = index_u64(a, cnt);
        let b = index_u64(b, cnt);
        if a == b {
            cnt += 8;
        } else {
            return cnt as I + ((a ^ b).leading_zeros() / u8::BITS) as I;
        };
    }
}

pub fn u64_unsafe_eq_if0(a: Seq, b: Seq) -> I {
    if a[0] != b[0] {
        return 0;
    }
    let mut cnt = 0;
    loop {
        let a = index_u64(a, cnt);
        let b = index_u64(b, cnt);
        if a == b {
            cnt += 8;
        } else {
            return cnt as I + ((a ^ b).leading_zeros() / u8::BITS) as I;
        };
    }
}

pub fn u64_unsafe_eq_if1(a: Seq, b: Seq) -> I {
    if a[0] != b[0] {
        return 0;
    }
    let mut cnt = 1;
    loop {
        let a = index_u64(a, cnt);
        let b = index_u64(b, cnt);
        if a == b {
            cnt += 8;
        } else {
            return cnt as I + ((a ^ b).leading_zeros() / u8::BITS) as I;
        };
    }
}

pub fn s256_unsafe_eq(a: Seq, b: Seq) -> I {
    const L: usize = 32;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        if a == b {
            cnt += L;
        } else {
            let mask = simd_a.simd_eq(simd_b);
            let eq = !mask.to_bitmask();
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                }) / u8::BITS as I;
        };
    }
}

pub fn s64_unsafe(a: Seq, b: Seq) -> I {
    const L: usize = 8;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if eq == 0 {
            cnt += L;
        } else {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        };
    }
}

pub fn s64_unsafe_nz(a: Seq, b: Seq) -> I {
    const L: usize = 8;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if let Some(eq) = NonZeroU8::new(eq) {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        } else {
            cnt += L;
        };
    }
}

pub fn s128_unsafe(a: Seq, b: Seq) -> I {
    const L: usize = 16;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if eq == 0 {
            cnt += L;
        } else {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        };
    }
}

pub fn s128_unsafe_nz(a: Seq, b: Seq) -> I {
    const L: usize = 16;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if let Some(eq) = NonZeroU16::new(eq) {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        } else {
            cnt += L;
        };
    }
}

pub fn s256_unsafe(a: Seq, b: Seq) -> I {
    const L: usize = 32;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if eq == 0 {
            cnt += L;
        } else {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        };
    }
}

pub fn s256_unsafe_nz(a: Seq, b: Seq) -> I {
    const L: usize = 32;
    let mut cnt = 0;
    loop {
        let simd_a = index_s::<L>(a, cnt);
        let simd_b = index_s::<L>(b, cnt);
        let mask = simd_a.simd_eq(simd_b);
        let eq = !mask.to_bitmask();
        if let Some(eq) = NonZeroU32::new(eq) {
            return cnt as I
                + (if cfg!(target_endian = "little") {
                    eq.leading_zeros() as I
                } else {
                    eq.trailing_zeros() as I
                });
        } else {
            cnt += L;
        };
    }
}

pub mod parallel {
    pub type ExtendFnParallel<const K: usize> = fn(Seq, Seq, Simd<I, K>, Simd<I, K>) -> Simd<I, K>;
    fn index<T: std::simd::SimdElement, const K: usize>(a: Seq, is: Simd<I, K>) -> Simd<T, K>
    where
        LaneCount<K>: SupportedLaneCount,
    {
        unsafe {
            Simd::from_array(array::from_fn(|k| {
                *(a.as_ptr().offset(is[k] as isize) as *const T)
            }))
        }
    }

    use super::*;
    use std::array;

    pub fn u32_unsafe_once_ss(a: Seq, b: Seq, is: Simd<I, 8>, js: Simd<I, 8>) -> Simd<I, 8> {
        let a = index::<u32, 8>(a, is);
        let b = index::<u32, 8>(b, js);
        let cmp = a ^ b;
        Simd::from(array::from_fn(|k| {
            cmp[k].leading_zeros() as I / u8::BITS as I
        }))
    }

    pub fn u16_unsafe_once_ss(a: Seq, b: Seq, is: Simd<I, 16>, js: Simd<I, 16>) -> Simd<I, 16> {
        let a = index::<u16, 16>(a, is);
        let b = index::<u16, 16>(b, js);
        let cmp = a ^ b;
        Simd::from(array::from_fn(|k| {
            cmp[k].leading_zeros() as I / u8::BITS as I
        }))
    }
}
