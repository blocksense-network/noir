use std::{fmt::Display, iter::{Product, Sum}, num::ParseIntError, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}, str::FromStr};

use ark_ff::{BigInt, BigInteger, FftField, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize, CanonicalSerializeWithFlags, Compress, Flags, SerializationError, Valid, Validate};
use num_bigint::BigUint;
use plonky2::field::{goldilocks_field::GoldilocksField as Plonky2GoldilocksField, types::{Field as Plonky2Field, PrimeField64}};
use rand::Rng;
use zeroize::Zeroize;

#[derive(Hash, Debug, Copy, Clone)]
pub struct GoldilocksField {
    data: u64
}

pub struct GoldilocksFieldIter {
}

impl Iterator for GoldilocksFieldIter {
    type Item = GoldilocksField;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl From<BigUint> for GoldilocksField {
    fn from(value: BigUint) -> Self {
        assert!(value < BigUint::from(GoldilocksField::MODULUS));
        GoldilocksField { data: value.try_into().unwrap() }
    }
}

impl From<BigInt<1>> for GoldilocksField {
    fn from(value: BigInt<1>) -> Self {
        assert!(BigInt::from(value) < GoldilocksField::MODULUS);
        // TODO: better impl
        let s = value.to_string();
        GoldilocksField { data: u64::from_str(s.as_str()).unwrap() }
    }
}

impl From<bool> for GoldilocksField {
    fn from(value: bool) -> Self {
        GoldilocksField { data: value.into() }
    }
}

impl From<u8> for GoldilocksField {
    fn from(value: u8) -> Self {
        GoldilocksField { data: value.into() }
    }
}

impl From<u16> for GoldilocksField {
    fn from(value: u16) -> Self {
        GoldilocksField { data: value.into() }
    }
}

impl From<u32> for GoldilocksField {
    fn from(value: u32) -> Self {
        GoldilocksField { data: value.into() }
    }
}

impl From<u64> for GoldilocksField {
    fn from(value: u64) -> Self {
        assert!(BigInt::from(value) < GoldilocksField::MODULUS);
        GoldilocksField { data: value }
    }
}

impl From<u128> for GoldilocksField {
    fn from(value: u128) -> Self {
        let u: u64 = value.try_into().unwrap();
        assert!(BigInt::from(u) < GoldilocksField::MODULUS);
        GoldilocksField { data: u }
    }
}

impl FromStr for GoldilocksField {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let a = u64::from_str(s)?;
        // TODO: range check
        Ok(GoldilocksField { data: a })
    }
}

impl FftField for GoldilocksField {
    // ???
    const GENERATOR: Self = GoldilocksField { data: 7 };

    // ???
    const TWO_ADICITY: u32 = 32;

    // ???
    const TWO_ADIC_ROOT_OF_UNITY: Self = GoldilocksField { data: 12345678 };
}

impl Field for GoldilocksField {
    // ???
    type BasePrimeField = Self;

    type BasePrimeFieldIter = GoldilocksFieldIter;

    // ???
    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = GoldilocksField { data: 0 };

    const ONE: Self = GoldilocksField { data: 1 };

    fn extension_degree() -> u64 {
        todo!()
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        todo!()
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        todo!()
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        todo!()
    }

    fn double(&self) -> Self {
        *self + *self
    }

    fn double_in_place(&mut self) -> &mut Self {
        *self += *self;
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        *self = - *self;
        self
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        todo!()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        *self * *self
    }

    fn square_in_place(&mut self) -> &mut Self {
        *self *= *self;
        self
    }

    fn inverse(&self) -> Option<Self> {
        if let Some(q) = Plonky2GoldilocksField::from_canonical_u64(self.data).try_inverse() {
            Some(GoldilocksField { data: q.to_canonical_u64() })
        } else {
            None
        }
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        if let Some(q) = Plonky2GoldilocksField::from_canonical_u64(self.data).try_inverse() {
            *self = GoldilocksField { data: q.to_canonical_u64() };
            Some(self)
        } else {
            None
        }
    }

    fn frobenius_map_in_place(&mut self, power: usize) {
        todo!()
    }
}

impl Product for GoldilocksField {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        todo!()
    }
}

impl<'a> Product<&'a GoldilocksField> for GoldilocksField {
    fn product<I: Iterator<Item = &'a GoldilocksField>>(iter: I) -> Self {
        todo!()
    }
}

impl Sum for GoldilocksField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        todo!()
    }
}

impl<'a> Sum<&'a GoldilocksField> for GoldilocksField {
    fn sum<I: Iterator<Item = &'a GoldilocksField>>(iter: I) -> Self {
        todo!()
    }
}

impl<'a> DivAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn div_assign(&mut self, rhs: &'a mut GoldilocksField) {
        *self = *self / rhs
    }
}

impl<'a> MulAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn mul_assign(&mut self, rhs: &'a mut GoldilocksField) {
        *self = *self * rhs
    }
}

impl<'a> SubAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn sub_assign(&mut self, rhs: &'a mut GoldilocksField) {
        *self = *self - rhs
    }
}

impl<'a> AddAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn add_assign(&mut self, rhs: &'a mut GoldilocksField) {
        *self = *self + rhs
    }
}

impl<'a> Div<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        self / *rhs
    }
}

impl<'a> Mul<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn mul(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        self * *rhs
    }
}

impl<'a> Sub<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        self - *rhs
    }
}

impl<'a> Add<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        self + *rhs
    }
}

impl<'a> DivAssign<&'a GoldilocksField> for GoldilocksField {
    fn div_assign(&mut self, rhs: &'a GoldilocksField) {
        *self = *self / rhs
    }
}

impl<'a> MulAssign<&'a GoldilocksField> for GoldilocksField {
    fn mul_assign(&mut self, rhs: &'a GoldilocksField) {
        *self = *self * rhs
    }
}

impl<'a> SubAssign<&'a GoldilocksField> for GoldilocksField {
    fn sub_assign(&mut self, rhs: &'a GoldilocksField) {
        *self = *self - rhs
    }
}

impl<'a> AddAssign<&'a GoldilocksField> for GoldilocksField {
    fn add_assign(&mut self, rhs: &'a GoldilocksField) {
        *self = *self + rhs
    }
}

impl<'a> Div<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: &'a GoldilocksField) -> Self::Output {
        self / *rhs
    }
}

impl<'a> Mul<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn mul(self, rhs: &'a GoldilocksField) -> Self::Output {
        self * *rhs
    }
}

impl<'a> Sub<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: &'a GoldilocksField) -> Self::Output {
        self - *rhs
    }
}

impl<'a> Add<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: &'a GoldilocksField) -> Self::Output {
        self + *rhs
    }
}

impl DivAssign for GoldilocksField {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl MulAssign for GoldilocksField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl SubAssign for GoldilocksField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl AddAssign for GoldilocksField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Div for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        GoldilocksField { data: (Plonky2GoldilocksField::from_canonical_u64(self.data) / Plonky2GoldilocksField::from_canonical_u64(rhs.data)).to_canonical_u64() }
    }
}

impl Sub for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.data >= rhs.data {
            GoldilocksField { data: self.data - rhs.data }
        } else {
            GoldilocksField { data: 18446744069414584321u64 - (rhs.data - self.data) }
        }
    }
}

impl Valid for GoldilocksField {
    fn check(&self) -> Result<(), SerializationError> {
        todo!()
    }
}

impl CanonicalSerialize for GoldilocksField {
    fn serialize_with_mode<W>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        todo!()
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        todo!()
    }
}

impl CanonicalSerializeWithFlags for GoldilocksField {
    fn serialize_with_flags<W, F: Flags>(
        &self,
        writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        todo!()
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        todo!()
    }
}

impl CanonicalDeserialize for GoldilocksField {
    fn deserialize_with_mode<R>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        todo!()
    }
}

impl CanonicalDeserializeWithFlags for GoldilocksField {
    fn deserialize_with_flags<R, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        todo!()
    }
}

impl Neg for GoldilocksField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        GoldilocksField { data: 0 } - self
    }
}

impl Ord for GoldilocksField {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

impl PartialOrd for GoldilocksField {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Zero for GoldilocksField {
    fn zero() -> Self {
        GoldilocksField { data: 0 }
    }

    fn is_zero(&self) -> bool {
        self.data == 0
    }
}

impl One for GoldilocksField {
    fn one() -> Self {
        GoldilocksField { data: 1 }
    }
}

impl Mul for GoldilocksField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        GoldilocksField { data: (((self.data as u128) * (rhs.data as u128)) % 18446744069414584321u128) as u64 }
    }
}

impl Add for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut s = (self.data as u128) + (rhs.data as u128);
        if s > 18446744069414584321 {
            s -= 18446744069414584321;
        }
        GoldilocksField { data: s as u64 }
    }
}

impl PartialEq for GoldilocksField {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for GoldilocksField {
}

impl Display for GoldilocksField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl From<GoldilocksField> for BigUint {
    fn from(value: GoldilocksField) -> Self {
        BigUint::from(value.data)
    }
}

impl From<GoldilocksField> for BigInt<1> {
    fn from(value: GoldilocksField) -> Self {
        BigInt::new([value.data])
    }
}

impl Zeroize for GoldilocksField {
    fn zeroize(&mut self) {
        self.data.zeroize()
    }
}

impl Default for GoldilocksField {
    fn default() -> Self {
        Self { data: Default::default() }
    }
}

impl rand::prelude::Distribution<GoldilocksField> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GoldilocksField {
        todo!()
    }
}

impl PrimeField for GoldilocksField {
    type BigInt = BigInt<1>;

    const MODULUS: Self::BigInt = BigInt::new([18446744069414584321]);

    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt = BigInt::new([9223372034707292160]);

    const MODULUS_BIT_SIZE: u32 = 64;

    const TRACE: Self::BigInt = BigInt::new([4294967295]);

    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt = BigInt::new([2147483647]);

    fn from_bigint(repr: Self::BigInt) -> Option<Self> {
        if repr >= Self::BigInt::zero() && repr < Self::MODULUS {
            let v = repr.to_bytes_le();
            let mut res: u64 = 0;
            let mut shift = 0;
            for b in v {
                res += (b as u64) << shift;
                shift += 8;
            }
            Some(Self { data: res } )
        } else {
            None
        }
    }

    fn into_bigint(self) -> Self::BigInt {
        Self::BigInt::new([self.data])
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use ark_ff::{BigInt, Field, PrimeField};
    use num_bigint::BigUint;

    use super::GoldilocksField;

    fn test_conversion_from_bool(b: bool, expect: &str) {
        let gf: GoldilocksField = b.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_bool() {
        test_conversion_from_bool(false, "0");
        test_conversion_from_bool(true, "1");
    }

    fn test_conversion_from_u8(u: u8, expect: String) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_u8() {
        for u in 0..=255 {
            test_conversion_from_u8(u as u8, u.to_string());
        }
    }

    fn test_conversion_from_u16(u: u16, expect: String) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_u16() {
        for u in 0..=65535 {
            test_conversion_from_u16(u as u16, u.to_string());
        }
    }

    fn test_conversion_from_u32(u: u32, expect: &str) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_u32() {
        test_conversion_from_u32(0, "0");
        test_conversion_from_u32(1, "1");
        test_conversion_from_u32(1234567890, "1234567890");
        test_conversion_from_u32(2147483647, "2147483647");  // 2^31-1
        test_conversion_from_u32(2147483648, "2147483648");  // 2^31
        test_conversion_from_u32(4294967295, "4294967295");  // 2^32-1
    }

    fn test_conversion_from_u64(u: u64, expect: &str) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_u64() {
        test_conversion_from_u64(0, "0");
        test_conversion_from_u64(1, "1");
        test_conversion_from_u64(1234567890, "1234567890");
        test_conversion_from_u64(2147483647, "2147483647");  // 2^31-1
        test_conversion_from_u64(2147483648, "2147483648");  // 2^31
        test_conversion_from_u64(4294967295, "4294967295");  // 2^32-1
        test_conversion_from_u64(18446744069414584320, "18446744069414584320");  // the Goldilocks field modulus - 1
    }

    #[test]
    #[should_panic]
    fn test_conversions_from_u64_too_big_number() {
        test_conversion_from_u64(18446744069414584321, "18446744069414584321");  // the Goldilocks field modulus
    }

    fn test_conversion_from_u128(u: u128, expect: &str) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_u128() {
        test_conversion_from_u128(0, "0");
        test_conversion_from_u128(1, "1");
        test_conversion_from_u128(1234567890, "1234567890");
        test_conversion_from_u128(2147483647, "2147483647");  // 2^31-1
        test_conversion_from_u128(2147483648, "2147483648");  // 2^31
        test_conversion_from_u128(4294967295, "4294967295");  // 2^32-1
        test_conversion_from_u128(18446744069414584320, "18446744069414584320");  // the Goldilocks field modulus - 1
    }

    #[test]
    #[should_panic]
    fn test_conversions_from_u128_too_big_number() {
        test_conversion_from_u128(18446744069414584321, "18446744069414584321");  // the Goldilocks field modulus
    }

    fn test_conversion_from_bigint(u: BigInt<1>, expect: &str) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_bigint() {
        test_conversion_from_bigint(BigInt!("0"), "0");
        test_conversion_from_bigint(BigInt!("1"), "1");
        test_conversion_from_bigint(BigInt!("1234567890"), "1234567890");
        test_conversion_from_bigint(BigInt!("2147483647"), "2147483647");  // 2^31-1
        test_conversion_from_bigint(BigInt!("2147483648"), "2147483648");  // 2^31
        test_conversion_from_bigint(BigInt!("4294967295"), "4294967295");  // 2^32-1
        test_conversion_from_bigint(BigInt!("18446744069414584320"), "18446744069414584320");  // the Goldilocks field modulus - 1
    }

    #[test]
    #[should_panic]
    fn test_conversions_from_bigint_too_big_number() {
        test_conversion_from_bigint(BigInt!("18446744069414584321"), "18446744069414584321");  // the Goldilocks field modulus
    }

    fn test_conversion_via_primefield_from_bigint(u: BigInt<1>, expect: Option<GoldilocksField>) {
        assert_eq!(GoldilocksField::from_bigint(u), expect);
    }

    #[test]
    fn test_primefield_from_bigint() {
        test_conversion_via_primefield_from_bigint(BigInt!("0"), Some(0u64.into()));
        test_conversion_via_primefield_from_bigint(BigInt!("1"), Some(1u64.into()));
        test_conversion_via_primefield_from_bigint(BigInt!("1234567890"), Some(1234567890u64.into()));
        test_conversion_via_primefield_from_bigint(BigInt!("2147483647"), Some(2147483647u64.into()));  // 2^31-1
        test_conversion_via_primefield_from_bigint(BigInt!("2147483648"), Some(2147483648u64.into()));  // 2^31
        test_conversion_via_primefield_from_bigint(BigInt!("4294967295"), Some(4294967295u64.into()));  // 2^32-1
        test_conversion_via_primefield_from_bigint(BigInt!("18446744069414584320"), Some(18446744069414584320u64.into()));  // the Goldilocks field modulus - 1
        test_conversion_via_primefield_from_bigint(BigInt!("18446744069414584321"), None);  // the Goldilocks field modulus
    }

    fn test_conversion_from_biguint(u: BigUint, expect: &str) {
        let gf: GoldilocksField = u.into();
        assert_eq!(format!("{}", gf), expect);
    }

    #[test]
    fn test_conversions_from_biguint() {
        test_conversion_from_biguint(BigUint::from_str("0").unwrap(), "0");
        test_conversion_from_biguint(BigUint::from_str("1").unwrap(), "1");
        test_conversion_from_biguint(BigUint::from_str("1234567890").unwrap(), "1234567890");
        test_conversion_from_biguint(BigUint::from_str("2147483647").unwrap(), "2147483647");  // 2^31-1
        test_conversion_from_biguint(BigUint::from_str("2147483648").unwrap(), "2147483648");  // 2^31
        test_conversion_from_biguint(BigUint::from_str("4294967295").unwrap(), "4294967295");  // 2^32-1
        test_conversion_from_biguint(BigUint::from_str("18446744069414584320").unwrap(), "18446744069414584320");  // the Goldilocks field modulus - 1
    }

    #[test]
    #[should_panic]
    fn test_conversions_from_biguint_too_big_number() {
        test_conversion_from_biguint(BigUint::from_str("18446744069414584321").unwrap(), "18446744069414584321");  // the Goldilocks field modulus
    }

    fn test_conversion_to_biguint(u: GoldilocksField, expect: BigUint) {
        let bu: BigUint = u.into();
        assert_eq!(bu, expect);
    }

    #[test]
    fn test_conversions_to_biguint() {
        test_conversion_to_biguint(0u64.into(), BigUint::from_str("0").unwrap());
        test_conversion_to_biguint(1u64.into(), BigUint::from_str("1").unwrap());
        test_conversion_to_biguint(1234567890u64.into(), BigUint::from_str("1234567890").unwrap());
        test_conversion_to_biguint(2147483647u64.into(), BigUint::from_str("2147483647").unwrap());  // 2^31-1
        test_conversion_to_biguint(2147483648u64.into(), BigUint::from_str("2147483648").unwrap());  // 2^31
        test_conversion_to_biguint(4294967295u64.into(), BigUint::from_str("4294967295").unwrap());  // 2^32-1
        test_conversion_to_biguint(18446744069414584320u64.into(), BigUint::from_str("18446744069414584320").unwrap());  // the Goldilocks field modulus - 1
    }

    fn test_conversion_to_bigint(u: GoldilocksField, expect: BigInt<1>) {
        let bi: BigInt<1> = u.into();
        assert_eq!(bi, expect);
        let bi2 = u.into_bigint();
        assert_eq!(bi2, expect);
    }

    #[test]
    fn test_conversions_to_bigint() {
        test_conversion_to_bigint(0u64.into(), BigInt!("0"));
        test_conversion_to_bigint(1u64.into(), BigInt!("1"));
        test_conversion_to_bigint(1234567890u64.into(), BigInt!("1234567890"));
        test_conversion_to_bigint(2147483647u64.into(), BigInt!("2147483647"));  // 2^31-1
        test_conversion_to_bigint(2147483648u64.into(), BigInt!("2147483648"));  // 2^31
        test_conversion_to_bigint(4294967295u64.into(), BigInt!("4294967295"));  // 2^32-1
        test_conversion_to_bigint(18446744069414584320u64.into(), BigInt!("18446744069414584320"));  // the Goldilocks field modulus - 1
    }

    fn test_single_add_borrow(f1: GoldilocksField, f2: &GoldilocksField, expect: GoldilocksField) {
        let fsum = f1 + f2;
        assert_eq!(fsum, expect);
        let mut fsum2 = f1;
        fsum2 += f2;
        assert_eq!(fsum2, expect);
    }

    fn test_single_add_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField, expect: GoldilocksField) {
        let fsum = f1 + f2.clone();
        assert_eq!(fsum, expect);
        let mut fsum2 = f1;
        fsum2 += f2;
        assert_eq!(fsum2, expect);
    }

    fn test_single_add(f1: GoldilocksField, f2: GoldilocksField, expect: GoldilocksField) {
        // test Add
        let fsum = f1 + f2;
        assert_eq!(fsum, expect);

        // test AddAssign
        let mut fsum2 = f1;
        fsum2 += f2;
        assert_eq!(fsum2, expect);

        // test impl<'a> Add<&'a GoldilocksField> for GoldilocksField
        // and  impl<'a> AddAssign<&'a GoldilocksField> for GoldilocksField
        test_single_add_borrow(f1, &f2, expect);

        // test impl<'a> Add<&'a mut GoldilocksField> for GoldilocksField
        // and  impl<'a> AddAssign<&'a mut GoldilocksField> for GoldilocksField
        let mut f2_mut = f2.clone();
        test_single_add_mut_borrow(f1, &mut f2_mut, expect);
    }

    #[test]
    fn test_add() {
        test_single_add(3u64.into(), 2u64.into(), 5u64.into());
        test_single_add(18446744069414584320u64.into(), 1000u64.into(), 999u64.into());
        test_single_add(18446744069414584320u64.into(), 18446744069414584320u64.into(), 18446744069414584319u64.into());
    }

    fn test_single_sub_borrow(f1: GoldilocksField, f2: &GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 - f2;
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 -= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_sub_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 - f2.clone();
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 -= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_sub(f1: GoldilocksField, f2: GoldilocksField, expect: GoldilocksField) {
        // test Sub
        let fdiff = f1 - f2;
        assert_eq!(fdiff, expect);

        // test SubAssign
        let mut fdiff2 = f1;
        fdiff2 -= f2;
        assert_eq!(fdiff2, expect);

        // test impl<'a> Sub<&'a GoldilocksField> for GoldilocksField
        // and  impl<'a> SubAssign<&'a GoldilocksField> for GoldilocksField
        test_single_sub_borrow(f1, &f2, expect);

        // test impl<'a> Sub<&'a mut GoldilocksField> for GoldilocksField
        // and  impl<'a> SubAssign<&'a mut GoldilocksField> for GoldilocksField
        let mut f2_mut = f2.clone();
        test_single_sub_mut_borrow(f1, &mut f2_mut, expect);
    }

    #[test]
    fn test_sub() {
        test_single_sub(5u64.into(), 2u64.into(), 3u64.into());
        test_single_sub(0u64.into(), 1u64.into(), 18446744069414584320u64.into());
        test_single_sub(999u64.into(), 1000u64.into(), 18446744069414584320u64.into());
        test_single_sub(998u64.into(), 18446744069414584319u64.into(), 1000u64.into());
    }

    fn test_single_neg(f1: GoldilocksField, expect: GoldilocksField) {
        // test Neg
        let fneg = -f1;
        assert_eq!(fneg, expect);

        // test neg_in_place()
        let mut fneg2 = f1;
        fneg2.neg_in_place();
        assert_eq!(fneg2, expect);
    }

    #[test]
    fn test_neg() {
        test_single_neg(0u64.into(), 0u64.into());
        test_single_neg(1u64.into(), 18446744069414584320u64.into());
        test_single_neg(18446744069414584320u64.into(), 1u64.into());
        test_single_neg(9223372034707292160u64.into(), 9223372034707292161u64.into());
        test_single_neg(9223372034707292161u64.into(), 9223372034707292160u64.into());
    }

    fn test_single_mul_borrow(f1: GoldilocksField, f2: &GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 * f2;
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 *= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_mul_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 * f2.clone();
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 *= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_mul(f1: GoldilocksField, f2: GoldilocksField, expect: GoldilocksField) {
        // test Mul
        let fmul = f1 * f2;
        assert_eq!(fmul, expect);

        // test MulAssign
        let mut fmul2 = f1;
        fmul2 *= f2;
        assert_eq!(fmul2, expect);

        // test impl<'a> Mul<&'a GoldilocksField> for GoldilocksField
        // and  impl<'a> MulAssign<&'a GoldilocksField> for GoldilocksField
        test_single_mul_borrow(f1, &f2, expect);

        // test impl<'a> Mul<&'a mut GoldilocksField> for GoldilocksField
        // and  impl<'a> MulAssign<&'a mut GoldilocksField> for GoldilocksField
        let mut f2_mut = f2.clone();
        test_single_mul_mut_borrow(f1, &mut f2_mut, expect);
    }

    #[test]
    fn test_mul() {
        test_single_mul(7u64.into(), 8u64.into(), 56u64.into());
        test_single_mul(0u64.into(), 0u64.into(), 0u64.into());
        test_single_mul(0u64.into(), 18446744069414584320u64.into(), 0u64.into());
        test_single_mul(18446744069414584320u64.into(), 0u64.into(), 0u64.into());
        test_single_mul(18446744069414584320u64.into(), 18446744069414584320u64.into(), 1u64.into());
        test_single_mul(9223372034707292163u64.into(), 2u64.into(), 5u64.into())
    }

    fn test_single_div_borrow(f1: GoldilocksField, f2: &GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 / f2;
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 /= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_div_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField, expect: GoldilocksField) {
        let fdiff = f1 / f2.clone();
        assert_eq!(fdiff, expect);
        let mut fdiff2 = f1;
        fdiff2 /= f2;
        assert_eq!(fdiff2, expect);
    }

    fn test_single_div(f1: GoldilocksField, f2: GoldilocksField, expect: GoldilocksField) {
        // test Div
        let fdiv = f1 / f2;
        assert_eq!(fdiv, expect);

        // test DivAssign
        let mut fdiv2 = f1;
        fdiv2 /= f2;
        assert_eq!(fdiv2, expect);

        // test impl<'a> Div<&'a GoldilocksField> for GoldilocksField
        // and  impl<'a> DivAssign<&'a GoldilocksField> for GoldilocksField
        test_single_div_borrow(f1, &f2, expect);

        // test impl<'a> Div<&'a mut GoldilocksField> for GoldilocksField
        // and  impl<'a> DivAssign<&'a mut GoldilocksField> for GoldilocksField
        let mut f2_mut = f2.clone();
        test_single_div_mut_borrow(f1, &mut f2_mut, expect);
    }

    #[test]
    fn test_div() {
        test_single_div(42u64.into(), 6u64.into(), 7u64.into());
        test_single_div(0u64.into(), 18446744069414584320u64.into(), 0u64.into());
        test_single_div(5u64.into(), 2u64.into(), 9223372034707292163u64.into());
    }

    #[test]
    #[should_panic]
    fn test_div_by_zero() {
        let f1: GoldilocksField = 1u64.into();
        let f2: GoldilocksField = 0u64.into();
        let _fdiv = f1 / f2;
    }

    #[test]
    #[should_panic]
    fn test_div_assign_by_zero() {
        let mut f1: GoldilocksField = 1u64.into();
        let f2: GoldilocksField = 0u64.into();
        f1 /= f2;
    }

    fn internal_test_single_div_by_zero_borrow(f1: GoldilocksField, f2: &GoldilocksField) {
        let _ = f1 / f2;
    }

    #[test]
    #[should_panic]
    fn test_div_by_zero_borrow() {
        let f1: GoldilocksField = 1u64.into();
        let f2: GoldilocksField = 0u64.into();
        internal_test_single_div_by_zero_borrow(f1, &f2);
    }

    fn internal_test_single_div_assign_by_zero_borrow(f1: GoldilocksField, f2: &GoldilocksField) {
        let mut fdiff2 = f1;
        fdiff2 /= f2;
    }

    #[test]
    #[should_panic]
    fn test_div_assign_by_zero_borrow() {
        let f1: GoldilocksField = 1u64.into();
        let f2: GoldilocksField = 0u64.into();
        internal_test_single_div_assign_by_zero_borrow(f1, &f2);
    }

    fn internal_test_single_div_by_zero_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField) {
        let _ = f1 / f2;
    }

    #[test]
    #[should_panic]
    fn test_div_by_zero_mut_borrow() {
        let f1: GoldilocksField = 1u64.into();
        let mut f2: GoldilocksField = 0u64.into();
        internal_test_single_div_by_zero_mut_borrow(f1, &mut f2);
    }

    fn internal_test_single_div_assign_by_zero_mut_borrow(f1: GoldilocksField, f2: &mut GoldilocksField) {
        let mut fdiff2 = f1;
        fdiff2 /= f2;
    }

    #[test]
    #[should_panic]
    fn test_div_assign_by_zero_mut_borrow() {
        let f1: GoldilocksField = 1u64.into();
        let mut f2: GoldilocksField = 0u64.into();
        internal_test_single_div_assign_by_zero_mut_borrow(f1, &mut f2);
    }

    fn internal_test_double(f: GoldilocksField, expect: GoldilocksField) {
        assert_eq!(f.double(), expect);
        let mut f2 = f;
        f2.double_in_place();
        assert_eq!(f2, expect);
    }

    #[test]
    fn test_double() {
        internal_test_double(0u64.into(), 0u64.into());
        internal_test_double(1u64.into(), 2u64.into());
        internal_test_double(1000u64.into(), 2000u64.into());
        internal_test_double(18446744069414584320u64.into(), 18446744069414584319u64.into());
        internal_test_double(9223372034707292163u64.into(), 5u64.into());
    }

    fn internal_test_square(f: GoldilocksField, expect: GoldilocksField) {
        assert_eq!(f.square(), expect);
        let mut f2 = f;
        f2.square_in_place();
        assert_eq!(f2, expect);
    }

    #[test]
    fn test_square() {
        internal_test_square(0u64.into(), 0u64.into());
        internal_test_square(7u64.into(), 49u64.into());
        internal_test_square(18446744069414584320u64.into(), 1u64.into());
    }

    fn internal_test_inverse(f: GoldilocksField, expect: Option<GoldilocksField>) {
        assert_eq!(f.inverse(), expect);
        let mut f2 = f;
        match f2.inverse_in_place() {
            Some(f2_some) => {
                assert!(expect.is_some());
                assert_eq!(*f2_some, expect.unwrap());
                assert_eq!(f2, expect.unwrap());
            },
            None => {
                assert!(expect.is_none());
                assert!(f2 == f);
            }
        }
    }

    #[test]
    fn test_inverse() {
        internal_test_inverse(0u64.into(), None);
        internal_test_inverse(1u64.into(), Some(1u64.into()));
        internal_test_inverse(2u64.into(), Some(9223372034707292161u64.into()));
    }
}
