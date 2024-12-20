use std::{fmt::Display, iter::{Product, Sum}, num::ParseIntError, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}, str::FromStr};

use ark_ff::{BigInt, BigInteger, FftField, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize, CanonicalSerializeWithFlags, Compress, Flags, SerializationError, Valid, Validate};
use num_bigint::BigUint;
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
        // TODO: range checking
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
        todo!()
    }

    fn double_in_place(&mut self) -> &mut Self {
        todo!()
    }

    fn neg_in_place(&mut self) -> &mut Self {
        todo!()
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        todo!()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        todo!()
    }

    fn square_in_place(&mut self) -> &mut Self {
        todo!()
    }

    fn inverse(&self) -> Option<Self> {
        todo!()
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        todo!()
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
        todo!()
    }
}

impl<'a> MulAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn mul_assign(&mut self, rhs: &'a mut GoldilocksField) {
        todo!()
    }
}

impl<'a> SubAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn sub_assign(&mut self, rhs: &'a mut GoldilocksField) {
        todo!()
    }
}

impl<'a> AddAssign<&'a mut GoldilocksField> for GoldilocksField {
    fn add_assign(&mut self, rhs: &'a mut GoldilocksField) {
        todo!()
    }
}

impl<'a> Div<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Mul<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn mul(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Sub<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Add<&'a mut GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: &'a mut GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> DivAssign<&'a GoldilocksField> for GoldilocksField {
    fn div_assign(&mut self, rhs: &'a GoldilocksField) {
        todo!()
    }
}

impl<'a> MulAssign<&'a GoldilocksField> for GoldilocksField {
    fn mul_assign(&mut self, rhs: &'a GoldilocksField) {
        todo!()
    }
}

impl<'a> SubAssign<&'a GoldilocksField> for GoldilocksField {
    fn sub_assign(&mut self, rhs: &'a GoldilocksField) {
        todo!()
    }
}

impl<'a> AddAssign<&'a GoldilocksField> for GoldilocksField {
    fn add_assign(&mut self, rhs: &'a GoldilocksField) {
        todo!()
    }
}

impl<'a> Div<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: &'a GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Mul<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn mul(self, rhs: &'a GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Sub<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: &'a GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl<'a> Add<&'a GoldilocksField> for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: &'a GoldilocksField) -> Self::Output {
        todo!()
    }
}

impl DivAssign for GoldilocksField {
    fn div_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl MulAssign for GoldilocksField {
    fn mul_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl SubAssign for GoldilocksField {
    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl AddAssign for GoldilocksField {
    fn add_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Div for GoldilocksField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Sub for GoldilocksField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
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
        todo!()
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
        todo!()
    }
}

impl Add for GoldilocksField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
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
    use ark_ff::{BigInt, PrimeField};

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
}
