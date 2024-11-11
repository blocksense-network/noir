use core::fmt::Debug;
use std::marker::PhantomData;

use super::{asm_writer::AsmWriter, config::P2Field};
use plonky2::{
    field::types::{Field, PrimeField64},
    iop::{
        generator::{GeneratedValues, SimpleGenerator},
        witness::{PartitionWitness, Witness, WitnessWrite},
    },
    plonk::circuit_data::CommonCircuitData,
    util::serialization::IoResult,
};

use plonky2::iop::target::Target;

#[derive(Debug, Clone)]
struct VariableIntDivGenerator {
    numerator: Target,
    denominator: Target,
    quotient: Target,
    remainder: Target,
    _phantom: PhantomData<P2Field>,
    signed: bool,
    bitsize: usize,
}

impl VariableIntDivGenerator {
    fn new(asm_writer: &mut AsmWriter, numerator: Target, denominator: Target, signed: bool, bitsize: usize) -> Self {
        Self {
            numerator,
            denominator,
            quotient: asm_writer.add_virtual_target(),
            remainder: asm_writer.add_virtual_target(),
            _phantom: PhantomData,
            signed,
            bitsize
        }
    }

    fn id() -> String {
        "VariableIntDivGenerator".to_string()
    }
}

impl SimpleGenerator<P2Field, 2> for VariableIntDivGenerator {
    fn id(&self) -> String {
        Self::id()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.numerator, self.denominator]
    }

    #[allow(unused_variables)]
    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<P2Field, 2>,
    ) -> IoResult<()> {
        todo!()
    }

    #[allow(unused_variables)]
    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<P2Field, 2>,
    ) -> IoResult<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<P2Field>,
        out_buffer: &mut GeneratedValues<P2Field>,
    ) {
        let numerator = witness.get_target(self.numerator).to_canonical_u64();
        let denominator = witness.get_target(self.denominator).to_canonical_u64();

        let mut quotient: u64 = 0;
        let mut remainder: u64 = 0;
        if denominator != 0 {
            if self.signed {
                match self.bitsize {
                    8 => {
                        quotient = (((numerator as i8) / (denominator as i8)) as u8) as u64;
                        remainder = (((numerator as i8) % (denominator as i8)) as u8) as u64;
                    }
                    16 => {
                        quotient = (((numerator as i16) / (denominator as i16)) as u16) as u64;
                        remainder = (((numerator as i16) % (denominator as i16)) as u16) as u64;
                    }
                    32 => {
                        quotient = (((numerator as i32) / (denominator as i32)) as u32) as u64;
                        remainder = (((numerator as i32) % (denominator as i32)) as u32) as u64;
                    }
                    64 => {
                        quotient = ((numerator as i64) / (denominator as i64)) as u64;
                        remainder = ((numerator as i64) % (denominator as i64)) as u64;
                    }
                    _ => unreachable!()
                }
            } else {
                match self.bitsize {
                    8 => {
                        quotient = ((numerator as u8) / (denominator as u8)) as u64;
                        remainder = ((numerator as u8) % (denominator as u8)) as u64;
                    }
                    16 => {
                        quotient = ((numerator as u16) / (denominator as u16)) as u64;
                        remainder = ((numerator as u16) % (denominator as u16)) as u64;
                    }
                    32 => {
                        quotient = ((numerator as u32) / (denominator as u32)) as u64;
                        remainder = ((numerator as u32) % (denominator as u32)) as u64;
                    }
                    64 => {
                        quotient = numerator / denominator;
                        remainder = numerator % denominator;
                    }
                    _ => unreachable!()
                }
            }
        }

        out_buffer.set_target(self.quotient, P2Field::from_canonical_u64(quotient));
        out_buffer.set_target(self.remainder, P2Field::from_canonical_u64(remainder));
    }
}

/// Add an integer division operation to a circuit, returning both the quotient and the remainder.
///
/// Can be used to implement both integer division and modulus. Also, can be used to implement
/// less-than.
///
/// This uses a custom `SimpleGenerator` internally, which will have performance implications.
pub(crate) fn add_div_mod(
    asm_writer: &mut AsmWriter,
    numerator: Target,
    denominator: Target,
    signed: bool,
    bitsize: usize,
) -> (Target, Target) {
    asm_writer.comment_divmod_begin(numerator, denominator);

    let generator = VariableIntDivGenerator::new(asm_writer, numerator, denominator, signed, bitsize);
    asm_writer.get_mut_builder().add_simple_generator(generator.clone());

    if !signed {
        let q_times_d = asm_writer.mul(generator.quotient, denominator);
        let q_times_d_plus_r = asm_writer.add(q_times_d, generator.remainder);
        let sanity = asm_writer.is_equal(numerator, q_times_d_plus_r);
        asm_writer.assert_bool(sanity);
    }
    let z = asm_writer.zero();
    let d_is_zero = asm_writer.is_equal(denominator, z);
    let d_is_not_zero = asm_writer.not(d_is_zero);
    asm_writer.assert_bool(d_is_not_zero);

    asm_writer.comment_divmod_end(generator.quotient, generator.remainder);

    (generator.quotient, generator.remainder)
}
