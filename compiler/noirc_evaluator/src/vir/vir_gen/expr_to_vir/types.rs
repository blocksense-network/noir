use std::sync::Arc;

use acvm::{AcirField, FieldElement};
use noirc_frontend::{ast::IntegerBitSize, monomorphization::ast::Type, shared::Signedness};
use num_bigint::BigInt;
use vir::ast::{Dt, IntRange, IntegerTypeBitwidth, Primitive, Typ, TypX};

pub fn const_to_vir_type(number: u32) -> Typ {
    Arc::new(TypX::ConstInt(BigInt::from(number)))
}

fn integer_type_to_vir_typx(signedness: &Signedness, bit_size: &IntegerBitSize) -> TypX {
    match signedness {
        Signedness::Unsigned => TypX::Int(IntRange::U(bit_size.bit_size() as u32)),
        Signedness::Signed => TypX::Int(IntRange::I(bit_size.bit_size() as u32)),
    }
}

// Returns the VIR unit type (also known as void type)
pub fn make_unit_vir_type() -> Typ {
    Arc::new(TypX::Datatype(Dt::Tuple(0), Arc::new(Vec::new()), Arc::new(Vec::new())))
}

pub fn ast_type_to_vir_type(ast_type: &Type) -> Typ {
    let typx = match ast_type {
        Type::Field => TypX::Int(IntRange::Int),
        Type::Array(len, inner_type) => TypX::Primitive(
            Primitive::Array,
            Arc::new(vec![ast_type_to_vir_type(&inner_type), const_to_vir_type(*len)]),
        ),
        Type::Integer(signedness, integer_bit_size) => {
            integer_type_to_vir_typx(signedness, integer_bit_size)
        }
        Type::Bool => TypX::Bool,
        Type::String(_) => todo!(),
        Type::FmtString(_, _) => todo!(),
        Type::Unit => TypX::Datatype(Dt::Tuple(0), Arc::new(Vec::new()), Arc::new(Vec::new())),
        Type::Tuple(items) => todo!(),
        Type::Slice(_) => todo!(),
        Type::Reference(_, _) => todo!(),
        Type::Function(items, _, _, _) => todo!(),
    };

    Arc::new(typx)
}

fn get_integer_type_bitwidth(integer_type: &Type) -> IntegerTypeBitwidth {
    match integer_type {
        Type::Field => IntegerTypeBitwidth::Width(FieldElement::max_num_bits()),
        Type::Integer(_, integer_bit_size) => {
            IntegerTypeBitwidth::Width(integer_bit_size.bit_size() as u32)
        }
        _ => unreachable!("Can get a bit width only of integer types"),
    }
}

/// Similar to `get_integer_type_bitwidth` but we have a different logic
/// for the types Field and Signed integer
pub fn get_bit_not_bitwidth(integer_type: &Type) -> Option<IntegerTypeBitwidth> {
    match integer_type {
        Type::Field | Type::Integer(Signedness::Signed, _) => None,
        Type::Integer(Signedness::Unsigned, _) => Some(get_integer_type_bitwidth(integer_type)),
        _ => unreachable!("Can get a bit width only of integer types"),
    }
}
