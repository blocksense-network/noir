use std::sync::Arc;

use acvm::{AcirField, FieldElement};
use noirc_frontend::{ast::{BinaryOpKind, IntegerBitSize}, monomorphization::ast::Type, shared::Signedness};
use num_bigint::BigInt;
use vir::ast::{Dt, IntRange, IntegerTypeBitwidth, Primitive, Typ, TypX};

pub fn const_to_vir_type(number: u32) -> Typ {
    Arc::new(TypX::ConstInt(BigInt::from(number)))
}

fn integer_type_to_vir_typx(signedness: &Signedness, bit_size: &IntegerBitSize) -> TypX {
    match signedness {
        Signedness::Unsigned => TypX::Int(IntRange::U(bit_size.bit_size().into())),
        Signedness::Signed => TypX::Int(IntRange::I(bit_size.bit_size().into())),
    }
}

// Returns the VIR unit type (also known as void type)
pub fn make_unit_vir_type() -> Typ {
    Arc::new(make_unit_vir_typx())
}

fn make_unit_vir_typx() -> TypX {
    TypX::Datatype(Dt::Tuple(0), Arc::new(Vec::new()), Arc::new(Vec::new()))
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
        Type::Unit => make_unit_vir_typx(),
        Type::Tuple(item_types) => build_tuple_type(
            item_types.iter().map(ast_type_to_vir_type).collect(),
        ),
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
/// for the types Field and Signed integer.
/// Will `panic` if the type is not an integer.
pub(crate) fn get_bit_not_bitwidth(integer_type: &Type) -> Option<IntegerTypeBitwidth> {
    match integer_type {
        Type::Field | Type::Integer(Signedness::Signed, _) => None,
        Type::Integer(Signedness::Unsigned, _) => Some(get_integer_type_bitwidth(integer_type)),
        _ => {
            // All Noir binary expression have gone through type checking
            // during the elaboration phase in `compiler/noirc_frontend/src/elaborator`.
            // Therefore we can assume that bit not operation only occurs with integers.
            unreachable!("Can get a bit width only of integer types")
        }
    }
}

pub fn get_binary_op_type(lhs_type: Typ, binary_op: &BinaryOpKind) -> Typ {
    if binary_op.is_comparator() || binary_op.is_equality() || binary_op.is_implication() {
        Arc::new(TypX::Bool)
    } else {
        lhs_type
    }
}

pub fn build_tuple_type(vir_types: Vec<Typ>) -> TypX {
    TypX::Datatype(Dt::Tuple(vir_types.len()), Arc::new(vir_types), Arc::new(Vec::new()))
}
