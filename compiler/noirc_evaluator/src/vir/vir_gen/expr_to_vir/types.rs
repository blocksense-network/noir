use std::sync::Arc;

use noirc_frontend::{ast::IntegerBitSize, monomorphization::ast::Type, shared::Signedness};
use num_bigint::BigInt;
use vir::ast::{IntRange, Primitive, Typ, TypX};

pub fn const_to_vir_type(number: u32) -> Typ {
    Arc::new(TypX::ConstInt(BigInt::from(number)))
}

fn integer_type_to_vir_typx(signedness: &Signedness, bit_size: &IntegerBitSize) -> TypX {
    match signedness {
        Signedness::Unsigned => TypX::Int(IntRange::U(bit_size.bit_size() as u32)),
        Signedness::Signed => TypX::Int(IntRange::I(bit_size.bit_size() as u32)),
    }
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
        Type::Unit => todo!(),
        Type::Tuple(items) => todo!(),
        Type::Slice(_) => todo!(),
        Type::Reference(_, _) => todo!(),
        Type::Function(items, _, _, _) => todo!(),
    };

    Arc::new(typx)
}
