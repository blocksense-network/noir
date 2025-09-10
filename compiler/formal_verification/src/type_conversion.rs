use noirc_frontend::{Kind, Type as NoirType, monomorphization::ast::Type as MastType};

pub fn convert_mast_to_noir_type(mast_type: MastType) -> NoirType {
    match mast_type {
        MastType::Field => NoirType::FieldElement,
        MastType::Array(len, element_type) => {
            // In noirc_frontend, the length of an array is a type itself.
            // We represent the concrete length from MAST as a `Type::Constant`.
            let length_type = Box::new(NoirType::Constant(len.into(), Kind::Normal));
            let converted_element_type = Box::new(convert_mast_to_noir_type(*element_type));
            NoirType::Array(length_type, converted_element_type)
        }
        MastType::Integer(sign, bits) => NoirType::Integer(sign, bits),
        MastType::Bool => NoirType::Bool,
        MastType::String(len) => {
            // Similar to arrays, the string length is a `Type::Constant`.
            let length_type = Box::new(NoirType::Constant(len.into(), Kind::Normal));
            NoirType::String(length_type)
        }
        MastType::FmtString(len, elements_type) => {
            let length_type = Box::new(NoirType::Constant(len.into(), Kind::Normal));
            let converted_elements_type = Box::new(convert_mast_to_noir_type(*elements_type));
            NoirType::FmtString(length_type, converted_elements_type)
        }
        MastType::Unit => NoirType::Unit,
        MastType::Tuple(mast_elements) => {
            // Recursively convert each type within the tuple.
            let noir_elements = mast_elements.into_iter().map(convert_mast_to_noir_type).collect();
            NoirType::Tuple(noir_elements)
        }
        MastType::Slice(element_type) => {
            // Recursively convert the slice's element type.
            let converted_element_type = Box::new(convert_mast_to_noir_type(*element_type));
            NoirType::Slice(converted_element_type)
        }
        MastType::Reference(element_type, mutable) => {
            let converted_element_type = Box::new(convert_mast_to_noir_type(*element_type));
            NoirType::Reference(converted_element_type, mutable)
        }
        MastType::Function(args, ret, env, unconstrained) => {
            // Recursively convert all function components: arguments, return type, and environment.
            let noir_args = args.into_iter().map(convert_mast_to_noir_type).collect();
            let noir_ret = Box::new(convert_mast_to_noir_type(*ret));
            let noir_env = Box::new(convert_mast_to_noir_type(*env));
            NoirType::Function(noir_args, noir_ret, noir_env, unconstrained)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::convert_mast_to_noir_type;
    use noirc_frontend::ast::IntegerBitSize;
    use noirc_frontend::shared::Signedness;
    use noirc_frontend::{Kind, Type as NoirType, monomorphization::ast::Type as MastType};

    #[test]
    fn test_convert_field() {
        let mast_type = MastType::Field;
        let expected_noir_type = NoirType::FieldElement;
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_integer() {
        let mast_type = MastType::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo);
        let expected_noir_type = NoirType::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo);
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_bool() {
        let mast_type = MastType::Bool;
        let expected_noir_type = NoirType::Bool;
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_unit() {
        let mast_type = MastType::Unit;
        let expected_noir_type = NoirType::Unit;
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_string() {
        let mast_type = MastType::String(10);
        let expected_noir_type =
            NoirType::String(Box::new(NoirType::Constant(10u32.into(), Kind::Normal)));
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_array() {
        let mast_type = MastType::Array(5, Box::new(MastType::Field));
        let expected_noir_type = NoirType::Array(
            Box::new(NoirType::Constant(5u32.into(), Kind::Normal)),
            Box::new(NoirType::FieldElement),
        );
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_slice() {
        let mast_type = MastType::Slice(Box::new(MastType::Bool));
        let expected_noir_type = NoirType::Slice(Box::new(NoirType::Bool));
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_tuple() {
        let mast_type = MastType::Tuple(vec![MastType::Field, MastType::Bool]);
        let expected_noir_type = NoirType::Tuple(vec![NoirType::FieldElement, NoirType::Bool]);
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }

    #[test]
    fn test_convert_reference() {
        // Immutable reference
        let mast_imm_ref = MastType::Reference(Box::new(MastType::Field), false);
        let expected_imm_ref = NoirType::Reference(Box::new(NoirType::FieldElement), false);
        assert_eq!(convert_mast_to_noir_type(mast_imm_ref), expected_imm_ref);

        // Mutable reference
        let mast_mut_ref = MastType::Reference(Box::new(MastType::Field), true);
        let expected_mut_ref = NoirType::Reference(Box::new(NoirType::FieldElement), true);
        assert_eq!(convert_mast_to_noir_type(mast_mut_ref), expected_mut_ref);
    }

    #[test]
    fn test_convert_function() {
        let mast_type = MastType::Function(
            vec![MastType::Field, MastType::Bool],
            Box::new(MastType::Unit),
            Box::new(MastType::Tuple(vec![])),
            false,
        );
        let expected_noir_type = NoirType::Function(
            vec![NoirType::FieldElement, NoirType::Bool],
            Box::new(NoirType::Unit),
            Box::new(NoirType::Tuple(vec![])),
            false,
        );
        assert_eq!(convert_mast_to_noir_type(mast_type), expected_noir_type);
    }
}
