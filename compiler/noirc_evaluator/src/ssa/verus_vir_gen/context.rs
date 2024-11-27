use std::{collections::HashMap, sync::Arc};

use vir::ast::{Dt, Expr, ExprX, FieldOpr, Ident, Param, SpannedTyped, Typ, TypX, VarIdent};

use super::{empty_span, function::get_function_return_values, BuildingKrateError, Function, ValueId};

pub(crate) struct ResultIdFixer {
    dt_tuple: Dt,
    dt_typs: Vec<Typ>,
    dt_len: Ident,
    result_span: Expr,
    id_map: HashMap<ValueId, Ident>,
}

impl ResultIdFixer {
    fn result_variable_map(func: &Function) -> Result<HashMap<ValueId, Ident>, BuildingKrateError> {
        let return_values = get_function_return_values(func)?;
        let mut result: HashMap<ValueId, Ident> = HashMap::new();

        for (i, &vid) in return_values.iter().enumerate() {
            result.insert(vid, Arc::new(i.to_string()));
        }

        Ok(result)
    }

    pub(crate) fn new(func: &Function, ret: &Param) -> Result<ResultIdFixer, BuildingKrateError> {
        let (mut dt_len, dt_typs, dt_tuple) = match &*ret.x.typ.clone() {
            TypX::Datatype(Dt::Tuple(len), typs, _) => {
                (len.to_string(), (**typs).clone(), Dt::Tuple(len.clone()))
            }
            _ => {
                return Err(BuildingKrateError::SomeError(
                    "Function return type is not a tuple".to_string(),
                ))
            }
        };

        dt_len.insert_str(0, "tuple%");
        let dt_len = Arc::new(dt_len);

        Ok(ResultIdFixer {
            dt_tuple,
            dt_typs,
            dt_len,
            result_span: SpannedTyped::new(
                &empty_span(),
                &ret.x.typ.clone(),
                ExprX::Var(VarIdent(
                    Arc::new("result".to_string()),
                    vir::ast::VarIdentDisambiguate::NoBodyParam,
                )),
            ),
            id_map: Self::result_variable_map(func).unwrap(),
        })
    }

    pub(crate) fn fix_id(&self, id: &ValueId) -> Option<Expr> {
        if !self.id_map.contains_key(id) {
            return None;
        }

        Some(SpannedTyped::new(
            &empty_span(),
            &self.dt_typs[(*self.id_map[id]).parse::<usize>().unwrap()],
            ExprX::UnaryOpr(
                vir::ast::UnaryOpr::Field(FieldOpr {
                    datatype: self.dt_tuple.clone(),
                    variant: self.dt_len.clone(),
                    field: self.id_map[id].clone(),
                    get_variant: false,
                    check: vir::ast::VariantCheck::None,
                }),
                self.result_span.clone(),
            ),
        ))
    }
}


pub(crate) struct SSAContext<'a> {
    pub result_id_fixer: Option<&'a ResultIdFixer>,
    pub side_effects_condition: Option<ValueId>,
}