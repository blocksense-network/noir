use std::{collections::HashMap, sync::Arc};

use vir::ast::{Dt, Expr, ExprX, FieldOpr, Ident, Param, SpannedTyped, Stmt, Typ, TypX, VarIdent};

use super::{
    empty_span, function::get_function_return_values, BuildingKrateError, Function, ValueId,
};

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

pub(crate) struct QuantifierContext {
    quant_indexes: Vec<Vec<String>>,
    quant_body: Vec<Vec<Stmt>>,
}

impl QuantifierContext {
    pub(crate) fn new() -> Self {
        QuantifierContext { quant_indexes: Vec::new(), quant_body: Vec::new() }
    }

    pub(crate) fn start_quantifier(&mut self, indexes: Vec<String>) {
        self.push_indexes(indexes);
        self.create_quantifier_body();
    }

    pub(crate) fn finish_quantifier(&mut self) -> (Option<Vec<String>>, Option<Vec<Stmt>>) {
        (self.pop_indexes(), self.pop_quantifier_body())
    }

    pub(crate) fn push_indexes(&mut self, indexes: Vec<String>) {
        self.quant_indexes.push(indexes);
    }

    pub(crate) fn pop_indexes(&mut self) -> Option<Vec<String>> {
        self.quant_indexes.pop()
    }

    pub(crate) fn create_quantifier_body(&mut self) {
        self.quant_body.push(Vec::new());
    }

    pub(crate) fn push_statement(&mut self, statement: Stmt) {
        if let Some(body_statements) = self.quant_body.last_mut() {
            body_statements.push(statement);
        } else {
            panic!("No quantifier body to push statement to");
        }
    }

    pub(crate) fn pop_quantifier_body(&mut self) -> Option<Vec<Stmt>> {
        self.quant_body.pop()
    }

    pub(crate) fn get_top_quant_indexes(&self) -> Option<&Vec<String>> {
        self.quant_indexes.last()
    }

    pub(crate) fn is_inside_quantifier_body(&self) -> bool {
        !self.quant_indexes.is_empty()
    }
}

pub(crate) struct SSAContext<'a> {
    pub result_id_fixer: Option<&'a ResultIdFixer>,
    pub side_effects_condition: Option<ValueId>,
    pub quantifier_context: QuantifierContext,
}

impl<'a> SSAContext<'a> {
    pub(crate) fn new(
        result_id_fixer: Option<&'a ResultIdFixer>,
        side_effects_condition: Option<ValueId>,
    ) -> Self {
        SSAContext {
            result_id_fixer,
            side_effects_condition,
            quantifier_context: QuantifierContext::new(),
        }
    }
}
