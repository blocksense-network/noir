use crate::{
    ast::{ExpressionKind, Ident, Pattern, QuantifierExpression, QuantifierType}, parser::ParserErrorReason, token::{Keyword, Token}
};

use super::{Parser, parse_many::separated_by_comma};

impl Parser<'_> {
    pub(super) fn parse_quantifier_expr(&mut self) -> Option<ExpressionKind> {
        let quantifier_type = if self.eat_keyword(Keyword::Forall) {
            QuantifierType::Forall
        } else if self.eat_keyword(Keyword::Exists) {
            QuantifierType::Exists
        } else {
            return None;
        };

        if !self.eat_left_paren() {
            self.expected_token(Token::LeftParen);
            return None;
        } 
        if !self.eat_pipe() {
            self.expected_token(Token::Pipe);
            return None;
        }

        let indexes = self.parse_quantifier_indexes()?;
        let body = self.parse_expression_or_error();

        if !self.eat_right_paren() {
            self.expected_token(Token::RightParen);
            return None;
        }

        Some(ExpressionKind::Quantifier(Box::new(QuantifierExpression {
            quantifier_type,
            indexes,
            body,
        })))
    }

    fn parse_quantifier_indexes(&mut self) -> Option<Vec<Ident>> {
        let start_location = self.current_token_location;
        let patterns = self.parse_many(
            "indexes",
            separated_by_comma().until(Token::Pipe),
            Self::parse_pattern_no_mut,
        );

        if patterns.is_empty() {
            self.push_error(ParserErrorReason::ExpectedIdentifierInIndexScope, self.location_since(start_location));
            return None;
        } 
        if !patterns.iter().all(|p| matches!(p, Pattern::Identifier(_))) {
            self.push_error(ParserErrorReason::InvalidPatternInIndexScope, self.location_since(start_location));
            return None;
        }

        Some(
            patterns
                .into_iter()
                .filter_map(|pattern| {
                    if let Pattern::Identifier(ident) = pattern { Some(ident) } else { None }
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use noirc_errors::Location;

    use crate::{
        ast::{ExpressionKind, Ident, QuantifierExpression, QuantifierType},
        parser::{parser::tests::{expect_no_errors, parse_all_failing}, Parser},
    };

    fn parse_quantifier_no_error(src: &str) -> QuantifierExpression {
        let mut parser = Parser::for_str_with_dummy_file(src);
        let expr = parser.parse_quantifier_expr();

        assert!(expr.is_some());
        if let ExpressionKind::Quantifier(quant_expr) = expr.unwrap() {
            assert_eq!(quant_expr.body.location.span.end() as usize, src.len() - 1);
            expect_no_errors(&parser.errors);
            return *quant_expr;
        } else {
            panic!("Expected successful parsing of quantifier expression");
        };
    }

    #[test]
    fn parse_forall_quantifier() {
        let src = "forall(|i, j| (i < 15) & (arr[i] > arr[j]))";
        let quant_expr = parse_quantifier_no_error(src);
        assert_eq!(quant_expr.quantifier_type, QuantifierType::Forall);
        assert!(
            quant_expr.indexes
                == vec![
                    Ident::new("i".to_string(), Location::dummy()),
                    Ident::new("j".to_string(), Location::dummy())
                ]
        );
        assert_eq!(quant_expr.body.to_string(), "(((i < 15)) & ((arr[i] > arr[j])))");
    }

    #[test]
    fn parse_exists_quantifier() {
        let src = "exists(|i| (i == 50) | (arr[i] > arr[i - 1]))";
        let quant_expr = parse_quantifier_no_error(src);
        assert_eq!(quant_expr.quantifier_type, QuantifierType::Exists);
        assert!(
            quant_expr.indexes
                == vec![
                    Ident::new("i".to_string(), Location::dummy()),
                ]
        );
        assert_eq!(quant_expr.body.to_string(), "(((i == 50)) | ((arr[i] > arr[(i - 1)])))");
    }

    #[test]
    fn parse_error_quantifiers() {
        parse_all_failing(vec![
            "exists(x > 5)",
            "forall(|| x > 5)",
            "forall(||)",
            "forall",
            "forall(|i x > 5)",
            "exists |j| x < 4",
            "exists()",
        ], |parser| parser.parse_quantifier_expr());
    }
}
