use std::collections::{HashMap, HashSet};

use crate::neuron::{AggregateOp, ArithmeticOp, ComparisonOp, Condition, Neuron, QorValue, Statement, TestExpect, is_unary_arith};
use crate::truth_value::TruthValue;

/// Parse errors with location information.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse warnings (non-fatal issues detected at parse time).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseWarning {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for ParseWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.message)
    }
}

/// Parser state: tracks position in the input.
struct Parser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
    col: usize,
    warnings: Vec<ParseWarning>,
    /// Pending @stratum annotation for the next rule.
    pending_stratum: Option<u32>,
    /// Strata collected for each statement (parallel to output vec).
    strata: Vec<Option<u32>>,
    /// Current @module prefix (None if top-level).
    current_module: Option<String>,
    /// Imports: short_name -> full_name.
    imports: HashMap<String, String>,
    /// Pending statements from multi-head named rules.
    pending: Vec<Statement>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser {
            input,
            pos: 0,
            line: 1,
            col: 1,
            warnings: Vec::new(),
            pending_stratum: None,
            strata: Vec::new(),
            current_module: None,
            imports: HashMap::new(),
            pending: Vec::new(),
        }
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            line: self.line,
            col: self.col,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(ch) = self.peek() {
                if ch.is_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }
            // Skip ;; comments
            if self.input[self.pos..].starts_with(";;") {
                while let Some(ch) = self.peek() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: char) -> Result<(), ParseError> {
        self.skip_whitespace_and_comments();
        match self.peek() {
            Some(c) if c == ch => {
                self.advance();
                Ok(())
            }
            Some(c) => Err(self.err(format!("expected '{}', found '{}'", ch, c))),
            None => Err(self.err(format!("expected '{}', found end of input", ch))),
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    /// Check if current position has an arrow: `→` (Unicode) or `->` (ASCII).
    fn peek_arrow(&self) -> bool {
        let remaining = &self.input[self.pos..];
        remaining.starts_with('→') || remaining.starts_with("->")
    }

    /// Consume an arrow token (`→` or `->`).
    fn consume_arrow(&mut self) {
        if self.input[self.pos..].starts_with('→') {
            self.advance(); // single Unicode char
        } else {
            self.advance(); // '-'
            self.advance(); // '>'
        }
    }

    /// Check if the next token is a keyword (e.g. "if") without consuming it.
    /// A keyword must be followed by a non-name character (space, '(', '<', etc).
    fn peek_keyword(&self, keyword: &str) -> bool {
        let remaining = &self.input[self.pos..];
        if remaining.starts_with(keyword) {
            let after = &remaining[keyword.len()..];
            after.is_empty()
                || after.starts_with(|c: char| {
                    !c.is_ascii_alphanumeric() && c != '_' && c != '-'
                })
        } else {
            false
        }
    }

    /// Consume a keyword (advance past it).
    fn consume_keyword(&mut self, keyword: &str) {
        for _ in keyword.chars() {
            self.advance();
        }
    }

    /// Parse a string literal: "..."
    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => return Ok(s),
                Some(ch) => s.push(ch),
                None => return Err(self.err("unterminated string")),
            }
        }
    }

    /// Parse a name: starts with letter or underscore (including Unicode letters),
    /// then letters/digits/_ /-/. are allowed.
    fn parse_name(&mut self) -> Result<String, ParseError> {
        let mut name = String::new();
        match self.peek() {
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                name.push(ch);
                self.advance();
            }
            Some(ch) => return Err(self.err(format!("expected name, found '{}'", ch))),
            None => return Err(self.err("expected name, found end of input")),
        }
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                name.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        Ok(name)
    }

    /// Parse a number: integer or float.
    fn parse_number(&mut self) -> Result<Neuron, ParseError> {
        let mut num_str = String::new();
        let mut is_float = false;

        // Optional negative sign
        if self.peek() == Some('-') {
            num_str.push('-');
            self.advance();
        }

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if num_str.is_empty() || num_str == "-" {
            return Err(self.err("expected number"));
        }

        if is_float {
            let val: f64 = num_str
                .parse()
                .map_err(|_| self.err(format!("invalid float: {}", num_str)))?;
            Ok(Neuron::Value(QorValue::Float(val)))
        } else {
            let val: i64 = num_str
                .parse()
                .map_err(|_| self.err(format!("invalid integer: {}", num_str)))?;
            Ok(Neuron::Value(QorValue::Int(val)))
        }
    }

    /// Parse a single symbol, variable, value, or nested expression.
    fn parse_neuron(&mut self) -> Result<Neuron, ParseError> {
        self.skip_whitespace_and_comments();

        match self.peek() {
            Some('(') => self.parse_expression(),
            Some('$') => {
                self.advance(); // skip $
                let name = self.parse_name()?;
                Ok(Neuron::Variable(name))
            }
            Some('"') => {
                let s = self.parse_string()?;
                Ok(Neuron::Value(QorValue::Str(s)))
            }
            Some(ch) if ch.is_ascii_digit() || ch == '-' => self.parse_number(),
            Some(ch) if ch.is_alphabetic() => {
                let name = self.parse_name()?;
                // Check for boolean literals
                match name.as_str() {
                    "true" => Ok(Neuron::Value(QorValue::Bool(true))),
                    "false" => Ok(Neuron::Value(QorValue::Bool(false))),
                    _ => Ok(Neuron::Symbol(name)),
                }
            }
            Some(ch) => Err(self.err(format!("unexpected character: '{}'", ch))),
            None => Err(self.err("unexpected end of input")),
        }
    }

    /// Parse an expression: (symbol ...)
    fn parse_expression(&mut self) -> Result<Neuron, ParseError> {
        self.expect('(')?;
        let mut children = Vec::new();

        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                Some(')') => {
                    self.advance();
                    break;
                }
                None => return Err(self.err("unterminated expression, expected ')'")),
                _ => {
                    children.push(self.parse_neuron()?);
                }
            }
        }

        if children.is_empty() {
            return Err(self.err("empty expression '()' is not allowed"));
        }

        Ok(Neuron::Expression(children))
    }

    /// Parse a truth value: <0.99> or <0.99, 0.85> or <auto>
    fn parse_truth_value(&mut self) -> Result<TruthValue, ParseError> {
        self.expect('<')?;
        self.skip_whitespace_and_comments();

        // Check for <auto> keyword — passthrough confidence propagation
        if self.peek_keyword("auto") {
            self.consume_keyword("auto");
            self.skip_whitespace_and_comments();
            self.expect('>')?;
            return Ok(TruthValue::new(1.0, 1.0));
        }

        let strength = self.parse_float()?;

        self.skip_whitespace_and_comments();

        let confidence = if self.peek() == Some(',') {
            self.advance(); // skip comma
            self.skip_whitespace_and_comments();
            self.parse_float()?
        } else {
            crate::truth_value::DEFAULT_CONFIDENCE
        };

        self.skip_whitespace_and_comments();
        self.expect('>')?;

        Ok(TruthValue::new(strength, confidence))
    }

    fn parse_float(&mut self) -> Result<f64, ParseError> {
        let mut num_str = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        num_str
            .parse()
            .map_err(|_| self.err(format!("expected number, got '{}'", num_str)))
    }

    /// Try to parse an optional truth value. Returns None if no '<' follows.
    fn try_parse_truth_value(&mut self) -> Result<Option<TruthValue>, ParseError> {
        self.skip_whitespace_and_comments();
        if self.peek() == Some('<') {
            Ok(Some(self.parse_truth_value()?))
        } else {
            Ok(None)
        }
    }

    /// Try to parse a comparison operator: >, <, >=, <=, ==, !=
    /// Returns None (without consuming input) if no operator found.
    fn try_parse_comparison_op(&mut self) -> Option<ComparisonOp> {
        let remaining = &self.input[self.pos..];
        if remaining.starts_with(">=") {
            self.advance(); self.advance();
            Some(ComparisonOp::Ge)
        } else if remaining.starts_with('>') {
            self.advance();
            Some(ComparisonOp::Gt)
        } else if remaining.starts_with("<=") {
            self.advance(); self.advance();
            Some(ComparisonOp::Le)
        } else if remaining.starts_with('<') {
            self.advance();
            Some(ComparisonOp::Lt)
        } else if remaining.starts_with("==") {
            self.advance(); self.advance();
            Some(ComparisonOp::Eq)
        } else if remaining.starts_with("!=") {
            self.advance(); self.advance();
            Some(ComparisonOp::Ne)
        } else {
            None
        }
    }

    /// Try to parse an aggregate operator: count, sum, min, max.
    /// Returns None (without consuming input) if no operator found.
    ///
    /// For min/max: disambiguate from arithmetic by peeking ahead.
    ///   Aggregate: (min (pattern) $bind -> $result) — next token is '('
    ///   Arithmetic: (min $a $b $result) — next token is '$' or number
    fn try_parse_aggregate_op(&mut self) -> Option<AggregateOp> {
        if self.peek_keyword("count") {
            self.consume_keyword("count");
            Some(AggregateOp::Count)
        } else if self.peek_keyword("sum") {
            self.consume_keyword("sum");
            Some(AggregateOp::Sum)
        } else if self.peek_keyword("min") {
            // Peek ahead: aggregate min needs '(' for pattern
            let after_pos = self.pos + 3; // len("min")
            let after = &self.input[after_pos..];
            let next_non_ws = after.trim_start().chars().next();
            if next_non_ws == Some('(') {
                self.consume_keyword("min");
                Some(AggregateOp::Min)
            } else {
                None // let arithmetic parser handle it
            }
        } else if self.peek_keyword("max") {
            let after_pos = self.pos + 3; // len("max")
            let after = &self.input[after_pos..];
            let next_non_ws = after.trim_start().chars().next();
            if next_non_ws == Some('(') {
                self.consume_keyword("max");
                Some(AggregateOp::Max)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Try to parse an arithmetic operator: +, -, *, /, %
    /// Only matches if the operator is followed by whitespace (to avoid
    /// confusing `-3` with subtraction).
    fn try_parse_arithmetic_op(&mut self) -> Option<ArithmeticOp> {
        let remaining = &self.input[self.pos..];
        if remaining.len() < 2 {
            return None;
        }
        // Multi-char keywords: min, max, sqrt, abs, power, digit-sum
        for (kw, op, len) in &[
            ("digit-sum ", ArithmeticOp::DigitSum, 9usize),
            ("digit-sum\t", ArithmeticOp::DigitSum, 9),
            ("power ", ArithmeticOp::Power, 5),
            ("power\t", ArithmeticOp::Power, 5),
            ("sqrt ", ArithmeticOp::Sqrt, 4),
            ("sqrt\t", ArithmeticOp::Sqrt, 4),
            ("abs ", ArithmeticOp::Abs, 3),
            ("abs\t", ArithmeticOp::Abs, 3),
            ("min ", ArithmeticOp::Min, 3),
            ("min\t", ArithmeticOp::Min, 3),
            ("max ", ArithmeticOp::Max, 3),
            ("max\t", ArithmeticOp::Max, 3),
        ] {
            if remaining.len() >= *len + 1 && remaining.starts_with(kw) {
                self.pos += *len;
                return Some(op.clone());
            }
        }
        let mut chars = remaining.chars();
        let op_char = chars.next()?;
        let next_char = chars.next()?;
        // Arithmetic op must be followed by whitespace
        if !next_char.is_whitespace() {
            return None;
        }
        match op_char {
            '+' => { self.advance(); Some(ArithmeticOp::Add) }
            '-' => { self.advance(); Some(ArithmeticOp::Sub) }
            '*' => { self.advance(); Some(ArithmeticOp::Mul) }
            '/' => { self.advance(); Some(ArithmeticOp::Div) }
            '%' => { self.advance(); Some(ArithmeticOp::Mod) }
            _ => None,
        }
    }

    /// Try to parse an optional `@decay float` annotation.
    fn try_parse_decay(&mut self) -> Result<Option<f64>, ParseError> {
        self.skip_whitespace_and_comments();
        if self.peek_keyword("@decay") {
            self.consume_keyword("@decay");
            self.skip_whitespace_and_comments();
            let rate = self.parse_float()?;
            Ok(Some(rate))
        } else {
            Ok(None)
        }
    }

    /// Parse a single statement: fact, query, rule, or test.
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        self.skip_whitespace_and_comments();

        match self.peek() {
            Some('?') => {
                self.advance(); // skip ?
                self.skip_whitespace_and_comments();
                let pattern = self.parse_expression()?;
                Ok(Statement::Query { pattern })
            }
            Some('(') => {
                // Peek for named rule: (rule name body → head)
                {
                    let saved = (self.pos, self.line, self.col);
                    self.advance(); // consume '('
                    self.skip_whitespace_and_comments();
                    if self.peek_keyword("rule") {
                        match self.parse_named_rule() {
                            Ok(mut stmts) => {
                                let first = stmts.remove(0);
                                // Store extra heads in pending buffer
                                for s in stmts {
                                    self.pending.push(s);
                                }
                                return Ok(first);
                            }
                            Err(_e) => {
                                // Backtrack: might be a fact like (rule-name something)
                                self.pos = saved.0;
                                self.line = saved.1;
                                self.col = saved.2;
                            }
                        }
                    } else {
                        self.pos = saved.0;
                        self.line = saved.1;
                        self.col = saved.2;
                    }
                }

                let head = self.parse_expression()?;
                self.skip_whitespace_and_comments();

                // Check for "if" keyword → rule
                if self.peek_keyword("if") {
                    self.consume_keyword("if");
                    let mut body = Vec::new();
                    loop {
                        self.skip_whitespace_and_comments();
                        if self.peek_keyword("not-present") {
                            // Negation-as-absence: not-present (grid-cell ...)
                            self.consume_keyword("not-present");
                            self.skip_whitespace_and_comments();
                            body.push(Condition::NegatedPresent(self.parse_expression()?));
                        } else if self.peek_keyword("not") {
                            // Negated condition: not (penguin $x)
                            self.consume_keyword("not");
                            self.skip_whitespace_and_comments();
                            body.push(Condition::Negated(self.parse_expression()?));
                        } else if self.peek() == Some('(') {
                            // Save position to backtrack if not a guard
                            let saved = (self.pos, self.line, self.col);
                            self.advance(); // consume '('
                            self.skip_whitespace_and_comments();

                            if let Some(op) = self.try_parse_comparison_op() {
                                // Guard condition: (> $v 70)
                                let lhs = self.parse_neuron()?;
                                let rhs = self.parse_neuron()?;
                                self.skip_whitespace_and_comments();
                                self.expect(')')?;
                                body.push(Condition::Guard(op, lhs, rhs));
                            } else if let Some(agg_op) = self.try_parse_aggregate_op() {
                                // Aggregate: (count (bird $x) $x -> $n)
                                self.skip_whitespace_and_comments();
                                let pattern = self.parse_expression()?;
                                self.skip_whitespace_and_comments();
                                self.expect('$')?;
                                let bind_var = self.parse_name()?;
                                self.skip_whitespace_and_comments();
                                // Expect "->"
                                if self.peek() != Some('-') {
                                    return Err(self.err("expected '->' in aggregate"));
                                }
                                self.advance();
                                if self.peek() != Some('>') {
                                    return Err(self.err("expected '->' in aggregate"));
                                }
                                self.advance();
                                self.skip_whitespace_and_comments();
                                self.expect('$')?;
                                let result_var = self.parse_name()?;
                                self.skip_whitespace_and_comments();
                                self.expect(')')?;
                                body.push(Condition::Aggregate {
                                    op: agg_op,
                                    pattern,
                                    bind_var,
                                    result_var,
                                });
                            } else if self.peek_keyword("ends-with") {
                                // Check: (ends-with $n $d) — last digit
                                self.consume_keyword("ends-with");
                                self.skip_whitespace_and_comments();
                                let num = self.parse_neuron()?;
                                self.skip_whitespace_and_comments();
                                let digit = self.parse_neuron()?;
                                self.skip_whitespace_and_comments();
                                self.expect(')')?;
                                body.push(Condition::EndsWith(num, digit));
                            } else if let Some(arith_op) = self.try_parse_arithmetic_op() {
                                self.skip_whitespace_and_comments();
                                let lhs = self.parse_neuron()?;
                                self.skip_whitespace_and_comments();
                                if is_unary_arith(&arith_op) {
                                    // Unary: (sqrt $a $result)
                                    self.expect('$')?;
                                    let result_var = self.parse_name()?;
                                    self.skip_whitespace_and_comments();
                                    self.expect(')')?;
                                    body.push(Condition::Arithmetic {
                                        op: arith_op, lhs,
                                        rhs: Neuron::Value(QorValue::Int(0)),
                                        result_var,
                                    });
                                } else {
                                    // Binary: (+ $a $b $result)
                                    let rhs = self.parse_neuron()?;
                                    self.skip_whitespace_and_comments();
                                    self.expect('$')?;
                                    let result_var = self.parse_name()?;
                                    self.skip_whitespace_and_comments();
                                    self.expect(')')?;
                                    body.push(Condition::Arithmetic {
                                        op: arith_op, lhs, rhs, result_var,
                                    });
                                }
                            } else if self.peek_keyword("lookup") {
                                // KB lookup: (lookup "predicate" $subject $result)
                                self.consume_keyword("lookup");
                                self.skip_whitespace_and_comments();
                                // Parse predicate (quoted string or bare symbol)
                                let predicate = if self.peek() == Some('"') {
                                    self.advance(); // skip opening quote
                                    let mut pred = String::new();
                                    while let Some(c) = self.peek() {
                                        if c == '"' { self.advance(); break; }
                                        pred.push(c);
                                        self.advance();
                                    }
                                    pred
                                } else {
                                    self.parse_name()?
                                };
                                self.skip_whitespace_and_comments();
                                // Parse subject (a neuron — could be $var or symbol)
                                let subject = self.parse_neuron()?;
                                self.skip_whitespace_and_comments();
                                // Parse result variable
                                self.expect('$')?;
                                let result_var = self.parse_name()?;
                                self.skip_whitespace_and_comments();
                                self.expect(')')?;
                                body.push(Condition::Lookup {
                                    predicate,
                                    subject,
                                    result_var,
                                });
                            } else {
                                // Not a guard or aggregate — backtrack and parse as normal expression
                                self.pos = saved.0;
                                self.line = saved.1;
                                self.col = saved.2;
                                body.push(Condition::Positive(self.parse_expression()?));
                            }
                        } else {
                            break;
                        }
                    }
                    if body.is_empty() {
                        return Err(self.err("rule must have at least one condition after 'if'"));
                    }
                    let tv = self.try_parse_truth_value()?;
                    let rule = Statement::Rule { head, body, tv };
                    self.validate_rule(&rule);
                    Ok(rule)
                } else {
                    let tv = self.try_parse_truth_value()?;
                    let decay = self.try_parse_decay()?;
                    Ok(Statement::Fact { neuron: head, tv, decay })
                }
            }
            Some(ch) => Err(self.err(format!("expected '(' or '?', found '{}'", ch))),
            None => Err(self.err("unexpected end of input")),
        }
    }

    /// Parse a named rule: `(rule name body-conditions → head <tv>)`
    /// Already consumed `(` and verified `rule` keyword is next.
    fn parse_named_rule(&mut self) -> Result<Vec<Statement>, ParseError> {
        self.consume_keyword("rule");
        self.skip_whitespace_and_comments();

        // Parse rule name (informational — stored as comment-like fact)
        let _rule_name = self.parse_name()?;

        // Parse body conditions until we hit → or ->
        let mut body = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.peek_arrow() {
                break;
            }
            if self.peek() == Some(')') {
                return Err(self.err("named rule missing arrow (→ or ->)"));
            }
            if self.at_end() {
                return Err(self.err("unexpected end of input in named rule"));
            }
            // Parse body condition (same logic as `if` body parsing)
            if self.peek_keyword("not-present") {
                self.consume_keyword("not-present");
                self.skip_whitespace_and_comments();
                body.push(Condition::NegatedPresent(self.parse_expression()?));
            } else if self.peek_keyword("not") {
                self.consume_keyword("not");
                self.skip_whitespace_and_comments();
                body.push(Condition::Negated(self.parse_expression()?));
            } else if self.peek() == Some('(') {
                let saved = (self.pos, self.line, self.col);
                self.advance();
                self.skip_whitespace_and_comments();
                if let Some(op) = self.try_parse_comparison_op() {
                    let lhs = self.parse_neuron()?;
                    let rhs = self.parse_neuron()?;
                    self.skip_whitespace_and_comments();
                    self.expect(')')?;
                    body.push(Condition::Guard(op, lhs, rhs));
                } else {
                    self.pos = saved.0;
                    self.line = saved.1;
                    self.col = saved.2;
                    body.push(Condition::Positive(self.parse_expression()?));
                }
            } else {
                body.push(Condition::Positive(self.parse_expression()?));
            }
        }

        // Consume arrow
        self.consume_arrow();

        // Parse head expression(s) and optional truth value until closing ')'
        let mut heads = Vec::new();
        let mut tv: Option<TruthValue> = None;
        loop {
            self.skip_whitespace_and_comments();
            if self.peek() == Some(')') {
                self.advance(); // consume ')'
                break;
            }
            if self.peek() == Some('<') {
                tv = self.try_parse_truth_value()?;
            } else if self.peek() == Some('(') {
                heads.push(self.parse_expression()?);
            } else if self.at_end() {
                return Err(self.err("unterminated named rule"));
            } else {
                // Skip unexpected token
                self.advance();
            }
        }

        // Also check for truth value AFTER the closing ')'
        self.skip_whitespace_and_comments();
        if tv.is_none() {
            if let Ok(Some(outer_tv)) = self.try_parse_truth_value() {
                tv = Some(outer_tv);
            }
        }

        if heads.is_empty() {
            return Err(self.err("named rule must have at least one head after arrow"));
        }

        // Emit one Statement::Rule per head expression
        let mut stmts = Vec::new();
        for head in heads {
            let rule = Statement::Rule {
                head,
                body: body.clone(),
                tv: tv.clone(),
            };
            self.validate_rule(&rule);
            stmts.push(rule);
        }
        Ok(stmts)
    }

    /// Parse a @test block: `@test name given (...) expect (...) expect not (...)`
    fn parse_test_block(&mut self) -> Result<Statement, ParseError> {
        self.consume_keyword("@test");
        self.skip_whitespace_and_comments();
        let name = self.parse_name()?;
        let mut given = Vec::new();
        let mut expect = Vec::new();

        loop {
            self.skip_whitespace_and_comments();
            if self.peek_keyword("given") {
                self.consume_keyword("given");
                self.skip_whitespace_and_comments();
                given.push(self.parse_expression()?);
            } else if self.peek_keyword("expect") {
                self.consume_keyword("expect");
                self.skip_whitespace_and_comments();
                if self.peek_keyword("not") {
                    self.consume_keyword("not");
                    self.skip_whitespace_and_comments();
                    expect.push(TestExpect::Absent(self.parse_expression()?));
                } else {
                    expect.push(TestExpect::Present(self.parse_expression()?));
                }
            } else {
                break;
            }
        }

        Ok(Statement::Test { name, given, expect })
    }

    /// Validate a rule's variable bindings and emit warnings for issues.
    fn validate_rule(&mut self, stmt: &Statement) {
        if let Statement::Rule { head, body, .. } = stmt {
            let rule_line = self.line;
            let rule_col = self.col;

            // Collect all variables and where they appear
            let head_vars = collect_variables(head);

            let mut positive_bound: HashSet<String> = HashSet::new();
            let mut arith_result_vars: HashSet<String> = HashSet::new();
            let mut agg_result_vars: HashSet<String> = HashSet::new();
            let mut all_body_vars: HashSet<String> = HashSet::new();

            for cond in body {
                match cond {
                    Condition::Positive(n) => {
                        for v in collect_variables(n) {
                            positive_bound.insert(v.clone());
                            all_body_vars.insert(v);
                        }
                    }
                    Condition::Negated(n) | Condition::NegatedPresent(n) => {
                        for v in collect_variables(n) {
                            all_body_vars.insert(v);
                        }
                    }
                    Condition::Guard(_, lhs, rhs) => {
                        for v in collect_variables(lhs) {
                            all_body_vars.insert(v);
                        }
                        for v in collect_variables(rhs) {
                            all_body_vars.insert(v);
                        }
                    }
                    Condition::Aggregate { pattern, bind_var, result_var, .. } => {
                        for v in collect_variables(pattern) {
                            all_body_vars.insert(v);
                        }
                        agg_result_vars.insert(result_var.clone());
                        all_body_vars.insert(result_var.clone());
                        all_body_vars.insert(bind_var.clone());
                    }
                    Condition::Arithmetic { lhs, rhs, result_var, .. } => {
                        for v in collect_variables(lhs) {
                            all_body_vars.insert(v);
                        }
                        for v in collect_variables(rhs) {
                            all_body_vars.insert(v);
                        }
                        arith_result_vars.insert(result_var.clone());
                        all_body_vars.insert(result_var.clone());
                    }
                    Condition::Lookup { subject, result_var, .. } => {
                        for v in collect_variables(subject) {
                            all_body_vars.insert(v);
                        }
                        positive_bound.insert(result_var.clone());
                        all_body_vars.insert(result_var.clone());
                    }
                    Condition::EndsWith(n, d) => {
                        for v in collect_variables(n) {
                            all_body_vars.insert(v);
                        }
                        for v in collect_variables(d) {
                            all_body_vars.insert(v);
                        }
                    }
                }
            }

            // Check: head variables must be bound by positive conditions, arithmetic, or aggregates
            for v in &head_vars {
                if !positive_bound.contains(v) && !arith_result_vars.contains(v)
                    && !agg_result_vars.contains(v)
                {
                    self.warnings.push(ParseWarning {
                        message: format!("variable ${} in rule head is not bound by any positive body condition", v),
                        line: rule_line,
                        col: rule_col,
                    });
                }
            }

            // Check: negation variables must be bound by preceding positive conditions
            for cond in body {
                match cond {
                    Condition::Negated(n) | Condition::NegatedPresent(n) => {
                        for v in collect_variables(n) {
                            if !positive_bound.contains(&v) && !arith_result_vars.contains(&v) {
                                self.warnings.push(ParseWarning {
                                    message: format!("variable ${} in negation is not bound by a positive condition", v),
                                    line: rule_line,
                                    col: rule_col,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Check: single-use variables without _ prefix → warning
            let mut var_counts: HashMap<String, usize> = HashMap::new();
            for v in &head_vars {
                *var_counts.entry(v.clone()).or_insert(0) += 1;
            }
            for v in &all_body_vars {
                *var_counts.entry(v.clone()).or_insert(0) += 1;
            }
            for (var, count) in &var_counts {
                if *count == 1 && !var.starts_with('_') {
                    self.warnings.push(ParseWarning {
                        message: format!("variable ${} is used only once (use $_{} to suppress)", var, var),
                        line: rule_line,
                        col: rule_col,
                    });
                }
            }
        }
    }

    /// Apply module prefix to a neuron's first symbol (predicate).
    /// - Imported names → resolved to full qualified name
    /// - Inside @module: unqualified names → prefixed with module name
    fn prefix_neuron(&self, neuron: &Neuron) -> Neuron {
        if let Neuron::Expression(parts) = neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                // Resolve imports to full qualified name
                if let Some(full) = self.imports.get(pred.as_str()) {
                    let mut new_parts = parts.clone();
                    new_parts[0] = Neuron::Symbol(full.clone());
                    return Neuron::Expression(new_parts);
                }
                // Inside a module: prefix unqualified names
                if let Some(ref module) = self.current_module {
                    if !pred.contains('.') {
                        let mut new_parts = parts.clone();
                        new_parts[0] = Neuron::Symbol(format!("{}.{}", module, pred));
                        return Neuron::Expression(new_parts);
                    }
                }
            }
        }
        neuron.clone()
    }

    /// Apply module prefix to all predicates in a condition.
    fn prefix_condition(&self, cond: &Condition) -> Condition {
        match cond {
            Condition::Positive(n) => Condition::Positive(self.prefix_neuron(n)),
            Condition::Negated(n) => Condition::Negated(self.prefix_neuron(n)),
            Condition::NegatedPresent(n) => Condition::NegatedPresent(self.prefix_neuron(n)),
            Condition::Guard(op, lhs, rhs) => Condition::Guard(op.clone(), lhs.clone(), rhs.clone()),
            Condition::Aggregate { op, pattern, bind_var, result_var } => Condition::Aggregate {
                op: op.clone(),
                pattern: self.prefix_neuron(pattern),
                bind_var: bind_var.clone(),
                result_var: result_var.clone(),
            },
            Condition::Arithmetic { op, lhs, rhs, result_var } => Condition::Arithmetic {
                op: op.clone(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                result_var: result_var.clone(),
            },
            Condition::Lookup { predicate, subject, result_var } => Condition::Lookup {
                predicate: predicate.clone(),
                subject: self.prefix_neuron(subject),
                result_var: result_var.clone(),
            },
            Condition::EndsWith(n, d) => Condition::EndsWith(n.clone(), d.clone()),
        }
    }

    /// Apply module prefix / import resolution to all predicates in a statement.
    fn prefix_statement(&self, stmt: Statement) -> Statement {
        if self.current_module.is_none() && self.imports.is_empty() {
            return stmt;
        }
        match stmt {
            Statement::Fact { neuron, tv, decay } => Statement::Fact {
                neuron: self.prefix_neuron(&neuron),
                tv,
                decay,
            },
            Statement::Rule { head, body, tv } => Statement::Rule {
                head: self.prefix_neuron(&head),
                body: body.iter().map(|c| self.prefix_condition(c)).collect(),
                tv,
            },
            Statement::Query { pattern } => Statement::Query {
                pattern: self.prefix_neuron(&pattern),
            },
            other => other,
        }
    }

    /// Parse an entire .qor source string into a list of statements.
    fn parse_program(&mut self) -> Result<Vec<Statement>, ParseError> {
        let mut statements = Vec::new();

        loop {
            self.skip_whitespace_and_comments();
            if self.at_end() {
                break;
            }

            // Handle @ annotations before statements
            if self.peek() == Some('@') {
                if self.peek_keyword("@test") {
                    let stmt = self.parse_test_block()?;
                    self.strata.push(None);
                    statements.push(stmt);
                    continue;
                } else if self.peek_keyword("@stratum") {
                    self.consume_keyword("@stratum");
                    self.expect('(')?;
                    self.skip_whitespace_and_comments();
                    let n_neuron = self.parse_number()?;
                    let n = match n_neuron {
                        Neuron::Value(QorValue::Int(i)) => i as u32,
                        _ => return Err(self.err("@stratum requires an integer")),
                    };
                    self.skip_whitespace_and_comments();
                    self.expect(')')?;
                    self.pending_stratum = Some(n);
                    continue;
                } else if self.peek_keyword("@module") {
                    self.consume_keyword("@module");
                    self.skip_whitespace_and_comments();
                    let name = self.parse_name()?;
                    self.current_module = Some(name);
                    continue;
                } else if self.peek_keyword("@end") {
                    self.consume_keyword("@end");
                    self.current_module = None;
                    continue;
                } else if self.peek_keyword("@import") {
                    self.consume_keyword("@import");
                    self.skip_whitespace_and_comments();
                    let full = self.parse_qualified_name()?;
                    let short = full.rsplit('.').next().unwrap_or(&full).to_string();
                    self.imports.insert(short, full);
                    continue;
                }
                // Unknown @ annotation — fall through to parse_statement
            }

            let stmt = self.parse_statement()?;
            let stmt = self.prefix_statement(stmt);
            self.strata.push(self.pending_stratum.take());
            statements.push(stmt);

            // Drain any pending statements from multi-head named rules
            while let Some(extra) = self.pending.pop() {
                let extra = self.prefix_statement(extra);
                self.strata.push(None);
                statements.push(extra);
            }
        }

        Ok(statements)
    }

    /// Parse a qualified name like "rsi.overbought" — allows dots.
    fn parse_qualified_name(&mut self) -> Result<String, ParseError> {
        let mut name = self.parse_name()?;
        while self.peek() == Some('.') {
            self.advance();
            let part = self.parse_name()?;
            name.push('.');
            name.push_str(&part);
        }
        Ok(name)
    }
}

// -- Helpers --

/// Recursively collect all variable names from a Neuron tree.
fn collect_variables(neuron: &Neuron) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_variables_inner(neuron, &mut vars);
    vars
}

fn collect_variables_inner(neuron: &Neuron, vars: &mut HashSet<String>) {
    match neuron {
        Neuron::Variable(name) => { vars.insert(name.clone()); }
        Neuron::Expression(children) => {
            for child in children {
                collect_variables_inner(child, vars);
            }
        }
        _ => {}
    }
}

// -- Public API --

/// Parse a QOR source string into a list of statements.
pub fn parse(input: &str) -> Result<Vec<Statement>, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_program()
}

/// Parse a QOR source string into statements + warnings.
pub fn parse_with_warnings(input: &str) -> Result<(Vec<Statement>, Vec<ParseWarning>), ParseError> {
    let mut parser = Parser::new(input);
    let stmts = parser.parse_program()?;
    Ok((stmts, parser.warnings))
}

/// Parse a QOR source string into statements + warnings + strata annotations.
pub fn parse_with_strata(input: &str) -> Result<(Vec<Statement>, Vec<ParseWarning>, Vec<Option<u32>>), ParseError> {
    let mut parser = Parser::new(input);
    let stmts = parser.parse_program()?;
    Ok((stmts, parser.warnings, parser.strata))
}

/// Parse a single neuron expression from a string.
pub fn parse_neuron(input: &str) -> Result<Neuron, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_neuron()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_fact() {
        let stmts = parse("(bird tweety) <0.99>").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Fact { neuron, tv, .. } => {
                assert_eq!(neuron.to_string(), "(bird tweety)");
                let tv = tv.unwrap();
                assert!((tv.strength - 0.99).abs() < 0.01);
                assert!((tv.confidence - 0.9).abs() < 0.01);
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_fact_with_two_component_tv() {
        let stmts = parse("(bird tweety) <0.99, 0.85>").unwrap();
        match &stmts[0] {
            Statement::Fact { tv, .. } => {
                let tv = tv.unwrap();
                assert!((tv.strength - 0.99).abs() < 0.01);
                assert!((tv.confidence - 0.85).abs() < 0.01);
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_fact_no_tv() {
        let stmts = parse("(bird tweety)").unwrap();
        match &stmts[0] {
            Statement::Fact { neuron, tv, .. } => {
                assert_eq!(neuron.to_string(), "(bird tweety)");
                assert!(tv.is_none());
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_query() {
        let stmts = parse("? (bird tweety)").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Query { pattern } => {
                assert_eq!(pattern.to_string(), "(bird tweety)");
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn test_parse_query_with_variable() {
        let stmts = parse("? (bird $x)").unwrap();
        match &stmts[0] {
            Statement::Query { pattern } => {
                assert_eq!(pattern.to_string(), "(bird $x)");
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn test_parse_multiple_statements() {
        let input = r#"
            ;; This is a comment
            (bird tweety) <0.99>
            (bird eagle)  <0.95>
            (fish salmon) <0.99>

            ? (bird $x)
        "#;
        let stmts = parse(input).unwrap();
        assert_eq!(stmts.len(), 4);
    }

    #[test]
    fn test_parse_string_value() {
        let stmts = parse(r#"(claim "eagles can fly" wikipedia) <0.92>"#).unwrap();
        match &stmts[0] {
            Statement::Fact { neuron, .. } => {
                assert_eq!(neuron.to_string(), r#"(claim "eagles can fly" wikipedia)"#);
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_number_value() {
        let stmts = parse("(price bitcoin 95432) <1.0>").unwrap();
        match &stmts[0] {
            Statement::Fact { neuron, .. } => {
                assert_eq!(neuron.to_string(), "(price bitcoin 95432)");
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_error_unterminated() {
        let result = parse("(bird tweety");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("expected ')'"));
    }

    #[test]
    fn test_parse_error_empty_expression() {
        let result = parse("()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_comments_ignored() {
        let input = r#"
            ;; this is a comment
            (bird tweety) <0.99>
            ;; another comment
        "#;
        let stmts = parse(input).unwrap();
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn test_parse_milestone_program() {
        let input = r#"
            ;; Phase 3 milestone
            (bird tweety)     <0.99>
            (bird eagle)      <0.95>
            (fish salmon)     <0.99>

            ? (bird tweety)
            ? (bird $x)
            ? (fish $x)
            ? (mammal $x)
        "#;
        let stmts = parse(input).unwrap();
        assert_eq!(stmts.len(), 7); // 3 facts + 4 queries
    }

    #[test]
    fn test_parse_rule_single_condition() {
        let stmts = parse("(flies $x) if (bird $x) <0.95>").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Rule { head, body, tv } => {
                assert_eq!(head.to_string(), "(flies $x)");
                assert_eq!(body.len(), 1);
                assert_eq!(body[0].to_string(), "(bird $x)");
                let tv = tv.unwrap();
                assert!((tv.strength - 0.95).abs() < 0.01);
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_rule_multiple_conditions() {
        let stmts = parse("(can-fly $x) if (bird $x) (healthy $x) <0.90>").unwrap();
        match &stmts[0] {
            Statement::Rule { head, body, tv } => {
                assert_eq!(head.to_string(), "(can-fly $x)");
                assert_eq!(body.len(), 2);
                assert_eq!(body[0].to_string(), "(bird $x)");
                assert_eq!(body[1].to_string(), "(healthy $x)");
                let tv = tv.unwrap();
                assert!((tv.strength - 0.90).abs() < 0.01);
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_rule_no_tv() {
        let stmts = parse("(flies $x) if (bird $x)").unwrap();
        match &stmts[0] {
            Statement::Rule { head, body, tv } => {
                assert_eq!(head.to_string(), "(flies $x)");
                assert_eq!(body.len(), 1);
                assert!(tv.is_none());
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_rule_with_negation() {
        let stmts = parse("(can-fly $x) if (bird $x) not (penguin $x) <0.95>").unwrap();
        match &stmts[0] {
            Statement::Rule { head, body, .. } => {
                assert_eq!(head.to_string(), "(can-fly $x)");
                assert_eq!(body.len(), 2);
                assert_eq!(body[0].to_string(), "(bird $x)");
                assert_eq!(body[1].to_string(), "not (penguin $x)");
                assert!(matches!(&body[0], Condition::Positive(_)));
                assert!(matches!(&body[1], Condition::Negated(_)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_rule_empty_body_is_error() {
        let result = parse("(flies $x) if <0.95>");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_if_not_keyword_in_name() {
        let stmts = parse("(information important)").unwrap();
        match &stmts[0] {
            Statement::Fact { neuron, .. } => {
                assert_eq!(neuron.to_string(), "(information important)");
            }
            _ => panic!("expected Fact, not Rule"),
        }
    }

    #[test]
    fn test_parse_decay_annotation() {
        let stmts = parse("(news headline) <0.95> @decay 0.05").unwrap();
        match &stmts[0] {
            Statement::Fact { neuron, tv, decay } => {
                assert_eq!(neuron.to_string(), "(news headline)");
                assert!(tv.is_some());
                assert!((decay.unwrap() - 0.05).abs() < 0.001);
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_fact_no_decay() {
        let stmts = parse("(bird tweety) <0.99>").unwrap();
        match &stmts[0] {
            Statement::Fact { decay, .. } => {
                assert!(decay.is_none());
            }
            _ => panic!("expected Fact"),
        }
    }

    #[test]
    fn test_parse_guard_gt() {
        let stmts = parse("(overbought $x) if (rsi $x $v) (> $v 70) <0.90>").unwrap();
        match &stmts[0] {
            Statement::Rule { head, body, tv } => {
                assert_eq!(head.to_string(), "(overbought $x)");
                assert_eq!(body.len(), 2);
                assert_eq!(body[0].to_string(), "(rsi $x $v)");
                assert_eq!(body[1].to_string(), "(> $v 70)");
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Gt, _, _)));
                assert!(tv.is_some());
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_guard_lt() {
        let stmts = parse("(oversold $x) if (rsi $x $v) (< $v 30) <0.90>").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body[1].to_string(), "(< $v 30)");
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Lt, _, _)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_guard_ge_le() {
        let stmts = parse("(in-range $x) if (val $x $v) (>= $v 10) (<= $v 90)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Ge, _, _)));
                assert!(matches!(&body[2], Condition::Guard(ComparisonOp::Le, _, _)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_guard_eq_ne() {
        let stmts = parse("(exact $x) if (val $x $v) (== $v 50)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Eq, _, _)));
            }
            _ => panic!("expected Rule"),
        }
        let stmts = parse("(not-zero $x) if (val $x $v) (!= $v 0)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Ne, _, _)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_guard_mixed_with_negation() {
        let stmts = parse("(buy $x) if (rsi $x $v) (< $v 30) not (blacklisted $x) <0.85>").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[0], Condition::Positive(_)));
                assert!(matches!(&body[1], Condition::Guard(ComparisonOp::Lt, _, _)));
                assert!(matches!(&body[2], Condition::Negated(_)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_program_with_rules() {
        let input = r#"
            (bird tweety) <0.99>
            (bird eagle)  <0.95>
            (flies $x) if (bird $x) <0.95>
            ? (flies $x)
        "#;
        let stmts = parse(input).unwrap();
        assert_eq!(stmts.len(), 4);
    }

    // ── Aggregate parsing ──

    #[test]
    fn test_parse_count() {
        let stmts = parse("(total-birds $n) if (count (bird $x) $x -> $n) <0.99>").unwrap();
        match &stmts[0] {
            Statement::Rule { head, body, tv } => {
                assert_eq!(head.to_string(), "(total-birds $n)");
                assert_eq!(body.len(), 1);
                match &body[0] {
                    Condition::Aggregate { op, pattern, bind_var, result_var } => {
                        assert_eq!(*op, AggregateOp::Count);
                        assert_eq!(pattern.to_string(), "(bird $x)");
                        assert_eq!(bind_var, "x");
                        assert_eq!(result_var, "n");
                    }
                    _ => panic!("expected Aggregate"),
                }
                assert!(tv.is_some());
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_sum() {
        let stmts = parse("(total-score $t) if (sum (score $x $v) $v -> $t)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                match &body[0] {
                    Condition::Aggregate { op, pattern, bind_var, result_var } => {
                        assert_eq!(*op, AggregateOp::Sum);
                        assert_eq!(pattern.to_string(), "(score $x $v)");
                        assert_eq!(bind_var, "v");
                        assert_eq!(result_var, "t");
                    }
                    _ => panic!("expected Aggregate"),
                }
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_min_max() {
        let stmts = parse("(lowest $m) if (min (score $x $v) $v -> $m)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert!(matches!(&body[0], Condition::Aggregate { op: AggregateOp::Min, .. }));
            }
            _ => panic!("expected Rule"),
        }
        let stmts = parse("(highest $m) if (max (score $x $v) $v -> $m)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert!(matches!(&body[0], Condition::Aggregate { op: AggregateOp::Max, .. }));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_aggregate_display() {
        let cond = Condition::Aggregate {
            op: AggregateOp::Count,
            pattern: Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]),
            bind_var: "x".into(),
            result_var: "n".into(),
        };
        assert_eq!(cond.to_string(), "(count (bird $x) $x -> $n)");
    }

    #[test]
    fn test_roundtrip_aggregate() {
        let input = "(total $n) if (count (bird $x) $x -> $n) <0.99>";
        let stmts = parse(input).unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                let display = body[0].to_string();
                assert_eq!(display, "(count (bird $x) $x -> $n)");
            }
            _ => panic!("expected Rule"),
        }
    }

    // ── Arithmetic parsing ──

    #[test]
    fn test_parse_arithmetic_add() {
        let stmts = parse("(shifted $nr $c $color) if (grid-cell ti $r $c $color) (+ $r 1 $nr)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(matches!(&body[1], Condition::Arithmetic { .. }));
                assert_eq!(body[1].to_string(), "(+ $r 1 $nr)");
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_sub() {
        let stmts = parse("(cropped $nr $nc $color) if (grid-cell ti $r $c $color) (- $r $r0 $nr) (- $c $c0 $nc)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[1], Condition::Arithmetic { .. }));
                assert!(matches!(&body[2], Condition::Arithmetic { .. }));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_mul_div_mod() {
        let stmts = parse("(result $r) if (val $a $b) (* $a $b $prod) (/ $prod 2 $half) (% $half 3 $r)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 4);
                assert!(matches!(&body[1], Condition::Arithmetic { .. }));
                assert!(matches!(&body[2], Condition::Arithmetic { .. }));
                assert!(matches!(&body[3], Condition::Arithmetic { .. }));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_negative_number_not_confused() {
        // (-3) should parse as expression with negative number, NOT as arithmetic
        let stmts = parse("(neg $x) if (val $x -3)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Condition::Positive(_)));
            }
            _ => panic!("expected Rule"),
        }
    }

    #[test]
    fn test_parse_arithmetic_with_guards() {
        let stmts = parse("(predict-cell $nr $c $color) if (grid-cell ti $r $c $color) (+ $r 1 $nr) (>= $nr 0)").unwrap();
        match &stmts[0] {
            Statement::Rule { body, .. } => {
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[0], Condition::Positive(_)));
                assert!(matches!(&body[1], Condition::Arithmetic { .. }));
                assert!(matches!(&body[2], Condition::Guard(ComparisonOp::Ge, _, _)));
            }
            _ => panic!("expected Rule"),
        }
    }
}
