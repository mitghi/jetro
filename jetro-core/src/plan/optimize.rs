//! Rule-based optimizer for `LogicalPlan`.
//!
//! The optimizer applies a set of rewrite rules bottom-up until no rule fires
//! (fixed-point iteration). Each rule receives ownership of a node and returns
//! either a rewritten plan (`Ok`) or the original plan unchanged (`Err`).

use crate::ir::logical::LogicalPlan;

/// A single rewrite rule applied to a `LogicalPlan` node.
pub(crate) trait Rule: Send + Sync {
    /// Try to rewrite `plan`. Return `Ok(new_plan)` if the rule fired; `Err(unchanged)` if not.
    fn apply(&self, plan: LogicalPlan) -> Result<LogicalPlan, LogicalPlan>;
}

/// Applies a set of rules to a `LogicalPlan` until no rule fires (fixed-point).
pub(crate) struct Optimizer {
    rules: Vec<Box<dyn Rule>>,
}

impl Optimizer {
    pub(crate) fn new(rules: Vec<Box<dyn Rule>>) -> Self {
        Self { rules }
    }

    /// Returns a default optimizer with the standard rule set.
    pub(crate) fn default_rules() -> Self {
        Self::new(vec![
            Box::new(rules::StrengthReduce),
            Box::new(rules::RedundantOps),
            Box::new(rules::FilterBeforeMap),
        ])
    }

    /// Optimizes `plan` bottom-up; repeats until no rule fires.
    pub(crate) fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        self.optimize_node(plan)
    }

    fn optimize_node(&self, plan: LogicalPlan) -> LogicalPlan {
        // 1. Recurse into children first (bottom-up)
        let plan = self.optimize_children(plan);
        // 2. Apply rules at this node until fixed point
        self.apply_rules(plan)
    }

    fn optimize_children(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan.take_input() {
            Ok((child, shell)) => {
                let optimized_child = self.optimize_node(*child);
                shell.with_input(optimized_child)
            }
            Err(leaf) => leaf,
        }
    }

    fn apply_rules(&self, mut plan: LogicalPlan) -> LogicalPlan {
        let mut changed = true;
        while changed {
            changed = false;
            for rule in &self.rules {
                match rule.apply(plan) {
                    Ok(new_plan) => {
                        plan = new_plan;
                        changed = true;
                    }
                    Err(unchanged) => {
                        plan = unchanged;
                    }
                }
            }
        }
        plan
    }
}

pub(crate) mod rules {
    use super::Rule;
    use crate::ir::logical::LogicalPlan;

    /// Replaces expensive patterns with cheaper equivalents:
    /// - `Sort(asc, no key) → Take(1)` → `Min`
    /// - `Sort(desc, no key) → Take(1)` → `Max`
    /// - `Sort(asc, no key) → First` → `Min`
    /// - `Sort(desc, no key) → First` → `Max`
    pub(crate) struct StrengthReduce;

    impl Rule for StrengthReduce {
        fn apply(&self, plan: LogicalPlan) -> Result<LogicalPlan, LogicalPlan> {
            // Match Take(1, Sort(...)) or First(Sort(...))
            // Track which outer form we had so we can rebuild correctly on Err.
            enum Outer { Take1, First }
            let (outer, inner) = match plan {
                LogicalPlan::Take { n: 1, input } => (Outer::Take1, input),
                LogicalPlan::First(input) => (Outer::First, input),
                other => return Err(other),
            };

            match *inner {
                LogicalPlan::Sort { input: sort_input, spec } => {
                    // Only collapse when there is no key expression (identity sort).
                    // A keyed sort changes semantics — we cannot simply emit Min/Max
                    // which operate on the raw value, not the key projection.
                    if spec.key.is_none() {
                        if spec.descending {
                            Ok(LogicalPlan::Max(sort_input))
                        } else {
                            Ok(LogicalPlan::Min(sort_input))
                        }
                    } else {
                        // Cannot collapse — rebuild the original outer node with sort inside.
                        let rebuilt_sort = LogicalPlan::Sort { input: sort_input, spec };
                        let rebuilt = match outer {
                            Outer::Take1 => LogicalPlan::Take {
                                n: 1,
                                input: Box::new(rebuilt_sort),
                            },
                            Outer::First => LogicalPlan::First(Box::new(rebuilt_sort)),
                        };
                        Err(rebuilt)
                    }
                }
                other => {
                    // Inner was not a Sort — rebuild the original outer node.
                    let rebuilt = match outer {
                        Outer::Take1 => LogicalPlan::Take {
                            n: 1,
                            input: Box::new(other),
                        },
                        Outer::First => LogicalPlan::First(Box::new(other)),
                    };
                    Err(rebuilt)
                }
            }
        }
    }

    /// Cancels algebraically inverse operations:
    /// - `Reverse(Reverse(x))` → `x`
    /// - `Take(n1)(Take(n2)(x))` → `Take(min(n1, n2))(x)`
    /// - `Skip(0)(x)` → `x`
    pub(crate) struct RedundantOps;

    impl Rule for RedundantOps {
        fn apply(&self, plan: LogicalPlan) -> Result<LogicalPlan, LogicalPlan> {
            match plan {
                LogicalPlan::Reverse { input } => {
                    if let LogicalPlan::Reverse { input: inner } = *input {
                        Ok(*inner)
                    } else {
                        Err(LogicalPlan::Reverse { input })
                    }
                }
                LogicalPlan::Take { n: n1, input } => {
                    if let LogicalPlan::Take { n: n2, input: inner } = *input {
                        Ok(LogicalPlan::Take {
                            n: n1.min(n2),
                            input: inner,
                        })
                    } else {
                        Err(LogicalPlan::Take { n: n1, input })
                    }
                }
                LogicalPlan::Skip { n: 0, input } => Ok(*input),
                _ => Err(plan),
            }
        }
    }

    /// Moves `Filter` before `Map` when the predicate does not reference
    /// the map's output. This reduces rows that need to be mapped.
    pub(crate) struct FilterBeforeMap;

    impl Rule for FilterBeforeMap {
        fn apply(&self, plan: LogicalPlan) -> Result<LogicalPlan, LogicalPlan> {
            // Filter { input: Map { input: x, projection: f }, predicate: p }
            // →  Map { input: Filter { input: x, predicate: p }, projection: f }
            // only when p does not reference the map output
            match plan {
                LogicalPlan::Filter { input, predicate }
                    if matches!(*input, LogicalPlan::Map { .. }) =>
                {
                    if let LogicalPlan::Map {
                        input: map_input,
                        projection,
                    } = *input
                    {
                        if is_independent_predicate(&predicate) {
                            return Ok(LogicalPlan::Map {
                                input: Box::new(LogicalPlan::Filter {
                                    input: map_input,
                                    predicate,
                                }),
                                projection,
                            });
                        }
                        return Err(LogicalPlan::Filter {
                            input: Box::new(LogicalPlan::Map {
                                input: map_input,
                                projection,
                            }),
                            predicate,
                        });
                    }
                    unreachable!()
                }
                _ => Err(plan),
            }
        }
    }

    /// Returns `true` when `expr` can be safely moved before a `Map` stage —
    /// it accesses only original source fields, not computed map outputs.
    ///
    /// Conservative: only allow literals, root/current references, simple field
    /// chains, and binary/unary combinations of the above.
    fn is_independent_predicate(expr: &crate::parse::ast::Expr) -> bool {
        use crate::parse::ast::Expr;
        match expr {
            // Literals — always safe
            Expr::Null | Expr::Bool(_) | Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => true,
            // Document root and current-item reference — safe to move
            Expr::Root | Expr::Current => true,
            // Plain identifier that resolves against the current row — safe
            Expr::Ident(_) => true,
            // Simple field navigation chains (e.g. `@.price`)
            Expr::Chain(base, steps) => {
                use crate::parse::ast::Step;
                let base_ok = is_independent_predicate(base);
                let steps_ok = steps.iter().all(|s| matches!(s, Step::Field(_) | Step::Index(_)));
                base_ok && steps_ok
            }
            // Compound expressions — recurse
            Expr::BinOp(lhs, _, rhs) => {
                is_independent_predicate(lhs) && is_independent_predicate(rhs)
            }
            Expr::Not(inner) => is_independent_predicate(inner),
            Expr::UnaryNeg(inner) => is_independent_predicate(inner),
            Expr::Coalesce(lhs, rhs) => {
                is_independent_predicate(lhs) && is_independent_predicate(rhs)
            }
            // Anything with method calls, lambdas, let-bindings, comprehensions — conservatively false
            _ => false,
        }
    }
}
