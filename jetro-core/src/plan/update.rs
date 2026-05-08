//! Shared planning helpers for functional batched updates.
//!
//! The parser owns syntax recognition. This module owns update-specific
//! semantic adapters that planner/compiler/executor code can share while the
//! first-class `UpdateBatch` path is being brought online.

use crate::parse::ast::{Expr, PatchOp, PathStep, Step};

/// Synthetic binding used to evaluate relative update expressions against the
/// selected value snapshot.
pub(crate) const UPDATE_FOCUS_BINDING: &str = "__update_focus";

/// Lightweight dependency summary for a functional update batch.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub(crate) struct UpdateDependencySummary {
    /// At least one selector, guard, dynamic index, or RHS reads `$`.
    pub(crate) reads_root: bool,
    /// At least one selected-object RHS or guard reads the original selected
    /// value snapshot through the synthetic update focus binding.
    pub(crate) reads_focus: bool,
    /// At least one guard, dynamic index, or RHS reads `@`.
    pub(crate) reads_current: bool,
    /// At least one expression depends on a runtime path step.
    pub(crate) has_dynamic_path: bool,
    /// At least one op has a guard.
    pub(crate) has_guards: bool,
}

/// Planner-visible trie for static update paths.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub(crate) struct UpdateTriePlan {
    pub(crate) root: UpdateTrieNode,
    pub(crate) op_count: usize,
    pub(crate) static_prefixes_only: bool,
    pub(crate) has_prefix_overlap: bool,
}

/// A node in the planner update trie.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub(crate) struct UpdateTrieNode {
    pub(crate) ops_here: Vec<usize>,
    pub(crate) fields: Vec<(String, UpdateTrieNode)>,
    pub(crate) indices: Vec<(i64, UpdateTrieNode)>,
}

/// Collects coarse dependencies used by update planning and safety tests.
pub(crate) fn analyze_update_batch(
    root: &Expr,
    selector: &[PathStep],
    ops: &[PatchOp],
) -> UpdateDependencySummary {
    let mut out = UpdateDependencySummary::default();
    collect_update_root(root, &mut out);
    collect_path(selector, &mut out);
    for op in ops {
        collect_path(&op.path, &mut out);
        collect_expr(&op.val, &mut out);
        if let Some(cond) = &op.cond {
            out.has_guards = true;
            collect_expr(cond, &mut out);
        }
    }
    out
}

/// Builds a source-order trie for relative update ops. Dynamic, wildcard, and
/// descendant steps stay executor-correct but are marked non-static so the
/// planner does not assume a single deterministic child prefix.
pub(crate) fn build_update_trie_plan(ops: &[PatchOp]) -> UpdateTriePlan {
    let mut out = UpdateTriePlan {
        root: UpdateTrieNode::default(),
        op_count: ops.len(),
        static_prefixes_only: true,
        has_prefix_overlap: false,
    };
    for (op_idx, op) in ops.iter().enumerate() {
        if !insert_static_path(&mut out.root, &op.path, 0, op_idx) {
            out.static_prefixes_only = false;
        }
    }
    out.has_prefix_overlap = !out.static_prefixes_only || trie_has_prefix_overlap(&out.root, false);
    out
}

fn insert_static_path(
    node: &mut UpdateTrieNode,
    path: &[PathStep],
    i: usize,
    op_idx: usize,
) -> bool {
    if i == path.len() {
        node.ops_here.push(op_idx);
        return true;
    }
    match &path[i] {
        PathStep::Field(name) => {
            let child = trie_field_child(node, name);
            insert_static_path(child, path, i + 1, op_idx)
        }
        PathStep::Index(idx) => {
            let child = trie_index_child(node, *idx);
            insert_static_path(child, path, i + 1, op_idx)
        }
        PathStep::DynIndex(_)
        | PathStep::Wildcard
        | PathStep::WildcardFilter(_)
        | PathStep::Descendant(_) => {
            node.ops_here.push(op_idx);
            false
        }
    }
}

fn trie_field_child<'a>(node: &'a mut UpdateTrieNode, name: &str) -> &'a mut UpdateTrieNode {
    if let Some(pos) = node.fields.iter().position(|(field, _)| field == name) {
        return &mut node.fields[pos].1;
    }
    node.fields
        .push((name.to_string(), UpdateTrieNode::default()));
    &mut node.fields.last_mut().unwrap().1
}

fn trie_index_child(node: &mut UpdateTrieNode, idx: i64) -> &mut UpdateTrieNode {
    if let Some(pos) = node.indices.iter().position(|(index, _)| *index == idx) {
        return &mut node.indices[pos].1;
    }
    node.indices.push((idx, UpdateTrieNode::default()));
    &mut node.indices.last_mut().unwrap().1
}

fn trie_has_prefix_overlap(node: &UpdateTrieNode, ancestor_has_op: bool) -> bool {
    if ancestor_has_op
        && (!node.ops_here.is_empty() || !node.fields.is_empty() || !node.indices.is_empty())
    {
        return true;
    }
    let here_has_op = ancestor_has_op || !node.ops_here.is_empty();
    node.fields
        .iter()
        .any(|(_, child)| trie_has_prefix_overlap(child, here_has_op))
        || node
            .indices
            .iter()
            .any(|(_, child)| trie_has_prefix_overlap(child, here_has_op))
}

fn collect_path(path: &[PathStep], out: &mut UpdateDependencySummary) {
    for step in path {
        match step {
            PathStep::DynIndex(expr) => {
                out.has_dynamic_path = true;
                collect_expr(expr, out);
            }
            PathStep::WildcardFilter(expr) => {
                out.has_dynamic_path = true;
                collect_expr(expr, out);
            }
            PathStep::Field(_)
            | PathStep::Index(_)
            | PathStep::Wildcard
            | PathStep::Descendant(_) => {}
        }
    }
}

fn collect_update_root(root: &Expr, out: &mut UpdateDependencySummary) {
    if !matches!(root, Expr::Root) {
        collect_expr(root, out);
    }
}

fn collect_expr(expr: &Expr, out: &mut UpdateDependencySummary) {
    match expr {
        Expr::Root => out.reads_root = true,
        Expr::Current => out.reads_current = true,
        Expr::Ident(name) if name == UPDATE_FOCUS_BINDING => out.reads_focus = true,
        Expr::Chain(base, steps) => {
            collect_expr(base, out);
            for step in steps {
                collect_step(step, out);
            }
        }
        Expr::BinOp(lhs, _, rhs) | Expr::Coalesce(lhs, rhs) => {
            collect_expr(lhs, out);
            collect_expr(rhs, out);
        }
        Expr::UnaryNeg(inner)
        | Expr::Not(inner)
        | Expr::Kind { expr: inner, .. }
        | Expr::Cast { expr: inner, .. } => collect_expr(inner, out),
        Expr::Object(fields) => {
            for field in fields {
                match field {
                    crate::parse::ast::ObjField::Kv { val, cond, .. } => {
                        collect_expr(val, out);
                        if let Some(cond) = cond {
                            collect_expr(cond, out);
                        }
                    }
                    crate::parse::ast::ObjField::Dynamic { key, val } => {
                        collect_expr(key, out);
                        collect_expr(val, out);
                    }
                    crate::parse::ast::ObjField::Spread(expr)
                    | crate::parse::ast::ObjField::SpreadDeep(expr) => collect_expr(expr, out),
                    crate::parse::ast::ObjField::Short(_) => {}
                }
            }
        }
        Expr::Array(items) => {
            for item in items {
                match item {
                    crate::parse::ast::ArrayElem::Expr(expr)
                    | crate::parse::ast::ArrayElem::Spread(expr) => collect_expr(expr, out),
                }
            }
        }
        Expr::Pipeline { base, steps } => {
            collect_expr(base, out);
            for step in steps {
                if let crate::parse::ast::PipeStep::Forward(expr) = step {
                    collect_expr(expr, out);
                }
            }
        }
        Expr::ListComp {
            expr, iter, cond, ..
        }
        | Expr::SetComp {
            expr, iter, cond, ..
        }
        | Expr::GenComp {
            expr, iter, cond, ..
        } => {
            collect_expr(iter, out);
            collect_expr(expr, out);
            if let Some(cond) = cond {
                collect_expr(cond, out);
            }
        }
        Expr::DictComp {
            key,
            val,
            iter,
            cond,
            ..
        } => {
            collect_expr(iter, out);
            collect_expr(key, out);
            collect_expr(val, out);
            if let Some(cond) = cond {
                collect_expr(cond, out);
            }
        }
        Expr::Lambda { body, .. } => collect_expr(body, out),
        Expr::Let { init, body, .. } => {
            collect_expr(init, out);
            collect_expr(body, out);
        }
        Expr::IfElse { cond, then_, else_ } => {
            collect_expr(cond, out);
            collect_expr(then_, out);
            collect_expr(else_, out);
        }
        Expr::Try { body, default } => {
            collect_expr(body, out);
            collect_expr(default, out);
        }
        Expr::GlobalCall { args, .. } => {
            for arg in args {
                match arg {
                    crate::parse::ast::Arg::Pos(expr) | crate::parse::ast::Arg::Named(_, expr) => {
                        collect_expr(expr, out)
                    }
                }
            }
        }
        Expr::Patch { root, ops } => {
            collect_update_root(root, out);
            for op in ops {
                collect_path(&op.path, out);
                collect_expr(&op.val, out);
                if let Some(cond) = &op.cond {
                    collect_expr(cond, out);
                }
            }
        }
        Expr::UpdateBatch {
            root,
            selector,
            ops,
        } => {
            collect_update_root(root, out);
            collect_path(selector, out);
            for op in ops {
                collect_path(&op.path, out);
                collect_expr(&op.val, out);
                if let Some(cond) = &op.cond {
                    collect_expr(cond, out);
                }
            }
        }
        Expr::FString(parts) => {
            for part in parts {
                if let crate::parse::ast::FStringPart::Interp { expr, .. } = part {
                    collect_expr(expr, out);
                }
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_expr(scrutinee, out);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_expr(guard, out);
                }
                collect_expr(&arm.body, out);
            }
        }
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::DeleteMark => {}
        Expr::Ident(_) => {}
    }
}

fn collect_step(step: &Step, out: &mut UpdateDependencySummary) {
    match step {
        Step::DynIndex(expr) | Step::InlineFilter(expr) => collect_expr(expr, out),
        Step::Method(_, args) | Step::OptMethod(_, args) => {
            for arg in args {
                match arg {
                    crate::parse::ast::Arg::Pos(expr) | crate::parse::ast::Arg::Named(_, expr) => {
                        collect_expr(expr, out)
                    }
                }
            }
        }
        Step::DeepMatch { arms, .. } => {
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_expr(guard, out);
                }
                collect_expr(&arm.body, out);
            }
        }
        Step::Field(_)
        | Step::OptField(_)
        | Step::Descendant(_)
        | Step::DescendAll
        | Step::Index(_)
        | Step::Slice(_, _, _)
        | Step::Wildcard
        | Step::Quantifier(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ast::{Arg, BinOp};

    #[test]
    fn dependency_summary_tracks_roots_current_dynamic_paths_and_guards() {
        let selector = vec![
            PathStep::Field("books".to_string()),
            PathStep::WildcardFilter(Box::new(Expr::BinOp(
                Box::new(Expr::Chain(
                    Box::new(Expr::Root),
                    vec![Step::Field("cutoff".to_string())],
                )),
                BinOp::Gt,
                Box::new(Expr::Int(0)),
            ))),
        ];
        let ops = vec![PatchOp {
            path: vec![PathStep::Field("tags".to_string())],
            val: Expr::Chain(
                Box::new(Expr::Current),
                vec![Step::Method(
                    "append".to_string(),
                    vec![Arg::Pos(Expr::Chain(
                        Box::new(Expr::Root),
                        vec![Step::Field("suffix".to_string())],
                    ))],
                )],
            ),
            cond: Some(Expr::BinOp(
                Box::new(Expr::Ident("year".to_string())),
                BinOp::Gt,
                Box::new(Expr::Int(1980)),
            )),
        }];

        let summary = analyze_update_batch(&Expr::Root, &selector, &ops);

        assert!(summary.reads_root);
        assert!(summary.reads_current);
        assert!(!summary.reads_focus);
        assert!(summary.has_dynamic_path);
        assert!(summary.has_guards);
    }

    #[test]
    fn dependency_summary_tracks_selected_focus_reads() {
        let selector = vec![PathStep::Field("books".to_string()), PathStep::Wildcard];
        let focus_read = Expr::Chain(
            Box::new(Expr::Ident(UPDATE_FOCUS_BINDING.to_string())),
            vec![Step::Field("tags".to_string())],
        );
        let ops = vec![PatchOp {
            path: vec![PathStep::Field("tags".to_string())],
            val: focus_read,
            cond: None,
        }];

        let summary = analyze_update_batch(&Expr::Root, &selector, &ops);

        assert!(summary.reads_focus);
        assert!(!summary.reads_current);
        assert!(!summary.reads_root);
    }

    #[test]
    fn dependency_summary_stays_empty_for_static_relative_updates() {
        let selector = vec![PathStep::Field("books".to_string()), PathStep::Wildcard];
        let ops = vec![PatchOp {
            path: vec![PathStep::Field("reviewed".to_string())],
            val: Expr::Bool(true),
            cond: None,
        }];

        let summary = analyze_update_batch(&Expr::Root, &selector, &ops);

        assert_eq!(summary, UpdateDependencySummary::default());
    }

    #[test]
    fn update_trie_groups_shared_static_prefixes() {
        let ops = vec![
            PatchOp {
                path: vec![
                    PathStep::Field("meta".to_string()),
                    PathStep::Field("score".to_string()),
                ],
                val: Expr::Int(1),
                cond: None,
            },
            PatchOp {
                path: vec![
                    PathStep::Field("meta".to_string()),
                    PathStep::Field("reviewed".to_string()),
                ],
                val: Expr::Bool(true),
                cond: None,
            },
        ];

        let trie = build_update_trie_plan(&ops);

        assert!(trie.static_prefixes_only);
        assert!(!trie.has_prefix_overlap);
        assert_eq!(trie.op_count, 2);
        assert_eq!(trie.root.fields.len(), 1);
        assert_eq!(trie.root.fields[0].0, "meta");
        assert_eq!(trie.root.fields[0].1.fields.len(), 2);
    }

    #[test]
    fn update_trie_marks_dynamic_and_prefix_overlap() {
        let ops = vec![
            PatchOp {
                path: vec![PathStep::Field("meta".to_string())],
                val: Expr::Object(Vec::new()),
                cond: None,
            },
            PatchOp {
                path: vec![
                    PathStep::Field("meta".to_string()),
                    PathStep::DynIndex(Expr::Ident("k".to_string())),
                ],
                val: Expr::Bool(true),
                cond: None,
            },
        ];

        let trie = build_update_trie_plan(&ops);

        assert!(!trie.static_prefixes_only);
        assert!(trie.has_prefix_overlap);
    }
}
