pub(crate) fn predicate_sink_label(sink: crate::builtins::BuiltinPredicateSink) -> &'static str {
    match sink {
        crate::builtins::BuiltinPredicateSink::Any => "any",
        crate::builtins::BuiltinPredicateSink::All => "all",
        crate::builtins::BuiltinPredicateSink::FindIndex => "find-index",
        crate::builtins::BuiltinPredicateSink::IndicesWhere => "indices-where",
        crate::builtins::BuiltinPredicateSink::FindOne => "find-one",
    }
}

pub(crate) fn numeric_reducer_label(
    reducer: crate::builtins::BuiltinNumericReducer,
) -> &'static str {
    match reducer {
        crate::builtins::BuiltinNumericReducer::Sum => "sum",
        crate::builtins::BuiltinNumericReducer::Avg => "avg",
        crate::builtins::BuiltinNumericReducer::Min => "min",
        crate::builtins::BuiltinNumericReducer::Max => "max",
    }
}

pub(crate) fn num_op_label(op: crate::exec::pipeline::NumOp) -> &'static str {
    match op {
        crate::exec::pipeline::NumOp::Sum => "sum",
        crate::exec::pipeline::NumOp::Min => "min",
        crate::exec::pipeline::NumOp::Max => "max",
        crate::exec::pipeline::NumOp::Avg => "avg",
    }
}

pub(crate) fn arg_extreme_sink_label(sink: crate::builtins::BuiltinArgExtremeSink) -> &'static str {
    match sink {
        crate::builtins::BuiltinArgExtremeSink::MaxBy => "max-by",
        crate::builtins::BuiltinArgExtremeSink::MinBy => "min-by",
    }
}

pub(crate) fn streaming_boundary_label(
    boundary: crate::builtins::BuiltinStreamingBoundary,
) -> &'static str {
    match boundary {
        crate::builtins::BuiltinStreamingBoundary::RowLocal => "row-local",
        crate::builtins::BuiltinStreamingBoundary::SourceStream => "source-stream",
        crate::builtins::BuiltinStreamingBoundary::BoundedState => "bounded-state",
        crate::builtins::BuiltinStreamingBoundary::PrefixState => "prefix-state",
        crate::builtins::BuiltinStreamingBoundary::Expanding => "expanding",
        crate::builtins::BuiltinStreamingBoundary::FullInputState => "full-input-state",
        crate::builtins::BuiltinStreamingBoundary::FullInputOrder => "full-input-order",
        crate::builtins::BuiltinStreamingBoundary::LegacyMaterialized => "legacy-materialized",
    }
}
