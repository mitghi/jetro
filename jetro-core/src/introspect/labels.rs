pub(crate) fn predicate_sink_label(sink: crate::builtins::BuiltinPredicateSink) -> &'static str {
    match sink {
        crate::builtins::BuiltinPredicateSink::Any => "any",
        crate::builtins::BuiltinPredicateSink::All => "all",
        crate::builtins::BuiltinPredicateSink::FindIndex => "find-index",
        crate::builtins::BuiltinPredicateSink::IndicesWhere => "indices-where",
        crate::builtins::BuiltinPredicateSink::FindOne => "find-one",
    }
}
