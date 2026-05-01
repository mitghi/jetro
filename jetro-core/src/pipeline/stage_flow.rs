pub(crate) enum StageFlow<T> {
    Continue(T),
    SkipRow,
    Stop,
    TerminalCollected,
}
