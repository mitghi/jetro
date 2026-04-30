use super::{BodyKernel, Pipeline, Sink, Stage};

impl Pipeline {
    /// Decompose a pipeline into its canonical `(stages, kernels, sink)`
    /// triple. Pure function: it does not mutate `self`.
    ///
    /// Fused sink variants were removed in Tier 3, so every pipeline is already
    /// in base form. This stable name remains the shared canonical-view
    /// boundary for composed Val/tape runners and columnar fast paths.
    pub fn canonical(&self) -> (Vec<Stage>, Vec<BodyKernel>, Sink) {
        (
            self.stages.clone(),
            self.stage_kernels.clone(),
            self.sink.clone(),
        )
    }
}
