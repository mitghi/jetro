//! Canonical decomposition of a `Pipeline` into its parts for testing and
//! introspection. Not on the hot execution path.

use super::{BodyKernel, Pipeline, Sink, Stage};

impl Pipeline {
    /// Returns cloned copies of the pipeline's stages, per-stage body kernels, and terminal sink.
    ///
    /// Intended for testing and introspection; not called on the hot execution path.
    pub fn canonical(&self) -> (Vec<Stage>, Vec<BodyKernel>, Sink) {
        (
            self.stages.clone(),
            self.stage_kernels.clone(),
            self.sink.clone(),
        )
    }
}
