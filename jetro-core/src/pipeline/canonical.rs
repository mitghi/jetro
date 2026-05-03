//! Canonical decomposition of a `Pipeline` into its parts for testing and
//! introspection. Not on the hot execution path.

use super::{BodyKernel, Pipeline, Sink, Stage};

impl Pipeline {
    /// Returns cloned copies of the pipeline's stages, body kernels, and terminal sink for testing and introspection.
    pub fn canonical(&self) -> (Vec<Stage>, Vec<BodyKernel>, Sink) {
        (
            self.stages.clone(),
            self.stage_kernels.clone(),
            self.sink.clone(),
        )
    }
}
