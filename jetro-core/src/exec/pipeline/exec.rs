//! Physical execution dispatch — routes each pipeline to the right backend
//! without runtime fallthrough. `exec_path` is computed once at lower time by
//! `select_exec_path`; `run_with_env` does a single match and never re-examines
//! the stage list.

use crate::{
    data::context::{Env, EvalError},
    data::value::Val,
    vm::VM,
};

use super::columnar;
use super::composed;
use super::indexed_exec;
use super::materialized_exec;
use super::{PhysicalExecPath, Pipeline, PipelineData};

impl Pipeline {
    /// Executes the pipeline against `root` using a freshly constructed environment.
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        self.run_with(root, None)
    }

    /// Executes the pipeline against `root`, optionally with a `PipelineData` cache for columnar promotion.
    pub fn run_with(&self, root: &Val, cache: Option<&dyn PipelineData>) -> Result<Val, EvalError> {
        let env = Env::new(root.clone());
        let mut vm = VM::new();
        self.run_with_env_and_vm(root, &env, cache, &mut vm)
    }

    /// Dispatches to the first applicable backend for this pipeline shape.
    /// The path was classified at lower time; no stage-list re-inspection occurs here.
    #[allow(dead_code)]
    pub fn run_with_env(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
    ) -> Result<Val, EvalError> {
        let mut vm = VM::new();
        self.run_with_env_and_vm(root, base_env, cache, &mut vm)
    }

    /// Dispatches with caller-owned VM state for compiled fallback programs.
    pub(crate) fn run_with_env_and_vm(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
        vm: &mut VM,
    ) -> Result<Val, EvalError> {
        match self.exec_path {
            PhysicalExecPath::Indexed => {
                if let Some(out) = indexed_exec::run(self, root, base_env, vm) {
                    return out;
                }
                self.run_columnar_or_below(root, base_env, cache, vm)
            }
            PhysicalExecPath::Columnar => self.run_columnar_or_below(root, base_env, cache, vm),
            PhysicalExecPath::Composed => composed::run_with_vm(self, root, base_env, vm)
                .unwrap_or_else(|| materialized_exec::run(self, root, base_env, vm)),
            PhysicalExecPath::Legacy => materialized_exec::run(self, root, base_env, vm),
        }
    }

    /// Tries the columnar paths then composed then legacy.
    /// Shared by `Indexed` (after indexed misses) and `Columnar`.
    fn run_columnar_or_below(
        &self,
        root: &Val,
        base_env: &Env,
        cache: Option<&dyn PipelineData>,
        vm: &mut VM,
    ) -> Result<Val, EvalError> {
        if let Some(out) = columnar::run_cached(self, root, cache) {
            return out;
        }
        if cache.is_none() {
            if let Some(out) = columnar::run_uncached(self, root) {
                return out;
            }
        }
        composed::run_with_vm(self, root, base_env, vm)
            .unwrap_or_else(|| materialized_exec::run(self, root, base_env, vm))
    }
}
