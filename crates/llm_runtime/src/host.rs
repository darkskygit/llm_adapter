use crate::{AccumulatedToolCall, ToolExecutionResult, ToolLoopEvent};

pub trait ToolExecutor<E> {
  fn execute(&mut self, call: &AccumulatedToolCall) -> Result<ToolExecutionResult, E>;
}

impl<E, F> ToolExecutor<E> for F
where
  F: FnMut(&AccumulatedToolCall) -> Result<ToolExecutionResult, E>,
{
  fn execute(&mut self, call: &AccumulatedToolCall) -> Result<ToolExecutionResult, E> {
    self(call)
  }
}

pub trait EventSink<E> {
  fn emit(&mut self, event: &ToolLoopEvent) -> Result<(), E>;
}

impl<E, F> EventSink<E> for F
where
  F: FnMut(&ToolLoopEvent) -> Result<(), E>,
{
  fn emit(&mut self, event: &ToolLoopEvent) -> Result<(), E> {
    self(event)
  }
}
