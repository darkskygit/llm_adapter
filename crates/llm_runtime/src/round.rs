use llm_adapter::core::StreamEvent;
use thiserror::Error;

use crate::{AccumulatedToolCall, ToolCallAccumulator, ToolLoopEvent};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RoundProcessorError {
  #[error("{message}")]
  StreamError { message: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundOutcome {
  pub tool_calls: Vec<AccumulatedToolCall>,
  pub final_done: Option<ToolLoopEvent>,
}

#[derive(Debug, Default)]
pub struct RoundProcessor {
  accumulator: ToolCallAccumulator,
  tool_calls: Vec<AccumulatedToolCall>,
  final_done: Option<ToolLoopEvent>,
}

impl RoundProcessor {
  pub fn process(&mut self, event: StreamEvent) -> Result<Vec<ToolLoopEvent>, RoundProcessorError> {
    match event {
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => {
        self.accumulator.feed_delta(call_id, name, arguments_delta);
        Ok(Vec::new())
      }
      StreamEvent::ToolCall {
        call_id,
        name,
        arguments,
        thought,
      } => {
        let call = self.accumulator.complete(call_id, name, arguments, thought);
        let event = ToolLoopEvent::from(call.clone());
        self.tool_calls.push(call);
        Ok(vec![event])
      }
      StreamEvent::Done { finish_reason, usage } => {
        self.final_done = Some(ToolLoopEvent::Done { finish_reason, usage });
        Ok(Vec::new())
      }
      StreamEvent::Error { message, .. } => Err(RoundProcessorError::StreamError { message }),
      StreamEvent::MessageStart { id, model } => Ok(vec![ToolLoopEvent::MessageStart { id, model }]),
      StreamEvent::TextDelta { text } => Ok(vec![ToolLoopEvent::TextDelta { text }]),
      StreamEvent::ReasoningDelta { text } => Ok(vec![ToolLoopEvent::ReasoningDelta { text }]),
      StreamEvent::ToolResult { .. } => Ok(Vec::new()),
      StreamEvent::Citation { index, url } => Ok(vec![ToolLoopEvent::Citation { index, url }]),
      StreamEvent::Usage { usage } => Ok(vec![ToolLoopEvent::Usage { usage }]),
    }
  }

  pub fn finish(mut self) -> RoundOutcome {
    self.tool_calls.extend(self.accumulator.drain_pending());
    RoundOutcome {
      tool_calls: self.tool_calls,
      final_done: self.final_done,
    }
  }
}

#[cfg(test)]
mod tests {
  use llm_adapter::core::StreamEvent;
  use serde_json::json;

  use super::RoundProcessor;
  use crate::ToolLoopEvent;

  #[test]
  fn should_hold_done_until_finish() {
    let mut processor = RoundProcessor::default();
    let emitted = processor
      .process(StreamEvent::Done {
        finish_reason: Some("tool_calls".to_string()),
        usage: None,
      })
      .unwrap();

    assert!(emitted.is_empty());
    let outcome = processor.finish();
    assert!(matches!(
      outcome.final_done,
      Some(ToolLoopEvent::Done {
        finish_reason: Some(reason),
        ..
      }) if reason == "tool_calls"
    ));
  }

  #[test]
  fn should_emit_completed_tool_call() {
    let mut processor = RoundProcessor::default();

    processor
      .process(StreamEvent::ToolCallDelta {
        call_id: "call_1".to_string(),
        name: Some("doc_read".to_string()),
        arguments_delta: "{\"doc_id\":\"a1\"}".to_string(),
      })
      .unwrap();

    let emitted = processor
      .process(StreamEvent::ToolCall {
        call_id: "call_1".to_string(),
        name: "doc_read".to_string(),
        arguments: json!({ "doc_id": "a1" }),
        thought: Some("need context".to_string()),
      })
      .unwrap();

    assert!(matches!(
      emitted.as_slice(),
      [ToolLoopEvent::ToolCall {
        call_id,
        name,
        thought: Some(thought),
        ..
      }] if call_id == "call_1" && name == "doc_read" && thought == "need context"
    ));
  }
}
