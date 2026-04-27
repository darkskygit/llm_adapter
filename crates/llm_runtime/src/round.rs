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

#[derive(Debug, Default)]
pub struct StreamRoundRunner {
  processor: RoundProcessor,
}

impl StreamRoundRunner {
  pub fn process_event<E, Emit>(&mut self, event: StreamEvent, emit: Emit) -> Result<(), E>
  where
    E: From<RoundProcessorError>,
    Emit: FnMut(ToolLoopEvent) -> Result<(), E>,
  {
    self.process_event_with(event, E::from, emit)
  }

  pub fn process_event_with<E, MapError, Emit>(
    &mut self,
    event: StreamEvent,
    map_error: MapError,
    mut emit: Emit,
  ) -> Result<(), E>
  where
    MapError: FnOnce(RoundProcessorError) -> E,
    Emit: FnMut(ToolLoopEvent) -> Result<(), E>,
  {
    for loop_event in self.processor.process(event).map_err(map_error)? {
      emit(loop_event)?;
    }

    Ok(())
  }

  pub fn finish(self) -> RoundOutcome {
    self.processor.finish()
  }
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

pub fn run_stream_round<E, Events, Emit>(events: Events, mut emit: Emit) -> Result<RoundOutcome, E>
where
  Events: IntoIterator<Item = Result<StreamEvent, E>>,
  Emit: FnMut(ToolLoopEvent) -> Result<(), E>,
  E: From<RoundProcessorError>,
{
  let mut processor = RoundProcessor::default();

  for event in events {
    for loop_event in processor.process(event?)? {
      emit(loop_event)?;
    }
  }

  Ok(processor.finish())
}

#[cfg(test)]
mod tests {
  use llm_adapter::core::StreamEvent;
  use serde_json::json;

  use super::{RoundProcessor, RoundProcessorError, run_stream_round};
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

  #[test]
  fn should_run_stream_round_and_keep_done_for_finish() {
    let events = vec![
      Ok(StreamEvent::TextDelta {
        text: "hello".to_string(),
      }),
      Ok(StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      }),
    ];
    let mut emitted = Vec::new();

    let outcome = run_stream_round(events, |event| {
      emitted.push(event);
      Ok::<_, RoundProcessorError>(())
    })
    .unwrap();

    assert!(matches!(
      emitted.as_slice(),
      [ToolLoopEvent::TextDelta { text }] if text == "hello"
    ));
    assert!(matches!(
      outcome.final_done,
      Some(ToolLoopEvent::Done {
        finish_reason: Some(reason),
        ..
      }) if reason == "stop"
    ));
  }

  #[test]
  fn should_run_stream_round_incrementally() {
    let mut runner = super::StreamRoundRunner::default();
    let mut emitted = Vec::new();

    runner
      .process_event(
        StreamEvent::TextDelta {
          text: "hello".to_string(),
        },
        |event| {
          emitted.push(event);
          Ok::<_, RoundProcessorError>(())
        },
      )
      .unwrap();

    runner
      .process_event(
        StreamEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        },
        |event| {
          emitted.push(event);
          Ok::<_, RoundProcessorError>(())
        },
      )
      .unwrap();

    assert!(matches!(
      emitted.as_slice(),
      [ToolLoopEvent::TextDelta { text }] if text == "hello"
    ));
    assert!(matches!(
      runner.finish().final_done,
      Some(ToolLoopEvent::Done {
        finish_reason: Some(reason),
        ..
      }) if reason == "stop"
    ));
  }

  #[test]
  fn should_roundtrip_tool_loop_event_serde_shape() {
    let event = ToolLoopEvent::ToolCall {
      call_id: "call_1".to_string(),
      name: "doc_read".to_string(),
      arguments: json!({ "doc_id": "a1" }),
      arguments_text: Some("{\"doc_id\":\"a1\"}".to_string()),
      arguments_error: None,
      thought: Some("need context".to_string()),
    };

    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], "tool_call");
    assert_eq!(serde_json::from_value::<ToolLoopEvent>(value).unwrap(), event);
  }
}
