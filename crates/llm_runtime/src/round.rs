use llm_adapter::{core::StreamEvent, middleware::StreamPipeline};
use thiserror::Error;

use crate::{AccumulatedToolCall, ToolLoopEvent, tool_call::ToolCallAccumulator};

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
struct RoundProcessor {
  accumulator: ToolCallAccumulator,
  tool_calls: Vec<AccumulatedToolCall>,
  final_done: Option<ToolLoopEvent>,
}

#[derive(Debug, Default)]
struct StreamRoundRunner {
  processor: RoundProcessor,
}

impl StreamRoundRunner {
  #[cfg(test)]
  fn process_event<E, Emit>(&mut self, event: StreamEvent, emit: Emit) -> Result<(), E>
  where
    E: From<RoundProcessorError>,
    Emit: FnMut(ToolLoopEvent) -> Result<(), E>,
  {
    self.process_event_with(event, E::from, emit)
  }

  fn process_event_with<E, MapError, Emit>(
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

  fn finish(self) -> RoundOutcome {
    self.processor.finish()
  }
}

impl RoundProcessor {
  fn process(&mut self, event: StreamEvent) -> Result<Vec<ToolLoopEvent>, RoundProcessorError> {
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

  fn finish(mut self) -> RoundOutcome {
    self.tool_calls.extend(self.accumulator.drain_pending());
    RoundOutcome {
      tool_calls: self.tool_calls,
      final_done: self.final_done,
    }
  }
}

#[cfg(test)]
fn run_stream_round<E, Events, Emit>(events: Events, mut emit: Emit) -> Result<RoundOutcome, E>
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

pub fn run_prepared_stream_round_with_fallback<E, Dispatch, Abort, AbortError, MapRoundError, Emit>(
  pipelines: &mut [StreamPipeline],
  mut dispatch_round: Dispatch,
  mut should_abort: Abort,
  mut abort_error: AbortError,
  mut map_round_error: MapRoundError,
  mut emit: Emit,
) -> Result<RoundOutcome, E>
where
  Dispatch: FnMut(&mut dyn FnMut(usize, StreamEvent) -> Result<bool, E>) -> Result<usize, E>,
  Abort: FnMut() -> bool,
  AbortError: FnMut() -> E,
  MapRoundError: FnMut(RoundProcessorError) -> E,
  Emit: FnMut(&ToolLoopEvent) -> Result<(), E>,
{
  let mut runners = pipelines
    .iter()
    .map(|_| StreamRoundRunner::default())
    .collect::<Vec<_>>();

  let selected_index = dispatch_round(&mut |index, event| {
    if should_abort() {
      return Err(abort_error());
    }

    let mut round_emitted = false;
    for event in pipelines[index].process(event) {
      runners[index].process_event_with(
        event,
        |error| map_round_error(error),
        |loop_event| {
          round_emitted = true;
          emit(&loop_event)
        },
      )?;
    }

    Ok(round_emitted)
  })?;

  if !should_abort() {
    for event in pipelines[selected_index].finish() {
      if should_abort() {
        break;
      }

      runners[selected_index].process_event_with(
        event,
        |error| map_round_error(error),
        |loop_event| emit(&loop_event),
      )?;
    }
  }

  Ok(runners.swap_remove(selected_index).finish())
}

#[cfg(test)]
mod tests {
  use llm_adapter::{
    core::StreamEvent,
    middleware::{MiddlewareConfig, StreamPipeline, resolve_stream_middleware_chain},
  };
  use serde_json::json;

  use super::{RoundProcessor, RoundProcessorError, run_prepared_stream_round_with_fallback, run_stream_round};
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
  fn should_run_prepared_stream_round_with_fallback_pipeline() {
    let mut pipelines = vec![StreamPipeline::new(
      resolve_stream_middleware_chain(&["stream_event_normalize".to_string()]).unwrap(),
      MiddlewareConfig::default(),
    )];
    let mut emitted = Vec::new();

    let outcome = run_prepared_stream_round_with_fallback(
      &mut pipelines,
      |on_event| {
        on_event(
          0,
          StreamEvent::MessageStart {
            id: Some("chat_1".to_string()),
            model: Some("gpt-4.1".to_string()),
          },
        )?;
        on_event(
          0,
          StreamEvent::TextDelta {
            text: "hello".to_string(),
          },
        )?;
        Ok::<_, String>(0)
      },
      || false,
      || "aborted".to_string(),
      |error| error.to_string(),
      |event| {
        emitted.push(event.clone());
        Ok(())
      },
    )
    .unwrap();

    assert!(outcome.tool_calls.is_empty());
    assert_eq!(
      emitted,
      vec![
        ToolLoopEvent::MessageStart {
          id: Some("chat_1".to_string()),
          model: Some("gpt-4.1".to_string()),
        },
        ToolLoopEvent::TextDelta {
          text: "hello".to_string(),
        },
      ]
    );
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
