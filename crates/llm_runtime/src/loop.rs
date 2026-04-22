use llm_adapter::core::CoreMessage;

use crate::{
  EventSink, RoundOutcome, ToolExecutor, ToolLoopEvent, ToolResultMessage, append_tool_turns,
};

pub fn run_tool_loop<E, DispatchRound, ExecuteTool, EmitEvent, MaxStepsError>(
  messages: &mut Vec<CoreMessage>,
  max_steps: usize,
  mut dispatch_round: DispatchRound,
  mut execute_tool: ExecuteTool,
  mut emit_event: EmitEvent,
  mut max_steps_error: MaxStepsError,
) -> Result<(), E>
where
  DispatchRound: FnMut(&[CoreMessage]) -> Result<RoundOutcome, E>,
  ExecuteTool: ToolExecutor<E>,
  EmitEvent: EventSink<E>,
  MaxStepsError: FnMut() -> E,
{
  for step in 0..max_steps {
    let outcome = dispatch_round(messages)?;
    if outcome.tool_calls.is_empty() {
      if let Some(done) = outcome.final_done {
        emit_event.emit(&done)?;
      }
      return Ok(());
    }

    if step == max_steps - 1 {
      return Err(max_steps_error());
    }

    let mut replay_results = Vec::with_capacity(outcome.tool_calls.len());
    for call in &outcome.tool_calls {
      let result = execute_tool.execute(call)?;
      emit_event.emit(&ToolLoopEvent::ToolResult {
        call_id: result.call_id.clone(),
        name: result.name.clone(),
        arguments: result.arguments.clone(),
        arguments_text: result.arguments_text.clone(),
        arguments_error: result.arguments_error.clone(),
        output: result.output.clone(),
        is_error: result.is_error,
      })?;
      replay_results.push(ToolResultMessage {
        call_id: result.call_id,
        output: result.output,
        is_error: result.is_error,
      });
    }

    append_tool_turns(messages, &outcome.tool_calls, &replay_results);
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use llm_adapter::core::{CoreContent, CoreMessage, CoreRole};
  use serde_json::json;
  use std::sync::{Arc, Mutex};

  use super::run_tool_loop;
  use crate::{AccumulatedToolCall, EventSink, RoundOutcome, ToolExecutionResult, ToolExecutor, ToolLoopEvent};

  #[test]
  fn should_replay_tool_results_before_emitting_done() {
    let mut messages = vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "read doc".to_string(),
      }],
    }];
    let mut rounds = vec![
      RoundOutcome {
        tool_calls: vec![crate::AccumulatedToolCall {
          id: "call_1".to_string(),
          name: "doc_read".to_string(),
          args: json!({ "doc_id": "a1" }),
          raw_arguments_text: Some("{\"doc_id\":\"a1\"}".to_string()),
          argument_parse_error: None,
          thought: Some("need context".to_string()),
        }],
        final_done: Some(ToolLoopEvent::Done {
          finish_reason: Some("tool_calls".to_string()),
          usage: None,
        }),
      },
      RoundOutcome {
        tool_calls: Vec::new(),
        final_done: Some(ToolLoopEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        }),
      },
    ]
    .into_iter();
    let mut emitted = Vec::new();

    run_tool_loop(
      &mut messages,
      4,
      |_| Ok(rounds.next().expect("missing round outcome")),
      |call| {
        Ok(ToolExecutionResult {
          call_id: call.id.clone(),
          name: call.name.clone(),
          arguments: call.args.clone(),
          arguments_text: call.raw_arguments_text.clone(),
          arguments_error: call.argument_parse_error.clone(),
          output: json!({ "markdown": "# doc" }),
          is_error: Some(false),
        })
      },
      |event| {
        emitted.push(event.clone());
        Ok(())
      },
      || "max steps".to_string(),
    )
    .unwrap();

    assert!(matches!(
      emitted.as_slice(),
      [
        ToolLoopEvent::ToolResult { call_id, .. },
        ToolLoopEvent::Done {
          finish_reason: Some(reason),
          ..
        }
      ] if call_id == "call_1" && reason == "stop"
    ));
    assert!(matches!(messages[1].role, CoreRole::Assistant));
    assert!(matches!(messages[2].role, CoreRole::Tool));
  }

  #[test]
  fn should_fail_when_tool_loop_reaches_max_steps() {
    let mut messages = vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "loop".to_string(),
      }],
    }];

    let error = run_tool_loop(
      &mut messages,
      1,
      |_| {
        Ok(RoundOutcome {
          tool_calls: vec![crate::AccumulatedToolCall {
            id: "call_1".to_string(),
            name: "doc_read".to_string(),
            args: json!({}),
            raw_arguments_text: None,
            argument_parse_error: None,
            thought: None,
          }],
          final_done: None,
        })
      },
      |_| unreachable!("should not execute tool when max steps reached"),
      |_| Ok(()),
      || "ToolCallLoop max steps reached".to_string(),
    )
    .unwrap_err();

    assert_eq!(error, "ToolCallLoop max steps reached");
  }

  struct RecordingExecutor;

  impl ToolExecutor<String> for RecordingExecutor {
    fn execute(&mut self, call: &AccumulatedToolCall) -> Result<ToolExecutionResult, String> {
      Ok(ToolExecutionResult {
        call_id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.args.clone(),
        arguments_text: call.raw_arguments_text.clone(),
        arguments_error: call.argument_parse_error.clone(),
        output: json!({ "ok": true }),
        is_error: Some(false),
      })
    }
  }

  struct RecordingSink(Arc<Mutex<Vec<ToolLoopEvent>>>);

  impl EventSink<String> for RecordingSink {
    fn emit(&mut self, event: &ToolLoopEvent) -> Result<(), String> {
      self
        .0
        .lock()
        .expect("recording sink poisoned")
        .push(event.clone());
      Ok(())
    }
  }

  #[test]
  fn should_accept_explicit_host_trait_implementations() {
    let mut messages = vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "run tool".to_string(),
      }],
    }];
    let mut rounds = vec![
      RoundOutcome {
        tool_calls: vec![AccumulatedToolCall {
          id: "call_1".to_string(),
          name: "doc_read".to_string(),
          args: json!({ "doc_id": "a1" }),
          raw_arguments_text: None,
          argument_parse_error: None,
          thought: None,
        }],
        final_done: Some(ToolLoopEvent::Done {
          finish_reason: Some("tool_calls".to_string()),
          usage: None,
        }),
      },
      RoundOutcome {
        tool_calls: Vec::new(),
        final_done: Some(ToolLoopEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        }),
      },
    ]
    .into_iter();
    let sink_events = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(sink_events.clone());

    run_tool_loop(
      &mut messages,
      4,
      |_| Ok(rounds.next().expect("missing round outcome")),
      RecordingExecutor,
      sink,
      || "max steps".to_string(),
    )
    .unwrap();

    assert!(matches!(
      sink_events
        .lock()
        .expect("recording sink poisoned")
        .as_slice(),
      [
        ToolLoopEvent::ToolResult { call_id, .. },
        ToolLoopEvent::Done {
          finish_reason: Some(reason),
          ..
        }
      ] if call_id == "call_1" && reason == "stop"
    ));
  }
}
