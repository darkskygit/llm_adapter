use std::collections::BTreeMap;

use serde_json::Value;

use crate::AccumulatedToolCall;

#[derive(Debug, Clone, Default, PartialEq)]
struct ToolCallState {
  name: Option<String>,
  arguments_text: String,
}

#[derive(Debug, Default)]
pub(crate) struct ToolCallAccumulator {
  states: BTreeMap<String, ToolCallState>,
}

impl ToolCallAccumulator {
  pub(crate) fn feed_delta(&mut self, call_id: String, name: Option<String>, arguments_delta: String) {
    let state = self.states.entry(call_id).or_default();
    if let Some(name) = name {
      state.name = Some(name);
    }
    state.arguments_text.push_str(&arguments_delta);
  }

  pub(crate) fn complete(
    &mut self,
    call_id: String,
    name: String,
    arguments: Value,
    thought: Option<String>,
  ) -> AccumulatedToolCall {
    let state = self.states.remove(&call_id);
    let (args, raw_arguments_text, argument_parse_error) = if let Some(raw_arguments_text) = state
      .as_ref()
      .map(|state| state.arguments_text.clone())
      .filter(|text| !text.is_empty())
    {
      parse_arguments_value(arguments, Some(raw_arguments_text))
    } else {
      parse_arguments_value(arguments, None)
    };

    AccumulatedToolCall {
      id: call_id,
      name: if name.is_empty() {
        state.and_then(|state| state.name).unwrap_or_default()
      } else {
        name
      },
      args,
      raw_arguments_text,
      argument_parse_error,
      thought,
    }
  }

  pub(crate) fn drain_pending(&mut self) -> Vec<AccumulatedToolCall> {
    let mut pending = Vec::new();

    for (call_id, state) in std::mem::take(&mut self.states) {
      let Some(name) = state.name else {
        continue;
      };
      let (args, raw_arguments_text, argument_parse_error) = parse_arguments_text(&state.arguments_text);
      pending.push(AccumulatedToolCall {
        id: call_id,
        name,
        args,
        raw_arguments_text,
        argument_parse_error,
        thought: None,
      });
    }

    pending
  }
}

fn parse_arguments_text(arguments_text: &str) -> (Value, Option<String>, Option<String>) {
  if arguments_text.trim().is_empty() {
    return (Value::Object(Default::default()), None, None);
  }

  match serde_json::from_str::<Value>(arguments_text) {
    Ok(value) => parse_arguments_value(value, Some(arguments_text.to_string())),
    Err(error) => (
      Value::Object(Default::default()),
      Some(arguments_text.to_string()),
      Some(error.to_string()),
    ),
  }
}

fn parse_arguments_value(
  arguments: Value,
  raw_arguments_text: Option<String>,
) -> (Value, Option<String>, Option<String>) {
  if arguments.is_object() {
    return (arguments, raw_arguments_text, None);
  }

  (
    Value::Object(Default::default()),
    raw_arguments_text,
    Some("Tool arguments must be a JSON object".to_string()),
  )
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::ToolCallAccumulator;
  use crate::AccumulatedToolCall;

  #[test]
  fn should_merge_delta_and_complete() {
    let mut accumulator = ToolCallAccumulator::default();

    accumulator.feed_delta(
      "call_1".to_string(),
      Some("doc_read".to_string()),
      "{\"doc_id\":\"a1\"}".to_string(),
    );

    let call = accumulator.complete(
      "call_1".to_string(),
      "doc_read".to_string(),
      json!({ "doc_id": "a1" }),
      Some("need context".to_string()),
    );

    assert_eq!(
      call,
      AccumulatedToolCall {
        id: "call_1".to_string(),
        name: "doc_read".to_string(),
        args: json!({ "doc_id": "a1" }),
        raw_arguments_text: Some("{\"doc_id\":\"a1\"}".to_string()),
        argument_parse_error: None,
        thought: Some("need context".to_string()),
      }
    );
  }

  #[test]
  fn should_preserve_invalid_json_when_draining() {
    let mut accumulator = ToolCallAccumulator::default();

    accumulator.feed_delta(
      "call_1".to_string(),
      Some("doc_read".to_string()),
      "{\"doc_id\":".to_string(),
    );

    let calls = accumulator.drain_pending();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].name, "doc_read");
    assert_eq!(calls[0].args, json!({}));
    assert_eq!(calls[0].raw_arguments_text.as_deref(), Some("{\"doc_id\":"));
    assert!(calls[0].argument_parse_error.is_some());
  }
}
