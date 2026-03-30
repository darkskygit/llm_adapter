use llm_adapter::core::{CoreContent, CoreMessage, CoreRole};

use crate::{AccumulatedToolCall, ToolResultMessage};

pub fn append_tool_turns(
  messages: &mut Vec<CoreMessage>,
  tool_calls: &[AccumulatedToolCall],
  tool_results: &[ToolResultMessage],
) {
  messages.push(CoreMessage {
    role: CoreRole::Assistant,
    content: tool_calls
      .iter()
      .map(|call| CoreContent::ToolCall {
        call_id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.args.clone(),
        thought: call.thought.clone(),
      })
      .collect(),
  });

  for result in tool_results {
    messages.push(CoreMessage {
      role: CoreRole::Tool,
      content: vec![CoreContent::ToolResult {
        call_id: result.call_id.clone(),
        output: result.output.clone(),
        is_error: result.is_error,
      }],
    });
  }
}

#[cfg(test)]
mod tests {
  use llm_adapter::core::{CoreContent, CoreMessage, CoreRole};
  use serde_json::json;

  use super::append_tool_turns;
  use crate::{AccumulatedToolCall, ToolResultMessage};

  #[test]
  fn should_replay_assistant_and_tool_messages() {
    let mut messages = vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "read doc".to_string(),
      }],
    }];

    append_tool_turns(
      &mut messages,
      &[AccumulatedToolCall {
        id: "call_1".to_string(),
        name: "doc_read".to_string(),
        args: json!({ "doc_id": "a1" }),
        raw_arguments_text: Some("{\"doc_id\":\"a1\"}".to_string()),
        argument_parse_error: None,
        thought: Some("need context".to_string()),
      }],
      &[ToolResultMessage {
        call_id: "call_1".to_string(),
        output: json!({ "markdown": "# doc" }),
        is_error: Some(false),
      }],
    );

    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[1].role, CoreRole::Assistant));
    assert!(matches!(messages[2].role, CoreRole::Tool));
  }
}
