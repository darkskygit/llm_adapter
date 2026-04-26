use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::chat::CoreUsage;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
  MessageStart {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
  },
  TextDelta {
    text: String,
  },
  ReasoningDelta {
    text: String,
  },
  ToolCallDelta {
    call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments_delta: String,
  },
  ToolCall {
    call_id: String,
    name: String,
    arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<String>,
  },
  ToolResult {
    call_id: String,
    output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
  },
  Citation {
    index: usize,
    url: String,
  },
  Usage {
    usage: CoreUsage,
  },
  Done {
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CoreUsage>,
  },
  Error {
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
  },
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn should_serialize_stream_event_with_usage() {
    let event = StreamEvent::Done {
      finish_reason: Some("stop".to_string()),
      usage: Some(CoreUsage {
        prompt_tokens: 100,
        completion_tokens: 20,
        total_tokens: 120,
        cached_tokens: Some(12),
      }),
    };

    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(
      value,
      json!({
        "type": "done",
        "finish_reason": "stop",
        "usage": {
          "prompt_tokens": 100,
          "completion_tokens": 20,
          "total_tokens": 120,
          "cached_tokens": 12
        }
      })
    );
  }
}
