use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreRole {
  System,
  User,
  Assistant,
  Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoreContent {
  Text {
    text: String,
  },
  Reasoning {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
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
  Image {
    source: Value,
  },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreToolDefinition {
  pub name: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<String>,
  pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreToolChoiceMode {
  Auto,
  None,
  Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CoreToolChoice {
  Mode(CoreToolChoiceMode),
  Specific { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreMessage {
  pub role: CoreRole,
  pub content: Vec<CoreContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreRequest {
  pub model: String,
  #[serde(default)]
  pub messages: Vec<CoreMessage>,
  #[serde(default)]
  pub stream: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub temperature: Option<f64>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub tools: Vec<CoreToolDefinition>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tool_choice: Option<CoreToolChoice>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub include: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CoreUsage {
  pub prompt_tokens: u32,
  pub completion_tokens: u32,
  pub total_tokens: u32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreResponse {
  pub id: String,
  pub model: String,
  pub message: CoreMessage,
  pub usage: CoreUsage,
  pub finish_reason: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning_details: Option<Value>,
}

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
  fn should_round_trip_core_request() {
    let request = CoreRequest {
      model: "gpt-4.1".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text {
          text: "hello".to_string(),
        }],
      }],
      stream: true,
      max_tokens: Some(1024),
      temperature: Some(0.5),
      tools: vec![CoreToolDefinition {
        name: "doc_read".to_string(),
        description: Some("Read a document".to_string()),
        parameters: json!({
          "type": "object",
          "properties": {
            "docId": { "type": "string" }
          }
        }),
      }],
      tool_choice: Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
    };

    let serialized = serde_json::to_value(&request).unwrap();
    let parsed: CoreRequest = serde_json::from_value(serialized).unwrap();
    assert_eq!(parsed, request);
  }

  #[test]
  fn should_support_tool_choice_string_and_object_forms() {
    let as_string = json!("required");
    let parsed_string: CoreToolChoice = serde_json::from_value(as_string).unwrap();
    assert_eq!(parsed_string, CoreToolChoice::Mode(CoreToolChoiceMode::Required));

    let as_object = json!({ "name": "doc_update" });
    let parsed_object: CoreToolChoice = serde_json::from_value(as_object).unwrap();
    assert_eq!(
      parsed_object,
      CoreToolChoice::Specific {
        name: "doc_update".to_string(),
      }
    );
  }

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
