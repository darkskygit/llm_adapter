use llm_adapter::core::CoreUsage;
#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct AccumulatedToolCall {
  pub id: String,
  pub name: String,
  pub args: Value,
  pub raw_arguments_text: Option<String>,
  pub argument_parse_error: Option<String>,
  pub thought: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultMessage {
  pub call_id: String,
  pub output: Value,
  pub is_error: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecutionResult {
  pub call_id: String,
  pub name: String,
  pub arguments: Value,
  pub arguments_text: Option<String>,
  pub arguments_error: Option<String>,
  pub output: Value,
  pub is_error: Option<bool>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ToolCallbackRequest {
  pub call_id: String,
  pub name: String,
  pub args: Value,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub raw_arguments_text: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub argument_parse_error: Option<String>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ToolCallbackResponse {
  pub call_id: String,
  pub name: String,
  pub args: Value,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub raw_arguments_text: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub argument_parse_error: Option<String>,
  pub output: Value,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub is_error: Option<bool>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolLoopEvent {
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
  ToolCall {
    call_id: String,
    name: String,
    arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<String>,
  },
  ToolResult {
    call_id: String,
    name: String,
    arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments_error: Option<String>,
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
}

impl From<AccumulatedToolCall> for ToolLoopEvent {
  fn from(call: AccumulatedToolCall) -> Self {
    Self::ToolCall {
      call_id: call.id,
      name: call.name,
      arguments: call.args,
      arguments_text: call.raw_arguments_text,
      arguments_error: call.argument_parse_error,
      thought: call.thought,
    }
  }
}
