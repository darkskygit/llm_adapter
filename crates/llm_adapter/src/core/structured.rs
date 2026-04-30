#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::chat::{CoreMessage, CoreRequest, CoreUsage, validate_messages};
use crate::protocol::ProtocolError;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredRequest {
  pub model: String,
  #[serde(default)]
  pub messages: Vec<CoreMessage>,
  pub schema: Value,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning: Option<Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub strict: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub response_mime_type: Option<String>,
}

impl StructuredRequest {
  pub fn validate(&self) -> Result<(), ProtocolError> {
    validate_messages(&self.messages)?;
    Ok(())
  }

  #[must_use]
  pub fn as_core_request(&self) -> CoreRequest {
    CoreRequest {
      model: self.model.clone(),
      messages: self.messages.clone(),
      stream: false,
      max_tokens: self.max_tokens,
      temperature: self.temperature,
      tools: Vec::new(),
      tool_choice: None,
      include: None,
      reasoning: self.reasoning.clone(),
      response_schema: Some(self.schema.clone()),
    }
  }
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredResponse {
  pub id: String,
  pub model: String,
  pub output_text: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub output_json: Option<Value>,
  pub usage: CoreUsage,
  pub finish_reason: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning_details: Option<Value>,
}
