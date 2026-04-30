#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::protocol::ProtocolError;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EmbeddingUsage {
  pub prompt_tokens: u32,
  pub total_tokens: u32,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
  pub model: String,
  pub inputs: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub dimensions: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub task_type: Option<String>,
}

impl EmbeddingRequest {
  pub fn validate(&self) -> Result<(), ProtocolError> {
    if self.inputs.is_empty() {
      return Err(ProtocolError::InvalidRequest {
        field: "inputs",
        message: "expected at least one input".to_string(),
      });
    }

    if self.inputs.iter().any(|input| input.is_empty()) {
      return Err(ProtocolError::InvalidRequest {
        field: "inputs",
        message: "inputs must not be empty".to_string(),
      });
    }

    Ok(())
  }
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
  pub model: String,
  pub embeddings: Vec<Vec<f64>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub usage: Option<EmbeddingUsage>,
}
