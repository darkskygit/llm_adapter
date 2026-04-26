use serde::{Deserialize, Serialize};

use crate::protocol::ProtocolError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankCandidate {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub id: Option<String>,
  pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankRequest {
  pub model: String,
  pub query: String,
  #[serde(default)]
  pub candidates: Vec<RerankCandidate>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub top_n: Option<u32>,
}

impl RerankRequest {
  pub fn validate(&self) -> Result<(), ProtocolError> {
    if self.query.trim().is_empty() {
      return Err(ProtocolError::InvalidRequest {
        field: "query",
        message: "query must not be empty".to_string(),
      });
    }

    if self.candidates.is_empty() {
      return Err(ProtocolError::InvalidRequest {
        field: "candidates",
        message: "expected at least one candidate".to_string(),
      });
    }

    if self.candidates.iter().any(|candidate| candidate.text.trim().is_empty()) {
      return Err(ProtocolError::InvalidRequest {
        field: "candidates",
        message: "candidate text must not be empty".to_string(),
      });
    }

    if matches!(self.top_n, Some(0)) {
      return Err(ProtocolError::InvalidRequest {
        field: "top_n",
        message: "top_n must be greater than 0".to_string(),
      });
    }

    Ok(())
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
  pub model: String,
  pub scores: Vec<f64>,
}
