use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::super::stream::StreamParseError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendProtocol {
  OpenaiChatCompletions,
  OpenaiResponses,
  AnthropicMessages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendRequestLayer {
  Anthropic,
  ChatCompletions,
  Responses,
  Vertex,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendConfig {
  pub base_url: String,
  pub auth_token: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub request_layer: Option<BackendRequestLayer>,
  #[serde(default)]
  pub headers: BTreeMap<String, String>,
  #[serde(default)]
  pub no_streaming: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HttpRequest {
  pub url: String,
  pub headers: Vec<(String, String)>,
  pub body: serde_json::Value,
  pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HttpResponse {
  pub status: u16,
  pub body: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HttpStreamResponse {
  pub status: u16,
  pub body: String,
}

#[derive(Debug, Error)]
pub enum BackendError {
  #[error("no backend available")]
  NoBackendAvailable,
  #[error("invalid backend config: {0}")]
  InvalidConfig(String),
  #[error("http transport error: {0}")]
  Http(String),
  #[error("upstream returned status {status}: {body}")]
  UpstreamStatus { status: u16, body: String },
  #[error("invalid response: {0}")]
  InvalidResponse(&'static str),
  #[error("json error: {0}")]
  Json(#[from] serde_json::Error),
  #[error(transparent)]
  Stream(#[from] StreamParseError),
}

pub trait BackendHttpClient {
  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError>;

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError>;
}
