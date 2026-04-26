use std::{collections::BTreeMap, ops::Index};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::super::stream::StreamParseError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatProtocol {
  OpenaiChatCompletions,
  OpenaiResponses,
  AnthropicMessages,
  GeminiGenerateContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredProtocol {
  OpenaiChatCompletions,
  OpenaiResponses,
  GeminiGenerateContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProtocol {
  Openai,
  Gemini,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankProtocol {
  OpenaiChatLogprobs,
  CloudflareWorkersAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageProtocol {
  OpenaiImages,
  GeminiGenerateContent,
  FalImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendRequestLayer {
  Anthropic,
  ChatCompletions,
  ChatCompletionsNoV1,
  CloudflareWorkersAi,
  GeminiApi,
  GeminiVertex,
  Responses,
  VertexAnthropic,
  OpenaiImages,
  Fal,
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
  pub body: HttpBody,
  pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HttpBody {
  Json(serde_json::Value),
  Multipart(Vec<MultipartPart>),
}

impl HttpBody {
  #[must_use]
  pub fn as_json(&self) -> Option<&serde_json::Value> {
    match self {
      Self::Json(value) => Some(value),
      Self::Multipart(_) => None,
    }
  }

  #[must_use]
  pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
    self.as_json()?.get(key)
  }
}

impl Index<&str> for HttpBody {
  type Output = serde_json::Value;

  fn index(&self, index: &str) -> &Self::Output {
    &self.as_json().expect("HTTP body is not JSON")[index]
  }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultipartPart {
  Text {
    name: String,
    value: String,
  },
  File {
    name: String,
    file_name: String,
    media_type: String,
    bytes: Vec<u8>,
  },
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
  #[error("invalid backend config: {message}")]
  InvalidConfig { message: String },
  #[error("invalid request field `{field}`: {message}")]
  InvalidRequest { field: &'static str, message: String },
  #[error("http transport error: {message}")]
  Transport { message: String },
  #[error("upstream returned status {status}: {body}")]
  UpstreamStatus { status: u16, body: String },
  #[error("invalid response field `{field}`: {message}")]
  InvalidResponse { field: &'static str, message: String },
  #[error("invalid_structured_output: {message}")]
  InvalidStructuredOutput { message: String },
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
