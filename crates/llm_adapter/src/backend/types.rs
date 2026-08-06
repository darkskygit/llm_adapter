use std::{collections::BTreeMap, fmt, ops::Index, str::FromStr};

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeroize::Zeroizing;

use super::super::{stream::StreamParseError, target::EgressPolicy};

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatProtocol {
  OpenaiChatCompletions,
  OpenaiResponses,
  AnthropicMessages,
  GeminiGenerateContent,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredProtocol {
  OpenaiChatCompletions,
  OpenaiResponses,
  GeminiGenerateContent,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProtocol {
  Openai,
  Gemini,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankProtocol {
  OpenaiChatLogprobs,
  CloudflareWorkersAi,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageProtocol {
  OpenaiImages,
  GeminiGenerateContent,
  FalImage,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendRequestLayer {
  Anthropic,
  ChatCompletions,
  CloudflareWorkersAi,
  GeminiApi,
  GeminiVertex,
  PerplexitySonar,
  Responses,
  VertexAnthropic,
  OpenaiImages,
  Fal,
}

fn normalize_protocol_name(value: &str) -> String {
  value.trim().replace('-', "_").to_ascii_lowercase()
}

impl FromStr for ChatProtocol {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "openai_chat" | "openai_chat_completions" | "chat_completions" => Ok(Self::OpenaiChatCompletions),
      "openai_responses" | "responses" => Ok(Self::OpenaiResponses),
      "anthropic" | "anthropic_messages" => Ok(Self::AnthropicMessages),
      "gemini" | "gemini_generate_content" => Ok(Self::GeminiGenerateContent),
      _ => Err(BackendError::InvalidRequest {
        field: "protocol",
        message: format!("unsupported chat protocol: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for ChatProtocol {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

impl FromStr for StructuredProtocol {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "openai_chat" | "openai_chat_completions" | "chat_completions" => Ok(Self::OpenaiChatCompletions),
      "openai_responses" | "responses" => Ok(Self::OpenaiResponses),
      "gemini" | "gemini_generate_content" => Ok(Self::GeminiGenerateContent),
      _ => Err(BackendError::InvalidRequest {
        field: "protocol",
        message: format!("unsupported structured protocol: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for StructuredProtocol {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

impl FromStr for EmbeddingProtocol {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "openai" | "openai_chat" | "openai_chat_completions" | "chat_completions" => Ok(Self::Openai),
      "gemini" | "gemini_generate_content" => Ok(Self::Gemini),
      _ => Err(BackendError::InvalidRequest {
        field: "protocol",
        message: format!("unsupported embedding protocol: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for EmbeddingProtocol {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

impl FromStr for RerankProtocol {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "openai_chat" | "openai_chat_completions" | "chat_completions" => Ok(Self::OpenaiChatLogprobs),
      "cloudflare_workers_ai" => Ok(Self::CloudflareWorkersAi),
      _ => Err(BackendError::InvalidRequest {
        field: "protocol",
        message: format!("unsupported rerank protocol: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for RerankProtocol {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

impl FromStr for ImageProtocol {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "openai_images" => Ok(Self::OpenaiImages),
      "gemini" | "gemini_generate_content" => Ok(Self::GeminiGenerateContent),
      "fal" | "fal_image" => Ok(Self::FalImage),
      _ => Err(BackendError::InvalidRequest {
        field: "protocol",
        message: format!("unsupported image protocol: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for ImageProtocol {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

impl FromStr for BackendRequestLayer {
  type Err = BackendError;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match normalize_protocol_name(value).as_str() {
      "anthropic" => Ok(Self::Anthropic),
      "chat_completions" => Ok(Self::ChatCompletions),
      "cloudflare_workers_ai" => Ok(Self::CloudflareWorkersAi),
      "gemini_api" => Ok(Self::GeminiApi),
      "gemini_vertex" => Ok(Self::GeminiVertex),
      "perplexity_sonar" | "sonar" => Ok(Self::PerplexitySonar),
      "responses" => Ok(Self::Responses),
      "vertex_anthropic" => Ok(Self::VertexAnthropic),
      "openai_images" => Ok(Self::OpenaiImages),
      "fal" => Ok(Self::Fal),
      _ => Err(BackendError::InvalidRequest {
        field: "request_layer",
        message: format!("unsupported request layer: {value}"),
      }),
    }
  }
}

impl TryFrom<&str> for BackendRequestLayer {
  type Error = BackendError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    value.parse()
  }
}

pub struct SensitiveString(Zeroizing<String>);

impl SensitiveString {
  #[must_use]
  pub fn new(value: String) -> Self {
    Self(Zeroizing::new(value))
  }

  #[must_use]
  pub fn expose(&self) -> &str {
    self.0.as_str()
  }

  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
}

impl From<String> for SensitiveString {
  fn from(value: String) -> Self {
    Self::new(value)
  }
}

impl From<&str> for SensitiveString {
  fn from(value: &str) -> Self {
    Self::new(value.to_string())
  }
}

impl fmt::Debug for SensitiveString {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter.write_str("SensitiveString([REDACTED])")
  }
}

impl PartialEq for SensitiveString {
  fn eq(&self, other: &Self) -> bool {
    self.expose() == other.expose()
  }
}

impl Eq for SensitiveString {}

#[derive(PartialEq, Eq)]
pub struct BackendConfig {
  pub base_url: String,
  pub auth_token: SensitiveString,
  pub request_layer: Option<BackendRequestLayer>,
  pub headers: BTreeMap<String, String>,
  pub no_streaming: bool,
  pub timeout_ms: Option<u64>,
  pub egress_policy: EgressPolicy,
}

#[derive(PartialEq)]
pub struct HttpRequest {
  pub url: String,
  pub headers: Vec<(String, String)>,
  pub body: HttpBody,
  pub timeout_ms: Option<u64>,
  pub egress_policy: EgressPolicy,
}

impl fmt::Debug for BackendConfig {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("BackendConfig")
      .field("base_url", &self.base_url)
      .field("auth_token", &"[REDACTED]")
      .field("request_layer", &self.request_layer)
      .field("headers", &self.headers.keys().collect::<Vec<_>>())
      .field("no_streaming", &self.no_streaming)
      .field("timeout_ms", &self.timeout_ms)
      .field("egress_policy", &self.egress_policy)
      .finish()
  }
}

impl fmt::Debug for HttpRequest {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("HttpRequest")
      .field("url", &self.url)
      .field(
        "header_names",
        &self.headers.iter().map(|(name, _)| name).collect::<Vec<_>>(),
      )
      .field("body", &self.body)
      .field("timeout_ms", &self.timeout_ms)
      .field("egress_policy", &self.egress_policy)
      .finish()
  }
}

impl Drop for HttpRequest {
  fn drop(&mut self) {
    use zeroize::Zeroize;
    self.headers.iter_mut().for_each(|(_, value)| value.zeroize());
  }
}

#[cfg(test)]
impl Clone for HttpRequest {
  fn clone(&self) -> Self {
    Self {
      url: self.url.clone(),
      headers: self.headers.clone(),
      body: self.body.clone(),
      timeout_ms: self.timeout_ms,
      egress_policy: self.egress_policy,
    }
  }
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

pub struct HttpUploadRequest {
  pub url: String,
  pub headers: Vec<(String, String)>,
  pub bytes: Vec<u8>,
  pub timeout_ms: Option<u64>,
  pub egress_policy: EgressPolicy,
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
  #[error("http timeout error: {message}")]
  Timeout { message: String },
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

  fn put_bytes(&self, _request: HttpUploadRequest) -> Result<(), BackendError> {
    Err(BackendError::InvalidConfig {
      message: "HTTP client does not support binary uploads".to_string(),
    })
  }

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError>;
}
