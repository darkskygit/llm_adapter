use std::{collections::BTreeMap, ops::Index, str::FromStr};

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::super::stream::StreamParseError;

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
  ChatCompletionsNoV1,
  CloudflareWorkersAi,
  GeminiApi,
  GeminiVertex,
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
      "chat_completions_no_v1" => Ok(Self::ChatCompletionsNoV1),
      "cloudflare_workers_ai" => Ok(Self::CloudflareWorkersAi),
      "gemini_api" => Ok(Self::GeminiApi),
      "gemini_vertex" => Ok(Self::GeminiVertex),
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

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendConfig {
  pub base_url: String,
  pub auth_token: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub request_layer: Option<BackendRequestLayer>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub headers: BTreeMap<String, String>,
  #[serde(default, skip_serializing_if = "is_false")]
  pub no_streaming: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub timeout_ms: Option<u64>,
}

fn is_false(value: &bool) -> bool {
  !*value
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
