use serde_json::Value;

use super::{BackendConfig, BackendError, BackendHttpClient, BackendProtocol, BackendRequestLayer};
use crate::core::{RerankRequest, RerankResponse};

mod anthropic;
mod anthropic_vertex;
mod chat_completions;
mod cloudflare_workers_ai;
mod gemini_api;
mod gemini_vertex;
mod responses;

use self::{
  anthropic::AnthropicRequestLayer,
  anthropic_vertex::VertexAnthropicRequestLayer,
  chat_completions::{ChatCompletionsNoV1RequestLayer, ChatCompletionsRequestLayer},
  cloudflare_workers_ai::CloudflareWorkersAiRequestLayer,
  gemini_api::GeminiApiRequestLayer,
  gemini_vertex::GeminiVertexRequestLayer,
  responses::ResponsesRequestLayer,
};

// Design note:
// We intentionally keep request-layer behavior behind trait-based
// implementations even though some current layers are simple. This keeps
// protocol concerns (payload encode/decode) separate from transport-shape
// concerns (URL/header/body rewrite), and allows adding provider-specific
// request surfaces without growing a single giant match with intertwined rules.
trait RequestLayerImpl {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String;
  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)>;

  fn rewrite_body(&self, body: Value) -> Value {
    body
  }

  fn build_embedding_url(&self, _base_url: &str, _model: &str) -> Result<String, BackendError> {
    Err(BackendError::InvalidConfig(
      "embedding dispatch is not supported for this request layer".to_string(),
    ))
  }

  fn build_embedding_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.build_headers(config, false)
  }

  fn rewrite_embedding_body(&self, body: Value) -> Value {
    body
  }

  fn build_rerank_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    Ok(self.build_url(base_url, model, false))
  }

  fn build_rerank_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.build_headers(config, false)
  }

  fn rewrite_rerank_body(&self, body: Value) -> Value {
    self.rewrite_body(body)
  }

  fn dispatch_rerank(
    &self,
    _client: &dyn BackendHttpClient,
    _config: &BackendConfig,
    _protocol: &BackendProtocol,
    _request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    Ok(None)
  }
}

const ANTHROPIC_LAYER: AnthropicRequestLayer = AnthropicRequestLayer;
const CHAT_COMPLETIONS_LAYER: ChatCompletionsRequestLayer = ChatCompletionsRequestLayer;
const CHAT_COMPLETIONS_NO_V1_LAYER: ChatCompletionsNoV1RequestLayer = ChatCompletionsNoV1RequestLayer;
const CLOUDFLARE_WORKERS_AI_LAYER: CloudflareWorkersAiRequestLayer = CloudflareWorkersAiRequestLayer;
const GEMINI_API_LAYER: GeminiApiRequestLayer = GeminiApiRequestLayer;
const GEMINI_VERTEX_LAYER: GeminiVertexRequestLayer = GeminiVertexRequestLayer;
const RESPONSES_LAYER: ResponsesRequestLayer = ResponsesRequestLayer;
const VERTEX_ANTHROPIC_LAYER: VertexAnthropicRequestLayer = VertexAnthropicRequestLayer;

impl BackendProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      BackendProtocol::OpenaiChatCompletions => "openai_chat_completions",
      BackendProtocol::OpenaiResponses => "openai_responses",
      BackendProtocol::AnthropicMessages => "anthropic_messages",
      BackendProtocol::GeminiGenerateContent => "gemini_generate_content",
    }
  }
}

impl BackendRequestLayer {
  pub(super) fn from_protocol(protocol: &BackendProtocol) -> Self {
    match protocol {
      BackendProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      BackendProtocol::OpenaiResponses => Self::Responses,
      BackendProtocol::AnthropicMessages => Self::Anthropic,
      BackendProtocol::GeminiGenerateContent => Self::GeminiApi,
    }
  }

  fn ensure_compatible(&self, protocol: &BackendProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (
        BackendRequestLayer::ChatCompletions,
        BackendProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::ChatCompletionsNoV1,
        BackendProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::CloudflareWorkersAi,
        BackendProtocol::OpenaiChatCompletions
      ) | (BackendRequestLayer::Responses, BackendProtocol::OpenaiResponses)
        | (BackendRequestLayer::Anthropic, BackendProtocol::AnthropicMessages)
        | (BackendRequestLayer::VertexAnthropic, BackendProtocol::AnthropicMessages)
        | (BackendRequestLayer::GeminiApi, BackendProtocol::GeminiGenerateContent)
        | (
          BackendRequestLayer::GeminiVertex,
          BackendProtocol::GeminiGenerateContent
        )
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig(format!(
        "request_layer `{}` is incompatible with protocol `{}`",
        self.as_str(),
        protocol.as_str(),
      )))
    }
  }

  fn as_str(&self) -> &'static str {
    match self {
      BackendRequestLayer::Anthropic => "anthropic",
      BackendRequestLayer::ChatCompletions => "chat_completions",
      BackendRequestLayer::ChatCompletionsNoV1 => "chat_completions_no_v1",
      BackendRequestLayer::CloudflareWorkersAi => "cloudflare_workers_ai",
      BackendRequestLayer::GeminiApi => "gemini_api",
      BackendRequestLayer::GeminiVertex => "gemini_vertex",
      BackendRequestLayer::Responses => "responses",
      BackendRequestLayer::VertexAnthropic => "vertex_anthropic",
    }
  }

  fn implementation(&self) -> &'static dyn RequestLayerImpl {
    match self {
      BackendRequestLayer::Anthropic => &ANTHROPIC_LAYER,
      BackendRequestLayer::ChatCompletions => &CHAT_COMPLETIONS_LAYER,
      BackendRequestLayer::ChatCompletionsNoV1 => &CHAT_COMPLETIONS_NO_V1_LAYER,
      BackendRequestLayer::CloudflareWorkersAi => &CLOUDFLARE_WORKERS_AI_LAYER,
      BackendRequestLayer::GeminiApi => &GEMINI_API_LAYER,
      BackendRequestLayer::GeminiVertex => &GEMINI_VERTEX_LAYER,
      BackendRequestLayer::Responses => &RESPONSES_LAYER,
      BackendRequestLayer::VertexAnthropic => &VERTEX_ANTHROPIC_LAYER,
    }
  }

  pub(super) fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    self.implementation().build_url(base_url, model, stream)
  }

  pub(super) fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    self.implementation().build_headers(config, stream)
  }

  pub(super) fn rewrite_body(&self, body: Value) -> Value {
    self.implementation().rewrite_body(body)
  }

  pub(super) fn build_embedding_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    self.implementation().build_embedding_url(base_url, model)
  }

  pub(super) fn build_embedding_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.implementation().build_embedding_headers(config)
  }

  pub(super) fn rewrite_embedding_body(&self, body: Value) -> Value {
    self.implementation().rewrite_embedding_body(body)
  }

  pub(super) fn build_rerank_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    self.implementation().build_rerank_url(base_url, model)
  }

  pub(super) fn build_rerank_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.implementation().build_rerank_headers(config)
  }

  pub(super) fn rewrite_rerank_body(&self, body: Value) -> Value {
    self.implementation().rewrite_rerank_body(body)
  }

  pub(super) fn dispatch_rerank(
    &self,
    client: &dyn BackendHttpClient,
    config: &BackendConfig,
    protocol: &BackendProtocol,
    request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    self.implementation().dispatch_rerank(client, config, protocol, request)
  }
}

pub(super) fn resolve_request_layer(
  config: &BackendConfig,
  protocol: &BackendProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_protocol(protocol));
  request_layer.ensure_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn build_extra_headers(config: &BackendConfig) -> Vec<(String, String)> {
  let mut headers: Vec<(String, String)> = config
    .headers
    .iter()
    .map(|(key, value)| (key.clone(), value.clone()))
    .collect();
  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}

fn join_url(base_url: &str, path: &str) -> String {
  format!("{}{}", base_url.trim_end_matches('/'), path)
}

fn build_bearer_headers(config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
  let mut headers = vec![
    ("content-type".to_string(), "application/json".to_string()),
    (
      "accept".to_string(),
      if stream {
        "text/event-stream".to_string()
      } else {
        "application/json".to_string()
      },
    ),
  ];

  if !config.auth_token.is_empty() {
    headers.push(("authorization".to_string(), format!("Bearer {}", config.auth_token)));
  }

  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}

fn build_api_key_headers(config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
  let mut headers = vec![
    ("content-type".to_string(), "application/json".to_string()),
    (
      "accept".to_string(),
      if stream {
        "text/event-stream".to_string()
      } else {
        "application/json".to_string()
      },
    ),
  ];

  if !config.auth_token.is_empty() {
    headers.push(("x-goog-api-key".to_string(), config.auth_token.clone()));
  }

  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}
