use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
  BackendConfig, BackendError, BackendHttpClient, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, ImageProtocol,
  RerankProtocol, StructuredProtocol,
};
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
    Err(BackendError::InvalidConfig {
      message: "embedding dispatch is not supported for this request layer".to_string(),
    })
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
    _protocol: &RerankProtocol,
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
const OPENAI_IMAGES_LAYER: OpenaiImagesRequestLayer = OpenaiImagesRequestLayer;
const FAL_LAYER: FalRequestLayer = FalRequestLayer;

#[derive(Debug, Clone, Copy)]
struct OpenaiImagesRequestLayer;

impl RequestLayerImpl for OpenaiImagesRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/v1/images/generations")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }
}

impl OpenaiImagesRequestLayer {
  fn build_image_url(&self, base_url: &str, edit: bool) -> String {
    join_url(
      base_url,
      if edit {
        "/v1/images/edits"
      } else {
        "/v1/images/generations"
      },
    )
  }
}

#[derive(Debug, Clone, Copy)]
struct FalRequestLayer;

impl RequestLayerImpl for FalRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, _stream: bool) -> String {
    join_url(base_url, &format!("/fal-ai/{model}"))
  }

  fn build_headers(&self, config: &BackendConfig, _stream: bool) -> Vec<(String, String)> {
    let mut headers = vec![("Authorization".to_string(), format!("key {}", config.auth_token))];
    headers.extend(config.headers.iter().map(|(key, value)| (key.clone(), value.clone())));
    headers
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentReferenceMode {
  Remote,
  Inline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentReferenceReason {
  NonUrlSource,
  UnsupportedScheme,
  GenericRemoteReference,
  GeminiApiFileUri,
  GeminiApiYoutubeUrl,
  GeminiApiInlineHttpUrl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentReferencePlan {
  pub mode: AttachmentReferenceMode,
  pub reason: AttachmentReferenceReason,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestIntentReasoning {
  #[serde(default)]
  pub enabled: bool,
  #[serde(default)]
  pub effort: Option<String>,
  #[serde(default)]
  pub budget_tokens: Option<u32>,
  #[serde(default)]
  pub include_reasoning: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RequestIntent {
  #[serde(default)]
  pub include: Vec<String>,
  #[serde(default)]
  pub reasoning: Option<RequestIntentReasoning>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResolvedRequestIntent {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub include: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning: Option<Value>,
}

impl ChatProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      ChatProtocol::OpenaiChatCompletions => "openai_chat_completions",
      ChatProtocol::OpenaiResponses => "openai_responses",
      ChatProtocol::AnthropicMessages => "anthropic_messages",
      ChatProtocol::GeminiGenerateContent => "gemini_generate_content",
    }
  }
}

impl StructuredProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      StructuredProtocol::OpenaiChatCompletions => "openai_chat_completions",
      StructuredProtocol::OpenaiResponses => "openai_responses",
      StructuredProtocol::GeminiGenerateContent => "gemini_generate_content",
    }
  }
}

impl EmbeddingProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      EmbeddingProtocol::Openai => "openai",
      EmbeddingProtocol::Gemini => "gemini",
    }
  }
}

impl RerankProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      RerankProtocol::OpenaiChatLogprobs => "openai_chat_logprobs",
      RerankProtocol::CloudflareWorkersAi => "cloudflare_workers_ai",
    }
  }
}

impl ImageProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      ImageProtocol::OpenaiImages => "openai_images",
      ImageProtocol::GeminiGenerateContent => "gemini_generate_content",
      ImageProtocol::FalImage => "fal_image",
    }
  }
}

impl BackendRequestLayer {
  pub(super) fn from_chat_protocol(protocol: &ChatProtocol) -> Self {
    match protocol {
      ChatProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      ChatProtocol::OpenaiResponses => Self::Responses,
      ChatProtocol::AnthropicMessages => Self::Anthropic,
      ChatProtocol::GeminiGenerateContent => Self::GeminiApi,
    }
  }

  pub(super) fn from_structured_protocol(protocol: &StructuredProtocol) -> Self {
    match protocol {
      StructuredProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      StructuredProtocol::OpenaiResponses => Self::Responses,
      StructuredProtocol::GeminiGenerateContent => Self::GeminiApi,
    }
  }

  pub(super) fn from_embedding_protocol(protocol: &EmbeddingProtocol) -> Self {
    match protocol {
      EmbeddingProtocol::Openai => Self::ChatCompletions,
      EmbeddingProtocol::Gemini => Self::GeminiApi,
    }
  }

  pub(super) fn from_rerank_protocol(protocol: &RerankProtocol) -> Self {
    match protocol {
      RerankProtocol::OpenaiChatLogprobs => Self::ChatCompletions,
      RerankProtocol::CloudflareWorkersAi => Self::CloudflareWorkersAi,
    }
  }

  pub(super) fn from_image_protocol(protocol: &ImageProtocol) -> Self {
    match protocol {
      ImageProtocol::OpenaiImages => Self::OpenaiImages,
      ImageProtocol::GeminiGenerateContent => Self::GeminiApi,
      ImageProtocol::FalImage => Self::Fal,
    }
  }

  fn ensure_chat_compatible(&self, protocol: &ChatProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (
        BackendRequestLayer::ChatCompletions,
        ChatProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::ChatCompletionsNoV1,
        ChatProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::CloudflareWorkersAi,
        ChatProtocol::OpenaiChatCompletions
      ) | (BackendRequestLayer::Responses, ChatProtocol::OpenaiResponses)
        | (BackendRequestLayer::Anthropic, ChatProtocol::AnthropicMessages)
        | (BackendRequestLayer::VertexAnthropic, ChatProtocol::AnthropicMessages)
        | (BackendRequestLayer::GeminiApi, ChatProtocol::GeminiGenerateContent)
        | (BackendRequestLayer::GeminiVertex, ChatProtocol::GeminiGenerateContent)
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with chat protocol `{}`",
          self.as_str(),
          protocol.as_str(),
        ),
      })
    }
  }

  fn ensure_structured_compatible(&self, protocol: &StructuredProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (
        BackendRequestLayer::ChatCompletions,
        StructuredProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::ChatCompletionsNoV1,
        StructuredProtocol::OpenaiChatCompletions
      ) | (
        BackendRequestLayer::CloudflareWorkersAi,
        StructuredProtocol::OpenaiChatCompletions
      ) | (BackendRequestLayer::Responses, StructuredProtocol::OpenaiResponses)
        | (
          BackendRequestLayer::GeminiApi,
          StructuredProtocol::GeminiGenerateContent
        )
        | (
          BackendRequestLayer::GeminiVertex,
          StructuredProtocol::GeminiGenerateContent
        )
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with structured protocol `{}`",
          self.as_str(),
          protocol.as_str(),
        ),
      })
    }
  }

  fn ensure_embedding_compatible(&self, protocol: &EmbeddingProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (BackendRequestLayer::ChatCompletions, EmbeddingProtocol::Openai)
        | (BackendRequestLayer::ChatCompletionsNoV1, EmbeddingProtocol::Openai)
        | (BackendRequestLayer::CloudflareWorkersAi, EmbeddingProtocol::Openai)
        | (BackendRequestLayer::Responses, EmbeddingProtocol::Openai)
        | (BackendRequestLayer::GeminiApi, EmbeddingProtocol::Gemini)
        | (BackendRequestLayer::GeminiVertex, EmbeddingProtocol::Gemini)
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with embedding protocol `{}`",
          self.as_str(),
          protocol.as_str(),
        ),
      })
    }
  }

  fn ensure_rerank_compatible(&self, protocol: &RerankProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (BackendRequestLayer::ChatCompletions, RerankProtocol::OpenaiChatLogprobs)
        | (
          BackendRequestLayer::ChatCompletionsNoV1,
          RerankProtocol::OpenaiChatLogprobs
        )
        | (
          BackendRequestLayer::CloudflareWorkersAi,
          RerankProtocol::CloudflareWorkersAi
        )
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with rerank protocol `{}`",
          self.as_str(),
          protocol.as_str(),
        ),
      })
    }
  }

  fn ensure_image_compatible(&self, protocol: &ImageProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (BackendRequestLayer::OpenaiImages, ImageProtocol::OpenaiImages)
        | (BackendRequestLayer::GeminiApi, ImageProtocol::GeminiGenerateContent)
        | (BackendRequestLayer::GeminiVertex, ImageProtocol::GeminiGenerateContent)
        | (BackendRequestLayer::Fal, ImageProtocol::FalImage)
    );

    if compatible {
      Ok(())
    } else {
      Err(BackendError::InvalidConfig {
        message: format!(
          "request_layer `{}` is incompatible with image protocol `{}`",
          self.as_str(),
          protocol.as_str(),
        ),
      })
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
      BackendRequestLayer::OpenaiImages => "openai_images",
      BackendRequestLayer::Fal => "fal",
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
      BackendRequestLayer::OpenaiImages => &OPENAI_IMAGES_LAYER,
      BackendRequestLayer::Fal => &FAL_LAYER,
    }
  }

  pub(super) fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    self.implementation().build_url(base_url, model, stream)
  }

  pub(super) fn build_image_url(&self, base_url: &str, model: &str, edit: bool) -> String {
    match self {
      BackendRequestLayer::OpenaiImages => OPENAI_IMAGES_LAYER.build_image_url(base_url, edit),
      _ => self.build_url(base_url, model, false),
    }
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

  pub fn plan_attachment_reference(&self, base_url: &str, source: &Value) -> AttachmentReferencePlan {
    let url = match source {
      Value::String(url) => Some(url.as_str()),
      Value::Object(object) => object.get("url").and_then(Value::as_str),
      _ => None,
    };

    let Some(url) = url else {
      return AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Inline,
        reason: AttachmentReferenceReason::NonUrlSource,
      };
    };

    match self {
      BackendRequestLayer::GeminiApi => plan_gemini_api_attachment_reference(url, base_url),
      _ => generic_remote_reference_plan(url),
    }
  }

  pub fn resolve_request_intent(&self, intent: RequestIntent) -> ResolvedRequestIntent {
    let mut include = intent.include;
    let mut reasoning = None;

    if let Some(reasoning_intent) = intent.reasoning.filter(|value| value.enabled) {
      if reasoning_intent.include_reasoning {
        include.push("reasoning".to_string());
      }

      reasoning = match self {
        BackendRequestLayer::Anthropic | BackendRequestLayer::VertexAnthropic => Some(Value::Object(
          [(
            "budget_tokens".to_string(),
            Value::from(reasoning_intent.budget_tokens.unwrap_or(12_000)),
          )]
          .into_iter()
          .collect(),
        )),
        _ => Some(Value::Object(match reasoning_intent.budget_tokens {
          Some(budget_tokens) => [("budget_tokens".to_string(), Value::from(budget_tokens))]
            .into_iter()
            .collect(),
          None => [(
            "effort".to_string(),
            Value::String(reasoning_intent.effort.unwrap_or_else(|| "medium".to_string())),
          )]
          .into_iter()
          .collect(),
        })),
      };
    }

    if !include.is_empty() {
      include.sort();
      include.dedup();
    }

    ResolvedRequestIntent {
      include: (!include.is_empty()).then_some(include),
      reasoning,
    }
  }

  pub(super) fn dispatch_rerank(
    &self,
    client: &dyn BackendHttpClient,
    config: &BackendConfig,
    protocol: &RerankProtocol,
    request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    self.implementation().dispatch_rerank(client, config, protocol, request)
  }
}

pub(super) fn resolve_chat_request_layer(
  config: &BackendConfig,
  protocol: &ChatProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_chat_protocol(protocol));
  request_layer.ensure_chat_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_structured_request_layer(
  config: &BackendConfig,
  protocol: &StructuredProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_structured_protocol(protocol));
  request_layer.ensure_structured_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_embedding_request_layer(
  config: &BackendConfig,
  protocol: &EmbeddingProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_embedding_protocol(protocol));
  request_layer.ensure_embedding_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_rerank_request_layer(
  config: &BackendConfig,
  protocol: &RerankProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_rerank_protocol(protocol));
  request_layer.ensure_rerank_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_image_request_layer(
  config: &BackendConfig,
  protocol: &ImageProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_image_protocol(protocol));
  request_layer.ensure_image_compatible(protocol)?;
  Ok(request_layer)
}

pub fn resolve_attachment_reference_plan(
  config: &BackendConfig,
  protocol: &ChatProtocol,
  source: &Value,
) -> Result<AttachmentReferencePlan, BackendError> {
  let request_layer = resolve_chat_request_layer(config, protocol)?;
  Ok(request_layer.plan_attachment_reference(&config.base_url, source))
}

pub fn resolve_request_intent(
  config: &BackendConfig,
  protocol: &ChatProtocol,
  intent: RequestIntent,
) -> Result<ResolvedRequestIntent, BackendError> {
  let request_layer = resolve_chat_request_layer(config, protocol)?;
  Ok(request_layer.resolve_request_intent(intent))
}

fn generic_remote_reference_plan(url: &str) -> AttachmentReferencePlan {
  let Ok(parsed) = url.parse::<url::Url>() else {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  };

  if !matches!(parsed.scheme(), "http" | "https" | "gs") {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  }

  AttachmentReferencePlan {
    mode: AttachmentReferenceMode::Remote,
    reason: AttachmentReferenceReason::GenericRemoteReference,
  }
}

fn plan_gemini_api_attachment_reference(url: &str, base_url: &str) -> AttachmentReferencePlan {
  let Ok(parsed) = url.parse::<url::Url>() else {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  };

  if !matches!(parsed.scheme(), "http" | "https") {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Inline,
      reason: AttachmentReferenceReason::UnsupportedScheme,
    };
  }

  if is_gemini_file_url(&parsed, base_url) {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Remote,
      reason: AttachmentReferenceReason::GeminiApiFileUri,
    };
  }

  if is_youtube_url(&parsed) {
    return AttachmentReferencePlan {
      mode: AttachmentReferenceMode::Remote,
      reason: AttachmentReferenceReason::GeminiApiYoutubeUrl,
    };
  }

  AttachmentReferencePlan {
    mode: AttachmentReferenceMode::Inline,
    reason: AttachmentReferenceReason::GeminiApiInlineHttpUrl,
  }
}

fn is_youtube_url(url: &url::Url) -> bool {
  let hostname = url.host_str().unwrap_or_default().to_ascii_lowercase();
  if hostname == "youtu.be" {
    let path = url.path().trim_matches('/');
    return !path.is_empty()
      && path
        .chars()
        .all(|char| char.is_ascii_alphanumeric() || char == '-' || char == '_');
  }

  if hostname != "youtube.com" && hostname != "www.youtube.com" {
    return false;
  }

  url.path() == "/watch" && url.query_pairs().any(|(key, value)| key == "v" && !value.is_empty())
}

fn is_gemini_file_url(url: &url::Url, base_url: &str) -> bool {
  let Ok(base) = base_url.parse::<url::Url>() else {
    return false;
  };

  if url.origin() != base.origin() {
    return false;
  }

  let base_path = base.path().trim_end_matches('/');
  let expected_prefix = format!("{base_path}/files/");
  url.path().starts_with(&expected_prefix)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{
    AttachmentReferenceMode, AttachmentReferencePlan, AttachmentReferenceReason, BackendRequestLayer, RequestIntent,
    RequestIntentReasoning,
  };

  #[test]
  fn gemini_api_should_inline_generic_http_attachments() {
    let plan = BackendRequestLayer::GeminiApi.plan_attachment_reference(
      "https://generativelanguage.googleapis.com/v1beta",
      &json!({ "url": "https://example.com/a.jpg" }),
    );

    assert_eq!(
      plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Inline,
        reason: AttachmentReferenceReason::GeminiApiInlineHttpUrl,
      }
    );
  }

  #[test]
  fn gemini_api_should_preserve_supported_remote_refs() {
    let base_url = "https://generativelanguage.googleapis.com/v1beta";

    let file_plan = BackendRequestLayer::GeminiApi.plan_attachment_reference(
      base_url,
      &json!({ "url": "https://generativelanguage.googleapis.com/v1beta/files/file-123" }),
    );
    assert_eq!(
      file_plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GeminiApiFileUri,
      }
    );

    let youtube_plan = BackendRequestLayer::GeminiApi
      .plan_attachment_reference(base_url, &json!({ "url": "https://www.youtube.com/watch?v=abc123" }));
    assert_eq!(
      youtube_plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GeminiApiYoutubeUrl,
      }
    );
  }

  #[test]
  fn gemini_vertex_should_preserve_remote_http_attachments() {
    let plan = BackendRequestLayer::GeminiVertex
      .plan_attachment_reference("https://vertex.example", &json!({ "url": "https://example.com/a.jpg" }));

    assert_eq!(
      plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GenericRemoteReference,
      }
    );
  }

  #[test]
  fn anthropic_request_intent_should_force_budget_tokens() {
    let resolved = BackendRequestLayer::Anthropic.resolve_request_intent(RequestIntent {
      include: vec!["citations".to_string()],
      reasoning: Some(RequestIntentReasoning {
        enabled: true,
        effort: Some("high".to_string()),
        budget_tokens: None,
        include_reasoning: false,
      }),
    });

    assert_eq!(resolved.include, Some(vec!["citations".to_string()]));
    assert_eq!(resolved.reasoning, Some(json!({ "budget_tokens": 12000 })));
  }

  #[test]
  fn openai_request_intent_should_merge_reasoning_include_and_effort() {
    let resolved = BackendRequestLayer::ChatCompletions.resolve_request_intent(RequestIntent {
      include: vec!["citations".to_string()],
      reasoning: Some(RequestIntentReasoning {
        enabled: true,
        effort: Some("high".to_string()),
        budget_tokens: None,
        include_reasoning: true,
      }),
    });

    assert_eq!(
      resolved.include,
      Some(vec!["citations".to_string(), "reasoning".to_string()])
    );
    assert_eq!(resolved.reasoning, Some(json!({ "effort": "high" })));
  }
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
