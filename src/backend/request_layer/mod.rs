use serde_json::Value;

use super::{BackendConfig, BackendError, BackendProtocol, BackendRequestLayer};

mod anthropic;
mod chat_completions;
mod responses;
mod vertex;

use self::{
  anthropic::AnthropicRequestLayer, chat_completions::ChatCompletionsRequestLayer, responses::ResponsesRequestLayer,
  vertex::VertexRequestLayer,
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
}

const ANTHROPIC_LAYER: AnthropicRequestLayer = AnthropicRequestLayer;
const CHAT_COMPLETIONS_LAYER: ChatCompletionsRequestLayer = ChatCompletionsRequestLayer;
const RESPONSES_LAYER: ResponsesRequestLayer = ResponsesRequestLayer;
const VERTEX_LAYER: VertexRequestLayer = VertexRequestLayer;

impl BackendProtocol {
  pub(super) fn as_str(&self) -> &'static str {
    match self {
      BackendProtocol::OpenaiChatCompletions => "openai_chat_completions",
      BackendProtocol::OpenaiResponses => "openai_responses",
      BackendProtocol::AnthropicMessages => "anthropic_messages",
    }
  }
}

impl BackendRequestLayer {
  pub(super) fn from_protocol(protocol: &BackendProtocol) -> Self {
    match protocol {
      BackendProtocol::OpenaiChatCompletions => Self::ChatCompletions,
      BackendProtocol::OpenaiResponses => Self::Responses,
      BackendProtocol::AnthropicMessages => Self::Anthropic,
    }
  }

  fn ensure_compatible(&self, protocol: &BackendProtocol) -> Result<(), BackendError> {
    let compatible = matches!(
      (self, protocol),
      (
        BackendRequestLayer::ChatCompletions,
        BackendProtocol::OpenaiChatCompletions
      ) | (BackendRequestLayer::Responses, BackendProtocol::OpenaiResponses)
        | (BackendRequestLayer::Anthropic, BackendProtocol::AnthropicMessages)
        | (BackendRequestLayer::Vertex, BackendProtocol::AnthropicMessages)
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
      BackendRequestLayer::Responses => "responses",
      BackendRequestLayer::Vertex => "vertex",
    }
  }

  fn implementation(&self) -> &'static dyn RequestLayerImpl {
    match self {
      BackendRequestLayer::Anthropic => &ANTHROPIC_LAYER,
      BackendRequestLayer::ChatCompletions => &CHAT_COMPLETIONS_LAYER,
      BackendRequestLayer::Responses => &RESPONSES_LAYER,
      BackendRequestLayer::Vertex => &VERTEX_LAYER,
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
