use std::{
  fmt,
  net::{IpAddr, SocketAddr, ToSocketAddrs},
};

use thiserror::Error;
use url::Url;
use zeroize::Zeroizing;

use crate::backend::{
  BackendConfig, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, ImageProtocol, RerankProtocol,
  StructuredProtocol,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendProvider {
  OpenAi,
  Anthropic,
  Gemini,
  GeminiVertex,
  AnthropicVertex,
  CloudflareWorkersAi,
  Fal,
  Perplexity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOperation {
  Chat,
  Structured,
  Embedding,
  Rerank,
  Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiDialect {
  Responses,
  ChatCompletions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendProtocol {
  Chat(ChatProtocol),
  Structured(StructuredProtocol),
  Embedding(EmbeddingProtocol),
  Rerank(RerankProtocol),
  Image(ImageProtocol),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendEndpoint {
  ProviderDefault,
  Custom(String),
}

pub struct BackendCredential(Zeroizing<String>);

impl BackendCredential {
  #[must_use]
  pub fn new(value: String) -> Self {
    Self(Zeroizing::new(value))
  }

  #[must_use]
  pub(crate) fn expose(&self) -> &str {
    self.0.as_str()
  }
}

impl fmt::Debug for BackendCredential {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter.write_str("BackendCredential([REDACTED])")
  }
}

pub struct BackendTargetInput {
  pub provider: BackendProvider,
  pub operation: BackendOperation,
  pub endpoint: BackendEndpoint,
  pub openai_dialect: Option<OpenAiDialect>,
  pub model: String,
  pub credential: BackendCredential,
  pub timeout_ms: Option<u64>,
  pub egress_policy: EgressPolicy,
}

pub struct CompiledBackendTarget {
  pub model: String,
  pub config: BackendConfig,
  pub protocol: BackendProtocol,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EgressPolicy {
  #[default]
  PublicOnly,
  AllowPrivate,
}

#[derive(Debug, Error)]
pub enum TargetCompileError {
  #[error("invalid target configuration: {0}")]
  InvalidConfig(String),
  #[error("model id is required")]
  EmptyModel,
  #[error("model id exceeds 512 bytes")]
  ModelTooLong,
  #[error("provider requires a custom endpoint")]
  EndpointRequired,
  #[error("invalid endpoint: {0}")]
  InvalidEndpoint(String),
  #[error("endpoint host could not be resolved")]
  UnresolvedEndpoint,
  #[error("endpoint resolves to a disallowed network address")]
  DisallowedAddress,
}

pub fn canonicalize_endpoint(value: &str) -> Result<String, TargetCompileError> {
  let mut url = Url::parse(value.trim()).map_err(|error| TargetCompileError::InvalidEndpoint(error.to_string()))?;
  if !matches!(url.scheme(), "http" | "https")
    || url.host_str().is_none()
    || !url.username().is_empty()
    || url.password().is_some()
  {
    return Err(TargetCompileError::InvalidEndpoint(
      "endpoint must be an HTTP(S) URL without user info".to_string(),
    ));
  }
  url.set_query(None);
  url.set_fragment(None);
  let path = url.path().trim_end_matches('/').to_string();
  url.set_path(if path.is_empty() { "/" } else { &path });
  Ok(url.to_string().trim_end_matches('/').to_string())
}

fn default_endpoint(provider: BackendProvider) -> Result<&'static str, TargetCompileError> {
  match provider {
    BackendProvider::OpenAi => Ok("https://api.openai.com/v1"),
    BackendProvider::Anthropic => Ok("https://api.anthropic.com"),
    BackendProvider::Gemini => Ok("https://generativelanguage.googleapis.com/v1beta"),
    BackendProvider::Fal => Ok("https://fal.run"),
    BackendProvider::Perplexity => Ok("https://api.perplexity.ai"),
    BackendProvider::GeminiVertex | BackendProvider::AnthropicVertex | BackendProvider::CloudflareWorkersAi => {
      Err(TargetCompileError::EndpointRequired)
    }
  }
}

pub fn compile_backend_target(input: BackendTargetInput) -> Result<CompiledBackendTarget, TargetCompileError> {
  let model = input.model.trim().to_string();
  if model.is_empty() {
    return Err(TargetCompileError::EmptyModel);
  }
  if model.len() > 512 {
    return Err(TargetCompileError::ModelTooLong);
  }
  let (protocol, request_layer) = if input.provider == BackendProvider::OpenAi {
    if matches!(&input.endpoint, BackendEndpoint::Custom(_)) && input.openai_dialect.is_none() {
      return Err(TargetCompileError::InvalidConfig(
        "OpenAI custom endpoint requires a dialect".to_string(),
      ));
    }
    compile_openai_binding(
      input.operation,
      input.openai_dialect.unwrap_or(OpenAiDialect::Responses),
    )?
  } else {
    if input.openai_dialect.is_some() {
      return Err(TargetCompileError::InvalidConfig(
        "OpenAI dialect is only valid for the OpenAI provider".to_string(),
      ));
    }
    compile_provider_binding(input.provider, input.operation)?
  };
  let base_url = match input.endpoint {
    BackendEndpoint::ProviderDefault => default_endpoint(input.provider)?.to_string(),
    BackendEndpoint::Custom(value) => canonicalize_endpoint(&value)?,
  };
  Ok(CompiledBackendTarget {
    model,
    protocol,
    config: BackendConfig {
      base_url,
      auth_token: input.credential.expose().into(),
      request_layer: Some(request_layer),
      headers: Default::default(),
      no_streaming: false,
      timeout_ms: input.timeout_ms,
      egress_policy: input.egress_policy,
    },
  })
}

fn compile_openai_binding(
  operation: BackendOperation,
  dialect: OpenAiDialect,
) -> Result<(BackendProtocol, BackendRequestLayer), TargetCompileError> {
  let binding = match (operation, dialect) {
    (BackendOperation::Chat, OpenAiDialect::Responses) => (
      BackendProtocol::Chat(ChatProtocol::OpenaiResponses),
      BackendRequestLayer::Responses,
    ),
    (BackendOperation::Chat, OpenAiDialect::ChatCompletions) => (
      BackendProtocol::Chat(ChatProtocol::OpenaiChatCompletions),
      BackendRequestLayer::ChatCompletions,
    ),
    (BackendOperation::Structured, OpenAiDialect::Responses) => (
      BackendProtocol::Structured(StructuredProtocol::OpenaiResponses),
      BackendRequestLayer::Responses,
    ),
    (BackendOperation::Structured, OpenAiDialect::ChatCompletions) => (
      BackendProtocol::Structured(StructuredProtocol::OpenaiChatCompletions),
      BackendRequestLayer::ChatCompletions,
    ),
    (BackendOperation::Embedding, _) => (
      BackendProtocol::Embedding(EmbeddingProtocol::Openai),
      BackendRequestLayer::Responses,
    ),
    (BackendOperation::Rerank, _) => (
      BackendProtocol::Rerank(RerankProtocol::OpenaiChatLogprobs),
      BackendRequestLayer::ChatCompletions,
    ),
    (BackendOperation::Image, _) => (
      BackendProtocol::Image(ImageProtocol::OpenaiImages),
      BackendRequestLayer::OpenaiImages,
    ),
  };
  Ok(binding)
}

fn compile_provider_binding(
  provider: BackendProvider,
  operation: BackendOperation,
) -> Result<(BackendProtocol, BackendRequestLayer), TargetCompileError> {
  let binding = match (provider, operation) {
    (BackendProvider::Anthropic, BackendOperation::Chat) => (
      BackendProtocol::Chat(ChatProtocol::AnthropicMessages),
      BackendRequestLayer::Anthropic,
    ),
    (BackendProvider::AnthropicVertex, BackendOperation::Chat) => (
      BackendProtocol::Chat(ChatProtocol::AnthropicMessages),
      BackendRequestLayer::VertexAnthropic,
    ),
    (BackendProvider::Gemini, BackendOperation::Chat) => (
      BackendProtocol::Chat(ChatProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiApi,
    ),
    (BackendProvider::GeminiVertex, BackendOperation::Chat) => (
      BackendProtocol::Chat(ChatProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiVertex,
    ),
    (BackendProvider::Gemini, BackendOperation::Structured) => (
      BackendProtocol::Structured(StructuredProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiApi,
    ),
    (BackendProvider::GeminiVertex, BackendOperation::Structured) => (
      BackendProtocol::Structured(StructuredProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiVertex,
    ),
    (BackendProvider::Gemini, BackendOperation::Embedding) => (
      BackendProtocol::Embedding(EmbeddingProtocol::Gemini),
      BackendRequestLayer::GeminiApi,
    ),
    (BackendProvider::GeminiVertex, BackendOperation::Embedding) => (
      BackendProtocol::Embedding(EmbeddingProtocol::Gemini),
      BackendRequestLayer::GeminiVertex,
    ),
    (BackendProvider::Gemini, BackendOperation::Image) => (
      BackendProtocol::Image(ImageProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiApi,
    ),
    (BackendProvider::GeminiVertex, BackendOperation::Image) => (
      BackendProtocol::Image(ImageProtocol::GeminiGenerateContent),
      BackendRequestLayer::GeminiVertex,
    ),
    (BackendProvider::CloudflareWorkersAi, BackendOperation::Rerank) => (
      BackendProtocol::Rerank(RerankProtocol::CloudflareWorkersAi),
      BackendRequestLayer::CloudflareWorkersAi,
    ),
    (BackendProvider::Fal, BackendOperation::Image) => (
      BackendProtocol::Image(ImageProtocol::FalImage),
      BackendRequestLayer::Fal,
    ),
    _ => {
      return Err(TargetCompileError::InvalidConfig(
        "provider does not support operation".to_string(),
      ));
    }
  };
  Ok(binding)
}

impl EgressPolicy {
  pub fn resolve(self, url: &str) -> Result<Vec<SocketAddr>, TargetCompileError> {
    let url = Url::parse(url).map_err(|error| TargetCompileError::InvalidEndpoint(error.to_string()))?;
    let host = url
      .host_str()
      .ok_or_else(|| TargetCompileError::InvalidEndpoint("missing host".to_string()))?;
    let port = url
      .port_or_known_default()
      .ok_or_else(|| TargetCompileError::InvalidEndpoint("missing port".to_string()))?;
    let addresses = (host, port)
      .to_socket_addrs()
      .map_err(|_| TargetCompileError::UnresolvedEndpoint)?
      .collect::<Vec<_>>();
    if addresses.is_empty() {
      return Err(TargetCompileError::UnresolvedEndpoint);
    }
    if self == Self::PublicOnly && addresses.iter().any(|address| !is_public(address.ip())) {
      return Err(TargetCompileError::DisallowedAddress);
    }
    Ok(addresses)
  }
}

fn is_public(ip: IpAddr) -> bool {
  match ip {
    IpAddr::V4(ip) => {
      !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.is_multicast())
    }
    IpAddr::V6(ip) => {
      !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (ip.segments()[0] & 0xfe00) == 0xfc00
        || (ip.segments()[0] & 0xffc0) == 0xfe80
        || (ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0db8))
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keeps_opaque_model_id_and_canonicalizes_custom_endpoint() {
    let target = compile_backend_target(BackendTargetInput {
      provider: BackendProvider::OpenAi,
      operation: BackendOperation::Chat,
      endpoint: BackendEndpoint::Custom("https://example.com/v1/?token=discarded#fragment".to_string()),
      openai_dialect: Some(OpenAiDialect::Responses),
      model: " vendor/model:latest ".to_string(),
      credential: BackendCredential::new("secret".to_string()),
      timeout_ms: None,
      egress_policy: EgressPolicy::PublicOnly,
    })
    .unwrap();
    assert_eq!(target.model, "vendor/model:latest");
    assert_eq!(target.config.base_url, "https://example.com/v1");
  }

  #[test]
  fn openai_provider_default_is_a_versioned_api_base() {
    for operation in [
      BackendOperation::Chat,
      BackendOperation::Structured,
      BackendOperation::Embedding,
      BackendOperation::Rerank,
      BackendOperation::Image,
    ] {
      let target = compile_backend_target(BackendTargetInput {
        provider: BackendProvider::OpenAi,
        operation,
        endpoint: BackendEndpoint::ProviderDefault,
        openai_dialect: None,
        model: "gpt-5.6-luna".to_string(),
        credential: BackendCredential::new("secret".to_string()),
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      })
      .unwrap();
      assert_eq!(target.config.base_url, "https://api.openai.com/v1");
    }
  }

  #[test]
  fn anthropic_provider_default_is_an_unversioned_api_base() {
    let target = compile_backend_target(BackendTargetInput {
      provider: BackendProvider::Anthropic,
      operation: BackendOperation::Chat,
      endpoint: BackendEndpoint::ProviderDefault,
      openai_dialect: None,
      model: "claude-sonnet-4-6".to_string(),
      credential: BackendCredential::new("secret".to_string()),
      timeout_ms: None,
      egress_policy: EgressPolicy::PublicOnly,
    })
    .unwrap();
    assert_eq!(target.config.base_url, "https://api.anthropic.com");
  }

  #[test]
  fn compiles_openai_dialect_as_an_atomic_protocol_and_request_layer() {
    let cases = [
      (
        BackendOperation::Chat,
        OpenAiDialect::Responses,
        BackendProtocol::Chat(ChatProtocol::OpenaiResponses),
        BackendRequestLayer::Responses,
      ),
      (
        BackendOperation::Chat,
        OpenAiDialect::ChatCompletions,
        BackendProtocol::Chat(ChatProtocol::OpenaiChatCompletions),
        BackendRequestLayer::ChatCompletions,
      ),
      (
        BackendOperation::Structured,
        OpenAiDialect::Responses,
        BackendProtocol::Structured(StructuredProtocol::OpenaiResponses),
        BackendRequestLayer::Responses,
      ),
      (
        BackendOperation::Structured,
        OpenAiDialect::ChatCompletions,
        BackendProtocol::Structured(StructuredProtocol::OpenaiChatCompletions),
        BackendRequestLayer::ChatCompletions,
      ),
      (
        BackendOperation::Embedding,
        OpenAiDialect::ChatCompletions,
        BackendProtocol::Embedding(EmbeddingProtocol::Openai),
        BackendRequestLayer::Responses,
      ),
      (
        BackendOperation::Rerank,
        OpenAiDialect::Responses,
        BackendProtocol::Rerank(RerankProtocol::OpenaiChatLogprobs),
        BackendRequestLayer::ChatCompletions,
      ),
      (
        BackendOperation::Image,
        OpenAiDialect::Responses,
        BackendProtocol::Image(ImageProtocol::OpenaiImages),
        BackendRequestLayer::OpenaiImages,
      ),
    ];

    for (operation, dialect, protocol, request_layer) in cases {
      let target = compile_backend_target(BackendTargetInput {
        provider: BackendProvider::OpenAi,
        operation,
        endpoint: BackendEndpoint::Custom("https://example.com/nested/api/v1".to_string()),
        openai_dialect: Some(dialect),
        model: "model".to_string(),
        credential: BackendCredential::new("secret".to_string()),
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      })
      .unwrap();

      assert_eq!(target.protocol, protocol);
      assert_eq!(target.config.request_layer, Some(request_layer));
      assert_eq!(target.config.base_url, "https://example.com/nested/api/v1");
    }
  }

  #[test]
  fn rejects_invalid_openai_dialect_placement() {
    let invalid = [
      BackendTargetInput {
        provider: BackendProvider::Anthropic,
        operation: BackendOperation::Chat,
        endpoint: BackendEndpoint::ProviderDefault,
        openai_dialect: Some(OpenAiDialect::Responses),
        model: "model".to_string(),
        credential: BackendCredential::new("secret".to_string()),
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      },
      BackendTargetInput {
        provider: BackendProvider::OpenAi,
        operation: BackendOperation::Chat,
        endpoint: BackendEndpoint::Custom("https://example.com/v1".to_string()),
        openai_dialect: None,
        model: "model".to_string(),
        credential: BackendCredential::new("secret".to_string()),
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      },
    ];

    assert!(
      invalid
        .into_iter()
        .all(|input| matches!(compile_backend_target(input), Err(TargetCompileError::InvalidConfig(_))))
    );
  }

  #[test]
  fn rejects_private_addresses_for_public_only_policy() {
    assert!(matches!(
      EgressPolicy::PublicOnly.resolve("http://127.0.0.1:3000"),
      Err(TargetCompileError::DisallowedAddress)
    ));
    assert!(EgressPolicy::AllowPrivate.resolve("http://127.0.0.1:3000").is_ok());
  }

  #[test]
  fn redacts_credential_debug() {
    assert_eq!(
      format!("{:?}", BackendCredential::new("never-print-this".to_string())),
      "BackendCredential([REDACTED])"
    );
  }
}
