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
    BackendProvider::Anthropic => Ok("https://api.anthropic.com/v1"),
    BackendProvider::Gemini => Ok("https://generativelanguage.googleapis.com/v1beta"),
    BackendProvider::Fal => Ok("https://fal.run"),
    BackendProvider::Perplexity => Ok("https://api.perplexity.ai"),
    BackendProvider::GeminiVertex | BackendProvider::AnthropicVertex | BackendProvider::CloudflareWorkersAi => {
      Err(TargetCompileError::EndpointRequired)
    }
  }
}

fn default_request_layer(provider: BackendProvider, operation: BackendOperation) -> BackendRequestLayer {
  match (provider, operation) {
    (BackendProvider::OpenAi, BackendOperation::Rerank) => BackendRequestLayer::ChatCompletions,
    (BackendProvider::OpenAi, BackendOperation::Image) => BackendRequestLayer::OpenaiImages,
    (BackendProvider::OpenAi, _) => BackendRequestLayer::Responses,
    (BackendProvider::Anthropic, _) => BackendRequestLayer::Anthropic,
    (BackendProvider::Gemini, _) => BackendRequestLayer::GeminiApi,
    (BackendProvider::GeminiVertex, _) => BackendRequestLayer::GeminiVertex,
    (BackendProvider::AnthropicVertex, _) => BackendRequestLayer::VertexAnthropic,
    (BackendProvider::CloudflareWorkersAi, _) => BackendRequestLayer::CloudflareWorkersAi,
    (BackendProvider::Fal, _) => BackendRequestLayer::Fal,
    (BackendProvider::Perplexity, _) => BackendRequestLayer::PerplexitySonar,
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
  let base_url = match input.endpoint {
    BackendEndpoint::ProviderDefault => default_endpoint(input.provider)?.to_string(),
    BackendEndpoint::Custom(value) => canonicalize_endpoint(&value)?,
  };
  let protocol = default_protocol(input.provider, input.operation)?;
  Ok(CompiledBackendTarget {
    model,
    protocol,
    config: BackendConfig {
      base_url,
      auth_token: input.credential.expose().into(),
      request_layer: Some(default_request_layer(input.provider, input.operation)),
      headers: Default::default(),
      no_streaming: false,
      timeout_ms: input.timeout_ms,
      egress_policy: input.egress_policy,
    },
  })
}

fn default_protocol(
  provider: BackendProvider,
  operation: BackendOperation,
) -> Result<BackendProtocol, TargetCompileError> {
  let protocol = match (provider, operation) {
    (BackendProvider::OpenAi, BackendOperation::Chat) => BackendProtocol::Chat(ChatProtocol::OpenaiResponses),
    (BackendProvider::OpenAi, BackendOperation::Structured) => {
      BackendProtocol::Structured(StructuredProtocol::OpenaiResponses)
    }
    (BackendProvider::OpenAi, BackendOperation::Embedding) => BackendProtocol::Embedding(EmbeddingProtocol::Openai),
    (BackendProvider::OpenAi, BackendOperation::Rerank) => BackendProtocol::Rerank(RerankProtocol::OpenaiChatLogprobs),
    (BackendProvider::OpenAi, BackendOperation::Image) => BackendProtocol::Image(ImageProtocol::OpenaiImages),
    (BackendProvider::Anthropic | BackendProvider::AnthropicVertex, BackendOperation::Chat) => {
      BackendProtocol::Chat(ChatProtocol::AnthropicMessages)
    }
    (BackendProvider::Gemini | BackendProvider::GeminiVertex, BackendOperation::Chat) => {
      BackendProtocol::Chat(ChatProtocol::GeminiGenerateContent)
    }
    (BackendProvider::Gemini | BackendProvider::GeminiVertex, BackendOperation::Structured) => {
      BackendProtocol::Structured(StructuredProtocol::GeminiGenerateContent)
    }
    (BackendProvider::Gemini | BackendProvider::GeminiVertex, BackendOperation::Embedding) => {
      BackendProtocol::Embedding(EmbeddingProtocol::Gemini)
    }
    (BackendProvider::Gemini | BackendProvider::GeminiVertex, BackendOperation::Image) => {
      BackendProtocol::Image(ImageProtocol::GeminiGenerateContent)
    }
    (BackendProvider::CloudflareWorkersAi, BackendOperation::Rerank) => {
      BackendProtocol::Rerank(RerankProtocol::CloudflareWorkersAi)
    }
    (BackendProvider::Fal, BackendOperation::Image) => BackendProtocol::Image(ImageProtocol::FalImage),
    _ => {
      return Err(TargetCompileError::InvalidConfig(
        "provider does not support operation".to_string(),
      ));
    }
  };
  Ok(protocol)
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
