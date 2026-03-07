use serde_json::Value;

use super::{RequestLayerImpl, build_bearer_headers};
use crate::backend::BackendConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct GeminiVertexRequestLayer;

impl RequestLayerImpl for GeminiVertexRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    build_gemini_vertex_url(base_url, model, stream)
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }

  fn rewrite_body(&self, mut body: Value) -> Value {
    if let Value::Object(payload) = &mut body {
      payload.remove("model");
      payload.remove("stream");
    }
    body
  }
}

fn build_gemini_vertex_url(base_url: &str, model: &str, stream: bool) -> String {
  let base_url = base_url.trim_end_matches('/');
  let method = if stream {
    "streamGenerateContent"
  } else {
    "generateContent"
  };

  let mut url = if base_url.ends_with(":generateContent") || base_url.ends_with(":streamGenerateContent") {
    base_url.to_string()
  } else if base_url.contains("/models/") {
    format!("{base_url}:{method}")
  } else if base_url.ends_with("/models") {
    format!("{base_url}/{model}:{method}")
  } else {
    format!("{base_url}/models/{model}:{method}")
  };

  if stream && !url.contains("alt=sse") {
    url.push_str(if url.contains('?') { "&alt=sse" } else { "?alt=sse" });
  }

  url
}
