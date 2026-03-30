use serde_json::Value;

use super::{super::BackendConfig, RequestLayerImpl, build_bearer_headers};

#[derive(Debug, Clone, Copy)]
pub(super) struct VertexAnthropicRequestLayer;

impl RequestLayerImpl for VertexAnthropicRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    build_vertex_url(base_url, model, stream)
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }

  fn rewrite_body(&self, mut body: Value) -> Value {
    let Value::Object(payload) = &mut body else {
      return body;
    };

    payload.remove("model");
    payload
      .entry("anthropic_version".to_string())
      .or_insert_with(|| Value::String("vertex-2023-10-16".to_string()));
    body
  }
}

fn build_vertex_url(base_url: &str, model: &str, stream: bool) -> String {
  let base_url = base_url.trim_end_matches('/');
  let method = if stream { "streamRawPredict" } else { "rawPredict" };

  if base_url.ends_with(":rawPredict") || base_url.ends_with(":streamRawPredict") {
    return base_url.to_string();
  }
  if base_url.contains("/models/") {
    return format!("{base_url}:{method}");
  }
  if base_url.ends_with("/models") {
    return format!("{base_url}/{model}:{method}");
  }
  format!("{base_url}/models/{model}:{method}")
}
