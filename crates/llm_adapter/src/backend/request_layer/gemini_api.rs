use serde_json::{Value, json};

use super::{RequestLayerImpl, build_api_key_headers};
use crate::backend::BackendConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct GeminiApiRequestLayer;

impl RequestLayerImpl for GeminiApiRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    build_gemini_url(base_url, model, stream)
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_api_key_headers(config, stream)
  }

  fn rewrite_body(&self, mut body: Value) -> Value {
    if let Value::Object(payload) = &mut body {
      payload.remove("model");
      payload.remove("stream");
    }
    body
  }

  fn build_embedding_url(&self, base_url: &str, model: &str) -> Result<String, crate::backend::BackendError> {
    Ok(build_gemini_embedding_url(base_url, model))
  }

  fn rewrite_embedding_body(&self, body: Value) -> Value {
    let Value::Object(payload) = body else {
      return body;
    };

    let model = payload
      .get("model")
      .and_then(Value::as_str)
      .unwrap_or_default()
      .to_string();
    let dimensions = payload.get("dimensions").and_then(Value::as_u64);
    let task_type = payload
      .get("task_type")
      .and_then(Value::as_str)
      .map(ToString::to_string);
    let inputs = payload
      .get("inputs")
      .and_then(Value::as_array)
      .cloned()
      .unwrap_or_default();

    Value::Object(serde_json::Map::from_iter([(
      "requests".to_string(),
      Value::Array(
        inputs
          .into_iter()
          .filter_map(|value| {
            let text = value.as_str()?.to_string();
            let mut request = serde_json::Map::from_iter([
              ("model".to_string(), Value::String(format!("models/{model}"))),
              (
                "content".to_string(),
                json!({
                  "role": "user",
                  "parts": [{ "text": text }],
                }),
              ),
            ]);
            if let Some(dimensions) = dimensions {
              request.insert("outputDimensionality".to_string(), Value::Number(dimensions.into()));
            }
            if let Some(task_type) = &task_type {
              request.insert("taskType".to_string(), Value::String(task_type.clone()));
            }
            Some(Value::Object(request))
          })
          .collect(),
      ),
    )]))
  }
}

fn build_gemini_url(base_url: &str, model: &str, stream: bool) -> String {
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

fn build_gemini_embedding_url(base_url: &str, model: &str) -> String {
  let base_url = base_url.trim_end_matches('/');
  if base_url.ends_with(":batchEmbedContents") {
    base_url.to_string()
  } else if base_url.contains("/models/") {
    format!("{base_url}:batchEmbedContents")
  } else if base_url.ends_with("/models") {
    format!("{base_url}/{model}:batchEmbedContents")
  } else {
    format!("{base_url}/models/{model}:batchEmbedContents")
  }
}
