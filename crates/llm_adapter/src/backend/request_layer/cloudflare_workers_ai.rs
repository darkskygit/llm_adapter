use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  super::{BackendConfig, BackendError, BackendHttpClient, BackendProtocol, HttpRequest},
  RequestLayerImpl, build_bearer_headers, build_extra_headers, join_url,
};
use crate::core::{RerankRequest, RerankResponse};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloudflareRerankStrategy {
  DefaultLogprobs,
  DisableThinkingLogprobs,
  NativeBge,
}

const CLOUDFLARE_RERANK_STRATEGIES: &[(&str, CloudflareRerankStrategy)] = &[
  ("@cf/baai/bge-reranker-base", CloudflareRerankStrategy::NativeBge),
  (
    "@cf/qwen/qwen3-30b-a3b-fp8",
    CloudflareRerankStrategy::DisableThinkingLogprobs,
  ),
  (
    "@cf/zai-org/glm-4.7-flash",
    CloudflareRerankStrategy::DisableThinkingLogprobs,
  ),
];

#[derive(Debug, Deserialize)]
struct CloudflareNativeRerankEnvelope {
  result: CloudflareNativeRerankResult,
}

#[derive(Debug, Deserialize)]
struct CloudflareNativeRerankResult {
  response: Vec<CloudflareNativeRerankScore>,
}

#[derive(Debug, Deserialize)]
struct CloudflareNativeRerankScore {
  id: usize,
  score: f64,
}

fn rerank_strategy_for_model(model: &str) -> CloudflareRerankStrategy {
  CLOUDFLARE_RERANK_STRATEGIES
    .iter()
    .find_map(|(candidate, strategy)| (*candidate == model).then_some(*strategy))
    .unwrap_or(CloudflareRerankStrategy::DefaultLogprobs)
}

fn rerank_strategy_for_body(body: &Value) -> CloudflareRerankStrategy {
  body
    .get("model")
    .and_then(Value::as_str)
    .map(rerank_strategy_for_model)
    .unwrap_or(CloudflareRerankStrategy::DefaultLogprobs)
}

fn merge_disable_thinking(mut body: Map<String, Value>) -> Value {
  let mut chat_template_kwargs = body
    .remove("chat_template_kwargs")
    .and_then(|value| value.as_object().cloned())
    .unwrap_or_default();
  chat_template_kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
  body.insert("chat_template_kwargs".to_string(), Value::Object(chat_template_kwargs));
  Value::Object(body)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CloudflareWorkersAiRequestLayer;

impl RequestLayerImpl for CloudflareWorkersAiRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/v1/chat/completions")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }

  fn build_embedding_url(&self, base_url: &str, _model: &str) -> Result<String, crate::backend::BackendError> {
    Ok(join_url(base_url, "/v1/embeddings"))
  }

  fn rewrite_rerank_body(&self, body: Value) -> Value {
    match (rerank_strategy_for_body(&body), body) {
      (CloudflareRerankStrategy::DisableThinkingLogprobs, Value::Object(body)) => merge_disable_thinking(body),
      (_, body) => body,
    }
  }

  fn dispatch_rerank(
    &self,
    client: &dyn BackendHttpClient,
    config: &BackendConfig,
    _protocol: &BackendProtocol,
    request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    if rerank_strategy_for_model(&request.model) != CloudflareRerankStrategy::NativeBge {
      return Ok(None);
    }

    let mut headers = self.build_rerank_headers(config);
    headers.extend(build_extra_headers(config));

    let mut body = Map::from_iter([
      ("query".to_string(), Value::String(request.query.clone())),
      (
        "contexts".to_string(),
        Value::Array(
          request
            .candidates
            .iter()
            .map(|candidate| json!({ "text": candidate.text }))
            .collect::<Vec<_>>(),
        ),
      ),
    ]);
    if let Some(top_k) = request.top_n {
      body.insert("top_k".to_string(), json!(top_k));
    }

    let response = client.post_json(HttpRequest {
      url: join_url(&config.base_url, &format!("/run/{}", request.model)),
      headers,
      body: Value::Object(body),
      timeout_ms: config.timeout_ms,
    })?;

    let envelope: CloudflareNativeRerankEnvelope = serde_json::from_value(response.body)?;
    let mut scores = vec![0.0; request.candidates.len()];
    for entry in envelope.result.response {
      if let Some(score) = scores.get_mut(entry.id) {
        *score = entry.score;
      }
    }

    Ok(Some(RerankResponse {
      model: request.model.clone(),
      scores,
    }))
  }
}
