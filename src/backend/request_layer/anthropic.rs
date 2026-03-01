use super::{super::BackendConfig, RequestLayerImpl, join_url};

#[derive(Debug, Clone, Copy)]
pub(super) struct AnthropicRequestLayer;

impl RequestLayerImpl for AnthropicRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/v1/messages")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_anthropic_headers(config, stream)
  }
}

fn build_anthropic_headers(config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
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
    ("x-api-key".to_string(), config.auth_token.clone()),
    ("anthropic-version".to_string(), "2023-06-01".to_string()),
  ];

  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}
