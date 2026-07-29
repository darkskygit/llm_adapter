use super::{super::BackendConfig, RequestLayerImpl, build_bearer_headers, join_url};

#[derive(Debug, Clone, Copy)]
pub(super) struct PerplexitySonarRequestLayer;

impl RequestLayerImpl for PerplexitySonarRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/v1/sonar")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }
}
