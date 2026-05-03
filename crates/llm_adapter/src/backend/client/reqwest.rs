use std::time::Duration;

use reqwest::{
  blocking::Client,
  header::{HeaderMap, HeaderName, HeaderValue},
};

use super::{
  super::{BackendError, BackendHttpClient, HttpRequest, HttpResponse},
  shared::{serialize_http_body, stream_utf8_chunks},
};

#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
  client: Client,
}

impl Default for ReqwestHttpClient {
  fn default() -> Self {
    let client = Client::builder()
      .build()
      .expect("failed to construct reqwest blocking client");
    Self { client }
  }
}

impl BackendHttpClient for ReqwestHttpClient {
  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
    let mut request = request;
    let body = serialize_http_body(&request.body, &mut request.headers)?;
    let headers = build_header_map(&request.headers)?;

    let mut request_builder = self.client.post(&request.url).headers(headers).body(body);

    if let Some(timeout_ms) = request.timeout_ms {
      request_builder = request_builder.timeout(Duration::from_millis(timeout_ms));
    }

    let response = request_builder.send().map_err(map_reqwest_error)?;

    let status = response.status().as_u16();
    let body = response.bytes().map_err(map_reqwest_error)?;

    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: String::from_utf8_lossy(&body).to_string(),
      });
    }

    let parsed_body = serde_json::from_slice(&body)?;
    Ok(HttpResponse {
      status,
      body: parsed_body,
    })
  }

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError> {
    let mut request = request;
    let body = serialize_http_body(&request.body, &mut request.headers)?;
    let headers = build_header_map(&request.headers)?;

    let mut request_builder = self.client.post(&request.url).headers(headers).body(body);

    if let Some(timeout_ms) = request.timeout_ms {
      request_builder = request_builder.timeout(Duration::from_millis(timeout_ms));
    }

    let mut response = request_builder.send().map_err(map_reqwest_error)?;

    let status = response.status().as_u16();

    if !(200..300).contains(&status) {
      let body = response.bytes().map_err(map_reqwest_error)?;
      return Err(BackendError::UpstreamStatus {
        status,
        body: String::from_utf8_lossy(&body).to_string(),
      });
    }

    stream_utf8_chunks(&mut response, on_chunk)
  }
}

fn map_reqwest_error(error: reqwest::Error) -> BackendError {
  if error.is_timeout() {
    BackendError::Timeout {
      message: error.to_string(),
    }
  } else {
    BackendError::Transport {
      message: error.to_string(),
    }
  }
}

fn build_header_map(headers: &[(String, String)]) -> Result<HeaderMap, BackendError> {
  let mut header_map = HeaderMap::new();

  for (key, value) in headers {
    let header_name = HeaderName::from_bytes(key.as_bytes()).map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
    let header_value = HeaderValue::from_str(value).map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
    header_map.insert(header_name, header_value);
  }

  Ok(header_map)
}
