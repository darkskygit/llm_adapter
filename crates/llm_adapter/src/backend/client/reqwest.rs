use std::{io::Read, time::Duration};

use reqwest::{
  blocking::Client,
  header::{HeaderMap, HeaderName, HeaderValue},
  redirect::Policy,
};
use url::Url;

use super::{
  super::{
    BackendError, BackendHttpClient, HttpMethod, HttpRawRequest, HttpRawResponse, HttpRequest, HttpResponse,
    HttpUploadRequest,
  },
  shared::{serialize_http_body, stream_utf8_chunks},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ReqwestHttpClient;

fn build_client(request: &HttpRequest) -> Result<Client, BackendError> {
  let url = Url::parse(&request.url).map_err(|error| BackendError::Transport {
    message: error.to_string(),
  })?;
  let host = url.host_str().ok_or_else(|| BackendError::Transport {
    message: "request URL has no host".to_string(),
  })?;
  let addresses = request
    .egress_policy
    .resolve(&request.url)
    .map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
  Client::builder()
    .redirect(Policy::none())
    .resolve_to_addrs(host, &addresses)
    .build()
    .map_err(map_reqwest_error)
}

impl BackendHttpClient for ReqwestHttpClient {
  fn execute(&self, mut request: HttpRawRequest) -> Result<HttpRawResponse, BackendError> {
    let headers = build_header_map(&request.headers)?;
    let client = build_raw_client(&request)?;
    let mut builder = match request.method {
      HttpMethod::Get => client.get(&request.url),
      HttpMethod::Post => client.post(&request.url),
      HttpMethod::Put => client.put(&request.url),
      HttpMethod::Delete => client.delete(&request.url),
    }
    .headers(headers)
    .body(std::mem::take(&mut request.body));
    if let Some(timeout_ms) = request.timeout_ms {
      builder = builder.timeout(Duration::from_millis(timeout_ms));
    }
    let mut response = builder.send().map_err(map_reqwest_error)?;
    let status = response.status().as_u16();
    let headers = response
      .headers()
      .iter()
      .map(|(name, value)| {
        (
          name.as_str().to_string(),
          value.to_str().unwrap_or_default().to_string(),
        )
      })
      .collect();
    let mut body = Vec::new();
    match request.max_response_bytes {
      Some(limit) => response
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut body)
        .map_err(|error| BackendError::Transport {
          message: error.to_string(),
        })?,
      None => response
        .read_to_end(&mut body)
        .map_err(|error| BackendError::Transport {
          message: error.to_string(),
        })?,
    };
    if request.max_response_bytes.is_some_and(|limit| body.len() > limit) {
      return Err(BackendError::Transport {
        message: "HTTP response exceeded configured size limit".to_string(),
      });
    }
    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: String::from_utf8_lossy(&body).to_string(),
      });
    }
    Ok(HttpRawResponse { status, headers, body })
  }

  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
    let mut request = request;
    let body = serialize_http_body(&request.body, &mut request.headers)?;
    let headers = build_header_map(&request.headers)?;

    let client = build_client(&request)?;
    let mut request_builder = client.post(&request.url).headers(headers).body(body);

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

    let client = build_client(&request)?;
    let mut request_builder = client.post(&request.url).headers(headers).body(body);

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

  fn put_bytes(&self, request: HttpUploadRequest) -> Result<(), BackendError> {
    let headers = build_header_map(&request.headers)?;
    let url = Url::parse(&request.url).map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
    let host = url.host_str().ok_or_else(|| BackendError::Transport {
      message: "request URL has no host".to_string(),
    })?;
    let addresses = request
      .egress_policy
      .resolve(&request.url)
      .map_err(|error| BackendError::Transport {
        message: error.to_string(),
      })?;
    let client = Client::builder()
      .redirect(Policy::none())
      .resolve_to_addrs(host, &addresses)
      .build()
      .map_err(map_reqwest_error)?;
    let mut builder = client.put(&request.url).headers(headers).body(request.bytes);
    if let Some(timeout_ms) = request.timeout_ms {
      builder = builder.timeout(Duration::from_millis(timeout_ms));
    }
    let response = builder.send().map_err(map_reqwest_error)?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: String::from_utf8_lossy(&response.bytes().map_err(map_reqwest_error)?).to_string(),
      });
    }
    Ok(())
  }
}

fn build_raw_client(request: &HttpRawRequest) -> Result<Client, BackendError> {
  let url = Url::parse(&request.url).map_err(|error| BackendError::Transport {
    message: error.to_string(),
  })?;
  let host = url.host_str().ok_or_else(|| BackendError::Transport {
    message: "request URL has no host".to_string(),
  })?;
  let addresses = request
    .egress_policy
    .resolve(&request.url)
    .map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
  Client::builder()
    .redirect(Policy::none())
    .resolve_to_addrs(host, &addresses)
    .build()
    .map_err(map_reqwest_error)
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
