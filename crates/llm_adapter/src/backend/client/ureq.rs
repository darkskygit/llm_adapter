use std::{fmt, io::Read, net::SocketAddr, time::Duration};

use ureq::{
  Agent, RequestBuilder,
  unversioned::{
    resolver::{ResolvedSocketAddrs, Resolver},
    transport::{DefaultConnector, NextTimeout},
  },
};

use super::{
  super::{BackendError, BackendHttpClient, HttpRequest, HttpResponse, HttpUploadRequest},
  shared::{map_io_error, serialize_http_body, stream_utf8_chunks},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct UreqHttpClient;

struct PinnedResolver {
  host: String,
  addresses: Vec<SocketAddr>,
}

impl fmt::Debug for PinnedResolver {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("PinnedResolver")
      .field("host", &self.host)
      .finish()
  }
}

impl Resolver for PinnedResolver {
  fn resolve(
    &self,
    uri: &ureq::http::Uri,
    _config: &ureq::config::Config,
    _timeout: NextTimeout,
  ) -> Result<ResolvedSocketAddrs, ureq::Error> {
    if uri.host() != Some(self.host.as_str()) {
      return Err(ureq::Error::HostNotFound);
    }
    let mut resolved = self.empty();
    self
      .addresses
      .iter()
      .take(16)
      .for_each(|address| resolved.push(*address));
    Ok(resolved)
  }
}

impl BackendHttpClient for UreqHttpClient {
  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
    let mut request = request;
    let body = serialize_http_body(&request.body, &mut request.headers)?;
    let mut response = self
      .build_request(&request)?
      .send(body.as_slice())
      .map_err(map_ureq_error)?;

    let status = response.status().as_u16();

    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: read_response_text(&mut response)?,
      });
    }

    let body = serde_json::from_slice(&read_response_bytes(&mut response)?)?;
    Ok(HttpResponse { status, body })
  }

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError> {
    let mut request = request;
    let body = serialize_http_body(&request.body, &mut request.headers)?;
    let mut response = self
      .build_request(&request)?
      .send(body.as_slice())
      .map_err(map_ureq_error)?;

    let status = response.status().as_u16();

    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: read_response_text(&mut response)?,
      });
    }

    let mut reader = response.body_mut().as_reader();
    stream_utf8_chunks(&mut reader, on_chunk)
  }

  fn put_bytes(&self, request: HttpUploadRequest) -> Result<(), BackendError> {
    let mut response = self
      .build_upload_request(&request)?
      .send(request.bytes.as_slice())
      .map_err(map_ureq_error)?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: read_response_text(&mut response)?,
      });
    }
    Ok(())
  }
}

impl UreqHttpClient {
  fn build_request(&self, request: &HttpRequest) -> Result<RequestBuilder<ureq::typestate::WithBody>, BackendError> {
    let url = url::Url::parse(&request.url).map_err(|error| BackendError::Transport {
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
    let config = Agent::config_builder().max_redirects(0).build();
    let agent = Agent::with_parts(
      config,
      DefaultConnector::new(),
      PinnedResolver {
        host: host.to_string(),
        addresses,
      },
    );
    let mut request_builder = agent.post(&request.url);
    for (key, value) in &request.headers {
      request_builder = request_builder.header(key.as_str(), value.as_str());
    }

    let mut config = request_builder.config().http_status_as_error(false);

    if let Some(timeout_ms) = request.timeout_ms {
      let timeout = Duration::from_millis(timeout_ms);
      config = config
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .timeout_connect(Some(timeout))
        .timeout_send_request(Some(timeout))
        .timeout_send_body(Some(timeout))
        .timeout_recv_response(Some(timeout))
        .timeout_recv_body(Some(timeout));
    }

    Ok(config.build())
  }

  fn build_upload_request(
    &self,
    request: &HttpUploadRequest,
  ) -> Result<RequestBuilder<ureq::typestate::WithBody>, BackendError> {
    let url = url::Url::parse(&request.url).map_err(|error| BackendError::Transport {
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
    let agent = Agent::with_parts(
      Agent::config_builder().max_redirects(0).build(),
      DefaultConnector::new(),
      PinnedResolver {
        host: host.to_string(),
        addresses,
      },
    );
    let mut builder = agent.put(&request.url);
    for (key, value) in &request.headers {
      builder = builder.header(key.as_str(), value.as_str());
    }
    let mut config = builder.config().http_status_as_error(false);
    if let Some(timeout_ms) = request.timeout_ms {
      let timeout = Duration::from_millis(timeout_ms);
      config = config
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .timeout_connect(Some(timeout))
        .timeout_send_request(Some(timeout))
        .timeout_send_body(Some(timeout))
        .timeout_recv_response(Some(timeout))
        .timeout_recv_body(Some(timeout));
    }
    Ok(config.build())
  }
}

fn map_ureq_error(error: ureq::Error) -> BackendError {
  if matches!(error, ureq::Error::Timeout(_)) {
    BackendError::Timeout {
      message: error.to_string(),
    }
  } else {
    BackendError::Transport {
      message: error.to_string(),
    }
  }
}

fn read_response_bytes(response: &mut ureq::http::Response<ureq::Body>) -> Result<Vec<u8>, BackendError> {
  let mut bytes = Vec::new();
  response
    .body_mut()
    .as_reader()
    .read_to_end(&mut bytes)
    .map_err(map_io_error)?;
  Ok(bytes)
}

fn read_response_text(response: &mut ureq::http::Response<ureq::Body>) -> Result<String, BackendError> {
  Ok(String::from_utf8_lossy(&read_response_bytes(response)?).to_string())
}
