use std::{io::Read, time::Duration};

use reqwest::{
  blocking::Client,
  header::{HeaderMap, HeaderName, HeaderValue},
  redirect::Policy,
};
use url::Url;

use super::{
  super::{
    BackendError, BackendHttpClient, HttpFailureContext, HttpFailureKind, HttpFailureStage, HttpMethod, HttpRawRequest,
    HttpRawResponse, HttpRequest, HttpResponse, HttpUploadRequest,
  },
  shared::{map_io_error, serialize_http_body, stream_utf8_chunks},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ReqwestHttpClient;

fn build_client(request: &HttpRequest) -> Result<Client, BackendError> {
  let url = Url::parse(&request.url).map_err(|error| BackendError::Transport {
    message: error.to_string(),
    context: Some(HttpFailureContext {
      stage: HttpFailureStage::Resolve,
      kind: HttpFailureKind::InvalidUrl,
      received_bytes: None,
    }),
  })?;
  let host = url.host_str().ok_or_else(|| BackendError::Transport {
    message: "request URL has no host".to_string(),
    context: Some(HttpFailureContext {
      stage: HttpFailureStage::Resolve,
      kind: HttpFailureKind::InvalidUrl,
      received_bytes: None,
    }),
  })?;
  let addresses = request
    .egress_policy
    .resolve(&request.url)
    .map_err(super::shared::map_egress_error)?;
  Client::builder()
    .redirect(Policy::none())
    .resolve_to_addrs(host, &addresses)
    .build()
    .map_err(|error| map_reqwest_error(error, HttpFailureStage::ClientBuild))
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
    let mut response = builder
      .send()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::Send))?;
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
        .map_err(map_io_error)?,
      None => response.read_to_end(&mut body).map_err(map_io_error)?,
    };
    if request.max_response_bytes.is_some_and(|limit| body.len() > limit) {
      return Err(BackendError::Transport {
        message: "HTTP response exceeded configured size limit".to_string(),
        context: Some(HttpFailureContext {
          stage: HttpFailureStage::BodyRead,
          kind: HttpFailureKind::Body,
          received_bytes: None,
        }),
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

    let response = request_builder
      .send()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::Send))?;

    let status = response.status().as_u16();
    let body = response
      .bytes()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::BodyRead))?;

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

    let mut response = request_builder
      .send()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::Send))?;

    let status = response.status().as_u16();

    if !(200..300).contains(&status) {
      let body = response
        .bytes()
        .map_err(|error| map_reqwest_error(error, HttpFailureStage::BodyRead))?;
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
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::Resolve,
        kind: HttpFailureKind::InvalidUrl,
        received_bytes: None,
      }),
    })?;
    let host = url.host_str().ok_or_else(|| BackendError::Transport {
      message: "request URL has no host".to_string(),
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::Resolve,
        kind: HttpFailureKind::InvalidUrl,
        received_bytes: None,
      }),
    })?;
    let addresses = request
      .egress_policy
      .resolve(&request.url)
      .map_err(super::shared::map_egress_error)?;
    let client = Client::builder()
      .redirect(Policy::none())
      .resolve_to_addrs(host, &addresses)
      .build()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::ClientBuild))?;
    let mut builder = client.put(&request.url).headers(headers).body(request.bytes);
    if let Some(timeout_ms) = request.timeout_ms {
      builder = builder.timeout(Duration::from_millis(timeout_ms));
    }
    let response = builder
      .send()
      .map_err(|error| map_reqwest_error(error, HttpFailureStage::Send))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: String::from_utf8_lossy(
          &response
            .bytes()
            .map_err(|error| map_reqwest_error(error, HttpFailureStage::BodyRead))?,
        )
        .to_string(),
      });
    }
    Ok(())
  }
}

fn build_raw_client(request: &HttpRawRequest) -> Result<Client, BackendError> {
  let url = Url::parse(&request.url).map_err(|error| BackendError::Transport {
    message: error.to_string(),
    context: Some(HttpFailureContext {
      stage: HttpFailureStage::Resolve,
      kind: HttpFailureKind::InvalidUrl,
      received_bytes: None,
    }),
  })?;
  let host = url.host_str().ok_or_else(|| BackendError::Transport {
    message: "request URL has no host".to_string(),
    context: Some(HttpFailureContext {
      stage: HttpFailureStage::Resolve,
      kind: HttpFailureKind::InvalidUrl,
      received_bytes: None,
    }),
  })?;
  let addresses = request
    .egress_policy
    .resolve(&request.url)
    .map_err(super::shared::map_egress_error)?;
  Client::builder()
    .redirect(Policy::none())
    .resolve_to_addrs(host, &addresses)
    .build()
    .map_err(|error| map_reqwest_error(error, HttpFailureStage::ClientBuild))
}

pub(super) fn reqwest_failure_kind(error: &reqwest::Error) -> HttpFailureKind {
  let mut source = std::error::Error::source(error);
  while let Some(cause) = source {
    if let Some(io) = cause.downcast_ref::<std::io::Error>() {
      match io.kind() {
        std::io::ErrorKind::TimedOut => return HttpFailureKind::Timeout,
        std::io::ErrorKind::ConnectionReset => return HttpFailureKind::ConnectionReset,
        std::io::ErrorKind::UnexpectedEof => return HttpFailureKind::UnexpectedEof,
        std::io::ErrorKind::BrokenPipe => return HttpFailureKind::BrokenPipe,
        _ => {}
      }
    }
    source = cause.source();
  }
  if error.is_timeout() {
    HttpFailureKind::Timeout
  } else if error.is_connect() {
    HttpFailureKind::Connect
  } else if error.is_body() {
    HttpFailureKind::Body
  } else if error.is_decode() {
    HttpFailureKind::Decode
  } else if error.is_request() {
    HttpFailureKind::Request
  } else {
    HttpFailureKind::Other
  }
}

fn map_reqwest_error(error: reqwest::Error, stage: HttpFailureStage) -> BackendError {
  let stage = if stage == HttpFailureStage::Send && error.is_connect() {
    HttpFailureStage::Connect
  } else {
    stage
  };
  let kind = reqwest_failure_kind(&error);
  let context = Some(HttpFailureContext {
    stage,
    kind,
    received_bytes: None,
  });
  if error.is_timeout() {
    BackendError::Timeout {
      message: super::shared::error_chain(&error),
      context,
    }
  } else {
    BackendError::Transport {
      message: super::shared::error_chain(&error),
      context,
    }
  }
}

fn build_header_map(headers: &[(String, String)]) -> Result<HeaderMap, BackendError> {
  let mut header_map = HeaderMap::new();

  for (key, value) in headers {
    let header_name = HeaderName::from_bytes(key.as_bytes()).map_err(|error| BackendError::Transport {
      message: error.to_string(),
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::ClientBuild,
        kind: HttpFailureKind::InvalidHeader,
        received_bytes: None,
      }),
    })?;
    let header_value = HeaderValue::from_str(value).map_err(|error| BackendError::Transport {
      message: error.to_string(),
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::ClientBuild,
        kind: HttpFailureKind::InvalidHeader,
        received_bytes: None,
      }),
    })?;
    header_map.insert(header_name, header_value);
  }

  Ok(header_map)
}

#[cfg(test)]
mod tests {
  use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
  };

  use super::*;
  use crate::{backend::HttpBody, target::EgressPolicy};

  #[test]
  fn invalid_headers_are_rejected_before_sending() {
    for (name, value) in [("bad header", "value"), ("x-test", "bad\nvalue")] {
      let error = build_header_map(&[(name.to_string(), value.to_string())]).unwrap_err();
      assert_eq!(error.transmission(), crate::backend::RequestTransmission::NotSent);
      assert!(matches!(
        error,
        BackendError::Transport {
          context: Some(HttpFailureContext {
            stage: HttpFailureStage::ClientBuild,
            kind: HttpFailureKind::InvalidHeader,
            ..
          }),
          ..
        }
      ));
    }
  }

  #[test]
  fn connection_failure_and_disconnect_after_request_headers_have_distinct_transmission() {
    use crate::backend::RequestTransmission;
    for before_handshake in [true, false] {
      let listener = TcpListener::bind("127.0.0.1:0").unwrap();
      let address = listener.local_addr().unwrap();
      let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        if !before_handshake {
          let mut request = Vec::new();
          let mut byte = [0];
          while !request.ends_with(b"\r\n\r\n") {
            socket.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
          }
        }
      });
      let error = ReqwestHttpClient
        .post_json(HttpRequest {
          url: format!(
            "{}://{address}/v1/chat/completions",
            if before_handshake { "https" } else { "http" }
          ),
          headers: vec![],
          body: HttpBody::Json(serde_json::json!({})),
          timeout_ms: Some(5_000),
          egress_policy: EgressPolicy::AllowPrivate,
        })
        .unwrap_err();
      assert_eq!(
        error.transmission(),
        if before_handshake {
          RequestTransmission::NotSent
        } else {
          RequestTransmission::PossiblySent
        }
      );
      assert!(
        matches!(error, BackendError::Transport { context: Some(HttpFailureContext { stage, .. }), .. } if stage == if before_handshake { HttpFailureStage::Connect } else { HttpFailureStage::Send })
      );
      server.join().unwrap();
    }
  }

  #[test]
  fn delayed_sse_body_preserves_nested_timeout_stage() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
      let (mut socket, _) = listener.accept().unwrap();
      let mut request = Vec::new();
      let mut byte = [0];
      while !request.ends_with(b"\r\n\r\n") {
        socket.read_exact(&mut byte).unwrap();
        request.push(byte[0]);
      }
      socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\ndata: first\n\n").unwrap();
      socket.flush().unwrap();
      thread::sleep(Duration::from_millis(1500));
    });
    let error = ReqwestHttpClient
      .post_sse(
        HttpRequest {
          url: format!("http://{address}/secret-path"),
          headers: vec![],
          body: HttpBody::Json(serde_json::json!({})),
          timeout_ms: Some(1000),
          egress_policy: EgressPolicy::AllowPrivate,
        },
        &mut |_| Ok(()),
      )
      .unwrap_err();
    assert!(matches!(
      error,
      BackendError::Timeout {
        context: Some(HttpFailureContext {
          stage: HttpFailureStage::BodyRead,
          kind: HttpFailureKind::Timeout,
          ..
        }),
        ..
      }
    ));
    server.join().unwrap();
  }
}
