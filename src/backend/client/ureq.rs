use std::{io::Read, time::Duration};

use ureq::{Agent, RequestBuilder};

use super::super::{BackendError, BackendHttpClient, HttpRequest, HttpResponse};
use super::shared::stream_utf8_chunks;

#[derive(Debug, Clone)]
pub struct UreqHttpClient {
  agent: Agent,
}

impl Default for UreqHttpClient {
  fn default() -> Self {
    Self {
      agent: Agent::new_with_defaults(),
    }
  }
}

impl BackendHttpClient for UreqHttpClient {
  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
    let body = serde_json::to_vec(&request.body)?;
    let mut response = self
      .build_request(&request)?
      .send(body.as_slice())
      .map_err(|error| BackendError::Http(error.to_string()))?;

    let status = response.status().as_u16();

    if !(200..300).contains(&status) {
      return Err(BackendError::UpstreamStatus {
        status,
        body: read_response_text(&mut response)?,
      });
    }

    let body = serde_json::from_reader(response.body_mut().as_reader())?;
    Ok(HttpResponse {
      status,
      body,
    })
  }

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError> {
    let body = serde_json::to_vec(&request.body)?;
    let mut response = self
      .build_request(&request)?
      .send(body.as_slice())
      .map_err(|error| BackendError::Http(error.to_string()))?;

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
}

impl UreqHttpClient {
  fn build_request(&self, request: &HttpRequest) -> Result<RequestBuilder<ureq::typestate::WithBody>, BackendError> {
    let mut request_builder = self.agent.post(&request.url);
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
}

fn read_response_text(response: &mut ureq::http::Response<ureq::Body>) -> Result<String, BackendError> {
  let mut bytes = Vec::new();
  response
    .body_mut()
    .as_reader()
    .read_to_end(&mut bytes)
    .map_err(|error| BackendError::Http(error.to_string()))?;
  Ok(String::from_utf8_lossy(&bytes).to_string())
}
