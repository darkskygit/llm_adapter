use std::io::{ErrorKind, Read};

use crate::backend::{BackendError, HttpBody, HttpFailureContext, HttpFailureKind, HttpFailureStage, MultipartPart};

const MULTIPART_BOUNDARY: &str = "llm-adapter-boundary";

pub(super) fn serialize_http_body(
  body: &HttpBody,
  headers: &mut Vec<(String, String)>,
) -> Result<Vec<u8>, BackendError> {
  match body {
    HttpBody::Json(value) => {
      ensure_header(headers, "content-type", "application/json");
      Ok(serde_json::to_vec(value)?)
    }
    HttpBody::Multipart(parts) => {
      set_header(
        headers,
        "content-type",
        &format!("multipart/form-data; boundary={MULTIPART_BOUNDARY}"),
      );
      encode_multipart_body(parts, MULTIPART_BOUNDARY)
    }
  }
}

fn ensure_header(headers: &mut Vec<(String, String)>, key: &str, value: &str) {
  if headers.iter().any(|(candidate, _)| candidate.eq_ignore_ascii_case(key)) {
    return;
  }
  headers.push((key.to_string(), value.to_string()));
}

fn set_header(headers: &mut Vec<(String, String)>, key: &str, value: &str) {
  if let Some((_, current)) = headers
    .iter_mut()
    .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
  {
    *current = value.to_string();
    return;
  }
  headers.push((key.to_string(), value.to_string()));
}

fn encode_multipart_body(parts: &[MultipartPart], boundary: &str) -> Result<Vec<u8>, BackendError> {
  let mut body = Vec::new();
  for part in parts {
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    match part {
      MultipartPart::Text { name, value } => {
        reject_multipart_boundary(value.as_bytes(), boundary)?;
        body.extend_from_slice(
          format!(
            "Content-Disposition: form-data; name=\"{}\"\r\n\r\n",
            escape_disposition_param(name, "multipart part name")?
          )
          .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
      }
      MultipartPart::File {
        name,
        file_name,
        media_type,
        bytes,
      } => {
        reject_multipart_boundary(bytes, boundary)?;
        validate_header_value(media_type, "multipart content type")?;
        body.extend_from_slice(
          format!(
            "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {media_type}\r\n\r\n",
            escape_disposition_param(name, "multipart part name")?,
            escape_disposition_param(file_name, "multipart file name")?
          )
          .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
      }
    }
  }
  body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
  Ok(body)
}

fn escape_disposition_param(value: &str, label: &'static str) -> Result<String, BackendError> {
  validate_header_value(value, label)?;
  let mut escaped = String::with_capacity(value.len());
  for character in value.chars() {
    match character {
      '"' => escaped.push_str("\\\""),
      '\\' => escaped.push_str("\\\\"),
      _ => escaped.push(character),
    }
  }
  Ok(escaped)
}

fn validate_header_value(value: &str, label: &'static str) -> Result<(), BackendError> {
  if value.bytes().any(|byte| byte.is_ascii_control()) {
    return Err(BackendError::Transport {
      message: format!("{label} contains control characters"),
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::ClientBuild,
        kind: HttpFailureKind::Request,
        received_bytes: None,
      }),
    });
  }
  Ok(())
}

fn reject_multipart_boundary(bytes: &[u8], boundary: &str) -> Result<(), BackendError> {
  let boundary = boundary.as_bytes();
  if bytes.windows(boundary.len()).any(|window| window == boundary) {
    return Err(BackendError::Transport {
      message: "multipart part body contains the boundary marker".to_string(),
      context: Some(HttpFailureContext {
        stage: HttpFailureStage::ClientBuild,
        kind: HttpFailureKind::Request,
        received_bytes: None,
      }),
    });
  }
  Ok(())
}

pub(super) fn stream_utf8_chunks(
  reader: &mut dyn Read,
  on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
  let mut buf = [0_u8; 8192];
  let mut pending = Vec::new();
  let mut received_bytes = 0u64;

  loop {
    let read = reader.read(&mut buf).map_err(|error| {
      let mut error = map_io_error(error);
      if let BackendError::Transport {
        context: Some(context), ..
      }
      | BackendError::Timeout {
        context: Some(context), ..
      } = &mut error
      {
        context.received_bytes = Some(received_bytes);
      }
      error
    })?;
    if read == 0 {
      break;
    }

    received_bytes += read as u64;
    pending.extend_from_slice(&buf[..read]);
    let mut consumed = 0usize;

    match std::str::from_utf8(&pending) {
      Ok(text) => {
        on_chunk(text)?;
        consumed = pending.len();
      }
      Err(error) => {
        let valid_up_to = error.valid_up_to();
        if valid_up_to > 0 {
          let valid_text = std::str::from_utf8(&pending[..valid_up_to]).map_err(|decode| BackendError::Transport {
            message: decode.to_string(),
            context: None,
          })?;
          on_chunk(valid_text)?;
          consumed = valid_up_to;
        }

        if error.error_len().is_some() {
          let fallback = String::from_utf8_lossy(&pending[consumed..]).to_string();
          on_chunk(&fallback)?;
          consumed = pending.len();
        }
      }
    }

    if consumed > 0 {
      pending.drain(..consumed);
    }
  }

  if !pending.is_empty() {
    let tail = String::from_utf8_lossy(&pending).to_string();
    on_chunk(&tail)?;
  }

  Ok(())
}

pub(super) fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
  let mut messages = vec![error.to_string()];
  let mut source = error.source();
  while let Some(cause) = source {
    messages.push(cause.to_string());
    source = cause.source();
  }
  messages.join(": ")
}

pub(super) fn map_io_error(error: std::io::Error) -> BackendError {
  let message = error_chain(&error);
  #[cfg(feature = "reqwest-client")]
  {
    let mut source: Option<&(dyn std::error::Error + 'static)> = error.get_ref().map(|inner| inner as _);
    while let Some(current) = source {
      if let Some(reqwest) = current.downcast_ref::<reqwest::Error>() {
        let kind = super::reqwest::reqwest_failure_kind(reqwest);
        let context = Some(HttpFailureContext {
          stage: HttpFailureStage::BodyRead,
          kind,
          received_bytes: None,
        });
        return if kind == HttpFailureKind::Timeout {
          BackendError::Timeout {
            message: message.clone(),
            context,
          }
        } else {
          BackendError::Transport {
            message: message.clone(),
            context,
          }
        };
      }
      source = current.source();
    }
  }
  let kind = match error.kind() {
    ErrorKind::TimedOut => HttpFailureKind::Timeout,
    ErrorKind::ConnectionReset => HttpFailureKind::ConnectionReset,
    ErrorKind::UnexpectedEof => HttpFailureKind::UnexpectedEof,
    ErrorKind::BrokenPipe => HttpFailureKind::BrokenPipe,
    _ => HttpFailureKind::Other,
  };
  let context = Some(HttpFailureContext {
    stage: HttpFailureStage::BodyRead,
    kind,
    received_bytes: None,
  });
  if kind == HttpFailureKind::Timeout {
    BackendError::Timeout {
      message: message.clone(),
      context,
    }
  } else {
    BackendError::Transport {
      message: message.clone(),
      context,
    }
  }
}

pub(super) fn map_egress_error(error: crate::target::TargetCompileError) -> BackendError {
  let kind = match &error {
    crate::target::TargetCompileError::UnresolvedEndpoint(_) => HttpFailureKind::Dns,
    crate::target::TargetCompileError::DisallowedAddress => HttpFailureKind::Egress,
    _ => HttpFailureKind::InvalidUrl,
  };
  BackendError::Transport {
    message: error_chain(&error),
    context: Some(HttpFailureContext {
      stage: HttpFailureStage::Resolve,
      kind,
      received_bytes: None,
    }),
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn should_escape_multipart_disposition_params() {
    let mut headers = Vec::new();
    let body = serialize_http_body(
      &HttpBody::Multipart(vec![MultipartPart::File {
        name: "image[]".to_string(),
        file_name: "a\"b\\c.png".to_string(),
        media_type: "image/png".to_string(),
        bytes: b"img".to_vec(),
      }]),
      &mut headers,
    )
    .unwrap();
    let body = String::from_utf8(body).unwrap();

    assert!(body.contains("filename=\"a\\\"b\\\\c.png\""));
    assert!(body.contains("Content-Type: image/png"));
  }

  #[test]
  fn should_map_timed_out_io_errors_to_timeout() {
    let error = map_io_error(std::io::Error::new(ErrorKind::TimedOut, "read timed out"));

    assert!(matches!(error, BackendError::Timeout { message, .. } if message == "read timed out"));
  }

  #[test]
  fn should_reject_multipart_header_injection() {
    let mut headers = Vec::new();
    let error = serialize_http_body(
      &HttpBody::Multipart(vec![MultipartPart::File {
        name: "image[]".to_string(),
        file_name: "in.png\r\nContent-Type: text/plain".to_string(),
        media_type: "image/png".to_string(),
        bytes: b"img".to_vec(),
      }]),
      &mut headers,
    )
    .unwrap_err();

    assert!(error.to_string().contains("control characters"));
  }

  #[test]
  fn should_reject_multipart_boundary_in_part_body() {
    let mut headers = Vec::new();
    let error = serialize_http_body(
      &HttpBody::Multipart(vec![MultipartPart::Text {
        name: "prompt".to_string(),
        value: format!("x\r\n--{MULTIPART_BOUNDARY}\r\nname=\"model\""),
      }]),
      &mut headers,
    )
    .unwrap_err();

    assert!(error.to_string().contains("boundary marker"));
  }

  #[test]
  fn should_serialize_json_body() {
    let mut headers = Vec::new();
    let body = serialize_http_body(&HttpBody::Json(json!({ "ok": true })), &mut headers).unwrap();

    assert_eq!(body, br#"{"ok":true}"#);
    assert_eq!(
      headers,
      vec![("content-type".to_string(), "application/json".to_string())]
    );
  }
  #[test]
  fn dns_cause_and_policy_rejection_have_distinct_unsent_evidence() {
    let dns = map_egress_error(crate::target::TargetCompileError::UnresolvedEndpoint(
      std::io::Error::new(ErrorKind::NotFound, "original resolver cause"),
    ));
    let policy = map_egress_error(crate::target::TargetCompileError::DisallowedAddress);
    for (error, kind) in [(dns, HttpFailureKind::Dns), (policy, HttpFailureKind::Egress)] {
      assert_eq!(error.transmission(), crate::backend::RequestTransmission::NotSent);
      assert!(
        matches!(error, BackendError::Transport { context: Some(HttpFailureContext { kind: found, .. }), .. } if found == kind)
      );
      if kind == HttpFailureKind::Dns {
        assert!(error.to_string().contains("original resolver cause"));
      }
    }
    for kind in [
      HttpFailureKind::Connect,
      HttpFailureKind::UnexpectedEof,
      HttpFailureKind::InvalidHeader,
    ] {
      let error = BackendError::Transport {
        message: "fixture".into(),
        context: Some(HttpFailureContext {
          stage: HttpFailureStage::Send,
          kind,
          received_bytes: Some(0),
        }),
      };
      assert_eq!(error.transmission(), crate::backend::RequestTransmission::PossiblySent);
    }
  }
}
