use std::io::Read;

use crate::backend::{BackendError, HttpBody, MultipartPart};

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
    });
  }
  Ok(())
}

fn reject_multipart_boundary(bytes: &[u8], boundary: &str) -> Result<(), BackendError> {
  let boundary = boundary.as_bytes();
  if bytes.windows(boundary.len()).any(|window| window == boundary) {
    return Err(BackendError::Transport {
      message: "multipart part body contains the boundary marker".to_string(),
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

  loop {
    let read = reader.read(&mut buf).map_err(|error| BackendError::Transport {
      message: error.to_string(),
    })?;
    if read == 0 {
      break;
    }

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
}
