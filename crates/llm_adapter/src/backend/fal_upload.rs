use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde_json::{Value, json};

use super::{BackendConfig, BackendError, BackendHttpClient, HttpBody, HttpRequest, HttpUploadRequest};
use crate::core::{ImageInput, image_data_url_parts};

const FAL_BASE_URL: &str = "https://fal.run";
const FAL_UPLOAD_INIT_URL: &str = "https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3";

pub(super) fn upload_inline_image(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  image: Option<&ImageInput>,
) -> Result<Option<String>, BackendError> {
  let Some(image) = image else {
    return Ok(None);
  };
  let (bytes, media_type, file_name) = match image {
    ImageInput::Url { url, .. } if !url.starts_with("data:") => return Ok(None),
    ImageInput::Url { url, .. } => {
      let (media_type, data) = image_data_url_parts(url).ok_or_else(|| BackendError::InvalidRequest {
        field: "images",
        message: "invalid image data URL".to_string(),
      })?;
      (decode_base64(data)?, media_type.to_string(), None)
    }
    ImageInput::Data {
      data_base64,
      media_type,
      file_name,
    } => (decode_base64(data_base64)?, media_type.clone(), file_name.clone()),
    ImageInput::Bytes {
      data,
      media_type,
      file_name,
    } => (data.clone(), media_type.clone(), file_name.clone()),
  };

  if !is_official_fal_endpoint(&config.base_url) {
    return Err(BackendError::InvalidRequest {
      field: "images",
      message: "custom Fal endpoints require URL image inputs".to_string(),
    });
  }

  let response = client.post_json(HttpRequest {
    url: FAL_UPLOAD_INIT_URL.to_string(),
    headers: vec![(
      "Authorization".to_string(),
      format!("Key {}", config.auth_token.expose()),
    )],
    body: HttpBody::Json(json!({
      "content_type": media_type,
      "file_name": file_name.unwrap_or_else(|| "input-image".to_string()),
    })),
    timeout_ms: config.timeout_ms,
    egress_policy: config.egress_policy,
  })?;
  let upload_url =
    response
      .body
      .get("upload_url")
      .and_then(Value::as_str)
      .ok_or_else(|| BackendError::InvalidResponse {
        field: "upload_url",
        message: "Fal upload initiation did not return an upload URL".to_string(),
      })?;
  let file_url = response
    .body
    .get("file_url")
    .and_then(Value::as_str)
    .ok_or_else(|| BackendError::InvalidResponse {
      field: "file_url",
      message: "Fal upload initiation did not return a file URL".to_string(),
    })?
    .to_string();
  client.put_bytes(HttpUploadRequest {
    url: upload_url.to_string(),
    headers: vec![("Content-Type".to_string(), media_type)],
    bytes,
    timeout_ms: config.timeout_ms,
    egress_policy: config.egress_policy,
  })?;
  Ok(Some(file_url))
}

fn decode_base64(data: &str) -> Result<Vec<u8>, BackendError> {
  BASE64_STANDARD
    .decode(data)
    .map_err(|error| BackendError::InvalidRequest {
      field: "images",
      message: format!("invalid base64 image data: {error}"),
    })
}

fn is_official_fal_endpoint(base_url: &str) -> bool {
  base_url.trim_end_matches('/') == FAL_BASE_URL
}
