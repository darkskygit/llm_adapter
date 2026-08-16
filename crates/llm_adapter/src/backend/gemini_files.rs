use std::{thread, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{BackendConfig, BackendError, BackendHttpClient, HttpMethod, HttpRawRequest, HttpRawResponse};
use crate::{core::CoreMessage, target::EgressPolicy};

const GEMINI_API_ORIGIN: &str = "https://generativelanguage.googleapis.com";
const MAX_ATTACHMENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;
const PROCESSING_POLL_LIMIT: usize = 120;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFile {
  name: String,
  uri: String,
  mime_type: String,
  state: Option<String>,
  error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct UploadResponse {
  file: GeminiFile,
}

pub(super) struct UploadedGeminiFiles<'a> {
  client: &'a dyn BackendHttpClient,
  config: &'a BackendConfig,
  names: Vec<String>,
}

impl UploadedGeminiFiles<'_> {
  pub(super) fn cleanup(mut self) {
    while let Some(name) = self.names.pop() {
      let _ = delete_file(self.client, self.config, &name);
    }
  }
}

pub(super) fn is_official_gemini_api(config: &BackendConfig) -> bool {
  config
    .base_url
    .parse::<url::Url>()
    .ok()
    .is_some_and(|url| url.scheme() == "https" && url.host_str() == Some("generativelanguage.googleapis.com"))
}

pub(super) fn prepare_messages<'a>(
  client: &'a dyn BackendHttpClient,
  config: &'a BackendConfig,
  messages: &mut [CoreMessage],
  upload_inline: bool,
) -> Result<UploadedGeminiFiles<'a>, BackendError> {
  let mut uploaded = UploadedGeminiFiles {
    client,
    config,
    names: Vec::new(),
  };
  let result = prepare_message_attachments(client, config, messages, upload_inline, &mut uploaded.names);
  if let Err(error) = result {
    uploaded.cleanup();
    return Err(error);
  }
  Ok(uploaded)
}

fn prepare_message_attachments(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  messages: &mut [CoreMessage],
  upload_inline: bool,
  uploaded_names: &mut Vec<String>,
) -> Result<(), BackendError> {
  let mut total_bytes = 0usize;
  for content in messages.iter_mut().flat_map(|message| &mut message.content) {
    let Some(source) = content.attachment_source_mut() else {
      continue;
    };
    let Some((bytes, media_type, display_name)) = materialize_source(client, config, source, upload_inline)? else {
      continue;
    };
    total_bytes = total_bytes.saturating_add(bytes.len());
    if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES {
      return Err(BackendError::InvalidRequest {
        field: "attachments",
        message: "Gemini attachments exceed the aggregate upload limit".to_string(),
      });
    }
    let file = upload_file(client, config, bytes, &media_type, display_name.as_deref())?;
    uploaded_names.push(file.name.clone());
    *source = json!({
      "url": file.uri,
      "media_type": file.mime_type,
    });
  }
  Ok(())
}

fn materialize_source(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  source: &Value,
  upload_inline: bool,
) -> Result<Option<(Vec<u8>, String, Option<String>)>, BackendError> {
  let object = match source {
    Value::String(url) => return materialize_url(client, config, url, None, None),
    Value::Object(object) => object,
    _ => return Ok(None),
  };
  let media_type = object
    .get("media_type")
    .or_else(|| object.get("mimeType"))
    .and_then(Value::as_str)
    .map(ToString::to_string);
  let display_name = object
    .get("file_name")
    .or_else(|| object.get("fileName"))
    .and_then(Value::as_str)
    .map(ToString::to_string);
  if let Some(url) = object.get("url").and_then(Value::as_str) {
    return materialize_url(client, config, url, media_type, display_name);
  }
  if !upload_inline {
    return Ok(None);
  }
  let Some(bytes) = inline_bytes(object)? else {
    return Ok(None);
  };
  validate_attachment_size(bytes.len())?;
  let media_type = media_type.ok_or_else(|| BackendError::InvalidRequest {
    field: "attachments",
    message: "inline Gemini attachment requires media_type".to_string(),
  })?;
  Ok(Some((bytes, media_type, display_name)))
}

fn inline_bytes(object: &serde_json::Map<String, Value>) -> Result<Option<Vec<u8>>, BackendError> {
  if let Some(encoded) = object.get("data").and_then(Value::as_str) {
    return decode_base64_attachment(encoded).map(Some);
  }
  match object.get("bytes") {
    Some(Value::String(encoded)) => decode_base64_attachment(encoded).map(Some),
    Some(Value::Array(values)) => values
      .iter()
      .map(|value| value.as_u64().and_then(|value| u8::try_from(value).ok()))
      .collect::<Option<Vec<_>>>()
      .ok_or_else(|| BackendError::InvalidRequest {
        field: "attachments",
        message: "attachment bytes must contain integers from 0 through 255".to_string(),
      })
      .map(Some),
    _ => Ok(None),
  }
}

fn decode_base64_attachment(encoded: &str) -> Result<Vec<u8>, BackendError> {
  BASE64_STANDARD
    .decode(encoded)
    .map_err(|error| BackendError::InvalidRequest {
      field: "attachments",
      message: format!("invalid base64 attachment: {error}"),
    })
}

fn materialize_url(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  url: &str,
  media_type: Option<String>,
  display_name: Option<String>,
) -> Result<Option<(Vec<u8>, String, Option<String>)>, BackendError> {
  if is_gemini_file_uri(url) || is_youtube_url(url) {
    return Ok(None);
  }
  if !url
    .parse::<url::Url>()
    .ok()
    .is_some_and(|url| matches!(url.scheme(), "http" | "https"))
  {
    return Ok(None);
  }
  let response = client.execute(HttpRawRequest {
    method: HttpMethod::Get,
    url: url.to_string(),
    headers: Vec::new(),
    body: Vec::new(),
    timeout_ms: config.timeout_ms,
    egress_policy: EgressPolicy::PublicOnly,
    max_response_bytes: Some(MAX_ATTACHMENT_BYTES),
  })?;
  validate_attachment_size(response.body.len())?;
  let response_media_type = response_header(&response, "content-type")
    .and_then(|value| value.split(';').next())
    .map(str::trim)
    .filter(|value| !value.is_empty())
    .map(ToString::to_string);
  let media_type = media_type
    .or(response_media_type)
    .unwrap_or_else(|| infer_media_type_from_url(url).to_string());
  Ok(Some((response.body, media_type, display_name)))
}

fn upload_file(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  bytes: Vec<u8>,
  media_type: &str,
  display_name: Option<&str>,
) -> Result<GeminiFile, BackendError> {
  let metadata = json!({
    "file": {
      "display_name": display_name.unwrap_or("affine-attachment"),
    }
  });
  let start = client.execute(HttpRawRequest {
    method: HttpMethod::Post,
    url: format!("{GEMINI_API_ORIGIN}/upload/v1beta/files"),
    headers: vec![
      ("x-goog-api-key".to_string(), config.auth_token.expose().to_string()),
      ("x-goog-upload-protocol".to_string(), "resumable".to_string()),
      ("x-goog-upload-command".to_string(), "start".to_string()),
      (
        "x-goog-upload-header-content-length".to_string(),
        bytes.len().to_string(),
      ),
      ("x-goog-upload-header-content-type".to_string(), media_type.to_string()),
      ("content-type".to_string(), "application/json".to_string()),
    ],
    body: serde_json::to_vec(&metadata)?,
    timeout_ms: config.timeout_ms,
    egress_policy: EgressPolicy::PublicOnly,
    max_response_bytes: Some(MAX_CONTROL_RESPONSE_BYTES),
  })?;
  let upload_url = response_header(&start, "x-goog-upload-url")
    .ok_or_else(|| BackendError::InvalidResponse {
      field: "x-goog-upload-url",
      message: "Gemini file upload did not return an upload URL".to_string(),
    })?
    .to_string();
  validate_upload_url(&upload_url)?;
  let finalize = client.execute(HttpRawRequest {
    method: HttpMethod::Post,
    url: upload_url,
    headers: vec![
      ("content-length".to_string(), bytes.len().to_string()),
      ("x-goog-upload-offset".to_string(), "0".to_string()),
      ("x-goog-upload-command".to_string(), "upload, finalize".to_string()),
      ("content-type".to_string(), media_type.to_string()),
    ],
    body: bytes,
    timeout_ms: config.timeout_ms,
    egress_policy: EgressPolicy::PublicOnly,
    max_response_bytes: Some(MAX_CONTROL_RESPONSE_BYTES),
  })?;
  let file = serde_json::from_slice::<UploadResponse>(&finalize.body)?.file;
  let name = file.name.clone();
  match wait_until_active(client, config, file) {
    Ok(file) => Ok(file),
    Err(error) => {
      let _ = delete_file(client, config, &name);
      Err(error)
    }
  }
}

fn wait_until_active(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  mut file: GeminiFile,
) -> Result<GeminiFile, BackendError> {
  for _ in 0..PROCESSING_POLL_LIMIT {
    match file.state.as_deref() {
      None | Some("ACTIVE") => return Ok(file),
      Some("FAILED") => {
        return Err(BackendError::InvalidResponse {
          field: "file.state",
          message: format!("Gemini file processing failed: {}", file.error.unwrap_or(Value::Null)),
        });
      }
      Some("PROCESSING") => {}
      Some(state) => {
        return Err(BackendError::InvalidResponse {
          field: "file.state",
          message: format!("unknown Gemini file state `{state}`"),
        });
      }
    }
    thread::sleep(Duration::from_secs(1));
    file = get_file(client, config, &file.name)?;
  }
  Err(BackendError::Timeout {
    message: "Gemini file did not become active before the processing deadline".to_string(),
  })
}

fn get_file(client: &dyn BackendHttpClient, config: &BackendConfig, name: &str) -> Result<GeminiFile, BackendError> {
  validate_file_name(name)?;
  let response = client.execute(HttpRawRequest {
    method: HttpMethod::Get,
    url: format!("{GEMINI_API_ORIGIN}/v1beta/{name}"),
    headers: vec![("x-goog-api-key".to_string(), config.auth_token.expose().to_string())],
    body: Vec::new(),
    timeout_ms: config.timeout_ms,
    egress_policy: EgressPolicy::PublicOnly,
    max_response_bytes: Some(MAX_CONTROL_RESPONSE_BYTES),
  })?;
  serde_json::from_slice(&response.body).map_err(Into::into)
}

fn delete_file(client: &dyn BackendHttpClient, config: &BackendConfig, name: &str) -> Result<(), BackendError> {
  validate_file_name(name)?;
  client.execute(HttpRawRequest {
    method: HttpMethod::Delete,
    url: format!("{GEMINI_API_ORIGIN}/v1beta/{name}"),
    headers: vec![("x-goog-api-key".to_string(), config.auth_token.expose().to_string())],
    body: Vec::new(),
    timeout_ms: config.timeout_ms,
    egress_policy: EgressPolicy::PublicOnly,
    max_response_bytes: Some(MAX_CONTROL_RESPONSE_BYTES),
  })?;
  Ok(())
}

fn validate_attachment_size(size: usize) -> Result<(), BackendError> {
  if size > MAX_ATTACHMENT_BYTES {
    Err(BackendError::InvalidRequest {
      field: "attachments",
      message: "Gemini attachment exceeds the per-file upload limit".to_string(),
    })
  } else {
    Ok(())
  }
}

fn validate_upload_url(upload_url: &str) -> Result<(), BackendError> {
  let url = upload_url
    .parse::<url::Url>()
    .map_err(|error| BackendError::InvalidResponse {
      field: "x-goog-upload-url",
      message: error.to_string(),
    })?;
  if url.scheme() != "https" || !url.host_str().is_some_and(|host| host.ends_with(".googleapis.com")) {
    return Err(BackendError::InvalidResponse {
      field: "x-goog-upload-url",
      message: "Gemini returned an untrusted upload URL".to_string(),
    });
  }
  Ok(())
}

fn validate_file_name(name: &str) -> Result<(), BackendError> {
  if name
    .strip_prefix("files/")
    .is_some_and(|id| !id.is_empty() && id.chars().all(|char| char.is_ascii_alphanumeric() || char == '-'))
  {
    Ok(())
  } else {
    Err(BackendError::InvalidResponse {
      field: "file.name",
      message: "Gemini returned an invalid file name".to_string(),
    })
  }
}

fn response_header<'a>(response: &'a HttpRawResponse, name: &str) -> Option<&'a str> {
  response
    .headers
    .iter()
    .find(|(header, _)| header.eq_ignore_ascii_case(name))
    .map(|(_, value)| value.as_str())
}

fn is_gemini_file_uri(value: &str) -> bool {
  value.parse::<url::Url>().ok().is_some_and(|url| {
    url.host_str() == Some("generativelanguage.googleapis.com") && url.path().starts_with("/v1beta/files/")
  })
}

fn is_youtube_url(value: &str) -> bool {
  let Ok(url) = value.parse::<url::Url>() else {
    return false;
  };
  let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
  if host == "youtu.be" {
    return !url.path().trim_matches('/').is_empty();
  }
  matches!(host.as_str(), "youtube.com" | "www.youtube.com")
    && url.path() == "/watch"
    && url.query_pairs().any(|(key, value)| key == "v" && !value.is_empty())
}

fn infer_media_type_from_url(url: &str) -> &'static str {
  let path = url.split(['?', '#']).next().unwrap_or(url).to_ascii_lowercase();
  match path.rsplit('.').next() {
    Some("m4a") | Some("mp4") => "audio/mp4",
    Some("mp3") => "audio/mpeg",
    Some("wav") => "audio/wav",
    Some("ogg") | Some("oga") => "audio/ogg",
    Some("webm") => "audio/webm",
    Some("png") => "image/png",
    Some("jpg") | Some("jpeg") => "image/jpeg",
    Some("webp") => "image/webp",
    Some("pdf") => "application/pdf",
    _ => "application/octet-stream",
  }
}
