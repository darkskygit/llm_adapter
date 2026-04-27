use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
  backend::BackendError,
  core::{CoreAttachmentKind, CoreContent, CoreMessage, CoreRole},
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptRole {
  System,
  User,
  Assistant,
}

impl From<&str> for PromptRole {
  fn from(role: &str) -> Self {
    match role {
      "system" => Self::System,
      "assistant" => Self::Assistant,
      _ => Self::User,
    }
  }
}

impl From<String> for PromptRole {
  fn from(role: String) -> Self {
    Self::from(role.as_str())
  }
}

impl From<PromptRole> for CoreRole {
  fn from(role: PromptRole) -> Self {
    match role {
      PromptRole::System => Self::System,
      PromptRole::User => Self::User,
      PromptRole::Assistant => Self::Assistant,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptAttachmentKind {
  Image,
  Audio,
  File,
}

impl From<&str> for PromptAttachmentKind {
  fn from(kind: &str) -> Self {
    match kind {
      "image" => Self::Image,
      "audio" => Self::Audio,
      _ => Self::File,
    }
  }
}

impl From<String> for PromptAttachmentKind {
  fn from(kind: String) -> Self {
    Self::from(kind.as_str())
  }
}

impl From<PromptAttachmentKind> for CoreAttachmentKind {
  fn from(kind: PromptAttachmentKind) -> Self {
    match kind {
      PromptAttachmentKind::Image => Self::Image,
      PromptAttachmentKind::Audio => Self::Audio,
      PromptAttachmentKind::File => Self::File,
    }
  }
}

impl From<&PromptAttachmentKind> for &'static str {
  fn from(kind: &PromptAttachmentKind) -> Self {
    match kind {
      PromptAttachmentKind::Image => "image",
      PromptAttachmentKind::Audio => "audio",
      PromptAttachmentKind::File => "file",
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptAttachmentSourceKind {
  Url,
  Data,
  Bytes,
  FileHandle,
}

impl From<&str> for PromptAttachmentSourceKind {
  fn from(kind: &str) -> Self {
    match kind {
      "url" => Self::Url,
      "data" => Self::Data,
      "bytes" => Self::Bytes,
      _ => Self::FileHandle,
    }
  }
}

impl From<String> for PromptAttachmentSourceKind {
  fn from(kind: String) -> Self {
    Self::from(kind.as_str())
  }
}

impl From<&PromptAttachmentSourceKind> for &'static str {
  fn from(kind: &PromptAttachmentSourceKind) -> Self {
    match kind {
      PromptAttachmentSourceKind::Url => "url",
      PromptAttachmentSourceKind::Data => "data",
      PromptAttachmentSourceKind::Bytes => "bytes",
      PromptAttachmentSourceKind::FileHandle => "file_handle",
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CanonicalPromptAttachment {
  pub kind: PromptAttachmentKind,
  pub source: Value,
  #[serde(default)]
  pub source_kind: Option<PromptAttachmentSourceKind>,
  #[serde(default)]
  pub is_remote: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptAttachmentProviderHint {
  #[serde(default)]
  pub provider: Option<String>,
  #[serde(default)]
  pub kind: Option<PromptAttachmentKind>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptAttachmentAliasInput {
  pub attachment: String,
  #[serde(default)]
  pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptAttachmentInputKind {
  Url,
  Data,
  Bytes,
  FileHandle,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptUrlAttachmentInput {
  pub kind: PromptAttachmentInputKind,
  pub url: String,
  #[serde(default)]
  pub data: Option<String>,
  #[serde(default)]
  pub encoding: Option<String>,
  #[serde(default)]
  pub mime_type: Option<String>,
  #[serde(default)]
  pub file_name: Option<String>,
  #[serde(default)]
  pub provider_hint: Option<PromptAttachmentProviderHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptDataAttachmentInput {
  pub kind: PromptAttachmentInputKind,
  pub data: String,
  pub mime_type: String,
  #[serde(default)]
  pub encoding: Option<String>,
  #[serde(default)]
  pub file_name: Option<String>,
  #[serde(default)]
  pub provider_hint: Option<PromptAttachmentProviderHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFileHandleAttachmentInput {
  pub kind: PromptAttachmentInputKind,
  pub file_handle: String,
  #[serde(default)]
  pub mime_type: Option<String>,
  #[serde(default)]
  pub file_name: Option<String>,
  #[serde(default)]
  pub provider_hint: Option<PromptAttachmentProviderHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PromptAttachmentInput {
  RawString(String),
  Alias(PromptAttachmentAliasInput),
  Url(PromptUrlAttachmentInput),
  Data(PromptDataAttachmentInput),
  FileHandle(PromptFileHandleAttachmentInput),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptMessageInput {
  pub role: PromptRole,
  pub content: String,
  #[serde(default)]
  pub attachments: Vec<PromptAttachmentInput>,
  #[serde(default)]
  pub response_format: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalPromptMessage {
  pub role: PromptRole,
  pub content: String,
  #[serde(default)]
  pub attachments: Vec<CanonicalPromptAttachment>,
  #[serde(default)]
  pub response_format: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AttachmentCapability {
  pub kinds: Vec<PromptAttachmentKind>,
  #[serde(default, alias = "sourceKinds")]
  pub source_kinds: Option<Vec<PromptAttachmentSourceKind>>,
  #[serde(default, alias = "allowRemoteUrls")]
  pub allow_remote_urls: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptModelConditions {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub input_types: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub attachment_kinds: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub attachment_source_kinds: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub has_remote_attachments: Option<bool>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub model_id: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output_type: Option<String>,
}

pub fn canonicalize_prompt_messages(messages: Vec<PromptMessageInput>) -> Vec<CanonicalPromptMessage> {
  messages
    .into_iter()
    .map(|message| CanonicalPromptMessage {
      role: message.role,
      content: message.content,
      attachments: message
        .attachments
        .into_iter()
        .map(canonicalize_prompt_attachment)
        .collect(),
      response_format: message.response_format,
    })
    .collect()
}

pub fn validate_attachment_capability(
  messages: &[CanonicalPromptMessage],
  attachment_capability: Option<&AttachmentCapability>,
) -> Result<(), BackendError> {
  let Some(attachment_capability) = attachment_capability else {
    return Ok(());
  };

  for message in messages {
    for attachment in &message.attachments {
      if !attachment_capability.kinds.contains(&attachment.kind) {
        let kind: &'static str = (&attachment.kind).into();
        return Err(BackendError::InvalidRequest {
          field: "attachments",
          message: format!("Native path does not support {kind} attachments"),
        });
      }

      if let (Some(source_kind), Some(source_kinds)) = (
        attachment.source_kind.as_ref(),
        attachment_capability.source_kinds.as_ref(),
      ) && !source_kinds.iter().any(|candidate| candidate == source_kind)
      {
        let source_kind_label: &'static str = source_kind.into();
        return Err(BackendError::InvalidRequest {
          field: "attachments",
          message: format!("Native path does not support {source_kind_label} attachment sources"),
        });
      }

      if attachment.is_remote == Some(true) && attachment_capability.allow_remote_urls == Some(false) {
        return Err(BackendError::InvalidRequest {
          field: "attachments",
          message: "Native path does not support remote attachment urls".to_string(),
        });
      }
    }
  }

  Ok(())
}

pub fn infer_model_conditions_from_prompt_messages(messages: Vec<CanonicalPromptMessage>) -> PromptModelConditions {
  let mut input_types = Vec::new();
  let mut attachment_kinds = Vec::new();
  let mut attachment_source_kinds = Vec::new();
  let mut has_remote_attachments = false;

  for message in messages {
    for attachment in message.attachments {
      let kind = String::from(<&'static str>::from(&attachment.kind));
      input_types.push(kind.clone());
      attachment_kinds.push(kind);

      if let Some(source_kind) = attachment.source_kind.as_ref() {
        attachment_source_kinds.push(String::from(<&'static str>::from(source_kind)));
      }

      has_remote_attachments |= attachment.is_remote == Some(true);
    }
  }

  PromptModelConditions {
    input_types: (!input_types.is_empty()).then(|| unique_strings(input_types)),
    attachment_kinds: (!attachment_kinds.is_empty()).then(|| unique_strings(attachment_kinds)),
    attachment_source_kinds: (!attachment_source_kinds.is_empty()).then(|| unique_strings(attachment_source_kinds)),
    has_remote_attachments: has_remote_attachments.then_some(true),
    model_id: None,
    output_type: None,
  }
}

pub fn materialize_core_messages(messages: Vec<CanonicalPromptMessage>) -> Vec<CoreMessage> {
  let mut iter = messages.into_iter();
  let first = iter.next();
  let mut core_messages = iter
    .filter(|message| !matches!(message.role, PromptRole::System))
    .map(message_to_core)
    .collect::<Vec<_>>();

  if let Some(message) = first {
    if matches!(message.role, PromptRole::System) && !message.content.is_empty() {
      core_messages.insert(
        0,
        CoreMessage {
          role: CoreRole::System,
          content: vec![CoreContent::Text { text: message.content }],
        },
      );
    } else if !matches!(message.role, PromptRole::System) {
      core_messages.insert(0, message_to_core(message));
    }
  }

  core_messages
}

pub fn parse_data_url(url: &str) -> Option<(String, String)> {
  if !url.starts_with("data:") {
    return None;
  }

  let comma_index = url.find(',')?;
  let meta = &url[5..comma_index];
  let payload = &url[comma_index + 1..];
  let parts = meta.split(';').collect::<Vec<_>>();
  let media_type = parts
    .first()
    .copied()
    .filter(|value| !value.is_empty())
    .unwrap_or("text/plain;charset=US-ASCII")
    .to_string();
  let is_base64 = parts.contains(&"base64");
  let data = if is_base64 {
    payload.to_string()
  } else {
    BASE64_STANDARD.encode(decode_percent_payload(payload))
  };

  Some((media_type, data))
}

fn canonicalize_prompt_attachment(attachment: PromptAttachmentInput) -> CanonicalPromptAttachment {
  match attachment {
    PromptAttachmentInput::RawString(url) => {
      if let Some((media_type, data)) = parse_data_url(&url) {
        return CanonicalPromptAttachment {
          kind: prompt_attachment_kind_from_media_type(Some(&media_type)),
          source: serde_json::json!({ "media_type": media_type, "data": data }),
          source_kind: Some(PromptAttachmentSourceKind::Data),
          is_remote: Some(false),
        };
      }

      let media_type = infer_media_type_from_attachment_path(&url, None);
      CanonicalPromptAttachment {
        kind: prompt_attachment_kind_from_media_type(Some(&media_type)),
        source: serde_json::json!({ "url": url.clone(), "media_type": media_type }),
        source_kind: Some(PromptAttachmentSourceKind::Url),
        is_remote: Some(is_remote_url(&url)),
      }
    }
    PromptAttachmentInput::Alias(alias) => {
      canonicalize_prompt_attachment(PromptAttachmentInput::Url(PromptUrlAttachmentInput {
        kind: PromptAttachmentInputKind::Url,
        url: alias.attachment,
        data: None,
        encoding: None,
        mime_type: alias.mime_type,
        file_name: None,
        provider_hint: None,
      }))
    }
    PromptAttachmentInput::Url(url_attachment) => {
      if let Some((data_media_type, data)) = parse_data_url(&url_attachment.url) {
        let media_type = url_attachment.mime_type.unwrap_or(data_media_type.clone());
        let mut source = serde_json::Map::new();
        source.insert("media_type".to_string(), Value::String(media_type.clone()));
        source.insert("data".to_string(), Value::String(data));
        append_attachment_metadata(
          &mut source,
          url_attachment.file_name,
          url_attachment.provider_hint.clone(),
        );

        return CanonicalPromptAttachment {
          kind: prompt_attachment_kind_from_hint_or_media_type(
            url_attachment
              .provider_hint
              .as_ref()
              .and_then(|hint| hint.kind.as_ref()),
            Some(&media_type),
          ),
          source: Value::Object(source),
          source_kind: Some(PromptAttachmentSourceKind::Data),
          is_remote: Some(false),
        };
      }

      let media_type = url_attachment.mime_type.unwrap_or_else(|| {
        infer_media_type_from_attachment_path(&url_attachment.url, url_attachment.file_name.as_deref())
      });
      let mut source = serde_json::Map::new();
      source.insert("url".to_string(), Value::String(url_attachment.url.clone()));
      source.insert("media_type".to_string(), Value::String(media_type.clone()));
      if let Some(data) = url_attachment.data {
        source.insert("data".to_string(), Value::String(data));
      }
      append_attachment_metadata(
        &mut source,
        url_attachment.file_name,
        url_attachment.provider_hint.clone(),
      );

      CanonicalPromptAttachment {
        kind: prompt_attachment_kind_from_hint_or_media_type(
          url_attachment
            .provider_hint
            .as_ref()
            .and_then(|hint| hint.kind.as_ref()),
          Some(&media_type),
        ),
        source: Value::Object(source),
        source_kind: Some(PromptAttachmentSourceKind::Url),
        is_remote: Some(is_remote_url(&url_attachment.url)),
      }
    }
    PromptAttachmentInput::Data(data_attachment) => {
      let source_kind = if data_attachment.kind == PromptAttachmentInputKind::Data {
        PromptAttachmentSourceKind::Data
      } else {
        PromptAttachmentSourceKind::Bytes
      };
      let data = if data_attachment.kind == PromptAttachmentInputKind::Data
        && data_attachment.encoding.as_deref() == Some("utf8")
      {
        BASE64_STANDARD.encode(data_attachment.data.as_bytes())
      } else {
        data_attachment.data.clone()
      };
      let mut source = serde_json::Map::new();
      source.insert(
        "media_type".to_string(),
        Value::String(data_attachment.mime_type.clone()),
      );
      source.insert("data".to_string(), Value::String(data));
      append_attachment_metadata(
        &mut source,
        data_attachment.file_name,
        data_attachment.provider_hint.clone(),
      );

      CanonicalPromptAttachment {
        kind: prompt_attachment_kind_from_hint_or_media_type(
          data_attachment
            .provider_hint
            .as_ref()
            .and_then(|hint| hint.kind.as_ref()),
          Some(&data_attachment.mime_type),
        ),
        source: Value::Object(source),
        source_kind: Some(source_kind),
        is_remote: Some(false),
      }
    }
    PromptAttachmentInput::FileHandle(file_handle_attachment) => {
      let mut source = serde_json::Map::new();
      source.insert(
        "file_handle".to_string(),
        Value::String(file_handle_attachment.file_handle.clone()),
      );
      if let Some(media_type) = &file_handle_attachment.mime_type {
        source.insert("media_type".to_string(), Value::String(media_type.clone()));
      }
      append_attachment_metadata(
        &mut source,
        file_handle_attachment.file_name,
        file_handle_attachment.provider_hint.clone(),
      );

      CanonicalPromptAttachment {
        kind: prompt_attachment_kind_from_hint_or_media_type(
          file_handle_attachment
            .provider_hint
            .as_ref()
            .and_then(|hint| hint.kind.as_ref()),
          file_handle_attachment.mime_type.as_deref(),
        ),
        source: Value::Object(source),
        source_kind: Some(PromptAttachmentSourceKind::FileHandle),
        is_remote: Some(false),
      }
    }
  }
}

fn message_to_core(message: CanonicalPromptMessage) -> CoreMessage {
  let mut content = Vec::new();

  if !message.content.is_empty() {
    content.push(CoreContent::Text { text: message.content });
  }

  content.extend(
    message
      .attachments
      .into_iter()
      .map(|attachment| CoreContent::from_attachment(attachment.kind.into(), attachment.source)),
  );

  CoreMessage {
    role: message.role.into(),
    content,
  }
}

fn decode_percent_payload(payload: &str) -> Vec<u8> {
  let bytes = payload.as_bytes();
  let mut decoded = Vec::with_capacity(bytes.len());
  let mut index = 0;

  while index < bytes.len() {
    if bytes[index] == b'%' && index + 2 < bytes.len() {
      let hi = (bytes[index + 1] as char).to_digit(16);
      let lo = (bytes[index + 2] as char).to_digit(16);
      if let (Some(hi), Some(lo)) = (hi, lo) {
        decoded.push(((hi << 4) + lo) as u8);
        index += 3;
        continue;
      }
    }

    decoded.push(bytes[index]);
    index += 1;
  }

  decoded
}

fn infer_media_type_from_path(path: &str) -> Option<&'static str> {
  let path = path.split(['?', '#']).next().unwrap_or(path);
  let extension = path.rsplit('.').next()?.to_ascii_lowercase();
  match extension.as_str() {
    "pdf" => Some("application/pdf"),
    "mp3" => Some("audio/mpeg"),
    "opus" => Some("audio/opus"),
    "ogg" => Some("audio/ogg"),
    "aac" | "m4a" => Some("audio/aac"),
    "flac" => Some("audio/flac"),
    "wav" => Some("audio/wav"),
    "png" => Some("image/png"),
    "jpeg" | "jpg" => Some("image/jpeg"),
    "webp" => Some("image/webp"),
    "txt" | "md" => Some("text/plain"),
    "mov" => Some("video/mov"),
    "mpeg" => Some("video/mpeg"),
    "mp4" => Some("video/mp4"),
    "avi" => Some("video/avi"),
    "wmv" => Some("video/wmv"),
    "flv" => Some("video/flv"),
    "ogv" => Some("video/ogg"),
    "webm" => Some("audio/webm"),
    _ => None,
  }
}

fn infer_media_type_from_attachment_path(url: &str, file_name: Option<&str>) -> String {
  infer_media_type_from_path(file_name.unwrap_or(url))
    .unwrap_or("application/octet-stream")
    .to_string()
}

fn prompt_attachment_kind_from_media_type(media_type: Option<&str>) -> PromptAttachmentKind {
  match media_type.unwrap_or_default() {
    value if value.starts_with("image/") => PromptAttachmentKind::Image,
    value if value.starts_with("audio/") => PromptAttachmentKind::Audio,
    _ => PromptAttachmentKind::File,
  }
}

fn prompt_attachment_kind_from_hint_or_media_type(
  hint: Option<&PromptAttachmentKind>,
  media_type: Option<&str>,
) -> PromptAttachmentKind {
  hint
    .cloned()
    .unwrap_or_else(|| prompt_attachment_kind_from_media_type(media_type))
}

fn is_remote_url(url: &str) -> bool {
  url.starts_with("http://") || url.starts_with("https://")
}

fn append_attachment_metadata(
  source: &mut serde_json::Map<String, Value>,
  file_name: Option<String>,
  provider_hint: Option<PromptAttachmentProviderHint>,
) {
  if let Some(file_name) = file_name {
    source.insert("file_name".to_string(), Value::String(file_name));
  }

  if let Some(provider_hint) = provider_hint {
    let mut value = serde_json::Map::new();
    if let Some(provider) = provider_hint.provider {
      value.insert("provider".to_string(), Value::String(provider));
    }
    if let Some(kind) = provider_hint.kind {
      value.insert(
        "kind".to_string(),
        Value::String(String::from(<&'static str>::from(&kind))),
      );
    }
    if !value.is_empty() {
      source.insert("provider_hint".to_string(), Value::Object(value));
    }
  }
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
  let mut unique = Vec::new();
  for value in values {
    if !unique.contains(&value) {
      unique.push(value);
    }
  }
  unique
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn canonicalizes_data_url_and_percent_payload() {
    let messages = canonicalize_prompt_messages(vec![PromptMessageInput {
      role: PromptRole::User,
      content: "see attached".to_string(),
      attachments: vec![PromptAttachmentInput::RawString(
        "data:text/plain,hello%20world".to_string(),
      )],
      response_format: None,
    }]);

    assert_eq!(messages[0].attachments[0].source["media_type"], "text/plain");
    assert_eq!(messages[0].attachments[0].source["data"], "aGVsbG8gd29ybGQ=");
    assert_eq!(
      messages[0].attachments[0].source_kind,
      Some(PromptAttachmentSourceKind::Data)
    );
  }

  #[test]
  fn preserves_remote_url_file_handle_and_provider_hint() {
    let messages = canonicalize_prompt_messages(vec![PromptMessageInput {
      role: PromptRole::User,
      content: "attachments".to_string(),
      attachments: vec![
        serde_json::from_value(json!({
          "kind": "url",
          "url": "https://example.com/audio.mp3",
          "providerHint": { "provider": "gemini", "kind": "audio" }
        }))
        .unwrap(),
        serde_json::from_value(json!({
          "kind": "file_handle",
          "fileHandle": "file_1",
          "mimeType": "application/pdf"
        }))
        .unwrap(),
      ],
      response_format: None,
    }]);

    assert_eq!(messages[0].attachments[0].kind, PromptAttachmentKind::Audio);
    assert_eq!(messages[0].attachments[0].is_remote, Some(true));
    assert_eq!(messages[0].attachments[0].source["provider_hint"]["provider"], "gemini");
    assert_eq!(
      messages[0].attachments[1].source_kind,
      Some(PromptAttachmentSourceKind::FileHandle)
    );
  }

  #[test]
  fn validates_attachment_capability_and_infers_conditions() {
    let messages = canonicalize_prompt_messages(vec![PromptMessageInput {
      role: PromptRole::User,
      content: "image".to_string(),
      attachments: vec![
        serde_json::from_value(json!({
          "kind": "url",
          "url": "https://example.com/image.png"
        }))
        .unwrap(),
      ],
      response_format: None,
    }]);

    let error = validate_attachment_capability(
      &messages,
      Some(&AttachmentCapability {
        kinds: vec![PromptAttachmentKind::Image],
        source_kinds: Some(vec![PromptAttachmentSourceKind::Url]),
        allow_remote_urls: Some(false),
      }),
    )
    .unwrap_err();
    assert!(error.to_string().contains("remote attachment urls"));

    let conditions = infer_model_conditions_from_prompt_messages(messages);
    assert_eq!(conditions.input_types, Some(vec!["image".to_string()]));
    assert_eq!(conditions.attachment_source_kinds, Some(vec!["url".to_string()]));
    assert_eq!(conditions.has_remote_attachments, Some(true));
  }
}
