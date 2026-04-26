use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::ProtocolError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreRole {
  System,
  User,
  Assistant,
  Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoreContent {
  Text {
    text: String,
  },
  Reasoning {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
  },
  ToolCall {
    call_id: String,
    name: String,
    arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<String>,
  },
  ToolResult {
    call_id: String,
    output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
  },
  Image {
    source: Value,
  },
  Audio {
    source: Value,
  },
  File {
    source: Value,
  },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAttachmentKind {
  Image,
  Audio,
  File,
}

impl CoreContent {
  #[must_use]
  pub fn from_attachment(kind: CoreAttachmentKind, source: Value) -> Self {
    match kind {
      CoreAttachmentKind::Image => Self::Image { source },
      CoreAttachmentKind::Audio => Self::Audio { source },
      CoreAttachmentKind::File => Self::File { source },
    }
  }

  #[must_use]
  pub fn attachment_kind(&self) -> Option<CoreAttachmentKind> {
    match self {
      Self::Image { .. } => Some(CoreAttachmentKind::Image),
      Self::Audio { .. } => Some(CoreAttachmentKind::Audio),
      Self::File { .. } => Some(CoreAttachmentKind::File),
      _ => None,
    }
  }

  #[must_use]
  pub fn attachment_source(&self) -> Option<&Value> {
    match self {
      Self::Image { source } | Self::Audio { source } | Self::File { source } => Some(source),
      _ => None,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreToolDefinition {
  pub name: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<String>,
  pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreToolChoiceMode {
  Auto,
  None,
  Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CoreToolChoice {
  Mode(CoreToolChoiceMode),
  Specific { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreMessage {
  pub role: CoreRole,
  pub content: Vec<CoreContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreRequest {
  pub model: String,
  #[serde(default)]
  pub messages: Vec<CoreMessage>,
  #[serde(default)]
  pub stream: bool,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub temperature: Option<f64>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub tools: Vec<CoreToolDefinition>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tool_choice: Option<CoreToolChoice>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub include: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning: Option<Value>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub response_schema: Option<Value>,
}

impl CoreRequest {
  pub fn validate(&self) -> Result<(), ProtocolError> {
    validate_messages(&self.messages)?;
    Ok(())
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CoreUsage {
  pub prompt_tokens: u32,
  pub completion_tokens: u32,
  pub total_tokens: u32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreResponse {
  pub id: String,
  pub model: String,
  pub message: CoreMessage,
  pub usage: CoreUsage,
  pub finish_reason: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning_details: Option<Value>,
}

pub(super) fn validate_messages(messages: &[CoreMessage]) -> Result<(), ProtocolError> {
  for message in messages {
    for content in &message.content {
      if let (Some(kind), Some(source)) = (content.attachment_kind(), content.attachment_source()) {
        validate_attachment_source(kind, source)?;
      }
    }
  }

  Ok(())
}

fn validate_attachment_source(kind: CoreAttachmentKind, source: &Value) -> Result<(), ProtocolError> {
  match source {
    Value::String(_) => Ok(()),
    Value::Object(object) => {
      if object.contains_key("url") && !matches!(object.get("url"), Some(Value::String(_))) {
        return Err(invalid_attachment_source(kind, "`url` must be a string"));
      }
      if object.contains_key("data") && !matches!(object.get("data"), Some(Value::String(_))) {
        return Err(invalid_attachment_source(kind, "`data` must be a string"));
      }
      if object.contains_key("bytes") && !matches!(object.get("bytes"), Some(Value::String(_) | Value::Array(_))) {
        return Err(invalid_attachment_source(
          kind,
          "`bytes` must be a base64 string or array",
        ));
      }
      if object.contains_key("file_handle")
        && !matches!(object.get("file_handle"), Some(Value::String(_) | Value::Object(_)))
      {
        return Err(invalid_attachment_source(
          kind,
          "`file_handle` must be a string or object",
        ));
      }
      if object.contains_key("file_id") && !matches!(object.get("file_id"), Some(Value::String(_))) {
        return Err(invalid_attachment_source(kind, "`file_id` must be a string"));
      }

      let has_known_source = ["url", "data", "bytes", "file_handle", "file_id"]
        .iter()
        .any(|key| object.contains_key(*key));
      if !has_known_source {
        return Err(invalid_attachment_source(
          kind,
          "expected one of `url`, `data`, `bytes`, `file_handle`, or `file_id`",
        ));
      }

      let media_type = object
        .get("media_type")
        .or_else(|| object.get("mime_type"))
        .or_else(|| object.get("mimeType"));
      let media_type = match media_type {
        Some(Value::String(media_type)) => Some(media_type.as_str()),
        Some(_) => return Err(invalid_attachment_source(kind, "`media_type` must be a string")),
        None => None,
      };

      if matches!(kind, CoreAttachmentKind::Image | CoreAttachmentKind::Audio)
        && media_type.is_none()
        && (object.contains_key("data") || object.contains_key("bytes"))
      {
        return Err(invalid_attachment_source(
          kind,
          "`media_type` is required for inline data or bytes sources",
        ));
      }

      match (kind, media_type) {
        (CoreAttachmentKind::Image, Some(media_type)) if !media_type.starts_with("image/") => {
          Err(invalid_attachment_source(kind, "`media_type` must start with `image/`"))
        }
        (CoreAttachmentKind::Audio, Some(media_type)) if !media_type.starts_with("audio/") => {
          Err(invalid_attachment_source(kind, "`media_type` must start with `audio/`"))
        }
        _ => Ok(()),
      }
    }
    _ => Err(invalid_attachment_source(kind, "expected string or object source")),
  }
}

fn invalid_attachment_source(kind: CoreAttachmentKind, message: &str) -> ProtocolError {
  let kind = match kind {
    CoreAttachmentKind::Image => "image",
    CoreAttachmentKind::Audio => "audio",
    CoreAttachmentKind::File => "file",
  };
  ProtocolError::InvalidRequest {
    field: "messages.content.source",
    message: format!("{kind} source {message}"),
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn should_round_trip_core_request() {
    let request = CoreRequest {
      model: "gpt-4.1".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text {
          text: "hello".to_string(),
        }],
      }],
      stream: true,
      max_tokens: Some(1024),
      temperature: Some(0.5),
      tools: vec![CoreToolDefinition {
        name: "doc_read".to_string(),
        description: Some("Read a document".to_string()),
        parameters: json!({
          "type": "object",
          "properties": {
            "docId": { "type": "string" }
          }
        }),
      }],
      tool_choice: Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
      response_schema: None,
    };

    let serialized = serde_json::to_value(&request).unwrap();
    let parsed: CoreRequest = serde_json::from_value(serialized).unwrap();
    assert_eq!(parsed, request);
  }

  #[test]
  fn should_support_tool_choice_string_and_object_forms() {
    let as_string = json!("required");
    let parsed_string: CoreToolChoice = serde_json::from_value(as_string).unwrap();
    assert_eq!(parsed_string, CoreToolChoice::Mode(CoreToolChoiceMode::Required));

    let as_object = json!({ "name": "doc_update" });
    let parsed_object: CoreToolChoice = serde_json::from_value(as_object).unwrap();
    assert_eq!(
      parsed_object,
      CoreToolChoice::Specific {
        name: "doc_update".to_string(),
      }
    );
  }

  #[test]
  fn should_round_trip_attachment_variants() {
    let content = vec![
      CoreContent::Image {
        source: json!({ "url": "https://example.com/a.png", "media_type": "image/png" }),
      },
      CoreContent::Audio {
        source: json!({ "data": "Zm9v", "media_type": "audio/wav" }),
      },
      CoreContent::File {
        source: json!({ "url": "https://example.com/a.pdf", "media_type": "application/pdf" }),
      },
    ];

    let serialized = serde_json::to_value(&content).unwrap();
    let parsed: Vec<CoreContent> = serde_json::from_value(serialized).unwrap();

    assert_eq!(parsed, content);
    assert_eq!(content[0].attachment_kind(), Some(CoreAttachmentKind::Image));
    assert_eq!(content[1].attachment_kind(), Some(CoreAttachmentKind::Audio));
    assert_eq!(content[2].attachment_kind(), Some(CoreAttachmentKind::File));
  }

  #[test]
  fn should_validate_attachment_sources() {
    let request = CoreRequest {
      model: "gemini-2.5-flash".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![
          CoreContent::Audio {
            source: json!({ "data": "Zm9v", "media_type": "audio/wav" }),
          },
          CoreContent::File {
            source: json!({ "file_handle": "file_1" }),
          },
        ],
      }],
      stream: false,
      max_tokens: None,
      temperature: None,
      tools: Vec::new(),
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    assert!(request.validate().is_ok());

    let invalid = CoreRequest {
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Audio {
          source: json!({ "url": "https://example.com/a.mp3", "media_type": "application/pdf" }),
        }],
      }],
      ..request
    };

    assert!(matches!(
      invalid.validate(),
      Err(ProtocolError::InvalidRequest {
        field: "messages.content.source",
        ..
      })
    ));
  }
}
