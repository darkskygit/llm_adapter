use serde_json::Value;

use super::{
  CoreAttachmentKind, CoreContent, CoreMessage, CoreRole, CoreUsage, ProtocolError, get_cached_tokens, get_u32_or,
  stringify_json,
};

pub(crate) fn parse_role(role: &str, field: &'static str) -> Result<CoreRole, ProtocolError> {
  match role {
    "system" => Ok(CoreRole::System),
    "user" => Ok(CoreRole::User),
    "assistant" => Ok(CoreRole::Assistant),
    "tool" => Ok(CoreRole::Tool),
    _ => Err(ProtocolError::InvalidValue {
      field,
      message: format!("unsupported role `{role}`"),
    }),
  }
}

pub(crate) fn parse_role_lossy(role: &str) -> CoreRole {
  match role {
    "system" => CoreRole::System,
    "user" => CoreRole::User,
    "tool" => CoreRole::Tool,
    _ => CoreRole::Assistant,
  }
}

pub(crate) fn core_role_to_string(role: &CoreRole) -> &'static str {
  match role {
    CoreRole::System => "system",
    CoreRole::User => "user",
    CoreRole::Assistant => "assistant",
    CoreRole::Tool => "tool",
  }
}

pub(crate) fn infer_media_type_from_url(url: &str) -> &'static str {
  let normalized = url.split('?').next().unwrap_or(url).to_ascii_lowercase();

  if normalized.ends_with(".png") {
    "image/png"
  } else if normalized.ends_with(".jpg") || normalized.ends_with(".jpeg") {
    "image/jpeg"
  } else if normalized.ends_with(".webp") {
    "image/webp"
  } else if normalized.ends_with(".gif") {
    "image/gif"
  } else if normalized.ends_with(".bmp") {
    "image/bmp"
  } else if normalized.ends_with(".svg") {
    "image/svg+xml"
  } else if normalized.ends_with(".mp3") {
    "audio/mpeg"
  } else if normalized.ends_with(".wav") {
    "audio/wav"
  } else if normalized.ends_with(".m4a") {
    "audio/mp4"
  } else if normalized.ends_with(".ogg") || normalized.ends_with(".oga") {
    "audio/ogg"
  } else if normalized.ends_with(".aac") {
    "audio/aac"
  } else if normalized.ends_with(".flac") {
    "audio/flac"
  } else if normalized.ends_with(".pdf") {
    "application/pdf"
  } else {
    "application/octet-stream"
  }
}

pub(crate) fn message_token_estimate(message: &CoreMessage) -> u32 {
  let mut text = String::new();
  for content in &message.content {
    match content {
      CoreContent::Text { text: value } => text.push_str(value),
      CoreContent::Reasoning { text: value, .. } => text.push_str(value),
      CoreContent::ToolCall { name, arguments, .. } => {
        text.push_str(name);
        text.push_str(&stringify_json(arguments));
      }
      CoreContent::ToolResult { output, .. } => text.push_str(&stringify_json(output)),
      CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => {}
    }
  }
  let chars = text.chars().count() as u32;
  (chars / 4).max(1)
}

fn source_media_type(source: &Value) -> Option<String> {
  if let Some(media_type) = source
    .as_object()
    .and_then(|object| {
      object
        .get("media_type")
        .or_else(|| object.get("mime_type"))
        .or_else(|| object.get("mimeType"))
    })
    .and_then(Value::as_str)
  {
    return Some(
      media_type
        .split(';')
        .next()
        .unwrap_or(media_type)
        .trim()
        .to_ascii_lowercase(),
    );
  }

  let url = match source {
    Value::String(url) => Some(url.as_str()),
    Value::Object(object) => object.get("url").and_then(Value::as_str),
    _ => None,
  }?;
  let data_url = url.strip_prefix("data:")?;
  let (meta, _) = data_url.split_once(',')?;
  Some(
    meta
      .split(';')
      .next()
      .unwrap_or("application/octet-stream")
      .trim()
      .to_ascii_lowercase(),
  )
}

fn kind_from_url(url: &str) -> Option<CoreAttachmentKind> {
  let normalized = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
  let media_type = infer_media_type_from_url(url);

  if media_type.starts_with("image/") {
    Some(CoreAttachmentKind::Image)
  } else if media_type.starts_with("audio/") {
    Some(CoreAttachmentKind::Audio)
  } else if normalized.contains('.') {
    Some(CoreAttachmentKind::File)
  } else {
    None
  }
}

fn attachment_kind_from_source(source: &Value, default_kind: CoreAttachmentKind) -> CoreAttachmentKind {
  if let Some(media_type) = source_media_type(source) {
    if media_type.starts_with("image/") {
      return CoreAttachmentKind::Image;
    }
    if media_type.starts_with("audio/") {
      return CoreAttachmentKind::Audio;
    }
    return CoreAttachmentKind::File;
  }

  let url = match source {
    Value::String(url) => Some(url.as_str()),
    Value::Object(object) => object.get("url").and_then(Value::as_str),
    _ => None,
  };
  url.and_then(kind_from_url).unwrap_or(default_kind)
}

pub(crate) fn attachment_content_from_source(source: Value, default_kind: CoreAttachmentKind) -> CoreContent {
  CoreContent::from_attachment(attachment_kind_from_source(&source, default_kind), source)
}

pub(crate) fn attachment_source(content: &CoreContent) -> Option<(&Value, CoreAttachmentKind)> {
  Some((content.attachment_source()?, content.attachment_kind()?))
}

pub(crate) fn usage_from_openai(usage: Option<&Value>, prompt_estimate: u32, completion_estimate: u32) -> CoreUsage {
  let usage = usage.unwrap_or(&Value::Null);
  let prompt_tokens = get_u32_or(usage, "prompt_tokens", prompt_estimate);
  let completion_tokens = get_u32_or(usage, "completion_tokens", completion_estimate);
  let total_tokens = get_u32_or(usage, "total_tokens", prompt_tokens + completion_tokens);
  let cached_tokens = get_cached_tokens(usage, &["prompt_tokens_details", "input_tokens_details"]);

  CoreUsage {
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cached_tokens,
  }
}

pub(crate) fn map_anthropic_finish_reason(reason: &str) -> String {
  match reason {
    "end_turn" | "stop_sequence" => "stop".to_string(),
    "max_tokens" => "length".to_string(),
    other => other.to_string(),
  }
}

pub(crate) fn map_responses_finish_reason(status: Option<&str>, finish_reason: Option<&str>) -> String {
  if let Some(reason) = finish_reason {
    return reason.to_string();
  }

  match status {
    Some("requires_action") => "tool_calls".to_string(),
    Some("incomplete") => "length".to_string(),
    _ => "stop".to_string(),
  }
}

pub(crate) fn usage_from_responses(usage: Option<&Value>, prompt_estimate: u32, completion_estimate: u32) -> CoreUsage {
  let usage = usage.unwrap_or(&Value::Null);
  let prompt_tokens = get_u32_or(usage, "input_tokens", prompt_estimate);
  let completion_tokens = get_u32_or(usage, "output_tokens", completion_estimate);
  let total_tokens = get_u32_or(usage, "total_tokens", prompt_tokens + completion_tokens);
  let cached_tokens = get_cached_tokens(usage, &["input_tokens_details"]);

  CoreUsage {
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cached_tokens,
  }
}

pub(crate) fn usage_from_anthropic(usage: Option<&Value>, prompt_estimate: u32, completion_estimate: u32) -> CoreUsage {
  let usage = usage.unwrap_or(&Value::Null);
  let input_tokens = get_u32_or(usage, "input_tokens", prompt_estimate);
  let output_tokens = get_u32_or(usage, "output_tokens", completion_estimate);
  let cache_read_input_tokens = get_u32_or(usage, "cache_read_input_tokens", 0);
  let cache_creation_input_tokens = get_u32_or(usage, "cache_creation_input_tokens", 0);
  let prompt_tokens = input_tokens
    .saturating_add(cache_read_input_tokens)
    .saturating_add(cache_creation_input_tokens);
  let completion_tokens = output_tokens;
  let total_tokens = prompt_tokens.saturating_add(completion_tokens);

  CoreUsage {
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cached_tokens: if cache_read_input_tokens > 0 {
      Some(cache_read_input_tokens)
    } else {
      None
    },
  }
}

pub(crate) fn map_gemini_finish_reason(reason: &str) -> String {
  match reason {
    "FINISH_REASON_MAX_TOKENS" => "length".to_string(),
    "FINISH_REASON_STOP" | "STOP" | "MAX_TOKENS" => {
      if reason == "MAX_TOKENS" {
        "length".to_string()
      } else {
        "stop".to_string()
      }
    }
    "FINISH_REASON_UNSPECIFIED" | "FINISH_REASON_OTHER" => "stop".to_string(),
    "FINISH_REASON_MALFORMED_FUNCTION_CALL" => "tool_calls".to_string(),
    other => other
      .strip_prefix("FINISH_REASON_")
      .unwrap_or(other)
      .to_ascii_lowercase(),
  }
}

pub(crate) fn usage_from_gemini(usage: Option<&Value>, prompt_estimate: u32, completion_estimate: u32) -> CoreUsage {
  let usage = usage.unwrap_or(&Value::Null);
  let prompt_tokens = get_u32_or(usage, "promptTokenCount", prompt_estimate);
  let completion_tokens = get_u32_or(usage, "candidatesTokenCount", completion_estimate);
  let total_tokens = get_u32_or(usage, "totalTokenCount", prompt_tokens + completion_tokens);
  let cached_tokens = get_u32_or(usage, "cachedContentTokenCount", 0);

  CoreUsage {
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cached_tokens: (cached_tokens > 0).then_some(cached_tokens),
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn should_infer_attachment_kind_from_media_type_and_url() {
    let pdf = attachment_content_from_source(
      json!({ "url": "https://example.com/manual.pdf" }),
      CoreAttachmentKind::Image,
    );
    let audio = attachment_content_from_source(
      json!({ "data": "Zm9v", "media_type": "audio/wav" }),
      CoreAttachmentKind::File,
    );

    assert!(matches!(pdf, CoreContent::File { .. }));
    assert!(matches!(audio, CoreContent::Audio { .. }));
  }
}
