use base64::{Engine as _, engine::general_purpose};
use serde_json::{Map, Value, json};

use super::{
  CoreAttachmentKind, CoreContent, CoreUsage, ProtocolError, attachment_content_from_source, attachment_source,
  get_first_str, get_str, get_str_or, parse_json_ref, stringify_json,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnthropicContentParseMode {
  Request,
  Response,
}

pub(crate) fn parse_content_blocks(
  value: Value,
  mode: AnthropicContentParseMode,
) -> Result<Vec<CoreContent>, ProtocolError> {
  match value {
    Value::String(text) => Ok(vec![CoreContent::Text { text }]),
    Value::Array(items) => {
      let mut content = Vec::new();
      for item in items {
        if let Some(block) = parse_content_block(&item, mode)? {
          content.push(block);
        }
      }
      Ok(content)
    }
    _ => Err(ProtocolError::InvalidRequest {
      field: "content",
      message: "expected string or array".to_string(),
    }),
  }
}

fn block_source(item: &Value, mode: AnthropicContentParseMode) -> Value {
  match mode {
    AnthropicContentParseMode::Request => item.get("source").cloned().unwrap_or_else(|| item.clone()),
    AnthropicContentParseMode::Response => item.get("source").cloned().unwrap_or(Value::Null),
  }
}

fn parse_content_block(item: &Value, mode: AnthropicContentParseMode) -> Result<Option<CoreContent>, ProtocolError> {
  if !item.is_object() {
    return Ok(None);
  }

  let typ = get_str_or(item, "type", "text");
  Ok(Some(match typ {
    "text" => {
      let Some(text) = get_str(item, "text") else {
        return Ok(None);
      };
      CoreContent::Text { text: text.to_string() }
    }
    "thinking" => {
      let text = get_first_str(item, &["thinking", "text"])
        .unwrap_or_default()
        .to_string();
      let signature = get_str(item, "signature").map(ToString::to_string);
      CoreContent::Reasoning { text, signature }
    }
    "tool_use" => {
      let call_id = match mode {
        AnthropicContentParseMode::Request => get_str(item, "id")
          .ok_or(ProtocolError::MissingResponseField("content[].id"))?
          .to_string(),
        AnthropicContentParseMode::Response => get_str_or(item, "id", "call_0").to_string(),
      };
      let name = match mode {
        AnthropicContentParseMode::Request => get_str(item, "name")
          .ok_or(ProtocolError::MissingResponseField("content[].name"))?
          .to_string(),
        AnthropicContentParseMode::Response => get_str_or(item, "name", "").to_string(),
      };
      let arguments = item.get("input").cloned().unwrap_or(Value::Null);
      let thought = get_str(item, "thought").map(ToString::to_string);
      CoreContent::ToolCall {
        call_id,
        name,
        arguments,
        thought,
      }
    }
    "tool_result" => {
      let call_id = match mode {
        AnthropicContentParseMode::Request => get_str(item, "tool_use_id")
          .ok_or(ProtocolError::MissingResponseField("content[].tool_use_id"))?
          .to_string(),
        AnthropicContentParseMode::Response => get_str_or(item, "tool_use_id", "call_0").to_string(),
      };
      let output = match mode {
        AnthropicContentParseMode::Request => item.get("content").cloned().unwrap_or(Value::Null),
        AnthropicContentParseMode::Response => item.get("content").map(parse_json_ref).unwrap_or(Value::Null),
      };
      CoreContent::ToolResult {
        call_id,
        output,
        is_error: item.get("is_error").and_then(Value::as_bool),
      }
    }
    "image" => attachment_content_from_source(block_source(item, mode), CoreAttachmentKind::Image),
    "document" => attachment_content_from_source(block_source(item, mode), CoreAttachmentKind::File),
    "audio" => attachment_content_from_source(block_source(item, mode), CoreAttachmentKind::Audio),
    _ => return Ok(None),
  }))
}

pub(crate) fn usage_to_anthropic_json(usage: &CoreUsage) -> Value {
  json!({
    "input_tokens": usage.prompt_tokens.saturating_sub(usage.cached_tokens.unwrap_or_default()),
    "output_tokens": usage.completion_tokens,
    "cache_read_input_tokens": usage.cached_tokens,
    "cache_creation_input_tokens": Value::Null,
  })
}

fn parse_base64_data_url(url: &str) -> Option<(String, String)> {
  let data_url = url.strip_prefix("data:")?;
  let (meta, payload) = data_url.split_once(',')?;
  let mut meta_parts = meta.split(';');
  let media_type = meta_parts.next().unwrap_or_default();
  let is_base64 = meta_parts.any(|part| part.eq_ignore_ascii_case("base64"));
  if !is_base64 {
    return None;
  }

  let media_type = if media_type.is_empty() {
    "application/octet-stream".to_string()
  } else {
    media_type.to_string()
  };
  Some((media_type, payload.to_string()))
}

fn infer_image_media_type_from_base64_data(data: &str) -> Option<&'static str> {
  let prefix_len = data.len().min(256);
  let usable_len = prefix_len - (prefix_len % 4);
  if usable_len == 0 {
    return None;
  }
  let prefix = &data[..usable_len];
  let decoded = general_purpose::STANDARD
    .decode(prefix)
    .or_else(|_| general_purpose::URL_SAFE.decode(prefix))
    .ok()?;

  if decoded.starts_with(&[0xFF, 0xD8, 0xFF]) {
    return Some("image/jpeg");
  }
  if decoded.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']) {
    return Some("image/png");
  }
  if decoded.starts_with(b"GIF87a") || decoded.starts_with(b"GIF89a") {
    return Some("image/gif");
  }
  if decoded.len() >= 12 && &decoded[..4] == b"RIFF" && &decoded[8..12] == b"WEBP" {
    return Some("image/webp");
  }
  None
}

fn normalize_image_url_source(url: &str) -> Value {
  if let Some((media_type, data)) = parse_base64_data_url(url) {
    let normalized_media_type = infer_image_media_type_from_base64_data(&data)
      .map(ToString::to_string)
      .unwrap_or(media_type);
    json!({ "type": "base64", "media_type": normalized_media_type, "data": data })
  } else {
    json!({ "type": "url", "url": url })
  }
}

fn normalize_attachment_source_to_anthropic(source: &Value) -> Value {
  match source {
    Value::Object(object) => {
      if object.get("type").is_some() {
        return source.clone();
      }
      if let Some(file_id) = object.get("file_id") {
        return json!({ "type": "file", "file_id": file_id });
      }
      if let (Some(Value::String(media_type)), Some(Value::String(data))) =
        (object.get("media_type"), object.get("data"))
      {
        return json!({ "type": "base64", "media_type": media_type, "data": data });
      }
      if let Some(Value::String(url)) = object.get("url") {
        return normalize_image_url_source(url);
      }
      source.clone()
    }
    Value::String(url) => normalize_image_url_source(url),
    _ => source.clone(),
  }
}

fn normalize_image_source_to_anthropic(source: &Value) -> Value {
  let normalized = normalize_attachment_source_to_anthropic(source);
  let Some(object) = normalized.as_object() else {
    return normalized;
  };
  if object.get("type").and_then(Value::as_str) != Some("base64") {
    return normalized;
  }

  let data = object.get("data").and_then(Value::as_str).unwrap_or_default();
  let media_type = object
    .get("media_type")
    .and_then(Value::as_str)
    .unwrap_or("application/octet-stream");
  let normalized_media_type = infer_image_media_type_from_base64_data(data)
    .map(ToString::to_string)
    .unwrap_or_else(|| media_type.to_string());

  json!({
    "type": "base64",
    "media_type": normalized_media_type,
    "data": data,
  })
}

fn is_anthropic_content_block(value: &Value) -> bool {
  get_str(value, "type").is_some()
}

fn tool_result_content_to_anthropic(output: &Value) -> Value {
  match output {
    Value::String(_) => output.clone(),
    Value::Array(items) if items.iter().all(is_anthropic_content_block) => output.clone(),
    _ => Value::Array(vec![json!({
      "type": "text",
      "text": stringify_json(output),
    })]),
  }
}

pub(crate) fn core_content_to_anthropic(content: &CoreContent, include_tool_thought: bool) -> Value {
  match content {
    CoreContent::Text { text } => json!({
      "type": "text",
      "text": text,
    }),
    CoreContent::Reasoning { text, signature } => json!({
      "type": "thinking",
      "thinking": text,
      "signature": signature,
    }),
    CoreContent::ToolCall {
      call_id,
      name,
      arguments,
      thought,
    } => {
      let mut block = Map::from_iter([
        ("type".to_string(), Value::String("tool_use".to_string())),
        ("id".to_string(), Value::String(call_id.clone())),
        ("name".to_string(), Value::String(name.clone())),
        ("input".to_string(), arguments.clone()),
      ]);
      if include_tool_thought && let Some(thought) = thought {
        block.insert("thought".to_string(), Value::String(thought.clone()));
      }
      Value::Object(block)
    }
    CoreContent::ToolResult {
      call_id,
      output,
      is_error,
    } => {
      let mut block = Map::from_iter([
        ("type".to_string(), Value::String("tool_result".to_string())),
        ("tool_use_id".to_string(), Value::String(call_id.clone())),
        ("content".to_string(), tool_result_content_to_anthropic(output)),
      ]);
      if let Some(is_error) = is_error {
        block.insert("is_error".to_string(), Value::Bool(*is_error));
      }
      Value::Object(block)
    }
    CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => {
      let Some((source, kind)) = attachment_source(content) else {
        return Value::Null;
      };
      let source = match kind {
        CoreAttachmentKind::Image => normalize_image_source_to_anthropic(source),
        CoreAttachmentKind::Audio | CoreAttachmentKind::File => normalize_attachment_source_to_anthropic(source),
      };
      let block_type = match kind {
        CoreAttachmentKind::Image => "image",
        CoreAttachmentKind::Audio => "audio",
        CoreAttachmentKind::File => "document",
      };
      json!({
        "type": block_type,
        "source": source,
      })
    }
  }
}
