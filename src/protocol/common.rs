use serde_json::Value;

use super::{
  CoreContent, CoreMessage, CoreRole, CoreUsage, ProtocolError, get_cached_tokens, get_u32_or, stringify_json,
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
      CoreContent::Image { .. } => {}
    }
  }
  let chars = text.chars().count() as u32;
  (chars / 4).max(1)
}

pub(crate) fn parse_text_or_array_content(value: Option<Value>) -> Result<Vec<CoreContent>, ProtocolError> {
  let Some(value) = value else {
    return Ok(Vec::new());
  };

  match value {
    Value::String(text) => Ok(vec![CoreContent::Text { text }]),
    Value::Array(items) => {
      let mut content = Vec::new();
      for item in items {
        match item {
          Value::String(text) => content.push(CoreContent::Text { text }),
          Value::Object(object) => {
            if let Some(Value::String(typ)) = object.get("type") {
              match typ.as_str() {
                "text" | "input_text" | "output_text" => {
                  if let Some(Value::String(text)) = object.get("text") {
                    content.push(CoreContent::Text { text: text.clone() });
                  }
                }
                "image_url" => {
                  let source = match object.get("image_url") {
                    Some(Value::String(url)) => {
                      let mut source = serde_json::Map::new();
                      source.insert("url".to_string(), Value::String(url.clone()));
                      Value::Object(source)
                    }
                    Some(value) => value.clone(),
                    None => Value::Object(object.clone()),
                  };
                  content.push(CoreContent::Image { source });
                }
                "input_image" => {
                  let mut source = serde_json::Map::new();
                  if let Some(image_url) = object.get("image_url") {
                    match image_url {
                      Value::String(url) => {
                        source.insert("url".to_string(), Value::String(url.clone()));
                      }
                      Value::Object(image_url_object) => {
                        if let Some(url) = image_url_object.get("url") {
                          source.insert("url".to_string(), url.clone());
                        }
                        for (key, value) in image_url_object {
                          if key != "url" {
                            source.insert(key.clone(), value.clone());
                          }
                        }
                      }
                      _ => {
                        source.insert("image_url".to_string(), image_url.clone());
                      }
                    }
                  }
                  if let Some(file_id) = object.get("file_id") {
                    source.insert("file_id".to_string(), file_id.clone());
                  }
                  if let Some(detail) = object.get("detail") {
                    source.insert("detail".to_string(), detail.clone());
                  }
                  if source.is_empty() {
                    for (key, value) in &object {
                      if key != "type" {
                        source.insert(key.clone(), value.clone());
                      }
                    }
                  }
                  let source = Value::Object(source);
                  content.push(CoreContent::Image { source });
                }
                "image" => {
                  let source = object
                    .get("source")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(object.clone()));
                  content.push(CoreContent::Image { source });
                }
                _ => {}
              }
            } else if let Some(Value::String(text)) = object.get("text") {
              content.push(CoreContent::Text { text: text.clone() });
            }
          }
          _ => {}
        }
      }
      Ok(content)
    }
    _ => Err(ProtocolError::InvalidValue {
      field: "content",
      message: "expected string or array".to_string(),
    }),
  }
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
