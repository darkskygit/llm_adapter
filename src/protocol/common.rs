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
                  let source = object
                    .get("image_url")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(object.clone()));
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
