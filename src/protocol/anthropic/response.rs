use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreResponse, CoreRole, CoreUsage, ProtocolError, get_first_str, get_str, get_str_or,
  message_token_estimate, parse_json_ref, parse_role_lossy, usage_from_anthropic,
};

fn usage_to_anthropic_json(usage: &CoreUsage) -> Value {
  json!({
    "input_tokens": usage.prompt_tokens.saturating_sub(usage.cached_tokens.unwrap_or_default()),
    "output_tokens": usage.completion_tokens,
    "cache_read_input_tokens": usage.cached_tokens,
    "cache_creation_input_tokens": Value::Null,
  })
}

pub fn decode(body: &Value) -> Result<CoreResponse, ProtocolError> {
  let id = get_str(body, "id")
    .ok_or(ProtocolError::MissingField("anthropic.id"))?
    .to_string();
  let model = get_str(body, "model")
    .ok_or(ProtocolError::MissingField("anthropic.model"))?
    .to_string();
  let role = parse_role_lossy(get_str_or(body, "role", "assistant"));

  let mut content = Vec::new();
  if let Some(blocks) = body.get("content").and_then(Value::as_array) {
    for block in blocks {
      let block_type = get_str_or(block, "type", "text");
      match block_type {
        "text" => {
          if let Some(text) = get_str(block, "text") {
            content.push(CoreContent::Text { text: text.to_string() });
          }
        }
        "thinking" => {
          let text = get_first_str(block, &["thinking", "text"])
            .unwrap_or_default()
            .to_string();
          content.push(CoreContent::Reasoning {
            text,
            signature: get_str(block, "signature").map(ToString::to_string),
          });
        }
        "tool_use" => {
          let call_id = get_str_or(block, "id", "call_0").to_string();
          let name = get_str_or(block, "name", "").to_string();
          let arguments = block.get("input").cloned().unwrap_or(Value::Null);
          content.push(CoreContent::ToolCall {
            call_id,
            name,
            arguments,
            thought: get_str(block, "thought").map(ToString::to_string),
          });
        }
        "tool_result" => {
          let call_id = get_str_or(block, "tool_use_id", "call_0").to_string();
          let output = block.get("content").map(parse_json_ref).unwrap_or(Value::Null);
          content.push(CoreContent::ToolResult {
            call_id,
            output,
            is_error: block.get("is_error").and_then(Value::as_bool),
          });
        }
        "image" => {
          let source = block.get("source").cloned().unwrap_or(Value::Null);
          content.push(CoreContent::Image { source });
        }
        _ => {}
      }
    }
  }

  let message = CoreMessage { role, content };
  let completion_estimate = message_token_estimate(&message);
  let usage = usage_from_anthropic(body.get("usage"), 0, completion_estimate);
  let finish_reason = get_str(body, "stop_reason")
    .map(super::map_anthropic_finish_reason)
    .unwrap_or_else(|| "stop".to_string());

  Ok(CoreResponse {
    id,
    model,
    message,
    usage,
    finish_reason,
    reasoning_details: body.get("reasoning_details").cloned(),
  })
}

#[must_use]
pub fn encode(response: &CoreResponse) -> Value {
  let role = match response.message.role {
    CoreRole::System => "system",
    CoreRole::User => "user",
    CoreRole::Assistant | CoreRole::Tool => "assistant",
  };

  let mut content = Vec::new();
  for block in &response.message.content {
    let value = match block {
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
      } => json!({
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": arguments,
        "thought": thought,
      }),
      CoreContent::ToolResult {
        call_id,
        output,
        is_error,
      } => json!({
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": output,
        "is_error": is_error,
      }),
      CoreContent::Image { source } => json!({
        "type": "image",
        "source": source,
      }),
    };
    content.push(value);
  }

  let mut payload = Map::from_iter([
    ("id".to_string(), json!(response.id)),
    ("type".to_string(), json!("message")),
    ("model".to_string(), json!(response.model)),
    ("role".to_string(), json!(role)),
    ("stop_reason".to_string(), json!(response.finish_reason)),
    ("stop_sequence".to_string(), Value::Null),
    ("content".to_string(), Value::Array(content)),
    ("usage".to_string(), usage_to_anthropic_json(&response.usage)),
  ]);

  if let Some(reasoning_details) = &response.reasoning_details {
    payload.insert("reasoning_details".to_string(), reasoning_details.clone());
  }

  Value::Object(payload)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn decode_should_cover_backend_case_and_usage() {
    let core = decode(&json!({
      "id": "msg_2",
      "model": "claude-sonnet-4-5-20250929",
      "role": "assistant",
      "content": [{ "type": "text", "text": "ok" }],
      "stop_reason": "end_turn",
      "usage": {
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0
      }
    }))
    .unwrap();

    assert_eq!(core.finish_reason, "stop");
    assert_eq!(core.usage.prompt_tokens, 3);
  }
}
