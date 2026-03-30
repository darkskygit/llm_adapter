use serde_json::{Map, Value, json};

use super::{
  CoreMessage, CoreResponse, CoreRole, ProtocolError,
  common::{AnthropicContentParseMode, core_content_to_anthropic, parse_content_blocks, usage_to_anthropic_json},
  get_str, get_str_or, message_token_estimate, parse_role_lossy, usage_from_anthropic,
};

pub fn decode(body: &Value) -> Result<CoreResponse, ProtocolError> {
  let id = get_str(body, "id")
    .ok_or(ProtocolError::MissingField("anthropic.id"))?
    .to_string();
  let model = get_str(body, "model")
    .ok_or(ProtocolError::MissingField("anthropic.model"))?
    .to_string();
  let role = parse_role_lossy(get_str_or(body, "role", "assistant"));

  let content = body
    .get("content")
    .cloned()
    .map(|content| parse_content_blocks(content, AnthropicContentParseMode::Response))
    .transpose()?
    .unwrap_or_default();

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
    let value = core_content_to_anthropic(block, true);
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
  use crate::core::{CoreContent, CoreUsage};

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

  #[test]
  fn should_round_trip_document_blocks() {
    let decoded = decode(&json!({
      "id": "msg_3",
      "model": "claude-sonnet-4-5-20250929",
      "role": "assistant",
      "content": [{
        "type": "document",
        "source": {
          "type": "base64",
          "media_type": "application/pdf",
          "data": "Zm9v"
        }
      }]
    }))
    .unwrap();

    assert!(matches!(
      &decoded.message.content[0],
      CoreContent::File { source } if source["media_type"] == "application/pdf"
    ));

    let payload = encode(&CoreResponse {
      id: "msg_4".to_string(),
      model: "claude-sonnet-4-5-20250929".to_string(),
      message: CoreMessage {
        role: CoreRole::Assistant,
        content: vec![CoreContent::File {
          source: json!({
            "url": "https://example.com/manual.pdf",
            "media_type": "application/pdf"
          }),
        }],
      },
      usage: CoreUsage::default(),
      finish_reason: "stop".to_string(),
      reasoning_details: None,
    });

    assert_eq!(payload["content"][0]["type"], "document");
    assert_eq!(payload["content"][0]["source"]["url"], "https://example.com/manual.pdf");
  }
}
