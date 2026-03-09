use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreResponse, CoreRole, OpenaiRequestFlavor, ProtocolError, core_role_to_string, get_str,
  get_str_or, message_token_estimate, parse_message_content, parse_role_lossy, stringify_json, usage_from_openai,
  usage_to_openai_json,
};

fn parse_message_from_response(message: &Value) -> Result<CoreMessage, ProtocolError> {
  let role = parse_role_lossy(get_str_or(message, "role", "assistant"));
  let tool_call_id = if role == CoreRole::Tool {
    get_str(message, "tool_call_id").map(ToString::to_string)
  } else {
    None
  };
  let content_value = message.get("content").cloned();
  let normalized_content = match content_value {
    Some(Value::String(_)) | Some(Value::Array(_)) => content_value,
    Some(Value::Null) | None => None,
    _ if tool_call_id.is_some() => content_value,
    _ => None,
  };
  let mut content = parse_message_content(
    normalized_content,
    tool_call_id,
    message.get("tool_calls").cloned(),
    "openai_chat.message.tool_calls[].id|call_id",
  )?;
  content.retain(|item| !matches!(item, CoreContent::Text { text } if text.is_empty()));

  if let Some(reasoning) = get_str(message, "reasoning_content")
    && !reasoning.is_empty()
  {
    content.push(CoreContent::Reasoning {
      text: reasoning.to_string(),
      signature: None,
    });
  }

  Ok(CoreMessage { role, content })
}

pub fn decode(body: &Value) -> Result<CoreResponse, ProtocolError> {
  let id = get_str(body, "id")
    .ok_or(ProtocolError::MissingField("openai_chat.id"))?
    .to_string();
  let model = get_str(body, "model")
    .ok_or(ProtocolError::MissingField("openai_chat.model"))?
    .to_string();
  let choice = body
    .get("choices")
    .and_then(Value::as_array)
    .and_then(|choices| choices.first())
    .ok_or(ProtocolError::MissingField("openai_chat.choices[0]"))?;
  let message = parse_message_from_response(
    choice
      .get("message")
      .ok_or(ProtocolError::MissingField("openai_chat.choices[0].message"))?,
  )?;
  let completion_estimate = message_token_estimate(&message);
  let usage = usage_from_openai(body.get("usage"), 0, completion_estimate);
  let finish_reason = get_str_or(choice, "finish_reason", "stop").to_string();

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
  let mut text_parts = Vec::new();
  let mut reasoning_parts = Vec::new();
  let mut tool_calls = Vec::new();
  let mut tool_result: Option<(String, Value)> = None;

  for content in &response.message.content {
    match content {
      CoreContent::Text { text } => text_parts.push(text.clone()),
      CoreContent::Reasoning { text, .. } => reasoning_parts.push(text.clone()),
      CoreContent::ToolCall {
        call_id,
        name,
        arguments,
        thought,
      } => {
        let mut tool_call = Map::from_iter([
          ("id".to_string(), json!(call_id)),
          ("type".to_string(), json!("function")),
          (
            "function".to_string(),
            json!({
              "name": name,
              "arguments": stringify_json(arguments)
            }),
          ),
        ]);
        if let Some(thought) = thought {
          tool_call.insert("thought".to_string(), json!(thought));
        }
        tool_calls.push(Value::Object(tool_call));
      }
      CoreContent::ToolResult { call_id, output, .. } => {
        tool_result = Some((call_id.clone(), output.clone()));
      }
      CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => {}
    }
  }

  let role = core_role_to_string(&response.message.role);

  let mut message = Map::from_iter([("role".to_string(), json!(role))]);
  if let Some((tool_call_id, output)) = tool_result {
    message.insert("tool_call_id".to_string(), json!(tool_call_id));
    message.insert("content".to_string(), json!(stringify_json(&output)));
  } else if !text_parts.is_empty() {
    message.insert("content".to_string(), json!(text_parts.join("")));
  } else {
    message.insert("content".to_string(), Value::Null);
  }

  if !reasoning_parts.is_empty() {
    message.insert("reasoning_content".to_string(), json!(reasoning_parts.join("")));
  }
  if !tool_calls.is_empty() {
    message.insert("tool_calls".to_string(), Value::Array(tool_calls));
  }

  let mut payload = Map::from_iter([
    ("id".to_string(), json!(response.id)),
    ("object".to_string(), json!("chat.completion")),
    ("created".to_string(), json!(0)),
    ("model".to_string(), json!(response.model)),
    (
      "choices".to_string(),
      json!([{
        "index": 0,
        "message": Value::Object(message),
        "finish_reason": response.finish_reason,
      }]),
    ),
    (
      "usage".to_string(),
      usage_to_openai_json(&response.usage, OpenaiRequestFlavor::ChatCompletions),
    ),
  ]);

  if let Some(reasoning_details) = &response.reasoning_details {
    payload.insert("reasoning_details".to_string(), reasoning_details.clone());
  }

  Value::Object(payload)
}

#[cfg(test)]
mod tests {
  use serde_json::{Value, json};

  use super::*;

  #[test]
  fn decode_should_cover_backend_tool_call_case() {
    let core = decode(&json!({
      "id": "chat_2",
      "model": "gpt-4.1",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": Value::Null,
          "tool_calls": [{
            "id": "call_2",
            "type": "function",
            "function": { "name": "doc_read", "arguments": "{\"docId\":\"a1\"}" }
          }]
        },
        "finish_reason": "tool_calls"
      }]
    }))
    .unwrap();

    assert_eq!(core.finish_reason, "tool_calls");
    assert!(core.message.content.iter().any(
      |content| matches!(content, CoreContent::ToolCall { call_id, name, .. } if call_id == "call_2" && name == "doc_read")
    ));
  }
}
