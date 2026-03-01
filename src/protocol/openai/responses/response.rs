use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreResponse, CoreRole, CoreUsage, ProtocolError, get_first_str, get_first_str_or, get_str,
  get_str_or, message_token_estimate, parse_json, parse_role_lossy, stringify_json, usage_from_responses,
};

fn usage_to_openai_json(usage: &CoreUsage) -> Value {
  let mut object = Map::from_iter([
    ("input_tokens".to_string(), json!(usage.prompt_tokens)),
    ("output_tokens".to_string(), json!(usage.completion_tokens)),
    ("total_tokens".to_string(), json!(usage.total_tokens)),
  ]);

  if let Some(cached_tokens) = usage.cached_tokens {
    object.insert(
      "input_tokens_details".to_string(),
      json!({ "cached_tokens": cached_tokens }),
    );
  }

  Value::Object(object)
}

pub fn decode(body: &Value) -> Result<CoreResponse, ProtocolError> {
  let id = get_str(body, "id")
    .ok_or(ProtocolError::MissingField("openai_responses.id"))?
    .to_string();
  let model = get_str(body, "model")
    .ok_or(ProtocolError::MissingField("openai_responses.model"))?
    .to_string();

  let mut role = CoreRole::Assistant;
  let mut content = Vec::new();
  if let Some(output_items) = body.get("output").and_then(Value::as_array) {
    for item in output_items {
      let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
      match item_type {
        "function_call" => {
          let call_id = get_first_str_or(item, &["call_id", "id"], "call_0").to_string();
          let name = get_str_or(item, "name", "").to_string();
          let arguments = item
            .get("arguments")
            .cloned()
            .map(parse_json)
            .unwrap_or_else(|| Value::Object(Map::new()));
          content.push(CoreContent::ToolCall {
            call_id,
            name,
            arguments,
            thought: None,
          });
        }
        "function_call_output" => {
          role = CoreRole::Tool;
          let call_id = get_first_str_or(item, &["call_id"], "call_0").to_string();
          let output = item.get("output").cloned().unwrap_or(Value::Null);
          content.push(CoreContent::ToolResult {
            call_id,
            output,
            is_error: item.get("is_error").and_then(Value::as_bool),
          });
        }
        "reasoning" => {
          if let Some(text) = item
            .get("summary")
            .and_then(Value::as_array)
            .and_then(|summary| summary.first())
            .and_then(|first| first.get("text"))
            .and_then(Value::as_str)
          {
            content.push(CoreContent::Reasoning {
              text: text.to_string(),
              signature: None,
            });
          }
        }
        "message" => {
          role = parse_role_lossy(get_str_or(item, "role", "assistant"));
          if let Some(contents) = item.get("content").and_then(Value::as_array) {
            for block in contents {
              if block.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(text) = get_str(block, "text") {
                  content.push(CoreContent::Text { text: text.to_string() });
                }
              } else if let Some(text) = get_str(block, "text") {
                content.push(CoreContent::Text { text: text.to_string() });
              }
            }
          }
        }
        _ => {}
      }
    }
  }

  if content.is_empty()
    && let Some(text) = get_first_str(body, &["output_text", "text"])
  {
    content.push(CoreContent::Text { text: text.to_string() });
  }

  let message = CoreMessage { role, content };
  let completion_estimate = message_token_estimate(&message);
  let usage = usage_from_responses(body.get("usage"), 0, completion_estimate);
  let finish_reason = super::map_responses_finish_reason(
    body.get("status").and_then(Value::as_str),
    get_str(body, "finish_reason"),
  );

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
  let mut tool_calls = Vec::new();
  let mut tool_results = Vec::new();

  for content in &response.message.content {
    match content {
      CoreContent::Text { text } | CoreContent::Reasoning { text, .. } => text_parts.push(text.clone()),
      CoreContent::ToolCall {
        call_id,
        name,
        arguments,
        ..
      } => {
        tool_calls.push(json!({
          "type": "function_call",
          "id": Value::Null,
          "call_id": call_id,
          "name": name,
          "arguments": stringify_json(arguments),
        }));
      }
      CoreContent::ToolResult { call_id, output, .. } => {
        tool_results.push((call_id.clone(), output.clone()));
      }
      CoreContent::Image { .. } => {}
    }
  }

  let output = if !tool_calls.is_empty() {
    Value::Array(tool_calls)
  } else {
    let role = match response.message.role {
      CoreRole::System => "system",
      CoreRole::User => "user",
      CoreRole::Assistant => "assistant",
      CoreRole::Tool => "tool",
    };
    let text = if !text_parts.is_empty() {
      text_parts.join("")
    } else if let Some((_call_id, output)) = tool_results.first() {
      stringify_json(output)
    } else {
      String::new()
    };

    Value::Array(vec![json!({
      "type": "message",
      "id": Value::Null,
      "status": "completed",
      "role": role,
      "content": [{
        "type": "output_text",
        "text": text,
        "annotations": [],
      }],
      "tool_call_id": tool_results.first().map(|(call_id, _)| call_id),
    })])
  };

  let mut payload = Map::from_iter([
    ("id".to_string(), json!(response.id)),
    ("object".to_string(), json!("response")),
    ("created_at".to_string(), json!(0)),
    ("model".to_string(), json!(response.model)),
    (
      "status".to_string(),
      json!(if !matches!(output, Value::Array(ref arr) if arr.is_empty())
        && output
          .as_array()
          .and_then(|arr| arr.first())
          .and_then(|item| item.get("type"))
          == Some(&Value::String("function_call".to_string()))
      {
        "requires_action"
      } else {
        "completed"
      }),
    ),
    ("output".to_string(), output),
    ("usage".to_string(), usage_to_openai_json(&response.usage)),
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
  use crate::core::CoreRole;

  #[test]
  fn decode_should_cover_backend_reasoning_and_tool_result_case() {
    let core = decode(&json!({
      "id": "resp_2",
      "model": "gpt-4.1",
      "status": "completed",
      "output": [
        {
          "type": "reasoning",
          "summary": [{ "type": "summary_text", "text": "thinking..." }]
        },
        {
          "type": "function_call_output",
          "call_id": "call_8",
          "output": { "ok": true },
          "is_error": false
        }
      ]
    }))
    .unwrap();

    assert_eq!(core.message.role, CoreRole::Tool);
    assert!(
      core
        .message
        .content
        .iter()
        .any(|content| matches!(content, CoreContent::ToolResult { call_id, .. } if call_id == "call_8"))
    );
    assert!(
      core
        .message
        .content
        .iter()
        .any(|content| matches!(content, CoreContent::Reasoning { text, .. } if text == "thinking..."))
    );
  }

  #[test]
  fn encode_should_emit_message_output_when_no_tool_calls() {
    let response = CoreResponse {
      id: "resp_2".to_string(),
      model: "gpt-4.1-mini".to_string(),
      message: CoreMessage {
        role: CoreRole::Assistant,
        content: vec![CoreContent::Text {
          text: "Final answer".to_string(),
        }],
      },
      usage: CoreUsage {
        prompt_tokens: 10,
        completion_tokens: 4,
        total_tokens: 14,
        cached_tokens: None,
      },
      finish_reason: "stop".to_string(),
      reasoning_details: None,
    };

    let payload = encode(&response);
    assert_eq!(payload["status"], "completed");
    assert_eq!(payload["output"][0]["type"], "message");
    assert_eq!(payload["output"][0]["content"][0]["text"], "Final answer");
  }
}
