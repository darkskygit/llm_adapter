use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreResponse, CoreRole, CoreUsage, ProtocolError, get_first_str, get_str,
  map_gemini_finish_reason, message_token_estimate, usage_from_gemini,
};

fn get_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
  keys.iter().find_map(|key| value.get(*key))
}

fn get_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
  get_value(value, keys).and_then(Value::as_str)
}

fn parse_function_response_output(value: &Value) -> (Value, Option<bool>) {
  let Some(object) = value.as_object() else {
    return (value.clone(), None);
  };

  match (object.get("output"), object.get("is_error").and_then(Value::as_bool)) {
    (Some(output), is_error) => (output.clone(), is_error),
    (None, is_error) => (Value::Object(object.clone()), is_error),
  }
}

fn parse_candidate_message(candidate: &Value) -> CoreMessage {
  let content = candidate.get("content").unwrap_or(&Value::Null);
  let role = match get_str(content, "role").unwrap_or("model") {
    "user" => CoreRole::User,
    "system" => CoreRole::System,
    _ => CoreRole::Assistant,
  };

  let mut parts = Vec::new();
  let mut call_ids_by_name = std::collections::HashMap::new();
  if let Some(items) = content.get("parts").and_then(Value::as_array) {
    for (index, item) in items.iter().enumerate() {
      if let Some(text) = get_string(item, &["text"]) {
        if item.get("thought").and_then(Value::as_bool).unwrap_or(false) {
          parts.push(CoreContent::Reasoning {
            text: text.to_string(),
            signature: get_string(item, &["thoughtSignature", "thought_signature"]).map(ToString::to_string),
          });
        } else {
          parts.push(CoreContent::Text { text: text.to_string() });
        }
        continue;
      }

      if let Some(function_call) = get_value(item, &["functionCall", "function_call"]) {
        let name = get_string(function_call, &["name"]).unwrap_or_default().to_string();
        parts.push(CoreContent::ToolCall {
          call_id: get_string(function_call, &["id", "callId", "call_id"])
            .map(ToString::to_string)
            .unwrap_or_else(|| {
              if name.is_empty() {
                format!("call_{index}")
              } else {
                format!("{name}:{index}")
              }
            }),
          name,
          arguments: function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
          thought: None,
        });
        if let Some(CoreContent::ToolCall { call_id, name, .. }) = parts.last() {
          call_ids_by_name.insert(name.clone(), call_id.clone());
        }
        continue;
      }

      if let Some(function_response) = get_value(item, &["functionResponse", "function_response"]) {
        let name = get_string(function_response, &["name"]).unwrap_or("call");
        let raw_response = function_response
          .get("response")
          .cloned()
          .unwrap_or_else(|| Value::Object(Map::new()));
        let (output, is_error) = parse_function_response_output(&raw_response);
        parts.push(CoreContent::ToolResult {
          call_id: get_string(&raw_response, &["call_id", "callId"])
            .map(ToString::to_string)
            .or_else(|| call_ids_by_name.get(name).cloned())
            .unwrap_or_else(|| format!("{name}:{index}")),
          output,
          is_error,
        });
        continue;
      }

      if let Some(inline_data) = get_value(item, &["inlineData", "inline_data"]) {
        parts.push(CoreContent::Image {
          source: json!({
            "type": "base64",
            "media_type": get_string(inline_data, &["mimeType", "mime_type"]).unwrap_or("application/octet-stream"),
            "data": get_string(inline_data, &["data"]).unwrap_or_default(),
          }),
        });
        continue;
      }

      if let Some(file_data) = get_value(item, &["fileData", "file_data"]) {
        parts.push(CoreContent::Image {
          source: json!({
            "url": get_string(file_data, &["fileUri", "file_uri"]).unwrap_or_default(),
            "media_type": get_string(file_data, &["mimeType", "mime_type"]).unwrap_or("application/octet-stream"),
          }),
        });
      }
    }
  }

  CoreMessage { role, content: parts }
}

fn usage_to_gemini_json(usage: &CoreUsage) -> Value {
  let mut payload = Map::from_iter([
    ("promptTokenCount".to_string(), json!(usage.prompt_tokens)),
    ("candidatesTokenCount".to_string(), json!(usage.completion_tokens)),
    ("totalTokenCount".to_string(), json!(usage.total_tokens)),
  ]);
  if let Some(cached_tokens) = usage.cached_tokens {
    payload.insert("cachedContentTokenCount".to_string(), json!(cached_tokens));
  }
  Value::Object(payload)
}

fn core_content_to_part(content: &CoreContent) -> Value {
  match content {
    CoreContent::Text { text } => json!({ "text": text }),
    CoreContent::Reasoning { text, signature } => json!({
      "text": text,
      "thought": true,
      "thoughtSignature": signature,
    }),
    CoreContent::ToolCall { name, arguments, .. } => json!({
      "functionCall": {
        "name": name,
        "args": arguments,
      }
    }),
    CoreContent::ToolResult { output, is_error, .. } => {
      let response = match (output, is_error) {
        (Value::Object(object), None) => Value::Object(object.clone()),
        _ => {
          let mut response = Map::from_iter([("output".to_string(), output.clone())]);
          if let Some(is_error) = is_error {
            response.insert("is_error".to_string(), Value::Bool(*is_error));
          }
          Value::Object(response)
        }
      };
      json!({
        "functionResponse": {
          "name": "tool_result",
          "response": response,
        }
      })
    }
    CoreContent::Image { source } => {
      if let Some(url) = get_str(source, "url") {
        json!({
          "fileData": {
            "fileUri": url,
            "mimeType": get_str(source, "media_type").unwrap_or("application/octet-stream"),
          }
        })
      } else {
        json!({
          "inlineData": {
            "mimeType": get_str(source, "media_type").unwrap_or("application/octet-stream"),
            "data": get_str(source, "data").unwrap_or_default(),
          }
        })
      }
    }
  }
}

fn finish_reason_to_gemini(response: &CoreResponse) -> &'static str {
  match response.finish_reason.as_str() {
    "length" => "FINISH_REASON_MAX_TOKENS",
    "tool_calls" => "FINISH_REASON_STOP",
    "stop" => "FINISH_REASON_STOP",
    _ => "FINISH_REASON_OTHER",
  }
}

pub fn decode(body: &Value) -> Result<CoreResponse, ProtocolError> {
  let candidate = body
    .get("candidates")
    .and_then(Value::as_array)
    .and_then(|candidates| candidates.first())
    .ok_or(ProtocolError::MissingField("gemini.candidates[0]"))?;
  let message = parse_candidate_message(candidate);
  let completion_estimate = message_token_estimate(&message);
  let mut finish_reason = get_str(candidate, "finishReason")
    .map(map_gemini_finish_reason)
    .unwrap_or_else(|| "stop".to_string());
  if finish_reason == "stop"
    && message
      .content
      .iter()
      .any(|content| matches!(content, CoreContent::ToolCall { .. }))
  {
    finish_reason = "tool_calls".to_string();
  }

  Ok(CoreResponse {
    id: get_first_str(body, &["responseId", "id"])
      .unwrap_or("gemini_response")
      .to_string(),
    model: get_first_str(body, &["modelVersion", "model"])
      .unwrap_or("gemini")
      .to_string(),
    message,
    usage: usage_from_gemini(body.get("usageMetadata"), 0, completion_estimate),
    finish_reason,
    reasoning_details: body.get("promptFeedback").cloned(),
  })
}

#[must_use]
pub fn encode(response: &CoreResponse) -> Value {
  let role = match response.message.role {
    CoreRole::Assistant => "model",
    CoreRole::User | CoreRole::Tool => "user",
    CoreRole::System => "system",
  };

  json!({
    "responseId": response.id,
    "modelVersion": response.model,
    "candidates": [{
      "content": {
        "role": role,
        "parts": response
          .message
          .content
          .iter()
          .map(core_content_to_part)
          .collect::<Vec<_>>(),
      },
      "finishReason": finish_reason_to_gemini(response),
    }],
    "usageMetadata": usage_to_gemini_json(&response.usage),
    "promptFeedback": response.reasoning_details,
  })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn decode_should_cover_tool_calls_reasoning_and_usage() {
    let core = decode(&json!({
      "responseId": "resp_1",
      "modelVersion": "gemini-2.5-flash",
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [
            { "text": "draft", "thought": true, "thoughtSignature": "sig_1" },
            { "functionCall": { "name": "doc_read", "args": { "docId": "a1" } } }
          ]
        },
        "finishReason": "FINISH_REASON_STOP"
      }],
      "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 4,
        "totalTokenCount": 14,
        "cachedContentTokenCount": 2
      }
    }))
    .unwrap();

    assert_eq!(core.id, "resp_1");
    assert_eq!(core.finish_reason, "tool_calls");
    assert_eq!(core.usage.cached_tokens, Some(2));
    assert!(core.message.content.iter().any(
      |content| matches!(content, CoreContent::Reasoning { signature: Some(signature), .. } if signature == "sig_1")
    ));
  }

  #[test]
  fn encode_should_emit_finish_reason_and_usage() {
    let response = CoreResponse {
      id: "resp_2".to_string(),
      model: "gemini-2.5-flash".to_string(),
      message: CoreMessage {
        role: CoreRole::Assistant,
        content: vec![CoreContent::ToolCall {
          call_id: "call_1".to_string(),
          name: "doc_read".to_string(),
          arguments: json!({ "docId": "a1" }),
          thought: None,
        }],
      },
      usage: CoreUsage {
        prompt_tokens: 8,
        completion_tokens: 3,
        total_tokens: 11,
        cached_tokens: Some(1),
      },
      finish_reason: "tool_calls".to_string(),
      reasoning_details: None,
    };

    let payload = encode(&response);

    assert_eq!(payload["candidates"][0]["finishReason"], "FINISH_REASON_STOP");
    assert_eq!(payload["usageMetadata"]["cachedContentTokenCount"], 1);
    assert_eq!(
      payload["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
      "doc_read"
    );
  }
}
