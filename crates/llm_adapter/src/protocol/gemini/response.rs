use serde_json::{Value, json};

use super::{
  CoreContent, CoreMessage, CoreResponse, CoreRole, ProtocolError, attachment_source,
  common::{attachment_source_to_part, parse_parts, tool_result_response, usage_to_gemini_json},
  get_first_str, get_str, map_gemini_finish_reason, message_token_estimate, usage_from_gemini,
};
use crate::backend::AttachmentReferencePlan;

fn parse_candidate_message(candidate: &Value) -> CoreMessage {
  let content = candidate.get("content").unwrap_or(&Value::Null);
  let role = match get_str(content, "role").unwrap_or("model") {
    "user" => CoreRole::User,
    "system" => CoreRole::System,
    _ => CoreRole::Assistant,
  };

  CoreMessage {
    role,
    content: content
      .get("parts")
      .and_then(Value::as_array)
      .map(|parts| parse_parts(parts))
      .unwrap_or_default(),
  }
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
      let response = tool_result_response(output, *is_error);
      json!({
        "functionResponse": {
          "name": "tool_result",
          "response": response,
        }
      })
    }
    CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => {
      let Some((source, _)) = attachment_source(content) else {
        return Value::Null;
      };
      attachment_source_to_part(source, AttachmentReferencePlan::Remote)
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
  use crate::core::CoreUsage;

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

  #[test]
  fn decode_should_preserve_audio_and_file_parts() {
    let core = decode(&json!({
      "responseId": "resp_3",
      "modelVersion": "gemini-2.5-flash",
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [
            {
              "fileData": {
                "mimeType": "application/pdf",
                "fileUri": "https://example.com/spec.pdf"
              }
            },
            {
              "inlineData": {
                "mimeType": "audio/wav",
                "data": "Zm9v"
              }
            }
          ]
        }
      }]
    }))
    .unwrap();

    assert!(matches!(
      &core.message.content[0],
      CoreContent::File { source } if source["url"] == "https://example.com/spec.pdf"
    ));
    assert!(matches!(
      &core.message.content[1],
      CoreContent::Audio { source } if source["media_type"] == "audio/wav" && source["data"] == "Zm9v"
    ));
  }

  #[test]
  fn encode_should_emit_file_and_audio_parts() {
    let response = CoreResponse {
      id: "resp_4".to_string(),
      model: "gemini-2.5-flash".to_string(),
      message: CoreMessage {
        role: CoreRole::Assistant,
        content: vec![
          CoreContent::File {
            source: json!({
              "url": "https://example.com/spec.pdf",
              "media_type": "application/pdf"
            }),
          },
          CoreContent::Audio {
            source: json!({
              "data": "Zm9v",
              "media_type": "audio/wav"
            }),
          },
        ],
      },
      usage: CoreUsage::default(),
      finish_reason: "stop".to_string(),
      reasoning_details: None,
    };

    let payload = encode(&response);

    assert_eq!(
      payload["candidates"][0]["content"]["parts"][0]["fileData"]["mimeType"],
      "application/pdf"
    );
    assert_eq!(
      payload["candidates"][0]["content"]["parts"][1]["inlineData"]["mimeType"],
      "audio/wav"
    );
  }
}
