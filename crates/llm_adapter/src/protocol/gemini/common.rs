use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::{CoreAttachmentKind, CoreContent, CoreUsage, attachment_content_from_source, infer_media_type_from_url};
use crate::backend::AttachmentReferencePlan;

pub(crate) fn get_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
  keys.iter().find_map(|key| value.get(*key))
}

pub(crate) fn get_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
  get_value(value, keys).and_then(Value::as_str)
}

pub(crate) fn parse_function_response_output(value: &Value) -> (Value, Option<bool>) {
  let Some(object) = value.as_object() else {
    return (value.clone(), None);
  };

  match (object.get("output"), object.get("is_error").and_then(Value::as_bool)) {
    (Some(output), is_error) => (output.clone(), is_error),
    (None, is_error) => (Value::Object(object.clone()), is_error),
  }
}

pub(crate) fn attachment_source_to_part(source: &Value, plan: AttachmentReferencePlan) -> Value {
  match source {
    Value::Object(object) => {
      let inline_part = || {
        object
          .get("media_type")
          .and_then(Value::as_str)
          .zip(object.get("data").and_then(Value::as_str))
          .map(|(media_type, data)| {
            json!({
              "inlineData": {
                "mimeType": media_type,
                "data": data,
              }
            })
          })
          .or_else(|| {
            object
              .get("mimeType")
              .and_then(Value::as_str)
              .zip(object.get("data").and_then(Value::as_str))
              .map(|(media_type, data)| {
                json!({
                  "inlineData": {
                    "mimeType": media_type,
                    "data": data,
                  }
                })
              })
          })
      };

      let remote_part = || {
        object.get("url").and_then(Value::as_str).map(|url| {
          json!({
            "fileData": {
              "mimeType": object
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or_else(|| infer_media_type_from_url(url)),
              "fileUri": url,
            }
          })
        })
      };

      match plan {
        AttachmentReferencePlan::Inline => {
          if let Some(part) = inline_part().or_else(remote_part) {
            return part;
          }
        }
        AttachmentReferencePlan::Remote => {
          if let Some(part) = remote_part().or_else(inline_part) {
            return part;
          }
        }
      }

      if let (Some(Value::String(media_type)), Some(Value::String(data))) =
        (object.get("media_type"), object.get("data"))
      {
        return json!({
          "inlineData": {
            "mimeType": media_type,
            "data": data,
          }
        });
      }

      if let Some(Value::String(url)) = object.get("url") {
        return json!({
          "fileData": {
            "mimeType": object
              .get("media_type")
              .and_then(Value::as_str)
              .unwrap_or_else(|| infer_media_type_from_url(url)),
            "fileUri": url,
          }
        });
      }

      if let (Some(Value::String(media_type)), Some(Value::String(data))) = (object.get("mimeType"), object.get("data"))
      {
        return json!({
          "inlineData": {
            "mimeType": media_type,
            "data": data,
          }
        });
      }
      Value::Object(object.clone())
    }
    Value::String(url) => json!({
      "fileData": {
        "mimeType": infer_media_type_from_url(url),
        "fileUri": url,
      }
    }),
    _ => Value::Null,
  }
}

pub(crate) fn parse_parts(parts: &[Value]) -> Vec<CoreContent> {
  let mut call_ids_by_name = HashMap::new();

  parts
    .iter()
    .enumerate()
    .filter_map(|(index, part)| {
      if let Some(text) = get_string(part, &["text"]) {
        return Some(if part.get("thought").and_then(Value::as_bool).unwrap_or(false) {
          CoreContent::Reasoning {
            text: text.to_string(),
            signature: get_string(part, &["thoughtSignature", "thought_signature"]).map(ToString::to_string),
          }
        } else {
          CoreContent::Text { text: text.to_string() }
        });
      }

      if let Some(function_call) = get_value(part, &["functionCall", "function_call"]) {
        let name = get_string(function_call, &["name"]).unwrap_or_default().to_string();
        let call_id = get_string(function_call, &["id", "callId", "call_id"])
          .map(ToString::to_string)
          .unwrap_or_else(|| {
            if name.is_empty() {
              format!("call_{index}")
            } else {
              format!("{name}:{index}")
            }
          });
        call_ids_by_name.insert(name.clone(), call_id.clone());
        return Some(CoreContent::ToolCall {
          call_id,
          name,
          arguments: function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
          thought: None,
        });
      }

      if let Some(function_response) = get_value(part, &["functionResponse", "function_response"]) {
        let name = get_string(function_response, &["name"]).unwrap_or("call");
        let raw_response = function_response
          .get("response")
          .cloned()
          .unwrap_or_else(|| Value::Object(Map::new()));
        let call_id = get_string(&raw_response, &["call_id", "callId"])
          .map(ToString::to_string)
          .or_else(|| call_ids_by_name.get(name).cloned())
          .unwrap_or_else(|| format!("{name}:{index}"));
        let (output, is_error) = parse_function_response_output(&raw_response);
        return Some(CoreContent::ToolResult {
          call_id,
          output,
          is_error,
        });
      }

      if let Some(inline_data) = get_value(part, &["inlineData", "inline_data"]) {
        return Some(attachment_content_from_source(
          json!({
            "media_type": get_string(inline_data, &["mimeType", "mime_type"]).unwrap_or("application/octet-stream"),
            "data": get_string(inline_data, &["data"]).unwrap_or_default(),
          }),
          CoreAttachmentKind::File,
        ));
      }

      if let Some(file_data) = get_value(part, &["fileData", "file_data"]) {
        return Some(attachment_content_from_source(
          json!({
            "url": get_string(file_data, &["fileUri", "file_uri"]).unwrap_or_default(),
            "media_type": get_string(file_data, &["mimeType", "mime_type"]).unwrap_or("application/octet-stream"),
          }),
          CoreAttachmentKind::File,
        ));
      }

      None
    })
    .collect()
}

pub(crate) fn tool_result_response(output: &Value, is_error: Option<bool>) -> Value {
  match (output, is_error) {
    (Value::Object(object), None) => Value::Object(object.clone()),
    _ => {
      let mut response = Map::from_iter([("output".to_string(), output.clone())]);
      if let Some(is_error) = is_error {
        response.insert("is_error".to_string(), Value::Bool(is_error));
      }
      Value::Object(response)
    }
  }
}

pub(crate) fn usage_to_gemini_json(usage: &CoreUsage) -> Value {
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
