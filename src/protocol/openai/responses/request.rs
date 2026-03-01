use serde::Deserialize;
use serde_json::Value;

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreRole, OpenaiDecodeRequestInput, OpenaiRequestFlavor, OpenaiTool,
  ProtocolError, decode_openai_request, encode_openai_request, messages_from_core, parse_content, parse_role,
  parse_tool_calls, tool_result_content,
};

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
  model: String,
  input: Value,
  stream: Option<bool>,
  max_output_tokens: Option<u32>,
  #[serde(default)]
  tools: Vec<OpenaiTool>,
  tool_choice: Option<Value>,
  include: Option<Vec<String>>,
  reasoning: Option<Value>,
}

fn parse_input_item(item: Value) -> Result<CoreMessage, ProtocolError> {
  match item {
    Value::String(text) => Ok(CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text { text }],
    }),
    Value::Object(object) => {
      if matches!(object.get("type"), Some(Value::String(typ)) if typ == "function_call_output") {
        let call_id = match object.get("call_id") {
          Some(Value::String(call_id)) => call_id.clone(),
          _ => return Err(ProtocolError::MissingField("input[].call_id")),
        };
        return Ok(CoreMessage {
          role: CoreRole::Tool,
          content: vec![tool_result_content(call_id, object.get("output").cloned())],
        });
      }

      let role = match object.get("role") {
        Some(Value::String(role)) => parse_role(role, "role")?,
        _ => return Err(ProtocolError::MissingField("input[].role")),
      };
      let content_field = object.get("content").cloned();
      let mut content = parse_content(content_field.clone())?;

      if let Some(Value::String(tool_call_id)) = object.get("tool_call_id") {
        content = vec![tool_result_content(tool_call_id.clone(), content_field)];
      } else if let Some(raw_tool_calls) = object.get("tool_calls") {
        content.extend(parse_tool_calls(raw_tool_calls, "input[].tool_calls[].id|call_id")?);
      }

      Ok(CoreMessage { role, content })
    }
    _ => Err(ProtocolError::InvalidValue {
      field: "input",
      message: "unsupported input item".to_string(),
    }),
  }
}

pub fn decode(request: &Value) -> Result<CoreRequest, ProtocolError> {
  let request: ResponsesRequest = serde_json::from_value(request.clone())?;

  let mut messages = Vec::new();
  match request.input {
    Value::Array(items) => {
      for item in items {
        messages.push(parse_input_item(item)?);
      }
    }
    other => {
      messages.push(parse_input_item(other)?);
    }
  }

  decode_openai_request(OpenaiDecodeRequestInput {
    model: request.model,
    messages,
    stream: request.stream,
    max_tokens: request.max_output_tokens,
    temperature: None,
    tools: request.tools,
    tool_choice: request.tool_choice,
    include: request.include,
    reasoning: request.reasoning,
    reasoning_effort: None,
  })
}

#[must_use]
pub fn encode(request: &CoreRequest, stream: bool) -> Value {
  encode_openai_request(
    request,
    stream,
    OpenaiRequestFlavor::Responses,
    messages_from_core(&request.messages, OpenaiRequestFlavor::Responses),
  )
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::core::{CoreContent, CoreRole, CoreToolChoice, CoreToolDefinition};

  #[test]
  fn decode_should_match_golden() {
    let input = json!({
      "model": "gpt-4.1",
      "input": [
        {
          "role": "user",
          "content": [
            { "type": "input_text", "text": "hello" },
            {
              "type": "input_image",
              "image_url": "https://example.com/a.png",
              "detail": "high"
            }
          ]
        },
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "call_2",
              "function": {
                "name": "doc_update",
                "arguments": "{\"docId\":\"123\",\"patch\":\"...\"}"
              }
            }
          ]
        },
        {
          "type": "function_call_output",
          "call_id": "call_2",
          "output": { "ok": true }
        }
      ],
      "max_output_tokens": 256,
      "tool_choice": { "name": "doc_update" }
    });

    let core = decode(&input).unwrap();
    let actual = serde_json::to_value(core).unwrap();

    let expected = json!({
      "model": "gpt-4.1",
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "hello" },
            { "type": "image", "source": { "url": "https://example.com/a.png", "detail": "high" } }
          ]
        },
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_call",
              "call_id": "call_2",
              "name": "doc_update",
              "arguments": { "docId": "123", "patch": "..." }
            }
          ]
        },
        {
          "role": "tool",
          "content": [
            {
              "type": "tool_result",
              "call_id": "call_2",
              "output": { "ok": true }
            }
          ]
        }
      ],
      "stream": false,
      "max_tokens": 256,
      "tool_choice": { "name": "doc_update" }
    });

    assert_eq!(actual, expected);
  }

  #[test]
  fn encode_should_match_backend_contract() {
    let core = CoreRequest {
      model: "gpt-4.1".to_string(),
      messages: vec![
        CoreMessage {
          role: CoreRole::User,
          content: vec![
            CoreContent::Text {
              text: "hello".to_string(),
            },
            CoreContent::Image {
              source: json!({
                "url": "https://example.com/a.png",
                "detail": "high"
              }),
            },
          ],
        },
        CoreMessage {
          role: CoreRole::Assistant,
          content: vec![CoreContent::ToolCall {
            call_id: "call_42".to_string(),
            name: "doc_read".to_string(),
            arguments: json!({ "docId": "a1" }),
            thought: Some("need context".to_string()),
          }],
        },
      ],
      stream: false,
      max_tokens: Some(256),
      temperature: Some(0.3),
      tools: vec![CoreToolDefinition {
        name: "doc_read".to_string(),
        description: Some("Read a doc".to_string()),
        parameters: json!({
          "type": "object",
          "properties": { "docId": { "type": "string" } }
        }),
      }],
      tool_choice: Some(CoreToolChoice::Specific {
        name: "doc_read".to_string(),
      }),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
    };

    let payload = encode(&core, true);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["max_output_tokens"], 256);
    assert_eq!(payload["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(payload["input"][0]["content"][1]["type"], "input_image");
    assert_eq!(
      payload["input"][0]["content"][1]["image_url"],
      "https://example.com/a.png"
    );
    assert_eq!(payload["input"][0]["content"][1]["detail"], "high");
    assert_eq!(payload["tools"][0]["name"], "doc_read");
    assert_eq!(payload["tool_choice"]["name"], Value::String("doc_read".to_string()));
    assert_eq!(payload["reasoning"]["effort"], "medium");
  }
}
