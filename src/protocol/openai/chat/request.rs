use serde::Deserialize;
use serde_json::Value;

use super::{
  CoreMessage, CoreRequest, OpenaiDecodeRequestInput, OpenaiRequestFlavor, OpenaiTool, ProtocolError,
  decode_openai_request, encode_openai_request, messages_from_core, parse_content, parse_role, parse_tool_calls,
  tool_result_content,
};

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
  model: String,
  #[serde(default)]
  messages: Vec<ChatMessage>,
  stream: Option<bool>,
  max_tokens: Option<u32>,
  max_completion_tokens: Option<u32>,
  temperature: Option<f64>,
  #[serde(default)]
  tools: Vec<OpenaiTool>,
  tool_choice: Option<Value>,
  include: Option<Vec<String>>,
  reasoning: Option<Value>,
  reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
  role: String,
  content: Option<Value>,
  tool_calls: Option<Value>,
  tool_call_id: Option<String>,
}

pub fn decode(request: &Value) -> Result<CoreRequest, ProtocolError> {
  let request: ChatCompletionRequest = serde_json::from_value(request.clone())?;

  let mut messages = Vec::with_capacity(request.messages.len());
  for message in request.messages {
    let role = parse_role(&message.role, "role")?;
    let content = if let Some(call_id) = message.tool_call_id {
      vec![tool_result_content(call_id, message.content)]
    } else {
      let mut content = parse_content(message.content)?;
      if let Some(raw_tool_calls) = message.tool_calls {
        content.extend(parse_tool_calls(&raw_tool_calls, "tool_calls[].id|call_id")?);
      }
      content
    };

    messages.push(CoreMessage { role, content });
  }

  decode_openai_request(OpenaiDecodeRequestInput {
    model: request.model,
    messages,
    stream: request.stream,
    max_tokens: request.max_completion_tokens.or(request.max_tokens),
    temperature: request.temperature,
    tools: request.tools,
    tool_choice: request.tool_choice,
    include: request.include,
    reasoning: request.reasoning,
    reasoning_effort: request.reasoning_effort,
  })
}

#[must_use]
pub fn encode(request: &CoreRequest, stream: bool) -> Value {
  encode_openai_request(
    request,
    stream,
    OpenaiRequestFlavor::ChatCompletions,
    messages_from_core(&request.messages),
  )
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::core::{CoreContent, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition};

  #[test]
  fn decode_should_match_golden() {
    let input = json!({
      "model": "gpt-4.1",
      "messages": [
        { "role": "system", "content": "You are helpful." },
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Read this file" },
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
          ]
        },
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "call_1",
              "type": "function",
              "function": {
                "name": "doc_read",
                "arguments": "{\"docId\":\"abc\"}"
              },
              "thought": "need to inspect"
            }
          ]
        },
        {
          "role": "tool",
          "tool_call_id": "call_1",
          "content": "{\"title\":\"Doc A\"}"
        }
      ],
      "stream": true,
      "max_completion_tokens": 512,
      "temperature": 0.2,
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "doc_read",
            "description": "Read a doc",
            "parameters": {
              "type": "object",
              "properties": {
                "docId": { "type": "string" }
              }
            }
          }
        }
      ],
      "tool_choice": "auto",
      "include": ["reasoning"],
      "reasoning": { "effort": "medium" }
    });

    let core = decode(&input).unwrap();
    let actual = serde_json::to_value(core).unwrap();

    let expected = json!({
      "model": "gpt-4.1",
      "messages": [
        {
          "role": "system",
          "content": [{ "type": "text", "text": "You are helpful." }]
        },
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Read this file" },
            { "type": "image", "source": { "url": "https://example.com/a.png" } }
          ]
        },
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_call",
              "call_id": "call_1",
              "name": "doc_read",
              "arguments": { "docId": "abc" },
              "thought": "need to inspect"
            }
          ]
        },
        {
          "role": "tool",
          "content": [
            {
              "type": "tool_result",
              "call_id": "call_1",
              "output": { "title": "Doc A" }
            }
          ]
        }
      ],
      "stream": true,
      "max_tokens": 512,
      "temperature": 0.2,
      "tools": [
        {
          "name": "doc_read",
          "description": "Read a doc",
          "parameters": {
            "type": "object",
            "properties": {
              "docId": { "type": "string" }
            }
          }
        }
      ],
      "tool_choice": "auto",
      "include": ["reasoning"],
      "reasoning": { "effort": "medium" }
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
          content: vec![CoreContent::Text {
            text: "hello".to_string(),
          }],
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
      tool_choice: Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
    };

    let payload = encode(&core, true);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["stream_options"]["include_usage"], true);
    assert_eq!(payload["max_completion_tokens"], 256);
    assert_eq!(payload["temperature"], 0.3);
    assert_eq!(
      payload["messages"][1]["tool_calls"][0]["function"]["arguments"],
      "{\"docId\":\"a1\"}"
    );
    assert_eq!(payload["reasoning_effort"], "medium");
  }
}
