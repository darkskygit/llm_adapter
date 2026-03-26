use serde::Deserialize;
use serde_json::Value;

use super::{
  CoreMessage, CoreRequest, OpenaiDecodeRequestInput, OpenaiRequestFlavor, OpenaiTool, ProtocolError,
  decode_openai_request, encode_openai_request, messages_from_core, parse_message_content, parse_role,
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
    let content = parse_message_content(
      message.content,
      message.tool_call_id,
      message.tool_calls,
      "tool_calls[].id|call_id",
    )?;

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
    messages_from_core(&request.messages, OpenaiRequestFlavor::ChatCompletions),
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
          content: vec![
            CoreContent::Text {
              text: "hello".to_string(),
            },
            CoreContent::Image {
              source: json!({
                "url": "https://example.com/a.png",
                "detail": "low"
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
      tool_choice: Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
      response_schema: None,
    };

    let payload = encode(&core, true);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["stream_options"]["include_usage"], true);
    assert_eq!(payload["max_completion_tokens"], 256);
    assert_eq!(payload["temperature"], 0.3);
    assert_eq!(payload["messages"][0]["content"][0]["type"], "text");
    assert_eq!(payload["messages"][0]["content"][1]["type"], "image_url");
    assert_eq!(
      payload["messages"][0]["content"][1]["image_url"]["url"],
      "https://example.com/a.png"
    );
    assert_eq!(payload["messages"][0]["content"][1]["image_url"]["detail"], "low");
    assert_eq!(
      payload["messages"][1]["tool_calls"][0]["function"]["arguments"],
      "{\"docId\":\"a1\"}"
    );
    assert_eq!(payload["messages"][1]["tool_calls"][0]["thought"], "need context");
    assert_eq!(payload["reasoning_effort"], "medium");
  }

  #[test]
  fn encode_should_omit_null_tool_call_thought() {
    let core = CoreRequest {
      model: "mistral-medium".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::Assistant,
        content: vec![CoreContent::ToolCall {
          call_id: "call_42".to_string(),
          name: "doc_read".to_string(),
          arguments: json!({ "docId": "a1" }),
          thought: None,
        }],
      }],
      stream: false,
      max_tokens: None,
      temperature: None,
      tools: Vec::new(),
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    let payload = encode(&core, false);

    assert!(
      payload["messages"][0]["tool_calls"][0]
        .get("thought")
        .is_none()
    );
  }

  #[test]
  fn decode_should_preserve_audio_input_parts() {
    let core = decode(&json!({
      "model": "gpt-audio",
      "messages": [{
        "role": "user",
        "content": [
          { "type": "text", "text": "What is in this recording?" },
          {
            "type": "input_audio",
            "input_audio": {
              "data": "Zm9v",
              "format": "wav"
            }
          }
        ]
      }]
    }))
    .unwrap();

    assert!(matches!(
      &core.messages[0].content[1],
      CoreContent::Audio { source } if source["media_type"] == "audio/wav" && source["data"] == "Zm9v"
    ));
  }

  #[test]
  fn encode_should_emit_audio_parts_for_chat_completions() {
    let core = CoreRequest {
      model: "gpt-audio".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![
          CoreContent::Text {
            text: "What is in this recording?".to_string(),
          },
          CoreContent::Audio {
            source: json!({
              "data": "Zm9v",
              "media_type": "audio/wav"
            }),
          },
        ],
      }],
      stream: false,
      max_tokens: None,
      temperature: None,
      tools: vec![],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    let payload = encode(&core, false);

    assert_eq!(payload["messages"][0]["content"][1]["type"], "input_audio");
    assert_eq!(payload["messages"][0]["content"][1]["input_audio"]["format"], "wav");
  }

  #[test]
  fn encode_should_emit_response_format_for_structured_outputs() {
    let payload = encode(
      &CoreRequest {
        model: "gpt-4.1".to_string(),
        messages: vec![CoreMessage {
          role: CoreRole::User,
          content: vec![CoreContent::Text {
            text: "Summarize AFFiNE.".to_string(),
          }],
        }],
        stream: false,
        max_tokens: Some(64),
        temperature: None,
        tools: vec![],
        tool_choice: None,
        include: None,
        reasoning: None,
        response_schema: Some(json!({
          "type": "object",
          "properties": {
            "summary": { "type": "string" }
          },
          "required": ["summary"],
          "additionalProperties": false
        })),
      },
      false,
    );

    assert_eq!(payload["response_format"]["type"], "json_schema");
    assert_eq!(payload["response_format"]["json_schema"]["name"], "structured_output");
    assert_eq!(
      payload["response_format"]["json_schema"]["schema"]["required"],
      json!(["summary"])
    );
  }
}
