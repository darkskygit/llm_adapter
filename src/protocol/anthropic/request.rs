use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition,
  ProtocolError,
  common::{AnthropicContentParseMode, core_content_to_anthropic, parse_content_blocks},
  core_role_to_string, get_str, get_u32, parse_role,
};

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
  model: String,
  #[serde(default)]
  messages: Vec<AnthropicMessage>,
  system: Option<Value>,
  max_tokens: Option<u32>,
  stream: Option<bool>,
  #[serde(default)]
  tools: Vec<AnthropicTool>,
  tool_choice: Option<Value>,
  thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
  role: String,
  content: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicTool {
  name: String,
  description: Option<String>,
  input_schema: Option<Value>,
  parameters: Option<Value>,
}

fn parse_tool_choice(value: Option<Value>) -> Result<Option<CoreToolChoice>, ProtocolError> {
  let Some(value) = value else {
    return Ok(None);
  };

  match value {
    Value::String(mode) => {
      let parsed = match mode.as_str() {
        "auto" => CoreToolChoice::Mode(CoreToolChoiceMode::Auto),
        "none" => CoreToolChoice::Mode(CoreToolChoiceMode::None),
        "required" | "any" => CoreToolChoice::Mode(CoreToolChoiceMode::Required),
        _ => {
          return Err(ProtocolError::InvalidValue {
            field: "tool_choice",
            message: format!("unsupported mode `{mode}`"),
          });
        }
      };
      Ok(Some(parsed))
    }
    Value::Object(object) => {
      if let Some(Value::String(name)) = object.get("name") {
        return Ok(Some(CoreToolChoice::Specific { name: name.clone() }));
      }
      if let Some(Value::String(name)) = object.get("tool_name") {
        return Ok(Some(CoreToolChoice::Specific { name: name.clone() }));
      }
      Err(ProtocolError::InvalidValue {
        field: "tool_choice",
        message: "unsupported object shape".to_string(),
      })
    }
    _ => Err(ProtocolError::InvalidValue {
      field: "tool_choice",
      message: "expected string or object".to_string(),
    }),
  }
}

fn core_tool_choice_to_anthropic(choice: Option<&CoreToolChoice>) -> Option<Value> {
  let choice = choice?;
  Some(match choice {
    CoreToolChoice::Mode(mode) => match mode {
      CoreToolChoiceMode::Auto => json!({ "type": "auto" }),
      CoreToolChoiceMode::None => json!({ "type": "none" }),
      CoreToolChoiceMode::Required => json!({ "type": "any" }),
    },
    CoreToolChoice::Specific { name } => json!({ "type": "tool", "name": name }),
  })
}

fn core_tool_to_anthropic(tool: &CoreToolDefinition) -> Value {
  json!({
    "name": tool.name,
    "description": tool.description,
    "input_schema": tool.parameters,
  })
}

fn core_message_to_anthropic(message: &CoreMessage) -> Value {
  let role = if message.role == CoreRole::Tool {
    "user"
  } else {
    core_role_to_string(&message.role)
  };
  let content = message
    .content
    .iter()
    .map(|content| core_content_to_anthropic(content, false))
    .collect::<Vec<_>>();
  json!({
    "role": role,
    "content": content,
  })
}

fn convert_effort_to_budget(effort: &str) -> Option<u32> {
  match effort.to_ascii_lowercase().as_str() {
    "low" => Some(2000),
    "medium" => Some(6000),
    "high" => Some(10000),
    _ => None,
  }
}

pub fn decode(request: &Value) -> Result<CoreRequest, ProtocolError> {
  let request: AnthropicRequest = serde_json::from_value(request.clone())?;

  let mut messages = Vec::new();
  if let Some(system) = request.system {
    let content = parse_content_blocks(system, AnthropicContentParseMode::Request)?;
    if !content.is_empty() {
      messages.push(CoreMessage {
        role: CoreRole::System,
        content,
      });
    }
  }

  for message in request.messages {
    let mut role = parse_role(&message.role, "role")?;
    let content = parse_content_blocks(message.content, AnthropicContentParseMode::Request)?;
    if content
      .iter()
      .any(|block| matches!(block, CoreContent::ToolResult { .. }))
    {
      role = CoreRole::Tool;
    }
    messages.push(CoreMessage { role, content });
  }

  Ok(CoreRequest {
    model: request.model,
    messages,
    stream: request.stream.unwrap_or(false),
    max_tokens: request.max_tokens,
    temperature: None,
    tools: request
      .tools
      .into_iter()
      .map(|tool| CoreToolDefinition {
        name: tool.name,
        description: tool.description,
        parameters: tool
          .input_schema
          .or(tool.parameters)
          .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
      })
      .collect(),
    tool_choice: parse_tool_choice(request.tool_choice)?,
    include: None,
    reasoning: request.thinking,
    response_schema: None,
  })
}

#[must_use]
pub fn encode(request: &CoreRequest, stream: bool) -> Value {
  let mut system_content = Vec::new();
  let mut messages = Vec::new();
  for message in &request.messages {
    if message.role == CoreRole::System {
      system_content.extend(
        message
          .content
          .iter()
          .map(|content| core_content_to_anthropic(content, false)),
      );
    } else {
      messages.push(core_message_to_anthropic(message));
    }
  }

  let mut payload = Map::from_iter([
    ("model".to_string(), Value::String(request.model.clone())),
    ("messages".to_string(), Value::Array(messages)),
    ("stream".to_string(), Value::Bool(stream)),
    (
      "max_tokens".to_string(),
      Value::Number((request.max_tokens.unwrap_or(4096)).into()),
    ),
  ]);

  if !system_content.is_empty() {
    payload.insert("system".to_string(), Value::Array(system_content));
  }
  if let Some(temperature) = request.temperature {
    payload.insert("temperature".to_string(), json!(temperature));
  }
  if !request.tools.is_empty() {
    payload.insert(
      "tools".to_string(),
      Value::Array(request.tools.iter().map(core_tool_to_anthropic).collect()),
    );
  }
  if let Some(tool_choice) = core_tool_choice_to_anthropic(request.tool_choice.as_ref()) {
    payload.insert("tool_choice".to_string(), tool_choice);
  }
  if let Some(reasoning) = &request.reasoning {
    let budget_tokens =
      get_u32(reasoning, "budget_tokens").or_else(|| get_str(reasoning, "effort").and_then(convert_effort_to_budget));
    if let Some(budget_tokens) = budget_tokens {
      payload.insert(
        "thinking".to_string(),
        json!({
          "type": "enabled",
          "budget_tokens": budget_tokens,
        }),
      );
    }
  }

  Value::Object(payload)
}

#[cfg(test)]
mod tests {
  use serde_json::{Value, json};

  use super::*;

  #[test]
  fn decode_should_match_golden() {
    let input = json!({
      "model": "claude-sonnet-4-5-20250929",
      "system": "You are a reviewer.",
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Summarize this change." },
            { "type": "thinking", "thinking": "break the task down", "signature": "sig_1" }
          ]
        },
        {
          "role": "assistant",
          "content": [
            { "type": "tool_use", "id": "call_3", "name": "doc_read", "input": { "docId": "xyz" } }
          ]
        },
        {
          "role": "assistant",
          "content": [
            { "type": "tool_result", "tool_use_id": "call_3", "content": { "text": "done" } }
          ]
        }
      ],
      "max_tokens": 1024,
      "stream": true
    });

    let core = decode(&input).unwrap();
    let actual = serde_json::to_value(core).unwrap();

    let expected = json!({
      "model": "claude-sonnet-4-5-20250929",
      "messages": [
        {
          "role": "system",
          "content": [{ "type": "text", "text": "You are a reviewer." }]
        },
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Summarize this change." },
            { "type": "reasoning", "text": "break the task down", "signature": "sig_1" }
          ]
        },
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_call",
              "call_id": "call_3",
              "name": "doc_read",
              "arguments": { "docId": "xyz" }
            }
          ]
        },
        {
          "role": "tool",
          "content": [
            {
              "type": "tool_result",
              "call_id": "call_3",
              "output": { "text": "done" }
            }
          ]
        }
      ],
      "stream": true,
      "max_tokens": 1024
    });

    assert_eq!(actual, expected);
  }

  #[test]
  fn encode_should_apply_anthropic_defaults() {
    let request = CoreRequest {
      model: "claude-sonnet-4-5-20250929".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text {
          text: "hello".to_string(),
        }],
      }],
      stream: false,
      max_tokens: None,
      temperature: Some(0.4),
      tools: vec![],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    let payload = encode(&request, false);
    assert_eq!(payload["max_tokens"], 4096);
    assert_eq!(payload["temperature"], 0.4);
  }

  #[test]
  fn encode_should_drop_tool_call_thought() {
    let request = request_with_single_content(
      CoreRole::Assistant,
      CoreContent::ToolCall {
        call_id: "call_1".to_string(),
        name: "doc_read".to_string(),
        arguments: json!({ "docId": "a1" }),
        thought: Some("internal chain of thought".to_string()),
      },
    );

    let payload = encode(&request, false);
    let tool_use = &payload["messages"][0]["content"][0];

    assert_eq!(tool_use["type"], "tool_use");
    assert!(tool_use.get("thought").is_none());
  }

  #[test]
  fn encode_should_normalize_tool_result_content_cases() {
    struct Case {
      name: &'static str,
      output: Value,
      is_error: Option<bool>,
      expected_content: Value,
      expected_is_error: Option<bool>,
    }

    let cases = vec![
      Case {
        name: "drop_null_is_error_and_stringify_object_output",
        output: json!({ "ok": true }),
        is_error: None,
        expected_content: json!([{ "type": "text", "text": "{\"ok\":true}" }]),
        expected_is_error: None,
      },
      Case {
        name: "normalize_structured_output_to_text_block",
        output: json!([{ "ok": true }]),
        is_error: Some(false),
        expected_content: json!([{ "type": "text", "text": "[{\"ok\":true}]" }]),
        expected_is_error: Some(false),
      },
      Case {
        name: "preserve_typed_content_blocks",
        output: json!([{ "type": "text", "text": "done" }]),
        is_error: Some(false),
        expected_content: json!([{ "type": "text", "text": "done" }]),
        expected_is_error: Some(false),
      },
    ];

    for case in cases {
      let request = request_with_single_content(
        CoreRole::Tool,
        CoreContent::ToolResult {
          call_id: "call_1".to_string(),
          output: case.output,
          is_error: case.is_error,
        },
      );

      let payload = encode(&request, false);
      let tool_result = &payload["messages"][0]["content"][0];

      assert_eq!(tool_result["type"], "tool_result", "{}", case.name);
      assert_eq!(tool_result["content"], case.expected_content, "{}", case.name);
      match case.expected_is_error {
        Some(is_error) => assert_eq!(tool_result["is_error"], is_error, "{}", case.name),
        None => assert!(tool_result.get("is_error").is_none(), "{}", case.name),
      }
    }
  }

  #[test]
  fn encode_should_normalize_image_source_cases() {
    struct Case {
      name: &'static str,
      source: Value,
      expected_source: Value,
    }

    let cases = vec![
      Case {
        name: "normalize_data_url_source",
        source: json!({ "url": "data:image/png;base64,Zm9v" }),
        expected_source: json!({
          "type": "base64",
          "media_type": "image/png",
          "data": "Zm9v"
        }),
      },
      Case {
        name: "preserve_typed_source",
        source: json!({
          "type": "base64",
          "media_type": "image/png",
          "data": "Zm9v"
        }),
        expected_source: json!({
          "type": "base64",
          "media_type": "image/png",
          "data": "Zm9v"
        }),
      },
      Case {
        name: "correct_mismatched_media_type",
        source: json!({ "url": "data:image/png;base64,/9j/4AAQSkZJRg==" }),
        expected_source: json!({
          "type": "base64",
          "media_type": "image/jpeg",
          "data": "/9j/4AAQSkZJRg=="
        }),
      },
    ];

    for case in cases {
      let request = request_with_single_content(CoreRole::User, CoreContent::Image { source: case.source });
      let payload = encode(&request, false);
      let source = &payload["messages"][0]["content"][0]["source"];

      assert_eq!(*source, case.expected_source, "{}", case.name);
    }
  }

  #[test]
  fn encode_should_emit_document_blocks_for_file_content() {
    let request = request_with_single_content(
      CoreRole::User,
      CoreContent::File {
        source: json!({
          "url": "https://example.com/manual.pdf",
          "media_type": "application/pdf"
        }),
      },
    );

    let payload = encode(&request, false);

    assert_eq!(payload["messages"][0]["content"][0]["type"], "document");
    assert_eq!(
      payload["messages"][0]["content"][0]["source"]["url"],
      "https://example.com/manual.pdf"
    );
  }

  #[test]
  fn decode_should_preserve_document_blocks() {
    let core = decode(&json!({
      "model": "claude-sonnet-4-5-20250929",
      "messages": [{
        "role": "user",
        "content": [{
          "type": "document",
          "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "Zm9v"
          }
        }]
      }]
    }))
    .unwrap();

    assert!(matches!(
      &core.messages[0].content[0],
      CoreContent::File { source } if source["media_type"] == "application/pdf" && source["data"] == "Zm9v"
    ));
  }

  fn request_with_single_content(role: CoreRole, content: CoreContent) -> CoreRequest {
    CoreRequest {
      model: "claude-sonnet-4-5-20250929".to_string(),
      messages: vec![CoreMessage {
        role,
        content: vec![content],
      }],
      stream: false,
      max_tokens: Some(128),
      temperature: None,
      tools: vec![],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    }
  }
}
