use base64::{Engine as _, engine::general_purpose};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition,
  ProtocolError, core_role_to_string, get_first_str, get_str, get_str_or, get_u32, parse_role, stringify_json,
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

fn parse_content_blocks(value: Value) -> Result<Vec<CoreContent>, ProtocolError> {
  match value {
    Value::String(text) => Ok(vec![CoreContent::Text { text }]),
    Value::Array(items) => {
      let mut content = Vec::new();
      for item in items {
        if !item.is_object() {
          continue;
        }

        let typ = get_str_or(&item, "type", "text");
        match typ {
          "text" => {
            if let Some(text) = get_str(&item, "text") {
              content.push(CoreContent::Text { text: text.to_string() });
            }
          }
          "thinking" => {
            let text = get_first_str(&item, &["thinking", "text"])
              .unwrap_or_default()
              .to_string();
            let signature = get_str(&item, "signature").map(ToString::to_string);
            content.push(CoreContent::Reasoning { text, signature });
          }
          "tool_use" => {
            let call_id = get_str(&item, "id")
              .ok_or(ProtocolError::MissingField("content[].id"))?
              .to_string();
            let name = get_str(&item, "name")
              .ok_or(ProtocolError::MissingField("content[].name"))?
              .to_string();
            let arguments = item.get("input").cloned().unwrap_or(Value::Null);
            let thought = get_str(&item, "thought").map(ToString::to_string);
            content.push(CoreContent::ToolCall {
              call_id,
              name,
              arguments,
              thought,
            });
          }
          "tool_result" => {
            let call_id = get_str(&item, "tool_use_id")
              .ok_or(ProtocolError::MissingField("content[].tool_use_id"))?
              .to_string();
            let output = item.get("content").cloned().unwrap_or(Value::Null);
            let is_error = item.get("is_error").and_then(Value::as_bool);
            content.push(CoreContent::ToolResult {
              call_id,
              output,
              is_error,
            });
          }
          "image" => {
            let source = item.get("source").cloned().unwrap_or_else(|| item.clone());
            content.push(CoreContent::Image { source });
          }
          _ => {}
        }
      }
      Ok(content)
    }
    _ => Err(ProtocolError::InvalidValue {
      field: "content",
      message: "expected string or array".to_string(),
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

fn parse_base64_data_url(url: &str) -> Option<(String, String)> {
  let data_url = url.strip_prefix("data:")?;
  let (meta, payload) = data_url.split_once(',')?;
  let mut meta_parts = meta.split(';');
  let media_type = meta_parts.next().unwrap_or_default();
  let is_base64 = meta_parts.any(|part| part.eq_ignore_ascii_case("base64"));
  if !is_base64 {
    return None;
  }

  let media_type = if media_type.is_empty() {
    "application/octet-stream".to_string()
  } else {
    media_type.to_string()
  };
  Some((media_type, payload.to_string()))
}

fn infer_image_media_type_from_base64_data(data: &str) -> Option<&'static str> {
  let prefix_len = data.len().min(256);
  let usable_len = prefix_len - (prefix_len % 4);
  if usable_len == 0 {
    return None;
  }
  let prefix = &data[..usable_len];
  let decoded = general_purpose::STANDARD
    .decode(prefix)
    .or_else(|_| general_purpose::URL_SAFE.decode(prefix))
    .ok()?;

  if decoded.starts_with(&[0xFF, 0xD8, 0xFF]) {
    return Some("image/jpeg");
  }
  if decoded.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']) {
    return Some("image/png");
  }
  if decoded.starts_with(b"GIF87a") || decoded.starts_with(b"GIF89a") {
    return Some("image/gif");
  }
  if decoded.len() >= 12 && &decoded[..4] == b"RIFF" && &decoded[8..12] == b"WEBP" {
    return Some("image/webp");
  }
  None
}

fn normalize_image_url_source(url: &str) -> Value {
  if let Some((media_type, data)) = parse_base64_data_url(url) {
    let normalized_media_type = infer_image_media_type_from_base64_data(&data)
      .map(ToString::to_string)
      .unwrap_or(media_type);
    json!({ "type": "base64", "media_type": normalized_media_type, "data": data })
  } else {
    json!({ "type": "url", "url": url })
  }
}

fn normalize_image_source_to_anthropic(source: &Value) -> Value {
  match source {
    Value::Object(object) => {
      if object.get("type").is_some() {
        return source.clone();
      }
      if let (Some(media_type), Some(data)) = (
        object.get("media_type").and_then(Value::as_str),
        object.get("data").and_then(Value::as_str),
      ) {
        return json!({ "type": "base64", "media_type": media_type, "data": data });
      }
      if let Some(url) = object.get("url").and_then(Value::as_str) {
        return normalize_image_url_source(url);
      }
      source.clone()
    }
    Value::String(url) => normalize_image_url_source(url),
    _ => source.clone(),
  }
}

fn is_anthropic_content_block(value: &Value) -> bool {
  value.get("type").and_then(Value::as_str).is_some()
}

fn tool_result_content_to_anthropic(output: &Value) -> Value {
  match output {
    Value::String(_) => output.clone(),
    Value::Array(items) if items.iter().all(is_anthropic_content_block) => output.clone(),
    _ => Value::Array(vec![json!({
      "type": "text",
      "text": stringify_json(output),
    })]),
  }
}

fn core_content_to_anthropic(content: &CoreContent) -> Value {
  match content {
    CoreContent::Text { text } => json!({
      "type": "text",
      "text": text,
    }),
    CoreContent::Reasoning { text, signature } => json!({
      "type": "thinking",
      "thinking": text,
      "signature": signature,
    }),
    CoreContent::ToolCall {
      call_id,
      name,
      arguments,
      thought: _,
    } => json!({
      "type": "tool_use",
      "id": call_id,
      "name": name,
      "input": arguments,
    }),
    CoreContent::ToolResult {
      call_id,
      output,
      is_error,
    } => {
      let mut block = Map::from_iter([
        ("type".to_string(), Value::String("tool_result".to_string())),
        ("tool_use_id".to_string(), Value::String(call_id.clone())),
        ("content".to_string(), tool_result_content_to_anthropic(output)),
      ]);
      if let Some(is_error) = is_error {
        block.insert("is_error".to_string(), Value::Bool(*is_error));
      }
      Value::Object(block)
    }
    CoreContent::Image { source } => json!({
      "type": "image",
      "source": normalize_image_source_to_anthropic(source),
    }),
  }
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
    .map(core_content_to_anthropic)
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
    let content = parse_content_blocks(system)?;
    if !content.is_empty() {
      messages.push(CoreMessage {
        role: CoreRole::System,
        content,
      });
    }
  }

  for message in request.messages {
    let mut role = parse_role(&message.role, "role")?;
    let content = parse_content_blocks(message.content)?;
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
  })
}

#[must_use]
pub fn encode(request: &CoreRequest, stream: bool) -> Value {
  let mut system_content = Vec::new();
  let mut messages = Vec::new();
  for message in &request.messages {
    if message.role == CoreRole::System {
      system_content.extend(message.content.iter().map(core_content_to_anthropic));
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
    }
  }
}
