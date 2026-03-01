use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition, ProtocolError,
  core_role_to_string, parse_json, parse_text_or_array_content, stringify_json,
};

#[derive(Debug, Deserialize)]
pub(crate) struct OpenaiTool {
  function: OpenaiFunctionDefinition,
}

#[derive(Debug, Deserialize)]
struct OpenaiFunctionDefinition {
  name: String,
  description: Option<String>,
  parameters: Value,
}

#[derive(Debug, Deserialize)]
struct OpenaiToolCall {
  id: Option<String>,
  call_id: Option<String>,
  function: OpenaiFunctionCall,
  thought: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenaiFunctionCall {
  name: String,
  arguments: Value,
}

pub(crate) fn openai_tools_to_core(tools: Vec<OpenaiTool>) -> Vec<CoreToolDefinition> {
  tools
    .into_iter()
    .map(|tool| CoreToolDefinition {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
    })
    .collect()
}

pub(crate) fn parse_tool_calls(raw_tool_calls: &Value, field: &'static str) -> Result<Vec<CoreContent>, ProtocolError> {
  let tool_calls: Vec<OpenaiToolCall> = serde_json::from_value(raw_tool_calls.clone())?;
  let mut content = Vec::with_capacity(tool_calls.len());

  for tool_call in tool_calls {
    let call_id = tool_call
      .id
      .or(tool_call.call_id)
      .ok_or(ProtocolError::MissingField(field))?;
    content.push(CoreContent::ToolCall {
      call_id,
      name: tool_call.function.name,
      arguments: parse_json(tool_call.function.arguments),
      thought: tool_call.thought,
    });
  }

  Ok(content)
}

pub(crate) fn tool_result_content(call_id: String, output: Option<Value>) -> CoreContent {
  CoreContent::ToolResult {
    call_id,
    output: parse_json(output.unwrap_or(Value::String(String::new()))),
    is_error: None,
  }
}

pub(crate) fn tool_choice_from_core(choice: Option<&CoreToolChoice>) -> Option<Value> {
  let choice = choice?;
  Some(match choice {
    CoreToolChoice::Mode(mode) => match mode {
      CoreToolChoiceMode::Auto => Value::String("auto".to_string()),
      CoreToolChoiceMode::None => Value::String("none".to_string()),
      CoreToolChoiceMode::Required => Value::String("required".to_string()),
    },
    CoreToolChoice::Specific { name } => {
      json!({ "type": "function", "function": { "name": name } })
    }
  })
}

pub(crate) fn tool_definition_from_core(tool: &CoreToolDefinition) -> Value {
  json!({
    "type": "function",
    "function": {
      "name": tool.name,
      "description": tool.description,
      "parameters": tool.parameters,
    }
  })
}

fn core_content_to_openai_content_part(content: &CoreContent) -> Vec<Value> {
  match content {
    CoreContent::Text { text } => vec![json!({ "type": "text", "text": text })],
    CoreContent::Image { source } => vec![json!({ "type": "image_url", "image_url": source })],
    _ => Vec::new(),
  }
}

fn core_message_to_openai(message: &CoreMessage) -> Vec<Value> {
  let mut text_parts = Vec::new();
  let mut reasoning_parts = Vec::new();
  let mut image_parts = Vec::new();
  let mut tool_calls = Vec::new();
  let mut tool_results = Vec::new();

  for content in &message.content {
    match content {
      CoreContent::Text { text } => text_parts.push(text.clone()),
      CoreContent::Reasoning { text, .. } => reasoning_parts.push(text.clone()),
      CoreContent::Image { .. } => image_parts.extend(core_content_to_openai_content_part(content)),
      CoreContent::ToolCall {
        call_id,
        name,
        arguments,
        thought,
      } => {
        tool_calls.push(json!({
          "id": call_id,
          "type": "function",
          "function": {
            "name": name,
            "arguments": stringify_json(arguments),
          },
          "thought": thought,
        }));
      }
      CoreContent::ToolResult {
        call_id,
        output,
        is_error,
      } => {
        tool_results.push((call_id.clone(), output.clone(), *is_error));
      }
    }
  }

  if tool_results.len() > 1 {
    return tool_results
      .into_iter()
      .map(|(call_id, output, is_error)| {
        json!({
          "role": "tool",
          "tool_call_id": call_id,
          "content": stringify_json(&output),
          "is_error": is_error,
        })
      })
      .collect();
  }

  let role = if !tool_results.is_empty() {
    "tool"
  } else {
    core_role_to_string(&message.role)
  };

  let content = if !image_parts.is_empty() {
    let mut merged = Vec::new();
    if !text_parts.is_empty() {
      merged.push(json!({ "type": "text", "text": text_parts.join("") }));
    }
    merged.extend(image_parts);
    Value::Array(merged)
  } else if let Some((_call_id, output, _)) = tool_results.first() {
    Value::String(stringify_json(output))
  } else if !text_parts.is_empty() {
    Value::String(text_parts.join(""))
  } else {
    Value::Null
  };

  let mut object = Map::from_iter([
    ("role".to_string(), Value::String(role.to_string())),
    ("content".to_string(), content),
  ]);

  if let Some((call_id, _, _)) = tool_results.first() {
    object.insert("tool_call_id".to_string(), Value::String(call_id.clone()));
  }

  if !reasoning_parts.is_empty() {
    object.insert("reasoning_content".to_string(), Value::String(reasoning_parts.join("")));
  }

  if !tool_calls.is_empty() {
    object.insert("tool_calls".to_string(), Value::Array(tool_calls));
  }

  vec![Value::Object(object)]
}

pub(crate) fn messages_from_core(messages: &[CoreMessage]) -> Vec<Value> {
  messages.iter().flat_map(core_message_to_openai).collect()
}

pub(crate) fn parse_tool_choice(value: Option<Value>) -> Result<Option<CoreToolChoice>, ProtocolError> {
  let Some(value) = value else {
    return Ok(None);
  };

  match value {
    Value::String(mode) => {
      let parsed = match mode.as_str() {
        "auto" => CoreToolChoice::Mode(CoreToolChoiceMode::Auto),
        "none" => CoreToolChoice::Mode(CoreToolChoiceMode::None),
        "required" => CoreToolChoice::Mode(CoreToolChoiceMode::Required),
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
      if let Some(Value::Object(function)) = object.get("function")
        && let Some(Value::String(name)) = function.get("name")
      {
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

pub(crate) fn parse_content(value: Option<Value>) -> Result<Vec<CoreContent>, ProtocolError> {
  parse_text_or_array_content(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenaiRequestFlavor {
  ChatCompletions,
  Responses,
}

pub(crate) struct OpenaiDecodeRequestInput {
  pub model: String,
  pub messages: Vec<CoreMessage>,
  pub stream: Option<bool>,
  pub max_tokens: Option<u32>,
  pub temperature: Option<f64>,
  pub tools: Vec<OpenaiTool>,
  pub tool_choice: Option<Value>,
  pub include: Option<Vec<String>>,
  pub reasoning: Option<Value>,
  pub reasoning_effort: Option<String>,
}

pub(crate) fn decode_openai_request(input: OpenaiDecodeRequestInput) -> Result<CoreRequest, ProtocolError> {
  Ok(CoreRequest {
    model: input.model,
    messages: input.messages,
    stream: input.stream.unwrap_or(false),
    max_tokens: input.max_tokens,
    temperature: input.temperature,
    tools: openai_tools_to_core(input.tools),
    tool_choice: parse_tool_choice(input.tool_choice)?,
    include: input.include,
    reasoning: input
      .reasoning
      .or_else(|| input.reasoning_effort.map(|effort| json!({ "effort": effort }))),
  })
}

pub(crate) fn encode_openai_request(
  request: &CoreRequest,
  stream: bool,
  flavor: OpenaiRequestFlavor,
  input: Vec<Value>,
) -> Value {
  let (input_key, max_tokens_key) = match flavor {
    OpenaiRequestFlavor::ChatCompletions => ("messages", "max_completion_tokens"),
    OpenaiRequestFlavor::Responses => ("input", "max_output_tokens"),
  };

  let mut payload = Map::from_iter([
    ("model".to_string(), Value::String(request.model.clone())),
    (input_key.to_string(), Value::Array(input)),
    ("stream".to_string(), Value::Bool(stream)),
  ]);

  if stream && flavor == OpenaiRequestFlavor::ChatCompletions {
    payload.insert("stream_options".to_string(), json!({ "include_usage": true }));
  }
  if let Some(max_tokens) = request.max_tokens {
    payload.insert(max_tokens_key.to_string(), json!(max_tokens));
  }
  if let Some(temperature) = request.temperature
    && flavor == OpenaiRequestFlavor::ChatCompletions
  {
    payload.insert("temperature".to_string(), json!(temperature));
  }
  if !request.tools.is_empty() {
    payload.insert(
      "tools".to_string(),
      Value::Array(request.tools.iter().map(tool_definition_from_core).collect()),
    );
  }
  if let Some(tool_choice) = tool_choice_from_core(request.tool_choice.as_ref()) {
    payload.insert("tool_choice".to_string(), tool_choice);
  }
  if let Some(include) = &request.include {
    payload.insert(
      "include".to_string(),
      Value::Array(include.iter().map(|item| Value::String(item.clone())).collect()),
    );
  }
  if let Some(reasoning) = &request.reasoning {
    if flavor == OpenaiRequestFlavor::ChatCompletions {
      if let Some(effort) = reasoning.get("effort").and_then(Value::as_str) {
        payload.insert("reasoning_effort".to_string(), Value::String(effort.to_string()));
      } else {
        payload.insert("reasoning".to_string(), reasoning.clone());
      }
    } else {
      payload.insert("reasoning".to_string(), reasoning.clone());
    }
  }

  Value::Object(payload)
}
