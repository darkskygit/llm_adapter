use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition, ProtocolError,
  core_role_to_string, get_str, parse_json, parse_text_or_array_content, stringify_json,
};

#[derive(Debug, Deserialize)]
pub(crate) struct OpenaiTool {
  #[serde(default)]
  function: Option<OpenaiFunctionDefinition>,
  #[serde(default)]
  name: Option<String>,
  #[serde(default)]
  description: Option<String>,
  #[serde(default)]
  parameters: Option<Value>,
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
    .filter_map(|tool| {
      if let Some(function) = tool.function {
        return Some(CoreToolDefinition {
          name: function.name,
          description: function.description,
          parameters: function.parameters,
        });
      }

      let name = tool.name?;
      Some(CoreToolDefinition {
        name,
        description: tool.description,
        parameters: tool
          .parameters
          .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
      })
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

fn tool_choice_from_core_for_responses(choice: Option<&CoreToolChoice>) -> Option<Value> {
  let choice = choice?;
  Some(match choice {
    CoreToolChoice::Mode(mode) => match mode {
      CoreToolChoiceMode::Auto => Value::String("auto".to_string()),
      CoreToolChoiceMode::None => Value::String("none".to_string()),
      CoreToolChoiceMode::Required => Value::String("required".to_string()),
    },
    CoreToolChoice::Specific { name } => {
      json!({ "type": "function", "name": name })
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

fn tool_definition_from_core_for_responses(tool: &CoreToolDefinition) -> Value {
  json!({
    "type": "function",
    "name": tool.name,
    "description": tool.description,
    "parameters": tool.parameters,
  })
}

fn to_openai_chat_image_source(source: &Value) -> Value {
  match source {
    Value::String(url) => json!({ "url": url }),
    Value::Object(object) => {
      if object.contains_key("url") {
        return Value::Object(object.clone());
      }
      if let Some(image_url) = object.get("image_url") {
        let mut normalized = Map::new();
        match image_url {
          Value::String(url) => {
            normalized.insert("url".to_string(), Value::String(url.clone()));
          }
          Value::Object(image_object) => {
            if let Some(url) = image_object.get("url") {
              normalized.insert("url".to_string(), url.clone());
            }
            for (key, value) in image_object {
              if key != "url" {
                normalized.insert(key.clone(), value.clone());
              }
            }
          }
          _ => {
            normalized.insert("url".to_string(), image_url.clone());
          }
        }
        for (key, value) in object {
          if key != "image_url" && key != "type" {
            normalized.insert(key.clone(), value.clone());
          }
        }
        return Value::Object(normalized);
      }
      Value::Object(object.clone())
    }
    _ => source.clone(),
  }
}

fn to_openai_responses_image_part(source: &Value) -> Value {
  let mut content = Map::new();
  content.insert("type".to_string(), Value::String("input_image".to_string()));

  match source {
    Value::String(url) => {
      content.insert("image_url".to_string(), Value::String(url.clone()));
    }
    Value::Object(object) => {
      for (key, value) in object {
        match key.as_str() {
          "type" => {}
          "url" => {
            content.insert("image_url".to_string(), value.clone());
          }
          "image_url" => {
            let normalized = if let Value::Object(image_object) = value {
              image_object.get("url").cloned().unwrap_or_else(|| value.clone())
            } else {
              value.clone()
            };
            content.insert("image_url".to_string(), normalized);
          }
          _ => {
            content.insert(key.clone(), value.clone());
          }
        }
      }
    }
    _ => {
      content.insert("image_url".to_string(), source.clone());
    }
  }

  Value::Object(content)
}

fn core_content_to_openai_content_part(content: &CoreContent, flavor: OpenaiRequestFlavor) -> Vec<Value> {
  match content {
    CoreContent::Text { text } => {
      let text_type = match flavor {
        OpenaiRequestFlavor::ChatCompletions => "text",
        OpenaiRequestFlavor::Responses => "input_text",
      };
      vec![json!({ "type": text_type, "text": text })]
    }
    CoreContent::Image { source } => match flavor {
      OpenaiRequestFlavor::ChatCompletions => {
        vec![json!({ "type": "image_url", "image_url": to_openai_chat_image_source(source) })]
      }
      OpenaiRequestFlavor::Responses => vec![to_openai_responses_image_part(source)],
    },
    _ => Vec::new(),
  }
}

fn core_message_to_openai(message: &CoreMessage, flavor: OpenaiRequestFlavor) -> Vec<Value> {
  let mut text_parts = Vec::new();
  let mut reasoning_parts = Vec::new();
  let mut image_parts = Vec::new();
  let mut tool_calls = Vec::new();
  let mut tool_results = Vec::new();

  for content in &message.content {
    match content {
      CoreContent::Text { text } => text_parts.push(text.clone()),
      CoreContent::Reasoning { text, .. } => reasoning_parts.push(text.clone()),
      CoreContent::Image { .. } => image_parts.extend(core_content_to_openai_content_part(content, flavor)),
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
      let text_type = match flavor {
        OpenaiRequestFlavor::ChatCompletions => "text",
        OpenaiRequestFlavor::Responses => "input_text",
      };
      merged.push(json!({ "type": text_type, "text": text_parts.join("") }));
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

pub(crate) fn messages_from_core(messages: &[CoreMessage], flavor: OpenaiRequestFlavor) -> Vec<Value> {
  messages
    .iter()
    .flat_map(|message| core_message_to_openai(message, flavor))
    .collect()
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
    let tools = match flavor {
      OpenaiRequestFlavor::ChatCompletions => request.tools.iter().map(tool_definition_from_core).collect(),
      OpenaiRequestFlavor::Responses => request
        .tools
        .iter()
        .map(tool_definition_from_core_for_responses)
        .collect(),
    };
    payload.insert("tools".to_string(), Value::Array(tools));
  }
  let tool_choice = match flavor {
    OpenaiRequestFlavor::ChatCompletions => tool_choice_from_core(request.tool_choice.as_ref()),
    OpenaiRequestFlavor::Responses => tool_choice_from_core_for_responses(request.tool_choice.as_ref()),
  };
  if let Some(tool_choice) = tool_choice {
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
      if let Some(effort) = get_str(reasoning, "effort") {
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
