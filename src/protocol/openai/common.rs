use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreAttachmentKind, CoreContent, CoreMessage, CoreRequest, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition,
  CoreUsage, ProtocolError, attachment_content_from_source, attachment_source, core_role_to_string, get_str,
  parse_json, stringify_json,
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

pub(crate) fn parse_message_content(
  content: Option<Value>,
  tool_call_id: Option<String>,
  raw_tool_calls: Option<Value>,
  tool_call_field: &'static str,
) -> Result<Vec<CoreContent>, ProtocolError> {
  if let Some(call_id) = tool_call_id {
    return Ok(vec![tool_result_content(call_id, content)]);
  }

  let mut content = parse_content(content)?;
  if let Some(raw_tool_calls) = raw_tool_calls {
    content.extend(parse_tool_calls(&raw_tool_calls, tool_call_field)?);
  }
  Ok(content)
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

fn parse_data_url(url: &str) -> Option<(&str, &str)> {
  let data_url = url.strip_prefix("data:")?;
  let (meta, data) = data_url.split_once(',')?;
  let media_type = meta.split(';').next().unwrap_or_default();
  Some((media_type, data))
}

fn audio_format_from_media_type(media_type: &str) -> Option<&'static str> {
  match media_type
    .split(';')
    .next()
    .unwrap_or(media_type)
    .trim()
    .to_ascii_lowercase()
    .as_str()
  {
    "audio/wav" | "audio/x-wav" => Some("wav"),
    "audio/mpeg" | "audio/mp3" => Some("mp3"),
    "audio/ogg" => Some("ogg"),
    "audio/flac" => Some("flac"),
    _ => None,
  }
}

fn to_openai_chat_audio_source(source: &Value) -> Option<Value> {
  match source {
    Value::Object(object) => {
      if let (Some(Value::String(data)), Some(format)) = (
        object.get("data"),
        object.get("format").and_then(Value::as_str).or_else(|| {
          object
            .get("media_type")
            .and_then(Value::as_str)
            .and_then(audio_format_from_media_type)
        }),
      ) {
        return Some(json!({ "data": data, "format": format }));
      }

      if let Some(Value::String(url)) = object.get("url")
        && let Some((media_type, data)) = parse_data_url(url)
        && let Some(format) = audio_format_from_media_type(media_type)
      {
        return Some(json!({ "data": data, "format": format }));
      }

      None
    }
    Value::String(url) => {
      let (media_type, data) = parse_data_url(url)?;
      let format = audio_format_from_media_type(media_type)?;
      Some(json!({ "data": data, "format": format }))
    }
    _ => None,
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

fn to_openai_responses_file_part(source: &Value) -> Value {
  let mut content = Map::new();
  content.insert("type".to_string(), Value::String("input_file".to_string()));

  match source {
    Value::String(url) => {
      content.insert("file_url".to_string(), Value::String(url.clone()));
    }
    Value::Object(object) => {
      for (key, value) in object {
        match key.as_str() {
          "type" => {}
          "url" => {
            content.insert("file_url".to_string(), value.clone());
          }
          "data" => {
            content.insert("file_data".to_string(), value.clone());
          }
          "media_type" => {
            content.insert("media_type".to_string(), value.clone());
          }
          _ => {
            content.insert(key.clone(), value.clone());
          }
        }
      }
    }
    _ => {
      content.insert("file_url".to_string(), source.clone());
    }
  }

  Value::Object(content)
}

fn image_source_from_openai_item(object: &serde_json::Map<String, Value>) -> Value {
  let mut source = serde_json::Map::new();
  if let Some(image_url) = object.get("image_url") {
    match image_url {
      Value::String(url) => {
        source.insert("url".to_string(), Value::String(url.clone()));
      }
      Value::Object(image_url_object) => {
        if let Some(url) = image_url_object.get("url") {
          source.insert("url".to_string(), url.clone());
        }
        for (key, value) in image_url_object {
          if key != "url" {
            source.insert(key.clone(), value.clone());
          }
        }
      }
      _ => {
        source.insert("image_url".to_string(), image_url.clone());
      }
    }
  }
  if let Some(file_id) = object.get("file_id") {
    source.insert("file_id".to_string(), file_id.clone());
  }
  if let Some(detail) = object.get("detail") {
    source.insert("detail".to_string(), detail.clone());
  }
  if source.is_empty() {
    for (key, value) in object {
      if key != "type" {
        source.insert(key.clone(), value.clone());
      }
    }
  }
  Value::Object(source)
}

fn audio_source_from_openai_item(object: &serde_json::Map<String, Value>) -> Value {
  let mut source = serde_json::Map::new();
  if let Some(Value::Object(audio)) = object.get("input_audio").or_else(|| object.get("audio")) {
    if let Some(data) = audio.get("data") {
      source.insert("data".to_string(), data.clone());
    }
    if let Some(format) = audio.get("format").and_then(Value::as_str) {
      let media_type = match format {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        other => other,
      };
      source.insert("format".to_string(), Value::String(format.to_string()));
      source.insert("media_type".to_string(), Value::String(media_type.to_string()));
    }
  }
  if let Some(Value::String(url)) = object.get("audio_url").or_else(|| object.get("url")) {
    source.insert("url".to_string(), Value::String(url.clone()));
  }
  if source.is_empty() {
    for (key, value) in object {
      if key != "type" {
        source.insert(key.clone(), value.clone());
      }
    }
  }
  Value::Object(source)
}

fn file_source_from_openai_item(object: &serde_json::Map<String, Value>) -> Value {
  let mut source = serde_json::Map::new();
  if let Some(file_url) = object.get("file_url").or_else(|| object.get("url")) {
    source.insert("url".to_string(), file_url.clone());
  }
  if let Some(file_data) = object.get("file_data") {
    source.insert("data".to_string(), file_data.clone());
  }
  if let Some(file_id) = object.get("file_id") {
    source.insert("file_id".to_string(), file_id.clone());
  }
  if let Some(filename) = object.get("filename") {
    source.insert("filename".to_string(), filename.clone());
  }
  if let Some(detail) = object.get("detail") {
    source.insert("detail".to_string(), detail.clone());
  }
  if let Some(media_type) = object
    .get("media_type")
    .or_else(|| object.get("mime_type"))
    .or_else(|| object.get("mimeType"))
  {
    source.insert("media_type".to_string(), media_type.clone());
  }
  if source.is_empty() {
    for (key, value) in object {
      if key != "type" {
        source.insert(key.clone(), value.clone());
      }
    }
  }
  Value::Object(source)
}

fn parse_text_or_array_content(value: Option<Value>) -> Result<Vec<CoreContent>, ProtocolError> {
  let Some(value) = value else {
    return Ok(Vec::new());
  };

  match value {
    Value::String(text) => Ok(vec![CoreContent::Text { text }]),
    Value::Array(items) => {
      let mut content = Vec::new();
      for item in items {
        match item {
          Value::String(text) => content.push(CoreContent::Text { text }),
          Value::Object(object) => {
            if let Some(Value::String(typ)) = object.get("type") {
              match typ.as_str() {
                "text" | "input_text" | "output_text" => {
                  if let Some(Value::String(text)) = object.get("text") {
                    content.push(CoreContent::Text { text: text.clone() });
                  }
                }
                "image_url" => {
                  let source = match object.get("image_url") {
                    Some(Value::String(url)) => {
                      let mut source = serde_json::Map::new();
                      source.insert("url".to_string(), Value::String(url.clone()));
                      Value::Object(source)
                    }
                    Some(value) => value.clone(),
                    None => Value::Object(object.clone()),
                  };
                  content.push(attachment_content_from_source(source, CoreAttachmentKind::Image));
                }
                "input_image" => {
                  let source = image_source_from_openai_item(&object);
                  content.push(attachment_content_from_source(source, CoreAttachmentKind::Image));
                }
                "image" => {
                  let source = object
                    .get("source")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(object.clone()));
                  content.push(attachment_content_from_source(source, CoreAttachmentKind::Image));
                }
                "input_audio" | "audio" => {
                  let source = if typ == "audio" {
                    object
                      .get("source")
                      .cloned()
                      .unwrap_or_else(|| audio_source_from_openai_item(&object))
                  } else {
                    audio_source_from_openai_item(&object)
                  };
                  content.push(attachment_content_from_source(source, CoreAttachmentKind::Audio));
                }
                "input_file" | "file" | "document" => {
                  let source = if matches!(typ.as_str(), "file" | "document") {
                    object
                      .get("source")
                      .cloned()
                      .unwrap_or_else(|| file_source_from_openai_item(&object))
                  } else {
                    file_source_from_openai_item(&object)
                  };
                  content.push(attachment_content_from_source(source, CoreAttachmentKind::File));
                }
                _ => {}
              }
            } else if let Some(Value::String(text)) = object.get("text") {
              content.push(CoreContent::Text { text: text.clone() });
            }
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
    CoreContent::Audio { source } => match flavor {
      OpenaiRequestFlavor::ChatCompletions => to_openai_chat_audio_source(source)
        .map(|input_audio| vec![json!({ "type": "input_audio", "input_audio": input_audio })])
        .unwrap_or_default(),
      OpenaiRequestFlavor::Responses => Vec::new(),
    },
    CoreContent::File { source } => match flavor {
      OpenaiRequestFlavor::ChatCompletions => Vec::new(),
      OpenaiRequestFlavor::Responses => vec![to_openai_responses_file_part(source)],
    },
    _ => Vec::new(),
  }
}

fn core_message_to_openai(message: &CoreMessage, flavor: OpenaiRequestFlavor) -> Vec<Value> {
  let mut text_parts = Vec::new();
  let mut reasoning_parts = Vec::new();
  let mut attachment_parts = Vec::new();
  let mut tool_calls = Vec::new();
  let mut tool_results = Vec::new();

  for content in &message.content {
    match content {
      CoreContent::Text { text } => text_parts.push(text.clone()),
      CoreContent::Reasoning { text, .. } => reasoning_parts.push(text.clone()),
      CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => {
        if attachment_source(content).is_some() {
          attachment_parts.extend(core_content_to_openai_content_part(content, flavor));
        }
      }
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

  let content = if !attachment_parts.is_empty() {
    let mut merged = Vec::new();
    if !text_parts.is_empty() {
      let text_type = match flavor {
        OpenaiRequestFlavor::ChatCompletions => "text",
        OpenaiRequestFlavor::Responses => "input_text",
      };
      merged.push(json!({ "type": text_type, "text": text_parts.join("") }));
    }
    merged.extend(attachment_parts);
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

pub(crate) fn usage_to_openai_json(usage: &CoreUsage, flavor: OpenaiRequestFlavor) -> Value {
  let mut object = match flavor {
    OpenaiRequestFlavor::ChatCompletions => Map::from_iter([
      ("prompt_tokens".to_string(), json!(usage.prompt_tokens)),
      ("completion_tokens".to_string(), json!(usage.completion_tokens)),
      ("total_tokens".to_string(), json!(usage.total_tokens)),
    ]),
    OpenaiRequestFlavor::Responses => Map::from_iter([
      ("input_tokens".to_string(), json!(usage.prompt_tokens)),
      ("output_tokens".to_string(), json!(usage.completion_tokens)),
      ("total_tokens".to_string(), json!(usage.total_tokens)),
    ]),
  };

  if let Some(cached_tokens) = usage.cached_tokens {
    object.insert(
      "input_tokens_details".to_string(),
      json!({ "cached_tokens": cached_tokens }),
    );
    if flavor == OpenaiRequestFlavor::ChatCompletions {
      object.insert(
        "prompt_tokens_details".to_string(),
        json!({ "cached_tokens": cached_tokens }),
      );
    }
  }

  Value::Object(object)
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
    response_schema: None,
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
  if let Some(response_schema) = &request.response_schema {
    match flavor {
      OpenaiRequestFlavor::ChatCompletions => {
        payload.insert(
          "response_format".to_string(),
          json!({
            "type": "json_schema",
            "json_schema": {
              "name": "structured_output",
              "strict": true,
              "schema": response_schema,
            }
          }),
        );
      }
      OpenaiRequestFlavor::Responses => {
        payload.insert(
          "text".to_string(),
          json!({
            "format": {
              "name": "structured_output",
              "type": "json_schema",
              "strict": true,
              "schema": response_schema,
            }
          }),
        );
      }
    }
  }

  Value::Object(payload)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn should_parse_openai_audio_and_file_content() {
    let content = parse_text_or_array_content(Some(json!([
      {
        "type": "input_audio",
        "input_audio": {
          "data": "Zm9v",
          "format": "wav"
        }
      },
      {
        "type": "input_file",
        "file_url": "https://example.com/manual.pdf",
        "filename": "manual.pdf"
      }
    ])))
    .unwrap();

    assert!(matches!(
      &content[0],
      CoreContent::Audio { source } if source["media_type"] == "audio/wav" && source["data"] == "Zm9v"
    ));
    assert!(matches!(
      &content[1],
      CoreContent::File { source } if source["url"] == "https://example.com/manual.pdf" && source["filename"] == "manual.pdf"
    ));
  }
}
