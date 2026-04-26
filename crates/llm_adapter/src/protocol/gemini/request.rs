use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRequest, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition,
  ProtocolError, attachment_source,
  common::{attachment_source_to_part, get_string, get_value, parse_parts, tool_result_response},
  get_str, get_u32,
};
use crate::backend::BackendRequestLayer;

#[derive(Debug, Deserialize)]
struct GeminiRequest {
  model: Option<String>,
  #[serde(default)]
  contents: Vec<GeminiContent>,
  #[serde(rename = "systemInstruction")]
  system_instruction: Option<GeminiContent>,
  #[serde(default)]
  tools: Vec<GeminiTool>,
  #[serde(rename = "toolConfig")]
  tool_config: Option<Value>,
  #[serde(rename = "generationConfig")]
  generation_config: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
  role: Option<String>,
  #[serde(default)]
  parts: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiTool {
  #[serde(rename = "functionDeclarations", default)]
  function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionDeclaration {
  name: String,
  description: Option<String>,
  parameters: Option<Value>,
}

fn parse_gemini_role(role: Option<&str>, has_function_response: bool) -> CoreRole {
  if has_function_response {
    CoreRole::Tool
  } else {
    match role.unwrap_or("model") {
      "user" => CoreRole::User,
      "system" => CoreRole::System,
      _ => CoreRole::Assistant,
    }
  }
}

fn parse_tool_choice(tool_config: Option<&Value>) -> Result<Option<CoreToolChoice>, ProtocolError> {
  let Some(tool_config) = tool_config else {
    return Ok(None);
  };
  let Some(function_calling_config) = tool_config.get("functionCallingConfig") else {
    return Ok(None);
  };
  let mode = get_string(function_calling_config, &["mode"]).unwrap_or("AUTO");
  match mode {
    "AUTO" => Ok(Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto))),
    "NONE" => Ok(Some(CoreToolChoice::Mode(CoreToolChoiceMode::None))),
    "ANY" => {
      let allowed = function_calling_config
        .get("allowedFunctionNames")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
      if allowed.len() == 1
        && let Some(name) = allowed[0].as_str()
      {
        return Ok(Some(CoreToolChoice::Specific { name: name.to_string() }));
      }
      Ok(Some(CoreToolChoice::Mode(CoreToolChoiceMode::Required)))
    }
    other => Err(ProtocolError::InvalidRequest {
      field: "toolConfig.functionCallingConfig.mode",
      message: format!("unsupported mode `{other}`"),
    }),
  }
}

fn effort_to_budget(effort: &str) -> Option<u32> {
  match effort.to_ascii_lowercase().as_str() {
    "minimal" => Some(256),
    "low" => Some(1024),
    "medium" => Some(4096),
    "high" => Some(8192),
    _ => None,
  }
}

fn budget_to_effort(budget: u32) -> &'static str {
  match budget {
    0..=512 => "minimal",
    513..=1536 => "low",
    1537..=6144 => "medium",
    _ => "high",
  }
}

fn is_gemini_3_model(model: &str) -> bool {
  model.starts_with("gemini-3")
}

fn is_gemini_3_flash_family(model: &str) -> bool {
  is_gemini_3_model(model) && model.contains("flash")
}

fn normalize_gemini_3_thinking_level(model: &str, level: &str) -> &'static str {
  match level.to_ascii_lowercase().as_str() {
    "minimal" if is_gemini_3_flash_family(model) => "minimal",
    "medium" if is_gemini_3_flash_family(model) => "medium",
    "low" | "minimal" | "medium" => "low",
    "high" => "high",
    _ if is_gemini_3_flash_family(model) => "minimal",
    _ => "low",
  }
}

fn decode_reasoning_config(generation_config: Option<&Value>) -> (Option<Vec<String>>, Option<Value>) {
  let Some(thinking_config) = generation_config.and_then(|value| value.get("thinkingConfig")) else {
    return (None, None);
  };

  let include = thinking_config
    .get("includeThoughts")
    .and_then(Value::as_bool)
    .filter(|enabled| *enabled)
    .map(|_| vec!["reasoning".to_string()]);

  let mut reasoning = Map::new();
  if let Some(budget_tokens) = get_u32(thinking_config, "thinkingBudget") {
    reasoning.insert("budget_tokens".to_string(), json!(budget_tokens));
    reasoning.insert("effort".to_string(), json!(budget_to_effort(budget_tokens)));
  }
  if let Some(level) = get_str(thinking_config, "thinkingLevel") {
    reasoning.insert("level".to_string(), json!(level.to_ascii_lowercase()));
  }

  (include, (!reasoning.is_empty()).then_some(Value::Object(reasoning)))
}

fn core_content_to_part(
  content: &CoreContent,
  tool_names: &mut HashMap<String, String>,
  request_layer: BackendRequestLayer,
  base_url: &str,
) -> Option<Value> {
  match content {
    CoreContent::Text { text } => Some(json!({ "text": text })),
    CoreContent::Reasoning { text, signature } => {
      let mut part = Map::from_iter([
        ("text".to_string(), Value::String(text.clone())),
        ("thought".to_string(), Value::Bool(true)),
      ]);
      if let Some(signature) = signature {
        part.insert("thoughtSignature".to_string(), Value::String(signature.clone()));
      }
      Some(Value::Object(part))
    }
    CoreContent::ToolCall {
      call_id,
      name,
      arguments,
      thought: _,
    } => {
      tool_names.insert(call_id.clone(), name.clone());
      Some(json!({
        "functionCall": {
          "name": name,
          "args": arguments,
        }
      }))
    }
    CoreContent::ToolResult {
      call_id,
      output,
      is_error,
    } => Some(json!({
      "functionResponse": {
        "name": tool_names.get(call_id).cloned().unwrap_or_else(|| call_id.clone()),
        "response": tool_result_response(output, *is_error),
      }
    })),
    CoreContent::Image { .. } | CoreContent::Audio { .. } | CoreContent::File { .. } => attachment_source(content)
      .map(|(source, _)| attachment_source_to_part(source, request_layer.plan_attachment_reference(base_url, source))),
  }
}

fn core_message_to_gemini(
  message: &CoreMessage,
  tool_names: &mut HashMap<String, String>,
  request_layer: BackendRequestLayer,
  base_url: &str,
) -> Option<Value> {
  let role = match message.role {
    CoreRole::Assistant => "model",
    CoreRole::User | CoreRole::Tool => "user",
    CoreRole::System => return None,
  };

  let parts: Vec<Value> = message
    .content
    .iter()
    .filter_map(|content| core_content_to_part(content, tool_names, request_layer, base_url))
    .filter(|part| !part.is_null())
    .collect();

  Some(json!({
    "role": role,
    "parts": parts,
  }))
}

fn sanitize_function_parameters(parameters: &Value) -> Value {
  match parameters {
    Value::Object(object) => {
      let mut sanitized = Map::new();
      for (key, value) in object {
        if key == "additionalProperties" {
          continue;
        }

        sanitized.insert(key.clone(), sanitize_function_parameters(value));
      }
      Value::Object(sanitized)
    }
    Value::Array(items) => Value::Array(items.iter().map(sanitize_function_parameters).collect::<Vec<_>>()),
    _ => parameters.clone(),
  }
}

fn core_tool_choice_to_gemini(choice: Option<&CoreToolChoice>) -> Option<Value> {
  let choice = choice?;
  Some(match choice {
    CoreToolChoice::Mode(CoreToolChoiceMode::Auto) => {
      json!({ "functionCallingConfig": { "mode": "AUTO" } })
    }
    CoreToolChoice::Mode(CoreToolChoiceMode::None) => {
      json!({ "functionCallingConfig": { "mode": "NONE" } })
    }
    CoreToolChoice::Mode(CoreToolChoiceMode::Required) => {
      json!({ "functionCallingConfig": { "mode": "ANY" } })
    }
    CoreToolChoice::Specific { name } => json!({
      "functionCallingConfig": {
        "mode": "ANY",
        "allowedFunctionNames": [name],
      }
    }),
  })
}

pub fn decode(request: &Value) -> Result<CoreRequest, ProtocolError> {
  let request: GeminiRequest = serde_json::from_value(request.clone())?;

  let mut messages = Vec::new();
  if let Some(system_instruction) = request.system_instruction {
    let content = parse_parts(&system_instruction.parts);
    if !content.is_empty() {
      messages.push(CoreMessage {
        role: CoreRole::System,
        content,
      });
    }
  }

  for content in request.contents {
    let has_function_response = content
      .parts
      .iter()
      .any(|part| get_value(part, &["functionResponse", "function_response"]).is_some());
    let role = parse_gemini_role(content.role.as_deref(), has_function_response);
    messages.push(CoreMessage {
      role,
      content: parse_parts(&content.parts),
    });
  }

  let (include, reasoning) = decode_reasoning_config(request.generation_config.as_ref());

  Ok(CoreRequest {
    model: request.model.unwrap_or_default(),
    messages,
    stream: false,
    max_tokens: request
      .generation_config
      .as_ref()
      .and_then(|config| get_u32(config, "maxOutputTokens")),
    temperature: request
      .generation_config
      .as_ref()
      .and_then(|config| config.get("temperature"))
      .and_then(Value::as_f64),
    tools: request
      .tools
      .into_iter()
      .flat_map(|tool| tool.function_declarations)
      .map(|tool| CoreToolDefinition {
        name: tool.name,
        description: tool.description,
        parameters: tool
          .parameters
          .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
      })
      .collect(),
    tool_choice: parse_tool_choice(request.tool_config.as_ref())?,
    include,
    reasoning,
    response_schema: request
      .generation_config
      .as_ref()
      .and_then(|config| config.get("responseSchema"))
      .cloned(),
  })
}

#[must_use]
pub fn encode(request: &CoreRequest, stream: bool, request_layer: BackendRequestLayer, base_url: &str) -> Value {
  let mut payload = Map::from_iter([("model".to_string(), Value::String(request.model.clone()))]);
  let mut tool_names = HashMap::new();

  let mut contents = Vec::new();
  let mut system_parts = Vec::new();
  for message in &request.messages {
    if message.role == CoreRole::System {
      system_parts.extend(
        message
          .content
          .iter()
          .filter_map(|content| core_content_to_part(content, &mut tool_names, request_layer, base_url))
          .filter(|part| !part.is_null()),
      );
    } else if let Some(content) = core_message_to_gemini(message, &mut tool_names, request_layer, base_url) {
      contents.push(content);
    }
  }

  payload.insert("contents".to_string(), Value::Array(contents));
  if !system_parts.is_empty() {
    payload.insert(
      "systemInstruction".to_string(),
      json!({
        "parts": system_parts,
      }),
    );
  }

  if !request.tools.is_empty() {
    payload.insert(
      "tools".to_string(),
      json!([{
        "functionDeclarations": request
          .tools
          .iter()
          .map(|tool| json!({
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_function_parameters(&tool.parameters),
          }))
          .collect::<Vec<_>>(),
      }]),
    );
  }
  if let Some(tool_config) = core_tool_choice_to_gemini(request.tool_choice.as_ref()) {
    payload.insert("toolConfig".to_string(), tool_config);
  }

  let include_thoughts = request
    .include
    .as_ref()
    .map(|include| include.iter().any(|value| value == "reasoning"))
    .unwrap_or(false)
    || request
      .messages
      .iter()
      .flat_map(|message| &message.content)
      .any(|content| matches!(content, CoreContent::Reasoning { .. }));
  let mut generation_config = Map::new();
  if let Some(max_tokens) = request.max_tokens {
    generation_config.insert("maxOutputTokens".to_string(), json!(max_tokens));
  }
  if let Some(temperature) = request.temperature {
    generation_config.insert("temperature".to_string(), json!(temperature));
  }
  if include_thoughts || request.reasoning.is_some() {
    let mut thinking_config = Map::new();
    if include_thoughts {
      thinking_config.insert("includeThoughts".to_string(), Value::Bool(true));
    }
    if let Some(reasoning) = &request.reasoning {
      let budget_tokens =
        get_u32(reasoning, "budget_tokens").or_else(|| get_str(reasoning, "effort").and_then(effort_to_budget));
      let level = get_str(reasoning, "level")
        .map(ToString::to_string)
        .or_else(|| budget_tokens.map(|budget| budget_to_effort(budget).to_string()));
      if is_gemini_3_model(&request.model) {
        if let Some(level) = level {
          thinking_config.insert(
            "thinkingLevel".to_string(),
            json!(normalize_gemini_3_thinking_level(&request.model, &level)),
          );
        }
      } else if let Some(budget_tokens) = budget_tokens {
        thinking_config.insert("thinkingBudget".to_string(), json!(budget_tokens));
      }
    }
    generation_config.insert("thinkingConfig".to_string(), Value::Object(thinking_config));
  }
  if let Some(response_schema) = &request.response_schema {
    generation_config.insert(
      "responseMimeType".to_string(),
      Value::String("application/json".to_string()),
    );
    generation_config.insert(
      "responseSchema".to_string(),
      sanitize_function_parameters(response_schema),
    );
  }
  if !generation_config.is_empty() {
    payload.insert("generationConfig".to_string(), Value::Object(generation_config));
  }
  if stream {
    payload.insert("stream".to_string(), Value::Bool(true));
  }

  Value::Object(payload)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn decode_should_match_golden() {
    let input = json!({
      "model": "gemini-2.5-flash",
      "systemInstruction": {
        "parts": [{ "text": "You are concise." }]
      },
      "contents": [
        {
          "role": "user",
          "parts": [
            { "text": "Read this image" },
            {
              "inlineData": {
                "mimeType": "image/png",
                "data": "Zm9v"
              }
            }
          ]
        },
        {
          "role": "model",
          "parts": [
            { "text": "thinking", "thought": true, "thoughtSignature": "sig_1" },
            { "functionCall": { "name": "doc_read", "args": { "docId": "a1" } } }
          ]
        },
        {
          "role": "user",
          "parts": [
            { "functionResponse": { "name": "doc_read", "response": { "output": { "ok": true }, "is_error": false } } }
          ]
        }
      ],
      "tools": [{
        "functionDeclarations": [{
          "name": "doc_read",
          "description": "Read a doc",
          "parameters": { "type": "object", "properties": { "docId": { "type": "string" } } }
        }]
      }],
      "toolConfig": {
        "functionCallingConfig": {
          "mode": "ANY",
          "allowedFunctionNames": ["doc_read"]
        }
      },
      "generationConfig": {
        "maxOutputTokens": 512,
        "temperature": 0.2,
        "thinkingConfig": {
          "includeThoughts": true,
          "thinkingBudget": 4096
        }
      }
    });

    let core = decode(&input).unwrap();
    let actual = serde_json::to_value(core).unwrap();

    let expected = json!({
      "model": "gemini-2.5-flash",
      "messages": [
        {
          "role": "system",
          "content": [{ "type": "text", "text": "You are concise." }]
        },
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Read this image" },
            { "type": "image", "source": { "media_type": "image/png", "data": "Zm9v" } }
          ]
        },
        {
          "role": "assistant",
          "content": [
            { "type": "reasoning", "text": "thinking", "signature": "sig_1" },
            { "type": "tool_call", "call_id": "doc_read:1", "name": "doc_read", "arguments": { "docId": "a1" } }
          ]
        },
        {
          "role": "tool",
          "content": [
            { "type": "tool_result", "call_id": "doc_read:0", "output": { "ok": true }, "is_error": false }
          ]
        }
      ],
      "stream": false,
      "max_tokens": 512,
      "temperature": 0.2,
      "tools": [{
        "name": "doc_read",
        "description": "Read a doc",
        "parameters": { "type": "object", "properties": { "docId": { "type": "string" } } }
      }],
      "tool_choice": { "name": "doc_read" },
      "include": ["reasoning"],
      "reasoning": { "budget_tokens": 4096, "effort": "medium" }
    });

    assert_eq!(actual, expected);
  }

  #[test]
  fn encode_should_match_backend_contract() {
    let request = CoreRequest {
      model: "gemini-2.5-flash".to_string(),
      messages: vec![
        CoreMessage {
          role: CoreRole::System,
          content: vec![CoreContent::Text {
            text: "You are concise.".to_string(),
          }],
        },
        CoreMessage {
          role: CoreRole::User,
          content: vec![
            CoreContent::Text {
              text: "hello".to_string(),
            },
            CoreContent::Image {
              source: json!({
                "url": "https://example.com/a.png",
                "media_type": "image/png"
              }),
            },
            CoreContent::File {
              source: json!({
                "url": "https://example.com/a.pdf",
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
        CoreMessage {
          role: CoreRole::Assistant,
          content: vec![
            CoreContent::Reasoning {
              text: "drafting".to_string(),
              signature: Some("sig_2".to_string()),
            },
            CoreContent::ToolCall {
              call_id: "call_1".to_string(),
              name: "doc_read".to_string(),
              arguments: json!({ "docId": "a1" }),
              thought: None,
            },
          ],
        },
        CoreMessage {
          role: CoreRole::Tool,
          content: vec![CoreContent::ToolResult {
            call_id: "call_1".to_string(),
            output: json!({ "title": "Doc A" }),
            is_error: Some(false),
          }],
        },
      ],
      stream: false,
      max_tokens: Some(256),
      temperature: Some(0.3),
      tools: vec![CoreToolDefinition {
        name: "doc_read".to_string(),
        description: Some("Read a doc".to_string()),
        parameters: json!({ "type": "object", "properties": { "docId": { "type": "string" } } }),
      }],
      tool_choice: Some(CoreToolChoice::Specific {
        name: "doc_read".to_string(),
      }),
      include: Some(vec!["reasoning".to_string()]),
      reasoning: Some(json!({ "effort": "medium" })),
      response_schema: None,
    };

    let payload = encode(
      &request,
      false,
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(payload["model"], "gemini-2.5-flash");
    assert_eq!(payload["systemInstruction"]["parts"][0]["text"], "You are concise.");
    assert_eq!(payload["contents"][0]["role"], "user");
    assert_eq!(
      payload["contents"][0]["parts"][1]["fileData"]["fileUri"],
      "https://example.com/a.png"
    );
    assert_eq!(
      payload["contents"][0]["parts"][2]["fileData"]["mimeType"],
      "application/pdf"
    );
    assert_eq!(
      payload["contents"][0]["parts"][3]["inlineData"]["mimeType"],
      "audio/wav"
    );
    assert_eq!(payload["contents"][1]["role"], "model");
    assert_eq!(payload["contents"][1]["parts"][0]["thought"], true);
    assert_eq!(
      payload["contents"][2]["parts"][0]["functionResponse"]["name"],
      "doc_read"
    );
    assert_eq!(
      payload["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
      "doc_read"
    );
    assert_eq!(payload["generationConfig"]["maxOutputTokens"], 256);
    assert_eq!(payload["generationConfig"]["thinkingConfig"]["includeThoughts"], true);
    assert_eq!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"], 4096);
  }

  #[test]
  fn decode_should_preserve_audio_and_file_parts() {
    let core = decode(&json!({
      "model": "gemini-2.5-flash",
      "contents": [{
        "role": "user",
        "parts": [
          {
            "fileData": {
              "mimeType": "application/pdf",
              "fileUri": "https://example.com/a.pdf"
            }
          },
          {
            "inlineData": {
              "mimeType": "audio/mpeg",
              "data": "Zm9v"
            }
          }
        ]
      }]
    }))
    .unwrap();

    assert!(matches!(
      &core.messages[0].content[0],
      CoreContent::File { source } if source["media_type"] == "application/pdf"
    ));
    assert!(matches!(
      &core.messages[0].content[1],
      CoreContent::Audio { source } if source["media_type"] == "audio/mpeg" && source["data"] == "Zm9v"
    ));
  }

  #[test]
  fn encode_should_strip_additional_properties_from_function_parameters() {
    let request = CoreRequest {
      model: "gemini-2.5-flash".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text {
          text: "hello".to_string(),
        }],
      }],
      stream: false,
      max_tokens: None,
      temperature: None,
      tools: vec![CoreToolDefinition {
        name: "doc_read".to_string(),
        description: None,
        parameters: json!({
          "type": "object",
          "properties": {
            "filters": {
              "type": "object",
              "properties": {
                "docId": { "type": "string" }
              },
              "additionalProperties": false
            }
          },
          "additionalProperties": false
        }),
      }],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    let payload = encode(
      &request,
      false,
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(
      payload["tools"][0]["functionDeclarations"][0]["parameters"],
      json!({
        "type": "object",
        "properties": {
          "filters": {
            "type": "object",
            "properties": {
              "docId": { "type": "string" }
            }
          }
        }
      })
    );
  }

  #[test]
  fn encode_should_emit_response_schema_for_structured_outputs() {
    let payload = encode(
      &CoreRequest {
        model: "gemini-2.5-flash".to_string(),
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
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(payload["generationConfig"]["responseMimeType"], "application/json");
    assert_eq!(
      payload["generationConfig"]["responseSchema"]["required"],
      json!(["summary"])
    );
    assert!(
      payload["generationConfig"]["responseSchema"]
        .get("additionalProperties")
        .is_none()
    );
  }

  #[test]
  fn encode_should_use_lowercase_thinking_level_for_gemini_3_pro() {
    let payload = encode(
      &CoreRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        messages: vec![CoreMessage {
          role: CoreRole::User,
          content: vec![CoreContent::Text {
            text: "Rank this candidate.".to_string(),
          }],
        }],
        stream: false,
        max_tokens: Some(64),
        temperature: None,
        tools: vec![],
        tool_choice: None,
        include: None,
        reasoning: Some(json!({ "budget_tokens": 256 })),
        response_schema: None,
      },
      false,
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(payload["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low");
    assert!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"].is_null());
  }

  #[test]
  fn encode_should_keep_minimal_thinking_level_for_gemini_3_flash_family() {
    let payload = encode(
      &CoreRequest {
        model: "gemini-3.1-flash-lite-preview".to_string(),
        messages: vec![CoreMessage {
          role: CoreRole::User,
          content: vec![CoreContent::Text {
            text: "Rank this candidate.".to_string(),
          }],
        }],
        stream: false,
        max_tokens: Some(64),
        temperature: None,
        tools: vec![],
        tool_choice: None,
        include: None,
        reasoning: Some(json!({ "level": "minimal" })),
        response_schema: None,
      },
      false,
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(
      payload["generationConfig"]["thinkingConfig"]["thinkingLevel"],
      "minimal"
    );
    assert!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"].is_null());
  }

  #[test]
  fn encode_should_preserve_gcs_audio_urls_as_file_data() {
    let request = CoreRequest {
      model: "gemini-2.5-flash".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Audio {
          source: json!({ "url": "gs://bucket/audio.opus" }),
        }],
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

    let payload = encode(
      &request,
      false,
      BackendRequestLayer::GeminiApi,
      "https://generativelanguage.googleapis.com/v1beta",
    );

    assert_eq!(
      payload["contents"][0]["parts"][0]["fileData"]["fileUri"],
      "gs://bucket/audio.opus"
    );
    assert_eq!(payload["contents"][0]["parts"][0]["fileData"]["mimeType"], "audio/opus");
  }
}
