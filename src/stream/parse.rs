use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value};

use super::{
  super::{
    core::{CoreUsage, StreamEvent},
    protocol::{
      map_anthropic_finish_reason, map_responses_finish_reason, usage_from_anthropic, usage_from_openai,
      usage_from_responses,
    },
    utils::{get_first_str, get_first_str_or, get_str, get_str_or, parse_json_ref, parse_json_string, stringify_json},
  },
  SseFrame, StreamParseError,
  sse::parse_sse_frames,
};

#[derive(Debug, Clone, Default)]
struct ToolCallState {
  name: Option<String>,
  arguments: String,
  thought: Option<String>,
  emitted: bool,
}

#[derive(Debug, Clone, Default)]
struct AnthropicToolBlockState {
  call_id: String,
  name: String,
  thought: Option<String>,
  arguments: String,
  emitted: bool,
}

fn emit_done_with_usage(events: &mut Vec<StreamEvent>, usage: Option<&CoreUsage>, finish_reason: String) {
  if let Some(usage) = usage {
    events.push(StreamEvent::Usage { usage: usage.clone() });
  }

  events.push(StreamEvent::Done {
    finish_reason: Some(finish_reason),
    usage: usage.cloned(),
  });
}

fn maybe_emit_tool_call(events: &mut Vec<StreamEvent>, call_id: &str, state: &mut ToolCallState) {
  if state.emitted {
    return;
  }

  let Some(name) = &state.name else {
    return;
  };

  let arguments = if state.arguments.is_empty() {
    Value::Object(Map::new())
  } else {
    parse_json_string(&state.arguments)
  };

  events.push(StreamEvent::ToolCall {
    call_id: call_id.to_string(),
    name: name.clone(),
    arguments,
    thought: state.thought.clone(),
  });

  state.emitted = true;
}

fn parse_stream_error(payload: &Value) -> StreamEvent {
  let error = payload.get("error").unwrap_or(payload);
  let message = get_first_str_or(error, &["message", "detail"], "upstream stream error").to_string();
  let code = get_first_str(error, &["code", "type"]).map(ToString::to_string);

  StreamEvent::Error { message, code }
}

fn extract_call_id(value: &Value) -> String {
  get_first_str_or(value, &["call_id", "id"], "call_0").to_string()
}

#[derive(Debug, Default)]
pub struct OpenaiChatStreamParser {
  started: bool,
  finished: bool,
  stream_id: Option<String>,
  stream_model: Option<String>,
  finish_reason: Option<String>,
  usage: Option<CoreUsage>,
  citation_by_index: BTreeMap<usize, String>,
  index_to_call_id: HashMap<i64, String>,
  tool_calls: BTreeMap<String, ToolCallState>,
}

impl OpenaiChatStreamParser {
  pub fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    if self.finished {
      return Ok(Vec::new());
    }

    if frame.data == "[DONE]" {
      self.finished = true;
      return Ok(self.flush_terminal_events());
    }

    let mut events = Vec::new();

    let json: Value = serde_json::from_str(&frame.data).map_err(|source| StreamParseError::InvalidJson {
      context: "openai_chat_stream",
      source,
    })?;

    if json.get("error").is_some() {
      events.push(parse_stream_error(&json));
      return Ok(events);
    }

    if self.stream_id.is_none() {
      self.stream_id = get_str(&json, "id").map(ToString::to_string);
    }
    if self.stream_model.is_none() {
      self.stream_model = get_str(&json, "model").map(ToString::to_string);
    }

    if !self.started {
      events.push(StreamEvent::MessageStart {
        id: self.stream_id.clone(),
        model: self.stream_model.clone(),
      });
      self.started = true;
    }

    if let Some(citations) = json.get("citations").and_then(Value::as_array) {
      for (offset, citation) in citations.iter().enumerate() {
        let Some(url) = citation.as_str() else {
          continue;
        };

        let index = offset + 1;
        if self.citation_by_index.get(&index).map(|existing| existing.as_str()) == Some(url) {
          continue;
        }

        self.citation_by_index.insert(index, url.to_string());
        events.push(StreamEvent::Citation {
          index,
          url: url.to_string(),
        });
      }
    }

    if let Some(choices) = json.get("choices").and_then(Value::as_array) {
      for choice in choices {
        self.handle_choice(choice, &mut events);
      }
    }

    if let Some(parsed_usage) = json.get("usage").map(|usage| usage_from_openai(Some(usage), 0, 0)) {
      self.usage = Some(parsed_usage);
    }

    Ok(events)
  }

  fn handle_choice(&mut self, choice: &Value, events: &mut Vec<StreamEvent>) {
    if let Some(delta) = choice.get("delta") {
      if let Some(text) = get_str(delta, "content")
        && !text.is_empty()
      {
        events.push(StreamEvent::TextDelta { text: text.to_string() });
      }

      if let Some(reasoning) = get_str(delta, "reasoning_content")
        && !reasoning.is_empty()
      {
        events.push(StreamEvent::ReasoningDelta {
          text: reasoning.to_string(),
        });
      }

      if let Some(tool_call_deltas) = delta.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_call_deltas {
          self.merge_tool_call_delta(tool_call, events);
        }
      }

      // Some OpenAI-compatible providers still emit the legacy function_call shape.
      if let Some(function_call_delta) = delta.get("function_call") {
        self.merge_legacy_function_call_delta(function_call_delta, events);
      }
    }

    // Some providers emit complete tool_calls on the final chunk in
    // message.tool_calls.
    if let Some(message_tool_calls) = choice.get("message").and_then(|message| message.get("tool_calls")) {
      match message_tool_calls {
        Value::Array(tool_calls) => {
          for tool_call in tool_calls {
            self.merge_tool_call_snapshot(tool_call);
          }
        }
        Value::Object(_) => self.merge_tool_call_snapshot(message_tool_calls),
        _ => {}
      }
    }

    if let Some(reason) = get_str(choice, "finish_reason") {
      self.finish_reason = Some(reason.to_string());
      if reason == "tool_calls" {
        for (call_id, state) in &mut self.tool_calls {
          maybe_emit_tool_call(events, call_id, state);
        }
      }
    }
  }

  fn merge_tool_call_delta(&mut self, tool_call: &Value, events: &mut Vec<StreamEvent>) {
    let index = tool_call.get("index").and_then(Value::as_i64);
    let explicit_call_id = get_first_str(tool_call, &["id", "call_id"]).map(ToString::to_string);
    let function = tool_call.get("function").unwrap_or(&Value::Null);
    let name = get_str(function, "name").map(ToString::to_string);

    let call_id = self.resolve_tool_call_id(index, explicit_call_id, name.clone());
    if let Some(index) = index {
      self.index_to_call_id.insert(index, call_id.clone());
    }

    let arguments_delta = get_str_or(function, "arguments", "").to_string();
    let thought = get_str(tool_call, "thought").map(ToString::to_string);

    events.push(StreamEvent::ToolCallDelta {
      call_id: call_id.clone(),
      name: name.clone(),
      arguments_delta: arguments_delta.clone(),
    });

    let state = self.tool_calls.entry(call_id).or_default();
    if let Some(name) = name {
      state.name = Some(name);
    }
    if !arguments_delta.is_empty() {
      state.arguments.push_str(&arguments_delta);
    }
    if thought.is_some() {
      state.thought = thought;
    }
  }

  fn merge_legacy_function_call_delta(&mut self, function_call: &Value, events: &mut Vec<StreamEvent>) {
    if !function_call.is_object() {
      return;
    }

    let explicit_call_id = get_first_str(function_call, &["id", "call_id"]).map(ToString::to_string);
    let name = get_str(function_call, "name").map(ToString::to_string);
    let call_id = self.resolve_tool_call_id(Some(0), explicit_call_id, name.clone());
    self.index_to_call_id.insert(0, call_id.clone());

    let arguments_delta = get_str_or(function_call, "arguments", "").to_string();

    events.push(StreamEvent::ToolCallDelta {
      call_id: call_id.clone(),
      name: name.clone(),
      arguments_delta: arguments_delta.clone(),
    });

    let state = self.tool_calls.entry(call_id).or_default();
    if let Some(name) = name {
      state.name = Some(name);
    }
    if !arguments_delta.is_empty() {
      state.arguments.push_str(&arguments_delta);
    }
  }

  fn merge_tool_call_snapshot(&mut self, tool_call: &Value) {
    if !tool_call.is_object() {
      return;
    }

    let index = tool_call.get("index").and_then(Value::as_i64);
    let explicit_call_id = get_first_str(tool_call, &["id", "call_id"]).map(ToString::to_string);
    let function = tool_call.get("function").unwrap_or(&Value::Null);
    let name = get_str(function, "name").map(ToString::to_string);
    let call_id = self.resolve_tool_call_id(index, explicit_call_id, name.clone());

    if let Some(index) = index {
      self.index_to_call_id.insert(index, call_id.clone());
    }

    let arguments = get_str(function, "arguments").map(ToString::to_string);
    let thought = get_str(tool_call, "thought").map(ToString::to_string);

    let state = self.tool_calls.entry(call_id).or_default();
    if let Some(name) = name {
      state.name = Some(name);
    }
    if let Some(arguments) = arguments {
      state.arguments = arguments;
    }
    if thought.is_some() {
      state.thought = thought;
    }
  }

  fn resolve_tool_call_id(&self, index: Option<i64>, explicit_call_id: Option<String>, name: Option<String>) -> String {
    explicit_call_id
      .or_else(|| index.and_then(|idx| self.index_to_call_id.get(&idx).cloned()))
      .or_else(|| {
        name.and_then(|value| {
          let trimmed = value.trim();
          if trimmed.is_empty() {
            None
          } else {
            Some(format!("{trimmed}:0"))
          }
        })
      })
      .unwrap_or_else(|| format!("call_{}", index.unwrap_or(self.tool_calls.len() as i64)))
  }

  pub fn finish(&mut self) -> Vec<StreamEvent> {
    if self.finished {
      return Vec::new();
    }

    self.finished = true;
    self.flush_terminal_events()
  }

  fn flush_terminal_events(&mut self) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for (call_id, state) in &mut self.tool_calls {
      maybe_emit_tool_call(&mut events, call_id, state);
    }

    emit_done_with_usage(
      &mut events,
      self.usage.as_ref(),
      self.finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
    );

    events
  }
}

#[derive(Debug, Default)]
pub struct OpenaiResponsesStreamParser {
  started: bool,
  finished: bool,
  stream_id: Option<String>,
  stream_model: Option<String>,
  finish_reason: Option<String>,
  status: Option<String>,
  usage: Option<CoreUsage>,
  tool_calls: BTreeMap<String, ToolCallState>,
}

impl OpenaiResponsesStreamParser {
  pub fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    if self.finished {
      return Ok(Vec::new());
    }

    if frame.data == "[DONE]" {
      self.finished = true;
      return Ok(self.flush_terminal_events());
    }

    let mut events = Vec::new();

    let json: Value = serde_json::from_str(&frame.data).map_err(|source| StreamParseError::InvalidJson {
      context: "openai_responses_stream",
      source,
    })?;

    let event_name = frame
      .event
      .as_deref()
      .or_else(|| get_str(&json, "type"))
      .unwrap_or_default();

    if self.stream_id.is_none() {
      self.stream_id = get_str(&json, "id").map(ToString::to_string);
    }
    if self.stream_model.is_none() {
      self.stream_model = get_str(&json, "model").map(ToString::to_string);
    }

    if !self.started && (self.stream_id.is_some() || self.stream_model.is_some() || event_name == "response.created") {
      events.push(StreamEvent::MessageStart {
        id: self.stream_id.clone(),
        model: self.stream_model.clone(),
      });
      self.started = true;
    }

    self.handle_event(event_name, &json, &mut events);

    Ok(events)
  }

  fn handle_event(&mut self, event_name: &str, json: &Value, events: &mut Vec<StreamEvent>) {
    match event_name {
      "response.output_text.delta" => {
        if let Some(delta) = get_str(json, "delta")
          && !delta.is_empty()
        {
          events.push(StreamEvent::TextDelta {
            text: delta.to_string(),
          });
        }
      }
      "response.reasoning.delta" => {
        let reasoning = get_first_str_or(json, &["delta", "text"], "");
        if !reasoning.is_empty() {
          events.push(StreamEvent::ReasoningDelta {
            text: reasoning.to_string(),
          });
        }
      }
      "response.function_call.delta" => {
        let call_id = extract_call_id(json);
        let name = get_str(json, "name").map(ToString::to_string);
        let delta = get_first_str_or(json, &["delta", "arguments_delta", "arguments"], "").to_string();

        events.push(StreamEvent::ToolCallDelta {
          call_id: call_id.clone(),
          name: name.clone(),
          arguments_delta: delta.clone(),
        });

        let state = self.tool_calls.entry(call_id).or_default();
        if let Some(name) = name {
          state.name = Some(name);
        }
        if !delta.is_empty() {
          state.arguments.push_str(&delta);
        }
      }
      "response.function_call.done" => {
        let call_id = extract_call_id(json);
        let name = get_str(json, "name").map(ToString::to_string);
        let arguments = get_str(json, "arguments").map(parse_json_string);

        let state = self.tool_calls.entry(call_id.clone()).or_default();
        if let Some(name) = name {
          state.name = Some(name);
        }
        if let Some(arguments) = arguments {
          state.arguments = stringify_json(&arguments);
        }

        maybe_emit_tool_call(events, &call_id, state);
      }
      "response.output_item.added" => {
        let item = json.get("item").cloned().unwrap_or_else(|| json.clone());
        match get_str_or(&item, "type", "") {
          "function_call" => {
            let call_id = extract_call_id(&item);
            let name = get_str_or(&item, "name", "").to_string();
            let arguments = get_str_or(&item, "arguments", "{}");
            events.push(StreamEvent::ToolCall {
              call_id,
              name,
              arguments: parse_json_string(arguments),
              thought: None,
            });
          }
          "function_call_output" => {
            let call_id = extract_call_id(&item);
            events.push(StreamEvent::ToolResult {
              call_id,
              output: item.get("output").map(parse_json_ref).unwrap_or(Value::Null),
              is_error: item.get("is_error").and_then(Value::as_bool),
            });
          }
          _ => {}
        }
      }
      "response.output_text.annotation.added" => {
        let annotation = json.get("annotation").unwrap_or(&Value::Null);
        if let Some(url) = get_first_str(annotation, &["url", "value"]) {
          let index = json
            .get("annotation_index")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .map(|value| value + 1)
            .unwrap_or(1);
          events.push(StreamEvent::Citation {
            index,
            url: url.to_string(),
          });
        }
      }
      "response.error" => {
        events.push(parse_stream_error(json));
      }
      "response.completed" => {
        self.status = get_str(json, "status")
          .map(ToString::to_string)
          .or_else(|| self.status.clone());
        self.finish_reason = get_str(json, "finish_reason")
          .map(ToString::to_string)
          .or_else(|| self.finish_reason.clone());
        if let Some(parsed_usage) = json.get("usage").map(|usage| usage_from_responses(Some(usage), 0, 0)) {
          self.usage = Some(parsed_usage);
        }
      }
      _ => {}
    }
  }

  pub fn finish(&mut self) -> Vec<StreamEvent> {
    if self.finished {
      return Vec::new();
    }

    self.finished = true;
    self.flush_terminal_events()
  }

  fn flush_terminal_events(&mut self) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for (call_id, state) in &mut self.tool_calls {
      maybe_emit_tool_call(&mut events, call_id, state);
    }

    emit_done_with_usage(
      &mut events,
      self.usage.as_ref(),
      map_responses_finish_reason(self.status.as_deref(), self.finish_reason.as_deref()),
    );

    events
  }
}

#[derive(Debug, Default)]
pub struct AnthropicStreamParser {
  started: bool,
  finished: bool,
  stream_id: Option<String>,
  stream_model: Option<String>,
  finish_reason: Option<String>,
  usage: Option<CoreUsage>,
  tool_blocks: HashMap<i64, AnthropicToolBlockState>,
}

impl AnthropicStreamParser {
  pub fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    if self.finished {
      return Ok(Vec::new());
    }

    if frame.data == "[DONE]" {
      self.finished = true;
      return Ok(self.flush_terminal_events());
    }

    let mut events = Vec::new();

    let json: Value = serde_json::from_str(&frame.data).map_err(|source| StreamParseError::InvalidJson {
      context: "anthropic_stream",
      source,
    })?;

    let event_name = frame
      .event
      .as_deref()
      .or_else(|| get_str(&json, "type"))
      .unwrap_or_default();

    if self.handle_event(event_name, &json, &mut events) {
      self.finished = true;
      return Ok(self.flush_terminal_events());
    }

    Ok(events)
  }

  fn handle_event(&mut self, event_name: &str, json: &Value, events: &mut Vec<StreamEvent>) -> bool {
    match event_name {
      "message_start" => {
        if let Some(message) = json.get("message") {
          self.stream_id = self
            .stream_id
            .clone()
            .or_else(|| get_str(message, "id").map(ToString::to_string));
          self.stream_model = self
            .stream_model
            .clone()
            .or_else(|| get_str(message, "model").map(ToString::to_string));
          if let Some(parsed_usage) = message
            .get("usage")
            .map(|usage| usage_from_anthropic(Some(usage), 0, 0))
          {
            self.usage = Some(parsed_usage);
          }
        }

        if !self.started {
          events.push(StreamEvent::MessageStart {
            id: self.stream_id.clone(),
            model: self.stream_model.clone(),
          });
          self.started = true;
        }
      }
      "content_block_start" => {
        let index = json.get("index").and_then(Value::as_i64).unwrap_or_default();
        let block = json
          .get("content_block")
          .cloned()
          .unwrap_or_else(|| Value::Object(Map::new()));

        match get_str_or(&block, "type", "") {
          "tool_use" => {
            let call_id = get_first_str_or(&block, &["id"], "call_0").to_string();
            let name = get_str_or(&block, "name", "").to_string();
            let thought = get_str(&block, "thought").map(ToString::to_string);

            self.tool_blocks.insert(
              index,
              AnthropicToolBlockState {
                call_id,
                name,
                thought,
                arguments: String::new(),
                emitted: false,
              },
            );
          }
          "tool_result" => {
            let call_id = get_first_str_or(&block, &["tool_use_id", "id"], "call_0").to_string();
            events.push(StreamEvent::ToolResult {
              call_id,
              output: block.get("content").map(parse_json_ref).unwrap_or(Value::Null),
              is_error: block.get("is_error").and_then(Value::as_bool),
            });
          }
          _ => {}
        }
      }
      "content_block_delta" => {
        let index = json.get("index").and_then(Value::as_i64).unwrap_or_default();
        let delta = json.get("delta").cloned().unwrap_or_else(|| Value::Object(Map::new()));

        let delta_type = get_str_or(&delta, "type", "text_delta");

        match delta_type {
          "text_delta" => {
            if let Some(text) = get_str(&delta, "text")
              && !text.is_empty()
            {
              events.push(StreamEvent::TextDelta { text: text.to_string() });
            }
          }
          "thinking_delta" => {
            if let Some(text) = get_str(&delta, "thinking")
              && !text.is_empty()
            {
              events.push(StreamEvent::ReasoningDelta { text: text.to_string() });
            }
          }
          "input_json_delta" => {
            if let Some(partial_json) = get_str(&delta, "partial_json")
              && let Some(block_state) = self.tool_blocks.get_mut(&index)
            {
              block_state.arguments.push_str(partial_json);
              events.push(StreamEvent::ToolCallDelta {
                call_id: block_state.call_id.clone(),
                name: Some(block_state.name.clone()),
                arguments_delta: partial_json.to_string(),
              });
            }
          }
          _ => {}
        }
      }
      "content_block_stop" => {
        let index = json.get("index").and_then(Value::as_i64).unwrap_or_default();
        if let Some(block_state) = self.tool_blocks.get_mut(&index)
          && !block_state.emitted
        {
          let arguments = if block_state.arguments.is_empty() {
            Value::Object(Map::new())
          } else {
            parse_json_string(&block_state.arguments)
          };

          events.push(StreamEvent::ToolCall {
            call_id: block_state.call_id.clone(),
            name: block_state.name.clone(),
            arguments,
            thought: block_state.thought.clone(),
          });

          block_state.emitted = true;
        }
      }
      "message_delta" => {
        if let Some(delta) = json.get("delta")
          && let Some(reason) = get_str(delta, "stop_reason")
        {
          self.finish_reason = Some(map_anthropic_finish_reason(reason));
        }

        if let Some(parsed_usage) = json.get("usage").map(|usage| usage_from_anthropic(Some(usage), 0, 0)) {
          self.usage = Some(parsed_usage);
        }
      }
      "message_stop" => {
        return true;
      }
      "error" => {
        events.push(parse_stream_error(json));
      }
      _ => {}
    }

    false
  }

  pub fn finish(&mut self) -> Vec<StreamEvent> {
    if self.finished {
      return Vec::new();
    }

    self.finished = true;
    self.flush_terminal_events()
  }

  fn flush_terminal_events(&mut self) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for block_state in self.tool_blocks.values_mut() {
      if !block_state.emitted {
        let arguments = if block_state.arguments.is_empty() {
          Value::Object(Map::new())
        } else {
          parse_json_string(&block_state.arguments)
        };

        events.push(StreamEvent::ToolCall {
          call_id: block_state.call_id.clone(),
          name: block_state.name.clone(),
          arguments,
          thought: block_state.thought.clone(),
        });

        block_state.emitted = true;
      }
    }

    emit_done_with_usage(
      &mut events,
      self.usage.as_ref(),
      self.finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
    );

    events
  }
}

trait StreamingEventParser {
  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError>;
  fn finish(&mut self) -> Vec<StreamEvent>;
}

impl StreamingEventParser for OpenaiChatStreamParser {
  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    OpenaiChatStreamParser::push_frame(self, frame)
  }

  fn finish(&mut self) -> Vec<StreamEvent> {
    OpenaiChatStreamParser::finish(self)
  }
}

impl StreamingEventParser for OpenaiResponsesStreamParser {
  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    OpenaiResponsesStreamParser::push_frame(self, frame)
  }

  fn finish(&mut self) -> Vec<StreamEvent> {
    OpenaiResponsesStreamParser::finish(self)
  }
}

impl StreamingEventParser for AnthropicStreamParser {
  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    AnthropicStreamParser::push_frame(self, frame)
  }

  fn finish(&mut self) -> Vec<StreamEvent> {
    AnthropicStreamParser::finish(self)
  }
}

fn run_parser<P: StreamingEventParser>(raw: &str, mut parser: P) -> Result<Vec<StreamEvent>, StreamParseError> {
  let frames = parse_sse_frames(raw);
  let mut events = Vec::new();

  for frame in frames {
    events.extend(parser.push_frame(frame)?);
  }
  events.extend(parser.finish());

  Ok(events)
}

pub fn parse_openai_chat_stream(raw: &str) -> Result<Vec<StreamEvent>, StreamParseError> {
  run_parser(raw, OpenaiChatStreamParser::default())
}

pub fn parse_openai_responses_stream(raw: &str) -> Result<Vec<StreamEvent>, StreamParseError> {
  run_parser(raw, OpenaiResponsesStreamParser::default())
}

pub fn parse_anthropic_stream(raw: &str) -> Result<Vec<StreamEvent>, StreamParseError> {
  run_parser(raw, AnthropicStreamParser::default())
}
