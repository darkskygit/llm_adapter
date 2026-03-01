use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::{
  super::{
    core::{CoreUsage, StreamEvent},
    protocol::stringify_json,
  },
  SseFrame,
  sse::encode_sse_frame,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEncodingTarget {
  OpenaiChatCompletions,
  OpenaiResponses,
  AnthropicMessages,
}

#[derive(Debug)]
pub enum IncrementalSseEncoder {
  OpenaiChat(OpenaiChatStreamEncoder),
  OpenaiResponses(OpenaiResponsesStreamEncoder),
  Anthropic(AnthropicStreamEncoder),
}

impl IncrementalSseEncoder {
  #[must_use]
  pub fn new(target: StreamEncodingTarget) -> Self {
    match target {
      StreamEncodingTarget::OpenaiChatCompletions => Self::OpenaiChat(OpenaiChatStreamEncoder::default()),
      StreamEncodingTarget::OpenaiResponses => Self::OpenaiResponses(OpenaiResponsesStreamEncoder::default()),
      StreamEncodingTarget::AnthropicMessages => Self::Anthropic(AnthropicStreamEncoder::default()),
    }
  }

  #[must_use]
  pub fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    match self {
      Self::OpenaiChat(encoder) => encoder.push_event(event),
      Self::OpenaiResponses(encoder) => encoder.push_event(event),
      Self::Anthropic(encoder) => encoder.push_event(event),
    }
  }

  #[must_use]
  pub fn finish(&mut self) -> Vec<SseFrame> {
    match self {
      Self::OpenaiChat(encoder) => encoder.finish(),
      Self::OpenaiResponses(encoder) => encoder.finish(),
      Self::Anthropic(encoder) => encoder.finish(),
    }
  }
}

#[derive(Debug, Default)]
pub struct OpenaiChatStreamEncoder {
  stream_id: String,
  stream_model: String,
  usage: Option<CoreUsage>,
  done_emitted: bool,
}

impl OpenaiChatStreamEncoder {
  #[must_use]
  pub fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    match event {
      StreamEvent::MessageStart { id, model } => {
        if let Some(id) = id {
          self.stream_id = id.clone();
        }
        if let Some(model) = model {
          self.stream_model = model.clone();
        }
        Vec::new()
      }
      StreamEvent::Usage { usage } => {
        self.usage = Some(usage.clone());
        Vec::new()
      }
      StreamEvent::TextDelta { text } => vec![json_frame(
        None,
        json!({
          "id": self.stream_id_or_default(),
          "object": "chat.completion.chunk",
          "created": 0,
          "model": self.stream_model_or_default(),
          "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": Value::Null }],
          "usage": Value::Null,
        }),
      )],
      StreamEvent::ReasoningDelta { text } => vec![json_frame(
        None,
        json!({
          "id": self.stream_id_or_default(),
          "object": "chat.completion.chunk",
          "created": 0,
          "model": self.stream_model_or_default(),
          "choices": [{ "index": 0, "delta": { "reasoning_content": text }, "finish_reason": Value::Null }],
          "usage": Value::Null,
        }),
      )],
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => vec![json_frame(
        None,
        json!({
          "id": self.stream_id_or_default(),
          "object": "chat.completion.chunk",
          "created": 0,
          "model": self.stream_model_or_default(),
          "choices": [{
            "index": 0,
            "delta": {
              "tool_calls": [{
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": { "name": name, "arguments": arguments_delta }
              }]
            },
            "finish_reason": Value::Null
          }],
          "usage": Value::Null,
        }),
      )],
      StreamEvent::ToolCall {
        call_id,
        name,
        arguments,
        ..
      } => vec![json_frame(
        None,
        json!({
          "id": self.stream_id_or_default(),
          "object": "chat.completion.chunk",
          "created": 0,
          "model": self.stream_model_or_default(),
          "choices": [{
            "index": 0,
            "delta": {
              "tool_calls": [{
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": { "name": name, "arguments": stringify_json(arguments) }
              }]
            },
            "finish_reason": Value::Null
          }],
          "usage": Value::Null,
        }),
      )],
      StreamEvent::ToolResult { output, .. } => {
        // Chat-completions stream has no native tool-result event. Map to text delta.
        vec![json_frame(
          None,
          json!({
            "id": self.stream_id_or_default(),
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.stream_model_or_default(),
            "choices": [{ "index": 0, "delta": { "content": stringify_json(output) }, "finish_reason": Value::Null }],
            "usage": Value::Null,
          }),
        )]
      }
      StreamEvent::Citation { index, url } => {
        let mut citations = vec![Value::Null; (*index).max(1)];
        citations[index.saturating_sub(1)] = Value::String(url.clone());

        vec![json_frame(
          None,
          json!({
            "id": self.stream_id_or_default(),
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.stream_model_or_default(),
            "citations": citations,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": Value::Null }],
            "usage": Value::Null,
          }),
        )]
      }
      StreamEvent::Error { message, code } => vec![json_frame(
        None,
        json!({
          "error": {
            "message": message,
            "code": code,
          }
        }),
      )],
      StreamEvent::Done { finish_reason, usage } => {
        if let Some(usage) = usage {
          self.usage = Some(usage.clone());
        }

        self.done_emitted = true;
        vec![
          json_frame(
            None,
            json!({
              "id": self.stream_id_or_default(),
              "object": "chat.completion.chunk",
              "created": 0,
              "model": self.stream_model_or_default(),
              "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason.clone().unwrap_or_else(|| "stop".to_string())
              }],
              "usage": self.usage,
            }),
          ),
          done_frame(),
        ]
      }
    }
  }

  #[must_use]
  pub fn finish(&mut self) -> Vec<SseFrame> {
    if self.done_emitted {
      return Vec::new();
    }

    self.done_emitted = true;
    vec![
      json_frame(
        None,
        json!({
          "id": self.stream_id_or_default(),
          "object": "chat.completion.chunk",
          "created": 0,
          "model": self.stream_model_or_default(),
          "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
          }],
          "usage": self.usage,
        }),
      ),
      done_frame(),
    ]
  }

  fn stream_id_or_default(&self) -> String {
    if self.stream_id.is_empty() {
      "chat_stream".to_string()
    } else {
      self.stream_id.clone()
    }
  }

  fn stream_model_or_default(&self) -> String {
    if self.stream_model.is_empty() {
      "unknown".to_string()
    } else {
      self.stream_model.clone()
    }
  }
}

#[derive(Debug, Default)]
pub struct OpenaiResponsesStreamEncoder {
  stream_id: String,
  stream_model: String,
  usage: Option<CoreUsage>,
  sequence_number: u32,
  created_emitted: bool,
  done_emitted: bool,
}

impl OpenaiResponsesStreamEncoder {
  #[must_use]
  pub fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    match event {
      StreamEvent::MessageStart { id, model } => {
        if let Some(id) = id {
          self.stream_id = id.clone();
        }
        if let Some(model) = model {
          self.stream_model = model.clone();
        }
        Vec::new()
      }
      StreamEvent::Usage { usage } => {
        self.usage = Some(usage.clone());
        Vec::new()
      }
      StreamEvent::TextDelta { text } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.output_text.delta"),
          json!({
            "type": "response.output_text.delta",
            "id": self.stream_id_or_default(),
            "object": "response.output_text.delta",
            "created_at": 0,
            "model": self.stream_model_or_default(),
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "annotations": [],
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::ReasoningDelta { text } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.reasoning.delta"),
          json!({
            "type": "response.reasoning.delta",
            "id": self.stream_id_or_default(),
            "object": "response.reasoning.delta",
            "created_at": 0,
            "model": self.stream_model_or_default(),
            "delta": text,
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.function_call.delta"),
          json!({
            "type": "response.function_call.delta",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "delta": arguments_delta,
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::ToolCall {
        call_id,
        name,
        arguments,
        ..
      } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.function_call.done"),
          json!({
            "type": "response.function_call.done",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": stringify_json(arguments),
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::ToolResult {
        call_id,
        output,
        is_error,
      } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.output_item.added"),
          json!({
            "type": "response.output_item.added",
            "id": self.stream_id_or_default(),
            "output_index": 0,
            "item": {
              "type": "function_call_output",
              "id": Value::Null,
              "call_id": call_id,
              "output": output,
              "is_error": is_error,
            },
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::Citation { index, url } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.output_text.annotation.added"),
          json!({
            "type": "response.output_text.annotation.added",
            "id": self.stream_id_or_default(),
            "output_index": 0,
            "content_index": 0,
            "annotation_index": index.saturating_sub(1),
            "annotation": {
              "type": "url_citation",
              "url": url,
            },
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::Error { message, code } => {
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.error"),
          json!({
            "type": "response.error",
            "id": self.stream_id_or_default(),
            "error": {
              "message": message,
              "code": code,
            },
            "sequence_number": self.sequence_number,
          }),
        ));
        frames
      }
      StreamEvent::Done { finish_reason, usage } => {
        if let Some(usage) = usage {
          self.usage = Some(usage.clone());
        }

        self.done_emitted = true;
        let mut frames = self.ensure_created();
        frames.push(self.next_json_frame(
          Some("response.completed"),
          json!({
            "type": "response.completed",
            "id": self.stream_id_or_default(),
            "object": "response.completed",
            "created_at": 0,
            "model": self.stream_model_or_default(),
            "status": if finish_reason.as_deref() == Some("tool_calls") { "requires_action" } else { "completed" },
            "finish_reason": finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
            "usage": self.usage,
            "sequence_number": self.sequence_number,
          }),
        ));
        frames.push(done_frame());
        frames
      }
    }
  }

  #[must_use]
  pub fn finish(&mut self) -> Vec<SseFrame> {
    if self.done_emitted {
      return Vec::new();
    }

    self.done_emitted = true;
    let mut frames = self.ensure_created();
    frames.push(self.next_json_frame(
      Some("response.completed"),
      json!({
        "type": "response.completed",
        "id": self.stream_id_or_default(),
        "object": "response.completed",
        "created_at": 0,
        "model": self.stream_model_or_default(),
        "status": "completed",
        "finish_reason": "stop",
        "usage": self.usage,
        "sequence_number": self.sequence_number,
      }),
    ));
    frames.push(done_frame());
    frames
  }

  fn ensure_created(&mut self) -> Vec<SseFrame> {
    if self.created_emitted {
      return Vec::new();
    }

    self.created_emitted = true;
    vec![self.next_json_frame(
      Some("response.created"),
      json!({
        "type": "response.created",
        "id": self.stream_id_or_default(),
        "object": "response",
        "created_at": 0,
        "model": self.stream_model_or_default(),
        "sequence_number": self.sequence_number,
      }),
    )]
  }

  fn next_json_frame(&mut self, event: Option<&str>, payload: Value) -> SseFrame {
    let frame = json_frame(event, payload);
    self.sequence_number = self.sequence_number.saturating_add(1);
    frame
  }

  fn stream_id_or_default(&self) -> String {
    if self.stream_id.is_empty() {
      "resp_stream".to_string()
    } else {
      self.stream_id.clone()
    }
  }

  fn stream_model_or_default(&self) -> String {
    if self.stream_model.is_empty() {
      "unknown".to_string()
    } else {
      self.stream_model.clone()
    }
  }
}

#[derive(Debug, Default)]
pub struct AnthropicStreamEncoder {
  stream_id: String,
  stream_model: String,
  usage: Option<CoreUsage>,
  message_started: bool,
  text_started: bool,
  thinking_started: bool,
  text_index: i64,
  thinking_index: i64,
  tool_indexes: HashMap<String, i64>,
  stopped_tool_indexes: HashSet<i64>,
  next_index: i64,
  done_emitted: bool,
}

impl AnthropicStreamEncoder {
  #[must_use]
  pub fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    match event {
      StreamEvent::MessageStart { id, model } => {
        if let Some(id) = id {
          self.stream_id = id.clone();
        }
        if let Some(model) = model {
          self.stream_model = model.clone();
        }
        Vec::new()
      }
      StreamEvent::Usage { usage } => {
        self.usage = Some(usage.clone());
        Vec::new()
      }
      StreamEvent::TextDelta { text } => {
        let mut frames = self.ensure_message_start();
        if !self.text_started {
          self.text_started = true;
          frames.push(json_frame(
            Some("content_block_start"),
            json!({
              "type": "content_block_start",
              "index": self.text_index,
              "content_block": {
                "type": "text",
                "text": "",
              }
            }),
          ));
        }

        frames.push(json_frame(
          Some("content_block_delta"),
          json!({
            "type": "content_block_delta",
            "index": self.text_index,
            "delta": {
              "type": "text_delta",
              "text": text,
            }
          }),
        ));
        frames
      }
      StreamEvent::ReasoningDelta { text } => {
        let mut frames = self.ensure_message_start();
        if !self.thinking_started {
          self.thinking_started = true;
          frames.push(json_frame(
            Some("content_block_start"),
            json!({
              "type": "content_block_start",
              "index": self.thinking_index,
              "content_block": {
                "type": "thinking",
                "thinking": "",
              }
            }),
          ));
        }

        frames.push(json_frame(
          Some("content_block_delta"),
          json!({
            "type": "content_block_delta",
            "index": self.thinking_index,
            "delta": {
              "type": "thinking_delta",
              "thinking": text,
            }
          }),
        ));
        frames
      }
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => {
        let mut frames = self.ensure_message_start();
        let tool_index = if let Some(tool_index) = self.tool_indexes.get(call_id) {
          *tool_index
        } else {
          let tool_index = self.next_tool_index();
          self.tool_indexes.insert(call_id.clone(), tool_index);
          frames.push(json_frame(
            Some("content_block_start"),
            json!({
              "type": "content_block_start",
              "index": tool_index,
              "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name.as_deref().unwrap_or_default(),
                "input": {},
              }
            }),
          ));
          tool_index
        };

        frames.push(json_frame(
          Some("content_block_delta"),
          json!({
            "type": "content_block_delta",
            "index": tool_index,
            "delta": {
              "type": "input_json_delta",
              "partial_json": arguments_delta,
            }
          }),
        ));
        frames
      }
      StreamEvent::ToolCall {
        call_id,
        name,
        arguments,
        ..
      } => {
        let mut frames = self.ensure_message_start();

        let tool_index = if let Some(tool_index) = self.tool_indexes.get(call_id) {
          *tool_index
        } else {
          let tool_index = self.next_tool_index();
          self.tool_indexes.insert(call_id.clone(), tool_index);
          frames.push(json_frame(
            Some("content_block_start"),
            json!({
              "type": "content_block_start",
              "index": tool_index,
              "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
              }
            }),
          ));
          frames.push(json_frame(
            Some("content_block_delta"),
            json!({
              "type": "content_block_delta",
              "index": tool_index,
              "delta": {
                "type": "input_json_delta",
                "partial_json": stringify_json(arguments),
              }
            }),
          ));
          tool_index
        };

        if !self.stopped_tool_indexes.contains(&tool_index) {
          frames.push(json_frame(
            Some("content_block_stop"),
            json!({
              "type": "content_block_stop",
              "index": tool_index,
            }),
          ));
          self.stopped_tool_indexes.insert(tool_index);
        }

        frames
      }
      StreamEvent::ToolResult {
        call_id,
        output,
        is_error,
      } => {
        let mut frames = self.ensure_message_start();
        let tool_index = self.next_tool_index();

        frames.push(json_frame(
          Some("content_block_start"),
          json!({
            "type": "content_block_start",
            "index": tool_index,
            "content_block": {
              "type": "tool_result",
              "tool_use_id": call_id,
              "content": tool_result_content_to_anthropic(output),
              "is_error": is_error,
            }
          }),
        ));
        frames.push(json_frame(
          Some("content_block_stop"),
          json!({
            "type": "content_block_stop",
            "index": tool_index,
          }),
        ));
        frames
      }
      StreamEvent::Citation { index, url } => {
        // Anthropic stream has no native citation delta block. Map citation to text.
        self.push_event(&StreamEvent::TextDelta {
          text: format!("[{}] {}", index, url),
        })
      }
      StreamEvent::Error { message, code } => {
        let mut frames = self.ensure_message_start();
        frames.push(json_frame(
          Some("error"),
          json!({
            "type": "error",
            "error": {
              "type": code.clone().unwrap_or_else(|| "stream_error".to_string()),
              "message": message,
            }
          }),
        ));
        frames
      }
      StreamEvent::Done { finish_reason, usage } => {
        if let Some(usage) = usage {
          self.usage = Some(usage.clone());
        }

        self.done_emitted = true;

        let mut frames = self.ensure_message_start();
        if self.thinking_started {
          frames.push(json_frame(
            Some("content_block_stop"),
            json!({
              "type": "content_block_stop",
              "index": self.thinking_index,
            }),
          ));
        }

        if self.text_started {
          frames.push(json_frame(
            Some("content_block_stop"),
            json!({
              "type": "content_block_stop",
              "index": self.text_index,
            }),
          ));
        }

        for tool_index in self.tool_indexes.values() {
          if self.stopped_tool_indexes.contains(tool_index) {
            continue;
          }
          frames.push(json_frame(
            Some("content_block_stop"),
            json!({
              "type": "content_block_stop",
              "index": tool_index,
            }),
          ));
          self.stopped_tool_indexes.insert(*tool_index);
        }

        frames.push(json_frame(
          Some("message_delta"),
          json!({
            "type": "message_delta",
            "delta": {
              "stop_reason": finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
              "stop_sequence": Value::Null,
            },
            "usage": {
              "input_tokens": self.usage.as_ref().map(|value| value.prompt_tokens).unwrap_or_default(),
              "output_tokens": self.usage.as_ref().map(|value| value.completion_tokens).unwrap_or_default(),
              "cache_read_input_tokens": self.usage.as_ref().and_then(|value| value.cached_tokens),
              "cache_creation_input_tokens": Value::Null,
            }
          }),
        ));
        frames.push(json_frame(
          Some("message_stop"),
          json!({
            "type": "message_stop",
          }),
        ));
        frames.push(done_frame());
        frames
      }
    }
  }

  #[must_use]
  pub fn finish(&mut self) -> Vec<SseFrame> {
    if self.done_emitted {
      return Vec::new();
    }

    self.push_event(&StreamEvent::Done {
      finish_reason: Some("stop".to_string()),
      usage: self.usage.clone(),
    })
  }

  fn ensure_message_start(&mut self) -> Vec<SseFrame> {
    if self.message_started {
      return Vec::new();
    }

    self.message_started = true;
    vec![json_frame(
      Some("message_start"),
      json!({
        "type": "message_start",
        "message": {
          "id": self.stream_id_or_default(),
          "type": "message",
          "role": "assistant",
          "model": self.stream_model_or_default(),
          "content": [],
          "usage": {
            "input_tokens": self.usage.as_ref().map(|value| value.prompt_tokens).unwrap_or_default(),
            "output_tokens": self.usage.as_ref().map(|value| value.completion_tokens).unwrap_or_default(),
            "cache_read_input_tokens": self.usage.as_ref().and_then(|value| value.cached_tokens),
            "cache_creation_input_tokens": Value::Null,
          }
        }
      }),
    )]
  }

  fn next_tool_index(&mut self) -> i64 {
    if self.next_index == 0 {
      self.next_index = 2;
    }
    let index = self.next_index;
    self.next_index += 1;
    index
  }

  fn stream_id_or_default(&self) -> String {
    if self.stream_id.is_empty() {
      "anthropic_stream".to_string()
    } else {
      self.stream_id.clone()
    }
  }

  fn stream_model_or_default(&self) -> String {
    if self.stream_model.is_empty() {
      "unknown".to_string()
    } else {
      self.stream_model.clone()
    }
  }
}

#[must_use]
pub fn encode_openai_chat_stream(events: &[StreamEvent]) -> String {
  let mut encoder = OpenaiChatStreamEncoder::default();
  encode_with_encoder(events, &mut encoder)
}

#[must_use]
pub fn encode_openai_responses_stream(events: &[StreamEvent]) -> String {
  let mut encoder = OpenaiResponsesStreamEncoder::default();
  encode_with_encoder(events, &mut encoder)
}

#[must_use]
pub fn encode_anthropic_stream(events: &[StreamEvent]) -> String {
  let mut encoder = AnthropicStreamEncoder::default();
  encode_with_encoder(events, &mut encoder)
}

fn encode_with_encoder<E>(events: &[StreamEvent], encoder: &mut E) -> String
where
  E: StreamEventEncoder,
{
  let mut out = String::new();

  for event in events {
    for frame in encoder.push_event(event) {
      out.push_str(&encode_sse_frame(&frame));
    }
  }

  for frame in encoder.finish() {
    out.push_str(&encode_sse_frame(&frame));
  }

  out
}

trait StreamEventEncoder {
  fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame>;
  fn finish(&mut self) -> Vec<SseFrame>;
}

impl StreamEventEncoder for OpenaiChatStreamEncoder {
  fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    OpenaiChatStreamEncoder::push_event(self, event)
  }

  fn finish(&mut self) -> Vec<SseFrame> {
    OpenaiChatStreamEncoder::finish(self)
  }
}

impl StreamEventEncoder for OpenaiResponsesStreamEncoder {
  fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    OpenaiResponsesStreamEncoder::push_event(self, event)
  }

  fn finish(&mut self) -> Vec<SseFrame> {
    OpenaiResponsesStreamEncoder::finish(self)
  }
}

impl StreamEventEncoder for AnthropicStreamEncoder {
  fn push_event(&mut self, event: &StreamEvent) -> Vec<SseFrame> {
    AnthropicStreamEncoder::push_event(self, event)
  }

  fn finish(&mut self) -> Vec<SseFrame> {
    AnthropicStreamEncoder::finish(self)
  }
}

fn done_frame() -> SseFrame {
  SseFrame {
    event: None,
    data: "[DONE]".to_string(),
  }
}

fn json_frame(event: Option<&str>, payload: Value) -> SseFrame {
  SseFrame {
    event: event.map(ToString::to_string),
    data: serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string()),
  }
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

fn is_anthropic_content_block(value: &Value) -> bool {
  value.get("type").and_then(Value::as_str).is_some()
}
