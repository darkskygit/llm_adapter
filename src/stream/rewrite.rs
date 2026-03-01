use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::{
  super::{
    core::{CoreUsage, StreamEvent},
    protocol::stringify_json,
  },
  SseFrame, StreamParseError,
  parse::{parse_anthropic_stream, parse_openai_chat_stream},
  sse::encode_sse_frame,
};

fn stream_event_to_chat_chunk(
  event: &StreamEvent,
  stream_id: &str,
  stream_model: &str,
  usage: Option<&CoreUsage>,
) -> Option<Value> {
  match event {
    StreamEvent::TextDelta { text } => Some(json!({
      "id": stream_id,
      "object": "chat.completion.chunk",
      "created": 0,
      "model": stream_model,
      "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": Value::Null }],
      "usage": Value::Null,
    })),
    StreamEvent::ReasoningDelta { text } => Some(json!({
      "id": stream_id,
      "object": "chat.completion.chunk",
      "created": 0,
      "model": stream_model,
      "choices": [{ "index": 0, "delta": { "reasoning_content": text }, "finish_reason": Value::Null }],
      "usage": Value::Null,
    })),
    StreamEvent::ToolCallDelta {
      call_id,
      name,
      arguments_delta,
    } => Some(json!({
      "id": stream_id,
      "object": "chat.completion.chunk",
      "created": 0,
      "model": stream_model,
      "choices": [{
        "index": 0,
        "delta": {
          "tool_calls": [
            {
              "index": 0,
              "id": call_id,
              "type": "function",
              "function": { "name": name, "arguments": arguments_delta }
            }
          ]
        },
        "finish_reason": Value::Null
      }],
      "usage": Value::Null,
    })),
    StreamEvent::ToolCall {
      call_id,
      name,
      arguments,
      ..
    } => Some(json!({
      "id": stream_id,
      "object": "chat.completion.chunk",
      "created": 0,
      "model": stream_model,
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
    })),
    StreamEvent::Done { finish_reason, .. } => Some(json!({
      "id": stream_id,
      "object": "chat.completion.chunk",
      "created": 0,
      "model": stream_model,
      "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason.clone().unwrap_or_else(|| "stop".to_string())
      }],
      "usage": usage,
    })),
    StreamEvent::MessageStart { .. } | StreamEvent::Usage { .. } => None,
    StreamEvent::ToolResult { .. } | StreamEvent::Error { .. } | StreamEvent::Citation { .. } => None,
  }
}

pub fn rewrite_anthropic_to_chat(raw: &str) -> Result<String, StreamParseError> {
  let events = parse_anthropic_stream(raw)?;

  let mut stream_id = "chat_stream".to_string();
  let mut stream_model = "unknown".to_string();
  let mut usage: Option<CoreUsage> = None;
  let mut out = String::new();

  for event in &events {
    match event {
      StreamEvent::MessageStart {
        id: Some(id),
        model: Some(model),
      } => {
        stream_id = id.clone();
        stream_model = model.clone();
      }
      StreamEvent::MessageStart {
        id: Some(id),
        model: None,
      } => {
        stream_id = id.clone();
      }
      StreamEvent::MessageStart {
        id: None,
        model: Some(model),
      } => {
        stream_model = model.clone();
      }
      StreamEvent::Usage { usage: parsed_usage } => {
        usage = Some(parsed_usage.clone());
      }
      _ => {}
    }

    if let Some(chunk) = stream_event_to_chat_chunk(event, &stream_id, &stream_model, usage.as_ref()) {
      let frame = SseFrame {
        event: None,
        data: serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string()),
      };
      out.push_str(&encode_sse_frame(&frame));

      if matches!(event, StreamEvent::Done { .. }) {
        out.push_str("data: [DONE]\n\n");
      }
    }
  }

  Ok(out)
}

pub fn rewrite_chat_to_responses(raw: &str) -> Result<String, StreamParseError> {
  let events = parse_openai_chat_stream(raw)?;

  let mut stream_id = "resp_stream".to_string();
  let mut stream_model = "unknown".to_string();
  let mut usage: Option<CoreUsage> = None;
  let mut seq: u32 = 0;
  let mut emitted_created = false;

  let mut out = String::new();

  for event in &events {
    if let StreamEvent::MessageStart { id, model } = event {
      if let Some(id) = id {
        stream_id = id.clone();
      }
      if let Some(model) = model {
        stream_model = model.clone();
      }
    }

    if let StreamEvent::Usage { usage: parsed_usage } = event {
      usage = Some(parsed_usage.clone());
      continue;
    }

    if !emitted_created {
      let frame = SseFrame {
        event: Some("response.created".to_string()),
        data: serde_json::to_string(&json!({
          "type": "response.created",
          "id": stream_id,
          "object": "response",
          "created_at": 0,
          "model": stream_model,
          "sequence_number": seq,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
      };
      out.push_str(&encode_sse_frame(&frame));
      seq = seq.saturating_add(1);
      emitted_created = true;
    }

    match event {
      StreamEvent::TextDelta { text } => {
        let frame = SseFrame {
          event: Some("response.output_text.delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "response.output_text.delta",
            "id": stream_id,
            "object": "response.output_text.delta",
            "created_at": 0,
            "model": stream_model,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "annotations": [],
            "sequence_number": seq,
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&frame));
        seq = seq.saturating_add(1);
      }
      StreamEvent::ReasoningDelta { text } => {
        let frame = SseFrame {
          event: Some("response.reasoning.delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "response.reasoning.delta",
            "id": stream_id,
            "object": "response.reasoning.delta",
            "created_at": 0,
            "model": stream_model,
            "delta": text,
            "sequence_number": seq,
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&frame));
        seq = seq.saturating_add(1);
      }
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => {
        let frame = SseFrame {
          event: Some("response.function_call.delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "response.function_call.delta",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "delta": arguments_delta,
            "sequence_number": seq,
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&frame));
        seq = seq.saturating_add(1);
      }
      StreamEvent::ToolCall {
        call_id,
        name,
        arguments,
        ..
      } => {
        let frame = SseFrame {
          event: Some("response.function_call.done".to_string()),
          data: serde_json::to_string(&json!({
            "type": "response.function_call.done",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": stringify_json(arguments),
            "sequence_number": seq,
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&frame));
        seq = seq.saturating_add(1);
      }
      StreamEvent::Done { finish_reason, .. } => {
        let frame = SseFrame {
          event: Some("response.completed".to_string()),
          data: serde_json::to_string(&json!({
            "type": "response.completed",
            "id": stream_id,
            "object": "response.completed",
            "created_at": 0,
            "model": stream_model,
            "status": if finish_reason.as_deref() == Some("tool_calls") { "requires_action" } else { "completed" },
            "finish_reason": finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
            "usage": usage,
            "sequence_number": seq,
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&frame));
        out.push_str("data: [DONE]\n\n");
      }
      StreamEvent::MessageStart { .. }
      | StreamEvent::Usage { .. }
      | StreamEvent::ToolResult { .. }
      | StreamEvent::Error { .. }
      | StreamEvent::Citation { .. } => {}
    }
  }

  Ok(out)
}

pub fn rewrite_chat_to_anthropic(raw: &str) -> Result<String, StreamParseError> {
  let events = parse_openai_chat_stream(raw)?;

  let mut stream_id = "anthropic_stream".to_string();
  let mut stream_model = "unknown".to_string();
  let mut usage: Option<CoreUsage> = None;

  let mut out = String::new();

  let mut message_started = false;
  let mut text_started = false;
  let mut thinking_started = false;
  let text_index = 0;
  let thinking_index = 1;
  let mut tool_indexes: HashMap<String, i64> = HashMap::new();
  let mut stopped_tool_indexes = HashSet::new();
  let mut next_index = 2;

  let ensure_message_start =
    |out: &mut String, message_started: &mut bool, stream_id: &str, stream_model: &str, usage: Option<&CoreUsage>| {
      if *message_started {
        return;
      }
      *message_started = true;

      let frame = SseFrame {
        event: Some("message_start".to_string()),
        data: serde_json::to_string(&json!({
          "type": "message_start",
          "message": {
            "id": stream_id,
            "type": "message",
            "role": "assistant",
            "model": stream_model,
            "content": [],
            "usage": {
              "input_tokens": usage.as_ref().map(|u| u.prompt_tokens).unwrap_or_default(),
              "output_tokens": usage.as_ref().map(|u| u.completion_tokens).unwrap_or_default(),
              "cache_read_input_tokens": usage.as_ref().and_then(|u| u.cached_tokens),
              "cache_creation_input_tokens": Value::Null,
            }
          }
        }))
        .unwrap_or_else(|_| "{}".to_string()),
      };
      out.push_str(&encode_sse_frame(&frame));
    };

  for event in &events {
    if let StreamEvent::MessageStart { id, model } = event {
      if let Some(id) = id {
        stream_id = id.clone();
      }
      if let Some(model) = model {
        stream_model = model.clone();
      }
      continue;
    }

    if let StreamEvent::Usage { usage: parsed_usage } = event {
      usage = Some(parsed_usage.clone());
      continue;
    }

    match event {
      StreamEvent::TextDelta { text } => {
        ensure_message_start(
          &mut out,
          &mut message_started,
          &stream_id,
          &stream_model,
          usage.as_ref(),
        );

        if !text_started {
          text_started = true;
          let start_frame = SseFrame {
            event: Some("content_block_start".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_start",
              "index": text_index,
              "content_block": {
                "type": "text",
                "text": "",
              }
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&start_frame));
        }

        let delta_frame = SseFrame {
          event: Some("content_block_delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "content_block_delta",
            "index": text_index,
            "delta": {
              "type": "text_delta",
              "text": text,
            }
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&delta_frame));
      }
      StreamEvent::ReasoningDelta { text } => {
        ensure_message_start(
          &mut out,
          &mut message_started,
          &stream_id,
          &stream_model,
          usage.as_ref(),
        );

        if !thinking_started {
          thinking_started = true;
          let start_frame = SseFrame {
            event: Some("content_block_start".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_start",
              "index": thinking_index,
              "content_block": {
                "type": "thinking",
                "thinking": "",
              }
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&start_frame));
        }

        let delta_frame = SseFrame {
          event: Some("content_block_delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "content_block_delta",
            "index": thinking_index,
            "delta": {
              "type": "thinking_delta",
              "thinking": text,
            }
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&delta_frame));
      }
      StreamEvent::ToolCallDelta {
        call_id,
        name,
        arguments_delta,
      } => {
        ensure_message_start(
          &mut out,
          &mut message_started,
          &stream_id,
          &stream_model,
          usage.as_ref(),
        );

        let tool_index = *tool_indexes.entry(call_id.clone()).or_insert_with(|| {
          let idx = next_index;
          next_index += 1;
          let start_frame = SseFrame {
            event: Some("content_block_start".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_start",
              "index": idx,
              "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
              }
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&start_frame));
          idx
        });

        let delta_frame = SseFrame {
          event: Some("content_block_delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "content_block_delta",
            "index": tool_index,
            "delta": {
              "type": "input_json_delta",
              "partial_json": arguments_delta,
            }
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&delta_frame));
      }
      StreamEvent::ToolCall { call_id, .. } => {
        if let Some(tool_index) = tool_indexes.get(call_id)
          && !stopped_tool_indexes.contains(tool_index)
        {
          let stop_frame = SseFrame {
            event: Some("content_block_stop".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_stop",
              "index": tool_index,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&stop_frame));
          stopped_tool_indexes.insert(*tool_index);
        }
      }
      StreamEvent::Done { finish_reason, .. } => {
        if thinking_started {
          let stop_frame = SseFrame {
            event: Some("content_block_stop".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_stop",
              "index": thinking_index,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&stop_frame));
        }

        if text_started {
          let stop_frame = SseFrame {
            event: Some("content_block_stop".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_stop",
              "index": text_index,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&stop_frame));
        }

        for tool_index in tool_indexes.values() {
          if stopped_tool_indexes.contains(tool_index) {
            continue;
          }

          let stop_frame = SseFrame {
            event: Some("content_block_stop".to_string()),
            data: serde_json::to_string(&json!({
              "type": "content_block_stop",
              "index": tool_index,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
          };
          out.push_str(&encode_sse_frame(&stop_frame));
          stopped_tool_indexes.insert(*tool_index);
        }

        let finish_reason = finish_reason.clone().unwrap_or_else(|| "stop".to_string());

        let message_delta = SseFrame {
          event: Some("message_delta".to_string()),
          data: serde_json::to_string(&json!({
            "type": "message_delta",
            "delta": {
              "stop_reason": finish_reason,
              "stop_sequence": Value::Null,
            },
            "usage": {
              "input_tokens": usage.as_ref().map(|u| u.prompt_tokens).unwrap_or_default(),
              "output_tokens": usage.as_ref().map(|u| u.completion_tokens).unwrap_or_default(),
              "cache_read_input_tokens": usage.as_ref().and_then(|u| u.cached_tokens),
              "cache_creation_input_tokens": Value::Null,
            }
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&message_delta));

        let message_stop = SseFrame {
          event: Some("message_stop".to_string()),
          data: serde_json::to_string(&json!({
            "type": "message_stop",
          }))
          .unwrap_or_else(|_| "{}".to_string()),
        };
        out.push_str(&encode_sse_frame(&message_stop));
        out.push_str("data: [DONE]\n\n");
      }
      StreamEvent::ToolResult { .. } | StreamEvent::Error { .. } | StreamEvent::Citation { .. } => {}
      StreamEvent::MessageStart { .. } | StreamEvent::Usage { .. } => {}
    }
  }

  Ok(out)
}
