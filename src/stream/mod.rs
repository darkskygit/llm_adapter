use thiserror::Error;

mod parse;
mod rewrite;
mod sse;

pub use parse::{
  AnthropicStreamParser, OpenaiChatStreamParser, OpenaiResponsesStreamParser, parse_anthropic_stream,
  parse_openai_chat_stream, parse_openai_responses_stream,
};
pub use rewrite::{rewrite_anthropic_to_chat, rewrite_chat_to_anthropic, rewrite_chat_to_responses};
pub use sse::{SseFrameDecoder, encode_sse_frame, parse_sse_frames};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseFrame {
  pub event: Option<String>,
  pub data: String,
}

#[derive(Debug, Error)]
pub enum StreamParseError {
  #[error("invalid json in {context}: {source}")]
  InvalidJson {
    context: &'static str,
    #[source]
    source: serde_json::Error,
  },
}

#[cfg(test)]
mod tests {
  use serde_json::{Value, json};

  use super::{super::core::StreamEvent, *};

  fn sample_chat_stream() -> String {
    let mut out = String::new();
    let chunk1 = json!({
      "id": "chat_1",
      "object": "chat.completion.chunk",
      "model": "gpt-4.1",
      "choices": [{
        "index": 0,
        "delta": {
          "role": "assistant",
          "content": "Hello ",
          "tool_calls": [{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "doc_read",
              "arguments": "{\"docId\":\""
            }
          }]
        },
        "finish_reason": Value::Null
      }]
    });
    out.push_str("data: ");
    out.push_str(&serde_json::to_string(&chunk1).unwrap());
    out.push_str("\n\n");

    let chunk2 = json!({
      "id": "chat_1",
      "object": "chat.completion.chunk",
      "model": "gpt-4.1",
      "citations": ["https://affine.pro"],
      "choices": [{
        "index": 0,
        "delta": {
          "content": "world",
          "tool_calls": [{
            "index": 0,
            "function": {
              "arguments": "a1\"}"
            }
          }]
        },
        "finish_reason": "tool_calls"
      }],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
      }
    });
    out.push_str("data: ");
    out.push_str(&serde_json::to_string(&chunk2).unwrap());
    out.push_str("\n\n");

    out.push_str("data: [DONE]\n\n");
    out
  }

  fn event_index(frames: &[SseFrame], name: &str) -> usize {
    frames
      .iter()
      .position(|frame| frame.event.as_deref() == Some(name))
      .unwrap_or_else(|| panic!("missing event `{name}`"))
  }

  #[test]
  fn should_parse_sse_frames() {
    let raw = "event: ping\ndata: 1\n\ndata: [DONE]\n\n";
    let frames = parse_sse_frames(raw);

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].event.as_deref(), Some("ping"));
    assert_eq!(frames[0].data, "1");
    assert_eq!(frames[1].data, "[DONE]");
  }

  #[test]
  fn should_parse_openai_chat_stream_with_tool_calls() {
    let parsed = parse_openai_chat_stream(&sample_chat_stream()).unwrap();

    assert!(
      parsed
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "Hello "))
    );
    assert!(
      parsed
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCallDelta { call_id, .. } if call_id == "call_1"))
    );
    assert!(parsed.iter().any(
      |event| matches!(event, StreamEvent::ToolCall { call_id, name, .. } if call_id == "call_1" && name == "doc_read")
    ));
    assert!(parsed.iter().any(
      |event| matches!(event, StreamEvent::Citation { index, url } if *index == 1 && url == "https://affine.pro")
    ));
    assert!(
      parsed
        .iter()
        .any(|event| matches!(event, StreamEvent::Done { finish_reason: Some(reason), .. } if reason == "tool_calls"))
    );
  }

  #[test]
  fn should_parse_openai_chat_stream_with_utf8_text_delta() {
    let mut raw = String::new();
    let chunk = json!({
      "id": "chat_utf8",
      "object": "chat.completion.chunk",
      "model": "gpt-4.1",
      "choices": [{
        "index": 0,
        "delta": {
          "role": "assistant",
          "content": "你好世界"
        },
        "finish_reason": "stop"
      }],
      "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3
      }
    });
    raw.push_str("data: ");
    raw.push_str(&serde_json::to_string(&chunk).unwrap());
    raw.push_str("\n\n");
    raw.push_str("data: [DONE]\n\n");

    let parsed = parse_openai_chat_stream(&raw).unwrap();
    assert!(
      parsed
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "你好世界"))
    );
  }

  #[test]
  fn should_rewrite_anthropic_to_chat() {
    let raw = concat!(
      "event: message_start\n",
      "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3\",\"usage\":{\"\
       input_tokens\":8,\"output_tokens\":0}}}\n\n",
      "event: content_block_delta\n",
      "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
      "event: message_delta\n",
      "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":8,\"\
       output_tokens\":2}}\n\n",
      "event: message_stop\n",
      "data: {\"type\":\"message_stop\"}\n\n",
      "data: [DONE]\n\n"
    );

    let rewritten = rewrite_anthropic_to_chat(raw).unwrap();
    let frames = parse_sse_frames(&rewritten);
    assert!(matches!(frames.last(), Some(frame) if frame.data == "[DONE]"));

    let events = parse_openai_chat_stream(&rewritten).unwrap();
    assert!(
      events
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "Hi"))
    );
    assert!(
      events
        .iter()
        .any(|event| matches!(event, StreamEvent::Done { finish_reason: Some(reason), .. } if reason == "stop"))
    );
  }

  #[test]
  fn should_rewrite_chat_to_responses() {
    let rewritten = rewrite_chat_to_responses(&sample_chat_stream()).unwrap();
    let frames = parse_sse_frames(&rewritten);
    assert!(matches!(frames.last(), Some(frame) if frame.data == "[DONE]"));

    let created = event_index(&frames, "response.created");
    let text_delta = event_index(&frames, "response.output_text.delta");
    let tool_delta = event_index(&frames, "response.function_call.delta");
    let completed = event_index(&frames, "response.completed");
    assert!(created < text_delta);
    assert!(text_delta < tool_delta);
    assert!(tool_delta < completed);

    let completed_payload: Value = serde_json::from_str(&frames[completed].data).unwrap();
    assert_eq!(completed_payload["finish_reason"], "tool_calls");

    let events = parse_openai_responses_stream(&rewritten).unwrap();
    assert!(events.iter().any(
      |event| matches!(event, StreamEvent::ToolCall { call_id, name, .. } if call_id == "call_1" && name == "doc_read")
    ));
    assert!(
      events
        .iter()
        .any(|event| matches!(event, StreamEvent::Done { finish_reason: Some(reason), .. } if reason == "tool_calls"))
    );
  }

  #[test]
  fn should_rewrite_chat_to_anthropic() {
    let rewritten = rewrite_chat_to_anthropic(&sample_chat_stream()).unwrap();
    let frames = parse_sse_frames(&rewritten);
    assert!(matches!(frames.last(), Some(frame) if frame.data == "[DONE]"));

    let message_start = event_index(&frames, "message_start");
    let text_delta = event_index(&frames, "content_block_delta");
    let message_delta = event_index(&frames, "message_delta");
    let message_stop = event_index(&frames, "message_stop");
    assert!(message_start < text_delta);
    assert!(text_delta < message_delta);
    assert!(message_delta < message_stop);

    let message_delta_payload: Value = serde_json::from_str(&frames[message_delta].data).unwrap();
    assert_eq!(message_delta_payload["delta"]["stop_reason"], "tool_calls");

    let events = parse_anthropic_stream(&rewritten).unwrap();
    assert!(matches!(events.first(), Some(StreamEvent::MessageStart { .. })));
    assert!(events.iter().any(
      |event| matches!(event, StreamEvent::ToolCall { call_id, name, .. } if call_id == "call_1" && name == "doc_read")
    ));
    assert!(matches!(
      events.last(),
      Some(StreamEvent::Done {
        finish_reason: Some(reason),
        ..
      }) if reason == "tool_calls"
    ));
  }

  #[test]
  fn stream_golden_should_preserve_order_tool_call_and_finish_reason_mapping() {
    let raw = concat!(
      "event: message_start\n",
      "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3\",\"usage\":{\"\
       input_tokens\":8,\"output_tokens\":0}}}\n\n",
      "event: content_block_start\n",
      "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\
       \"name\":\"doc_read\",\"input\":{}}}\n\n",
      "event: content_block_delta\n",
      r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"docId\":\"a1\"}"}}"#,
      "\n\n",
      "event: content_block_stop\n",
      "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
      "event: message_delta\n",
      "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":8,\"\
       output_tokens\":2}}\n\n",
      "event: message_stop\n",
      "data: {\"type\":\"message_stop\"}\n\n",
      "data: [DONE]\n\n"
    );

    let events = parse_anthropic_stream(raw).unwrap();

    assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(
      events[1],
      StreamEvent::ToolCallDelta { ref call_id, .. } if call_id == "call_1"
    ));
    assert!(matches!(
      events[2],
      StreamEvent::ToolCall { ref call_id, ref name, .. } if call_id == "call_1" && name == "doc_read"
    ));
    assert!(matches!(events[3], StreamEvent::Usage { .. }));
    assert!(matches!(
      events[4],
      StreamEvent::Done {
        finish_reason: Some(ref reason),
        ..
      } if reason == "stop"
    ));
  }
}
