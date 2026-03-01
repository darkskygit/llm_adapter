use serde_json::{Value, json};

use super::{ProtocolError, anthropic, openai};
use crate::{core::CoreResponse, test_support::sample_response_with_reasoning_tool_call};

#[test]
fn decode_should_support_string_number_and_call_id_fallback_across_providers() {
  struct Case {
    name: &'static str,
    decode: fn(&Value) -> Result<CoreResponse, ProtocolError>,
    input: Value,
    expected_prompt_tokens: u32,
    expected_completion_tokens: u32,
    expected_total_tokens: u32,
    expected_cached_tokens: Option<u32>,
    expected_call_id: &'static str,
    expected_finish_reason: Option<&'static str>,
    expected_tool_result_output: Option<Value>,
  }

  let cases = vec![
    Case {
      name: "openai_chat",
      decode: openai::chat::response::decode,
      input: json!({
        "id": "chat_3",
        "model": "gpt-4.1",
        "choices": [{
          "index": 0,
          "message": {
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
              "call_id": "call_fallback_1",
              "type": "function",
              "function": { "name": "doc_read", "arguments": "{\"docId\":\"a1\"}" }
            }]
          },
          "finish_reason": "tool_calls"
        }],
        "usage": {
          "prompt_tokens": "10",
          "completion_tokens": "2",
          "total_tokens": "12",
          "prompt_tokens_details": { "cached_tokens": "3" }
        }
      }),
      expected_prompt_tokens: 10,
      expected_completion_tokens: 2,
      expected_total_tokens: 12,
      expected_cached_tokens: Some(3),
      expected_call_id: "call_fallback_1",
      expected_finish_reason: None,
      expected_tool_result_output: None,
    },
    Case {
      name: "openai_responses",
      decode: openai::responses::response::decode,
      input: json!({
        "id": "resp_3",
        "model": "gpt-4.1",
        "status": "completed",
        "usage": {
          "input_tokens": "8",
          "output_tokens": "4",
          "total_tokens": "12",
          "input_tokens_details": { "cached_tokens": "2" }
        },
        "output": [{
          "type": "function_call",
          "id": "call_fallback_2",
          "name": "doc_read",
          "arguments": "{\"docId\":\"a2\"}"
        }]
      }),
      expected_prompt_tokens: 8,
      expected_completion_tokens: 4,
      expected_total_tokens: 12,
      expected_cached_tokens: Some(2),
      expected_call_id: "call_fallback_2",
      expected_finish_reason: None,
      expected_tool_result_output: None,
    },
    Case {
      name: "anthropic",
      decode: anthropic::response::decode,
      input: json!({
        "id": "msg_3",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
          { "type": "thinking", "text": "drafting plan" },
          { "type": "tool_use", "name": "doc_read", "input": { "docId": "a3" } },
          { "type": "tool_result", "content": "{\"ok\":true}" }
        ],
        "usage": {
          "input_tokens": "7",
          "output_tokens": "5",
          "cache_read_input_tokens": "2",
          "cache_creation_input_tokens": "1"
        },
        "stop_reason": "max_tokens"
      }),
      expected_prompt_tokens: 10,
      expected_completion_tokens: 5,
      expected_total_tokens: 15,
      expected_cached_tokens: Some(2),
      expected_call_id: "call_0",
      expected_finish_reason: Some("length"),
      expected_tool_result_output: Some(json!({ "ok": true })),
    },
  ];

  for case in cases {
    let core = (case.decode)(&case.input).unwrap_or_else(|error| panic!("case `{}` failed: {error}", case.name));

    assert_eq!(
      core.usage.prompt_tokens, case.expected_prompt_tokens,
      "{} prompt_tokens",
      case.name
    );
    assert_eq!(
      core.usage.completion_tokens, case.expected_completion_tokens,
      "{} completion_tokens",
      case.name
    );
    assert_eq!(
      core.usage.total_tokens, case.expected_total_tokens,
      "{} total_tokens",
      case.name
    );
    assert_eq!(
      core.usage.cached_tokens, case.expected_cached_tokens,
      "{} cached_tokens",
      case.name
    );

    assert!(
      core.message.content.iter().any(|content| matches!(
        content,
        crate::core::CoreContent::ToolCall { call_id, .. } if call_id == case.expected_call_id
      )),
      "{} call_id fallback",
      case.name
    );

    if let Some(finish_reason) = case.expected_finish_reason {
      assert_eq!(core.finish_reason, finish_reason, "{} finish_reason", case.name);
    }

    if let Some(expected_output) = case.expected_tool_result_output {
      assert!(
        core.message.content.iter().any(|content| matches!(
          content,
          crate::core::CoreContent::ToolResult { call_id, output, .. }
            if call_id == case.expected_call_id && output == &expected_output
        )),
        "{} tool_result output",
        case.name
      );
    }
  }
}

#[test]
fn encode_should_preserve_tool_call_reasoning_and_usage_across_providers() {
  struct Case {
    encode: fn(&CoreResponse) -> Value,
    verify: fn(&Value),
  }

  fn verify_chat(payload: &Value) {
    assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(payload["usage"]["prompt_tokens_details"]["cached_tokens"], 12);
    assert_eq!(
      payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
      "doc_read"
    );
  }

  fn verify_responses(payload: &Value) {
    assert_eq!(payload["status"], "requires_action");
    assert_eq!(payload["usage"]["input_tokens_details"]["cached_tokens"], 12);
    assert_eq!(payload["output"][0]["type"], "function_call");
  }

  fn verify_anthropic(payload: &Value) {
    assert_eq!(payload["type"], "message");
    assert_eq!(payload["usage"]["cache_read_input_tokens"], 12);
    assert_eq!(payload["content"][0]["type"], "thinking");
  }

  let cases = [
    Case {
      encode: openai::chat::response::encode,
      verify: verify_chat,
    },
    Case {
      encode: openai::responses::response::encode,
      verify: verify_responses,
    },
    Case {
      encode: anthropic::response::encode,
      verify: verify_anthropic,
    },
  ];

  let response = sample_response_with_reasoning_tool_call();

  for case in cases {
    let payload = (case.encode)(&response);
    (case.verify)(&payload);
  }
}
