use serde_json::{Value, json};

use super::{
  super::{
    core::StreamEvent,
    test_support::{
      MockHttpClient, MockHttpResponse, sample_backend_config, sample_backend_config_with_header, sample_request,
      sse_done, sse_event,
    },
  },
  BackendError, BackendProtocol, BackendRequestLayer, HttpResponse, HttpStreamResponse, dispatch_request,
  dispatch_stream, dispatch_stream_with_handler,
};

#[test]
fn should_dispatch_openai_chat_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "chat_1",
      "model": "gpt-4.1",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Hi!"
        },
        "finish_reason": "stop"
      }],
      "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
        "prompt_tokens_details": {
          "cached_tokens": 3
        }
      }
    }),
  }))]);

  let response = dispatch_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiChatCompletions,
    &sample_request(),
  )
  .unwrap();

  assert_eq!(response.id, "chat_1");

  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(requests[0].url, "https://api.example.com/v1/chat/completions");
  assert_eq!(requests[0].body["stream"], Value::Bool(false));
  assert_eq!(
    requests[0].headers,
    vec![
      ("accept".to_string(), "application/json".to_string()),
      ("authorization".to_string(), "Bearer token-1".to_string()),
      ("content-type".to_string(), "application/json".to_string()),
      ("x-test-header".to_string(), "1".to_string()),
    ]
  );
}

#[test]
fn should_dispatch_openai_responses_stream() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: vec![
      sse_event(
        "response.created",
        json!({
          "type": "response.created",
          "id": "resp_1",
          "model": "gpt-4.1",
        }),
      ),
      sse_event(
        "response.output_text.delta",
        json!({
          "type": "response.output_text.delta",
          "id": "resp_1",
          "model": "gpt-4.1",
          "delta": "Hi",
        }),
      ),
      sse_event(
        "response.function_call.delta",
        json!({
          "type": "response.function_call.delta",
          "call_id": "call_1",
          "name": "doc_read",
          "delta": r#"{"docId":"#,
        }),
      ),
      sse_event(
        "response.function_call.done",
        json!({
          "type": "response.function_call.done",
          "call_id": "call_1",
          "name": "doc_read",
          "arguments": r#"{"docId":"a1"}"#,
        }),
      ),
      sse_event(
        "response.completed",
        json!({
          "type": "response.completed",
          "status": "requires_action",
          "finish_reason": "tool_calls",
          "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
          },
        }),
      ),
      sse_done(),
    ]
    .concat(),
  }))]);

  let events = dispatch_stream(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &sample_request(),
  )
  .unwrap();

  assert!(!events.is_empty());
  assert!(events.iter().any(|event| matches!(event, StreamEvent::Done { .. })));

  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(requests[0].url, "https://api.example.com/v1/responses");
  assert_eq!(requests[0].body["stream"], Value::Bool(true));
}

#[test]
fn should_dispatch_stream_with_handler_incrementally() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: vec![
      sse_event(
        "response.created",
        json!({
          "type": "response.created",
          "id": "resp_2",
          "model": "gpt-4.1",
        }),
      ),
      sse_event(
        "response.output_text.delta",
        json!({
          "type": "response.output_text.delta",
          "id": "resp_2",
          "model": "gpt-4.1",
          "delta": "streaming",
        }),
      ),
      sse_done(),
    ]
    .concat(),
  }))]);

  let mut seen = Vec::new();
  let result = dispatch_stream_with_handler(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &sample_request(),
    &mut |event| {
      seen.push(event.clone());
      Err(BackendError::Http("stop_after_first_event".to_string()))
    },
  );

  assert!(matches!(result, Err(BackendError::Http(reason)) if reason == "stop_after_first_event"));
  assert_eq!(seen.len(), 1);
}

#[test]
fn should_dispatch_anthropic_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "msg_1",
      "model": "claude-sonnet-4-5-20250929",
      "role": "assistant",
      "content": [
        { "type": "thinking", "thinking": "analyzing" },
        { "type": "text", "text": "Done" }
      ],
      "stop_reason": "end_turn",
      "usage": {
        "input_tokens": 9,
        "output_tokens": 3,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0
      }
    }),
  }))]);

  let mut request = sample_request();
  request.model = "claude-sonnet-4-5-20250929".to_string();

  let response = dispatch_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::AnthropicMessages,
    &request,
  )
  .unwrap();

  assert_eq!(response.id, "msg_1");

  let requests = client.requests();
  assert_eq!(requests[0].url, "https://api.example.com/v1/messages");
  assert_eq!(requests[0].body["stream"], Value::Bool(false));
  assert_eq!(
    requests[0].headers,
    vec![
      ("accept".to_string(), "application/json".to_string()),
      ("anthropic-version".to_string(), "2023-06-01".to_string()),
      ("content-type".to_string(), "application/json".to_string()),
      ("x-api-key".to_string(), "token-1".to_string()),
      ("x-test-header".to_string(), "1".to_string()),
    ]
  );
}

#[test]
fn should_apply_anthropic_request_defaults_in_dispatch() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "msg_2",
      "model": "claude-sonnet-4-5-20250929",
      "role": "assistant",
      "content": [{ "type": "text", "text": "ok" }],
      "stop_reason": "end_turn",
      "usage": { "input_tokens": 1, "output_tokens": 1 }
    }),
  }))]);

  let mut request = sample_request();
  request.model = "claude-sonnet-4-5-20250929".to_string();
  request.max_tokens = None;
  request.temperature = Some(0.4);

  dispatch_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::AnthropicMessages,
    &request,
  )
  .unwrap();
  let requests = client.requests();
  assert_eq!(requests[0].body["max_tokens"], 4096);
  assert_eq!(requests[0].body["temperature"], 0.4);
}

#[test]
fn should_dispatch_vertex_anthropic_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "msg_vrtx_1",
      "model": "claude-sonnet-4-5@20250929",
      "role": "assistant",
      "content": [{ "type": "text", "text": "ok" }],
      "stop_reason": "end_turn",
      "usage": { "input_tokens": 1, "output_tokens": 1 }
    }),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url =
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic".to_string();
  config.request_layer = Some(BackendRequestLayer::Vertex);

  let mut request = sample_request();
  request.model = "claude-sonnet-4-5@20250929".to_string();

  dispatch_request(&client, &config, BackendProtocol::AnthropicMessages, &request).unwrap();
  let requests = client.requests();

  assert_eq!(
    requests[0].url,
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
  );
  assert!(requests[0].body.get("model").is_none());
  assert_eq!(requests[0].body["anthropic_version"], "vertex-2023-10-16");
  assert_eq!(
    requests[0].headers,
    vec![
      ("accept".to_string(), "application/json".to_string()),
      ("authorization".to_string(), "Bearer token-1".to_string()),
      ("content-type".to_string(), "application/json".to_string()),
      ("x-test-header".to_string(), "1".to_string()),
    ]
  );
}

#[test]
fn should_dispatch_vertex_anthropic_stream() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: vec![
      sse_event(
        "message_start",
        json!({
          "type": "message_start",
          "message": {
            "id": "msg_vrtx_2",
            "model": "claude-sonnet-4-5@20250929",
            "usage": { "input_tokens": 8, "output_tokens": 0 },
          },
        }),
      ),
      sse_event(
        "content_block_delta",
        json!({
          "type": "content_block_delta",
          "index": 0,
          "delta": { "type": "text_delta", "text": "Hi" },
        }),
      ),
      sse_event(
        "message_delta",
        json!({
          "type": "message_delta",
          "delta": { "stop_reason": "end_turn" },
          "usage": { "input_tokens": 8, "output_tokens": 2 },
        }),
      ),
      sse_event("message_stop", json!({ "type": "message_stop" })),
      sse_done(),
    ]
    .concat(),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url =
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic".to_string();
  config.request_layer = Some(BackendRequestLayer::Vertex);

  let mut request = sample_request();
  request.model = "claude-sonnet-4-5@20250929".to_string();

  let events = dispatch_stream(&client, &config, BackendProtocol::AnthropicMessages, &request).unwrap();
  assert!(events.iter().any(|event| matches!(event, StreamEvent::Done { .. })));

  let requests = client.requests();
  assert_eq!(
    requests[0].url,
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict",
  );
  assert_eq!(requests[0].body["stream"], Value::Bool(true));
  assert_eq!(requests[0].body["anthropic_version"], "vertex-2023-10-16");
}

#[test]
fn should_reject_incompatible_request_layer() {
  let client = MockHttpClient::default();
  let mut config = sample_backend_config(false);
  config.request_layer = Some(BackendRequestLayer::Responses);

  let result = dispatch_request(&client, &config, BackendProtocol::AnthropicMessages, &sample_request());
  assert!(matches!(result, Err(BackendError::InvalidConfig(_))));
}

#[test]
fn should_surface_upstream_error() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
    status: 500,
    body: "boom".to_string(),
  }))]);

  let result = dispatch_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiChatCompletions,
    &sample_request(),
  );

  assert!(matches!(result, Err(BackendError::UpstreamStatus { status: 500, .. })));
}
