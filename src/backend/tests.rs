use serde_json::{Value, json};

use super::{
  super::{
    core::{
      CoreContent, CoreMessage, CoreRole, EmbeddingRequest, RerankCandidate, RerankRequest, StreamEvent,
      StructuredRequest,
    },
    stream::StreamEncodingTarget,
    test_support::{
      MockHttpClient, MockHttpResponse, sample_backend_config, sample_backend_config_with_header, sample_request,
      sse_done, sse_event,
    },
  },
  BackendError, BackendProtocol, BackendRequestLayer, HttpResponse, HttpStreamResponse, collect_stream_events,
  dispatch_embedding_request, dispatch_request, dispatch_rerank_request, dispatch_stream_encoded_with,
  dispatch_stream_events_with, dispatch_structured_request,
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
fn should_dispatch_gemini_api_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "responseId": "gem_resp_1",
      "modelVersion": "gemini-2.5-flash",
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [{ "text": "Hi!" }]
        },
        "finishReason": "FINISH_REASON_STOP"
      }],
      "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 4,
        "totalTokenCount": 16
      }
    }),
  }))]);

  let mut request = sample_request();
  request.model = "gemini-2.5-flash".to_string();

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiApi);

  let response = dispatch_request(&client, &config, BackendProtocol::GeminiGenerateContent, &request).unwrap();
  assert_eq!(response.id, "gem_resp_1");

  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(
    requests[0].url,
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
  );
  assert!(requests[0].body.get("model").is_none());
  assert!(requests[0].body.get("stream").is_none());
  assert_eq!(requests[0].body["generationConfig"]["maxOutputTokens"], 128);
  assert_eq!(
    requests[0].headers,
    vec![
      ("accept".to_string(), "application/json".to_string()),
      ("content-type".to_string(), "application/json".to_string()),
      ("x-goog-api-key".to_string(), "token-1".to_string()),
      ("x-test-header".to_string(), "1".to_string()),
    ]
  );
}

#[test]
fn should_dispatch_openai_structured_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "resp_struct_1",
      "model": "gpt-4.1",
      "status": "completed",
      "output": [{
        "type": "message",
        "role": "assistant",
        "content": [{
          "type": "output_text",
          "text": "{\"summary\":\"AFFiNE\"}"
        }]
      }],
      "usage": {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14
      }
    }),
  }))]);

  let request = StructuredRequest {
    model: "gpt-4.1".to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "Summarize AFFiNE.".to_string(),
      }],
    }],
    schema: json!({
      "type": "object",
      "properties": {
        "summary": { "type": "string" }
      },
      "required": ["summary"],
      "additionalProperties": false
    }),
    max_tokens: Some(64),
    temperature: Some(0.2),
    reasoning: Some(json!({ "effort": "medium" })),
    strict: Some(true),
    response_mime_type: Some("application/json".to_string()),
  };

  let response = dispatch_structured_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &request,
  )
  .unwrap();

  assert_eq!(response.output_text, "{\"summary\":\"AFFiNE\"}");
  assert_eq!(response.output_json, Some(json!({ "summary": "AFFiNE" })));
  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(requests[0].url, "https://api.example.com/v1/responses");
  assert_eq!(requests[0].body["text"]["format"]["type"], "json_schema");
  assert_eq!(requests[0].body["text"]["format"]["schema"], request.schema);
}

#[test]
fn should_extract_output_json_from_fenced_structured_response() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "resp_2",
      "model": "gpt-4.1",
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "```json\n{\"summary\":\"AFFiNE\"}\n```"
        }]
      }],
      "usage": {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16
      }
    }),
  }))]);

  let request = StructuredRequest {
    model: "gpt-4.1".to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "Summarize AFFiNE.".to_string(),
      }],
    }],
    schema: json!({
      "type": "object",
      "properties": {
        "summary": { "type": "string" }
      },
      "required": ["summary"]
    }),
    max_tokens: Some(64),
    temperature: None,
    reasoning: None,
    strict: Some(true),
    response_mime_type: Some("application/json".to_string()),
  };

  let response = dispatch_structured_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &request,
  )
  .unwrap();

  assert_eq!(
    response.output_json,
    Some(json!({
      "summary": "AFFiNE"
    }))
  );
}

#[test]
fn should_fail_structured_dispatch_with_typed_error_when_output_is_not_json() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "id": "resp_3",
      "model": "gpt-4.1",
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "summary: AFFiNE"
        }]
      }],
      "usage": {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16
      }
    }),
  }))]);

  let request = StructuredRequest {
    model: "gpt-4.1".to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "Summarize AFFiNE.".to_string(),
      }],
    }],
    schema: json!({
      "type": "object",
      "properties": {
        "summary": { "type": "string" }
      },
      "required": ["summary"]
    }),
    max_tokens: Some(64),
    temperature: None,
    reasoning: None,
    strict: Some(true),
    response_mime_type: Some("application/json".to_string()),
  };

  let error = dispatch_structured_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &request,
  )
  .unwrap_err();

  assert!(matches!(
    error,
    BackendError::InvalidStructuredOutput { .. }
  ));
  assert_eq!(
    error.to_string(),
    "invalid_structured_output: structured response did not contain valid JSON: summary: AFFiNE"
  );
}

#[test]
fn should_dispatch_gemini_embedding_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "embeddings": [
        { "values": [0.1, 0.2] },
        { "values": [0.3, 0.4] }
      ]
    }),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiApi);

  let response = dispatch_embedding_request(
    &client,
    &config,
    BackendProtocol::GeminiGenerateContent,
    &EmbeddingRequest {
      model: "gemini-embedding-001".to_string(),
      inputs: vec!["hello".to_string(), "world".to_string()],
      dimensions: Some(256),
      task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
    },
  )
  .unwrap();

  assert_eq!(response.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
  let requests = client.requests();
  assert_eq!(
    requests[0].url,
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
  );
  assert_eq!(requests[0].body["requests"][0]["model"], "models/gemini-embedding-001");
  assert_eq!(requests[0].body["requests"][0]["outputDimensionality"], 256);
  assert_eq!(requests[0].body["requests"][0]["taskType"], "RETRIEVAL_DOCUMENT");
}

#[test]
fn should_dispatch_gemini_vertex_embedding_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "predictions": [{
        "embeddings": {
          "values": [0.1, 0.2],
          "statistics": { "token_count": 3 }
        }
      }]
    }),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://vertex.example/v1/projects/p1/locations/us-central1/publishers/google".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiVertex);

  let response = dispatch_embedding_request(
    &client,
    &config,
    BackendProtocol::GeminiGenerateContent,
    &EmbeddingRequest {
      model: "gemini-embedding-001".to_string(),
      inputs: vec!["hello".to_string()],
      dimensions: Some(128),
      task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
    },
  )
  .unwrap();

  assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);
  assert_eq!(response.usage.unwrap().total_tokens, 3);
  let requests = client.requests();
  assert_eq!(
    requests[0].url,
    "https://vertex.example/v1/projects/p1/locations/us-central1/publishers/google/models/gemini-embedding-001:predict"
  );
  assert_eq!(requests[0].body["instances"][0]["task_type"], "RETRIEVAL_DOCUMENT");
  assert_eq!(requests[0].body["parameters"]["outputDimensionality"], 128);
}

#[test]
fn should_dispatch_openai_rerank_request() {
  let client = MockHttpClient::with_json_responses(vec![
    MockHttpResponse::Json(Ok(HttpResponse {
      status: 200,
      body: json!({
        "model": "gpt-5.2",
        "choices": [{
          "logprobs": {
            "content": [{
              "top_logprobs": [
                { "token": " Yes", "logprob": -0.1 },
                { "token": " No", "logprob": -2.0 }
              ]
            }]
          }
        }]
      }),
    })),
    MockHttpResponse::Json(Ok(HttpResponse {
      status: 200,
      body: json!({
        "model": "gpt-5.2",
        "choices": [{
          "logprobs": {
            "content": [{
              "top_logprobs": [
                { "token": " Yes", "logprob": -2.0 },
                { "token": " No", "logprob": -0.1 }
              ]
            }]
          }
        }]
      }),
    })),
  ]);

  let response = dispatch_rerank_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiChatCompletions,
    &RerankRequest {
      model: "gpt-5.2".to_string(),
      query: "programming".to_string(),
      candidates: vec![
        RerankCandidate {
          id: Some("js".to_string()),
          text: "Is JavaScript relevant?".to_string(),
        },
        RerankCandidate {
          id: Some("weather".to_string()),
          text: "Is weather relevant?".to_string(),
        },
      ],
      top_n: None,
    },
  )
  .unwrap();

  assert_eq!(response.model, "gpt-5.2");
  assert_eq!(response.scores.len(), 2);
  assert!(response.scores[0] > 0.8);
  assert!(response.scores[1] < 0.2);

  let requests = client.requests();
  assert_eq!(requests.len(), 2);
  assert_eq!(requests[0].url, "https://api.example.com/v1/chat/completions");
  assert_eq!(requests[0].body["logprobs"], true);
  assert_eq!(requests[0].body["top_logprobs"], 5);
  assert_eq!(requests[0].body["max_completion_tokens"], 16);
  assert_eq!(requests[0].body["reasoning_effort"], "none");
}

#[test]
fn should_dispatch_cloudflare_rerank_request() {
  let client = MockHttpClient::with_json_responses(vec![
    MockHttpResponse::Json(Ok(HttpResponse {
      status: 200,
      body: json!({
        "model": "@cf/moonshotai/kimi-k2.5",
        "choices": [{
          "logprobs": {
            "content": [{
              "top_logprobs": [
                { "token": "Yes", "logprob": -0.1 },
                { "token": "No", "logprob": -2.0 }
              ]
            }]
          }
        }]
      }),
    })),
    MockHttpResponse::Json(Ok(HttpResponse {
      status: 200,
      body: json!({
        "model": "@cf/moonshotai/kimi-k2.5",
        "choices": [{
          "logprobs": {
            "content": [{
              "top_logprobs": [
                { "token": "Yes", "logprob": -2.0 },
                { "token": "No", "logprob": -0.1 }
              ]
            }]
          }
        }]
      }),
    })),
  ]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://api.cloudflare.com/client/v4/accounts/a1/ai".to_string();
  config.request_layer = Some(BackendRequestLayer::CloudflareWorkersAi);

  let response = dispatch_rerank_request(
    &client,
    &config,
    BackendProtocol::OpenaiChatCompletions,
    &RerankRequest {
      model: "@cf/moonshotai/kimi-k2.5".to_string(),
      query: "programming".to_string(),
      candidates: vec![
        RerankCandidate {
          id: Some("rust".to_string()),
          text: "Rust ownership guide".to_string(),
        },
        RerankCandidate {
          id: Some("weather".to_string()),
          text: "Sunny and warm tomorrow".to_string(),
        },
      ],
      top_n: None,
    },
  )
  .unwrap();

  assert_eq!(response.model, "@cf/moonshotai/kimi-k2.5");
  assert!(response.scores[0] > 0.8);
  assert!(response.scores[1] < 0.2);

  let requests = client.requests();
  assert_eq!(requests.len(), 2);
  assert_eq!(
    requests[0].url,
    "https://api.cloudflare.com/client/v4/accounts/a1/ai/v1/chat/completions"
  );
  assert_eq!(requests[0].body["logprobs"], true);
  assert_eq!(requests[0].body["top_logprobs"], 5);
  assert!(requests[0].body.get("chat_template_kwargs").is_none());
}

#[test]
fn should_disable_thinking_for_cloudflare_logprobs_profiles() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "model": "@cf/zai-org/glm-4.7-flash",
      "choices": [{
        "logprobs": {
          "content": [{
            "top_logprobs": [
              { "token": "Yes", "logprob": -0.1 },
              { "token": "No", "logprob": -2.0 }
            ]
          }]
        }
      }]
    }),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://api.cloudflare.com/client/v4/accounts/a1/ai".to_string();
  config.request_layer = Some(BackendRequestLayer::CloudflareWorkersAi);

  let response = dispatch_rerank_request(
    &client,
    &config,
    BackendProtocol::OpenaiChatCompletions,
    &RerankRequest {
      model: "@cf/zai-org/glm-4.7-flash".to_string(),
      query: "programming".to_string(),
      candidates: vec![RerankCandidate {
        id: Some("rust".to_string()),
        text: "Rust ownership guide".to_string(),
      }],
      top_n: None,
    },
  )
  .unwrap();

  assert_eq!(response.model, "@cf/zai-org/glm-4.7-flash");
  assert_eq!(response.scores.len(), 1);

  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(
    requests[0].body["chat_template_kwargs"]["enable_thinking"],
    Value::Bool(false)
  );
}

#[test]
fn should_dispatch_cloudflare_native_bge_rerank_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "result": {
        "response": [
          { "id": 1, "score": 0.12 },
          { "id": 0, "score": 0.91 }
        ]
      }
    }),
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url = "https://api.cloudflare.com/client/v4/accounts/a1/ai".to_string();
  config.request_layer = Some(BackendRequestLayer::CloudflareWorkersAi);

  let response = dispatch_rerank_request(
    &client,
    &config,
    BackendProtocol::OpenaiChatCompletions,
    &RerankRequest {
      model: "@cf/baai/bge-reranker-base".to_string(),
      query: "programming".to_string(),
      candidates: vec![
        RerankCandidate {
          id: Some("rust".to_string()),
          text: "Rust ownership guide".to_string(),
        },
        RerankCandidate {
          id: Some("weather".to_string()),
          text: "Sunny and warm tomorrow".to_string(),
        },
      ],
      top_n: Some(1),
    },
  )
  .unwrap();

  assert_eq!(response.model, "@cf/baai/bge-reranker-base");
  assert_eq!(response.scores, vec![0.91, 0.12]);

  let requests = client.requests();
  assert_eq!(requests.len(), 1);
  assert_eq!(
    requests[0].url,
    "https://api.cloudflare.com/client/v4/accounts/a1/ai/run/@cf/baai/bge-reranker-base"
  );
  assert_eq!(requests[0].body["query"], "programming");
  assert_eq!(requests[0].body["top_k"], 1);
  assert_eq!(requests[0].body["contexts"][0]["text"], "Rust ownership guide");
  assert_eq!(requests[0].body["contexts"][1]["text"], "Sunny and warm tomorrow");
}

#[test]
fn should_reject_non_logprobs_rerank_protocols() {
  let client = MockHttpClient::with_json_responses(vec![]);

  let err = dispatch_rerank_request(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::AnthropicMessages,
    &RerankRequest {
      model: "claude-sonnet-4-5-20250929".to_string(),
      query: "programming".to_string(),
      candidates: vec![RerankCandidate {
        id: Some("rust".to_string()),
        text: "Rust ownership guide".to_string(),
      }],
      top_n: None,
    },
  )
  .unwrap_err();
  assert!(matches!(err, BackendError::InvalidResponse("protocol")));

  let mut gemini_config = sample_backend_config_with_header(false);
  gemini_config.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
  gemini_config.request_layer = Some(BackendRequestLayer::GeminiApi);

  let err = dispatch_rerank_request(
    &client,
    &gemini_config,
    BackendProtocol::GeminiGenerateContent,
    &RerankRequest {
      model: "gemini-2.5-flash".to_string(),
      query: "programming".to_string(),
      candidates: vec![RerankCandidate {
        id: Some("rust".to_string()),
        text: "Rust ownership guide".to_string(),
      }],
      top_n: None,
    },
  )
  .unwrap_err();
  assert!(matches!(err, BackendError::InvalidResponse("protocol")));
}

#[test]
fn should_dispatch_openai_responses_stream() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: [
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

  let events = collect_stream_events(
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
fn should_dispatch_stream_events_with_incrementally() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: [
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
  let result = dispatch_stream_events_with(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::OpenaiResponses,
    &sample_request(),
    |event| {
      seen.push(event.clone());
      Err(BackendError::Http("stop_after_first_event".to_string()))
    },
  );

  assert!(matches!(result, Err(BackendError::Http(reason)) if reason == "stop_after_first_event"));
  assert_eq!(seen.len(), 1);
}

#[test]
fn should_dispatch_stream_encoded_with_incrementally() {
  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body: [
      sse_event(
        "message_start",
        json!({
          "type": "message_start",
          "message": {
            "id": "msg_22",
            "model": "claude-sonnet-4-5-20250929",
            "usage": { "input_tokens": 4, "output_tokens": 0 },
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
          "usage": { "input_tokens": 4, "output_tokens": 2 },
        }),
      ),
      sse_event("message_stop", json!({ "type": "message_stop" })),
      sse_done(),
    ]
    .concat(),
  }))]);

  let mut chunks = Vec::new();
  dispatch_stream_encoded_with(
    &client,
    &sample_backend_config_with_header(false),
    BackendProtocol::AnthropicMessages,
    StreamEncodingTarget::OpenaiResponses,
    &sample_request(),
    |chunk| {
      chunks.push(chunk.to_string());
      Ok(())
    },
  )
  .unwrap();

  assert!(!chunks.is_empty());
  assert!(chunks.iter().any(|chunk| chunk.contains("event: response.created")));
  assert!(chunks.iter().any(|chunk| chunk.contains("event: response.completed")));
  assert!(chunks.iter().any(|chunk| chunk.contains("data: [DONE]")));
  assert!(chunks.len() >= 3);
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
  config.request_layer = Some(BackendRequestLayer::VertexAnthropic);

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
    body: [
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
  config.request_layer = Some(BackendRequestLayer::VertexAnthropic);

  let mut request = sample_request();
  request.model = "claude-sonnet-4-5@20250929".to_string();

  let events = collect_stream_events(&client, &config, BackendProtocol::AnthropicMessages, &request).unwrap();
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
fn should_dispatch_gemini_vertex_stream() {
  let body = [
    format!(
      "data: {}\n\n",
      serde_json::to_string(&json!({
        "responseId": "gem_stream_1",
        "modelVersion": "gemini-2.5-flash-001",
        "candidates": [{
          "content": {
            "parts": [{ "text": "Hel" }]
          }
        }]
      }))
      .unwrap()
    ),
    format!(
      "data: {}\n\n",
      serde_json::to_string(&json!({
        "responseId": "gem_stream_1",
        "modelVersion": "gemini-2.5-flash-001",
        "candidates": [{
          "content": {
            "parts": [
              { "text": "Hello" },
              { "functionCall": { "name": "doc_read", "args": { "docId": "a1" } } }
            ]
          },
          "finishReason": "FINISH_REASON_STOP"
        }],
        "usageMetadata": {
          "promptTokenCount": 8,
          "candidatesTokenCount": 2,
          "totalTokenCount": 10
        }
      }))
      .unwrap()
    ),
    "data: [DONE]\n\n".to_string(),
  ]
  .concat();

  let client = MockHttpClient::with_stream_responses(vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
    status: 200,
    body,
  }))]);

  let mut config = sample_backend_config_with_header(false);
  config.base_url =
    "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/publishers/google".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiVertex);

  let mut request = sample_request();
  request.model = "gemini-2.5-flash".to_string();

  let events = collect_stream_events(&client, &config, BackendProtocol::GeminiGenerateContent, &request).unwrap();
  assert!(
    events
      .iter()
      .any(|event| matches!(event, StreamEvent::TextDelta { .. }))
  );
  assert!(events.iter().any(
    |event| matches!(event, StreamEvent::ToolCall { call_id, name, .. } if call_id == "doc_read:1" && name == "doc_read")
  ));
  assert!(
    events
      .iter()
      .any(|event| matches!(event, StreamEvent::Done { finish_reason: Some(reason), .. } if reason == "tool_calls"))
  );

  let requests = client.requests();
  assert_eq!(
    requests[0].url,
    "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
  );
  assert!(requests[0].body.get("model").is_none());
  assert!(requests[0].body.get("stream").is_none());
  assert_eq!(
    requests[0].headers,
    vec![
      ("accept".to_string(), "text/event-stream".to_string()),
      ("authorization".to_string(), "Bearer token-1".to_string()),
      ("content-type".to_string(), "application/json".to_string()),
      ("x-test-header".to_string(), "1".to_string()),
    ]
  );
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
