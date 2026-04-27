use serde_json::{Value, json};

use super::{
  super::{
    core::{
      CoreContent, CoreMessage, CoreRole, EmbeddingRequest, ImageFormat, ImageInput, ImageOptions,
      ImageProviderOptions, ImageRequest, RerankCandidate, RerankRequest, StreamEvent, StructuredRequest,
    },
    protocol::fal::options::{FalImageOptions, FalImageOutputFormat, FalImageSize, FalImageSizePreset},
    stream::StreamEncodingTarget,
    test_support::{
      MockHttpClient, MockHttpResponse, sample_backend_config, sample_backend_config_with_header, sample_request,
      sse_done, sse_event,
    },
  },
  BackendConfig, BackendError, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, HttpBody, HttpResponse,
  HttpStreamResponse, ImageProtocol, MultipartPart, RerankProtocol, StructuredProtocol, collect_stream_events,
  dispatch_embedding_request, dispatch_image_request, dispatch_request, dispatch_rerank_request,
  dispatch_stream_encoded_with, dispatch_stream_events_with, dispatch_structured_request,
};

#[test]
fn should_parse_protocol_and_request_layer_aliases() {
  assert_eq!(
    "chat-completions".parse::<ChatProtocol>().unwrap(),
    ChatProtocol::OpenaiChatCompletions
  );
  assert_eq!(
    "openai-responses".parse::<StructuredProtocol>().unwrap(),
    StructuredProtocol::OpenaiResponses
  );
  assert_eq!(
    "openai_chat".parse::<EmbeddingProtocol>().unwrap(),
    EmbeddingProtocol::Openai
  );
  assert_eq!(
    "cloudflare-workers-ai".parse::<RerankProtocol>().unwrap(),
    RerankProtocol::CloudflareWorkersAi
  );
  assert_eq!("fal-image".parse::<ImageProtocol>().unwrap(), ImageProtocol::FalImage);
  assert_eq!(
    "gemini-vertex".parse::<BackendRequestLayer>().unwrap(),
    BackendRequestLayer::GeminiVertex
  );
}

#[test]
fn should_return_stable_parse_errors() {
  let error = "unknown".parse::<ChatProtocol>().unwrap_err();
  assert!(matches!(error, BackendError::InvalidRequest { field: "protocol", .. }));
}

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
    ChatProtocol::OpenaiChatCompletions,
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
fn should_dispatch_openai_image_generate_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "data": [{ "b64_json": "aW1n" }],
      "usage": { "input_tokens": 3, "output_tokens": 5, "total_tokens": 8 }
    }),
  }))]);
  let request = ImageRequest::generate(
    "gpt-image-1".to_string(),
    "draw a quiet workspace".to_string(),
    ImageOptions {
      n: Some(1),
      size: Some("1024x1024".to_string()),
      quality: Some("high".to_string()),
      output_format: Some(ImageFormat::Webp),
      ..Default::default()
    },
    ImageProviderOptions::default(),
  );

  let response = dispatch_image_request(
    &client,
    &BackendConfig {
      request_layer: Some(BackendRequestLayer::OpenaiImages),
      ..sample_backend_config(false)
    },
    ImageProtocol::OpenaiImages,
    &request,
  )
  .unwrap();

  assert_eq!(response.images[0].data_base64.as_deref(), Some("aW1n"));
  assert_eq!(response.images[0].media_type, "image/webp");
  assert_eq!(response.usage.unwrap().total_tokens, Some(8));
  let requests = client.requests();
  assert_eq!(requests[0].url, "https://api.example.com/v1/images/generations");
  assert_eq!(requests[0].body["model"], "gpt-image-1");
  assert_eq!(requests[0].body["quality"], "high");
  assert_eq!(requests[0].body["output_format"], "webp");
}

#[test]
fn should_dispatch_openai_image_edit_as_multipart() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({ "data": [{ "url": "https://cdn.example.com/out.png" }] }),
  }))]);
  let request = ImageRequest::edit(
    "gpt-image-1".to_string(),
    "remove background".to_string(),
    vec![ImageInput::Data {
      data_base64: "aW1n".to_string(),
      media_type: "image/png".to_string(),
      file_name: Some("in.png".to_string()),
    }],
    None,
    ImageOptions::default(),
    ImageProviderOptions::default(),
  );

  dispatch_image_request(
    &client,
    &BackendConfig {
      request_layer: Some(BackendRequestLayer::OpenaiImages),
      ..sample_backend_config(false)
    },
    ImageProtocol::OpenaiImages,
    &request,
  )
  .unwrap();

  let requests = client.requests();
  assert_eq!(requests[0].url, "https://api.example.com/v1/images/edits");
  assert!(matches!(
    &requests[0].body,
    HttpBody::Multipart(parts)
      if parts.iter().any(|part| matches!(part, MultipartPart::Text { name, value } if name == "prompt" && value == "remove background"))
        && parts.iter().any(|part| matches!(part, MultipartPart::File { name, file_name, media_type, bytes } if name == "image[]" && file_name == "in.png" && media_type == "image/png" && bytes == b"img"))
  ));
}

#[test]
fn should_reject_fal_workflow_image_request() {
  let client = MockHttpClient::with_json_responses(vec![]);
  let error = dispatch_image_request(
    &client,
    &BackendConfig {
      base_url: "https://fal.run".to_string(),
      request_layer: Some(BackendRequestLayer::Fal),
      ..sample_backend_config(false)
    },
    ImageProtocol::FalImage,
    &ImageRequest::generate(
      "workflows/foo".to_string(),
      "test".to_string(),
      ImageOptions::default(),
      ImageProviderOptions::default(),
    ),
  )
  .unwrap_err();

  assert!(error.to_string().contains("model"));
}

#[test]
fn should_reject_fal_inline_image_input() {
  let client = MockHttpClient::default();
  let request = ImageRequest::edit(
    "flux-1/schnell".to_string(),
    "sticker".to_string(),
    vec![ImageInput::Data {
      data_base64: "aW1hZ2U=".to_string(),
      media_type: "image/png".to_string(),
      file_name: None,
    }],
    None,
    ImageOptions::default(),
    ImageProviderOptions::default(),
  );

  let error = dispatch_image_request(
    &client,
    &BackendConfig {
      base_url: "https://fal.run".to_string(),
      request_layer: Some(BackendRequestLayer::Fal),
      ..sample_backend_config(false)
    },
    ImageProtocol::FalImage,
    &request,
  )
  .unwrap_err();

  assert!(error.to_string().contains("images"));
}

#[test]
fn should_dispatch_fal_ordinary_image_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "images": [{ "url": "https://fal.example.com/out.webp", "content_type": "image/webp", "width": 512, "height": 512 }]
    }),
  }))]);
  let request = ImageRequest::generate(
    "flux-1/schnell".to_string(),
    "sticker".to_string(),
    ImageOptions {
      seed: Some(7),
      ..Default::default()
    },
    ImageProviderOptions::Fal(FalImageOptions {
      image_size: Some(FalImageSize::Preset(FalImageSizePreset::SquareHd)),
      num_images: Some(2),
      enable_safety_checker: Some(false),
      output_format: Some(FalImageOutputFormat::Webp),
      model_name: Some("dev".to_string()),
      ..Default::default()
    }),
  );

  let response = dispatch_image_request(
    &client,
    &BackendConfig {
      base_url: "https://fal.run".to_string(),
      request_layer: Some(BackendRequestLayer::Fal),
      ..sample_backend_config(false)
    },
    ImageProtocol::FalImage,
    &request,
  )
  .unwrap();

  assert_eq!(
    response.images[0].url.as_deref(),
    Some("https://fal.example.com/out.webp")
  );
  let requests = client.requests();
  assert_eq!(requests[0].url, "https://fal.run/fal-ai/flux-1/schnell");
  assert_eq!(requests[0].body["prompt"], "sticker");
  assert_eq!(requests[0].body["model_name"], "dev");
  assert_eq!(requests[0].body["seed"], 7);
  assert_eq!(requests[0].body["image_size"], "square_hd");
  assert_eq!(requests[0].body["num_images"], 2);
  assert_eq!(requests[0].body["enable_safety_checker"], false);
  assert_eq!(requests[0].body["output_format"], "webp");
  assert_eq!(requests[0].body["sync_mode"], true);
}

#[test]
fn should_dispatch_gemini_nano_banana_image_generate_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [
            { "text": "ok" },
            { "inlineData": { "mimeType": "image/png", "data": "aW1n" } }
          ]
        }
      }]
    }),
  }))]);
  let request = ImageRequest::generate(
    "gemini-2.5-flash-image".to_string(),
    "draw a quiet workspace".to_string(),
    ImageOptions::default(),
    ImageProviderOptions::default(),
  );

  let mut config = sample_backend_config(false);
  config.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiApi);

  let response = dispatch_image_request(&client, &config, ImageProtocol::GeminiGenerateContent, &request).unwrap();

  assert_eq!(response.images[0].data_base64.as_deref(), Some("aW1n"));
  assert_eq!(response.images[0].media_type, "image/png");
  assert_eq!(response.text.as_deref(), Some("ok"));
  let requests = client.requests();
  assert_eq!(
    requests[0].url,
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
  );
  assert_eq!(
    requests[0].body["generationConfig"]["responseModalities"],
    json!(["TEXT", "IMAGE"])
  );
  assert_eq!(
    requests[0].body["contents"][0]["parts"][0]["text"],
    "draw a quiet workspace"
  );
  assert!(requests[0].body.get("model").is_none());
}

#[test]
fn should_dispatch_gemini_nano_banana_image_edit_request() {
  let client = MockHttpClient::with_json_responses(vec![MockHttpResponse::Json(Ok(HttpResponse {
    status: 200,
    body: json!({
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [
            { "inlineData": { "mimeType": "image/png", "data": "ZWRpdA==" } }
          ]
        }
      }]
    }),
  }))]);
  let request = ImageRequest::edit(
    "gemini-2.5-flash-image".to_string(),
    "turn this into a clean sticker".to_string(),
    vec![ImageInput::Data {
      data_base64: "aW1n".to_string(),
      media_type: "image/png".to_string(),
      file_name: Some("source.png".to_string()),
    }],
    None,
    ImageOptions::default(),
    ImageProviderOptions::default(),
  );

  let mut config = sample_backend_config(false);
  config.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
  config.request_layer = Some(BackendRequestLayer::GeminiApi);

  let response = dispatch_image_request(&client, &config, ImageProtocol::GeminiGenerateContent, &request).unwrap();

  assert_eq!(response.images[0].data_base64.as_deref(), Some("ZWRpdA=="));
  let requests = client.requests();
  assert_eq!(
    requests[0].body["contents"][0]["parts"][1]["inlineData"],
    json!({ "mimeType": "image/png", "data": "aW1n" })
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

  let response = dispatch_request(&client, &config, ChatProtocol::GeminiGenerateContent, &request).unwrap();
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
    StructuredProtocol::OpenaiResponses,
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
    StructuredProtocol::OpenaiResponses,
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
    StructuredProtocol::OpenaiResponses,
    &request,
  )
  .unwrap_err();

  assert!(matches!(error, BackendError::InvalidStructuredOutput { .. }));
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
    EmbeddingProtocol::Gemini,
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
    EmbeddingProtocol::Gemini,
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
    RerankProtocol::OpenaiChatLogprobs,
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
    RerankProtocol::CloudflareWorkersAi,
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
    RerankProtocol::CloudflareWorkersAi,
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
    RerankProtocol::CloudflareWorkersAi,
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
    ChatProtocol::OpenaiResponses,
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
    ChatProtocol::OpenaiResponses,
    &sample_request(),
    |event| {
      seen.push(event.clone());
      Err(BackendError::Transport {
        message: "stop_after_first_event".to_string(),
      })
    },
  );

  assert!(matches!(result, Err(BackendError::Transport { message: reason }) if reason == "stop_after_first_event"));
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
    ChatProtocol::AnthropicMessages,
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
    ChatProtocol::AnthropicMessages,
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
    ChatProtocol::AnthropicMessages,
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

  dispatch_request(&client, &config, ChatProtocol::AnthropicMessages, &request).unwrap();
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

  let events = collect_stream_events(&client, &config, ChatProtocol::AnthropicMessages, &request).unwrap();
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

  let events = collect_stream_events(&client, &config, ChatProtocol::GeminiGenerateContent, &request).unwrap();
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

  let result = dispatch_request(&client, &config, ChatProtocol::AnthropicMessages, &sample_request());
  assert!(matches!(result, Err(BackendError::InvalidConfig { .. })));
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
    ChatProtocol::OpenaiChatCompletions,
    &sample_request(),
  );

  assert!(matches!(result, Err(BackendError::UpstreamStatus { status: 500, .. })));
}
