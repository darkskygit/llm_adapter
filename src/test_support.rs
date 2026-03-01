use std::{
  collections::BTreeMap,
  sync::{Arc, Mutex},
};

use serde_json::{Value, json};

use crate::{
  backend::{BackendConfig, BackendError, BackendHttpClient, HttpRequest, HttpResponse, HttpStreamResponse},
  core::{
    CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreToolChoice, CoreToolChoiceMode,
    CoreToolDefinition, CoreUsage,
  },
};

pub(crate) fn sample_request() -> CoreRequest {
  CoreRequest {
    model: "gpt-4.1".to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "hello".to_string(),
      }],
    }],
    stream: false,
    max_tokens: Some(128),
    temperature: Some(0.2),
    tools: vec![CoreToolDefinition {
      name: "doc_read".to_string(),
      description: Some("Read a document".to_string()),
      parameters: json!({ "type": "object", "properties": { "docId": { "type": "string" }}}),
    }],
    tool_choice: Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
    include: Some(vec!["reasoning".to_string()]),
    reasoning: Some(json!({ "effort": "medium" })),
  }
}

pub(crate) fn sample_backend_config(no_streaming: bool) -> BackendConfig {
  BackendConfig {
    base_url: "https://api.example.com".to_string(),
    auth_token: "token-1".to_string(),
    request_layer: None,
    headers: BTreeMap::new(),
    no_streaming,
    timeout_ms: None,
  }
}

pub(crate) fn sample_backend_config_with_header(no_streaming: bool) -> BackendConfig {
  BackendConfig {
    base_url: "https://api.example.com".to_string(),
    auth_token: "token-1".to_string(),
    request_layer: None,
    headers: BTreeMap::from_iter([("x-test-header".to_string(), "1".to_string())]),
    no_streaming,
    timeout_ms: Some(10_000),
  }
}

pub(crate) fn sample_response_with_reasoning_tool_call() -> CoreResponse {
  CoreResponse {
    id: "resp_1".to_string(),
    model: "gpt-4.1".to_string(),
    message: CoreMessage {
      role: CoreRole::Assistant,
      content: vec![
        CoreContent::Reasoning {
          text: "inspect document".to_string(),
          signature: None,
        },
        CoreContent::Text {
          text: "Need to call a tool.".to_string(),
        },
        CoreContent::ToolCall {
          call_id: "call_10".to_string(),
          name: "doc_read".to_string(),
          arguments: json!({ "docId": "abc" }),
          thought: Some("fetch context".to_string()),
        },
      ],
    },
    usage: CoreUsage {
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      cached_tokens: Some(12),
    },
    finish_reason: "tool_calls".to_string(),
    reasoning_details: Some(json!({ "effort": "medium" })),
  }
}

#[derive(Debug)]
pub(crate) enum MockHttpResponse {
  Json(Result<HttpResponse, BackendError>),
  Stream(Result<HttpStreamResponse, BackendError>),
}

#[derive(Debug, Clone)]
pub(crate) struct MockHttpClient {
  requests: Arc<Mutex<Vec<HttpRequest>>>,
  json_responses: Arc<Mutex<Vec<MockHttpResponse>>>,
  stream_responses: Arc<Mutex<Vec<MockHttpResponse>>>,
  stream_chunk_size: usize,
}

impl Default for MockHttpClient {
  fn default() -> Self {
    Self {
      requests: Arc::new(Mutex::new(Vec::new())),
      json_responses: Arc::new(Mutex::new(Vec::new())),
      stream_responses: Arc::new(Mutex::new(Vec::new())),
      stream_chunk_size: 17,
    }
  }
}

impl MockHttpClient {
  pub(crate) fn new(json_responses: Vec<MockHttpResponse>, stream_responses: Vec<MockHttpResponse>) -> Self {
    Self {
      json_responses: Arc::new(Mutex::new(json_responses)),
      stream_responses: Arc::new(Mutex::new(stream_responses)),
      ..Self::default()
    }
  }

  pub(crate) fn with_json_responses(responses: Vec<MockHttpResponse>) -> Self {
    Self {
      json_responses: Arc::new(Mutex::new(responses)),
      ..Self::default()
    }
  }

  pub(crate) fn with_stream_responses(responses: Vec<MockHttpResponse>) -> Self {
    Self {
      stream_responses: Arc::new(Mutex::new(responses)),
      ..Self::default()
    }
  }

  pub(crate) fn requests(&self) -> Vec<HttpRequest> {
    self.requests.lock().unwrap().clone()
  }
}

impl BackendHttpClient for MockHttpClient {
  fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
    self.requests.lock().unwrap().push(request);

    match self.json_responses.lock().unwrap().remove(0) {
      MockHttpResponse::Json(response) => response,
      MockHttpResponse::Stream(_) => unreachable!(),
    }
  }

  fn post_sse(
    &self,
    request: HttpRequest,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
  ) -> Result<(), BackendError> {
    self.requests.lock().unwrap().push(request);

    match self.stream_responses.lock().unwrap().remove(0) {
      MockHttpResponse::Stream(response) => {
        let response = response?;
        for chunk in response.body.as_bytes().chunks(self.stream_chunk_size) {
          let text = std::str::from_utf8(chunk).map_err(|error| BackendError::Http(error.to_string()))?;
          on_chunk(text)?;
        }
        Ok(())
      }
      MockHttpResponse::Json(_) => unreachable!(),
    }
  }
}

pub(crate) fn sse_event(event: &str, data: Value) -> String {
  format!("event: {event}\\ndata: {data}\\n\\n")
}

pub(crate) fn sse_done() -> String {
  "data: [DONE]\\n\\n".to_string()
}
