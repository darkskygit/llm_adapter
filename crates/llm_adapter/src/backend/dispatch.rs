use serde_json::Value;

use super::{
  super::{
    core::{
      CoreContent, CoreRequest, CoreResponse, EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse,
      StreamEvent, StructuredRequest, StructuredResponse,
    },
    protocol::{ProtocolError, anthropic, gemini, openai},
    stream::{
      AnthropicStreamParser, GeminiStreamParser, IncrementalSseEncoder, OpenaiChatStreamParser,
      OpenaiResponsesStreamParser, SseFrame, SseFrameDecoder, StreamEncodingTarget, StreamParseError, encode_sse_frame,
    },
  },
  BackendConfig, BackendError, BackendHttpClient, BackendProtocol, BackendRequestLayer, HttpRequest,
  request_layer::{build_extra_headers, resolve_request_layer},
};

impl BackendProtocol {
  fn encode_request(&self, request: &CoreRequest, stream: bool) -> Value {
    match self {
      BackendProtocol::OpenaiChatCompletions => openai::chat::request::encode(request, stream),
      BackendProtocol::OpenaiResponses => openai::responses::request::encode(request, stream),
      BackendProtocol::AnthropicMessages => anthropic::request::encode(request, stream),
      BackendProtocol::GeminiGenerateContent => gemini::request::encode(request, stream),
    }
  }

  fn decode_response(&self, body: &Value) -> Result<CoreResponse, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions => openai::chat::response::decode(body),
      BackendProtocol::OpenaiResponses => openai::responses::response::decode(body),
      BackendProtocol::AnthropicMessages => anthropic::response::decode(body),
      BackendProtocol::GeminiGenerateContent => gemini::response::decode(body),
    }
  }

  fn encode_structured_request(&self, request: &StructuredRequest) -> Result<Value, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions
      | BackendProtocol::OpenaiResponses
      | BackendProtocol::GeminiGenerateContent => Ok(self.encode_request(&request.as_core_request(), false)),
      BackendProtocol::AnthropicMessages => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "structured dispatch is unsupported for anthropic_messages".to_string(),
      }),
    }
  }

  fn decode_structured_response(&self, body: &Value) -> Result<StructuredResponse, BackendError> {
    let response = self.decode_response(body).map_err(map_protocol_error)?;
    let output_text = extract_text_output(&response).map_err(map_protocol_error)?;
    let output_json = parse_structured_output(&output_text).ok_or_else(|| BackendError::InvalidStructuredOutput {
      message: format!(
        "structured response did not contain valid JSON: {}",
        truncate_structured_output_preview(&output_text)
      ),
    })?;
    Ok(StructuredResponse {
      id: response.id,
      model: response.model,
      output_text,
      output_json: Some(output_json),
      usage: response.usage,
      finish_reason: response.finish_reason,
      reasoning_details: response.reasoning_details,
    })
  }

  fn encode_embedding_request(&self, request: &EmbeddingRequest) -> Result<Value, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions | BackendProtocol::OpenaiResponses => {
        Ok(openai::embedding::encode(request))
      }
      BackendProtocol::GeminiGenerateContent => Ok(gemini::embedding::encode(request)),
      BackendProtocol::AnthropicMessages => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "embedding dispatch is unsupported for anthropic_messages".to_string(),
      }),
    }
  }

  fn decode_embedding_response(&self, body: &Value) -> Result<EmbeddingResponse, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions | BackendProtocol::OpenaiResponses => openai::embedding::decode(body),
      BackendProtocol::GeminiGenerateContent => gemini::embedding::decode(body),
      BackendProtocol::AnthropicMessages => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "embedding dispatch is unsupported for anthropic_messages".to_string(),
      }),
    }
  }

  fn encode_rerank_request(&self, request: &RerankRequest, candidate_index: usize) -> Result<Value, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions => Ok(openai::chat::rerank::encode(request, candidate_index)?),
      BackendProtocol::OpenaiResponses => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for openai_responses".to_string(),
      }),
      BackendProtocol::AnthropicMessages => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for anthropic_messages".to_string(),
      }),
      BackendProtocol::GeminiGenerateContent => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for gemini_generate_content".to_string(),
      }),
    }
  }

  fn decode_rerank_response(&self, body: &Value, request: &RerankRequest) -> Result<(String, f64), ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions => openai::chat::rerank::decode(body, request),
      BackendProtocol::OpenaiResponses => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for openai_responses".to_string(),
      }),
      BackendProtocol::AnthropicMessages => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for anthropic_messages".to_string(),
      }),
      BackendProtocol::GeminiGenerateContent => Err(ProtocolError::InvalidValue {
        field: "protocol",
        message: "rerank dispatch is unsupported for gemini_generate_content".to_string(),
      }),
    }
  }
}

enum IncrementalStreamParser {
  OpenaiChat(OpenaiChatStreamParser),
  OpenaiResponses(OpenaiResponsesStreamParser),
  Anthropic(AnthropicStreamParser),
  Gemini(GeminiStreamParser),
}

impl IncrementalStreamParser {
  fn new(protocol: &BackendProtocol) -> Self {
    match protocol {
      BackendProtocol::OpenaiChatCompletions => Self::OpenaiChat(OpenaiChatStreamParser::default()),
      BackendProtocol::OpenaiResponses => Self::OpenaiResponses(OpenaiResponsesStreamParser::default()),
      BackendProtocol::AnthropicMessages => Self::Anthropic(AnthropicStreamParser::default()),
      BackendProtocol::GeminiGenerateContent => Self::Gemini(GeminiStreamParser::default()),
    }
  }

  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    match self {
      IncrementalStreamParser::OpenaiChat(parser) => parser.push_frame(frame),
      IncrementalStreamParser::OpenaiResponses(parser) => parser.push_frame(frame),
      IncrementalStreamParser::Anthropic(parser) => parser.push_frame(frame),
      IncrementalStreamParser::Gemini(parser) => parser.push_frame(frame),
    }
  }

  fn finish(&mut self) -> Vec<StreamEvent> {
    match self {
      IncrementalStreamParser::OpenaiChat(parser) => parser.finish(),
      IncrementalStreamParser::OpenaiResponses(parser) => parser.finish(),
      IncrementalStreamParser::Anthropic(parser) => parser.finish(),
      IncrementalStreamParser::Gemini(parser) => parser.finish(),
    }
  }
}

fn build_http_request(
  config: &BackendConfig,
  protocol: &BackendProtocol,
  request: &CoreRequest,
  stream: bool,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_protocol_error)?;
  let request_layer = resolve_request_layer(config, protocol)?;
  let mut headers = request_layer.build_headers(config, stream);
  headers.extend(build_extra_headers(config));

  let body = request_layer.rewrite_body(protocol.encode_request(request, stream));

  Ok(HttpRequest {
    url: request_layer.build_url(&config.base_url, &request.model, stream),
    headers,
    body,
    timeout_ms: config.timeout_ms,
  })
}

fn build_structured_http_request(
  config: &BackendConfig,
  protocol: &BackendProtocol,
  request: &StructuredRequest,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_protocol_error)?;
  let request_layer = resolve_request_layer(config, protocol)?;
  let mut headers = request_layer.build_headers(config, false);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_url(&config.base_url, &request.model, false),
    headers,
    body: request_layer.rewrite_body(
      protocol
        .encode_structured_request(request)
        .map_err(map_protocol_error)?,
    ),
    timeout_ms: config.timeout_ms,
  })
}

fn build_embedding_http_request(
  config: &BackendConfig,
  protocol: &BackendProtocol,
  request: &EmbeddingRequest,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_protocol_error)?;
  let request_layer = resolve_request_layer(config, protocol)?;
  let mut headers = request_layer.build_embedding_headers(config);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_embedding_url(&config.base_url, &request.model)?,
    headers,
    body: request_layer.rewrite_embedding_body(protocol.encode_embedding_request(request).map_err(map_protocol_error)?),
    timeout_ms: config.timeout_ms,
  })
}

fn build_rerank_http_request(
  request_layer: &BackendRequestLayer,
  config: &BackendConfig,
  protocol: &BackendProtocol,
  request: &RerankRequest,
  candidate_index: usize,
) -> Result<HttpRequest, BackendError> {
  let mut headers = request_layer.build_rerank_headers(config);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_rerank_url(&config.base_url, &request.model)?,
    headers,
    body: request_layer.rewrite_rerank_body(
      protocol
        .encode_rerank_request(request, candidate_index)
        .map_err(map_protocol_error)?,
    ),
    timeout_ms: config.timeout_ms,
  })
}

pub fn dispatch_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &CoreRequest,
) -> Result<CoreResponse, BackendError> {
  let response = client.post_json(build_http_request(config, &protocol, request, false)?)?;
  protocol.decode_response(&response.body).map_err(map_protocol_error)
}

pub fn dispatch_structured_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &StructuredRequest,
) -> Result<StructuredResponse, BackendError> {
  let response = client.post_json(build_structured_http_request(config, &protocol, request)?)?;
  protocol.decode_structured_response(&response.body)
}

pub fn dispatch_embedding_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, BackendError> {
  let response = client.post_json(build_embedding_http_request(config, &protocol, request)?)?;
  protocol
    .decode_embedding_response(&response.body)
    .map_err(map_protocol_error)
}

pub fn dispatch_rerank_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &RerankRequest,
) -> Result<RerankResponse, BackendError> {
  request.validate().map_err(map_protocol_error)?;
  let request_layer = resolve_request_layer(config, &protocol)?;

  if let Some(response) = request_layer.dispatch_rerank(client, config, &protocol, request)? {
    return Ok(response);
  }

  let mut model: Option<String> = None;
  let mut scores = Vec::with_capacity(request.candidates.len());

  for candidate_index in 0..request.candidates.len() {
    let response = client.post_json(build_rerank_http_request(
      &request_layer,
      config,
      &protocol,
      request,
      candidate_index,
    )?)?;
    let (response_model, score) = protocol
      .decode_rerank_response(&response.body, request)
      .map_err(map_protocol_error)?;
    if model.is_none() {
      model = Some(response_model);
    }
    scores.push(score);
  }

  Ok(RerankResponse {
    model: model.unwrap_or_else(|| request.model.clone()),
    scores,
  })
}

pub fn collect_stream_events(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &CoreRequest,
) -> Result<Vec<StreamEvent>, BackendError> {
  let mut events = Vec::new();
  dispatch_stream_events_with(client, config, protocol, request, |event| {
    events.push(event);
    Ok(())
  })?;
  Ok(events)
}

pub fn dispatch_stream_events_with<F>(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &CoreRequest,
  mut on_event: F,
) -> Result<(), BackendError>
where
  F: FnMut(StreamEvent) -> Result<(), BackendError>,
{
  let mut frame_decoder = SseFrameDecoder::default();
  let mut parser = IncrementalStreamParser::new(&protocol);

  let request = build_http_request(config, &protocol, request, true)?;
  client.post_sse(request, &mut |chunk| {
    for frame in frame_decoder.push_chunk(chunk) {
      for event in parser.push_frame(frame).map_err(BackendError::from)? {
        on_event(event)?;
      }
    }
    Ok(())
  })?;

  for frame in frame_decoder.finish() {
    for event in parser.push_frame(frame).map_err(BackendError::from)? {
      on_event(event)?;
    }
  }
  for event in parser.finish() {
    on_event(event)?;
  }

  Ok(())
}

pub fn collect_stream_encoded(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  source_protocol: BackendProtocol,
  target: StreamEncodingTarget,
  request: &CoreRequest,
) -> Result<String, BackendError> {
  let mut out = String::new();
  dispatch_stream_encoded_with(client, config, source_protocol, target, request, |chunk| {
    out.push_str(chunk);
    Ok(())
  })?;
  Ok(out)
}

pub fn dispatch_stream_encoded_with<F>(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  source_protocol: BackendProtocol,
  target: StreamEncodingTarget,
  request: &CoreRequest,
  mut on_chunk: F,
) -> Result<(), BackendError>
where
  F: FnMut(&str) -> Result<(), BackendError>,
{
  let mut encoder = IncrementalSseEncoder::new(target);
  dispatch_stream_events_with(client, config, source_protocol, request, |event| {
    for frame in encoder.push_event(&event) {
      let serialized = encode_sse_frame(&frame);
      on_chunk(&serialized)?;
    }
    Ok(())
  })?;

  for frame in encoder.finish() {
    let serialized = encode_sse_frame(&frame);
    on_chunk(&serialized)?;
  }

  Ok(())
}

fn map_protocol_error(error: ProtocolError) -> BackendError {
  match error {
    ProtocolError::MissingField(field) => BackendError::InvalidResponse(field),
    ProtocolError::InvalidValue { field, .. } => BackendError::InvalidResponse(field),
    ProtocolError::Json(error) => BackendError::Json(error),
  }
}

fn extract_text_output(response: &CoreResponse) -> Result<String, ProtocolError> {
  let mut text = String::new();
  for content in &response.message.content {
    match content {
      CoreContent::Text { text: delta } => text.push_str(delta),
      CoreContent::Reasoning { .. } => {}
      _ => {}
    }
  }

  if text.is_empty() {
    Err(ProtocolError::InvalidValue {
      field: "message.content",
      message: "structured response did not contain text content".to_string(),
    })
  } else {
    Ok(text)
  }
}

fn parse_structured_output(output_text: &str) -> Option<Value> {
  let normalized = normalize_structured_text(output_text);

  for candidate in structured_output_candidates(&normalized) {
    if let Some(candidate) = candidate
      && let Ok(value) = serde_json::from_str::<Value>(candidate)
    {
      return Some(value);
    }
  }

  None
}

fn truncate_structured_output_preview(output_text: &str) -> String {
  const MAX_PREVIEW_CHARS: usize = 200;

  let mut preview = String::new();
  for ch in output_text.trim().chars().take(MAX_PREVIEW_CHARS) {
    preview.push(ch);
  }

  if output_text.trim().chars().count() > MAX_PREVIEW_CHARS {
    preview.push_str("...");
  }

  preview
}

fn structured_output_candidates(normalized: &str) -> [Option<&str>; 3] {
  let object_slice = match (normalized.find('{'), normalized.rfind('}')) {
    (Some(start), Some(end)) if end > start => Some(&normalized[start..=end]),
    _ => None,
  };
  let array_slice = match (normalized.find('['), normalized.rfind(']')) {
    (Some(start), Some(end)) if end > start => Some(&normalized[start..=end]),
    _ => None,
  };

  [Some(normalized), object_slice, array_slice]
}

fn normalize_structured_text(output_text: &str) -> String {
  let trimmed = output_text.trim();
  let trimmed = trimmed.strip_prefix("ny\n").unwrap_or(trimmed).trim();
  if trimmed.starts_with("```") || trimmed.ends_with("```") {
    return strip_fenced_code_block(trimmed).trim().to_string();
  }
  trimmed.to_string()
}

fn strip_fenced_code_block(text: &str) -> &str {
  let Some(without_open) = text.strip_prefix("```") else {
    return text;
  };
  let without_language = match without_open.find('\n') {
    Some(index) => &without_open[index + 1..],
    None => without_open,
  };
  without_language.strip_suffix("\n```").unwrap_or(without_language)
}
