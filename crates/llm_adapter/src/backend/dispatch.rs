use serde_json::Value;

use super::{
  super::{
    core::{
      CoreContent, CoreRequest, CoreResponse, EmbeddingRequest, EmbeddingResponse, ImageRequest, ImageResponse,
      RerankRequest, RerankResponse, StreamEvent, StructuredRequest, StructuredResponse,
    },
    protocol::{ProtocolError, anthropic, fal, gemini, openai},
    stream::{
      AnthropicStreamParser, GeminiStreamParser, OpenaiChatStreamParser, OpenaiResponsesStreamParser, SseFrame,
      SseFrameDecoder, StreamParseError,
    },
  },
  BackendConfig, BackendError, BackendHttpClient, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, HttpBody,
  HttpRequest, ImageProtocol, RerankProtocol, StructuredProtocol,
  request_layer::{
    build_extra_headers, resolve_chat_request_layer, resolve_embedding_request_layer, resolve_image_request_layer,
    resolve_rerank_request_layer, resolve_structured_request_layer,
  },
};
#[cfg(test)]
use crate::stream::{IncrementalSseEncoder, StreamEncodingTarget, encode_sse_frame};

impl ChatProtocol {
  fn encode_request(
    &self,
    request: &CoreRequest,
    stream: bool,
    request_layer: BackendRequestLayer,
    base_url: &str,
  ) -> Value {
    match self {
      ChatProtocol::OpenaiChatCompletions => openai::chat::request::encode(request, stream),
      ChatProtocol::OpenaiResponses => openai::responses::request::encode(request, stream),
      ChatProtocol::AnthropicMessages => anthropic::request::encode(request, stream),
      ChatProtocol::GeminiGenerateContent => gemini::request::encode(request, stream, request_layer, base_url),
    }
  }

  fn decode_response(&self, body: &Value) -> Result<CoreResponse, ProtocolError> {
    match self {
      ChatProtocol::OpenaiChatCompletions => openai::chat::response::decode(body),
      ChatProtocol::OpenaiResponses => openai::responses::response::decode(body),
      ChatProtocol::AnthropicMessages => anthropic::response::decode(body),
      ChatProtocol::GeminiGenerateContent => gemini::response::decode(body),
    }
  }
}

impl StructuredProtocol {
  fn encode_structured_request(
    &self,
    request: &StructuredRequest,
    request_layer: BackendRequestLayer,
    base_url: &str,
  ) -> Result<Value, ProtocolError> {
    match self {
      StructuredProtocol::OpenaiChatCompletions => Ok(openai::chat::request::encode(&request.as_core_request(), false)),
      StructuredProtocol::OpenaiResponses => Ok(openai::responses::request::encode(&request.as_core_request(), false)),
      StructuredProtocol::GeminiGenerateContent => Ok(gemini::request::encode(
        &request.as_core_request(),
        false,
        request_layer,
        base_url,
      )),
    }
  }

  fn decode_structured_response(&self, body: &Value) -> Result<StructuredResponse, BackendError> {
    let response = match self {
      StructuredProtocol::OpenaiChatCompletions => openai::chat::response::decode(body),
      StructuredProtocol::OpenaiResponses => openai::responses::response::decode(body),
      StructuredProtocol::GeminiGenerateContent => gemini::response::decode(body),
    }
    .map_err(map_response_protocol_error)?;
    let output_text = extract_text_output(&response).map_err(map_response_protocol_error)?;
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
}

impl EmbeddingProtocol {
  fn encode_embedding_request(&self, request: &EmbeddingRequest) -> Result<Value, ProtocolError> {
    match self {
      EmbeddingProtocol::Openai => Ok(openai::embedding::encode(request)),
      EmbeddingProtocol::Gemini => Ok(gemini::embedding::encode(request)),
    }
  }

  fn decode_embedding_response(&self, body: &Value) -> Result<EmbeddingResponse, ProtocolError> {
    match self {
      EmbeddingProtocol::Openai => openai::embedding::decode(body),
      EmbeddingProtocol::Gemini => gemini::embedding::decode(body),
    }
  }
}

impl RerankProtocol {
  fn encode_rerank_request(&self, request: &RerankRequest, candidate_index: usize) -> Result<Value, ProtocolError> {
    match self {
      RerankProtocol::OpenaiChatLogprobs | RerankProtocol::CloudflareWorkersAi => {
        Ok(openai::chat::rerank::encode(request, candidate_index)?)
      }
    }
  }

  fn decode_rerank_response(&self, body: &Value, request: &RerankRequest) -> Result<(String, f64), ProtocolError> {
    match self {
      RerankProtocol::OpenaiChatLogprobs | RerankProtocol::CloudflareWorkersAi => {
        openai::chat::rerank::decode(body, request)
      }
    }
  }
}

impl ImageProtocol {
  fn encode_image_request(
    &self,
    request: &ImageRequest,
    request_layer: BackendRequestLayer,
    base_url: &str,
  ) -> Result<HttpBody, ProtocolError> {
    match self {
      ImageProtocol::OpenaiImages => openai::images::encode(request),
      ImageProtocol::GeminiGenerateContent => gemini::image::encode(request, request_layer, base_url),
      ImageProtocol::FalImage => Ok(HttpBody::Json(fal::encode(request)?)),
    }
  }

  fn decode_image_response(&self, body: &Value, request: &ImageRequest) -> Result<ImageResponse, BackendError> {
    match self {
      ImageProtocol::OpenaiImages => openai::images::decode(body, request),
      ImageProtocol::GeminiGenerateContent => gemini::image::decode(body).map_err(map_response_protocol_error),
      ImageProtocol::FalImage => fal::decode(body),
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
  fn new(protocol: &ChatProtocol) -> Self {
    match protocol {
      ChatProtocol::OpenaiChatCompletions => Self::OpenaiChat(OpenaiChatStreamParser::default()),
      ChatProtocol::OpenaiResponses => Self::OpenaiResponses(OpenaiResponsesStreamParser::default()),
      ChatProtocol::AnthropicMessages => Self::Anthropic(AnthropicStreamParser::default()),
      ChatProtocol::GeminiGenerateContent => Self::Gemini(GeminiStreamParser::default()),
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
  protocol: &ChatProtocol,
  request: &CoreRequest,
  stream: bool,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_request_protocol_error)?;
  let request_layer = resolve_chat_request_layer(config, protocol)?;
  let mut headers = request_layer.build_headers(config, stream);
  headers.extend(build_extra_headers(config));

  let body = request_layer.rewrite_body(protocol.encode_request(request, stream, request_layer, &config.base_url));

  Ok(HttpRequest {
    url: request_layer.build_url(&config.base_url, &request.model, stream),
    headers,
    body: HttpBody::Json(body),
    timeout_ms: config.timeout_ms,
  })
}

fn build_structured_http_request(
  config: &BackendConfig,
  protocol: &StructuredProtocol,
  request: &StructuredRequest,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_request_protocol_error)?;
  let request_layer = resolve_structured_request_layer(config, protocol)?;
  let mut headers = request_layer.build_headers(config, false);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_url(&config.base_url, &request.model, false),
    headers,
    body: HttpBody::Json(
      request_layer.rewrite_body(
        protocol
          .encode_structured_request(request, request_layer, &config.base_url)
          .map_err(map_request_protocol_error)?,
      ),
    ),
    timeout_ms: config.timeout_ms,
  })
}

fn build_embedding_http_request(
  config: &BackendConfig,
  protocol: &EmbeddingProtocol,
  request: &EmbeddingRequest,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_request_protocol_error)?;
  let request_layer = resolve_embedding_request_layer(config, protocol)?;
  let mut headers = request_layer.build_embedding_headers(config);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_embedding_url(&config.base_url, &request.model)?,
    headers,
    body: HttpBody::Json(
      request_layer.rewrite_embedding_body(
        protocol
          .encode_embedding_request(request)
          .map_err(map_request_protocol_error)?,
      ),
    ),
    timeout_ms: config.timeout_ms,
  })
}

fn build_rerank_http_request(
  request_layer: &BackendRequestLayer,
  config: &BackendConfig,
  protocol: &RerankProtocol,
  request: &RerankRequest,
  candidate_index: usize,
) -> Result<HttpRequest, BackendError> {
  let mut headers = request_layer.build_rerank_headers(config);
  headers.extend(build_extra_headers(config));

  Ok(HttpRequest {
    url: request_layer.build_rerank_url(&config.base_url, &request.model)?,
    headers,
    body: HttpBody::Json(
      request_layer.rewrite_rerank_body(
        protocol
          .encode_rerank_request(request, candidate_index)
          .map_err(map_request_protocol_error)?,
      ),
    ),
    timeout_ms: config.timeout_ms,
  })
}

fn build_image_http_request(
  config: &BackendConfig,
  protocol: &ImageProtocol,
  request: &ImageRequest,
) -> Result<HttpRequest, BackendError> {
  request.validate().map_err(map_request_protocol_error)?;
  let request_layer = resolve_image_request_layer(config, protocol)?;
  let mut headers = request_layer.build_headers(config, false);
  headers.extend(build_extra_headers(config));
  let edit = request.is_edit();
  let body = protocol
    .encode_image_request(request, request_layer, &config.base_url)
    .map_err(map_request_protocol_error)?;
  let body = match body {
    HttpBody::Json(body) => HttpBody::Json(request_layer.rewrite_body(body)),
    body => body,
  };

  Ok(HttpRequest {
    url: request_layer.build_image_url(&config.base_url, request.model(), edit),
    headers,
    body,
    timeout_ms: config.timeout_ms,
  })
}

pub fn dispatch_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: ChatProtocol,
  request: &CoreRequest,
) -> Result<CoreResponse, BackendError> {
  let response = client.post_json(build_http_request(config, &protocol, request, false)?)?;
  protocol
    .decode_response(&response.body)
    .map_err(map_response_protocol_error)
}

pub fn dispatch_structured_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: StructuredProtocol,
  request: &StructuredRequest,
) -> Result<StructuredResponse, BackendError> {
  let response = client.post_json(build_structured_http_request(config, &protocol, request)?)?;
  protocol.decode_structured_response(&response.body)
}

pub fn dispatch_embedding_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: EmbeddingProtocol,
  request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, BackendError> {
  let response = client.post_json(build_embedding_http_request(config, &protocol, request)?)?;
  protocol
    .decode_embedding_response(&response.body)
    .map_err(map_response_protocol_error)
}

pub fn dispatch_rerank_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: RerankProtocol,
  request: &RerankRequest,
) -> Result<RerankResponse, BackendError> {
  request.validate().map_err(map_request_protocol_error)?;
  let request_layer = resolve_rerank_request_layer(config, &protocol)?;

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
      .map_err(map_response_protocol_error)?;
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

pub fn dispatch_image_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: ImageProtocol,
  request: &ImageRequest,
) -> Result<ImageResponse, BackendError> {
  let response = client.post_json(build_image_http_request(config, &protocol, request)?)?;
  protocol.decode_image_response(&response.body, request)
}

pub fn collect_stream_events(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: ChatProtocol,
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
  protocol: ChatProtocol,
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

#[cfg(test)]
pub(super) fn dispatch_stream_encoded_with<F>(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  source_protocol: ChatProtocol,
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

fn map_request_protocol_error(error: ProtocolError) -> BackendError {
  match error {
    ProtocolError::MissingResponseField(field) => BackendError::InvalidRequest {
      field,
      message: "missing field".to_string(),
    },
    ProtocolError::InvalidRequest { field, message } | ProtocolError::InvalidResponse { field, message } => {
      BackendError::InvalidRequest { field, message }
    }
    ProtocolError::Json(error) => BackendError::Json(error),
  }
}

fn map_response_protocol_error(error: ProtocolError) -> BackendError {
  match error {
    ProtocolError::MissingResponseField(field) => BackendError::InvalidResponse {
      field,
      message: "missing field".to_string(),
    },
    ProtocolError::InvalidRequest { field, message } | ProtocolError::InvalidResponse { field, message } => {
      BackendError::InvalidResponse { field, message }
    }
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
    Err(ProtocolError::InvalidResponse {
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
