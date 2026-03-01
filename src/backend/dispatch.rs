use serde_json::Value;

use super::{
  super::{
    core::{CoreRequest, CoreResponse, StreamEvent},
    protocol::{ProtocolError, anthropic, openai},
    stream::{
      AnthropicStreamParser, IncrementalSseEncoder, OpenaiChatStreamParser, OpenaiResponsesStreamParser, SseFrame,
      SseFrameDecoder, StreamEncodingTarget, StreamParseError, encode_sse_frame,
    },
  },
  BackendConfig, BackendError, BackendHttpClient, BackendProtocol, HttpRequest,
  request_layer::{build_extra_headers, resolve_request_layer},
};

impl BackendProtocol {
  fn encode_request(&self, request: &CoreRequest, stream: bool) -> Value {
    match self {
      BackendProtocol::OpenaiChatCompletions => openai::chat::request::encode(request, stream),
      BackendProtocol::OpenaiResponses => openai::responses::request::encode(request, stream),
      BackendProtocol::AnthropicMessages => anthropic::request::encode(request, stream),
    }
  }

  fn decode_response(&self, body: &Value) -> Result<CoreResponse, ProtocolError> {
    match self {
      BackendProtocol::OpenaiChatCompletions => openai::chat::response::decode(body),
      BackendProtocol::OpenaiResponses => openai::responses::response::decode(body),
      BackendProtocol::AnthropicMessages => anthropic::response::decode(body),
    }
  }
}

enum IncrementalStreamParser {
  OpenaiChat(OpenaiChatStreamParser),
  OpenaiResponses(OpenaiResponsesStreamParser),
  Anthropic(AnthropicStreamParser),
}

impl IncrementalStreamParser {
  fn new(protocol: &BackendProtocol) -> Self {
    match protocol {
      BackendProtocol::OpenaiChatCompletions => Self::OpenaiChat(OpenaiChatStreamParser::default()),
      BackendProtocol::OpenaiResponses => Self::OpenaiResponses(OpenaiResponsesStreamParser::default()),
      BackendProtocol::AnthropicMessages => Self::Anthropic(AnthropicStreamParser::default()),
    }
  }

  fn push_frame(&mut self, frame: SseFrame) -> Result<Vec<StreamEvent>, StreamParseError> {
    match self {
      IncrementalStreamParser::OpenaiChat(parser) => parser.push_frame(frame),
      IncrementalStreamParser::OpenaiResponses(parser) => parser.push_frame(frame),
      IncrementalStreamParser::Anthropic(parser) => parser.push_frame(frame),
    }
  }

  fn finish(&mut self) -> Vec<StreamEvent> {
    match self {
      IncrementalStreamParser::OpenaiChat(parser) => parser.finish(),
      IncrementalStreamParser::OpenaiResponses(parser) => parser.finish(),
      IncrementalStreamParser::Anthropic(parser) => parser.finish(),
    }
  }
}

fn build_http_request(
  config: &BackendConfig,
  protocol: &BackendProtocol,
  request: &CoreRequest,
  stream: bool,
) -> Result<HttpRequest, BackendError> {
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

pub fn dispatch_request(
  client: &dyn BackendHttpClient,
  config: &BackendConfig,
  protocol: BackendProtocol,
  request: &CoreRequest,
) -> Result<CoreResponse, BackendError> {
  let response = client.post_json(build_http_request(config, &protocol, request, false)?)?;
  protocol.decode_response(&response.body).map_err(map_protocol_error)
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
