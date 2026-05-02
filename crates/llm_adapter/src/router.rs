//! Fallback routing helpers for host applications.
//!
//! The public surface is centered on prepared routes so hosts can normalize
//! requests before dispatch and apply shared middleware consistently.

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
  backend::{
    BackendConfig, BackendError, BackendHttpClient, ChatProtocol, EmbeddingProtocol, ImageProtocol, RerankProtocol,
    StructuredProtocol, dispatch_embedding_request, dispatch_image_request, dispatch_request, dispatch_rerank_request,
    dispatch_stream_events_with, dispatch_structured_request,
  },
  core::{
    CoreRequest, CoreResponse, EmbeddingRequest, EmbeddingResponse, ImageRequest, ImageResponse, RerankRequest,
    RerankResponse, StreamEvent, StructuredRequest, StructuredResponse,
  },
  middleware::StreamPipeline,
};

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SerializablePreparedRoute<TRequest = Value> {
  pub provider_id: String,
  pub protocol: String,
  pub model: String,
  #[serde(alias = "backendConfig")]
  pub config: BackendConfig,
  pub request: TRequest,
}

#[derive(Debug, Clone)]
pub struct RoutedBackend {
  pub provider_id: String,
  pub protocol: ChatProtocol,
  pub model: String,
  pub config: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct RoutedImageBackend {
  pub provider_id: String,
  pub protocol: ImageProtocol,
  pub model: String,
  pub config: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct RoutedStructuredBackend {
  pub provider_id: String,
  pub protocol: StructuredProtocol,
  pub model: String,
  pub config: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct RoutedEmbeddingBackend {
  pub provider_id: String,
  pub protocol: EmbeddingProtocol,
  pub model: String,
  pub config: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct RoutedRerankBackend {
  pub provider_id: String,
  pub protocol: RerankProtocol,
  pub model: String,
  pub config: BackendConfig,
}

pub type PreparedChatRoute = (RoutedBackend, CoreRequest);
pub type PreparedStructuredRoute = (RoutedStructuredBackend, StructuredRequest);
pub type PreparedEmbeddingRoute = (RoutedEmbeddingBackend, EmbeddingRequest);
pub type PreparedRerankRoute = (RoutedRerankBackend, RerankRequest);
pub type PreparedImageRoute = (RoutedImageBackend, ImageRequest);
pub type PreparedStreamPipelineRoute = (PreparedChatRoute, StreamPipeline);

pub fn normalize_prepared_routes(value: Value) -> Result<Value, BackendError> {
  let routes: Vec<SerializablePreparedRoute> =
    serde_json::from_value(value).map_err(|error| BackendError::InvalidRequest {
      field: "preparedRoutes",
      message: error.to_string(),
    })?;
  serde_json::to_value(routes).map_err(|error| BackendError::InvalidRequest {
    field: "preparedRoutes",
    message: error.to_string(),
  })
}

pub fn serializable_prepared_routes_from_value<TRequest>(
  value: Value,
) -> Result<Vec<SerializablePreparedRoute<TRequest>>, BackendError>
where
  TRequest: for<'de> Deserialize<'de>,
{
  serde_json::from_value(value).map_err(|error| BackendError::InvalidRequest {
    field: "preparedRoutes",
    message: error.to_string(),
  })
}

pub fn serializable_prepared_routes_from_str<TRequest>(
  value: &str,
) -> Result<Vec<SerializablePreparedRoute<TRequest>>, BackendError>
where
  TRequest: for<'de> Deserialize<'de>,
{
  serde_json::from_str(value).map_err(|error| BackendError::InvalidRequest {
    field: "preparedRoutes",
    message: format!("Invalid JSON payload: {error}"),
  })
}

pub fn prepared_chat_routes_from_serializable<TRequest, Prepare>(
  routes: Vec<SerializablePreparedRoute<TRequest>>,
  mut prepare: Prepare,
) -> Result<Vec<PreparedChatRoute>, BackendError>
where
  Prepare:
    FnMut(TRequest, ChatProtocol, Option<crate::backend::BackendRequestLayer>) -> Result<CoreRequest, BackendError>,
{
  routes
    .into_iter()
    .map(|route| {
      let protocol = ChatProtocol::try_from(route.protocol.as_str())?;
      let request = prepare(route.request, protocol, route.config.request_layer)?;
      Ok((
        RoutedBackend {
          provider_id: route.provider_id,
          protocol,
          model: route.model,
          config: route.config,
        },
        request,
      ))
    })
    .collect()
}

pub fn prepared_structured_routes_from_serializable<TRequest, Prepare>(
  routes: Vec<SerializablePreparedRoute<TRequest>>,
  mut prepare: Prepare,
) -> Result<Vec<PreparedStructuredRoute>, BackendError>
where
  Prepare: FnMut(
    TRequest,
    StructuredProtocol,
    Option<crate::backend::BackendRequestLayer>,
  ) -> Result<StructuredRequest, BackendError>,
{
  routes
    .into_iter()
    .map(|route| {
      let protocol = StructuredProtocol::try_from(route.protocol.as_str())?;
      let request = prepare(route.request, protocol, route.config.request_layer)?;
      Ok((
        RoutedStructuredBackend {
          provider_id: route.provider_id,
          protocol,
          model: route.model,
          config: route.config,
        },
        request,
      ))
    })
    .collect()
}

pub fn prepared_embedding_routes_from_serializable<TRequest, Prepare>(
  routes: Vec<SerializablePreparedRoute<TRequest>>,
  mut prepare: Prepare,
) -> Result<Vec<PreparedEmbeddingRoute>, BackendError>
where
  Prepare: FnMut(TRequest) -> Result<EmbeddingRequest, BackendError>,
{
  routes
    .into_iter()
    .map(|route| {
      let protocol = EmbeddingProtocol::try_from(route.protocol.as_str())?;
      let request = prepare(route.request)?;
      Ok((
        RoutedEmbeddingBackend {
          provider_id: route.provider_id,
          protocol,
          model: route.model,
          config: route.config,
        },
        request,
      ))
    })
    .collect()
}

pub fn prepared_rerank_routes_from_serializable<TRequest, Prepare>(
  routes: Vec<SerializablePreparedRoute<TRequest>>,
  mut prepare: Prepare,
) -> Result<Vec<PreparedRerankRoute>, BackendError>
where
  Prepare: FnMut(TRequest) -> Result<RerankRequest, BackendError>,
{
  routes
    .into_iter()
    .map(|route| {
      let protocol = RerankProtocol::try_from(route.protocol.as_str())?;
      let request = prepare(route.request)?;
      Ok((
        RoutedRerankBackend {
          provider_id: route.provider_id,
          protocol,
          model: route.model,
          config: route.config,
        },
        request,
      ))
    })
    .collect()
}

pub fn prepared_image_routes_from_serializable<TRequest, Prepare>(
  routes: Vec<SerializablePreparedRoute<TRequest>>,
  mut prepare: Prepare,
) -> Result<Vec<PreparedImageRoute>, BackendError>
where
  Prepare: FnMut(TRequest) -> Result<ImageRequest, BackendError>,
{
  routes
    .into_iter()
    .map(|route| {
      let protocol = ImageProtocol::try_from(route.protocol.as_str())?;
      let request = prepare(route.request)?;
      Ok((
        RoutedImageBackend {
          provider_id: route.provider_id,
          protocol,
          model: route.model,
          config: route.config,
        },
        request,
      ))
    })
    .collect()
}

#[cfg(test)]
fn dispatch_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[RoutedBackend],
  request: &CoreRequest,
) -> Result<(String, CoreResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for route in routes {
    let routed_request = with_model(request, &route.model);
    match dispatch_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) => return Ok((route.provider_id.clone(), response)),
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

#[cfg(test)]
fn dispatch_stream_with_fallback<F>(
  client: &dyn BackendHttpClient,
  routes: &[RoutedBackend],
  request: &CoreRequest,
  mut on_event: F,
) -> Result<String, BackendError>
where
  F: FnMut(StreamEvent) -> Result<(), BackendError>,
{
  let mut last_error: Option<BackendError> = None;

  for route in routes {
    if route.config.no_streaming {
      continue;
    }

    let routed_request = with_model(request, &route.model);
    let mut emitted = false;
    match dispatch_stream_events_with(client, &route.config, route.protocol, &routed_request, |event| {
      emitted = true;
      on_event(event)
    }) {
      Ok(()) => return Ok(route.provider_id.clone()),
      Err(error) => {
        if emitted {
          return Err(error);
        }
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

pub fn dispatch_prepared_stream_with_fallback<F>(
  client: &dyn BackendHttpClient,
  routes: &[PreparedChatRoute],
  on_event: F,
) -> Result<String, BackendError>
where
  F: FnMut(usize, StreamEvent) -> Result<bool, BackendError>,
{
  dispatch_prepared_stream_with_fallback_index(client, routes, on_event).map(|(_, provider_id)| provider_id)
}

pub fn dispatch_prepared_stream_with_fallback_index<F>(
  client: &dyn BackendHttpClient,
  routes: &[PreparedChatRoute],
  mut on_event: F,
) -> Result<(usize, String), BackendError>
where
  F: FnMut(usize, StreamEvent) -> Result<bool, BackendError>,
{
  let mut last_error: Option<BackendError> = None;

  for (index, (route, request)) in routes.iter().enumerate() {
    if route.config.no_streaming {
      continue;
    }

    let mut routed_request = request.clone();
    routed_request.model = route.model.clone();
    let mut emitted = false;
    match dispatch_stream_events_with(client, &route.config, route.protocol, &routed_request, |event| {
      emitted |= on_event(index, event)?;
      Ok(())
    }) {
      Ok(()) => return Ok((index, route.provider_id.clone())),
      Err(error) => {
        if emitted {
          return Err(error);
        }
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

#[cfg(test)]
fn collect_stream_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[RoutedBackend],
  request: &CoreRequest,
) -> Result<(String, Vec<StreamEvent>), BackendError> {
  let mut events = Vec::new();
  let provider_id = dispatch_stream_with_fallback(client, routes, request, |event| {
    events.push(event);
    Ok(())
  })?;
  Ok((provider_id, events))
}

pub fn dispatch_prepared_stream_with_pipeline<Abort, AbortError, Emit>(
  client: &dyn BackendHttpClient,
  routes: &mut [PreparedStreamPipelineRoute],
  mut should_abort: Abort,
  mut abort_error: AbortError,
  mut emit: Emit,
) -> Result<String, BackendError>
where
  Abort: FnMut() -> bool,
  AbortError: FnMut() -> BackendError,
  Emit: FnMut(&StreamEvent) -> Result<(), BackendError>,
{
  let adapter_routes = routes.iter().map(|(route, _)| route.clone()).collect::<Vec<_>>();
  let (selected_index, provider_id) =
    dispatch_prepared_stream_with_fallback_index(client, &adapter_routes, |index, event| {
      if should_abort() {
        return Err(abort_error());
      }

      let mut emitted = false;
      for event in routes[index].1.process(event) {
        emit(&event)?;
        emitted = true;
      }

      Ok(emitted)
    })?;

  if !should_abort() {
    for event in routes[selected_index].1.finish() {
      if should_abort() {
        break;
      }
      emit(&event)?;
    }
  }

  Ok(provider_id)
}

pub fn dispatch_prepared_chat_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[PreparedChatRoute],
) -> Result<(String, CoreResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for (route, request) in routes {
    let mut routed_request = request.clone();
    routed_request.model = route.model.clone();
    match dispatch_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) => return Ok((route.provider_id.clone(), response)),
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

pub fn dispatch_structured_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[PreparedStructuredRoute],
) -> Result<(String, StructuredResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for (route, request) in routes {
    let mut routed_request = request.clone();
    routed_request.model = route.model.clone();
    match dispatch_structured_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) => return Ok((route.provider_id.clone(), response)),
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

pub fn dispatch_embedding_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[PreparedEmbeddingRoute],
) -> Result<(String, EmbeddingResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for (route, request) in routes {
    let mut routed_request = request.clone();
    routed_request.model = route.model.clone();
    match dispatch_embedding_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) => return Ok((route.provider_id.clone(), response)),
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

pub fn dispatch_rerank_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[PreparedRerankRoute],
) -> Result<(String, RerankResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for (route, request) in routes {
    let mut routed_request = request.clone();
    routed_request.model = route.model.clone();
    match dispatch_rerank_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) => return Ok((route.provider_id.clone(), response)),
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

pub fn dispatch_image_with_fallback(
  client: &dyn BackendHttpClient,
  routes: &[PreparedImageRoute],
) -> Result<(String, ImageResponse), BackendError> {
  let mut last_error: Option<BackendError> = None;

  for (route, request) in routes {
    let mut routed_request = request.clone();
    routed_request.set_model(route.model.clone());
    match dispatch_image_request(client, &route.config, route.protocol, &routed_request) {
      Ok(response) if !response.images.is_empty() => return Ok((route.provider_id.clone(), response)),
      Ok(_) => {
        last_error = Some(BackendError::InvalidResponse {
          field: "images",
          message: "expected at least one image".to_string(),
        })
      }
      Err(error) => {
        last_error = Some(error);
      }
    }
  }

  Err(last_error.unwrap_or(BackendError::NoBackendAvailable))
}

#[cfg(test)]
fn with_model(request: &CoreRequest, model: &str) -> CoreRequest {
  let mut routed_request = request.clone();
  routed_request.model = model.to_string();
  routed_request
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::{
    backend::{HttpResponse, HttpStreamResponse},
    core::{EmbeddingRequest, ImageOptions, ImageProviderOptions, RerankCandidate},
    test_support::{MockHttpClient, MockHttpResponse, sample_backend_config, sample_request},
  };

  #[test]
  fn should_fallback_for_non_stream_dispatch() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Json(Ok(HttpResponse {
          status: 200,
          body: json!({
            "id": "chat_2",
            "model": "gpt-4.1",
            "choices": [{
              "index": 0,
              "message": { "role": "assistant", "content": "ok" },
              "finish_reason": "stop"
            }],
            "usage": {
              "prompt_tokens": 10,
              "completion_tokens": 5,
              "total_tokens": 15
            }
          }),
        })),
      ],
      vec![],
    );

    let routes = vec![
      RoutedBackend {
        provider_id: "openai-primary".to_string(),
        protocol: ChatProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
      RoutedBackend {
        provider_id: "openai-fallback".to_string(),
        protocol: ChatProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
    ];

    let (provider_id, response) = dispatch_with_fallback(&client, &routes, &sample_request()).unwrap();

    assert_eq!(provider_id, "openai-fallback");
    assert_eq!(response.id, "chat_2");
  }

  #[test]
  fn should_fallback_for_prepared_chat_routes() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Json(Ok(HttpResponse {
          status: 200,
          body: json!({
            "id": "chat_2",
            "model": "gpt-4.1",
            "choices": [{
              "index": 0,
              "message": { "role": "assistant", "content": "ok" },
              "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
          }),
        })),
      ],
      vec![],
    );
    let request = sample_request();
    let routes = vec![
      (
        RoutedBackend {
          provider_id: "primary".to_string(),
          protocol: ChatProtocol::OpenaiChatCompletions,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(false),
        },
        request.clone(),
      ),
      (
        RoutedBackend {
          provider_id: "fallback".to_string(),
          protocol: ChatProtocol::OpenaiChatCompletions,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (provider_id, response) = dispatch_prepared_chat_with_fallback(&client, &routes).unwrap();

    assert_eq!(provider_id, "fallback");
    assert_eq!(response.id, "chat_2");
  }

  #[test]
  fn should_skip_no_streaming_route() {
    let client = MockHttpClient::new(
      vec![],
      vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
        status: 200,
        body: concat!(
          "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\"\
           :0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
          "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\"\
           :0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"\
           total_tokens\":15}}\n\n",
          "data: [DONE]\n\n"
        )
        .to_string(),
      }))],
    );

    let routes = vec![
      RoutedBackend {
        provider_id: "anthropic-no-stream".to_string(),
        protocol: ChatProtocol::AnthropicMessages,
        model: "claude-sonnet-4-5-20250929".to_string(),
        config: sample_backend_config(true),
      },
      RoutedBackend {
        provider_id: "openai-stream".to_string(),
        protocol: ChatProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
    ];

    let (provider_id, stream) = collect_stream_with_fallback(&client, &routes, &sample_request()).unwrap();

    assert_eq!(provider_id, "openai-stream");
    assert!(
      stream
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "hello"))
    );
  }

  #[test]
  fn should_return_selected_prepared_stream_route_index() {
    let client = MockHttpClient::new(
      vec![],
      vec![MockHttpResponse::Stream(Ok(HttpStreamResponse {
        status: 200,
        body: concat!(
          "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\"\
           :0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
          "data: [DONE]\n\n"
        )
        .to_string(),
      }))],
    );
    let request = sample_request();
    let routes = vec![
      (
        RoutedBackend {
          provider_id: "no-stream".to_string(),
          protocol: ChatProtocol::OpenaiChatCompletions,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(true),
        },
        request.clone(),
      ),
      (
        RoutedBackend {
          provider_id: "stream".to_string(),
          protocol: ChatProtocol::OpenaiChatCompletions,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (index, provider_id) =
      dispatch_prepared_stream_with_fallback_index(&client, &routes, |_index, _event| Ok(true)).unwrap();

    assert_eq!(index, 1);
    assert_eq!(provider_id, "stream");
  }

  #[test]
  fn should_flush_only_selected_prepared_stream_pipeline() {
    let client = MockHttpClient::new(
      vec![],
      vec![
        MockHttpResponse::Stream(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Stream(Ok(HttpStreamResponse {
          status: 200,
          body: concat!(
            "data: {\"id\":\"chat_2\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"\
             index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n"
          )
          .to_string(),
        })),
      ],
    );
    let request = sample_request();
    let mut routes = vec![
      (
        (
          RoutedBackend {
            provider_id: "primary".to_string(),
            protocol: ChatProtocol::OpenaiChatCompletions,
            model: "gpt-4.1".to_string(),
            config: sample_backend_config(false),
          },
          request.clone(),
        ),
        StreamPipeline::new(
          crate::middleware::resolve_stream_middleware_chain(&["stream_event_normalize".to_string()]).unwrap(),
          Default::default(),
        ),
      ),
      (
        (
          RoutedBackend {
            provider_id: "fallback".to_string(),
            protocol: ChatProtocol::OpenaiChatCompletions,
            model: "gpt-4.1".to_string(),
            config: sample_backend_config(false),
          },
          request,
        ),
        StreamPipeline::new(
          crate::middleware::resolve_stream_middleware_chain(&["stream_event_normalize".to_string()]).unwrap(),
          Default::default(),
        ),
      ),
    ];
    let mut emitted = Vec::new();

    let provider_id = dispatch_prepared_stream_with_pipeline(
      &client,
      &mut routes,
      || false,
      || BackendError::NoBackendAvailable,
      |event| {
        emitted.push(event.clone());
        Ok(())
      },
    )
    .unwrap();

    assert_eq!(provider_id, "fallback");
    assert!(
      emitted
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text } if text == "ok"))
    );
  }

  #[test]
  fn should_fallback_for_prepared_embedding_routes() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Json(Ok(HttpResponse {
          status: 200,
          body: json!({
            "model": "text-embedding-3-small",
            "data": [{ "embedding": [0.1, 0.2] }]
          }),
        })),
      ],
      vec![],
    );
    let request = EmbeddingRequest {
      model: "text-embedding-3-small".to_string(),
      inputs: vec!["hello".to_string()],
      dimensions: None,
      task_type: None,
    };
    let routes = vec![
      (
        RoutedEmbeddingBackend {
          provider_id: "primary".to_string(),
          protocol: EmbeddingProtocol::Openai,
          model: "text-embedding-3-small".to_string(),
          config: sample_backend_config(false),
        },
        request.clone(),
      ),
      (
        RoutedEmbeddingBackend {
          provider_id: "fallback".to_string(),
          protocol: EmbeddingProtocol::Openai,
          model: "text-embedding-3-small".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (provider_id, response) = dispatch_embedding_with_fallback(&client, &routes).unwrap();

    assert_eq!(provider_id, "fallback");
    assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);
  }

  #[test]
  fn should_fallback_for_prepared_structured_routes() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Json(Ok(HttpResponse {
          status: 200,
          body: json!({
            "id": "resp_1",
            "model": "gpt-4.1",
            "output": [{
              "type": "message",
              "content": [{ "type": "output_text", "text": "{\"summary\":\"ok\"}" }]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
          }),
        })),
      ],
      vec![],
    );
    let request = StructuredRequest {
      model: "gpt-4.1".to_string(),
      messages: sample_request().messages,
      schema: json!({
        "type": "object",
        "properties": { "summary": { "type": "string" } },
        "required": ["summary"]
      }),
      max_tokens: None,
      temperature: None,
      reasoning: None,
      strict: Some(true),
      response_mime_type: None,
    };
    let routes = vec![
      (
        RoutedStructuredBackend {
          provider_id: "primary".to_string(),
          protocol: StructuredProtocol::OpenaiResponses,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(false),
        },
        request.clone(),
      ),
      (
        RoutedStructuredBackend {
          provider_id: "fallback".to_string(),
          protocol: StructuredProtocol::OpenaiResponses,
          model: "gpt-4.1".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (provider_id, response) = dispatch_structured_with_fallback(&client, &routes).unwrap();

    assert_eq!(provider_id, "fallback");
    assert_eq!(response.output_json, Some(json!({ "summary": "ok" })));
  }

  #[test]
  fn should_fallback_for_prepared_rerank_routes() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
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
      ],
      vec![],
    );
    let request = RerankRequest {
      model: "gpt-5.2".to_string(),
      query: "programming".to_string(),
      candidates: vec![RerankCandidate {
        id: Some("js".to_string()),
        text: "Is JavaScript relevant?".to_string(),
      }],
      top_n: None,
    };
    let routes = vec![
      (
        RoutedRerankBackend {
          provider_id: "primary".to_string(),
          protocol: RerankProtocol::OpenaiChatLogprobs,
          model: "gpt-5.2".to_string(),
          config: sample_backend_config(false),
        },
        request.clone(),
      ),
      (
        RoutedRerankBackend {
          provider_id: "fallback".to_string(),
          protocol: RerankProtocol::OpenaiChatLogprobs,
          model: "gpt-5.2".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (provider_id, response) = dispatch_rerank_with_fallback(&client, &routes).unwrap();

    assert_eq!(provider_id, "fallback");
    assert_eq!(response.scores.len(), 1);
  }

  #[test]
  fn should_fallback_for_prepared_image_routes() {
    let client = MockHttpClient::new(
      vec![
        MockHttpResponse::Json(Err(BackendError::UpstreamStatus {
          status: 500,
          body: "boom".to_string(),
        })),
        MockHttpResponse::Json(Ok(HttpResponse {
          status: 200,
          body: json!({
            "data": [{ "b64_json": "aGVsbG8=" }],
            "usage": { "total_tokens": 3 }
          }),
        })),
      ],
      vec![],
    );
    let request = ImageRequest::generate(
      "gpt-image-1".to_string(),
      "draw a square".to_string(),
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );
    let routes = vec![
      (
        RoutedImageBackend {
          provider_id: "image-primary".to_string(),
          protocol: ImageProtocol::OpenaiImages,
          model: "gpt-image-1".to_string(),
          config: sample_backend_config(false),
        },
        request.clone(),
      ),
      (
        RoutedImageBackend {
          provider_id: "image-fallback".to_string(),
          protocol: ImageProtocol::OpenaiImages,
          model: "gpt-image-1".to_string(),
          config: sample_backend_config(false),
        },
        request,
      ),
    ];

    let (provider_id, response) = dispatch_image_with_fallback(&client, &routes).unwrap();

    assert_eq!(provider_id, "image-fallback");
    assert_eq!(response.images.len(), 1);
  }

  #[test]
  fn should_not_fallback_after_stream_has_started() {
    let client = MockHttpClient::new(
      vec![],
      vec![
        MockHttpResponse::Stream(Ok(HttpStreamResponse {
          status: 200,
          body: concat!(
            "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"\
             index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n"
          )
          .to_string(),
        })),
        MockHttpResponse::Stream(Ok(HttpStreamResponse {
          status: 200,
          body: concat!(
            "data: {\"id\":\"chat_2\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"\
             index\":0,\"delta\":{\"content\":\"fallback\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n"
          )
          .to_string(),
        })),
      ],
    );

    let routes = vec![
      RoutedBackend {
        provider_id: "openai-primary".to_string(),
        protocol: ChatProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
      RoutedBackend {
        provider_id: "openai-fallback".to_string(),
        protocol: ChatProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
    ];

    let result = dispatch_stream_with_fallback(&client, &routes, &sample_request(), |_event| {
      Err(BackendError::Transport {
        message: "stream consumer failed".to_string(),
      })
    });

    assert!(matches!(result, Err(BackendError::Transport { message }) if message == "stream consumer failed"));
  }

  #[test]
  fn should_fail_when_no_route_available() {
    let client = MockHttpClient::new(vec![], vec![]);
    let result = dispatch_with_fallback(&client, &[], &sample_request());

    assert!(matches!(result, Err(BackendError::NoBackendAvailable)));
  }

  #[test]
  fn should_normalize_serializable_prepared_routes() {
    let normalized = normalize_prepared_routes(json!([
      {
        "provider_id": "openai-main",
        "protocol": "openai_chat",
        "model": "gpt-5-mini",
        "config": {
          "base_url": "https://api.openai.com/v1",
          "auth_token": "token"
        },
        "request": {
          "model": "gpt-5-mini",
          "messages": []
        }
      }
    ]))
    .unwrap();

    assert_eq!(normalized[0]["provider_id"], json!("openai-main"));
    assert!(normalized[0]["config"]["headers"].is_null());
    assert!(normalized[0]["config"]["no_streaming"].is_null());

    let error = normalize_prepared_routes(json!([
      {
        "provider_id": "openai-main",
        "protocol": "openai_chat",
        "model": "gpt-5-mini",
        "config": { "base_url": "https://api.openai.com/v1" },
        "request": {}
      }
    ]))
    .unwrap_err();

    assert!(matches!(
      error,
      BackendError::InvalidRequest {
        field: "preparedRoutes",
        ..
      }
    ));
  }
}
