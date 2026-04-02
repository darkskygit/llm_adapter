//! Fallback routing helpers for host applications.
//!
//! These helpers are intentionally public so callers can build multi-provider
//! retry/fallback policies on top of this library.

use crate::{
  backend::{
    BackendConfig, BackendError, BackendHttpClient, BackendProtocol, dispatch_request, dispatch_stream_events_with,
  },
  core::{CoreRequest, CoreResponse, StreamEvent},
};

#[derive(Debug, Clone)]
pub struct RoutedBackend {
  pub provider_id: String,
  pub protocol: BackendProtocol,
  pub model: String,
  pub config: BackendConfig,
}

pub fn dispatch_with_fallback(
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

pub fn dispatch_stream_with_fallback<F>(
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

pub fn collect_stream_with_fallback(
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
        protocol: BackendProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
      RoutedBackend {
        provider_id: "openai-fallback".to_string(),
        protocol: BackendProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
    ];

    let (provider_id, response) = dispatch_with_fallback(&client, &routes, &sample_request()).unwrap();

    assert_eq!(provider_id, "openai-fallback");
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
        protocol: BackendProtocol::AnthropicMessages,
        model: "claude-sonnet-4-5-20250929".to_string(),
        config: sample_backend_config(true),
      },
      RoutedBackend {
        provider_id: "openai-stream".to_string(),
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
      RoutedBackend {
        provider_id: "openai-fallback".to_string(),
        protocol: BackendProtocol::OpenaiChatCompletions,
        model: "gpt-4.1".to_string(),
        config: sample_backend_config(false),
      },
    ];

    let result = dispatch_stream_with_fallback(&client, &routes, &sample_request(), |_event| {
      Err(BackendError::Http("stream consumer failed".to_string()))
    });

    assert!(matches!(result, Err(BackendError::Http(message)) if message == "stream consumer failed"));
  }

  #[test]
  fn should_fail_when_no_route_available() {
    let client = MockHttpClient::new(vec![], vec![]);
    let result = dispatch_with_fallback(&client, &[], &sample_request());

    assert!(matches!(result, Err(BackendError::NoBackendAvailable)));
  }
}
