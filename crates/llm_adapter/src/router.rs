use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

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
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutableProtocol {
  Chat(ChatProtocol),
  Structured(StructuredProtocol),
  Embedding(EmbeddingProtocol),
  Rerank(RerankProtocol),
  Image(ImageProtocol),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutableRequest {
  Chat(CoreRequest),
  Structured(StructuredRequest),
  Embedding(EmbeddingRequest),
  Rerank(RerankRequest),
  Image(Box<ImageRequest>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutableResponse {
  Chat(CoreResponse),
  Structured(StructuredResponse),
  Embedding(EmbeddingResponse),
  Rerank(RerankResponse),
  Image(ImageResponse),
}

pub struct ExecutablePreparedRoute {
  pub protocol: ExecutableProtocol,
  pub model: String,
  pub config: BackendConfig,
  pub request: ExecutableRequest,
}

impl ExecutablePreparedRoute {
  pub fn new(
    protocol: ExecutableProtocol,
    model: String,
    config: BackendConfig,
    mut request: ExecutableRequest,
  ) -> Result<Self, BackendError> {
    if !request_matches_protocol(&request, protocol) {
      return Err(BackendError::InvalidRequest {
        field: "protocol",
        message: "request kind does not match executable protocol".to_string(),
      });
    }
    set_request_model(&mut request, &model);
    normalize_request_for_protocol(&mut request, protocol);
    Ok(Self {
      protocol,
      model,
      config,
      request,
    })
  }
}

fn normalize_request_for_protocol(request: &mut ExecutableRequest, protocol: ExecutableProtocol) {
  if protocol != ExecutableProtocol::Image(ImageProtocol::FalImage) {
    return;
  }
  let ExecutableRequest::Image(image) = request else {
    return;
  };
  let ImageRequest::Edit(request) = image.as_mut() else {
    return;
  };
  for image in &mut request.images {
    let replacement = match image {
      crate::core::ImageInput::Data {
        data_base64,
        media_type,
        ..
      } => Some(crate::core::ImageInput::Url {
        url: format!("data:{media_type};base64,{data_base64}"),
        media_type: Some(media_type.clone()),
      }),
      crate::core::ImageInput::Bytes { data, media_type, .. } => Some(crate::core::ImageInput::Url {
        url: format!("data:{media_type};base64,{}", STANDARD.encode(data)),
        media_type: Some(media_type.clone()),
      }),
      crate::core::ImageInput::Url { .. } => None,
    };
    if let Some(replacement) = replacement {
      *image = replacement;
    }
  }
}

fn request_matches_protocol(request: &ExecutableRequest, protocol: ExecutableProtocol) -> bool {
  matches!(
    (request, protocol),
    (ExecutableRequest::Chat(_), ExecutableProtocol::Chat(_))
      | (ExecutableRequest::Structured(_), ExecutableProtocol::Structured(_))
      | (ExecutableRequest::Embedding(_), ExecutableProtocol::Embedding(_))
      | (ExecutableRequest::Rerank(_), ExecutableProtocol::Rerank(_))
      | (ExecutableRequest::Image(_), ExecutableProtocol::Image(_))
  )
}

fn set_request_model(request: &mut ExecutableRequest, model: &str) {
  match request {
    ExecutableRequest::Chat(request) => request.model = model.to_string(),
    ExecutableRequest::Structured(request) => request.model = model.to_string(),
    ExecutableRequest::Embedding(request) => request.model = model.to_string(),
    ExecutableRequest::Rerank(request) => request.model = model.to_string(),
    ExecutableRequest::Image(request) => match request.as_mut() {
      ImageRequest::Generate(request) => request.model = model.to_string(),
      ImageRequest::Edit(request) => request.model = model.to_string(),
    },
  }
}

pub fn dispatch_prepared_route(
  client: &dyn BackendHttpClient,
  route: &ExecutablePreparedRoute,
) -> Result<ExecutableResponse, BackendError> {
  match (&route.protocol, &route.request) {
    (ExecutableProtocol::Chat(protocol), ExecutableRequest::Chat(request)) => {
      dispatch_request(client, &route.config, *protocol, request).map(ExecutableResponse::Chat)
    }
    (ExecutableProtocol::Structured(protocol), ExecutableRequest::Structured(request)) => {
      dispatch_structured_request(client, &route.config, *protocol, request).map(ExecutableResponse::Structured)
    }
    (ExecutableProtocol::Embedding(protocol), ExecutableRequest::Embedding(request)) => {
      dispatch_embedding_request(client, &route.config, *protocol, request).map(ExecutableResponse::Embedding)
    }
    (ExecutableProtocol::Rerank(protocol), ExecutableRequest::Rerank(request)) => {
      dispatch_rerank_request(client, &route.config, *protocol, request).map(ExecutableResponse::Rerank)
    }
    (ExecutableProtocol::Image(protocol), ExecutableRequest::Image(request)) => {
      dispatch_image_request(client, &route.config, *protocol, request).map(ExecutableResponse::Image)
    }
    _ => Err(BackendError::InvalidRequest {
      field: "protocol",
      message: "request kind does not match executable protocol".to_string(),
    }),
  }
}

pub fn dispatch_prepared_stream<Emit>(
  client: &dyn BackendHttpClient,
  route: &ExecutablePreparedRoute,
  emit: Emit,
) -> Result<(), BackendError>
where
  Emit: FnMut(StreamEvent) -> Result<(), BackendError>,
{
  match (&route.protocol, &route.request) {
    (ExecutableProtocol::Chat(protocol), ExecutableRequest::Chat(request)) => {
      dispatch_stream_events_with(client, &route.config, *protocol, request, emit)
    }
    _ => Err(BackendError::InvalidRequest {
      field: "protocol",
      message: "stream dispatch requires a chat route".to_string(),
    }),
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RedactedRouteTrace {
  pub route_id: String,
  pub outcome: String,
  pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use serde_json::json;

  use super::*;
  use crate::{
    backend::{BackendRequestLayer, HttpRequest, HttpResponse},
    core::{CoreMessage, CoreRole},
    target::EgressPolicy,
  };

  struct Client;

  impl BackendHttpClient for Client {
    fn post_json(&self, _request: HttpRequest) -> Result<HttpResponse, BackendError> {
      Ok(HttpResponse {
        status: 200,
        body: json!({
          "id": "response_1",
          "model": "opaque/model:latest",
          "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        }),
      })
    }

    fn post_sse(
      &self,
      _request: HttpRequest,
      _on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
      Ok(())
    }
  }

  #[test]
  fn dispatches_one_non_serializable_route() {
    let request = CoreRequest {
      model: "ignored".to_string(),
      messages: vec![CoreMessage {
        role: CoreRole::User,
        content: vec![],
      }],
      stream: false,
      max_tokens: None,
      temperature: None,
      tools: Vec::new(),
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };
    let route = ExecutablePreparedRoute::new(
      ExecutableProtocol::Chat(ChatProtocol::OpenaiChatCompletions),
      "opaque/model:latest".to_string(),
      BackendConfig {
        base_url: "https://example.com/v1".to_string(),
        auth_token: "secret".into(),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
        no_streaming: false,
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      },
      ExecutableRequest::Chat(request),
    )
    .unwrap();

    assert!(matches!(
      dispatch_prepared_route(&Client, &route).unwrap(),
      ExecutableResponse::Chat(_)
    ));
  }
}
