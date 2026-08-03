use llm_adapter::{
  backend::{BackendError, BackendHttpClient},
  core::{CoreMessage, CoreUsage, ImageUsage, StreamEvent},
  router::{ExecutablePreparedRoute, ExecutableResponse, dispatch_prepared_route, dispatch_prepared_stream},
};
use thiserror::Error;

use crate::{RoundOutcome, RoundProcessorError, round::StreamRoundRunner};

pub struct CompiledRoute {
  route_id: String,
  route: ExecutablePreparedRoute,
}

impl CompiledRoute {
  #[must_use]
  pub fn new(route_id: String, route: ExecutablePreparedRoute) -> Self {
    Self { route_id, route }
  }

  #[must_use]
  pub fn route_id(&self) -> &str {
    &self.route_id
  }
}

pub struct CompiledPlan {
  candidates: Vec<CompiledRoute>,
}

#[derive(Debug, Error)]
pub enum CompiledPlanError {
  #[error("compiled plan requires at least one route")]
  NoCandidates,
  #[error(transparent)]
  Backend(#[from] BackendError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeRouteEvent {
  Selected { route_id: String },
  Failed { route_id: String, error_kind: String },
  Usage { route_id: String, usage: RuntimeUsage },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeUsage {
  Tokens(CoreUsage),
  Image(ImageUsage),
}

impl CompiledPlan {
  pub fn new(candidates: Vec<CompiledRoute>) -> Result<Self, CompiledPlanError> {
    if candidates.is_empty() {
      Err(CompiledPlanError::NoCandidates)
    } else {
      Ok(Self { candidates })
    }
  }

  pub fn route_ids(&self) -> impl Iterator<Item = &str> {
    self.candidates.iter().map(|candidate| candidate.route_id.as_str())
  }

  pub fn replace_chat_messages(&mut self, messages: &[CoreMessage]) -> Result<(), CompiledPlanError> {
    for candidate in &mut self.candidates {
      let llm_adapter::router::ExecutableRequest::Chat(request) = &mut candidate.route.request else {
        return Err(
          BackendError::InvalidRequest {
            field: "request",
            message: "tool loop requires chat routes".to_string(),
          }
          .into(),
        );
      };
      request.messages = messages.to_vec();
      request.stream = true;
    }
    Ok(())
  }
}

pub fn dispatch_compiled_round<Abort, EmitEvent, EmitRoute>(
  client: &dyn BackendHttpClient,
  plan: &mut CompiledPlan,
  messages: &[CoreMessage],
  mut should_abort: Abort,
  mut emit_event: EmitEvent,
  mut emit_route: EmitRoute,
) -> Result<RoundOutcome, CompiledPlanError>
where
  Abort: FnMut() -> bool,
  EmitEvent: FnMut(&crate::ToolLoopEvent) -> Result<(), BackendError>,
  EmitRoute: FnMut(RuntimeRouteEvent),
{
  plan.replace_chat_messages(messages)?;
  let mut last_error = None;
  for candidate in &plan.candidates {
    if should_abort() {
      return Err(
        BackendError::Transport {
          message: "stream aborted".to_string(),
        }
        .into(),
      );
    }
    let mut runner = StreamRoundRunner::default();
    let mut emitted_content = false;
    let result = dispatch_prepared_stream(client, &candidate.route, |event| {
      if should_abort() {
        return Err(BackendError::Transport {
          message: "stream aborted".to_string(),
        });
      }
      if let StreamEvent::Usage { usage } = &event {
        emit_route(RuntimeRouteEvent::Usage {
          route_id: candidate.route_id.clone(),
          usage: RuntimeUsage::Tokens(usage.clone()),
        });
      }
      runner.process_event_with(
        event,
        |error: RoundProcessorError| BackendError::Transport {
          message: error.to_string(),
        },
        |event| {
          emitted_content = true;
          emit_event(&event)
        },
      )
    });
    match result {
      Ok(()) => {
        emit_route(RuntimeRouteEvent::Selected {
          route_id: candidate.route_id.clone(),
        });
        return Ok(runner.finish());
      }
      Err(error) => {
        emit_route(RuntimeRouteEvent::Failed {
          route_id: candidate.route_id.clone(),
          error_kind: error_kind(&error).to_string(),
        });
        if emitted_content {
          return Err(error.into());
        }
        last_error = Some(error);
      }
    }
  }
  Err(last_error.unwrap_or(BackendError::NoBackendAvailable).into())
}

pub fn dispatch_compiled_plan<Emit>(
  client: &dyn BackendHttpClient,
  plan: &CompiledPlan,
  mut emit: Emit,
) -> Result<ExecutableResponse, CompiledPlanError>
where
  Emit: FnMut(RuntimeRouteEvent),
{
  let mut last_error = None;
  for candidate in &plan.candidates {
    match dispatch_prepared_route(client, &candidate.route) {
      Ok(response) => {
        emit(RuntimeRouteEvent::Selected {
          route_id: candidate.route_id.clone(),
        });
        if let Some(usage) = response_usage(&response) {
          emit(RuntimeRouteEvent::Usage {
            route_id: candidate.route_id.clone(),
            usage,
          });
        }
        return Ok(response);
      }
      Err(error) => {
        emit(RuntimeRouteEvent::Failed {
          route_id: candidate.route_id.clone(),
          error_kind: error_kind(&error).to_string(),
        });
        last_error = Some(error);
      }
    }
  }
  Err(last_error.unwrap_or(BackendError::NoBackendAvailable).into())
}

pub fn dispatch_compiled_stream<EmitEvent, EmitRoute>(
  client: &dyn BackendHttpClient,
  plan: &CompiledPlan,
  mut emit_event: EmitEvent,
  mut emit_route: EmitRoute,
) -> Result<(), CompiledPlanError>
where
  EmitEvent: FnMut(StreamEvent) -> Result<(), BackendError>,
  EmitRoute: FnMut(RuntimeRouteEvent),
{
  let mut last_error = None;
  for candidate in &plan.candidates {
    let mut emitted_content = false;
    let result = dispatch_prepared_stream(client, &candidate.route, |event| {
      emitted_content = true;
      if let StreamEvent::Usage { usage } = &event {
        emit_route(RuntimeRouteEvent::Usage {
          route_id: candidate.route_id.clone(),
          usage: RuntimeUsage::Tokens(usage.clone()),
        });
      }
      emit_event(event)
    });
    match result {
      Ok(()) => {
        emit_route(RuntimeRouteEvent::Selected {
          route_id: candidate.route_id.clone(),
        });
        return Ok(());
      }
      Err(error) => {
        emit_route(RuntimeRouteEvent::Failed {
          route_id: candidate.route_id.clone(),
          error_kind: error_kind(&error).to_string(),
        });
        if emitted_content {
          return Err(error.into());
        }
        last_error = Some(error);
      }
    }
  }
  Err(last_error.unwrap_or(BackendError::NoBackendAvailable).into())
}

fn response_usage(response: &ExecutableResponse) -> Option<RuntimeUsage> {
  match response {
    ExecutableResponse::Chat(response) => Some(RuntimeUsage::Tokens(response.usage.clone())),
    ExecutableResponse::Structured(response) => Some(RuntimeUsage::Tokens(response.usage.clone())),
    ExecutableResponse::Image(response) => response.usage.clone().map(RuntimeUsage::Image),
    ExecutableResponse::Embedding(_) | ExecutableResponse::Rerank(_) => None,
  }
}

fn error_kind(error: &BackendError) -> &'static str {
  match error {
    BackendError::NoBackendAvailable => "no_backend_available",
    BackendError::InvalidConfig { .. } => "invalid_config",
    BackendError::InvalidRequest { .. } => "invalid_request",
    BackendError::Transport { .. } => "transport",
    BackendError::Timeout { .. } => "timeout",
    BackendError::UpstreamStatus { .. } => "upstream_status",
    BackendError::InvalidResponse { .. } => "invalid_response",
    BackendError::InvalidStructuredOutput { .. } => "invalid_structured_output",
    BackendError::Json(_) => "json",
    BackendError::Stream(_) => "stream",
  }
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use llm_adapter::{
    backend::{BackendConfig, BackendHttpClient, BackendRequestLayer, ChatProtocol, HttpRequest, HttpResponse},
    core::{CoreMessage, CoreRequest, CoreRole},
    router::{ExecutablePreparedRoute, ExecutableProtocol, ExecutableRequest},
    target::EgressPolicy,
  };
  use serde_json::json;

  use super::*;

  struct Client;

  impl BackendHttpClient for Client {
    fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, BackendError> {
      if request.url.contains("failed.example") {
        return Err(BackendError::Transport {
          message: "failed".to_string(),
        });
      }
      Ok(HttpResponse {
        status: 200,
        body: json!({"id":"ok","model":"m","choices":[{"message":{"role":"assistant","content":"ok"}}]}),
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

  fn route(base_url: &str) -> ExecutablePreparedRoute {
    ExecutablePreparedRoute::new(
      ExecutableProtocol::Chat(ChatProtocol::OpenaiChatCompletions),
      "m".to_string(),
      BackendConfig {
        base_url: base_url.to_string(),
        auth_token: "secret".into(),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
        no_streaming: false,
        timeout_ms: None,
        egress_policy: EgressPolicy::PublicOnly,
      },
      ExecutableRequest::Chat(CoreRequest {
        model: "m".to_string(),
        messages: vec![CoreMessage {
          role: CoreRole::User,
          content: Vec::new(),
        }],
        stream: false,
        max_tokens: None,
        temperature: None,
        tools: Vec::new(),
        tool_choice: None,
        include: None,
        reasoning: None,
        response_schema: None,
      }),
    )
    .unwrap()
  }

  #[test]
  fn owns_candidate_fallback_and_emits_only_route_ids() {
    let plan = CompiledPlan::new(vec![
      CompiledRoute::new("route-a".to_string(), route("https://failed.example/v1")),
      CompiledRoute::new("route-b".to_string(), route("https://ok.example/v1")),
    ])
    .unwrap();
    let mut events = Vec::new();
    dispatch_compiled_plan(&Client, &plan, |event| events.push(event)).unwrap();
    assert!(matches!(events[0], RuntimeRouteEvent::Failed { ref route_id, .. } if route_id == "route-a"));
    assert!(matches!(events[1], RuntimeRouteEvent::Selected { ref route_id } if route_id == "route-b"));
  }
}
