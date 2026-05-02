use llm_adapter::core::{CoreRequest, EmbeddingRequest, ImageRequest, RerankRequest, StructuredRequest};
#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutionPlanCompileError {
  #[error("invalid execution plan: {0}")]
  InvalidJson(#[from] serde_json::Error),
  #[error("ExecutionPlan cannot serialize host-only {owner}.{key}")]
  HostOnlyState { owner: String, key: String },
}

pub fn compile_execution_plan_value(value: Value) -> Result<SerializableExecutionPlan, ExecutionPlanCompileError> {
  let plan = serde_json::from_value::<SerializableExecutionPlan>(value)?;
  reject_host_only_execution_plan_state(&plan)?;
  Ok(plan)
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct SerializableExecutionPlan {
  pub routes: Vec<ExecutionRoute>,
  pub request: ExecutionPlanRequest,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub transport: Option<ExecutionTransport>,
  pub route_policy: ExecutionRoutePolicy,
  pub runtime_policy: ExecutionRuntimePolicy,
  pub attachment_policy: ExecutionAttachmentPolicy,
  pub response_postprocess: ExecutionResponsePostprocess,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub host_context: Option<ExecutionHostContext>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionRoute {
  pub provider_id: String,
  pub protocol: String,
  pub model: String,
  pub backend_config: Value,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind")]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub enum ExecutionPlanRequest {
  #[serde(rename = "text")]
  Text {
    cond: Value,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "streamText")]
  StreamText {
    cond: Value,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "streamObject")]
  StreamObject {
    cond: Value,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "structured")]
  Structured {
    cond: Value,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "image")]
  Image {
    cond: Value,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "embedding")]
  Embedding {
    cond: Value,
    #[serde(rename = "modelId")]
    model_id: String,
    input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
  #[serde(rename = "rerank")]
  Rerank {
    cond: Value,
    #[serde(rename = "modelId")]
    model_id: String,
    request: RerankExecutionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
  },
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct RerankExecutionRequest {
  pub query: String,
  pub candidates: Vec<RerankExecutionCandidate>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub top_k: Option<u32>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct RerankExecutionCandidate {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub id: Option<String>,
  pub text: String,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind")]
#[serde(deny_unknown_fields)]
pub enum ExecutionTransport {
  #[serde(rename = "chat")]
  Chat { request: CoreRequest },
  #[serde(rename = "structured")]
  Structured { request: StructuredRequest },
  #[serde(rename = "embedding")]
  Embedding { request: EmbeddingRequest },
  #[serde(rename = "rerank")]
  Rerank { request: RerankRequest },
  #[serde(rename = "image")]
  Image { request: ImageRequest },
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionRoutePolicy {
  pub fallback_order: Vec<String>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionRuntimePolicy {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub prefer: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_steps: Option<u32>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionAttachmentPolicy {
  pub materialize_remote_attachments: bool,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionResponsePostprocess {
  pub mode: String,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ExecutionHostContext {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub current_messages: Option<Vec<Value>>,
}

fn request_options(request: &ExecutionPlanRequest) -> Option<&Value> {
  match request {
    ExecutionPlanRequest::Text { options, .. }
    | ExecutionPlanRequest::StreamText { options, .. }
    | ExecutionPlanRequest::StreamObject { options, .. }
    | ExecutionPlanRequest::Structured { options, .. }
    | ExecutionPlanRequest::Image { options, .. }
    | ExecutionPlanRequest::Embedding { options, .. }
    | ExecutionPlanRequest::Rerank { options, .. } => options.as_ref(),
  }
}

fn reject_option_keys(options: Option<&Value>, owner: &str) -> Result<(), ExecutionPlanCompileError> {
  let Some(options) = options.and_then(Value::as_object) else {
    return Ok(());
  };

  for key in ["signal", "user", "session", "workspace"] {
    if options.contains_key(key) {
      return Err(ExecutionPlanCompileError::HostOnlyState {
        owner: owner.to_string(),
        key: key.to_string(),
      });
    }
  }

  Ok(())
}

pub fn reject_host_only_execution_plan_state(
  plan: &SerializableExecutionPlan,
) -> Result<(), ExecutionPlanCompileError> {
  reject_option_keys(request_options(&plan.request), "request.options")
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{SerializableExecutionPlan, compile_execution_plan_value};

  #[test]
  fn rejects_unknown_execution_plan_fields() {
    let error = serde_json::from_value::<SerializableExecutionPlan>(json!({
      "routes": [],
      "request": { "kind": "text", "cond": {}, "messages": [] },
      "routePolicy": { "fallbackOrder": [] },
      "runtimePolicy": {},
      "attachmentPolicy": { "materializeRemoteAttachments": true },
      "responsePostprocess": { "mode": "text" },
      "extra": true
    }))
    .unwrap_err();

    assert!(error.to_string().contains("unknown field"));
  }

  #[test]
  fn compile_rejects_host_only_request_options() {
    let error = compile_execution_plan_value(json!({
      "routes": [],
      "request": {
        "kind": "text",
        "cond": {},
        "messages": [],
        "options": { "signal": {} }
      },
      "routePolicy": { "fallbackOrder": [] },
      "runtimePolicy": {},
      "attachmentPolicy": { "materializeRemoteAttachments": true },
      "responsePostprocess": { "mode": "text" }
    }))
    .unwrap_err();

    assert!(error.to_string().contains("request.options.signal"));
  }
}
