use jsonschema::Draft;
#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeStepExecution {
  pub id: String,
  pub kind: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub input: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub state_patch: Option<Value>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeRuntimeStatus {
  Created,
  Running,
  Succeeded,
  Failed,
  Aborted,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeDefinition {
  pub id: String,
  pub version: String,
  #[serde(default)]
  pub steps: Vec<RecipeStepExecution>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeStepState {
  pub id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub input: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub error: Option<StepExecutionError>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeRuntimeEvent {
  pub event_type: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub step_id: Option<String>,
  pub status: RecipeRuntimeStatus,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub result: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub error: Option<StepExecutionError>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeRuntimeTrace {
  pub recipe_id: String,
  pub recipe_version: String,
  pub status: RecipeRuntimeStatus,
  #[serde(default)]
  pub events: Vec<RecipeRuntimeEvent>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub error_code: Option<String>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeRuntimeOutput {
  pub result: Value,
  pub state: Value,
  pub status: RecipeRuntimeStatus,
  pub steps: Vec<RecipeStepState>,
  pub trace: RecipeRuntimeTrace,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[error("{code}: {message}")]
pub struct StepExecutionError {
  pub code: String,
  pub message: String,
}

impl StepExecutionError {
  pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
    Self {
      code: code.into(),
      message: message.into(),
    }
  }
}

pub trait RecipeStepExecutor {
  fn execute_step(
    &mut self,
    step: &RecipeStepExecution,
    input: Option<Value>,
    state: &Value,
  ) -> Result<Value, StepExecutionError>;
}

pub fn run_recipe_runtime<E, Emit, Abort>(
  recipe: RecipeDefinition,
  input: Value,
  executor: &mut E,
  mut emit: Emit,
  mut should_abort: Abort,
) -> RecipeRuntimeOutput
where
  E: RecipeStepExecutor,
  Emit: FnMut(&RecipeRuntimeEvent),
  Abort: FnMut() -> bool,
{
  let mut state = input.clone();
  let mut result = input;
  let mut steps = Vec::new();
  let mut trace = RecipeRuntimeTrace {
    recipe_id: recipe.id.clone(),
    recipe_version: recipe.version.clone(),
    status: RecipeRuntimeStatus::Running,
    events: Vec::new(),
    error_code: None,
  };

  record_recipe_event(
    &mut trace,
    &mut emit,
    RecipeRuntimeEvent {
      event_type: "recipe_start".to_string(),
      step_id: None,
      status: RecipeRuntimeStatus::Running,
      result: None,
      error: None,
    },
  );

  for step in recipe.steps {
    if should_abort() {
      fail_recipe(&mut trace, &mut emit, "aborted", "Recipe execution was aborted");
      break;
    }

    let step_input = step
      .input
      .as_ref()
      .map(|input| resolve_state_refs(input, &state))
      .unwrap_or_else(|| state.clone());
    record_recipe_event(
      &mut trace,
      &mut emit,
      RecipeRuntimeEvent {
        event_type: "step_start".to_string(),
        step_id: Some(step.id.clone()),
        status: RecipeRuntimeStatus::Running,
        result: None,
        error: None,
      },
    );

    let execution = RecipeStepExecution {
      input: Some(step_input.clone()),
      ..step.clone()
    };
    let output = execute_recipe_step(executor, &execution, Some(step_input.clone()), &state);
    match output {
      Ok(output) => {
        if let Some(output_key) = step_input.get("outputKey").and_then(Value::as_str) {
          set_state_path(&mut state, output_key, output.clone());
        }
        if step.kind == "final" {
          result = output.clone();
        }
        if let Some(patch) = step.state_patch {
          let patch = resolve_state_refs(&patch, &state);
          merge_state_patch(&mut state, patch);
        }
        steps.push(RecipeStepState {
          id: step.id.clone(),
          input: Some(step_input),
          output: Some(output),
          error: None,
        });
        record_recipe_event(
          &mut trace,
          &mut emit,
          RecipeRuntimeEvent {
            event_type: "step_end".to_string(),
            step_id: Some(step.id),
            status: RecipeRuntimeStatus::Running,
            result: Some(result.clone()),
            error: None,
          },
        );
      }
      Err(error) => {
        steps.push(RecipeStepState {
          id: step.id.clone(),
          input: Some(step_input),
          output: None,
          error: Some(error.clone()),
        });
        trace.status = RecipeRuntimeStatus::Failed;
        trace.error_code = Some(error.code.clone());
        record_recipe_event(
          &mut trace,
          &mut emit,
          RecipeRuntimeEvent {
            event_type: "error".to_string(),
            step_id: Some(step.id),
            status: RecipeRuntimeStatus::Failed,
            result: None,
            error: Some(error),
          },
        );
        return RecipeRuntimeOutput {
          result,
          state,
          status: RecipeRuntimeStatus::Failed,
          steps,
          trace,
        };
      }
    }
  }

  if trace.status != RecipeRuntimeStatus::Failed && trace.status != RecipeRuntimeStatus::Aborted {
    trace.status = RecipeRuntimeStatus::Succeeded;
    record_recipe_event(
      &mut trace,
      &mut emit,
      RecipeRuntimeEvent {
        event_type: "recipe_done".to_string(),
        step_id: None,
        status: RecipeRuntimeStatus::Succeeded,
        result: Some(result.clone()),
        error: None,
      },
    );
  }

  RecipeRuntimeOutput {
    result,
    state,
    status: trace.status.clone(),
    steps,
    trace,
  }
}

fn execute_recipe_step<E: RecipeStepExecutor>(
  executor: &mut E,
  step: &RecipeStepExecution,
  input: Option<Value>,
  state: &Value,
) -> Result<Value, StepExecutionError> {
  match step.kind.as_str() {
    "validateJson" => execute_validate_json_step(input),
    "transform" => match execute_transform_step(input.clone(), state)? {
      Some(value) => Ok(value),
      None => executor.execute_step(step, input, state),
    },
    "final" => match execute_transform_step(input.clone(), state)? {
      Some(value) => Ok(value),
      None => Ok(input.unwrap_or_else(|| state.clone())),
    },
    _ => executor.execute_step(step, input, state),
  }
}

fn record_recipe_event<Emit>(trace: &mut RecipeRuntimeTrace, emit: &mut Emit, event: RecipeRuntimeEvent)
where
  Emit: FnMut(&RecipeRuntimeEvent),
{
  emit(&event);
  trace.events.push(event);
}

fn fail_recipe<Emit>(
  trace: &mut RecipeRuntimeTrace,
  emit: &mut Emit,
  code: impl Into<String>,
  message: impl Into<String>,
) where
  Emit: FnMut(&RecipeRuntimeEvent),
{
  let error = StepExecutionError::new(code, message);
  trace.status = RecipeRuntimeStatus::Aborted;
  trace.error_code = Some(error.code.clone());
  record_recipe_event(
    trace,
    emit,
    RecipeRuntimeEvent {
      event_type: "error".to_string(),
      step_id: None,
      status: RecipeRuntimeStatus::Aborted,
      result: None,
      error: Some(error),
    },
  );
}

fn resolve_state_refs(value: &Value, state: &Value) -> Value {
  match value {
    Value::Array(items) => Value::Array(items.iter().map(|item| resolve_state_refs(item, state)).collect()),
    Value::Object(map) if map.len() == 1 && map.contains_key("$state") => resolve_state_ref(value, state),
    Value::Object(map) => Value::Object(
      map
        .iter()
        .map(|(key, value)| (key.clone(), resolve_state_refs(value, state)))
        .collect(),
    ),
    value => value.clone(),
  }
}

pub fn resolve_state_ref(value: &Value, state: &Value) -> Value {
  let Some(path) = value.get("$state").and_then(Value::as_str) else {
    return value.clone();
  };
  let mut current = state;
  for segment in path.split('.') {
    let Some(next) = current.get(segment) else {
      return Value::Null;
    };
    current = next;
  }
  current.clone()
}

fn merge_state_patch(state: &mut Value, patch: Value) {
  match (state, patch) {
    (Value::Object(state), Value::Object(patch)) => {
      for (key, value) in patch {
        state.insert(key, value);
      }
    }
    (state, patch) => {
      *state = patch;
    }
  }
}

fn set_state_path(state: &mut Value, path: &str, value: Value) {
  if !state.is_object() {
    *state = Value::Object(Map::new());
  }

  let mut current = state;
  let mut segments = path.split('.').peekable();
  while let Some(segment) = segments.next() {
    if segments.peek().is_none() {
      if let Value::Object(map) = current {
        map.insert(segment.to_string(), value);
      }
      return;
    }

    if let Value::Object(map) = current {
      current = map
        .entry(segment.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
      if !current.is_object() {
        *current = Value::Object(Map::new());
      }
    }
  }
}

pub fn validate_json_schema(label: &str, schema: &Value, value: &Value) -> Result<(), StepExecutionError> {
  let compiled = jsonschema::options()
    .with_draft(Draft::Draft7)
    .build(schema)
    .map_err(|error| {
      StepExecutionError::new(
        "invalid_schema",
        format!("Failed to compile action {label} schema: {error}"),
      )
    })?;

  let details = compiled
    .iter_errors(value)
    .map(|error| error.to_string())
    .collect::<Vec<_>>();
  if details.is_empty() {
    Ok(())
  } else {
    Err(StepExecutionError::new(
      "invalid_value",
      format!("Action {label} does not match JSON schema: {}", details.join("; ")),
    ))
  }
}

pub fn execute_validate_json_step(input: Option<Value>) -> Result<Value, StepExecutionError> {
  let Some(input) = input else {
    return Ok(Value::Bool(false));
  };
  if let Some(text) = input.as_str() {
    return Ok(Value::Bool(serde_json::from_str::<Value>(text).is_ok()));
  }

  let schema = input.get("schema");
  let value = input.get("value");
  if let (Some(schema), Some(value)) = (schema, value) {
    validate_json_schema("step output", schema, value).map_or_else(
      |error| match error.code.as_str() {
        "invalid_schema" => Err(error),
        _ => Ok(Value::Bool(false)),
      },
      |()| Ok(Value::Bool(true)),
    )
  } else {
    Ok(Value::Bool(false))
  }
}

pub fn execute_transform_step(input: Option<Value>, state: &Value) -> Result<Option<Value>, StepExecutionError> {
  let Some(input) = input else {
    return Ok(Some(state.clone()));
  };

  if let Some(parse_json) = input.get("parseJson") {
    let text = resolve_state_ref(parse_json, state);
    let Some(text) = text.as_str() else {
      return Ok(Some(Value::Null));
    };
    return serde_json::from_str::<Value>(text)
      .map(Some)
      .map_err(|error| StepExecutionError::new("invalid_step", format!("transform parseJson failed: {error}")));
  }
  if let Some(stringify) = input.get("stringify") {
    let value = resolve_state_ref(stringify, state);
    return serde_json::to_string(&value)
      .map(Value::String)
      .map(Some)
      .map_err(|error| StepExecutionError::new("invalid_step", format!("transform stringify failed: {error}")));
  }
  if let Some(copy) = input.get("copy") {
    return Ok(Some(resolve_state_ref(copy, state)));
  }
  if let Some(merge) = input.get("merge").and_then(Value::as_array) {
    let mut merged = Value::Object(Map::new());
    for value in merge {
      merge_state_patch(&mut merged, resolve_state_ref(value, state));
    }
    return Ok(Some(merged));
  }

  Ok(None)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{
    RecipeDefinition, RecipeRuntimeStatus, RecipeStepExecution, RecipeStepExecutor, StepExecutionError,
    execute_transform_step, execute_validate_json_step, merge_state_patch, resolve_state_refs, run_recipe_runtime,
    set_state_path, validate_json_schema,
  };

  struct EchoExecutor;

  impl RecipeStepExecutor for EchoExecutor {
    fn execute_step(
      &mut self,
      _step: &RecipeStepExecution,
      input: Option<serde_json::Value>,
      _state: &serde_json::Value,
    ) -> Result<serde_json::Value, StepExecutionError> {
      Ok(input.unwrap_or(serde_json::Value::Null))
    }
  }

  #[test]
  fn resolves_state_refs_recursively() {
    let state = json!({ "summary": { "title": "Weekly" }, "count": 2 });
    let value = json!({
      "title": { "$state": "summary.title" },
      "items": [{ "$state": "count" }, { "$state": "missing" }]
    });

    assert_eq!(
      resolve_state_refs(&value, &state),
      json!({ "title": "Weekly", "items": [2, null] })
    );
  }

  #[test]
  fn applies_state_patch_and_nested_output_path() {
    let mut state = json!({ "a": 1 });
    merge_state_patch(&mut state, json!({ "b": 2 }));
    set_state_path(&mut state, "nested.value", json!("ok"));

    assert_eq!(state, json!({ "a": 1, "b": 2, "nested": { "value": "ok" } }));
  }

  #[test]
  fn validates_draft7_json_schema() {
    let schema = json!({
      "type": "object",
      "required": ["title"],
      "properties": { "title": { "type": "string" } }
    });

    assert!(validate_json_schema("output", &schema, &json!({ "title": "ok" })).is_ok());
    assert!(validate_json_schema("output", &schema, &json!({ "title": 1 })).is_err());
  }

  #[test]
  fn runs_validate_json_primitive() {
    assert_eq!(
      execute_validate_json_step(Some(json!("{\"ok\":true}"))).unwrap(),
      json!(true)
    );
    assert_eq!(execute_validate_json_step(Some(json!("{"))).unwrap(), json!(false));
    assert_eq!(
      execute_validate_json_step(Some(json!({
        "schema": { "type": "object", "required": ["ok"] },
        "value": {}
      })))
      .unwrap(),
      json!(false)
    );
  }

  #[test]
  fn runs_transform_primitives() {
    let state = json!({ "raw": "{\"ok\":true}", "left": { "a": 1 }, "right": { "b": 2 } });

    assert_eq!(
      execute_transform_step(Some(json!({ "parseJson": { "$state": "raw" } })), &state).unwrap(),
      Some(json!({ "ok": true }))
    );
    assert_eq!(
      execute_transform_step(
        Some(json!({ "merge": [{ "$state": "left" }, { "$state": "right" }] })),
        &state
      )
      .unwrap(),
      Some(json!({ "a": 1, "b": 2 }))
    );
    assert_eq!(
      execute_transform_step(Some(json!({ "custom": true })), &state).unwrap(),
      None
    );
  }

  #[test]
  fn runs_generic_recipe_runtime_with_builtin_steps() {
    let recipe = RecipeDefinition {
      id: "recipe".to_string(),
      version: "1".to_string(),
      steps: vec![
        RecipeStepExecution {
          id: "parse".to_string(),
          kind: "transform".to_string(),
          input: Some(json!({
            "parseJson": { "$state": "raw" },
            "outputKey": "parsed"
          })),
          state_patch: None,
        },
        RecipeStepExecution {
          id: "final".to_string(),
          kind: "final".to_string(),
          input: Some(json!({ "$state": "parsed" })),
          state_patch: None,
        },
      ],
    };
    let mut executor = EchoExecutor;
    let mut events = Vec::new();

    let output = run_recipe_runtime(
      recipe,
      json!({ "raw": "{\"ok\":true}" }),
      &mut executor,
      |event| events.push(event.event_type.clone()),
      || false,
    );

    assert_eq!(output.status, RecipeRuntimeStatus::Succeeded);
    assert_eq!(output.result, json!({ "ok": true }));
    assert_eq!(output.state["parsed"], json!({ "ok": true }));
    assert_eq!(events.first().map(String::as_str), Some("recipe_start"));
    assert_eq!(events.last().map(String::as_str), Some("recipe_done"));
  }

  #[test]
  fn reports_generic_recipe_step_failure() {
    struct FailingExecutor;
    impl RecipeStepExecutor for FailingExecutor {
      fn execute_step(
        &mut self,
        _step: &RecipeStepExecution,
        _input: Option<serde_json::Value>,
        _state: &serde_json::Value,
      ) -> Result<serde_json::Value, StepExecutionError> {
        Err(StepExecutionError::new("bad_step", "failed"))
      }
    }

    let recipe = RecipeDefinition {
      id: "recipe".to_string(),
      version: "1".to_string(),
      steps: vec![RecipeStepExecution {
        id: "product".to_string(),
        kind: "productStep".to_string(),
        input: None,
        state_patch: None,
      }],
    };
    let mut executor = FailingExecutor;

    let output = run_recipe_runtime(recipe, json!({}), &mut executor, |_| {}, || false);

    assert_eq!(output.status, RecipeRuntimeStatus::Failed);
    assert_eq!(output.trace.error_code.as_deref(), Some("bad_step"));
    assert_eq!(output.steps[0].error.as_ref().unwrap().message, "failed");
  }

  #[test]
  fn delegates_unknown_transform_to_executor() {
    struct CustomTransformExecutor;
    impl RecipeStepExecutor for CustomTransformExecutor {
      fn execute_step(
        &mut self,
        step: &RecipeStepExecution,
        _input: Option<serde_json::Value>,
        _state: &serde_json::Value,
      ) -> Result<serde_json::Value, StepExecutionError> {
        assert_eq!(step.kind, "transform");
        Ok(json!("custom-output"))
      }
    }

    let recipe = RecipeDefinition {
      id: "recipe".to_string(),
      version: "1".to_string(),
      steps: vec![
        RecipeStepExecution {
          id: "custom".to_string(),
          kind: "transform".to_string(),
          input: Some(json!({ "custom": true, "outputKey": "value" })),
          state_patch: None,
        },
        RecipeStepExecution {
          id: "final".to_string(),
          kind: "final".to_string(),
          input: Some(json!({ "$state": "value" })),
          state_patch: None,
        },
      ],
    };
    let mut executor = CustomTransformExecutor;

    let output = run_recipe_runtime(recipe, json!({}), &mut executor, |_| {}, || false);

    assert_eq!(output.status, RecipeRuntimeStatus::Succeeded);
    assert_eq!(output.result, json!("custom-output"));
  }

  #[test]
  fn final_step_uses_transform_primitives() {
    let recipe = RecipeDefinition {
      id: "recipe".to_string(),
      version: "1".to_string(),
      steps: vec![RecipeStepExecution {
        id: "final".to_string(),
        kind: "final".to_string(),
        input: Some(json!({ "copy": { "$state": "value" } })),
        state_patch: None,
      }],
    };
    let mut executor = EchoExecutor;

    let output = run_recipe_runtime(recipe, json!({ "value": "ok" }), &mut executor, |_| {}, || false);

    assert_eq!(output.status, RecipeRuntimeStatus::Succeeded);
    assert_eq!(output.result, json!("ok"));
  }
}
