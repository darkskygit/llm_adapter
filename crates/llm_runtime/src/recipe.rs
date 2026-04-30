use jsonschema::Draft;
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct RecipeStepExecution {
  pub id: String,
  pub kind: String,
  pub input: Option<Value>,
  pub state_patch: Option<Value>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
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

pub fn resolve_state_refs(value: &Value, state: &Value) -> Value {
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

pub fn merge_state_patch(state: &mut Value, patch: Value) {
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

pub fn set_state_path(state: &mut Value, path: &str, value: Value) {
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
    execute_transform_step, execute_validate_json_step, merge_state_patch, resolve_state_refs, set_state_path,
    validate_json_schema,
  };

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
}
