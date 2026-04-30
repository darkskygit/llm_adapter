use jsonschema::Draft;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaValidationError {
  #[error("Failed to compile JSON schema: {0}")]
  Compile(String),
  #[error("Structured output does not match JSON schema: {0}")]
  Validate(String),
}

pub fn validate_json_schema(schema: &Value, value: &Value) -> Result<(), SchemaValidationError> {
  let compiled = jsonschema::options()
    .with_draft(Draft::Draft7)
    .build(schema)
    .map_err(|error| SchemaValidationError::Compile(error.to_string()))?;
  let details = compiled
    .iter_errors(value)
    .map(|error| error.to_string())
    .collect::<Vec<_>>();

  if details.is_empty() {
    Ok(())
  } else {
    Err(SchemaValidationError::Validate(details.join("; ")))
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{SchemaValidationError, validate_json_schema};

  #[test]
  fn validates_draft7_schema() {
    let schema = json!({
      "type": "object",
      "required": ["title"],
      "properties": { "title": { "type": "string" } }
    });

    assert!(validate_json_schema(&schema, &json!({ "title": "ok" })).is_ok());
    assert!(matches!(
      validate_json_schema(&schema, &json!({ "title": 1 })),
      Err(SchemaValidationError::Validate(_))
    ));
  }
}
