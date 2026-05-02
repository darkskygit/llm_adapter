use jsonschema::Draft;
use serde_json::Value;
use sha2::{Digest, Sha256};
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

pub fn canonical_json_string(value: &Value) -> String {
  let mut output = String::new();
  write_canonical_json(value, &mut output);
  output
}

pub fn canonical_json_sha256(value: &Value) -> String {
  let mut hasher = Sha256::new();
  hasher.update(canonical_json_string(value).as_bytes());
  to_hex(&hasher.finalize())
}

fn write_canonical_json(value: &Value, output: &mut String) {
  match value {
    Value::Null => output.push_str("null"),
    Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
    Value::Number(value) => output.push_str(&value.to_string()),
    Value::String(value) => {
      output.push_str(&serde_json::to_string(value).expect("string serialization should not fail"))
    }
    Value::Array(values) => {
      output.push('[');
      for (index, value) in values.iter().enumerate() {
        if index > 0 {
          output.push(',');
        }
        write_canonical_json(value, output);
      }
      output.push(']');
    }
    Value::Object(map) => {
      output.push('{');
      let mut entries = map.iter().collect::<Vec<_>>();
      entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
      for (index, (key, value)) in entries.into_iter().enumerate() {
        if index > 0 {
          output.push(',');
        }
        output.push_str(&serde_json::to_string(key).expect("object key serialization should not fail"));
        output.push(':');
        write_canonical_json(value, output);
      }
      output.push('}');
    }
  }
}

fn to_hex(bytes: &[u8]) -> String {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut output = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    output.push(HEX[(byte >> 4) as usize] as char);
    output.push(HEX[(byte & 0x0f) as usize] as char);
  }
  output
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{SchemaValidationError, canonical_json_sha256, canonical_json_string, validate_json_schema};

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

  #[test]
  fn canonicalizes_object_keys_before_hashing() {
    let left = json!({
      "type": "object",
      "properties": {
        "title": { "type": "string" },
        "count": { "type": "number" }
      },
      "required": ["title", "count"]
    });
    let right = json!({
      "required": ["title", "count"],
      "properties": {
        "count": { "type": "number" },
        "title": { "type": "string" }
      },
      "type": "object"
    });

    assert_eq!(canonical_json_string(&left), canonical_json_string(&right));
    assert_eq!(canonical_json_sha256(&left), canonical_json_sha256(&right));
  }
}
