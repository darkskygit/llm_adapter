use serde_json::{Value, json};

use super::ProtocolError;
use crate::core::{EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};

pub fn encode(request: &EmbeddingRequest) -> Value {
  let mut payload = serde_json::Map::from_iter([
    ("model".to_string(), Value::String(request.model.clone())),
    (
      "input".to_string(),
      Value::Array(request.inputs.iter().cloned().map(Value::String).collect()),
    ),
    ("encoding_format".to_string(), Value::String("float".to_string())),
  ]);

  if let Some(dimensions) = request.dimensions {
    payload.insert("dimensions".to_string(), json!(dimensions));
  }

  Value::Object(payload)
}

pub fn decode(body: &Value) -> Result<EmbeddingResponse, ProtocolError> {
  let model = body
    .get("model")
    .and_then(Value::as_str)
    .ok_or(ProtocolError::MissingField("model"))?
    .to_string();
  let data = body
    .get("data")
    .and_then(Value::as_array)
    .ok_or(ProtocolError::MissingField("data"))?;

  let embeddings = data
    .iter()
    .map(|item| {
      let embedding = item
        .get("embedding")
        .and_then(Value::as_array)
        .ok_or(ProtocolError::MissingField("data[].embedding"))?;
      embedding
        .iter()
        .map(|value| {
          value.as_f64().ok_or(ProtocolError::InvalidValue {
            field: "data[].embedding[]",
            message: "expected number".to_string(),
          })
        })
        .collect()
    })
    .collect::<Result<Vec<Vec<f64>>, ProtocolError>>()?;

  let usage = body.get("usage").map(|usage| EmbeddingUsage {
    prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_default() as u32,
    total_tokens: usage.get("total_tokens").and_then(Value::as_u64).unwrap_or_default() as u32,
  });

  Ok(EmbeddingResponse {
    model,
    embeddings,
    usage,
  })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn encode_should_match_backend_contract() {
    let payload = encode(&EmbeddingRequest {
      model: "text-embedding-3-small".to_string(),
      inputs: vec!["hello".to_string(), "world".to_string()],
      dimensions: Some(256),
      task_type: None,
    });

    assert_eq!(payload["model"], "text-embedding-3-small");
    assert_eq!(payload["input"], json!(["hello", "world"]));
    assert_eq!(payload["dimensions"], 256);
    assert_eq!(payload["encoding_format"], "float");
  }

  #[test]
  fn decode_should_match_backend_contract() {
    let response = decode(&json!({
      "model": "text-embedding-3-small",
      "data": [
        { "embedding": [0.1, 0.2] },
        { "embedding": [0.3, 0.4] }
      ],
      "usage": {
        "prompt_tokens": 8,
        "total_tokens": 8
      }
    }))
    .unwrap();

    assert_eq!(response.model, "text-embedding-3-small");
    assert_eq!(response.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    assert_eq!(
      response.usage,
      Some(EmbeddingUsage {
        prompt_tokens: 8,
        total_tokens: 8,
      })
    );
  }
}
