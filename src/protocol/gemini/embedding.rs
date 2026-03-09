use serde_json::{Value, json};

use super::ProtocolError;
use crate::core::{EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};

pub fn encode(request: &EmbeddingRequest) -> Value {
  let mut payload = serde_json::Map::from_iter([
    ("model".to_string(), Value::String(request.model.clone())),
    (
      "inputs".to_string(),
      Value::Array(request.inputs.iter().cloned().map(Value::String).collect()),
    ),
  ]);

  if let Some(dimensions) = request.dimensions {
    payload.insert("dimensions".to_string(), json!(dimensions));
  }
  if let Some(task_type) = &request.task_type {
    payload.insert("task_type".to_string(), Value::String(task_type.to_string()));
  }

  Value::Object(payload)
}

pub fn decode(body: &Value) -> Result<EmbeddingResponse, ProtocolError> {
  if let Some(embeddings) = body.get("embeddings").and_then(Value::as_array) {
    return Ok(EmbeddingResponse {
      model: body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string(),
      embeddings: parse_values(embeddings, "embeddings[].values")?,
      usage: None,
    });
  }

  let predictions = body
    .get("predictions")
    .and_then(Value::as_array)
    .ok_or(ProtocolError::MissingField("predictions|embeddings"))?;
  let usage = Some(EmbeddingUsage {
    prompt_tokens: predictions
      .iter()
      .map(|prediction| {
        prediction
          .get("embeddings")
          .and_then(|value| value.get("statistics"))
          .and_then(|value| value.get("token_count"))
          .and_then(Value::as_u64)
          .unwrap_or_default() as u32
      })
      .sum(),
    total_tokens: predictions
      .iter()
      .map(|prediction| {
        prediction
          .get("embeddings")
          .and_then(|value| value.get("statistics"))
          .and_then(|value| value.get("token_count"))
          .and_then(Value::as_u64)
          .unwrap_or_default() as u32
      })
      .sum(),
  });

  Ok(EmbeddingResponse {
    model: body
      .get("model")
      .and_then(Value::as_str)
      .unwrap_or_default()
      .to_string(),
    embeddings: parse_prediction_values(predictions)?,
    usage,
  })
}

fn parse_values(items: &[Value], field: &'static str) -> Result<Vec<Vec<f64>>, ProtocolError> {
  items
    .iter()
    .map(|item| {
      let values = item
        .get("values")
        .and_then(Value::as_array)
        .ok_or(ProtocolError::MissingField(field))?;
      values
        .iter()
        .map(|value| {
          value.as_f64().ok_or(ProtocolError::InvalidValue {
            field,
            message: "expected number".to_string(),
          })
        })
        .collect()
    })
    .collect()
}

fn parse_prediction_values(predictions: &[Value]) -> Result<Vec<Vec<f64>>, ProtocolError> {
  predictions
    .iter()
    .map(|prediction| {
      let values = prediction
        .get("embeddings")
        .and_then(|value| value.get("values"))
        .and_then(Value::as_array)
        .ok_or(ProtocolError::MissingField("predictions[].embeddings.values"))?;
      values
        .iter()
        .map(|value| {
          value.as_f64().ok_or(ProtocolError::InvalidValue {
            field: "predictions[].embeddings.values[]",
            message: "expected number".to_string(),
          })
        })
        .collect()
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn encode_should_match_backend_contract() {
    let payload = encode(&EmbeddingRequest {
      model: "gemini-embedding-001".to_string(),
      inputs: vec!["hello".to_string(), "world".to_string()],
      dimensions: Some(256),
      task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
    });

    assert_eq!(payload["model"], "gemini-embedding-001");
    assert_eq!(payload["inputs"], json!(["hello", "world"]));
    assert_eq!(payload["dimensions"], 256);
    assert_eq!(payload["task_type"], "RETRIEVAL_DOCUMENT");
  }

  #[test]
  fn decode_should_parse_gemini_api_response() {
    let response = decode(&json!({
      "embeddings": [
        { "values": [0.1, 0.2] },
        { "values": [0.3, 0.4] }
      ]
    }))
    .unwrap();

    assert_eq!(response.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    assert_eq!(response.usage, None);
  }

  #[test]
  fn decode_should_parse_gemini_vertex_response() {
    let response = decode(&json!({
      "predictions": [
        {
          "embeddings": {
            "values": [0.1, 0.2],
            "statistics": { "token_count": 3 }
          }
        },
        {
          "embeddings": {
            "values": [0.3, 0.4],
            "statistics": { "token_count": 4 }
          }
        }
      ]
    }))
    .unwrap();

    assert_eq!(response.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    assert_eq!(
      response.usage,
      Some(EmbeddingUsage {
        prompt_tokens: 7,
        total_tokens: 7,
      })
    );
  }
}
