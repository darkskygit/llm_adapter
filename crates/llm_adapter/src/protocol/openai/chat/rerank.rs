use std::collections::BTreeSet;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
  CoreContent, CoreMessage, CoreRole, OpenaiRequestFlavor, ProtocolError, RerankRequest, get_str, messages_from_core,
};

const DEFAULT_TOP_LOGPROBS: u32 = 5;
const DEFAULT_MAX_TOKENS: u32 = 16;
const DEFAULT_SYSTEM_PROMPT: &str = r#"Judge whether the Document meets the requirements based on the Query and the Instruct provided. The answer must be "yes" or "no"."#;
const DEFAULT_USER_TEMPLATE: &str = "<Instruct>: Given a document search result, determine whether the result is \
                                     relevant to the query.\n<Query>: {query}\n<Document>: {document}";
const TOKEN_PREFIXES: [&str; 12] = ["_", ":", ".", " ", "\"", "-", "\t", ",", "(", "=", "_", "“"];

#[derive(Debug, Deserialize)]
struct ChatCompletionRerankResponse {
  model: String,
  #[serde(default)]
  choices: Vec<ChatCompletionRerankChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRerankChoice {
  logprobs: Option<ChatCompletionRerankLogprobs>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRerankLogprobs {
  #[serde(default)]
  content: Vec<ChatCompletionRerankContent>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRerankContent {
  #[serde(default)]
  top_logprobs: Vec<ChatCompletionRerankTopLogprob>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRerankTopLogprob {
  token: String,
  logprob: f64,
}

#[must_use]
fn default_positive_tokens() -> Vec<String> {
  vec![
    "Yes".to_string(),
    "Relevant".to_string(),
    "Related".to_string(),
    "True".to_string(),
    "Correct".to_string(),
  ]
}

#[must_use]
fn default_negative_tokens() -> Vec<String> {
  vec![
    "No".to_string(),
    "Irrelevant".to_string(),
    "Unrelated".to_string(),
    "False".to_string(),
    "Incorrect".to_string(),
  ]
}

fn default_reasoning(model: &str) -> Option<Value> {
  if model.starts_with("gpt-5.2") {
    Some(json!({ "effort": "none" }))
  } else {
    None
  }
}

fn build_default_messages(request: &RerankRequest, candidate_index: usize) -> Result<Vec<CoreMessage>, ProtocolError> {
  let candidate = request
    .candidates
    .get(candidate_index)
    .ok_or(ProtocolError::InvalidRequest {
      field: "candidates",
      message: format!("candidate index {candidate_index} is out of bounds"),
    })?;

  Ok(vec![
    CoreMessage {
      role: CoreRole::System,
      content: vec![CoreContent::Text {
        text: DEFAULT_SYSTEM_PROMPT.to_string(),
      }],
    },
    CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: DEFAULT_USER_TEMPLATE
          .replace("{query}", &request.query)
          .replace("{document}", &candidate.text),
      }],
    },
  ])
}

pub fn encode(request: &RerankRequest, candidate_index: usize) -> Result<Value, ProtocolError> {
  let batch = build_default_messages(request, candidate_index)?;
  let mut payload = Map::from_iter([
    ("model".to_string(), Value::String(request.model.clone())),
    (
      "messages".to_string(),
      Value::Array(messages_from_core(&batch, OpenaiRequestFlavor::ChatCompletions)),
    ),
    ("stream".to_string(), Value::Bool(false)),
    ("temperature".to_string(), json!(0.0)),
    ("logprobs".to_string(), Value::Bool(true)),
    ("top_logprobs".to_string(), json!(DEFAULT_TOP_LOGPROBS)),
  ]);

  let reasoning = default_reasoning(&request.model);
  if let Some(reasoning) = &reasoning {
    if let Some(effort) = get_str(reasoning, "effort") {
      payload.insert("reasoning_effort".to_string(), json!(effort));
    } else {
      payload.insert("reasoning".to_string(), reasoning.clone());
    }
    payload.insert("max_completion_tokens".to_string(), json!(DEFAULT_MAX_TOKENS));
  } else {
    payload.insert("max_tokens".to_string(), json!(DEFAULT_MAX_TOKENS));
  }

  Ok(Value::Object(payload))
}

pub fn decode(body: &Value, _request: &RerankRequest) -> Result<(String, f64), ProtocolError> {
  let response: ChatCompletionRerankResponse = serde_json::from_value(body.clone())?;
  let choice = response
    .choices
    .first()
    .ok_or(ProtocolError::MissingResponseField("openai_chat.choices[0]"))?;

  let Some(logprobs) = &choice.logprobs else {
    return Ok((response.model, 0.0));
  };
  let Some(content) = logprobs.content.first() else {
    return Ok((response.model, 0.0));
  };

  let positive = best_logprob(&content.top_logprobs, &default_positive_tokens());
  let negative = best_logprob(&content.top_logprobs, &default_negative_tokens());
  let p_positive = positive.exp();
  let p_negative = negative.exp();
  let score = if (p_positive + p_negative) == 0.0 {
    0.0
  } else {
    p_positive / (p_positive + p_negative)
  };

  Ok((response.model, score))
}

fn best_logprob(top_logprobs: &[ChatCompletionRerankTopLogprob], tokens: &[String]) -> f64 {
  let variants = tokens
    .iter()
    .flat_map(|token| token_variants(token))
    .collect::<BTreeSet<_>>();

  variants.iter().fold(f64::NEG_INFINITY, |best, candidate| {
    let candidate_best = top_logprobs
      .iter()
      .filter(|entry| entry.token == *candidate)
      .map(|entry| entry.logprob)
      .fold(f64::NEG_INFINITY, f64::max);
    best.max(candidate_best)
  })
}

fn token_variants(token: &str) -> BTreeSet<String> {
  let mut variants = BTreeSet::from([token.to_string(), token.to_lowercase(), token.to_uppercase()]);
  for prefix in TOKEN_PREFIXES {
    let prefixed = format!("{prefix}{token}");
    variants.insert(prefixed.clone());
    variants.insert(prefixed.to_lowercase());
    variants.insert(prefixed.to_uppercase());
  }
  variants
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  fn sample_request() -> RerankRequest {
    RerankRequest {
      model: "gpt-5.2".to_string(),
      query: "Is this relevant?".to_string(),
      candidates: vec![crate::core::RerankCandidate {
        id: Some("doc-1".to_string()),
        text: "Rust is useful.".to_string(),
      }],
      top_n: None,
    }
  }

  #[test]
  fn encode_should_use_chat_completions_logprobs_contract() {
    let request = sample_request();
    let payload = encode(&request, 0).unwrap();

    assert_eq!(payload["model"], "gpt-5.2");
    assert_eq!(payload["stream"], false);
    assert_eq!(payload["temperature"], 0.0);
    assert_eq!(payload["logprobs"], true);
    assert_eq!(payload["top_logprobs"], 5);
    assert_eq!(payload["reasoning_effort"], "none");
    assert_eq!(payload["max_completion_tokens"], 16);
    assert!(payload.get("max_tokens").is_none());
    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["messages"][1]["role"], "user");
  }

  #[test]
  fn encode_should_fallback_to_default_messages_when_batches_are_missing() {
    let request = RerankRequest {
      model: "gpt-4.1".to_string(),
      query: "javascript".to_string(),
      candidates: vec![crate::core::RerankCandidate {
        id: None,
        text: "React is useful.".to_string(),
      }],
      top_n: None,
    };

    let payload = encode(&request, 0).unwrap();

    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["messages"][1]["role"], "user");
    assert!(
      payload["messages"][1]["content"]
        .as_str()
        .unwrap()
        .contains("javascript")
    );
    assert!(
      payload["messages"][1]["content"]
        .as_str()
        .unwrap()
        .contains("React is useful.")
    );
  }

  #[test]
  fn decode_should_score_prefixed_yes_no_tokens() {
    let request = sample_request();
    let (model, score) = decode(
      &json!({
        "model": "gpt-5.2",
        "choices": [{
          "logprobs": {
            "content": [{
              "top_logprobs": [
                { "token": " Yes", "logprob": -0.1 },
                { "token": "No", "logprob": -2.0 }
              ]
            }]
          }
        }]
      }),
      &request,
    )
    .unwrap();

    assert_eq!(model, "gpt-5.2");
    assert!(score > 0.8);
  }

  #[test]
  fn decode_should_fallback_to_zero_when_logprobs_are_missing() {
    let request = sample_request();
    let (_, score) = decode(
      &json!({
        "model": "gpt-5.2",
        "choices": [{}]
      }),
      &request,
    )
    .unwrap();

    assert_eq!(score, 0.0);
  }
}
