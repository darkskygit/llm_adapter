use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
  #[error("missing field `{0}`")]
  MissingField(&'static str),
  #[error("invalid value for field `{field}`: {message}")]
  InvalidValue { field: &'static str, message: String },
  #[error("json error: {0}")]
  Json(#[from] serde_json::Error),
}
