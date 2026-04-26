use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
  #[error("missing response field `{0}`")]
  MissingResponseField(&'static str),
  #[error("invalid request field `{field}`: {message}")]
  InvalidRequest { field: &'static str, message: String },
  #[error("invalid response field `{field}`: {message}")]
  InvalidResponse { field: &'static str, message: String },
  #[error("json error: {0}")]
  Json(#[from] serde_json::Error),
}
