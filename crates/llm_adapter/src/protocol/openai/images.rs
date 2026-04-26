use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
  backend::{BackendError, HttpBody, MultipartPart},
  core::{ImageArtifact, ImageFormat, ImageInput, ImageRequest, ImageResponse, ImageUsage},
  protocol::{ProtocolError, get_u32},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OpenAiImageOptions {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub input_fidelity: Option<String>,
}

pub fn encode(request: &ImageRequest) -> Result<HttpBody, ProtocolError> {
  match request {
    ImageRequest::Generate(_) => Ok(HttpBody::Json(encode_generate(request))),
    ImageRequest::Edit(_) => encode_edit(request),
  }
}

fn insert_option<T: Into<Value>>(payload: &mut Map<String, Value>, key: &str, value: Option<T>) {
  if let Some(value) = value {
    payload.insert(key.to_string(), value.into());
  }
}

fn encode_generate(request: &ImageRequest) -> Value {
  let mut payload = Map::from_iter([
    ("model".to_string(), json!(request.model())),
    ("prompt".to_string(), json!(request.prompt())),
  ]);
  insert_common_options(&mut payload, request);
  Value::Object(payload)
}

fn encode_edit(request: &ImageRequest) -> Result<HttpBody, ProtocolError> {
  let mut parts = vec![
    MultipartPart::Text {
      name: "model".to_string(),
      value: request.model().to_string(),
    },
    MultipartPart::Text {
      name: "prompt".to_string(),
      value: request.prompt().to_string(),
    },
  ];

  append_common_text_options(&mut parts, request);
  for (index, image) in request.images().iter().enumerate() {
    parts.push(input_to_part("image[]", image, index)?);
  }
  if let Some(mask) = request.mask() {
    parts.push(input_to_part("mask", mask, 0)?);
  }
  Ok(HttpBody::Multipart(parts))
}

fn insert_common_options(payload: &mut Map<String, Value>, request: &ImageRequest) {
  insert_option(payload, "n", request.options().n.map(Value::from));
  insert_option(payload, "size", request.options().size.clone().map(Value::from));
  insert_option(payload, "quality", request.options().quality.clone().map(Value::from));
  insert_option(
    payload,
    "output_format",
    request
      .options()
      .output_format
      .map(|format| Value::String(format.as_str().to_string())),
  );
  insert_option(
    payload,
    "output_compression",
    request.options().output_compression.map(Value::from),
  );
  insert_option(
    payload,
    "background",
    request.options().background.clone().map(Value::from),
  );
  if let Some(input_fidelity) = request
    .provider_options()
    .openai()
    .and_then(|extension| extension.input_fidelity.clone())
  {
    payload.insert("input_fidelity".to_string(), Value::String(input_fidelity));
  }
}

fn append_common_text_options(parts: &mut Vec<MultipartPart>, request: &ImageRequest) {
  let mut payload = Map::new();
  insert_common_options(&mut payload, request);
  for (name, value) in payload {
    if let Some(value) = json_scalar_to_string(value) {
      parts.push(MultipartPart::Text { name, value });
    }
  }
}

fn json_scalar_to_string(value: Value) -> Option<String> {
  match value {
    Value::String(value) => Some(value),
    Value::Number(value) => Some(value.to_string()),
    Value::Bool(value) => Some(value.to_string()),
    _ => None,
  }
}

fn input_to_part(name: &str, input: &ImageInput, index: usize) -> Result<MultipartPart, ProtocolError> {
  match input {
    ImageInput::Bytes {
      data,
      media_type,
      file_name,
    } => Ok(MultipartPart::File {
      name: name.to_string(),
      file_name: file_name
        .clone()
        .unwrap_or_else(|| format!("{index}.{}", extension(media_type))),
      media_type: media_type.clone(),
      bytes: data.clone(),
    }),
    ImageInput::Data {
      data_base64,
      media_type,
      file_name,
    } => {
      let bytes = STANDARD
        .decode(data_base64)
        .map_err(|_| ProtocolError::InvalidRequest {
          field: "images",
          message: "image data_base64 must be valid base64".to_string(),
        })?;
      Ok(MultipartPart::File {
        name: name.to_string(),
        file_name: file_name
          .clone()
          .unwrap_or_else(|| format!("{index}.{}", extension(media_type))),
        media_type: media_type.clone(),
        bytes,
      })
    }
    ImageInput::Url { .. } => Err(ProtocolError::InvalidRequest {
      field: "images",
      message: "OpenAI image edits require host-materialized image data".to_string(),
    }),
  }
}

fn extension(media_type: &str) -> &'static str {
  match media_type {
    "image/jpeg" => "jpg",
    "image/webp" => "webp",
    _ => "png",
  }
}

pub fn decode(body: &Value, request: &ImageRequest) -> Result<ImageResponse, BackendError> {
  if let Some(error) = body.get("error") {
    return Err(BackendError::UpstreamStatus {
      status: 400,
      body: error.to_string(),
    });
  }

  let media_type = request
    .options()
    .output_format
    .unwrap_or(ImageFormat::Png)
    .media_type()
    .to_string();
  let data = body
    .get("data")
    .and_then(Value::as_array)
    .ok_or(BackendError::InvalidResponse {
      field: "data",
      message: "missing field".to_string(),
    })?;
  let images = data
    .iter()
    .filter_map(|item| {
      let url = item.get("url").and_then(Value::as_str).map(ToString::to_string);
      let data_base64 = item.get("b64_json").and_then(Value::as_str).map(ToString::to_string);
      (url.is_some() || data_base64.is_some()).then(|| ImageArtifact {
        url,
        data_base64,
        media_type: media_type.clone(),
        width: None,
        height: None,
        provider_metadata: item.clone(),
      })
    })
    .collect();

  Ok(ImageResponse {
    images,
    text: None,
    usage: body.get("usage").map(|usage| ImageUsage {
      input_tokens: get_u32(usage, "input_tokens").or_else(|| get_u32(usage, "prompt_tokens")),
      output_tokens: get_u32(usage, "output_tokens").or_else(|| get_u32(usage, "completion_tokens")),
      total_tokens: get_u32(usage, "total_tokens"),
    }),
    provider_metadata: body.clone(),
  })
}
