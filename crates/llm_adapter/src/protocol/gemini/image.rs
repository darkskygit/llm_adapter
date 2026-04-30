use base64::{Engine as _, engine::general_purpose::STANDARD};
#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
  backend::{BackendRequestLayer, HttpBody},
  core::{CoreContent, CoreMessage, CoreRequest, CoreRole, ImageArtifact, ImageInput, ImageRequest, ImageResponse},
  protocol::{ProtocolError, gemini, get_u32, infer_media_type_from_url},
};

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GeminiImageOptions {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub response_modalities: Option<Vec<String>>,
}

pub fn encode(
  request: &ImageRequest,
  request_layer: BackendRequestLayer,
  base_url: &str,
) -> Result<HttpBody, ProtocolError> {
  let mut content = vec![CoreContent::Text {
    text: request.prompt().to_string(),
  }];
  for image in request.images() {
    content.push(CoreContent::Image {
      source: image_to_source(image),
    });
  }
  let mut core = CoreRequest {
    model: request.model().to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content,
    }],
    stream: false,
    max_tokens: None,
    temperature: None,
    tools: Vec::new(),
    tool_choice: None,
    include: None,
    reasoning: None,
    response_schema: None,
  };
  core.response_schema = None;
  let mut payload = gemini::request::encode(&core, false, request_layer, base_url);
  if let Value::Object(object) = &mut payload {
    let modalities = request
      .provider_options()
      .gemini()
      .and_then(|extension| extension.response_modalities.clone())
      .unwrap_or_else(|| vec!["TEXT".to_string(), "IMAGE".to_string()]);
    let generation_config = object
      .entry("generationConfig")
      .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(config) = generation_config {
      config.insert(
        "responseModalities".to_string(),
        Value::Array(modalities.into_iter().map(Value::String).collect()),
      );
    }
  }
  Ok(HttpBody::Json(payload))
}

fn image_to_source(image: &ImageInput) -> Value {
  match image {
    ImageInput::Url { url, media_type } => json!({
      "url": url,
      "media_type": media_type.clone().unwrap_or_else(|| infer_media_type_from_url(url).to_string()),
    }),
    ImageInput::Data {
      data_base64,
      media_type,
      file_name,
    } => json!({
      "data": data_base64,
      "media_type": media_type,
      "file_name": file_name,
    }),
    ImageInput::Bytes {
      data,
      media_type,
      file_name,
    } => json!({
      "data": STANDARD.encode(data),
      "media_type": media_type,
      "file_name": file_name,
    }),
  }
}

pub fn decode(body: &Value) -> Result<ImageResponse, ProtocolError> {
  let candidates = body
    .get("candidates")
    .and_then(Value::as_array)
    .ok_or(ProtocolError::MissingResponseField("gemini.candidates"))?;
  let mut images = Vec::new();
  let mut text = String::new();
  for candidate in candidates {
    let parts = candidate
      .get("content")
      .and_then(|content| content.get("parts"))
      .and_then(Value::as_array)
      .into_iter()
      .flatten();
    for part in parts {
      if let Some(delta) = part.get("text").and_then(Value::as_str) {
        text.push_str(delta);
      }
      if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data"))
        && let Some(data) = inline.get("data").and_then(Value::as_str)
      {
        images.push(ImageArtifact {
          url: None,
          data_base64: Some(data.to_string()),
          media_type: inline
            .get("mimeType")
            .or_else(|| inline.get("mime_type"))
            .and_then(Value::as_str)
            .unwrap_or("image/png")
            .to_string(),
          width: None,
          height: None,
          provider_metadata: part.clone(),
        });
      }
      if let Some(file) = part.get("fileData").or_else(|| part.get("file_data"))
        && let Some(url) = file
          .get("fileUri")
          .or_else(|| file.get("file_uri"))
          .and_then(Value::as_str)
      {
        images.push(ImageArtifact {
          url: Some(url.to_string()),
          data_base64: None,
          media_type: file
            .get("mimeType")
            .or_else(|| file.get("mime_type"))
            .and_then(Value::as_str)
            .unwrap_or("image/png")
            .to_string(),
          width: get_u32(file, "width"),
          height: get_u32(file, "height"),
          provider_metadata: part.clone(),
        });
      }
    }
  }
  Ok(ImageResponse {
    images,
    text: (!text.is_empty()).then_some(text),
    usage: None,
    provider_metadata: body.clone(),
  })
}
