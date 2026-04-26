pub mod options;

use serde_json::{Map, Value, json};

use self::options::FalImageOutputFormat;
use crate::{
  backend::BackendError,
  core::{ImageArtifact, ImageFormat, ImageInput, ImageRequest, ImageResponse},
  protocol::{ProtocolError, get_u32},
};

pub fn encode(request: &ImageRequest) -> Result<Value, ProtocolError> {
  if request.model().starts_with("workflows/") {
    return Err(ProtocolError::InvalidRequest {
      field: "model",
      message: "Fal workflows are not supported by native image dispatch".to_string(),
    });
  }

  let mut payload = Map::new();
  if !request.prompt().trim().is_empty() {
    payload.insert("prompt".to_string(), json!(request.prompt()));
  }
  if let Some(first) = request.images().first() {
    let url = image_url(first)?;
    payload.insert("image_url".to_string(), json!(url));
  }
  if let Some(seed) = request.options().seed {
    payload.insert("seed".to_string(), json!(seed));
  }
  let fal = request.provider_options().fal();
  if let Some(image_size) = fal.and_then(|extension| extension.image_size.as_ref()) {
    payload.insert("image_size".to_string(), serde_json::to_value(image_size)?);
  } else if let Some(size) = &request.options().size {
    payload.insert("image_size".to_string(), json!(size));
  }
  if let Some(aspect_ratio) = fal
    .and_then(|extension| extension.aspect_ratio.clone())
    .or_else(|| request.options().aspect_ratio.clone())
  {
    payload.insert("aspect_ratio".to_string(), json!(aspect_ratio));
  }
  if let Some(num_images) = fal.and_then(|extension| extension.num_images).or(request.options().n) {
    payload.insert("num_images".to_string(), json!(num_images));
  }
  if let Some(output_format) = fal
    .and_then(|extension| extension.output_format.map(FalImageOutputFormat::as_str))
    .or_else(|| request.options().output_format.map(image_format_to_fal_output_format))
  {
    payload.insert("output_format".to_string(), json!(output_format));
  }
  if let Some(enable_prompt_expansion) = fal.and_then(|extension| extension.enable_prompt_expansion) {
    payload.insert("enable_prompt_expansion".to_string(), json!(enable_prompt_expansion));
  }
  if let Some(model_name) = fal.and_then(|extension| extension.model_name.clone()) {
    payload.insert("model_name".to_string(), json!(model_name));
  }
  if let Some(loras) = fal.and_then(|extension| extension.loras.clone()) {
    payload.insert("loras".to_string(), loras);
  }
  if let Some(controlnets) = fal.and_then(|extension| extension.controlnets.clone()) {
    payload.insert("controlnets".to_string(), controlnets);
  }
  if let Some(extension) = fal
    && let Some(extra) = extension.extra.as_ref().and_then(Value::as_object)
  {
    for (key, value) in extra {
      payload.insert(key.clone(), value.clone());
    }
  }
  payload.insert(
    "sync_mode".to_string(),
    Value::Bool(fal.and_then(|extension| extension.sync_mode).unwrap_or(true)),
  );
  payload.insert(
    "enable_safety_checker".to_string(),
    Value::Bool(
      fal
        .and_then(|extension| extension.enable_safety_checker)
        .unwrap_or(true),
    ),
  );

  Ok(Value::Object(payload))
}

fn image_format_to_fal_output_format(format: ImageFormat) -> &'static str {
  match format {
    ImageFormat::Png => "png",
    ImageFormat::Jpeg => "jpeg",
    ImageFormat::Webp => "webp",
  }
}

fn image_url(input: &ImageInput) -> Result<String, ProtocolError> {
  match input {
    ImageInput::Url { url, .. } => Ok(url.clone()),
    ImageInput::Data { .. } | ImageInput::Bytes { .. } => Err(ProtocolError::InvalidRequest {
      field: "images",
      message: "Fal image dispatch requires URL image inputs".to_string(),
    }),
  }
}

pub fn decode(body: &Value) -> Result<ImageResponse, BackendError> {
  let mut images = Vec::new();
  push_image(body.get("image"), &mut images);
  if let Some(items) = body.get("images").and_then(Value::as_array) {
    for item in items {
      push_image(Some(item), &mut images);
    }
  }

  if images.is_empty() {
    return Err(BackendError::InvalidResponse {
      field: "images",
      message: "expected at least one image".to_string(),
    });
  }

  Ok(ImageResponse {
    images,
    text: body.get("output").and_then(Value::as_str).map(ToString::to_string),
    usage: None,
    provider_metadata: body.clone(),
  })
}

fn push_image(value: Option<&Value>, images: &mut Vec<ImageArtifact>) {
  let Some(value) = value else {
    return;
  };
  let Some(url) = value.get("url").and_then(Value::as_str) else {
    return;
  };
  images.push(ImageArtifact {
    url: Some(url.to_string()),
    data_base64: None,
    media_type: value
      .get("content_type")
      .and_then(Value::as_str)
      .unwrap_or("image/png")
      .to_string(),
    width: get_u32(value, "width"),
    height: get_u32(value, "height"),
    provider_metadata: value.clone(),
  });
}
