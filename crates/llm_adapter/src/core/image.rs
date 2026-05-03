#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::{
  ProtocolError, fal::options::FalImageOptions, gemini::image::GeminiImageOptions, openai::images::OpenAiImageOptions,
};

const MAX_IMAGE_PROMPT_CHARS: usize = 32_000;
const MAX_IMAGE_INPUTS: usize = 16;
const MAX_IMAGE_INPUT_BYTES: usize = 25 * 1024 * 1024;
const MAX_IMAGE_INPUT_BASE64_CHARS: usize = MAX_IMAGE_INPUT_BYTES.div_ceil(3) * 4;
const MAX_IMAGE_TOTAL_BYTES: usize = 100 * 1024 * 1024;
const MAX_IMAGE_URL_CHARS: usize = 8 * 1024;
const MAX_IMAGE_FILE_NAME_CHARS: usize = 255;
const MAX_IMAGE_MEDIA_TYPE_CHARS: usize = 127;
const MAX_IMAGE_OUTPUT_COUNT: u32 = 10;
const MAX_IMAGE_MODEL_CHARS: usize = 256;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
  Png,
  Jpeg,
  Webp,
}

impl ImageFormat {
  #[must_use]
  pub fn media_type(self) -> &'static str {
    match self {
      Self::Png => "image/png",
      Self::Jpeg => "image/jpeg",
      Self::Webp => "image/webp",
    }
  }

  #[must_use]
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Png => "png",
      Self::Jpeg => "jpeg",
      Self::Webp => "webp",
    }
  }
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageInput {
  Url {
    url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    media_type: Option<String>,
  },
  Data {
    data_base64: String,
    media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_name: Option<String>,
  },
  Bytes {
    data: Vec<u8>,
    media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_name: Option<String>,
  },
}

impl ImageInput {
  pub fn validate(&self, field: &'static str) -> Result<(), ProtocolError> {
    let invalid = |message: &str| ProtocolError::InvalidRequest {
      field,
      message: message.to_string(),
    };

    match self {
      Self::Url { url, media_type } => {
        if url.starts_with("data:") {
          let Some((data_media_type, data_base64)) = image_data_url_parts(url) else {
            return Err(invalid("invalid image data URL"));
          };
          validate_image_media_type(data_media_type, field)?;
          if data_base64.len() > MAX_IMAGE_INPUT_BASE64_CHARS {
            return Err(invalid("image data exceeds the per-image size limit"));
          }
          if decoded_base64_len(data_base64) > MAX_IMAGE_INPUT_BYTES {
            return Err(invalid("image data exceeds the per-image size limit"));
          }
        } else if url.len() > MAX_IMAGE_URL_CHARS {
          return Err(invalid("`url` is too long"));
        }
        if let Some(media_type) = media_type {
          validate_image_media_type(media_type, field)?;
        }
        Ok(())
      }
      Self::Data {
        data_base64,
        media_type,
        file_name,
      } => {
        validate_image_media_type(media_type, field)?;
        validate_image_file_name(file_name.as_deref(), field)?;
        if data_base64.len() > MAX_IMAGE_INPUT_BASE64_CHARS {
          return Err(invalid("image data exceeds the per-image size limit"));
        }
        if decoded_base64_len(data_base64) > MAX_IMAGE_INPUT_BYTES {
          return Err(invalid("image data exceeds the per-image size limit"));
        }
        Ok(())
      }
      Self::Bytes {
        data,
        media_type,
        file_name,
      } => {
        validate_image_media_type(media_type, field)?;
        validate_image_file_name(file_name.as_deref(), field)?;
        if data.len() > MAX_IMAGE_INPUT_BYTES {
          return Err(invalid("image data exceeds the per-image size limit"));
        }
        Ok(())
      }
    }
  }

  #[must_use]
  pub fn media_type(&self) -> Option<&str> {
    match self {
      Self::Url { media_type, .. } => media_type.as_deref(),
      Self::Data { media_type, .. } | Self::Bytes { media_type, .. } => Some(media_type),
    }
  }

  #[must_use]
  fn estimated_byte_len(&self) -> usize {
    match self {
      Self::Url { url, .. } => image_data_url_parts(url)
        .map(|(_, data_base64)| decoded_base64_len(data_base64))
        .unwrap_or(0),
      Self::Data { data_base64, .. } => decoded_base64_len(data_base64),
      Self::Bytes { data, .. } => data.len(),
    }
  }
}

fn image_data_url_parts(url: &str) -> Option<(&str, &str)> {
  let rest = url.strip_prefix("data:")?;
  let (metadata, data_base64) = rest.split_once(',')?;
  let media_type = metadata.strip_suffix(";base64")?;
  (!media_type.is_empty()).then_some((media_type, data_base64))
}

fn validate_image_media_type(media_type: &str, field: &'static str) -> Result<(), ProtocolError> {
  if media_type.len() > MAX_IMAGE_MEDIA_TYPE_CHARS {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "`media_type` is too long".to_string(),
    });
  }
  if !media_type.starts_with("image/") {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "`media_type` must start with `image/`".to_string(),
    });
  }
  if media_type.bytes().any(|byte| byte.is_ascii_control()) {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "`media_type` must not contain control characters".to_string(),
    });
  }
  Ok(())
}

fn validate_image_file_name(file_name: Option<&str>, field: &'static str) -> Result<(), ProtocolError> {
  let Some(file_name) = file_name else {
    return Ok(());
  };
  if file_name.is_empty() || file_name.len() > MAX_IMAGE_FILE_NAME_CHARS {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "`file_name` length is invalid".to_string(),
    });
  }
  if file_name.bytes().any(|byte| byte.is_ascii_control()) {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "`file_name` must not contain control characters".to_string(),
    });
  }
  Ok(())
}

fn decoded_base64_len(value: &str) -> usize {
  let without_padding = value.trim_end_matches('=').len();
  without_padding.saturating_mul(3) / 4
}

fn validate_image_model(model: &str) -> Result<(), ProtocolError> {
  if model.trim().is_empty() {
    return Err(ProtocolError::InvalidRequest {
      field: "model",
      message: "model must not be empty".to_string(),
    });
  }
  if model.len() > MAX_IMAGE_MODEL_CHARS {
    return Err(ProtocolError::InvalidRequest {
      field: "model",
      message: "model is too long".to_string(),
    });
  }
  if model
    .bytes()
    .any(|byte| byte.is_ascii_control() || matches!(byte, b'\\' | b'?' | b'#'))
  {
    return Err(ProtocolError::InvalidRequest {
      field: "model",
      message: "model contains invalid path characters".to_string(),
    });
  }
  if model
    .split('/')
    .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
  {
    return Err(ProtocolError::InvalidRequest {
      field: "model",
      message: "model contains invalid path segments".to_string(),
    });
  }
  Ok(())
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ImageOptions {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub n: Option<u32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub aspect_ratio: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub quality: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output_format: Option<ImageFormat>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output_compression: Option<u8>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub background: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub seed: Option<u64>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "provider", content = "options", rename_all = "snake_case")]
pub enum ImageProviderOptions {
  #[default]
  None,
  Openai(OpenAiImageOptions),
  Gemini(GeminiImageOptions),
  Fal(FalImageOptions),
  Extra(Value),
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ImageRequest {
  Generate(ImageGenerateRequest),
  Edit(ImageEditRequest),
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerateRequest {
  pub model: String,
  pub prompt: String,
  #[serde(default)]
  pub options: ImageOptions,
  #[serde(default)]
  pub provider_options: ImageProviderOptions,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageEditRequest {
  pub model: String,
  pub prompt: String,
  pub images: Vec<ImageInput>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub mask: Option<ImageInput>,
  #[serde(default)]
  pub options: ImageOptions,
  #[serde(default)]
  pub provider_options: ImageProviderOptions,
}

impl ImageProviderOptions {
  #[must_use]
  pub fn openai(&self) -> Option<&OpenAiImageOptions> {
    match self {
      Self::Openai(options) => Some(options),
      _ => None,
    }
  }

  #[must_use]
  pub fn gemini(&self) -> Option<&GeminiImageOptions> {
    match self {
      Self::Gemini(options) => Some(options),
      _ => None,
    }
  }

  #[must_use]
  pub fn fal(&self) -> Option<&FalImageOptions> {
    match self {
      Self::Fal(options) => Some(options),
      _ => None,
    }
  }
}

impl ImageRequest {
  #[must_use]
  pub fn generate(
    model: String,
    prompt: String,
    options: ImageOptions,
    provider_options: ImageProviderOptions,
  ) -> Self {
    Self::Generate(ImageGenerateRequest {
      model,
      prompt,
      options,
      provider_options,
    })
  }

  #[must_use]
  pub fn edit(
    model: String,
    prompt: String,
    images: Vec<ImageInput>,
    mask: Option<ImageInput>,
    options: ImageOptions,
    provider_options: ImageProviderOptions,
  ) -> Self {
    Self::Edit(ImageEditRequest {
      model,
      prompt,
      images,
      mask,
      options,
      provider_options,
    })
  }

  #[must_use]
  pub fn model(&self) -> &str {
    match self {
      Self::Generate(request) => &request.model,
      Self::Edit(request) => &request.model,
    }
  }

  pub fn set_model(&mut self, model: String) {
    match self {
      Self::Generate(request) => request.model = model,
      Self::Edit(request) => request.model = model,
    }
  }

  #[must_use]
  pub fn prompt(&self) -> &str {
    match self {
      Self::Generate(request) => &request.prompt,
      Self::Edit(request) => &request.prompt,
    }
  }

  #[must_use]
  pub fn options(&self) -> &ImageOptions {
    match self {
      Self::Generate(request) => &request.options,
      Self::Edit(request) => &request.options,
    }
  }

  #[must_use]
  pub fn provider_options(&self) -> &ImageProviderOptions {
    match self {
      Self::Generate(request) => &request.provider_options,
      Self::Edit(request) => &request.provider_options,
    }
  }

  #[must_use]
  pub fn images(&self) -> &[ImageInput] {
    match self {
      Self::Generate(_) => &[],
      Self::Edit(request) => &request.images,
    }
  }

  #[must_use]
  pub fn mask(&self) -> Option<&ImageInput> {
    match self {
      Self::Generate(_) => None,
      Self::Edit(request) => request.mask.as_ref(),
    }
  }

  #[must_use]
  pub fn is_edit(&self) -> bool {
    matches!(self, Self::Edit(_))
  }

  pub fn validate(&self) -> Result<(), ProtocolError> {
    validate_image_model(self.model())?;
    if self.prompt().trim().is_empty() {
      return Err(ProtocolError::InvalidRequest {
        field: "prompt",
        message: "prompt must not be empty".to_string(),
      });
    }
    if self.prompt().chars().count() > MAX_IMAGE_PROMPT_CHARS {
      return Err(ProtocolError::InvalidRequest {
        field: "prompt",
        message: "prompt is too long".to_string(),
      });
    }
    if matches!(self, Self::Edit(request) if request.images.is_empty()) {
      return Err(ProtocolError::InvalidRequest {
        field: "images",
        message: "edit requires at least one image".to_string(),
      });
    }
    if self.images().len() > MAX_IMAGE_INPUTS {
      return Err(ProtocolError::InvalidRequest {
        field: "images",
        message: "too many image inputs".to_string(),
      });
    }
    if self.options().output_compression.is_some_and(|value| value > 100) {
      return Err(ProtocolError::InvalidRequest {
        field: "output_compression",
        message: "output_compression must be between 0 and 100".to_string(),
      });
    }
    if self
      .options()
      .n
      .is_some_and(|value| value == 0 || value > MAX_IMAGE_OUTPUT_COUNT)
    {
      return Err(ProtocolError::InvalidRequest {
        field: "n",
        message: "n must be between 1 and 10".to_string(),
      });
    }
    let mut total_bytes = 0usize;
    for image in self.images() {
      image.validate("images")?;
      total_bytes = total_bytes.saturating_add(image.estimated_byte_len());
    }
    if let Some(mask) = self.mask() {
      mask.validate("mask")?;
      total_bytes = total_bytes.saturating_add(mask.estimated_byte_len());
    }
    if let Some(fal) = self.provider_options().fal() {
      fal.validate()?;
    }
    if total_bytes > MAX_IMAGE_TOTAL_BYTES {
      return Err(ProtocolError::InvalidRequest {
        field: "images",
        message: "image inputs exceed the total size limit".to_string(),
      });
    }
    Ok(())
  }
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageUsage {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub input_tokens: Option<u32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output_tokens: Option<u32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub total_tokens: Option<u32>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageArtifact {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub url: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub data_base64: Option<String>,
  pub media_type: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub width: Option<u32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub height: Option<u32>,
  #[serde(default)]
  pub provider_metadata: Value,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
  pub images: Vec<ImageArtifact>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub text: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub usage: Option<ImageUsage>,
  #[serde(default)]
  pub provider_metadata: Value,
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::protocol::fal::options::{FalImageOptions, FalImageSize};

  #[test]
  fn should_validate_image_request_limits() {
    let request = ImageRequest::edit(
      "gpt-image-1".to_string(),
      "edit".to_string(),
      vec![ImageInput::Bytes {
        data: vec![0; MAX_IMAGE_INPUT_BYTES + 1],
        media_type: "image/png".to_string(),
        file_name: Some("in.png".to_string()),
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "images", .. })
    ));

    let request = ImageRequest::edit(
      "gpt-image-1".to_string(),
      "edit".to_string(),
      vec![ImageInput::Bytes {
        data: vec![1],
        media_type: "image/png\r\nX-Test: 1".to_string(),
        file_name: Some("in.png".to_string()),
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "images", .. })
    ));
  }

  #[test]
  fn should_reject_unsafe_image_file_name() {
    let request = ImageRequest::edit(
      "gpt-image-1".to_string(),
      "edit".to_string(),
      vec![ImageInput::Data {
        data_base64: "aW1n".to_string(),
        media_type: "image/png".to_string(),
        file_name: Some("in.png\"\r\nContent-Type: text/plain".to_string()),
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "images", .. })
    ));
  }

  #[test]
  fn should_validate_image_data_url_inputs() {
    let request = ImageRequest::edit(
      "fal/image-to-image".to_string(),
      "edit".to_string(),
      vec![ImageInput::Url {
        url: format!("data:image/png;base64,{}", "aW1n".repeat(3_000)),
        media_type: Some("image/png".to_string()),
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(request.validate().is_ok());

    let request = ImageRequest::edit(
      "fal/image-to-image".to_string(),
      "edit".to_string(),
      vec![ImageInput::Url {
        url: "data:text/plain;base64,aW1n".to_string(),
        media_type: None,
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "images", .. })
    ));
  }

  #[test]
  fn should_keep_regular_image_url_length_limit() {
    let request = ImageRequest::edit(
      "gpt-image-1".to_string(),
      "edit".to_string(),
      vec![ImageInput::Url {
        url: format!("https://example.com/{}.png", "x".repeat(MAX_IMAGE_URL_CHARS)),
        media_type: Some("image/png".to_string()),
      }],
      None,
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "images", .. })
    ));
  }

  #[test]
  fn should_reject_unsafe_image_model() {
    let request = ImageRequest::generate(
      "flux-1/../requests?x=1".to_string(),
      "draw".to_string(),
      ImageOptions::default(),
      ImageProviderOptions::default(),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest { field: "model", .. })
    ));
  }

  #[test]
  fn should_validate_fal_typed_options() {
    let request = ImageRequest::generate(
      "flux-1/schnell".to_string(),
      "draw".to_string(),
      ImageOptions::default(),
      ImageProviderOptions::Fal(FalImageOptions {
        image_size: Some(FalImageSize::Dimensions {
          width: 8193,
          height: 512,
        }),
        ..Default::default()
      }),
    );

    assert!(matches!(
      request.validate(),
      Err(ProtocolError::InvalidRequest {
        field: "provider_options.fal.image_size",
        ..
      })
    ));
  }
}
