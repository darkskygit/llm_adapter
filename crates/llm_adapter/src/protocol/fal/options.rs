use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::ProtocolError;

const MAX_FAL_IMAGE_DIMENSION: u32 = 8192;
const MAX_FAL_ASPECT_RATIO_CHARS: usize = 32;
const MAX_FAL_IMAGE_OUTPUT_COUNT: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FalImageSizePreset {
  SquareHd,
  Square,
  #[serde(rename = "portrait_4_3")]
  Portrait4_3,
  #[serde(rename = "portrait_16_9")]
  Portrait16_9,
  #[serde(rename = "landscape_4_3")]
  Landscape4_3,
  #[serde(rename = "landscape_16_9")]
  Landscape16_9,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FalImageSize {
  Preset(FalImageSizePreset),
  Dimensions { width: u32, height: u32 },
}

impl FalImageSize {
  fn validate(&self) -> Result<(), ProtocolError> {
    match self {
      Self::Preset(_) => Ok(()),
      Self::Dimensions { width, height } if *width > 0 && *height > 0 => {
        if *width > MAX_FAL_IMAGE_DIMENSION || *height > MAX_FAL_IMAGE_DIMENSION {
          return Err(ProtocolError::InvalidRequest {
            field: "provider_options.fal.image_size",
            message: "image dimensions are too large".to_string(),
          });
        }
        Ok(())
      }
      Self::Dimensions { .. } => Err(ProtocolError::InvalidRequest {
        field: "provider_options.fal.image_size",
        message: "image dimensions must be positive".to_string(),
      }),
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FalImageOutputFormat {
  Png,
  Jpeg,
  Webp,
}

impl FalImageOutputFormat {
  #[must_use]
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Png => "png",
      Self::Jpeg => "jpeg",
      Self::Webp => "webp",
    }
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FalImageOptions {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub model_name: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub image_size: Option<FalImageSize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub aspect_ratio: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub num_images: Option<u32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub enable_safety_checker: Option<bool>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub output_format: Option<FalImageOutputFormat>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sync_mode: Option<bool>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub enable_prompt_expansion: Option<bool>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub loras: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub controlnets: Option<Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub extra: Option<Value>,
}

impl FalImageOptions {
  pub fn validate(&self) -> Result<(), ProtocolError> {
    if let Some(model_name) = &self.model_name {
      validate_safe_fal_string(model_name, "provider_options.fal.model_name", 128)?;
    }
    if let Some(image_size) = &self.image_size {
      image_size.validate()?;
    }
    if let Some(aspect_ratio) = &self.aspect_ratio {
      validate_safe_fal_string(
        aspect_ratio,
        "provider_options.fal.aspect_ratio",
        MAX_FAL_ASPECT_RATIO_CHARS,
      )?;
    }
    if self
      .num_images
      .is_some_and(|value| value == 0 || value > MAX_FAL_IMAGE_OUTPUT_COUNT)
    {
      return Err(ProtocolError::InvalidRequest {
        field: "provider_options.fal.num_images",
        message: "num_images must be between 1 and 10".to_string(),
      });
    }
    Ok(())
  }
}

fn validate_safe_fal_string(value: &str, field: &'static str, max_len: usize) -> Result<(), ProtocolError> {
  if value.is_empty() || value.len() > max_len {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "value length is invalid".to_string(),
    });
  }
  if value.bytes().any(|byte| byte.is_ascii_control()) {
    return Err(ProtocolError::InvalidRequest {
      field,
      message: "value must not contain control characters".to_string(),
    });
  }
  Ok(())
}
