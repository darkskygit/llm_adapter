use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
  backend::BackendError,
  core::{
    AttachmentCapability, CanonicalPromptAttachment, CoreRequest, CoreToolChoice, CoreToolChoiceMode,
    CoreToolDefinition, ImageFormat, ImageInput, ImageOptions, ImageProviderOptions, ImageRequest,
    PromptAttachmentKind, PromptAttachmentSourceKind, PromptMessageInput, StructuredRequest,
    canonicalize_prompt_messages, materialize_core_messages, validate_attachment_capability,
  },
  protocol::{fal::options::FalImageOptions, gemini::image::GeminiImageOptions, openai::images::OpenAiImageOptions},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalChatRequest {
  pub model: String,
  pub messages: Vec<PromptMessageInput>,
  #[serde(default)]
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub temperature: Option<f64>,
  #[serde(default)]
  pub tools: Vec<CoreToolDefinition>,
  #[serde(default)]
  pub include: Option<Vec<String>>,
  #[serde(default)]
  pub reasoning: Option<Value>,
  #[serde(default)]
  pub response_schema: Option<Value>,
  #[serde(default)]
  pub attachment_capability: Option<AttachmentCapability>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalStructuredRequest {
  pub model: String,
  pub messages: Vec<PromptMessageInput>,
  #[serde(default)]
  pub schema: Option<Value>,
  #[serde(default)]
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub temperature: Option<f64>,
  #[serde(default)]
  pub reasoning: Option<Value>,
  #[serde(default)]
  pub strict: Option<bool>,
  #[serde(default)]
  pub response_mime_type: Option<String>,
  #[serde(default)]
  pub attachment_capability: Option<AttachmentCapability>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRequestBuildOptions {
  pub quality: Option<String>,
  pub seed: Option<u64>,
  pub model_name: Option<String>,
  pub loras: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRequestFromMessages {
  pub model: String,
  pub protocol: String,
  pub messages: Vec<PromptMessageInput>,
  #[serde(default)]
  pub options: Option<ImageRequestBuildOptions>,
}

pub fn build_canonical_chat_request(request: CanonicalChatRequest) -> Result<CoreRequest, BackendError> {
  let messages = canonicalize_prompt_messages(request.messages);
  validate_attachment_capability(&messages, request.attachment_capability.as_ref())?;
  let tool_choice = (!request.tools.is_empty()).then_some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto));

  Ok(CoreRequest {
    model: request.model,
    messages: materialize_core_messages(messages),
    stream: true,
    max_tokens: request.max_tokens,
    temperature: request.temperature,
    tools: request.tools,
    tool_choice,
    include: request.include,
    reasoning: request.reasoning,
    response_schema: request.response_schema,
  })
}

pub fn build_canonical_structured_request(
  request: CanonicalStructuredRequest,
) -> Result<StructuredRequest, BackendError> {
  let messages = canonicalize_prompt_messages(request.messages);
  validate_attachment_capability(&messages, request.attachment_capability.as_ref())?;
  let schema = request.schema.ok_or_else(|| BackendError::InvalidRequest {
    field: "schema",
    message: "Schema is required".to_string(),
  })?;

  Ok(StructuredRequest {
    model: request.model,
    messages: materialize_core_messages(messages),
    schema,
    max_tokens: request.max_tokens,
    temperature: request.temperature,
    reasoning: request.reasoning,
    strict: Some(request.strict.unwrap_or(true)),
    response_mime_type: request.response_mime_type,
  })
}

fn required_string<'a>(
  source: &'a serde_json::Map<String, Value>,
  field: &'static str,
  message: &'static str,
) -> Result<&'a str, BackendError> {
  source
    .get(field)
    .and_then(Value::as_str)
    .ok_or_else(|| BackendError::InvalidRequest {
      field,
      message: message.to_string(),
    })
}

fn image_input_from_attachment(attachment: &CanonicalPromptAttachment) -> Result<Option<ImageInput>, BackendError> {
  if attachment.kind != PromptAttachmentKind::Image {
    return Ok(None);
  }

  let source = attachment
    .source
    .as_object()
    .ok_or_else(|| BackendError::InvalidRequest {
      field: "attachments",
      message: "Image attachment source must be an object".to_string(),
    })?;
  let media_type = source
    .get("media_type")
    .and_then(Value::as_str)
    .map(ToString::to_string);
  let file_name = source.get("file_name").and_then(Value::as_str).map(ToString::to_string);

  match attachment.source_kind.as_ref() {
    Some(PromptAttachmentSourceKind::Url) => Ok(Some(ImageInput::Url {
      url: required_string(source, "url", "Image url attachment requires url")?.to_string(),
      media_type,
    })),
    Some(PromptAttachmentSourceKind::Data) => Ok(Some(ImageInput::Data {
      data_base64: required_string(source, "data", "Image data attachment requires data")?.to_string(),
      media_type: media_type.ok_or_else(|| BackendError::InvalidRequest {
        field: "attachments",
        message: "Image data attachment requires media_type".to_string(),
      })?,
      file_name,
    })),
    Some(PromptAttachmentSourceKind::Bytes) => {
      let data = required_string(source, "data", "Image bytes attachment requires data")?;
      Ok(Some(ImageInput::Bytes {
        data: BASE64_STANDARD
          .decode(data.as_bytes())
          .map_err(|_| BackendError::InvalidRequest {
            field: "attachments",
            message: "Image bytes attachment data must be base64".to_string(),
          })?,
        media_type: media_type.ok_or_else(|| BackendError::InvalidRequest {
          field: "attachments",
          message: "Image bytes attachment requires media_type".to_string(),
        })?,
        file_name,
      }))
    }
    Some(PromptAttachmentSourceKind::FileHandle) => Err(BackendError::InvalidRequest {
      field: "attachments",
      message: "Image file_handle attachments must be materialized".to_string(),
    }),
    None => Err(BackendError::InvalidRequest {
      field: "attachments",
      message: "Image attachment source kind is required".to_string(),
    }),
  }
}

fn image_provider_options(protocol: &str, options: &ImageRequestBuildOptions) -> ImageProviderOptions {
  match protocol {
    "openai_images" => ImageProviderOptions::Openai(OpenAiImageOptions::default()),
    "gemini" => ImageProviderOptions::Gemini(GeminiImageOptions::default()),
    "fal_image" => ImageProviderOptions::Fal(FalImageOptions {
      model_name: options.model_name.clone(),
      loras: options.loras.clone(),
      ..Default::default()
    }),
    _ => ImageProviderOptions::None,
  }
}

pub fn build_image_request_from_prompt_messages(
  request: ImageRequestFromMessages,
) -> Result<ImageRequest, BackendError> {
  let options = request.options.unwrap_or_default();
  let messages = canonicalize_prompt_messages(request.messages);
  let message = messages.last().ok_or_else(|| BackendError::InvalidRequest {
    field: "messages",
    message: "Prompt message is required".to_string(),
  })?;
  let prompt = message.content.trim();
  if prompt.is_empty() {
    return Err(BackendError::InvalidRequest {
      field: "prompt",
      message: "Prompt is required".to_string(),
    });
  }

  let images = message
    .attachments
    .iter()
    .map(image_input_from_attachment)
    .collect::<Result<Vec<_>, _>>()?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
  let provider_options = image_provider_options(&request.protocol, &options);
  let request_options = ImageOptions {
    quality: options.quality,
    seed: options.seed,
    output_format: Some(ImageFormat::Webp),
    ..Default::default()
  };
  let request = if images.is_empty() {
    ImageRequest::generate(request.model, prompt.to_string(), request_options, provider_options)
  } else {
    ImageRequest::edit(
      request.model,
      prompt.to_string(),
      images,
      None,
      request_options,
      provider_options,
    )
  };

  request.validate().map_err(|error| BackendError::InvalidRequest {
    field: "image_request",
    message: error.to_string(),
  })?;
  Ok(request)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{
    CanonicalChatRequest, CanonicalStructuredRequest, ImageRequestFromMessages, build_canonical_chat_request,
    build_canonical_structured_request, build_image_request_from_prompt_messages,
  };

  #[test]
  fn builds_canonical_chat_request_with_attachments_and_tools() {
    let request: CanonicalChatRequest = serde_json::from_value(json!({
      "model": "gpt-4.1",
      "messages": [
        { "role": "system", "content": "system instruction" },
        {
          "role": "user",
          "content": "hello",
          "attachments": [{ "kind": "url", "url": "https://affine.pro/image.png" }]
        },
        { "role": "system", "content": "ignored" }
      ],
      "tools": [{ "name": "doc_read", "parameters": { "type": "object" } }]
    }))
    .unwrap();

    let built = build_canonical_chat_request(request).unwrap();

    assert_eq!(built.messages.len(), 2);
    assert!(built.tool_choice.is_some());
    assert_eq!(
      serde_json::to_value(&built.messages[1].content[1]).unwrap()["type"],
      "image"
    );
  }

  #[test]
  fn requires_structured_schema() {
    let request: CanonicalStructuredRequest = serde_json::from_value(json!({
      "model": "gpt-4.1",
      "messages": [{ "role": "user", "content": "hello" }]
    }))
    .unwrap();

    assert!(build_canonical_structured_request(request).is_err());
  }

  #[test]
  fn builds_image_edit_request_from_prompt_messages() {
    let request: ImageRequestFromMessages = serde_json::from_value(json!({
      "model": "gpt-image-1",
      "protocol": "openai_images",
      "messages": [{
        "role": "user",
        "content": "remove background",
        "attachments": [{
          "kind": "data",
          "data": "aW1n",
          "mimeType": "image/png"
        }]
      }]
    }))
    .unwrap();

    let built = build_image_request_from_prompt_messages(request).unwrap();

    assert!(built.is_edit());
    assert_eq!(built.images().len(), 1);
  }
}
