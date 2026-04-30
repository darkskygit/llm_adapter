mod chat;
mod embedding;
mod image;
mod model_registry;
mod prompt;
pub mod prompt_template;
mod request_builder;
mod rerank;
mod stream;
mod structured;

pub use chat::{
  CoreAttachmentKind, CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreToolChoice,
  CoreToolChoiceMode, CoreToolDefinition, CoreUsage,
};
pub use embedding::{EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};
pub use image::{
  ImageArtifact, ImageEditRequest, ImageFormat, ImageGenerateRequest, ImageInput, ImageOptions, ImageProviderOptions,
  ImageRequest, ImageResponse, ImageUsage,
};
pub use model_registry::{
  CandidateModel, CapabilityAttachment, ModelCapability, ModelConditions, ModelRegistryRoute, ModelRegistryVariant,
  default_model_registry_variants, matches_model_capability, matches_requested_model_list,
  normalize_requested_model_id, resolve_model_registry_variant, select_model_id, select_model_registry_variant,
};
pub use prompt::{
  AttachmentCapability, CanonicalPromptAttachment, CanonicalPromptMessage, PromptAttachmentAliasInput,
  PromptAttachmentInput, PromptAttachmentInputKind, PromptAttachmentKind, PromptAttachmentProviderHint,
  PromptAttachmentSourceKind, PromptDataAttachmentInput, PromptFileHandleAttachmentInput, PromptMessageInput,
  PromptModelConditions, PromptRole, PromptUrlAttachmentInput, canonicalize_prompt_messages,
  infer_model_conditions_from_prompt_messages, materialize_core_messages, parse_data_url,
  validate_attachment_capability,
};
pub use request_builder::{
  CanonicalChatRequest, CanonicalStructuredRequest, ImageRequestBuildOptions, ImageRequestFromMessages,
  build_canonical_chat_request, build_canonical_structured_request, build_image_request_from_prompt_messages,
};
pub use rerank::{RerankCandidate, RerankRequest, RerankResponse};
pub use stream::StreamEvent;
pub use structured::{StructuredRequest, StructuredResponse};
