mod common;
mod error;

pub mod anthropic;
pub mod fal;
pub mod gemini;
pub mod openai;

#[cfg(test)]
mod tests;

pub(crate) use common::{
  attachment_content_from_source, attachment_source, infer_media_type_from_url, map_anthropic_finish_reason,
  map_gemini_finish_reason, map_responses_finish_reason, usage_from_anthropic, usage_from_gemini, usage_from_openai,
  usage_from_responses,
};
use common::{core_role_to_string, message_token_estimate, parse_role, parse_role_lossy};
pub use error::ProtocolError;

use super::{
  core::{
    CoreAttachmentKind, CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreToolChoice,
    CoreToolChoiceMode, CoreToolDefinition, CoreUsage, RerankRequest,
  },
  utils::{
    get_cached_tokens, get_first_str, get_first_str_or, get_str, get_str_or, get_u32, get_u32_or, parse_json,
    parse_json_ref, stringify_json,
  },
};
