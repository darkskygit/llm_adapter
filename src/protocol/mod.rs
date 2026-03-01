mod common;
mod error;
mod utils;

pub mod anthropic;
pub mod openai;

#[cfg(test)]
mod tests;

use common::{core_role_to_string, message_token_estimate, parse_role, parse_role_lossy, parse_text_or_array_content};
pub(crate) use common::{
  map_anthropic_finish_reason, map_responses_finish_reason, usage_from_anthropic, usage_from_openai,
  usage_from_responses,
};
pub use error::ProtocolError;
use utils::{
  get_cached_tokens, get_first_str, get_first_str_or, get_str, get_str_or, get_u32, get_u32_or, parse_json,
  parse_json_ref,
};
pub(crate) use utils::{parse_json_string, stringify_json};

use super::core::{
  CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreToolChoice, CoreToolChoiceMode,
  CoreToolDefinition, CoreUsage,
};
