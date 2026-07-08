use super::helpers::{ModelRegistryVariantContract, capability, variant};

pub(super) fn kimi_variants() -> Vec<ModelRegistryVariantContract> {
  vec![
    variant(
      "kimi",
      "kimi-k2.7-code-highspeed",
      "kimi-k2.7-code-highspeed",
      &["kimi-k2.7-code-highspeed"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], true)],
      Some("openai_chat"),
      Some("chat_completions"),
      &["omit_tool_choice", "reasoning_supported"],
      Some("Kimi K2.7 Code Highspeed"),
    ),
    variant(
      "kimi",
      "kimi-k2.7-code",
      "kimi-k2.7-code",
      &["kimi-k2.7-code"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], false)],
      Some("openai_chat"),
      Some("chat_completions"),
      &["omit_tool_choice", "reasoning_supported"],
      Some("Kimi K2.7 Code"),
    ),
    variant(
      "kimi",
      "kimi-k2.6",
      "kimi-k2.6",
      &["kimi-k2.6"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], false)],
      Some("openai_chat"),
      Some("chat_completions"),
      &["omit_tool_choice", "reasoning_supported"],
      Some("Kimi K2.6"),
    ),
  ]
}
