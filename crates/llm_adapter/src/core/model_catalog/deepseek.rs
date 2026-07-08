use super::helpers::{ModelRegistryVariantContract, capability, variant};

pub(super) fn deepseek_variants() -> Vec<ModelRegistryVariantContract> {
  vec![
    variant(
      "deepseek",
      "deepseek-v4-pro",
      "deepseek-v4-pro",
      &["deepseek-v4-pro"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], true)],
      Some("openai_chat"),
      Some("chat_completions_no_v1"),
      &["reasoning_supported"],
      Some("DeepSeek V4 Pro"),
    ),
    variant(
      "deepseek",
      "deepseek-v4-flash",
      "deepseek-v4-flash",
      &["deepseek-v4-flash"],
      &["deepseek-chat", "deepseek-reasoner"],
      vec![capability(&["text"], &["text", "object", "structured"], false)],
      Some("openai_chat"),
      Some("chat_completions_no_v1"),
      &["reasoning_supported"],
      Some("DeepSeek V4 Flash"),
    ),
  ]
}
