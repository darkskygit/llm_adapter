use super::helpers::{
  CapabilityAttachmentContract, ModelRegistryVariantContract, capability, capability_with_attachments, variant,
};

pub(super) fn kimi_variants(image_attachment: &CapabilityAttachmentContract) -> Vec<ModelRegistryVariantContract> {
  vec![
    variant(
      "kimi",
      "kimi-k3",
      "kimi-k3",
      &["kimi-k3"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object", "structured"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        true,
      )],
      Some("openai_chat"),
      Some("chat_completions"),
      &["omit_tool_choice", "reasoning_supported"],
      Some("Kimi K3"),
    ),
    variant(
      "kimi",
      "kimi-k2.7-code-highspeed",
      "kimi-k2.7-code-highspeed",
      &["kimi-k2.7-code-highspeed"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], false)],
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
