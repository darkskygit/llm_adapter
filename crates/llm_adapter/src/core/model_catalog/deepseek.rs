use super::helpers::{
  CapabilityAttachmentContract, ModelRegistryVariantContract, capability, capability_with_attachments, variant,
};

pub(super) fn deepseek_variants(image_attachment: &CapabilityAttachmentContract) -> Vec<ModelRegistryVariantContract> {
  vec![
    variant(
      "deepseek",
      "deepseek-v4-pro",
      "deepseek-v4-pro",
      &["deepseek-v4-pro"],
      &[],
      vec![capability(&["text"], &["text", "object", "structured"], true)],
      Some("openai_chat"),
      Some("chat_completions"),
      &["reasoning_supported"],
      Some("DeepSeek V4 Pro"),
    ),
    variant(
      "deepseek",
      "deepseek-flash",
      "deepseek-flash",
      &["deepseek-flash"],
      &["deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object", "structured"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_chat"),
      Some("chat_completions"),
      &["reasoning_supported"],
      Some("DeepSeek V4.1 Flash"),
    ),
  ]
}
