use std::collections::BTreeMap;

use super::helpers::{
  CapabilityAttachmentContract, ModelRegistryVariantContract, capability, capability_with_attachments, route, variant,
};

pub(super) fn openai_variants(image_attachment: &CapabilityAttachmentContract) -> Vec<ModelRegistryVariantContract> {
  vec![
    variant(
      "openai_responses",
      "gpt-4o",
      "gpt-4o",
      &["gpt-4o"],
      &["gpt-4o-2024-08-06"],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_responses"),
      Some("responses"),
      &[],
      Some("GPT 4o"),
    ),
    {
      let mut model = variant(
        "openai_responses",
        "gpt-4o-mini",
        "gpt-4o-mini",
        &["gpt-4o-mini"],
        &["gpt-4o-mini-2024-07-18"],
        vec![capability_with_attachments(
          &["text", "image"],
          &["text", "object", "rerank"],
          image_attachment.clone(),
          Some(image_attachment.clone()),
          false,
        )],
        Some("openai_responses"),
        Some("responses"),
        &[],
        Some("GPT 4o Mini"),
      );
      model.route_overrides = Some(BTreeMap::from([(
        "rerank".to_string(),
        route(Some("openai_chat"), Some("chat_completions")),
      )]));
      model
    },
    {
      let mut model = variant(
        "openai_responses",
        "gpt-4.1",
        "gpt-4.1",
        &["gpt-4.1"],
        &["gpt-4.1-2025-04-14"],
        vec![capability_with_attachments(
          &["text", "image"],
          &["text", "object", "rerank", "structured"],
          image_attachment.clone(),
          Some(image_attachment.clone()),
          true,
        )],
        Some("openai_responses"),
        Some("responses"),
        &[],
        Some("GPT 4.1"),
      );
      model.route_overrides = Some(BTreeMap::from([(
        "rerank".to_string(),
        route(Some("openai_chat"), Some("chat_completions")),
      )]));
      model
    },
    {
      let mut model = variant(
        "openai_responses",
        "gpt-4.1-mini",
        "gpt-4.1-mini",
        &["gpt-4.1-mini"],
        &[],
        vec![capability_with_attachments(
          &["text", "image"],
          &["text", "object", "rerank", "structured"],
          image_attachment.clone(),
          Some(image_attachment.clone()),
          false,
        )],
        Some("openai_responses"),
        Some("responses"),
        &[],
        Some("GPT 4.1 Mini"),
      );
      model.route_overrides = Some(BTreeMap::from([(
        "rerank".to_string(),
        route(Some("openai_chat"), Some("chat_completions")),
      )]));
      model
    },
    {
      let mut model = variant(
        "openai_responses",
        "gpt-4.1-nano",
        "gpt-4.1-nano",
        &["gpt-4.1-nano"],
        &[],
        vec![capability_with_attachments(
          &["text", "image"],
          &["text", "object", "rerank", "structured"],
          image_attachment.clone(),
          Some(image_attachment.clone()),
          false,
        )],
        Some("openai_responses"),
        Some("responses"),
        &[],
        Some("GPT 4.1 Nano"),
      );
      model.route_overrides = Some(BTreeMap::from([(
        "rerank".to_string(),
        route(Some("openai_chat"), Some("chat_completions")),
      )]));
      model
    },
    variant(
      "openai_responses",
      "gpt-5",
      "gpt-5",
      &["gpt-5"],
      &["gpt-5-2025-08-07"],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object", "structured"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_responses"),
      Some("responses"),
      &[],
      Some("GPT 5"),
    ),
    variant(
      "openai_responses",
      "gpt-5-mini",
      "gpt-5-mini",
      &["gpt-5-mini"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object", "structured"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_responses"),
      Some("responses"),
      &[],
      Some("GPT 5 Mini"),
    ),
    {
      let mut model = variant(
        "openai_responses",
        "gpt-5.2",
        "gpt-5.2",
        &["gpt-5.2"],
        &["gpt-5.2-2025-12-11"],
        vec![capability_with_attachments(
          &["text", "image"],
          &["text", "object", "rerank", "structured"],
          image_attachment.clone(),
          Some(image_attachment.clone()),
          false,
        )],
        Some("openai_responses"),
        Some("responses"),
        &[],
        Some("GPT 5.2"),
      );
      model.route_overrides = Some(BTreeMap::from([(
        "rerank".to_string(),
        route(Some("openai_chat"), Some("chat_completions")),
      )]));
      model
    },
    variant(
      "openai_responses",
      "gpt-5-nano",
      "gpt-5-nano",
      &["gpt-5-nano"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object", "structured"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_responses"),
      Some("responses"),
      &[],
      Some("GPT 5 Nano"),
    ),
    variant(
      "openai_chat",
      "o1",
      "o1",
      &["o1"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_chat"),
      Some("chat_completions"),
      &[],
      Some("GPT O1"),
    ),
    variant(
      "openai_chat",
      "o3",
      "o3",
      &["o3"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_chat"),
      Some("chat_completions"),
      &[],
      Some("GPT O3"),
    ),
    variant(
      "openai_chat",
      "o4-mini",
      "o4-mini",
      &["o4-mini"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["text", "object"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      Some("openai_chat"),
      Some("chat_completions"),
      &[],
      Some("GPT O4 Mini"),
    ),
    variant(
      "openai_responses",
      "text-embedding-3-large",
      "text-embedding-3-large",
      &["text-embedding-3-large"],
      &[],
      vec![capability(&["text"], &["embedding"], true)],
      Some("openai_responses"),
      Some("responses"),
      &[],
      None,
    ),
    variant(
      "openai_responses",
      "text-embedding-3-small",
      "text-embedding-3-small",
      &["text-embedding-3-small"],
      &[],
      vec![capability(&["text"], &["embedding"], false)],
      Some("openai_responses"),
      Some("responses"),
      &[],
      None,
    ),
    variant(
      "openai_responses",
      "dall-e-3",
      "dall-e-3",
      &["dall-e-3"],
      &[],
      vec![capability(&["text"], &["image"], false)],
      Some("openai_images"),
      Some("openai_images"),
      &[],
      None,
    ),
    variant(
      "openai_responses",
      "gpt-image-1",
      "gpt-image-1",
      &["gpt-image-1"],
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &["image"],
        image_attachment.clone(),
        Some(image_attachment.clone()),
        true,
      )],
      Some("openai_images"),
      Some("openai_images"),
      &[],
      None,
    ),
  ]
}
