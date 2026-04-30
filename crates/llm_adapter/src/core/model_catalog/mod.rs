mod anthropic;
mod gemini;
mod helpers;
mod openai;
mod other;

use self::{
  anthropic::anthropic_variants,
  gemini::gemini_variants,
  helpers::{ModelRegistryVariantContract, attachment},
  openai::openai_variants,
  other::{cloudflare_variants, fal_variants, morph_variants, perplexity_variants},
};

pub(crate) fn registry_variants() -> Vec<ModelRegistryVariantContract> {
  let image_attachment = attachment(&["image"], &["url", "data"], true);
  let gemini_attachment = attachment(
    &["image", "audio", "file"],
    &["url", "data", "bytes", "file_handle"],
    true,
  );

  let mut variants = Vec::new();
  variants.extend(openai_variants(&image_attachment));
  variants.extend(cloudflare_variants());
  variants.extend(fal_variants());
  variants.extend(gemini_variants(&gemini_attachment));
  variants.extend(perplexity_variants());
  variants.extend(anthropic_variants(&image_attachment));
  variants.extend(morph_variants());
  variants
}
