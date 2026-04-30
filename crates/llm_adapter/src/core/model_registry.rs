use std::collections::BTreeMap;

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[path = "model_catalog/mod.rs"]
mod catalog;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelConditions {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub input_types: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub attachment_kinds: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub attachment_source_kinds: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub has_remote_attachments: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub model_id: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub output_type: Option<String>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityAttachment {
  pub kinds: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub source_kinds: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub allow_remote_urls: Option<bool>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapability {
  pub input: Vec<String>,
  pub output: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub attachments: Option<CapabilityAttachment>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub structured_attachments: Option<CapabilityAttachment>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub default_for_output_type: Option<bool>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct CandidateModel {
  pub id: String,
  pub capabilities: Vec<ModelCapability>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelRegistryVariant {
  pub backend_kind: String,
  pub canonical_key: String,
  pub raw_model_id: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub display_name: Option<String>,
  pub aliases: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub legacy_aliases: Option<Vec<String>>,
  pub capabilities: Vec<ModelCapability>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub protocol: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub request_layer: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub route_overrides: Option<BTreeMap<String, ModelRegistryRoute>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub behavior_flags: Option<Vec<String>>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelRegistryRoute {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub protocol: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub request_layer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputType {
  Text,
  Image,
  Audio,
  File,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputType {
  Text,
  Object,
  Structured,
  Embedding,
  Rerank,
  Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachmentKind {
  Image,
  Audio,
  File,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachmentSourceKind {
  Url,
  Data,
  Bytes,
  FileHandle,
}

impl TryFrom<&str> for InputType {
  type Error = String;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    match value {
      "text" => Ok(Self::Text),
      "image" => Ok(Self::Image),
      "audio" => Ok(Self::Audio),
      "file" => Ok(Self::File),
      other => Err(format!("Unsupported capability input type: {other}")),
    }
  }
}

impl From<&InputType> for &'static str {
  fn from(value: &InputType) -> Self {
    match value {
      InputType::Text => "text",
      InputType::Image => "image",
      InputType::Audio => "audio",
      InputType::File => "file",
    }
  }
}

impl TryFrom<&str> for OutputType {
  type Error = String;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    match value {
      "text" => Ok(Self::Text),
      "object" => Ok(Self::Object),
      "structured" => Ok(Self::Structured),
      "embedding" => Ok(Self::Embedding),
      "rerank" => Ok(Self::Rerank),
      "image" => Ok(Self::Image),
      other => Err(format!("Unsupported capability output type: {other}")),
    }
  }
}

impl From<&OutputType> for &'static str {
  fn from(value: &OutputType) -> Self {
    match value {
      OutputType::Text => "text",
      OutputType::Object => "object",
      OutputType::Structured => "structured",
      OutputType::Embedding => "embedding",
      OutputType::Rerank => "rerank",
      OutputType::Image => "image",
    }
  }
}

impl TryFrom<&str> for AttachmentKind {
  type Error = String;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    match value {
      "image" => Ok(Self::Image),
      "audio" => Ok(Self::Audio),
      "file" => Ok(Self::File),
      other => Err(format!("Unsupported attachment kind: {other}")),
    }
  }
}

impl From<&AttachmentKind> for &'static str {
  fn from(value: &AttachmentKind) -> Self {
    match value {
      AttachmentKind::Image => "image",
      AttachmentKind::Audio => "audio",
      AttachmentKind::File => "file",
    }
  }
}

impl From<AttachmentKind> for InputType {
  fn from(value: AttachmentKind) -> Self {
    match value {
      AttachmentKind::Image => Self::Image,
      AttachmentKind::Audio => Self::Audio,
      AttachmentKind::File => Self::File,
    }
  }
}

impl TryFrom<&str> for AttachmentSourceKind {
  type Error = String;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    match value {
      "url" => Ok(Self::Url),
      "data" => Ok(Self::Data),
      "bytes" => Ok(Self::Bytes),
      "file_handle" => Ok(Self::FileHandle),
      other => Err(format!("Unsupported attachment source kind: {other}")),
    }
  }
}

impl From<&AttachmentSourceKind> for &'static str {
  fn from(value: &AttachmentSourceKind) -> Self {
    match value {
      AttachmentSourceKind::Url => "url",
      AttachmentSourceKind::Data => "data",
      AttachmentSourceKind::Bytes => "bytes",
      AttachmentSourceKind::FileHandle => "file_handle",
    }
  }
}

fn parse_enum_vec<T>(values: &[String]) -> Result<Vec<T>, String>
where
  for<'a> T: TryFrom<&'a str, Error = String>,
{
  values.iter().map(|value| T::try_from(value.as_str())).collect()
}

fn attachment_input_type(kind: AttachmentKind) -> InputType {
  kind.into()
}

#[derive(Debug, Clone)]
struct ModelMatchConditions {
  input_types: Vec<InputType>,
  attachment_kinds: Vec<AttachmentKind>,
  attachment_source_kinds: Vec<AttachmentSourceKind>,
  has_remote_attachments: bool,
  model_id: Option<String>,
  output_type: Option<OutputType>,
}

impl TryFrom<ModelConditions> for ModelMatchConditions {
  type Error = String;

  fn try_from(value: ModelConditions) -> Result<Self, Self::Error> {
    Ok(Self {
      input_types: parse_enum_vec(value.input_types.as_deref().unwrap_or(&[]))?,
      attachment_kinds: parse_enum_vec(value.attachment_kinds.as_deref().unwrap_or(&[]))?,
      attachment_source_kinds: parse_enum_vec(value.attachment_source_kinds.as_deref().unwrap_or(&[]))?,
      has_remote_attachments: value.has_remote_attachments.unwrap_or(false),
      model_id: value.model_id,
      output_type: value.output_type.as_deref().map(OutputType::try_from).transpose()?,
    })
  }
}

fn resolve_attachment_capability(
  capability: &ModelCapability,
  output_type: Option<OutputType>,
) -> Option<&CapabilityAttachment> {
  if output_type == Some(OutputType::Structured) {
    capability
      .structured_attachments
      .as_ref()
      .or(capability.attachments.as_ref())
  } else {
    capability.attachments.as_ref()
  }
}

fn matches_attachment_capability(capability: &ModelCapability, cond: &ModelMatchConditions) -> bool {
  if cond.attachment_kinds.is_empty() && cond.attachment_source_kinds.is_empty() && !cond.has_remote_attachments {
    return true;
  }

  if let Some(attachment_capability) = resolve_attachment_capability(capability, cond.output_type) {
    if cond.attachment_kinds.iter().any(|kind| {
      attachment_capability
        .kinds
        .iter()
        .all(|candidate| candidate != <&'static str>::from(kind))
    }) {
      return false;
    }

    if attachment_capability.source_kinds.as_ref().is_some_and(|source_kinds| {
      cond.attachment_source_kinds.iter().any(|kind| {
        source_kinds
          .iter()
          .all(|candidate| candidate != <&'static str>::from(kind))
      })
    }) {
      return false;
    }

    if cond.has_remote_attachments && attachment_capability.allow_remote_urls == Some(false) {
      return false;
    }

    return true;
  }

  !cond.attachment_kinds.iter().any(|kind| {
    capability
      .input
      .iter()
      .all(|candidate| candidate != <&'static str>::from(&attachment_input_type(*kind)))
  })
}

pub fn matches_model_capability(capability: &ModelCapability, cond: &ModelConditions) -> Result<bool, String> {
  let cond = ModelMatchConditions::try_from(cond.clone())?;
  Ok(matches_model_capability_inner(capability, &cond))
}

fn matches_model_capability_inner(capability: &ModelCapability, cond: &ModelMatchConditions) -> bool {
  if cond.output_type.is_some_and(|output_type| {
    capability
      .output
      .iter()
      .all(|value| value != <&'static str>::from(&output_type))
  }) {
    return false;
  }

  if cond.input_types.iter().any(|value| {
    capability
      .input
      .iter()
      .all(|input| input != <&'static str>::from(value))
  }) {
    return false;
  }

  matches_attachment_capability(capability, cond)
}

pub fn select_model_id(models: &[CandidateModel], cond: &ModelConditions) -> Result<Option<String>, String> {
  let cond = ModelMatchConditions::try_from(cond.clone())?;
  Ok(select_model_id_inner(models, &cond).map(str::to_owned))
}

fn select_model_id_inner<'a>(models: &'a [CandidateModel], cond: &ModelMatchConditions) -> Option<&'a str> {
  if let Some(model_id) = cond.model_id.as_deref() {
    return models
      .iter()
      .find(|model| {
        model.id == model_id
          && model
            .capabilities
            .iter()
            .any(|capability| matches_model_capability_inner(capability, cond))
      })
      .map(|model| model.id.as_str());
  }

  let output_type = cond.output_type?;
  models
    .iter()
    .find(|model| {
      model.capabilities.iter().any(|capability| {
        matches_model_capability_inner(capability, cond)
          && capability.default_for_output_type == Some(true)
          && capability
            .output
            .iter()
            .any(|value| value == <&'static str>::from(&output_type))
      })
    })
    .map(|model| model.id.as_str())
}

pub fn normalize_requested_model_id<'a>(provider_ids: &[String], model_id: &'a str) -> &'a str {
  let Some(separator_index) = model_id.find('/') else {
    return model_id;
  };
  if separator_index == 0 {
    return model_id;
  }

  let provider_id = &model_id[..separator_index];
  if provider_ids.iter().any(|candidate| candidate == provider_id) {
    &model_id[separator_index + 1..]
  } else {
    model_id
  }
}

pub fn matches_requested_model_list(
  provider_ids: &[String],
  models: &[String],
  requested_model_id: Option<&str>,
) -> bool {
  let Some(requested_model_id) = requested_model_id else {
    return false;
  };
  models.iter().any(|model| model == requested_model_id)
    || models
      .iter()
      .any(|model| model == normalize_requested_model_id(provider_ids, requested_model_id))
}

fn variant_matches(variant: &ModelRegistryVariant, cond: &ModelConditions) -> Result<bool, String> {
  Ok(
    variant
      .capabilities
      .iter()
      .any(|capability| matches_model_capability(capability, cond).unwrap_or(false)),
  )
}

fn resolve_unique_variant<'a>(
  variants: Vec<&'a ModelRegistryVariant>,
  label: &str,
  value: &str,
) -> Result<Option<&'a ModelRegistryVariant>, String> {
  match variants.len() {
    0 => Ok(None),
    1 => Ok(variants.into_iter().next()),
    _ => Err(format!("Ambiguous {label}: {value}")),
  }
}

pub fn resolve_model_registry_variant<'a>(
  variants: &'a [ModelRegistryVariant],
  backend_kind: Option<&str>,
  model_id: &str,
) -> Result<Option<(&'a ModelRegistryVariant, &'static str)>, String> {
  let scoped = variants
    .iter()
    .filter(|variant| backend_kind.is_none_or(|kind| variant.backend_kind == kind))
    .collect::<Vec<_>>();
  let try_match = |matched: Vec<&'a ModelRegistryVariant>, label: &'static str| {
    resolve_unique_variant(matched, label, model_id).map(|variant| variant.map(|entry| (entry, label)))
  };

  let candidates = if let Some(kind) = backend_kind {
    scoped
      .into_iter()
      .filter(|variant| variant.backend_kind == kind)
      .collect::<Vec<_>>()
  } else {
    scoped
  };

  try_match(
    candidates
      .iter()
      .copied()
      .filter(|variant| variant.raw_model_id == model_id)
      .collect(),
    "raw_model_id",
  )
  .and_then(|matched| {
    if matched.is_some() {
      return Ok(matched);
    }
    try_match(
      candidates
        .iter()
        .copied()
        .filter(|variant| variant.canonical_key == model_id)
        .collect(),
      "canonical",
    )
  })
  .and_then(|matched| {
    if matched.is_some() {
      return Ok(matched);
    }
    try_match(
      candidates
        .iter()
        .copied()
        .filter(|variant| variant.aliases.iter().any(|alias| alias == model_id))
        .collect(),
      "alias",
    )
  })
  .and_then(|matched| {
    if matched.is_some() {
      return Ok(matched);
    }
    try_match(
      candidates
        .iter()
        .copied()
        .filter(|variant| {
          variant
            .legacy_aliases
            .as_ref()
            .is_some_and(|aliases| aliases.iter().any(|alias| alias == model_id))
        })
        .collect(),
      "legacy_alias",
    )
  })
}

pub fn select_model_registry_variant<'a>(
  variants: &'a [ModelRegistryVariant],
  backend_kind: &str,
  cond: &ModelConditions,
) -> Result<Option<&'a ModelRegistryVariant>, String> {
  if let Some(model_id) = cond.model_id.as_deref() {
    let resolved = resolve_model_registry_variant(variants, Some(backend_kind), model_id)?;
    if let Some((variant, _)) = resolved {
      return variant_matches(variant, cond).map(|matches| matches.then_some(variant));
    }
    return Ok(None);
  }

  let matches = variants
    .iter()
    .filter(|variant| variant.backend_kind == backend_kind && variant_matches(variant, cond).unwrap_or(false))
    .collect::<Vec<_>>();

  let output_type = cond.output_type.as_deref().map(OutputType::try_from).transpose()?;
  Ok(
    matches
      .iter()
      .copied()
      .find(|variant| {
        variant.capabilities.iter().any(|capability| {
          capability.default_for_output_type == Some(true)
            && output_type.is_none_or(|output| {
              capability
                .output
                .iter()
                .any(|value| value == <&'static str>::from(&output))
            })
        })
      })
      .or_else(|| matches.into_iter().next()),
  )
}

#[must_use]
pub fn default_model_registry_variants() -> Vec<ModelRegistryVariant> {
  catalog::registry_variants()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn capability(input: &[&str], output: &[&str], default_for_output_type: bool) -> ModelCapability {
    ModelCapability {
      input: input.iter().map(|value| value.to_string()).collect(),
      output: output.iter().map(|value| value.to_string()).collect(),
      attachments: None,
      structured_attachments: None,
      default_for_output_type: Some(default_for_output_type),
    }
  }

  #[test]
  fn selects_default_model_for_output_type() {
    let models = vec![
      CandidateModel {
        id: "text-default".to_string(),
        capabilities: vec![capability(&["text"], &["text"], true)],
      },
      CandidateModel {
        id: "text-secondary".to_string(),
        capabilities: vec![capability(&["text"], &["text"], false)],
      },
    ];
    let cond = ModelConditions {
      input_types: Some(vec!["text".to_string()]),
      output_type: Some("text".to_string()),
      attachment_kinds: None,
      attachment_source_kinds: None,
      has_remote_attachments: None,
      model_id: None,
    };

    assert_eq!(
      select_model_id(&models, &cond).unwrap().as_deref(),
      Some("text-default")
    );
  }

  #[test]
  fn rejects_remote_attachments_when_capability_disallows_them() {
    let capability = ModelCapability {
      input: vec!["text".to_string(), "image".to_string()],
      output: vec!["text".to_string()],
      attachments: Some(CapabilityAttachment {
        kinds: vec!["image".to_string()],
        source_kinds: Some(vec!["url".to_string()]),
        allow_remote_urls: Some(false),
      }),
      structured_attachments: None,
      default_for_output_type: Some(true),
    };
    let cond = ModelConditions {
      input_types: Some(vec!["text".to_string(), "image".to_string()]),
      attachment_kinds: Some(vec!["image".to_string()]),
      attachment_source_kinds: Some(vec!["url".to_string()]),
      has_remote_attachments: Some(true),
      model_id: None,
      output_type: Some("text".to_string()),
    };

    assert!(!matches_model_capability(&capability, &cond).unwrap());
  }

  #[test]
  fn resolves_registry_aliases_and_ambiguity() {
    let variants = vec![
      ModelRegistryVariant {
        backend_kind: "a".to_string(),
        canonical_key: "same".to_string(),
        raw_model_id: "raw-a".to_string(),
        display_name: None,
        aliases: vec!["alias-a".to_string()],
        legacy_aliases: Some(vec!["legacy-a".to_string()]),
        capabilities: vec![capability(&["text"], &["text"], true)],
        protocol: None,
        request_layer: None,
        route_overrides: None,
        behavior_flags: None,
      },
      ModelRegistryVariant {
        backend_kind: "b".to_string(),
        canonical_key: "same".to_string(),
        raw_model_id: "raw-b".to_string(),
        display_name: None,
        aliases: Vec::new(),
        legacy_aliases: None,
        capabilities: vec![capability(&["text"], &["text"], true)],
        protocol: None,
        request_layer: None,
        route_overrides: None,
        behavior_flags: None,
      },
    ];

    assert_eq!(
      resolve_model_registry_variant(&variants, Some("a"), "legacy-a")
        .unwrap()
        .unwrap()
        .1,
      "legacy_alias"
    );
    assert!(resolve_model_registry_variant(&variants, None, "same").is_err());
  }

  #[test]
  fn default_catalog_resolves_backend_scoped_aliases() {
    let variants = default_model_registry_variants();
    let (variant, matched_by) =
      resolve_model_registry_variant(&variants, Some("anthropic_vertex"), "claude-sonnet-4.5")
        .unwrap()
        .unwrap();

    assert_eq!(matched_by, "canonical");
    assert_eq!(variant.raw_model_id, "claude-sonnet-4-5@20250929");
  }

  #[test]
  fn default_catalog_selects_backend_default_by_output() {
    let variants = default_model_registry_variants();
    let variant = select_model_registry_variant(
      &variants,
      "gemini_api",
      &ModelConditions {
        input_types: Some(vec!["text".to_string()]),
        attachment_kinds: None,
        attachment_source_kinds: None,
        has_remote_attachments: None,
        model_id: None,
        output_type: Some("embedding".to_string()),
      },
    )
    .unwrap()
    .unwrap();

    assert_eq!(variant.raw_model_id, "gemini-embedding-001");
  }
}
