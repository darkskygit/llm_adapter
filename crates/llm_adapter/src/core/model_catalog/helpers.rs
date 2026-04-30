use super::super::{CapabilityAttachment, ModelCapability, ModelRegistryRoute, ModelRegistryVariant};

pub(super) type CapabilityAttachmentContract = CapabilityAttachment;
pub(super) type CapabilityModelCapability = ModelCapability;
pub(super) type ModelRegistryRouteContract = ModelRegistryRoute;
pub(super) type ModelRegistryVariantContract = ModelRegistryVariant;

pub(super) fn attachment(
  kinds: &[&str],
  source_kinds: &[&str],
  allow_remote_urls: bool,
) -> CapabilityAttachmentContract {
  CapabilityAttachmentContract {
    kinds: kinds.iter().map(|value| value.to_string()).collect(),
    source_kinds: if source_kinds.is_empty() {
      None
    } else {
      Some(source_kinds.iter().map(|value| value.to_string()).collect())
    },
    allow_remote_urls: Some(allow_remote_urls),
  }
}

pub(super) fn capability(input: &[&str], output: &[&str], default_for_output_type: bool) -> CapabilityModelCapability {
  CapabilityModelCapability {
    input: input.iter().map(|value| value.to_string()).collect(),
    output: output.iter().map(|value| value.to_string()).collect(),
    attachments: None,
    structured_attachments: None,
    default_for_output_type: default_for_output_type.then_some(true),
  }
}

pub(super) fn capability_with_attachments(
  input: &[&str],
  output: &[&str],
  attachments: CapabilityAttachmentContract,
  structured_attachments: Option<CapabilityAttachmentContract>,
  default_for_output_type: bool,
) -> CapabilityModelCapability {
  CapabilityModelCapability {
    input: input.iter().map(|value| value.to_string()).collect(),
    output: output.iter().map(|value| value.to_string()).collect(),
    attachments: Some(attachments),
    structured_attachments,
    default_for_output_type: default_for_output_type.then_some(true),
  }
}

#[expect(
  clippy::too_many_arguments,
  reason = "catalog entries map directly to registry fields"
)]
pub(super) fn variant(
  backend_kind: &str,
  canonical_key: &str,
  raw_model_id: &str,
  aliases: &[&str],
  legacy_aliases: &[&str],
  capabilities: Vec<CapabilityModelCapability>,
  protocol: Option<&str>,
  request_layer: Option<&str>,
  behavior_flags: &[&str],
  display_name: Option<&str>,
) -> ModelRegistryVariantContract {
  ModelRegistryVariantContract {
    backend_kind: backend_kind.to_string(),
    canonical_key: canonical_key.to_string(),
    raw_model_id: raw_model_id.to_string(),
    display_name: display_name.map(|value| value.to_string()),
    aliases: aliases.iter().map(|value| value.to_string()).collect(),
    legacy_aliases: if legacy_aliases.is_empty() {
      None
    } else {
      Some(legacy_aliases.iter().map(|value| value.to_string()).collect())
    },
    capabilities,
    protocol: protocol.map(|value| value.to_string()),
    request_layer: request_layer.map(|value| value.to_string()),
    route_overrides: None,
    behavior_flags: if behavior_flags.is_empty() {
      None
    } else {
      Some(behavior_flags.iter().map(|value| value.to_string()).collect())
    },
  }
}

pub(super) fn route(protocol: Option<&str>, request_layer: Option<&str>) -> ModelRegistryRouteContract {
  ModelRegistryRouteContract {
    protocol: protocol.map(|value| value.to_string()),
    request_layer: request_layer.map(|value| value.to_string()),
  }
}
