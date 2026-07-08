use super::helpers::{ModelRegistryVariantContract, capability, variant};

fn openai_compatible_variant(
  backend_kind: &str,
  raw_model_id: &str,
  aliases: &[&str],
  default_for_output_type: bool,
  display_name: &str,
) -> ModelRegistryVariantContract {
  variant(
    backend_kind,
    raw_model_id,
    raw_model_id,
    aliases,
    &[],
    vec![capability(
      &["text"],
      &["text", "object", "structured"],
      default_for_output_type,
    )],
    Some("openai_chat"),
    Some("chat_completions"),
    &["omit_tool_choice"],
    Some(display_name),
  )
}

pub(super) fn opencode_go_variants() -> Vec<ModelRegistryVariantContract> {
  vec![
    openai_compatible_variant(
      "opencode_go",
      "kimi-k2.7-code",
      &["opencode-go/kimi-k2.7-code"],
      true,
      "OpenCode Go Kimi K2.7 Code",
    ),
    openai_compatible_variant(
      "opencode_go",
      "kimi-k2.6",
      &["opencode-go/kimi-k2.6"],
      false,
      "OpenCode Go Kimi K2.6",
    ),
    openai_compatible_variant(
      "opencode_go",
      "deepseek-v4-pro",
      &["opencode-go/deepseek-v4-pro"],
      false,
      "OpenCode Go DeepSeek V4 Pro",
    ),
    openai_compatible_variant(
      "opencode_go",
      "deepseek-v4-flash",
      &["opencode-go/deepseek-v4-flash"],
      false,
      "OpenCode Go DeepSeek V4 Flash",
    ),
    openai_compatible_variant(
      "opencode_go",
      "glm-5.2",
      &["opencode-go/glm-5.2"],
      false,
      "OpenCode Go GLM 5.2",
    ),
    openai_compatible_variant(
      "opencode_go",
      "glm-5.1",
      &["opencode-go/glm-5.1"],
      false,
      "OpenCode Go GLM 5.1",
    ),
    openai_compatible_variant(
      "opencode_go",
      "mimo-v2.5",
      &["opencode-go/mimo-v2.5"],
      false,
      "OpenCode Go MiMo V2.5",
    ),
    openai_compatible_variant(
      "opencode_go",
      "mimo-v2.5-pro",
      &["opencode-go/mimo-v2.5-pro"],
      false,
      "OpenCode Go MiMo V2.5 Pro",
    ),
  ]
}

pub(super) fn opencode_zen_variants() -> Vec<ModelRegistryVariantContract> {
  vec![
    openai_compatible_variant(
      "opencode_zen",
      "kimi-k2.7-code",
      &["opencode/kimi-k2.7-code"],
      true,
      "OpenCode Zen Kimi K2.7 Code",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "kimi-k2.6",
      &["opencode/kimi-k2.6"],
      false,
      "OpenCode Zen Kimi K2.6",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "kimi-k2.5",
      &["opencode/kimi-k2.5"],
      false,
      "OpenCode Zen Kimi K2.5",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "deepseek-v4-pro",
      &["opencode/deepseek-v4-pro"],
      false,
      "OpenCode Zen DeepSeek V4 Pro",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "deepseek-v4-flash",
      &["opencode/deepseek-v4-flash"],
      false,
      "OpenCode Zen DeepSeek V4 Flash",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "deepseek-v4-flash-free",
      &["opencode/deepseek-v4-flash-free"],
      false,
      "OpenCode Zen DeepSeek V4 Flash Free",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "glm-5.2",
      &["opencode/glm-5.2"],
      false,
      "OpenCode Zen GLM 5.2",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "glm-5.1",
      &["opencode/glm-5.1"],
      false,
      "OpenCode Zen GLM 5.1",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "glm-5",
      &["opencode/glm-5"],
      false,
      "OpenCode Zen GLM 5",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "minimax-m3",
      &["opencode/minimax-m3"],
      false,
      "OpenCode Zen MiniMax M3",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "minimax-m2.7",
      &["opencode/minimax-m2.7"],
      false,
      "OpenCode Zen MiniMax M2.7",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "minimax-m2.5",
      &["opencode/minimax-m2.5"],
      false,
      "OpenCode Zen MiniMax M2.5",
    ),
    openai_compatible_variant(
      "opencode_zen",
      "mimo-v2.5-free",
      &["opencode/mimo-v2.5-free"],
      false,
      "OpenCode Zen MiMo V2.5 Free",
    ),
  ]
}
