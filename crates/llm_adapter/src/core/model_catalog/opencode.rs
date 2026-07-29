use super::helpers::{ModelRegistryVariantContract, capability, variant};

fn openai_compatible_variant(
  backend_kind: &str,
  alias_prefix: &str,
  raw_model_id: &str,
  default_for_output_type: bool,
  display_name: &str,
) -> ModelRegistryVariantContract {
  variant(
    backend_kind,
    raw_model_id,
    raw_model_id,
    &[&format!("{alias_prefix}/{raw_model_id}")],
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

fn openai_compatible_variants(
  backend_kind: &str,
  alias_prefix: &str,
  models: &[(&str, &str)],
) -> Vec<ModelRegistryVariantContract> {
  models
    .iter()
    .enumerate()
    .map(|(index, (model, display_name))| {
      openai_compatible_variant(backend_kind, alias_prefix, model, index == 0, display_name)
    })
    .collect()
}

pub(super) fn opencode_go_variants() -> Vec<ModelRegistryVariantContract> {
  openai_compatible_variants(
    "opencode_go",
    "opencode-go",
    &[
      ("kimi-k3", "OpenCode Go Kimi K3"),
      ("kimi-k2.7-code", "OpenCode Go Kimi K2.7 Code"),
      ("kimi-k2.6", "OpenCode Go Kimi K2.6"),
      ("kimi-k2.5", "OpenCode Go Kimi K2.5"),
      ("deepseek-v4-pro", "OpenCode Go DeepSeek V4 Pro"),
      ("deepseek-v4-flash", "OpenCode Go DeepSeek V4 Flash"),
      ("glm-5.2", "OpenCode Go GLM 5.2"),
      ("glm-5.1", "OpenCode Go GLM 5.1"),
      ("glm-5", "OpenCode Go GLM 5"),
      ("grok-4.5", "OpenCode Go Grok 4.5"),
      ("hy3", "OpenCode Go HY 3"),
      ("hy3-preview", "OpenCode Go HY 3 Preview"),
      ("mimo-v2.5", "OpenCode Go MiMo V2.5"),
      ("mimo-v2.5-pro", "OpenCode Go MiMo V2.5 Pro"),
      ("mimo-v2-omni", "OpenCode Go MiMo V2 Omni"),
      ("mimo-v2-pro", "OpenCode Go MiMo V2 Pro"),
      ("minimax-m3", "OpenCode Go MiniMax M3"),
      ("minimax-m2.7", "OpenCode Go MiniMax M2.7"),
      ("minimax-m2.5", "OpenCode Go MiniMax M2.5"),
      ("qwen3.7-max", "OpenCode Go Qwen3.7 Max"),
      ("qwen3.7-plus", "OpenCode Go Qwen3.7 Plus"),
      ("qwen3.6-plus", "OpenCode Go Qwen3.6 Plus"),
      ("qwen3.5-plus", "OpenCode Go Qwen3.5 Plus"),
    ],
  )
}

pub(super) fn opencode_zen_variants() -> Vec<ModelRegistryVariantContract> {
  // Zen also exposes Responses, Anthropic, and Gemini-native models. This
  // backend catalog intentionally contains only its Chat Completions routes.
  openai_compatible_variants(
    "opencode_zen",
    "opencode",
    &[
      ("kimi-k3", "OpenCode Zen Kimi K3"),
      ("kimi-k2.7-code", "OpenCode Zen Kimi K2.7 Code"),
      ("kimi-k2.6", "OpenCode Zen Kimi K2.6"),
      ("kimi-k2.5", "OpenCode Zen Kimi K2.5"),
      ("deepseek-v4-pro", "OpenCode Zen DeepSeek V4 Pro"),
      ("deepseek-v4-flash", "OpenCode Zen DeepSeek V4 Flash"),
      ("deepseek-v4-flash-free", "OpenCode Zen DeepSeek V4 Flash Free"),
      ("glm-5.2", "OpenCode Zen GLM 5.2"),
      ("glm-5.1", "OpenCode Zen GLM 5.1"),
      ("glm-5", "OpenCode Zen GLM 5"),
      ("minimax-m3", "OpenCode Zen MiniMax M3"),
      ("minimax-m2.7", "OpenCode Zen MiniMax M2.7"),
      ("minimax-m2.5", "OpenCode Zen MiniMax M2.5"),
      ("mimo-v2.5-free", "OpenCode Zen MiMo V2.5 Free"),
      ("grok-4.5", "OpenCode Zen Grok 4.5"),
      ("grok-build-0.1", "OpenCode Zen Grok Build 0.1"),
      ("big-pickle", "OpenCode Zen Big Pickle"),
      ("laguna-s-2.1-free", "OpenCode Zen Laguna S 2.1 Free"),
      ("ling-3.0-flash-free", "OpenCode Zen Ling 3.0 Flash Free"),
      ("north-mini-code-free", "OpenCode Zen North Mini Code Free"),
      ("nemotron-3-ultra-free", "OpenCode Zen Nemotron 3 Ultra Free"),
    ],
  )
}
