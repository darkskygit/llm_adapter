use std::{
  collections::{BTreeMap, HashMap},
  fs,
  path::{Path, PathBuf},
  time::Instant,
};

use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use clap::{Parser, Subcommand};
use llm_adapter::{
  backend::{
    BackendConfig, BackendRequestLayer, ChatProtocol, DefaultHttpClient, ImageProtocol, collect_stream_events,
    dispatch_image_request, dispatch_request,
  },
  core::{
    CoreContent, CoreMessage, CoreRequest, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition,
    ImageFormat, ImageInput, ImageOptions, ImageProviderOptions, ImageRequest,
  },
  protocol::{
    fal::options::{FalImageOptions, FalImageSize, FalImageSizePreset},
    gemini::image::GeminiImageOptions,
    openai::images::OpenAiImageOptions,
  },
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_EDIT_IMAGE_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAn0lEQVR42u3awQ2AMAhAUaZzMHc28WBSV9AIUeo7MMB/txZi24/x5wkAAAAAAAAAAAAAAAAAAACgdpZ1XJ5pAO5Ev4ERX4+vRoivh1dDRKf4CoToFp+NEB3jMxEAdI3PQojO8RkIAAA0j3+KAAAAAAAAAAAAAACAtwAAAH6E/AkCsBewGbIbtB12H+BCxI0QAAAAAAAAAAAAAAAAAMw9J3ZsEO1X7TiaAAAAAElFTkSuQmCC";

#[derive(Parser)]
#[command(name = "llm_compat")]
#[command(about = "Cross-provider compatibility checks powered by llm_adapter")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Run compatibility checks
  Run {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Restrict to specific provider names (repeatable)
    #[arg(short, long, value_name = "NAME")]
    provider: Vec<String>,
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
  },
  /// Generate example config file
  Config {
    /// Output file path
    #[arg(short, long, value_name = "FILE", default_value = "llm-compat.toml")]
    output: PathBuf,
  },
  /// List configured providers
  Providers {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
  },
}

fn main() -> Result<()> {
  let cli = Cli::parse();

  match &cli.command {
    Commands::Run {
      config,
      provider,
      verbose,
    } => run_compat(config.as_ref(), provider, *verbose),
    Commands::Config { output } => generate_config(output),
    Commands::Providers { config } => list_providers(config.as_ref()),
  }
}

fn run_compat(config_path: Option<&PathBuf>, providers: &[String], verbose_override: bool) -> Result<()> {
  let mut config = load_config(config_path)?;
  if verbose_override {
    config.settings.verbose = true;
  }

  let runner = CompatibilityRunner::new(config, providers)?;
  let report = runner.run();
  report.print();

  if report.total_failed > 0 || report.provider_failed > 0 {
    return Err(anyhow!(
      "compatibility checks failed: providers_failed={}, cases_failed={}",
      report.provider_failed,
      report.total_failed
    ));
  }

  Ok(())
}

fn generate_config(output_path: &Path) -> Result<()> {
  let config = CompatibilityConfig::default();
  let content = toml::to_string_pretty(&config)?;
  fs::write(output_path, content).with_context(|| format!("failed to write {}", output_path.display()))?;
  println!("Generated compatibility config: {}", output_path.display());
  Ok(())
}

fn list_providers(config_path: Option<&PathBuf>) -> Result<()> {
  let config = load_config(config_path)?;
  let mut names: Vec<&String> = config.providers.keys().collect();
  names.sort();

  if names.is_empty() {
    println!("No providers configured.");
    return Ok(());
  }

  println!("Configured providers:");
  for name in names {
    if let Some(provider) = config.providers.get(name) {
      println!(
        "  {}: model={} chat_protocol={:?} image_protocol={:?} enabled={} requires_auth={} tool_choice_strategy={:?}",
        name,
        provider.model,
        provider.chat_protocol,
        provider.image_protocol,
        provider.enabled,
        provider.requires_auth,
        provider.effective_tool_choice_strategy(),
      );
    }
  }

  Ok(())
}

fn load_config(config_path: Option<&PathBuf>) -> Result<CompatibilityConfig> {
  if let Some(path) = config_path {
    return read_config(path);
  }

  let default_paths = ["llm-compat.toml", "compat.toml", "config.toml"];
  for path in default_paths {
    let path_buf = PathBuf::from(path);
    if path_buf.exists() {
      let config = read_config(&path_buf)?;
      println!("Loaded config from: {}", path_buf.display());
      return Ok(config);
    }
  }

  println!("No config file found, using default compatibility configuration");
  Ok(CompatibilityConfig::default())
}

fn read_config(path: &Path) -> Result<CompatibilityConfig> {
  let content = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
  let config: CompatibilityConfig =
    toml::from_str(&content).with_context(|| format!("invalid TOML in {}", path.display()))?;
  Ok(config)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CompatibilityConfig {
  #[serde(default)]
  settings: CompatSettings,
  #[serde(default)]
  tests: TestSuiteConfig,
  #[serde(default)]
  image: ImageCaseConfig,
  #[serde(default)]
  providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CompatSettings {
  #[serde(default = "default_timeout_seconds")]
  timeout_seconds: u64,
  #[serde(default)]
  verbose: bool,
  #[serde(default)]
  fail_fast: bool,
}

impl Default for CompatSettings {
  fn default() -> Self {
    Self {
      timeout_seconds: default_timeout_seconds(),
      verbose: false,
      fail_fast: false,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TestSuiteConfig {
  #[serde(default = "default_true")]
  basic_text: bool,
  #[serde(default = "default_true")]
  stream_text: bool,
  #[serde(default = "default_true")]
  conversation_memory: bool,
  #[serde(default = "default_true")]
  tool_call: bool,
  #[serde(default)]
  image_generation: bool,
  #[serde(default)]
  image_edit: bool,
  #[serde(default = "default_expected_token")]
  expected_token: String,
  #[serde(default = "default_magic_number")]
  magic_number: u32,
}

impl Default for TestSuiteConfig {
  fn default() -> Self {
    Self {
      basic_text: true,
      stream_text: true,
      conversation_memory: true,
      tool_call: true,
      image_generation: false,
      image_edit: false,
      expected_token: default_expected_token(),
      magic_number: default_magic_number(),
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageCaseConfig {
  #[serde(default)]
  output_dir: Option<PathBuf>,
  #[serde(default)]
  generation_prompt: Option<String>,
  #[serde(default)]
  edit_prompt: Option<String>,
  #[serde(default)]
  edit_input: Option<ImageInputConfig>,
  #[serde(default)]
  n: Option<u32>,
  #[serde(default)]
  size: Option<String>,
  #[serde(default)]
  aspect_ratio: Option<String>,
  #[serde(default)]
  quality: Option<String>,
  #[serde(default)]
  output_format: Option<ImageFormat>,
  #[serde(default)]
  output_compression: Option<u8>,
  #[serde(default)]
  background: Option<String>,
  #[serde(default)]
  seed: Option<u64>,
  #[serde(default)]
  openai_input_fidelity: Option<String>,
  #[serde(default)]
  fal_model_name: Option<String>,
  #[serde(default)]
  fal_image_size: Option<FalImageSize>,
  #[serde(default)]
  fal_num_images: Option<u32>,
  #[serde(default)]
  fal_enable_safety_checker: Option<bool>,
  #[serde(default)]
  fal_sync_mode: Option<bool>,
  #[serde(default)]
  fal_enable_prompt_expansion: Option<bool>,
  #[serde(default)]
  gemini_response_modalities: Option<Vec<String>>,
}

impl Default for ImageCaseConfig {
  fn default() -> Self {
    Self::empty()
  }
}

impl ImageCaseConfig {
  fn empty() -> Self {
    Self {
      output_dir: None,
      generation_prompt: None,
      edit_prompt: None,
      edit_input: None,
      n: None,
      size: None,
      aspect_ratio: None,
      quality: None,
      output_format: None,
      output_compression: None,
      background: None,
      seed: None,
      openai_input_fidelity: None,
      fal_model_name: None,
      fal_image_size: None,
      fal_num_images: None,
      fal_enable_safety_checker: None,
      fal_sync_mode: None,
      fal_enable_prompt_expansion: None,
      gemini_response_modalities: None,
    }
  }

  fn generated_example() -> Self {
    Self {
      output_dir: Some(PathBuf::from("llm-compat-output")),
      generation_prompt: Some("A small blue square app icon on a plain white background".to_string()),
      edit_prompt: Some("Convert the image into a clean sticker with a white background".to_string()),
      n: Some(1),
      size: Some("1024x1024".to_string()),
      aspect_ratio: Some("1:1".to_string()),
      output_format: Some(ImageFormat::Png),
      seed: Some(7),
      fal_num_images: Some(1),
      fal_enable_safety_checker: Some(true),
      fal_sync_mode: Some(true),
      gemini_response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
      ..Self::empty()
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageInputConfig {
  #[serde(default)]
  url: Option<String>,
  #[serde(default)]
  data_base64: Option<String>,
  #[serde(default)]
  media_type: Option<String>,
  #[serde(default)]
  file_name: Option<String>,
}

impl Default for ImageInputConfig {
  fn default() -> Self {
    Self {
      url: None,
      data_base64: Some(DEFAULT_EDIT_IMAGE_BASE64.to_string()),
      media_type: Some("image/png".to_string()),
      file_name: Some("compat-edit-source.png".to_string()),
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ProviderConfig {
  #[serde(default = "default_true")]
  enabled: bool,
  base_url: String,
  model: String,
  #[serde(default)]
  auth_token: Option<String>,
  #[serde(default)]
  auth_token_env: Option<String>,
  #[serde(default = "default_false")]
  requires_auth: bool,
  #[serde(default = "default_chat_protocol")]
  chat_protocol: Option<ChatProtocol>,
  #[serde(default)]
  image_protocol: Option<ImageProtocol>,
  #[serde(default)]
  request_layer: Option<BackendRequestLayer>,
  #[serde(default)]
  headers: BTreeMap<String, String>,
  #[serde(default = "default_true")]
  supports_stream: bool,
  #[serde(default = "default_true")]
  supports_tools: bool,
  #[serde(default)]
  max_tokens: Option<u32>,
  #[serde(default)]
  temperature: Option<f64>,
  #[serde(default)]
  tool_choice_strategy: Option<ToolChoiceStrategy>,
  #[serde(default = "default_true")]
  supports_image_generation: bool,
  #[serde(default = "default_true")]
  supports_image_edit: bool,
  #[serde(default)]
  image: ImageCaseConfig,
}

impl Default for ProviderConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      base_url: String::new(),
      model: String::new(),
      auth_token: None,
      auth_token_env: None,
      requires_auth: false,
      chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
      image_protocol: None,
      request_layer: None,
      headers: BTreeMap::new(),
      supports_stream: true,
      supports_tools: true,
      max_tokens: Some(128),
      temperature: Some(0.0),
      tool_choice_strategy: None,
      supports_image_generation: true,
      supports_image_edit: true,
      image: ImageCaseConfig::default(),
    }
  }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ToolChoiceStrategy {
  Specific,
  Required,
  Auto,
  None,
  Omit,
}

impl ProviderConfig {
  fn effective_tool_choice_strategy(&self) -> ToolChoiceStrategy {
    self.tool_choice_strategy.unwrap_or(ToolChoiceStrategy::Specific)
  }
}

impl Default for CompatibilityConfig {
  fn default() -> Self {
    let mut providers = HashMap::new();

    providers.insert(
      "openai".to_string(),
      ProviderConfig {
        base_url: "https://api.openai.com".to_string(),
        model: "gpt-5.2".to_string(),
        auth_token_env: Some("OPENAI_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiResponses),
        request_layer: Some(BackendRequestLayer::Responses),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "anthropic".to_string(),
      ProviderConfig {
        base_url: "https://api.anthropic.com".to_string(),
        model: "claude-sonnet-4-5-20250929".to_string(),
        auth_token_env: Some("ANTHROPIC_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::AnthropicMessages),
        request_layer: Some(BackendRequestLayer::Anthropic),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "gemini".to_string(),
      ProviderConfig {
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        model: "gemini-2.5-flash".to_string(),
        auth_token_env: Some("GEMINI_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::GeminiGenerateContent),
        image_protocol: None,
        request_layer: Some(BackendRequestLayer::GeminiApi),
        supports_image_generation: false,
        supports_image_edit: false,
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "gemini_vertex".to_string(),
      ProviderConfig {
        enabled: false,
        base_url: "https://us-central1-aiplatform.googleapis.com/v1/projects/your-project/locations/us-central1/publishers/google".to_string(),
        model: "gemini-2.5-flash".to_string(),
        auth_token_env: Some("VERTEX_TOKEN".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::GeminiGenerateContent),
        image_protocol: None,
        request_layer: Some(BackendRequestLayer::GeminiVertex),
        supports_image_generation: false,
        supports_image_edit: false,
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "openai_image".to_string(),
      ProviderConfig {
        enabled: false,
        base_url: "https://api.openai.com".to_string(),
        model: "gpt-image-1".to_string(),
        auth_token_env: Some("OPENAI_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: None,
        image_protocol: Some(ImageProtocol::OpenaiImages),
        request_layer: Some(BackendRequestLayer::OpenaiImages),
        supports_stream: false,
        supports_tools: false,
        image: ImageCaseConfig {
          quality: Some("low".to_string()),
          ..ImageCaseConfig::empty()
        },
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "gemini_nano_banana".to_string(),
      ProviderConfig {
        enabled: false,
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        model: "gemini-2.5-flash-image".to_string(),
        auth_token_env: Some("GEMINI_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: None,
        image_protocol: Some(ImageProtocol::GeminiGenerateContent),
        request_layer: Some(BackendRequestLayer::GeminiApi),
        supports_stream: false,
        supports_tools: false,
        image: ImageCaseConfig {
          gemini_response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
          ..ImageCaseConfig::empty()
        },
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "fal_image_generation".to_string(),
      ProviderConfig {
        enabled: false,
        base_url: "https://fal.run".to_string(),
        model: "flux-1/schnell".to_string(),
        auth_token_env: Some("FAL_KEY".to_string()),
        requires_auth: true,
        chat_protocol: None,
        image_protocol: Some(ImageProtocol::FalImage),
        request_layer: Some(BackendRequestLayer::Fal),
        supports_stream: false,
        supports_tools: false,
        supports_image_edit: false,
        image: ImageCaseConfig {
          fal_image_size: Some(FalImageSize::Preset(FalImageSizePreset::SquareHd)),
          output_format: Some(ImageFormat::Png),
          ..ImageCaseConfig::empty()
        },
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "fal_image_edit".to_string(),
      ProviderConfig {
        enabled: false,
        base_url: "https://fal.run".to_string(),
        model: "lcm-sd15-i2i".to_string(),
        auth_token_env: Some("FAL_KEY".to_string()),
        requires_auth: true,
        chat_protocol: None,
        image_protocol: Some(ImageProtocol::FalImage),
        request_layer: Some(BackendRequestLayer::Fal),
        supports_stream: false,
        supports_tools: false,
        supports_image_generation: false,
        image: ImageCaseConfig {
          edit_input: Some(ImageInputConfig {
            url: Some("https://raw.githubusercontent.com/github/explore/main/topics/rust/rust.png".to_string()),
            data_base64: None,
            media_type: Some("image/png".to_string()),
            file_name: None,
          }),
          ..ImageCaseConfig::empty()
        },
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "groq".to_string(),
      ProviderConfig {
        base_url: "https://api.groq.com/openai".to_string(),
        model: "llama-3.3-70b-versatile".to_string(),
        auth_token_env: Some("GROQ_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "xai".to_string(),
      ProviderConfig {
        base_url: "https://api.x.ai".to_string(),
        model: "grok-3-mini".to_string(),
        auth_token_env: Some("XAI_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "openrouter".to_string(),
      ProviderConfig {
        base_url: "https://openrouter.ai/api".to_string(),
        model: "openai/gpt-5.2".to_string(),
        auth_token_env: Some("OPENROUTER_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "moonshot".to_string(),
      ProviderConfig {
        base_url: "https://api.moonshot.ai".to_string(),
        model: "kimi-k2.5".to_string(),
        auth_token_env: Some("MOONSHOT_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        tool_choice_strategy: Some(ToolChoiceStrategy::Omit),
        temperature: Some(1.0),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "minimax".to_string(),
      ProviderConfig {
        base_url: "https://api.minimax.io".to_string(),
        model: "MiniMax-M2.1".to_string(),
        auth_token_env: Some("MINIMAX_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "zhipu".to_string(),
      ProviderConfig {
        base_url: "https://api.z.ai/api/paas/v4".to_string(),
        model: "glm-4.6v".to_string(),
        auth_token_env: Some("ZHIPU_API_KEY".to_string()),
        requires_auth: true,
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletionsNoV1),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "ollama".to_string(),
      ProviderConfig {
        base_url: "http://localhost:11434".to_string(),
        model: "qwen3:14b".to_string(),
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        requires_auth: false,
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "llama_server".to_string(),
      ProviderConfig {
        base_url: "http://localhost:8080".to_string(),
        model: "local-model".to_string(),
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        requires_auth: false,
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "lmstudio".to_string(),
      ProviderConfig {
        base_url: "http://localhost:1234".to_string(),
        model: "local-model".to_string(),
        chat_protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        requires_auth: false,
        tool_choice_strategy: Some(ToolChoiceStrategy::Required),
        ..ProviderConfig::default()
      },
    );

    Self {
      settings: CompatSettings::default(),
      tests: TestSuiteConfig::default(),
      image: ImageCaseConfig::generated_example(),
      providers,
    }
  }
}

fn default_timeout_seconds() -> u64 {
  30
}

fn default_true() -> bool {
  true
}

fn default_false() -> bool {
  false
}

fn default_expected_token() -> String {
  "compat-ok".to_string()
}

fn default_magic_number() -> u32 {
  39
}

fn default_chat_protocol() -> Option<ChatProtocol> {
  Some(ChatProtocol::OpenaiChatCompletions)
}

struct CompatibilityRunner {
  settings: CompatSettings,
  tests: TestSuiteConfig,
  image: ImageCaseConfig,
  providers: Vec<(String, ProviderConfig)>,
  client: DefaultHttpClient,
}

impl CompatibilityRunner {
  fn new(config: CompatibilityConfig, selected_providers: &[String]) -> Result<Self> {
    let mut providers: Vec<(String, ProviderConfig)> = config
      .providers
      .into_iter()
      .filter(|(_, provider)| provider.enabled)
      .collect();
    providers.sort_by(|a, b| a.0.cmp(&b.0));

    if !selected_providers.is_empty() {
      let allowed: std::collections::BTreeSet<&str> = selected_providers.iter().map(String::as_str).collect();
      providers.retain(|(name, _)| allowed.contains(name.as_str()));
      if providers.is_empty() {
        return Err(anyhow!("no matching providers found for {:?}", selected_providers));
      }
    }

    Ok(Self {
      settings: config.settings,
      tests: config.tests,
      image: config.image,
      providers,
      client: DefaultHttpClient::default(),
    })
  }

  fn run(&self) -> CompatibilityReport {
    let mut provider_reports = Vec::new();

    for (provider_name, provider_cfg) in &self.providers {
      let report = self.run_provider(provider_name, provider_cfg);
      provider_reports.push(report);
    }

    CompatibilityReport::from_reports(provider_reports)
  }

  fn run_provider(&self, provider_name: &str, provider_cfg: &ProviderConfig) -> ProviderReport {
    let mut report = ProviderReport::new(provider_name, provider_cfg.model.clone());

    let auth_token = resolve_token(
      provider_cfg.auth_token.as_deref(),
      provider_cfg.auth_token_env.as_deref(),
    );
    if provider_cfg.requires_auth && auth_token.is_empty() {
      report.push_case(CaseOutcome::skipped(
        "provider_init",
        format!(
          "missing auth token (set auth_token or env `{}`)",
          provider_cfg.auth_token_env.as_deref().unwrap_or("<unset>")
        ),
      ));
      return report;
    }

    let backend = BackendConfig {
      base_url: provider_cfg.base_url.clone(),
      auth_token,
      request_layer: provider_cfg.request_layer,
      headers: provider_cfg.headers.clone(),
      no_streaming: false,
      timeout_ms: Some(self.settings.timeout_seconds.saturating_mul(1000)),
    };

    if self.tests.basic_text {
      let outcome = if provider_cfg.chat_protocol.is_some() {
        self.case_basic_text(provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("basic_text", "provider protocol is not a text protocol")
      };
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.stream_text {
      let outcome = if provider_cfg.chat_protocol.is_none() {
        CaseOutcome::skipped("stream_text", "provider protocol is not a text protocol")
      } else if provider_cfg.supports_stream {
        self.case_stream_text(provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("stream_text", "provider marked supports_stream=false")
      };
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.conversation_memory {
      let outcome = if provider_cfg.chat_protocol.is_some() {
        self.case_conversation_memory(provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("conversation_memory", "provider protocol is not a text protocol")
      };
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.tool_call {
      let outcome = if provider_cfg.chat_protocol.is_none() {
        CaseOutcome::skipped("tool_call", "provider protocol is not a text protocol")
      } else if provider_cfg.supports_tools {
        self.case_tool_call(provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("tool_call", "provider marked supports_tools=false")
      };
      report.push_case(outcome);
    }

    if self.tests.image_generation {
      let outcome = if provider_cfg.image_protocol.is_none() {
        CaseOutcome::skipped("image_generation", "provider protocol is not an image protocol")
      } else if provider_cfg.supports_image_generation {
        self.case_image_generation(provider_name, provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("image_generation", "provider marked supports_image_generation=false")
      };
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.image_edit {
      let outcome = if provider_cfg.image_protocol.is_none() {
        CaseOutcome::skipped("image_edit", "provider protocol is not an image protocol")
      } else if provider_cfg.supports_image_edit {
        self.case_image_edit(provider_name, provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("image_edit", "provider marked supports_image_edit=false")
      };
      report.push_case(outcome);
    }

    report
  }

  fn case_basic_text(&self, provider_cfg: &ProviderConfig, backend: &BackendConfig) -> CaseOutcome {
    let case_name = "basic_text";
    let prompt = format!("Reply with exactly '{}' and nothing else.", self.tests.expected_token);
    let request = self.build_request(
      provider_cfg,
      vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text { text: prompt }],
      }],
    );

    let started = Instant::now();
    match dispatch_request(
      &self.client,
      backend,
      provider_cfg.chat_protocol.expect("chat protocol"),
      &request,
    ) {
      Ok(response) => {
        let elapsed = started.elapsed().as_millis() as u64;
        let text = message_text(&response.message);
        if text.to_lowercase().contains(&self.tests.expected_token.to_lowercase()) {
          CaseOutcome::passed(
            case_name,
            elapsed,
            format!("matched token, usage={}", response.usage.total_tokens),
          )
        } else {
          CaseOutcome::failed(
            case_name,
            elapsed,
            format!("token not found in response: {:?}", truncate_text(&text, 160)),
          )
        }
      }
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn case_stream_text(&self, provider_cfg: &ProviderConfig, backend: &BackendConfig) -> CaseOutcome {
    let case_name = "stream_text";
    let prompt = format!("Reply with exactly '{}' and nothing else.", self.tests.expected_token);
    let mut request = self.build_request(
      provider_cfg,
      vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text { text: prompt }],
      }],
    );
    request.stream = true;

    let started = Instant::now();
    match collect_stream_events(
      &self.client,
      backend,
      provider_cfg.chat_protocol.expect("chat protocol"),
      &request,
    ) {
      Ok(events) => {
        let elapsed = started.elapsed().as_millis() as u64;
        let text = stream_text(&events);
        let has_done = events
          .iter()
          .any(|event| matches!(event, llm_adapter::core::StreamEvent::Done { .. }));
        if !has_done {
          return CaseOutcome::failed(case_name, elapsed, "missing done event".to_string());
        }
        if text.to_lowercase().contains(&self.tests.expected_token.to_lowercase()) {
          CaseOutcome::passed(
            case_name,
            elapsed,
            format!("stream matched token, events={}", events.len()),
          )
        } else {
          CaseOutcome::failed(
            case_name,
            elapsed,
            format!("token not found in stream output: {:?}", truncate_text(&text, 160)),
          )
        }
      }
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn case_conversation_memory(&self, provider_cfg: &ProviderConfig, backend: &BackendConfig) -> CaseOutcome {
    let case_name = "conversation_memory";
    let magic = self.tests.magic_number;
    let request = self.build_request(
      provider_cfg,
      vec![
        CoreMessage {
          role: CoreRole::User,
          content: vec![CoreContent::Text {
            text: format!("Remember this number for later: {}.", magic),
          }],
        },
        CoreMessage {
          role: CoreRole::User,
          content: vec![CoreContent::Text {
            text: "What number did I ask you to remember? Reply with number only.".to_string(),
          }],
        },
      ],
    );

    let started = Instant::now();
    match dispatch_request(
      &self.client,
      backend,
      provider_cfg.chat_protocol.expect("chat protocol"),
      &request,
    ) {
      Ok(response) => {
        let elapsed = started.elapsed().as_millis() as u64;
        let text = message_text(&response.message);
        if text.contains(&magic.to_string()) {
          CaseOutcome::passed(case_name, elapsed, format!("memory confirmed: {}", magic))
        } else {
          CaseOutcome::failed(
            case_name,
            elapsed,
            format!("expected {} in response: {:?}", magic, truncate_text(&text, 160)),
          )
        }
      }
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn case_tool_call(&self, provider_cfg: &ProviderConfig, backend: &BackendConfig) -> CaseOutcome {
    let case_name = "tool_call";
    let tool_name = "get_weather";
    let mut request = self.build_request(
      provider_cfg,
      vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Text {
          text: "Call get_weather with location=Tokyo and unit=c. Do not answer in natural language.".to_string(),
        }],
      }],
    );
    request.tools = vec![CoreToolDefinition {
      name: tool_name.to_string(),
      description: Some("Get weather".to_string()),
      parameters: json!({
        "type": "object",
        "properties": {
          "location": { "type": "string" },
          "unit": { "type": "string", "enum": ["c", "f"] }
        },
        "required": ["location", "unit"],
        "additionalProperties": false
      }),
    }];
    request.tool_choice = match provider_cfg.effective_tool_choice_strategy() {
      ToolChoiceStrategy::Specific => Some(CoreToolChoice::Specific {
        name: tool_name.to_string(),
      }),
      ToolChoiceStrategy::Required => Some(CoreToolChoice::Mode(CoreToolChoiceMode::Required)),
      ToolChoiceStrategy::Auto => Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto)),
      ToolChoiceStrategy::None => Some(CoreToolChoice::Mode(CoreToolChoiceMode::None)),
      ToolChoiceStrategy::Omit => None,
    };

    let started = Instant::now();
    match dispatch_request(
      &self.client,
      backend,
      provider_cfg.chat_protocol.expect("chat protocol"),
      &request,
    ) {
      Ok(response) => {
        let elapsed = started.elapsed().as_millis() as u64;
        evaluate_tool_call_response(case_name, elapsed, &response)
      }
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn case_image_generation(
    &self,
    provider_name: &str,
    provider_cfg: &ProviderConfig,
    backend: &BackendConfig,
  ) -> CaseOutcome {
    let case_name = "image_generation";
    let request = ImageRequest::generate(
      provider_cfg.model.clone(),
      self.image_generation_prompt(provider_cfg),
      self.image_options(provider_cfg),
      self.image_provider_options(provider_cfg),
    );

    let started = Instant::now();
    match dispatch_image_request(
      &self.client,
      backend,
      provider_cfg.image_protocol.expect("image protocol"),
      &request,
    ) {
      Ok(response) => self.evaluate_image_response(
        provider_name,
        case_name,
        started.elapsed().as_millis() as u64,
        &response,
      ),
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn case_image_edit(
    &self,
    provider_name: &str,
    provider_cfg: &ProviderConfig,
    backend: &BackendConfig,
  ) -> CaseOutcome {
    let case_name = "image_edit";
    let input = match self.image_edit_input(provider_cfg) {
      Ok(input) => input,
      Err(error) => return CaseOutcome::failed(case_name, 0, error.to_string()),
    };
    let request = ImageRequest::edit(
      provider_cfg.model.clone(),
      self.image_edit_prompt(provider_cfg),
      vec![input],
      None,
      self.image_options(provider_cfg),
      self.image_provider_options(provider_cfg),
    );

    let started = Instant::now();
    match dispatch_image_request(
      &self.client,
      backend,
      provider_cfg.image_protocol.expect("image protocol"),
      &request,
    ) {
      Ok(response) => self.evaluate_image_response(
        provider_name,
        case_name,
        started.elapsed().as_millis() as u64,
        &response,
      ),
      Err(error) => CaseOutcome::failed(case_name, started.elapsed().as_millis() as u64, error.to_string()),
    }
  }

  fn build_request(&self, provider_cfg: &ProviderConfig, messages: Vec<CoreMessage>) -> CoreRequest {
    CoreRequest {
      model: provider_cfg.model.clone(),
      messages,
      stream: false,
      max_tokens: provider_cfg.max_tokens.or(Some(128)),
      temperature: provider_cfg.temperature.or(Some(0.0)),
      tools: vec![],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    }
  }

  fn image_generation_prompt(&self, provider_cfg: &ProviderConfig) -> String {
    provider_cfg
      .image
      .generation_prompt
      .clone()
      .or_else(|| self.image.generation_prompt.clone())
      .unwrap_or_else(|| "A small blue square app icon on a plain white background".to_string())
  }

  fn image_edit_prompt(&self, provider_cfg: &ProviderConfig) -> String {
    provider_cfg
      .image
      .edit_prompt
      .clone()
      .or_else(|| self.image.edit_prompt.clone())
      .unwrap_or_else(|| "Convert the image into a clean sticker with a white background".to_string())
  }

  fn image_options(&self, provider_cfg: &ProviderConfig) -> ImageOptions {
    ImageOptions {
      n: provider_cfg.image.n.or(self.image.n),
      size: provider_cfg.image.size.clone().or_else(|| self.image.size.clone()),
      aspect_ratio: provider_cfg
        .image
        .aspect_ratio
        .clone()
        .or_else(|| self.image.aspect_ratio.clone()),
      quality: provider_cfg
        .image
        .quality
        .clone()
        .or_else(|| self.image.quality.clone()),
      output_format: provider_cfg.image.output_format.or(self.image.output_format),
      output_compression: provider_cfg.image.output_compression.or(self.image.output_compression),
      background: provider_cfg
        .image
        .background
        .clone()
        .or_else(|| self.image.background.clone()),
      seed: provider_cfg.image.seed.or(self.image.seed),
    }
  }

  fn image_provider_options(&self, provider_cfg: &ProviderConfig) -> ImageProviderOptions {
    if let Some(options) = Some(OpenAiImageOptions {
      input_fidelity: provider_cfg
        .image
        .openai_input_fidelity
        .clone()
        .or_else(|| self.image.openai_input_fidelity.clone()),
    })
    .filter(|options| options.input_fidelity.is_some())
    {
      return ImageProviderOptions::Openai(options);
    }

    if let Some(options) = Some(GeminiImageOptions {
      response_modalities: provider_cfg
        .image
        .gemini_response_modalities
        .clone()
        .or_else(|| self.image.gemini_response_modalities.clone()),
    })
    .filter(|options| options.response_modalities.is_some())
    {
      return ImageProviderOptions::Gemini(options);
    }

    if let Some(options) = Some(FalImageOptions {
      model_name: provider_cfg
        .image
        .fal_model_name
        .clone()
        .or_else(|| self.image.fal_model_name.clone()),
      image_size: provider_cfg
        .image
        .fal_image_size
        .clone()
        .or_else(|| self.image.fal_image_size.clone()),
      aspect_ratio: None,
      num_images: provider_cfg.image.fal_num_images.or(self.image.fal_num_images),
      enable_safety_checker: provider_cfg
        .image
        .fal_enable_safety_checker
        .or(self.image.fal_enable_safety_checker),
      output_format: None,
      sync_mode: provider_cfg.image.fal_sync_mode.or(self.image.fal_sync_mode),
      enable_prompt_expansion: provider_cfg
        .image
        .fal_enable_prompt_expansion
        .or(self.image.fal_enable_prompt_expansion),
      loras: None,
      controlnets: None,
      extra: None,
    })
    .filter(|options| {
      options.model_name.is_some()
        || options.image_size.is_some()
        || options.num_images.is_some()
        || options.enable_safety_checker.is_some()
        || options.sync_mode.is_some()
        || options.enable_prompt_expansion.is_some()
    }) {
      return ImageProviderOptions::Fal(options);
    }

    ImageProviderOptions::default()
  }

  fn image_edit_input(&self, provider_cfg: &ProviderConfig) -> Result<ImageInput> {
    let input = provider_cfg
      .image
      .edit_input
      .as_ref()
      .or(self.image.edit_input.as_ref());
    image_input_from_config(input)
  }

  fn image_output_dir(&self) -> PathBuf {
    self
      .image
      .output_dir
      .clone()
      .unwrap_or_else(|| PathBuf::from("llm-compat-output"))
  }

  fn evaluate_image_response(
    &self,
    provider_name: &str,
    case_name: &str,
    elapsed: u64,
    response: &llm_adapter::core::ImageResponse,
  ) -> CaseOutcome {
    evaluate_image_response(
      case_name,
      elapsed,
      response,
      persist_image_response(&self.image_output_dir(), provider_name, case_name, response),
    )
  }
}

fn image_input_from_config(input: Option<&ImageInputConfig>) -> Result<ImageInput> {
  if let Some(input) = input {
    if let Some(url) = &input.url {
      return Ok(ImageInput::Url {
        url: url.clone(),
        media_type: input.media_type.clone(),
      });
    }
    if let Some(data_base64) = &input.data_base64 {
      return Ok(ImageInput::Data {
        data_base64: data_base64.clone(),
        media_type: input.media_type.clone().unwrap_or_else(|| "image/png".to_string()),
        file_name: input.file_name.clone(),
      });
    }
    return Err(anyhow!("image edit input must set either url or data_base64"));
  }

  Ok(ImageInput::Data {
    data_base64: DEFAULT_EDIT_IMAGE_BASE64.to_string(),
    media_type: "image/png".to_string(),
    file_name: Some("compat-edit-source.png".to_string()),
  })
}

fn resolve_token(explicit: Option<&str>, env_name: Option<&str>) -> String {
  if let Some(token) = explicit {
    return token.to_string();
  }
  if let Some(env_name) = env_name {
    if let Ok(token) = std::env::var(env_name) {
      return token;
    }
  }
  String::new()
}

fn message_text(message: &CoreMessage) -> String {
  let mut out = String::new();
  for content in &message.content {
    match content {
      CoreContent::Text { text } => out.push_str(text),
      CoreContent::Reasoning { text, .. } => out.push_str(text),
      _ => {}
    }
  }
  out
}

fn stream_text(events: &[llm_adapter::core::StreamEvent]) -> String {
  let mut out = String::new();
  for event in events {
    match event {
      llm_adapter::core::StreamEvent::TextDelta { text } => out.push_str(text),
      llm_adapter::core::StreamEvent::ReasoningDelta { text } => out.push_str(text),
      _ => {}
    }
  }
  out
}

fn truncate_text(value: &str, max_chars: usize) -> String {
  let mut result = String::new();
  for (index, ch) in value.chars().enumerate() {
    if index >= max_chars {
      result.push_str("...");
      break;
    }
    result.push(ch);
  }
  result
}

fn evaluate_tool_call_response(
  case_name: &str,
  elapsed: u64,
  response: &llm_adapter::core::CoreResponse,
) -> CaseOutcome {
  for content in &response.message.content {
    if let CoreContent::ToolCall {
      name,
      arguments,
      call_id: _,
      thought: _,
    } = content
      && name == "get_weather"
    {
      let location = arguments.get("location").and_then(serde_json::Value::as_str);
      let unit = arguments.get("unit").and_then(serde_json::Value::as_str);
      if location.is_some() && unit.is_some() {
        return CaseOutcome::passed(
          case_name,
          elapsed,
          format!(
            "tool call ok: location={}, unit={}",
            location.unwrap_or(""),
            unit.unwrap_or("")
          ),
        );
      }
      return CaseOutcome::passed(
        case_name,
        elapsed,
        "tool call present (arguments partially filled)".to_string(),
      );
    }
  }

  let text = message_text(&response.message);
  CaseOutcome::failed(
    case_name,
    elapsed,
    format!("no tool call found, message={:?}", truncate_text(&text, 160)),
  )
}

fn evaluate_image_response(
  case_name: &str,
  elapsed: u64,
  response: &llm_adapter::core::ImageResponse,
  saved: Result<Vec<PathBuf>>,
) -> CaseOutcome {
  if response.images.is_empty() {
    return CaseOutcome::failed(case_name, elapsed, "no image artifacts returned".to_string());
  }

  let usable_images = response
    .images
    .iter()
    .filter(|image| image.url.is_some() || image.data_base64.is_some())
    .count();
  if usable_images == 0 {
    return CaseOutcome::failed(
      case_name,
      elapsed,
      "image artifacts do not contain url or data".to_string(),
    );
  }

  let first = &response.images[0];
  let location = if let Some(url) = &first.url {
    summarize_url(url)
  } else if let Some(data) = &first.data_base64 {
    format!("inline_base64_chars={}", data.len())
  } else {
    "empty_artifact".to_string()
  };

  let saved_detail = match saved {
    Ok(paths) if paths.is_empty() => "saved=none".to_string(),
    Ok(paths) => format!(
      "saved={}",
      paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(",")
    ),
    Err(error) => format!("save_error={error}"),
  };

  CaseOutcome::passed(
    case_name,
    elapsed,
    format!(
      "images={}, usable={}, first_media_type={}, first={}, {}",
      response.images.len(),
      usable_images,
      first.media_type,
      location,
      saved_detail
    ),
  )
}

fn summarize_url(value: &str) -> String {
  match url::Url::parse(value) {
    Ok(url) => format!("url_host={}", url.host_str().unwrap_or("<none>")),
    Err(_) => "url_present".to_string(),
  }
}

fn persist_image_response(
  output_dir: &Path,
  provider_name: &str,
  case_name: &str,
  response: &llm_adapter::core::ImageResponse,
) -> Result<Vec<PathBuf>> {
  fs::create_dir_all(output_dir)
    .with_context(|| format!("failed to create image output dir {}", output_dir.display()))?;
  let prefix = format!(
    "{}_{}",
    sanitize_path_segment(provider_name),
    sanitize_path_segment(case_name)
  );
  let mut paths = Vec::new();

  for (index, image) in response.images.iter().enumerate() {
    if let Some(data) = &image.data_base64 {
      let path = output_dir.join(format!("{prefix}_{index}.{}", image_extension(&image.media_type)));
      fs::write(
        &path,
        STANDARD.decode(data).context("failed to decode image data_base64")?,
      )
      .with_context(|| format!("failed to write {}", path.display()))?;
      paths.push(path);
      continue;
    }

    if let Some(url) = &image.url {
      if let Some((media_type, data)) = parse_data_url(url) {
        let path = output_dir.join(format!("{prefix}_{index}.{}", image_extension(media_type)));
        fs::write(&path, STANDARD.decode(data).context("failed to decode image data url")?)
          .with_context(|| format!("failed to write {}", path.display()))?;
        paths.push(path);
      } else {
        let path = output_dir.join(format!("{prefix}_{index}.url.txt"));
        fs::write(&path, url).with_context(|| format!("failed to write {}", path.display()))?;
        paths.push(path);
      }
    }
  }

  let manifest_path = output_dir.join(format!("{prefix}.json"));
  let artifacts = response
    .images
    .iter()
    .map(|image| {
      json!({
        "media_type": image.media_type,
        "width": image.width,
        "height": image.height,
        "has_inline_data": image.data_base64.is_some(),
        "has_url": image.url.is_some(),
        "url_kind": image.url.as_deref().map(|url| if url.starts_with("data:") { "data_url" } else { "remote_url" }),
      })
    })
    .collect::<Vec<_>>();
  fs::write(
    &manifest_path,
    serde_json::to_vec_pretty(&json!({
      "provider": provider_name,
      "case": case_name,
      "images": artifacts,
      "text": response.text,
    }))?,
  )
  .with_context(|| format!("failed to write {}", manifest_path.display()))?;
  paths.push(manifest_path);

  Ok(paths)
}

fn parse_data_url(value: &str) -> Option<(&str, &str)> {
  let payload = value.strip_prefix("data:")?;
  let (metadata, data) = payload.split_once(',')?;
  let mut parts = metadata.split(';');
  let media_type = parts.next().filter(|value| !value.is_empty())?;
  parts
    .any(|part| part.eq_ignore_ascii_case("base64"))
    .then_some((media_type, data))
}

fn image_extension(media_type: &str) -> &'static str {
  match media_type {
    "image/jpeg" | "image/jpg" => "jpg",
    "image/webp" => "webp",
    "image/png" => "png",
    _ => "img",
  }
}

fn sanitize_path_segment(value: &str) -> String {
  value
    .chars()
    .map(|ch| {
      if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
        ch
      } else {
        '_'
      }
    })
    .collect()
}

#[derive(Debug, Clone)]
struct CaseOutcome {
  name: String,
  status: CaseStatus,
  duration_ms: Option<u64>,
  detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseStatus {
  Passed,
  Failed,
  Skipped,
}

impl CaseOutcome {
  fn passed(name: &str, duration_ms: u64, detail: String) -> Self {
    Self {
      name: name.to_string(),
      status: CaseStatus::Passed,
      duration_ms: Some(duration_ms),
      detail,
    }
  }

  fn failed(name: &str, duration_ms: u64, detail: String) -> Self {
    Self {
      name: name.to_string(),
      status: CaseStatus::Failed,
      duration_ms: Some(duration_ms),
      detail,
    }
  }

  fn skipped(name: &str, detail: impl Into<String>) -> Self {
    Self {
      name: name.to_string(),
      status: CaseStatus::Skipped,
      duration_ms: None,
      detail: detail.into(),
    }
  }
}

#[derive(Debug, Clone)]
struct ProviderReport {
  provider_name: String,
  model: String,
  cases: Vec<CaseOutcome>,
}

impl ProviderReport {
  fn new(provider_name: &str, model: String) -> Self {
    Self {
      provider_name: provider_name.to_string(),
      model,
      cases: Vec::new(),
    }
  }

  fn push_case(&mut self, outcome: CaseOutcome) {
    self.cases.push(outcome);
  }

  fn has_failures(&self) -> bool {
    self.cases.iter().any(|case| case.status == CaseStatus::Failed)
  }
}

#[derive(Debug, Clone)]
struct CompatibilityReport {
  providers: Vec<ProviderReport>,
  provider_passed: usize,
  provider_failed: usize,
  provider_skipped: usize,
  total_passed: usize,
  total_failed: usize,
  total_skipped: usize,
}

impl CompatibilityReport {
  fn from_reports(providers: Vec<ProviderReport>) -> Self {
    let mut provider_passed = 0usize;
    let mut provider_failed = 0usize;
    let mut provider_skipped = 0usize;
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut total_skipped = 0usize;

    for provider in &providers {
      let mut provider_failed_any = false;
      let mut provider_all_skipped = true;
      for case in &provider.cases {
        match case.status {
          CaseStatus::Passed => {
            total_passed += 1;
            provider_all_skipped = false;
          }
          CaseStatus::Failed => {
            total_failed += 1;
            provider_failed_any = true;
            provider_all_skipped = false;
          }
          CaseStatus::Skipped => {
            total_skipped += 1;
          }
        }
      }

      if provider_failed_any {
        provider_failed += 1;
      } else if provider_all_skipped {
        provider_skipped += 1;
      } else {
        provider_passed += 1;
      }
    }

    Self {
      providers,
      provider_passed,
      provider_failed,
      provider_skipped,
      total_passed,
      total_failed,
      total_skipped,
    }
  }

  fn print(&self) {
    println!("=== LLM Compatibility Report ===");
    for provider in &self.providers {
      println!();
      println!("Provider: {} ({})", provider.provider_name, provider.model);
      for case in &provider.cases {
        let status = match case.status {
          CaseStatus::Passed => "PASS",
          CaseStatus::Failed => "FAIL",
          CaseStatus::Skipped => "SKIP",
        };
        match case.duration_ms {
          Some(ms) => println!("  [{}] {} ({} ms) - {}", status, case.name, ms, case.detail),
          None => println!("  [{}] {} - {}", status, case.name, case.detail),
        }
      }
    }

    println!();
    println!("=== Summary ===");
    println!(
      "Providers: {} passed, {} failed, {} skipped",
      self.provider_passed, self.provider_failed, self.provider_skipped
    );
    println!(
      "Cases: {} passed, {} failed, {} skipped",
      self.total_passed, self.total_failed, self.total_skipped
    );
  }
}
