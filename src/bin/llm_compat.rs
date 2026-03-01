use std::{
  collections::{BTreeMap, HashMap},
  fs,
  path::{Path, PathBuf},
  time::Instant,
};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use llm_adapter::{
  backend::{
    BackendConfig, BackendProtocol, BackendRequestLayer, ReqwestHttpClient, collect_stream_events, dispatch_request,
  },
  core::{CoreContent, CoreMessage, CoreRequest, CoreRole, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

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
        "  {}: model={} protocol={:?} enabled={} requires_auth={} tool_choice_strategy={:?}",
        name,
        provider.model,
        provider.protocol,
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
      expected_token: default_expected_token(),
      magic_number: default_magic_number(),
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
  #[serde(default = "default_protocol")]
  protocol: BackendProtocol,
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
      protocol: BackendProtocol::OpenaiChatCompletions,
      request_layer: None,
      headers: BTreeMap::new(),
      supports_stream: true,
      supports_tools: true,
      max_tokens: Some(128),
      temperature: Some(0.0),
      tool_choice_strategy: None,
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
        protocol: BackendProtocol::OpenaiResponses,
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
        protocol: BackendProtocol::AnthropicMessages,
        request_layer: Some(BackendRequestLayer::Anthropic),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "gemini".to_string(),
      ProviderConfig {
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
        model: "gemini-2.5-flash".to_string(),
        auth_token_env: Some("GEMINI_API_KEY".to_string()),
        requires_auth: true,
        protocol: BackendProtocol::OpenaiChatCompletions,
        request_layer: Some(BackendRequestLayer::ChatCompletions),
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
        request_layer: Some(BackendRequestLayer::ChatCompletionsNoV1),
        ..ProviderConfig::default()
      },
    );
    providers.insert(
      "ollama".to_string(),
      ProviderConfig {
        base_url: "http://localhost:11434".to_string(),
        model: "qwen3:14b".to_string(),
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
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
        protocol: BackendProtocol::OpenaiChatCompletions,
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        requires_auth: false,
        tool_choice_strategy: Some(ToolChoiceStrategy::Required),
        ..ProviderConfig::default()
      },
    );

    Self {
      settings: CompatSettings::default(),
      tests: TestSuiteConfig::default(),
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

fn default_protocol() -> BackendProtocol {
  BackendProtocol::OpenaiChatCompletions
}

struct CompatibilityRunner {
  settings: CompatSettings,
  tests: TestSuiteConfig,
  providers: Vec<(String, ProviderConfig)>,
  client: ReqwestHttpClient,
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
      providers,
      client: ReqwestHttpClient::default(),
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
      let outcome = self.case_basic_text(provider_cfg, &backend);
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.stream_text {
      let outcome = if provider_cfg.supports_stream {
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
      let outcome = self.case_conversation_memory(provider_cfg, &backend);
      report.push_case(outcome);
      if self.settings.fail_fast && report.has_failures() {
        return report;
      }
    }

    if self.tests.tool_call {
      let outcome = if provider_cfg.supports_tools {
        self.case_tool_call(provider_cfg, &backend)
      } else {
        CaseOutcome::skipped("tool_call", "provider marked supports_tools=false")
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
    match dispatch_request(&self.client, backend, provider_cfg.protocol, &request) {
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
    match collect_stream_events(&self.client, backend, provider_cfg.protocol, &request) {
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
    match dispatch_request(&self.client, backend, provider_cfg.protocol, &request) {
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
    match dispatch_request(&self.client, backend, provider_cfg.protocol, &request) {
      Ok(response) => {
        let elapsed = started.elapsed().as_millis() as u64;
        evaluate_tool_call_response(case_name, elapsed, &response)
      }
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
    }
  }
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
