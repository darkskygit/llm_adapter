use std::{
  collections::{BTreeMap, HashMap},
  fs,
  path::{Path, PathBuf},
  sync::Arc,
  thread,
  time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use clap::{Parser, Subcommand};
use llm_adapter::{
  backend::{BackendConfig, BackendRequestLayer, ChatProtocol, DefaultHttpClient, dispatch_request},
  core::{CoreContent, CoreMessage, CoreRequest, CoreRole},
};
use rand::prelude::IndexedRandom;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Parser)]
#[command(name = "llm_benchmark")]
#[command(about = "Local/remote LLM benchmark tool backed by llm_adapter backend APIs")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Run benchmarks
  Run {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Override provider name
    #[arg(short, long)]
    provider: Option<String>,
    /// Override model
    #[arg(short, long)]
    model: Option<String>,
    /// Override prompt name
    #[arg(long)]
    prompt: Option<String>,
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
  },
  /// Generate an example configuration file
  Config {
    /// Output file path
    #[arg(short, long, value_name = "FILE", default_value = "llm-benchmark.toml")]
    output: PathBuf,
  },
  /// List prompt templates in config
  Prompts {
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
      model,
      prompt,
      verbose,
    } => run_benchmark(
      config.as_ref(),
      provider.as_ref(),
      model.as_ref(),
      prompt.as_ref(),
      *verbose,
    ),
    Commands::Config { output } => generate_config(output),
    Commands::Prompts { config } => list_prompts(config.as_ref()),
  }
}

fn run_benchmark(
  config_path: Option<&PathBuf>,
  provider_override: Option<&String>,
  model_override: Option<&String>,
  prompt_override: Option<&String>,
  verbose_override: bool,
) -> Result<()> {
  let mut config = load_config(config_path)?;

  if let Some(provider) = provider_override {
    config.benchmark.provider = provider.clone();
  }
  if let Some(model) = model_override {
    config.benchmark.model = model.clone();
  }
  if let Some(prompt) = prompt_override {
    config.benchmark.prompt = prompt.clone();
  }
  if verbose_override {
    config.settings.verbose = true;
  }

  let runner = BenchmarkRunner::new(config)?;
  let results = runner.run()?;

  println!("=== Benchmark Summary ===");
  for result in &results {
    println!(
      "Concurrency {}: {:.1} tokens/s, {:.1} req/s, {:.1}ms avg latency",
      result.concurrency, result.tokens_per_second, result.requests_per_second, result.average_latency_ms
    );
  }

  Ok(())
}

fn load_config(config_path: Option<&PathBuf>) -> Result<BenchmarkConfig> {
  if let Some(path) = config_path {
    return read_config(path);
  }

  let default_paths = ["llm-benchmark.toml", "benchmark.toml", "config.toml"];
  for path in default_paths {
    let path_buf = PathBuf::from(path);
    if path_buf.exists() {
      let config = read_config(&path_buf)?;
      println!("Loaded config from: {}", path_buf.display());
      return Ok(config);
    }
  }

  println!("No config file found, using default configuration");
  Ok(BenchmarkConfig::default())
}

fn read_config(path: &Path) -> Result<BenchmarkConfig> {
  let content = fs::read_to_string(path).with_context(|| format!("failed to read config: {}", path.display()))?;
  let config: BenchmarkConfig =
    toml::from_str(&content).with_context(|| format!("invalid TOML in {}", path.display()))?;
  Ok(config)
}

fn generate_config(output_path: &PathBuf) -> Result<()> {
  let config = BenchmarkConfig::default();
  let toml_content = toml::to_string_pretty(&config)?;
  fs::write(output_path, toml_content).with_context(|| format!("failed to write config: {}", output_path.display()))?;
  println!("Generated example configuration file: {}", output_path.display());
  Ok(())
}

fn list_prompts(config_path: Option<&PathBuf>) -> Result<()> {
  let config = load_config(config_path)?;
  let prompt_manager = PromptManager::new(config.prompts);

  println!("Available prompts:");
  let prompt_names = prompt_manager.list_prompts();
  if prompt_names.is_empty() {
    println!("  No prompts configured");
    return Ok(());
  }

  for name in prompt_names {
    let prompt = prompt_manager.get_prompt(name)?;
    println!("  {} - {}", name, prompt.description);
    println!("    Modalities: {:?}", prompt.modalities);
  }

  Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkConfig {
  #[serde(default)]
  pub settings: Settings,
  #[serde(default)]
  pub benchmark: BenchmarkSettings,
  #[serde(default)]
  pub providers: HashMap<String, ProviderConfig>,
  #[serde(default)]
  pub prompts: HashMap<String, PromptTemplate>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Settings {
  #[serde(default)]
  pub user_agent: Option<String>,
  #[serde(default = "default_timeout_seconds")]
  pub timeout_seconds: u64,
  #[serde(default)]
  pub verbose: bool,
}

impl Default for Settings {
  fn default() -> Self {
    Self {
      user_agent: None,
      timeout_seconds: default_timeout_seconds(),
      verbose: false,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkSettings {
  #[serde(default = "default_total_requests")]
  pub total_requests: usize,
  #[serde(default = "default_concurrency_levels")]
  pub concurrency_levels: Vec<usize>,
  #[serde(default = "default_provider_name")]
  pub provider: String,
  #[serde(default = "default_model_name")]
  pub model: String,
  #[serde(default = "default_prompt_name")]
  pub prompt: String,
  #[serde(default)]
  pub temperature: Option<f64>,
  #[serde(default)]
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub test_interval_seconds: Option<u64>,
}

impl Default for BenchmarkSettings {
  fn default() -> Self {
    Self {
      total_requests: default_total_requests(),
      concurrency_levels: default_concurrency_levels(),
      provider: default_provider_name(),
      model: default_model_name(),
      prompt: default_prompt_name(),
      temperature: None,
      max_tokens: None,
      test_interval_seconds: None,
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
  #[serde(rename = "ollama")]
  Ollama {
    base_url: String,
    #[serde(default)]
    auth_token: Option<String>,
    #[serde(default)]
    auth_token_env: Option<String>,
    #[serde(default)]
    protocol: Option<ChatProtocol>,
    #[serde(default)]
    request_layer: Option<BackendRequestLayer>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
  },
  #[serde(rename = "llama_server")]
  LlamaServer {
    base_url: String,
    #[serde(default)]
    auth_token: Option<String>,
    #[serde(default)]
    auth_token_env: Option<String>,
    #[serde(default)]
    protocol: Option<ChatProtocol>,
    #[serde(default)]
    request_layer: Option<BackendRequestLayer>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
  },
  #[serde(rename = "lmstudio")]
  Lmstudio {
    base_url: String,
    #[serde(default)]
    auth_token: Option<String>,
    #[serde(default)]
    auth_token_env: Option<String>,
    #[serde(default)]
    protocol: Option<ChatProtocol>,
    #[serde(default)]
    request_layer: Option<BackendRequestLayer>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
  },
  #[serde(rename = "openai")]
  OpenAI {
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    protocol: Option<ChatProtocol>,
    #[serde(default)]
    request_layer: Option<BackendRequestLayer>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
  },
  #[serde(rename = "anthropic")]
  Anthropic {
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    request_layer: Option<BackendRequestLayer>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
  },
}

impl ProviderConfig {
  fn resolve(&self, provider_name: &str) -> Result<ResolvedProvider> {
    match self {
      ProviderConfig::Ollama {
        base_url,
        auth_token,
        auth_token_env,
        protocol,
        request_layer,
        headers,
      }
      | ProviderConfig::Lmstudio {
        base_url,
        auth_token,
        auth_token_env,
        protocol,
        request_layer,
        headers,
      }
      | ProviderConfig::LlamaServer {
        base_url,
        auth_token,
        auth_token_env,
        protocol,
        request_layer,
        headers,
      } => {
        let auth_token = resolve_token(auth_token, auth_token_env.as_deref(), None)?;
        let protocol = protocol.unwrap_or(ChatProtocol::OpenaiChatCompletions);
        Ok(ResolvedProvider {
          name: provider_name.to_string(),
          protocol,
          backend: BackendConfig {
            base_url: base_url.clone(),
            auth_token,
            request_layer: *request_layer,
            headers: headers.clone(),
            no_streaming: false,
            timeout_ms: None,
          },
        })
      }
      ProviderConfig::OpenAI {
        base_url,
        api_key,
        api_key_env,
        protocol,
        request_layer,
        headers,
      } => {
        let auth_token = resolve_token(api_key, api_key_env.as_deref(), Some("OPENAI_API_KEY"))?;
        if auth_token.is_empty() {
          return Err(anyhow!(
            "provider `{provider_name}` requires api_key/api_key_env/OPENAI_API_KEY",
          ));
        }

        Ok(ResolvedProvider {
          name: provider_name.to_string(),
          protocol: protocol.unwrap_or(ChatProtocol::OpenaiChatCompletions),
          backend: BackendConfig {
            base_url: base_url.clone(),
            auth_token,
            request_layer: *request_layer,
            headers: headers.clone(),
            no_streaming: false,
            timeout_ms: None,
          },
        })
      }
      ProviderConfig::Anthropic {
        base_url,
        api_key,
        api_key_env,
        request_layer,
        headers,
      } => {
        let auth_token = resolve_token(api_key, api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"))?;
        if auth_token.is_empty() {
          return Err(anyhow!(
            "provider `{provider_name}` requires api_key/api_key_env/ANTHROPIC_API_KEY",
          ));
        }

        Ok(ResolvedProvider {
          name: provider_name.to_string(),
          protocol: ChatProtocol::AnthropicMessages,
          backend: BackendConfig {
            base_url: base_url.clone(),
            auth_token,
            request_layer: *request_layer,
            headers: headers.clone(),
            no_streaming: false,
            timeout_ms: None,
          },
        })
      }
    }
  }
}

fn resolve_token(explicit: &Option<String>, env_name: Option<&str>, default_env: Option<&str>) -> Result<String> {
  if let Some(token) = explicit {
    return Ok(token.clone());
  }

  if let Some(env_name) = env_name
    && let Ok(token) = std::env::var(env_name)
  {
    return Ok(token);
  }

  if let Some(env_name) = default_env
    && let Ok(token) = std::env::var(env_name)
  {
    return Ok(token);
  }

  Ok(String::new())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTemplate {
  pub description: String,
  pub messages: Vec<MessageTemplate>,
  #[serde(default)]
  pub modalities: Vec<Modality>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageTemplate {
  pub role: String,
  pub content: ContentTemplate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ContentTemplate {
  Text(String),
  Multimodal(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
  #[serde(rename = "text")]
  Text { text: String },
  #[serde(rename = "image")]
  Image {
    image_url: Option<String>,
    image_path: Option<String>,
  },
  #[serde(rename = "audio")]
  Audio {
    audio_url: Option<String>,
    audio_path: Option<String>,
  },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
  Text,
  Image,
  Audio,
  Video,
}

impl Default for BenchmarkConfig {
  fn default() -> Self {
    let mut providers = HashMap::new();
    providers.insert(
      "ollama".to_string(),
      ProviderConfig::Ollama {
        base_url: "http://localhost:11434".to_string(),
        auth_token: None,
        auth_token_env: None,
        protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
      },
    );
    providers.insert(
      "llama_server".to_string(),
      ProviderConfig::LlamaServer {
        base_url: "http://localhost:8080".to_string(),
        auth_token: None,
        auth_token_env: None,
        protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
      },
    );
    providers.insert(
      "lmstudio".to_string(),
      ProviderConfig::Lmstudio {
        base_url: "http://localhost:1234".to_string(),
        auth_token: None,
        auth_token_env: None,
        protocol: Some(ChatProtocol::OpenaiChatCompletions),
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
      },
    );

    let mut prompts = HashMap::new();
    prompts.insert(
      "ceph_question".to_string(),
      PromptTemplate {
        description: "Question about PVE Ceph cluster".to_string(),
        messages: vec![MessageTemplate {
          role: "user".to_string(),
          content: ContentTemplate::Text(
            "Can a PVE Ceph cluster aggregate storage capacity across nodes? Answer within 1000 words.".to_string(),
          ),
        }],
        modalities: vec![Modality::Text],
      },
    );

    Self {
      settings: Settings {
        user_agent: Some("llm-benchmark/0.1.0".to_string()),
        timeout_seconds: default_timeout_seconds(),
        verbose: false,
      },
      benchmark: BenchmarkSettings {
        total_requests: default_total_requests(),
        concurrency_levels: default_concurrency_levels(),
        provider: default_provider_name(),
        model: default_model_name(),
        prompt: default_prompt_name(),
        temperature: Some(0.7),
        max_tokens: None,
        test_interval_seconds: Some(1),
      },
      providers,
      prompts,
    }
  }
}

fn default_timeout_seconds() -> u64 {
  30
}

fn default_total_requests() -> usize {
  50
}

fn default_concurrency_levels() -> Vec<usize> {
  vec![2, 4, 8]
}

fn default_provider_name() -> String {
  "ollama".to_string()
}

fn default_model_name() -> String {
  "qwen3:14b".to_string()
}

fn default_prompt_name() -> String {
  "ceph_question".to_string()
}

struct PromptManager {
  prompts: HashMap<String, PromptTemplate>,
}

impl PromptManager {
  fn new(prompts: HashMap<String, PromptTemplate>) -> Self {
    Self { prompts }
  }

  fn get_prompt(&self, name: &str) -> Result<&PromptTemplate> {
    if name == "random" {
      let keys: Vec<&String> = self.prompts.keys().collect();
      if keys.is_empty() {
        return Err(anyhow!("no prompts available"));
      }
      let mut rng = rand::rng();
      let selected = keys
        .choose(&mut rng)
        .ok_or_else(|| anyhow!("failed to select random prompt"))?;
      return self
        .prompts
        .get(*selected)
        .ok_or_else(|| anyhow!("randomly selected prompt not found"));
    }

    self
      .prompts
      .get(name)
      .ok_or_else(|| anyhow!("prompt `{name}` not found"))
  }

  fn build_messages(&self, template: &PromptTemplate) -> Result<Vec<CoreMessage>> {
    template
      .messages
      .iter()
      .map(|message| {
        Ok(CoreMessage {
          role: parse_role(&message.role)?,
          content: self.build_content(&message.content)?,
        })
      })
      .collect()
  }

  fn build_content(&self, content_template: &ContentTemplate) -> Result<Vec<CoreContent>> {
    match content_template {
      ContentTemplate::Text(text) => Ok(vec![CoreContent::Text { text: text.clone() }]),
      ContentTemplate::Multimodal(parts) => {
        let mut content = Vec::new();
        for part in parts {
          match part {
            ContentPart::Text { text } => content.push(CoreContent::Text { text: text.clone() }),
            ContentPart::Image { image_url, image_path } => {
              let source = build_image_source(image_url.as_deref(), image_path.as_deref())?;
              content.push(CoreContent::Image { source });
            }
            ContentPart::Audio { audio_url, audio_path } => {
              let source = build_audio_source(audio_url.as_deref(), audio_path.as_deref())?;
              content.push(CoreContent::Audio { source });
            }
          }
        }
        Ok(content)
      }
    }
  }

  fn list_prompts(&self) -> Vec<&str> {
    let mut names: Vec<&str> = self.prompts.keys().map(String::as_str).collect();
    names.sort_unstable();
    names
  }
}

fn parse_role(role: &str) -> Result<CoreRole> {
  match role {
    "system" => Ok(CoreRole::System),
    "user" => Ok(CoreRole::User),
    "assistant" => Ok(CoreRole::Assistant),
    "tool" => Ok(CoreRole::Tool),
    _ => Err(anyhow!("unsupported role `{role}`")),
  }
}

fn build_image_source(image_url: Option<&str>, image_path: Option<&str>) -> Result<serde_json::Value> {
  if let Some(url) = image_url {
    return Ok(json!({ "url": url }));
  }
  if let Some(path) = image_path {
    let data_url = load_image_as_data_url(path)?;
    return Ok(json!({ "url": data_url }));
  }
  Err(anyhow!("image content requires `image_url` or `image_path`"))
}

fn build_audio_source(audio_url: Option<&str>, audio_path: Option<&str>) -> Result<serde_json::Value> {
  if let Some(url) = audio_url {
    return Ok(json!({ "url": url }));
  }
  if let Some(path) = audio_path {
    let bytes = fs::read(path).with_context(|| format!("failed to read audio file: {path}"))?;
    let media_type = detect_audio_mime_type(Path::new(path));
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    return Ok(json!({ "media_type": media_type, "data": encoded }));
  }
  Err(anyhow!("audio content requires `audio_url` or `audio_path`"))
}

fn load_image_as_data_url(path: &str) -> Result<String> {
  let path = Path::new(path);
  let bytes = fs::read(path).with_context(|| format!("failed to read image file: {}", path.display()))?;
  let mime_type = detect_image_mime_type(path);
  let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
  Ok(format!("data:{mime_type};base64,{encoded}"))
}

fn detect_image_mime_type(path: &Path) -> &'static str {
  match path
    .extension()
    .and_then(|ext| ext.to_str())
    .unwrap_or_default()
    .to_ascii_lowercase()
    .as_str()
  {
    "jpg" | "jpeg" => "image/jpeg",
    "png" => "image/png",
    "gif" => "image/gif",
    "webp" => "image/webp",
    _ => "image/jpeg",
  }
}

fn detect_audio_mime_type(path: &Path) -> &'static str {
  match path
    .extension()
    .and_then(|ext| ext.to_str())
    .unwrap_or_default()
    .to_ascii_lowercase()
    .as_str()
  {
    "wav" => "audio/wav",
    "ogg" | "oga" => "audio/ogg",
    "flac" => "audio/flac",
    _ => "audio/mpeg",
  }
}

struct BenchmarkRunner {
  config: BenchmarkConfig,
  provider: ResolvedProvider,
  prompt_manager: PromptManager,
  client: Arc<DefaultHttpClient>,
}

#[derive(Debug, Clone)]
struct ResolvedProvider {
  name: String,
  protocol: ChatProtocol,
  backend: BackendConfig,
}

#[derive(Debug)]
struct BenchmarkResult {
  concurrency: usize,
  total_requests: usize,
  successful_requests: usize,
  failed_requests: usize,
  total_tokens: u32,
  duration_seconds: f64,
  tokens_per_second: f64,
  requests_per_second: f64,
  average_latency_ms: f64,
  errors: Vec<String>,
}

#[derive(Debug)]
struct TestResult {
  success: bool,
  duration_ms: u64,
  tokens: Option<u32>,
  error: Option<String>,
}

impl BenchmarkRunner {
  fn new(config: BenchmarkConfig) -> Result<Self> {
    let provider_config = config
      .providers
      .get(&config.benchmark.provider)
      .ok_or_else(|| anyhow!("provider `{}` not found in config", config.benchmark.provider))?;

    let mut provider = provider_config.resolve(&config.benchmark.provider)?;
    if provider.backend.timeout_ms.is_none() {
      provider.backend.timeout_ms = Some(config.settings.timeout_seconds.saturating_mul(1000));
    }

    let prompt_manager = PromptManager::new(config.prompts.clone());
    let client = Arc::new(DefaultHttpClient::default());

    Ok(Self {
      config,
      provider,
      prompt_manager,
      client,
    })
  }

  fn run(&self) -> Result<Vec<BenchmarkResult>> {
    if self.config.settings.user_agent.is_some() {
      println!("Note: settings.user_agent is ignored by DefaultHttpClient in llm_adapter.");
    }

    println!("Starting benchmark with provider: {}", self.provider.name);
    println!("Protocol: {:?}", self.provider.protocol);
    println!("Model: {}", self.config.benchmark.model);
    println!("Prompt: {}", self.config.benchmark.prompt);
    println!("Total requests per test: {}", self.config.benchmark.total_requests);
    println!();

    let mut results = Vec::new();
    for &concurrency in &self.config.benchmark.concurrency_levels {
      if concurrency == 0 {
        return Err(anyhow!("concurrency level cannot be 0"));
      }

      if let Some(interval) = self.config.benchmark.test_interval_seconds
        && !results.is_empty()
      {
        println!("Waiting {} seconds before next test...", interval);
        thread::sleep(Duration::from_secs(interval));
      }

      let result = self.run_single_test(concurrency)?;
      self.print_result(&result);
      results.push(result);
    }

    Ok(results)
  }

  fn run_single_test(&self, concurrency: usize) -> Result<BenchmarkResult> {
    println!("Running test with concurrency: {}", concurrency);
    let prompt_template = self.prompt_manager.get_prompt(&self.config.benchmark.prompt)?;
    let messages = self.prompt_manager.build_messages(prompt_template)?;

    let request = CoreRequest {
      model: self.config.benchmark.model.clone(),
      messages,
      stream: false,
      max_tokens: self.config.benchmark.max_tokens,
      temperature: self.config.benchmark.temperature,
      tools: vec![],
      tool_choice: None,
      include: None,
      reasoning: None,
      response_schema: None,
    };

    let total_requests = self.config.benchmark.total_requests;
    let requests_per_worker = total_requests / concurrency;
    let remaining_requests = total_requests % concurrency;
    let start_time = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);

    for worker_id in 0..concurrency {
      let worker_requests = if worker_id < remaining_requests {
        requests_per_worker + 1
      } else {
        requests_per_worker
      };
      if worker_requests == 0 {
        continue;
      }

      let client = Arc::clone(&self.client);
      let backend = self.provider.backend.clone();
      let protocol = self.provider.protocol;
      let request = request.clone();
      let verbose = self.config.settings.verbose;

      handles.push(thread::spawn(move || {
        worker_loop(client, backend, protocol, request, worker_requests, worker_id, verbose)
      }));
    }

    let mut successful_requests = 0usize;
    let mut failed_requests = 0usize;
    let mut total_tokens = 0u32;
    let mut total_latency_ms = 0u64;
    let mut errors = Vec::new();

    for handle in handles {
      let results = handle.join().map_err(|_| anyhow!("worker thread panicked"))?;
      for result in results {
        if result.success {
          successful_requests += 1;
          total_latency_ms = total_latency_ms.saturating_add(result.duration_ms);
          if let Some(tokens) = result.tokens {
            total_tokens = total_tokens.saturating_add(tokens);
          }
        } else {
          failed_requests += 1;
          if let Some(error) = result.error {
            errors.push(error);
          }
        }
      }
    }

    let duration_seconds = start_time.elapsed().as_secs_f64();
    let tokens_per_second = if duration_seconds > 0.0 {
      total_tokens as f64 / duration_seconds
    } else {
      0.0
    };
    let requests_per_second = if duration_seconds > 0.0 {
      successful_requests as f64 / duration_seconds
    } else {
      0.0
    };
    let average_latency_ms = if successful_requests > 0 {
      total_latency_ms as f64 / successful_requests as f64
    } else {
      0.0
    };

    Ok(BenchmarkResult {
      concurrency,
      total_requests,
      successful_requests,
      failed_requests,
      total_tokens,
      duration_seconds,
      tokens_per_second,
      requests_per_second,
      average_latency_ms,
      errors,
    })
  }

  fn print_result(&self, result: &BenchmarkResult) {
    println!();
    println!("Concurrency: {}", result.concurrency);
    println!("Success: {}/{}", result.successful_requests, result.total_requests);
    println!("Failed: {}", result.failed_requests);
    println!("Total tokens: {}", result.total_tokens);
    println!("Duration: {:.2} s", result.duration_seconds);
    println!("Token TPS: {:.1} tokens/s", result.tokens_per_second);
    println!("Request RPS: {:.1} requests/s", result.requests_per_second);
    println!("Avg latency: {:.1} ms", result.average_latency_ms);
    if !result.errors.is_empty() {
      println!("Error samples (first 5 of {}):", result.errors.len());
      for (index, error) in result.errors.iter().take(5).enumerate() {
        println!("  {}: {}", index + 1, error);
      }
    }
    println!();
  }
}

fn worker_loop(
  client: Arc<DefaultHttpClient>,
  backend: BackendConfig,
  protocol: ChatProtocol,
  request: CoreRequest,
  num_requests: usize,
  worker_id: usize,
  verbose: bool,
) -> Vec<TestResult> {
  let mut results = Vec::with_capacity(num_requests);
  for index in 0..num_requests {
    if verbose && worker_id == 0 {
      println!("Sending request {}/{}", index + 1, num_requests);
    }

    let start = Instant::now();
    match dispatch_request(client.as_ref(), &backend, protocol, &request) {
      Ok(response) => {
        results.push(TestResult {
          success: true,
          duration_ms: start.elapsed().as_millis() as u64,
          tokens: Some(response.usage.total_tokens),
          error: None,
        });
      }
      Err(error) => {
        results.push(TestResult {
          success: false,
          duration_ms: start.elapsed().as_millis() as u64,
          tokens: None,
          error: Some(error.to_string()),
        });
      }
    }
  }
  results
}
