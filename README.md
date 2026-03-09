# llm_adapter

A small Rust library for adapting multiple LLM provider APIs into one internal request/response model.

`llm_adapter` gives you:

- A provider-neutral core model (`CoreRequest`, `CoreResponse`, `StreamEvent`)
- Protocol codecs for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini GenerateContent
- Streaming SSE parsing and cross-protocol stream rewriting helpers
- Optional middleware and fallback routing building blocks for host applications

## Status

Early-stage crate (`0.1.0`) focused on API shape stability and test coverage.

## Supported Backends

Protocol codecs for:

- OpenAI Chat Completions
- OpenAI Responses
- Gemini GenerateContent
- Anthropic Messages

Endpoint shapes for:

- OpenAI Chat Completions / Responses (`BackendRequestLayer::ChatCompletions`/`BackendRequestLayer::ChatCompletionsNoV1` / `BackendRequestLayer::Responses`)
- Google Gemini (`BackendRequestLayer::GeminiApi`/`BackendRequestLayer::GeminiVertex`)
- Anthropic (`BackendRequestLayer::Anthropic`/`BackendRequestLayer::VertexAnthropic`)

## Add To Your Project

```toml
[dependencies]
llm_adapter = { version = "0.1.0" }
```

## Quick Start

```rust
use std::collections::BTreeMap;

use llm_adapter::{
  backend::{
    dispatch_request, BackendConfig, BackendProtocol, ReqwestHttpClient,
  },
  core::{CoreContent, CoreMessage, CoreRequest, CoreRole},
};

fn main() -> Result<(), llm_adapter::backend::BackendError> {
  let client = ReqwestHttpClient::default();

  let config = BackendConfig {
    base_url: "https://api.openai.com".to_string(),
    auth_token: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
    request_layer: None,
    headers: BTreeMap::new(),
    no_streaming: false,
    timeout_ms: Some(15_000),
  };

  let request = CoreRequest {
    model: "gpt-4.1".to_string(),
    messages: vec![CoreMessage {
      role: CoreRole::User,
      content: vec![CoreContent::Text {
        text: "Say hello in one sentence.".to_string(),
      }],
    }],
    stream: false,
    max_tokens: Some(128),
    temperature: Some(0.2),
    tools: vec![],
    tool_choice: None,
    include: None,
    reasoning: None,
  };

  let response = dispatch_request(
    &client,
    &config,
    BackendProtocol::OpenaiChatCompletions,
    &request,
  )?;

  println!("id={} finish_reason={}", response.id, response.finish_reason);
  Ok(())
}
```

## Streaming

Use `collect_stream_events` for collect-all behavior, or `dispatch_stream_events_with` for incremental processing:

```rust
use llm_adapter::backend::{dispatch_stream_events_with, BackendError, BackendProtocol};

// inside your function, with `client`, `config`, `request` ready
let mut on_event = |event| {
  println!("{event:?}");
  Ok::<(), BackendError>(())
};

dispatch_stream_events_with(
  &client,
  &config,
  BackendProtocol::OpenaiResponses,
  &request,
  on_event,
)?;
```

## Fallback Routing and Middleware

This crate exposes reusable orchestration helpers:

- `router::dispatch_with_fallback`
- `router::dispatch_stream_with_fallback`
- `middleware::run_request_middleware_chain`
- `middleware::run_stream_middleware_chain`

They are designed for host apps that want custom retry, fallback, and policy pipelines.

## Benchmark CLI

This repository also ships a benchmark binary, now using `llm_adapter::backend::dispatch_request` instead of manual endpoint requests.

```bash
cargo run --bin llm_benchmark --features benchmark-cli -- config
cargo run --bin llm_benchmark --features benchmark-cli -- run -c llm-benchmark.toml
cargo run --bin llm_benchmark --features benchmark-cli -- prompts -c llm-benchmark.toml
```

Configuration auto-discovery order:

- `llm-benchmark.toml`
- `benchmark.toml`
- `config.toml`

## Compatibility CLI

`llm_compat` provides provider compatibility checks.

```bash
cargo run --bin llm_compat --features benchmark-cli -- config
cargo run --bin llm_compat --features benchmark-cli -- providers -c llm-compat.toml
cargo run --bin llm_compat --features benchmark-cli -- run -c llm-compat.toml
```

## Development

```bash
cargo test
```

## License

Licensed under [AGPL-3.0-only](./LICENSE).
