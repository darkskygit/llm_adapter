# llm_adapter

A small Rust library for adapting multiple LLM provider APIs into one internal request/response model.

`llm_adapter` gives you:

- A provider-neutral core model (`CoreRequest`, `CoreResponse`, `StreamEvent`)
- Protocol codecs for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini GenerateContent
- Image generation/edit support for GPT Image, Gemini Nano Banana, and Fal
- Streaming SSE parsing and cross-protocol stream rewriting helpers
- Optional middleware and fallback routing building blocks for host applications

## Status

Early-stage crate (`0.2.0`) focused on API shape stability and test coverage.

## Supported Backends

Protocol codecs for:

- OpenAI Chat Completions / Responses / Images / Embeddings
- Gemini GenerateContent / Images / Embeddings
- Anthropic Messages
- Fal image models

Protocol selection is ability-specific:

- `ChatProtocol` for chat/streaming text
- `StructuredProtocol` for schema-constrained responses
- `EmbeddingProtocol` for embeddings
- `RerankProtocol` for reranking
- `ImageProtocol` for image generation/editing

Endpoint shapes for:

- OpenAI Chat Completions / Responses (`BackendRequestLayer::ChatCompletions`/`BackendRequestLayer::ChatCompletionsNoV1` / `BackendRequestLayer::CloudflareWorkersAi` / `BackendRequestLayer::Responses`)
- OpenAI Images (`BackendRequestLayer::OpenaiImages`)
- Google Gemini (`BackendRequestLayer::GeminiApi`/`BackendRequestLayer::GeminiVertex`)
- Anthropic (`BackendRequestLayer::Anthropic`/`BackendRequestLayer::VertexAnthropic`)
- Fal image models (`BackendRequestLayer::Fal`)

## Add To Your Project

```toml
[dependencies]
llm_adapter = { version = "0.2.0" }
```

## Quick Start

```rust
use std::collections::BTreeMap;

use llm_adapter::{
  backend::{
    dispatch_request, BackendConfig, ChatProtocol, ReqwestHttpClient,
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
    response_schema: None,
  };

  let response = dispatch_request(
    &client,
    &config,
    ChatProtocol::OpenaiChatCompletions,
    &request,
  )?;

  println!("id={} finish_reason={}", response.id, response.finish_reason);
  Ok(())
}
```

## Streaming

Use `collect_stream_events` for collect-all behavior, or `dispatch_stream_events_with` for incremental processing:

```rust
use llm_adapter::backend::{dispatch_stream_events_with, BackendError, ChatProtocol};

// inside your function, with `client`, `config`, `request` ready
let mut on_event = |event| {
  println!("{event:?}");
  Ok::<(), BackendError>(())
};

dispatch_stream_events_with(
  &client,
  &config,
  ChatProtocol::OpenaiResponses,
  &request,
  on_event,
)?;
```

## Image Generation And Editing

Use `dispatch_image_request` with `ImageRequest` for image generation and image editing. OpenAI edit requests are encoded as multipart form uploads; Fal image editing requires URL image inputs.

```rust
use std::collections::BTreeMap;

use llm_adapter::{
  backend::{dispatch_image_request, BackendConfig, BackendRequestLayer, ImageProtocol, ReqwestHttpClient},
  core::{ImageOptions, ImageProviderOptions, ImageRequest},
};

fn main() -> Result<(), llm_adapter::backend::BackendError> {
  let client = ReqwestHttpClient::default();
  let config = BackendConfig {
    base_url: "https://api.openai.com".to_string(),
    auth_token: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required"),
    request_layer: Some(BackendRequestLayer::OpenaiImages),
    headers: BTreeMap::new(),
    no_streaming: false,
    timeout_ms: Some(60_000),
  };

  let request = ImageRequest::generate(
    "gpt-image-1".to_string(),
    "Create a compact blue app icon on a white background.".to_string(),
    ImageOptions {
      n: Some(1),
      size: Some("1024x1024".to_string()),
      quality: Some("low".to_string()),
      ..ImageOptions::default()
    },
    ImageProviderOptions::default(),
  );

  let response = dispatch_image_request(&client, &config, ImageProtocol::OpenaiImages, &request)?;
  println!("images={}", response.images.len());
  Ok(())
}
```

For edits, use `ImageRequest::edit` and pass at least one `ImageInput`:

```rust
use llm_adapter::core::{ImageInput, ImageOptions, ImageProviderOptions, ImageRequest};

let request = ImageRequest::edit(
  "gpt-image-1".to_string(),
  "Convert this image into a clean sticker.".to_string(),
  vec![ImageInput::Data {
    data_base64: "...".to_string(),
    media_type: "image/png".to_string(),
    file_name: Some("source.png".to_string()),
  }],
  None,
  ImageOptions::default(),
  ImageProviderOptions::default(),
);
```

Provider-specific image options are represented as a single tagged value on the request, for example
`ImageProviderOptions::Openai(OpenAiImageOptions { ... })` or
`ImageProviderOptions::Fal(FalImageOptions { ... })`.

Provider-specific image options live under the protocol modules:

- `protocol::openai::images::OpenAiImageOptions`
- `protocol::gemini::image::GeminiImageOptions`
- `protocol::fal::options::{FalImageOptions, FalImageSize, FalImageOutputFormat}`

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

The generated config includes `[image]` defaults and disabled image-provider examples for OpenAI Images, Gemini Nano Banana (`gemini-2.5-flash-image`), and Fal image generation/editing. Image outputs are written under `[image].output_dir` as decoded image files plus JSON manifests.

`llm_compat` provider entries use ability-specific fields:

- `chat_protocol = "openai_responses"` for chat-capable providers
- `image_protocol = "openai_images"` for image-only providers
- omit `chat_protocol` or set it to `null` for image-only providers

To run the checked-in Gemini Nano Banana image generation/editing smoke test:

```bash
GEMINI_API_KEY=... cargo run -p llm_adapter --features benchmark-cli --bin llm_compat -- run -c llm-compat.toml -p gemini_nano_banana -v
```

## Development

```bash
cargo test
```

## License

Licensed under [AGPL-3.0-only](./LICENSE).
