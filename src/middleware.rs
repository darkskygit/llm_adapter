//! Middleware extension points for host applications.
//!
//! This crate's internal dispatch pipeline does not call these hooks directly.
//! They are exposed as public building blocks for external orchestrators.

use std::{collections::VecDeque, fmt::Write};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::{CoreContent, CoreRequest, CoreRole, StreamEvent};

pub type RequestMiddleware = fn(CoreRequest, &MiddlewareConfig) -> CoreRequest;
pub type StreamMiddleware = fn(StreamEvent, &mut PipelineContext, &MiddlewareConfig) -> Option<StreamEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdditionalPropertiesPolicy {
  #[default]
  Preserve,
  Forbid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaKeywordPolicy {
  #[default]
  Preserve,
  Drop,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MiddlewareConfig {
  #[serde(default)]
  pub additional_properties_policy: AdditionalPropertiesPolicy,
  #[serde(default)]
  pub property_format_policy: SchemaKeywordPolicy,
  #[serde(default)]
  pub property_min_length_policy: SchemaKeywordPolicy,
  #[serde(default)]
  pub array_min_items_policy: SchemaKeywordPolicy,
  #[serde(default)]
  pub array_max_items_policy: SchemaKeywordPolicy,
  #[serde(default)]
  pub max_tokens_cap: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineContext {
  pub citations: Vec<String>,
  pub pattern_buffer: String,
  pub last_chunk_type: Option<String>,
  pending_text_delta: Option<String>,
  pending_reasoning_delta: Option<String>,
  queued_events: VecDeque<StreamEvent>,
}

impl PipelineContext {
  pub fn drain_queued_events(&mut self) -> Vec<StreamEvent> {
    self.queued_events.drain(..).collect()
  }

  pub fn flush_pending_deltas(&mut self) {
    if let Some(text) = self.pending_reasoning_delta.take() {
      self.queued_events.push_back(StreamEvent::ReasoningDelta { text });
    }
    if let Some(text) = self.pending_text_delta.take() {
      self.queued_events.push_back(StreamEvent::TextDelta { text });
    }
  }

  fn flush_pending_text_by_newline(&mut self) {
    flush_pending_delta_by_newline(&mut self.pending_text_delta, into_text_delta, &mut self.queued_events);
  }

  fn flush_pending_reasoning_by_newline(&mut self) {
    flush_pending_delta_by_newline(
      &mut self.pending_reasoning_delta,
      into_reasoning_delta,
      &mut self.queued_events,
    );
  }
}

fn into_text_delta(text: String) -> StreamEvent {
  StreamEvent::TextDelta { text }
}

fn into_reasoning_delta(text: String) -> StreamEvent {
  StreamEvent::ReasoningDelta { text }
}

fn flush_pending_delta_by_newline(
  pending: &mut Option<String>,
  to_event: fn(String) -> StreamEvent,
  queued_events: &mut VecDeque<StreamEvent>,
) {
  const DELTA_FLUSH_THRESHOLD_CHARS: usize = 16;

  let Some(buffer) = pending.take() else {
    return;
  };

  let mut start = 0usize;
  let mut segment_chars = 0usize;
  for (index, ch) in buffer.char_indices() {
    segment_chars += 1;

    let should_flush = ch == '\n' || segment_chars >= DELTA_FLUSH_THRESHOLD_CHARS;
    if should_flush {
      let end = index + ch.len_utf8();
      queued_events.push_back(to_event(buffer[start..end].to_string()));
      start = end;
      segment_chars = 0;
    }
  }

  if start < buffer.len() {
    *pending = Some(buffer[start..].to_string());
  }
}

pub fn run_request_middleware_chain(
  request: CoreRequest,
  config: &MiddlewareConfig,
  chain: &[RequestMiddleware],
) -> CoreRequest {
  chain
    .iter()
    .fold(request, |current, middleware| middleware(current, config))
}

pub fn run_stream_middleware_chain(
  event: StreamEvent,
  context: &mut PipelineContext,
  config: &MiddlewareConfig,
  chain: &[StreamMiddleware],
) -> Vec<StreamEvent> {
  let mut output = Vec::new();
  let mut pending = VecDeque::from([(event, 0usize)]);

  while let Some((event, start_index)) = pending.pop_front() {
    let mut current = Some(event);
    let mut queued_during_event = false;

    for (index, middleware) in chain.iter().enumerate().skip(start_index) {
      let Some(value) = current.take() else {
        break;
      };

      current = middleware(value, context, config);
      let next_index = index + 1;

      for queued in context.drain_queued_events() {
        queued_during_event = true;
        pending.push_back((queued, next_index));
      }
    }

    if let Some(event) = current {
      if queued_during_event {
        pending.push_back((event, chain.len()));
      } else {
        output.push(event);
      }
    }
  }

  output
}

#[must_use]
pub fn normalize_messages(mut request: CoreRequest, _config: &MiddlewareConfig) -> CoreRequest {
  for message in &mut request.messages {
    message.content = normalize_message_content(&message.content);

    if message.role != CoreRole::Tool
      && message
        .content
        .iter()
        .any(|content| matches!(content, CoreContent::ToolResult { .. }))
    {
      message.role = CoreRole::Tool;
    }

    if message.content.is_empty() {
      message.content.push(CoreContent::Text {
        text: "[no content]".to_string(),
      });
    }
  }

  request
}

fn normalize_message_content(content: &[CoreContent]) -> Vec<CoreContent> {
  let mut normalized = Vec::new();
  let mut current_text = String::new();

  for block in content {
    match block {
      CoreContent::Text { text } => {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
          current_text.push_str(trimmed);
        }
      }
      other => {
        if !current_text.is_empty() {
          normalized.push(CoreContent::Text {
            text: std::mem::take(&mut current_text),
          });
        }
        normalized.push(other.clone());
      }
    }
  }

  if !current_text.is_empty() {
    normalized.push(CoreContent::Text { text: current_text });
  }

  normalized
}

#[must_use]
pub fn tool_schema_rewrite(mut request: CoreRequest, config: &MiddlewareConfig) -> CoreRequest {
  for tool in &mut request.tools {
    tool.parameters = rewrite_schema(&tool.parameters, config);
  }
  request
}

#[must_use]
pub fn clamp_max_tokens(mut request: CoreRequest, config: &MiddlewareConfig) -> CoreRequest {
  let cap = config
    .max_tokens_cap
    .unwrap_or_else(|| default_max_tokens_cap(&request.model));

  if let Some(max_tokens) = request.max_tokens {
    request.max_tokens = Some(max_tokens.min(cap));
  }

  request
}

fn default_max_tokens_cap(model: &str) -> u32 {
  let normalized = model.to_ascii_lowercase();

  if normalized.starts_with("gpt-5") {
    return 8192;
  }
  if normalized.starts_with("gpt-4.1") || normalized.starts_with("gpt-4o") || normalized.starts_with('o') {
    return 8192;
  }
  if normalized.starts_with("claude") {
    return 8192;
  }
  if normalized.starts_with("sonar") || normalized.starts_with("morph") {
    return 4096;
  }

  4096
}

fn rewrite_schema(value: &Value, config: &MiddlewareConfig) -> Value {
  match value {
    Value::Object(object) => {
      let value_type = object
        .get("type")
        .and_then(Value::as_str)
        .map(|kind| kind.to_ascii_lowercase());
      let is_object = value_type.as_deref() == Some("object");
      let is_array = value_type.as_deref() == Some("array");

      let mut rewritten = Map::new();
      for (key, child) in object {
        match key.as_str() {
          "format" if matches!(config.property_format_policy, SchemaKeywordPolicy::Drop) => {}
          "minLength" if matches!(config.property_min_length_policy, SchemaKeywordPolicy::Drop) => {}
          "minItems" if is_array && matches!(config.array_min_items_policy, SchemaKeywordPolicy::Drop) => {}
          "maxItems" if is_array && matches!(config.array_max_items_policy, SchemaKeywordPolicy::Drop) => {}
          _ => {
            rewritten.insert(key.clone(), rewrite_schema(child, config));
          }
        }
      }

      if is_object && !rewritten.contains_key("properties") {
        rewritten.insert("properties".to_string(), Value::Object(Map::new()));
      }

      if is_object && matches!(config.additional_properties_policy, AdditionalPropertiesPolicy::Forbid) {
        rewritten.insert("additionalProperties".to_string(), Value::Bool(false));
      }

      Value::Object(rewritten)
    }
    Value::Array(items) => Value::Array(
      items
        .iter()
        .map(|item| rewrite_schema(item, config))
        .collect::<Vec<_>>(),
    ),
    other => other.clone(),
  }
}

pub fn stream_event_normalize(
  event: StreamEvent,
  context: &mut PipelineContext,
  _config: &MiddlewareConfig,
) -> Option<StreamEvent> {
  match event {
    StreamEvent::TextDelta { text } => {
      if let Some(pending) = &mut context.pending_text_delta {
        pending.push_str(&text);
      } else {
        context.flush_pending_deltas();
        context.pending_text_delta = Some(text);
      }
      context.flush_pending_text_by_newline();
      context.last_chunk_type = Some("text-delta".to_string());
      None
    }
    StreamEvent::ReasoningDelta { text } => {
      if let Some(pending) = &mut context.pending_reasoning_delta {
        pending.push_str(&text);
      } else {
        context.flush_pending_deltas();
        context.pending_reasoning_delta = Some(text);
      }
      context.flush_pending_reasoning_by_newline();
      context.last_chunk_type = Some("reasoning-delta".to_string());
      None
    }
    StreamEvent::Done { .. } => {
      context.flush_pending_deltas();
      Some(event)
    }
    _ => {
      context.flush_pending_deltas();
      context.last_chunk_type = Some(event_type_name(&event).to_string());
      Some(event)
    }
  }
}

pub fn citation_indexing(
  event: StreamEvent,
  context: &mut PipelineContext,
  _config: &MiddlewareConfig,
) -> Option<StreamEvent> {
  match event {
    StreamEvent::Citation { index, url } => {
      upsert_citation(&mut context.citations, index, url.clone());
      Some(StreamEvent::Citation { index, url })
    }
    StreamEvent::TextDelta { text } => {
      let combined = format!("{}{}", context.pattern_buffer, text);
      let (parsed, remainder) = parse_citations(&combined, &mut context.citations, &mut context.queued_events, false);
      context.pattern_buffer = remainder;
      Some(StreamEvent::TextDelta { text: parsed })
    }
    StreamEvent::Done { finish_reason, usage } => {
      if !context.pattern_buffer.is_empty() {
        let remaining = std::mem::take(&mut context.pattern_buffer);
        let (parsed, remainder) = parse_citations(&remaining, &mut context.citations, &mut context.queued_events, true);
        if !parsed.is_empty() {
          context.queued_events.push_back(StreamEvent::TextDelta { text: parsed });
        }
        if !remainder.is_empty() {
          context
            .queued_events
            .push_back(StreamEvent::TextDelta { text: remainder });
        }
      }

      Some(StreamEvent::Done { finish_reason, usage })
    }
    _ => Some(event),
  }
}

fn parse_citations(
  input: &str,
  citations: &mut Vec<String>,
  queued_events: &mut VecDeque<StreamEvent>,
  flush_all: bool,
) -> (String, String) {
  let mut output = String::new();
  let mut index = 0;

  while index < input.len() {
    if input[index..].starts_with("([") {
      match parse_wrapped_link(input, index) {
        ParseOutcome::Matched { end, text, url } => {
          let Some(url) = url else {
            if let Some((ch, next_index)) = next_char_at(input, index) {
              output.push(ch);
              index = next_index;
            } else {
              break;
            }
            continue;
          };
          let (citation_index, is_new) = citation_index(citations, &url);
          if is_new {
            queued_events.push_back(StreamEvent::Citation {
              index: citation_index,
              url: url.clone(),
            });
          }
          let _ = text;
          let _ = write!(output, "[^{citation_index}]");
          index = end;
          continue;
        }
        ParseOutcome::NeedMore if !flush_all => {
          return (output, input[index..].to_string());
        }
        ParseOutcome::NeedMore => {
          output.push_str(&input[index..]);
          index = input.len();
          continue;
        }
        ParseOutcome::Failed => {}
      }
    }

    if input[index..].starts_with('[') {
      match parse_bracket_pattern(input, index) {
        ParseOutcome::Matched { end, text, url } => {
          if let Some(url) = url {
            let _ = write!(output, "[{text}]({url})");
          } else if let Ok(value) = text.parse::<usize>() {
            if value > 0 && value <= citations.len() {
              let _ = write!(output, "[^{value}]");
            } else {
              let _ = write!(output, "[{text}]");
            }
          } else {
            let _ = write!(output, "[{text}]");
          }
          index = end;
          continue;
        }
        ParseOutcome::NeedMore if !flush_all => {
          return (output, input[index..].to_string());
        }
        ParseOutcome::NeedMore => {
          output.push_str(&input[index..]);
          index = input.len();
          continue;
        }
        ParseOutcome::Failed => {}
      }
    }

    if let Some((ch, next_index)) = next_char_at(input, index) {
      output.push(ch);
      index = next_index;
    } else {
      break;
    }
  }

  (output, String::new())
}

fn next_char_at(input: &str, start: usize) -> Option<(char, usize)> {
  let ch = input.get(start..)?.chars().next()?;
  Some((ch, start + ch.len_utf8()))
}

fn citation_index(citations: &mut Vec<String>, url: &str) -> (usize, bool) {
  if let Some(position) = citations.iter().position(|citation| citation == url) {
    return (position + 1, false);
  }

  citations.push(url.to_string());
  (citations.len(), true)
}

fn upsert_citation(citations: &mut Vec<String>, index: usize, url: String) {
  if index == 0 {
    return;
  }

  let slot = index - 1;
  if citations.len() <= slot {
    citations.resize(slot + 1, String::new());
  }
  citations[slot] = url;
}

enum ParseOutcome {
  Matched {
    end: usize,
    text: String,
    url: Option<String>,
  },
  NeedMore,
  Failed,
}

fn parse_wrapped_link(input: &str, start: usize) -> ParseOutcome {
  if !input[start..].starts_with("([") {
    return ParseOutcome::Failed;
  }

  match parse_bracket_pattern(input, start + 1) {
    ParseOutcome::Matched { end, text, url } => {
      let Some((ch, next_index)) = next_char_at(input, end) else {
        if end >= input.len() {
          return ParseOutcome::NeedMore;
        }
        return ParseOutcome::Failed;
      };
      if ch != ')' {
        return ParseOutcome::Failed;
      }

      if let Some(url) = url {
        ParseOutcome::Matched {
          end: next_index,
          text,
          url: Some(url),
        }
      } else {
        ParseOutcome::Failed
      }
    }
    ParseOutcome::NeedMore => ParseOutcome::NeedMore,
    ParseOutcome::Failed => ParseOutcome::Failed,
  }
}

fn parse_bracket_pattern(input: &str, start: usize) -> ParseOutcome {
  if !input[start..].starts_with('[') {
    return ParseOutcome::Failed;
  }

  let content_start = start + '['.len_utf8();
  let mut index = content_start;
  let content_end;
  let after_bracket;

  loop {
    let Some((ch, next_index)) = next_char_at(input, index) else {
      return ParseOutcome::NeedMore;
    };
    if ch == '[' {
      return ParseOutcome::Failed;
    }
    if ch == ']' {
      content_end = index;
      after_bracket = next_index;
      break;
    }
    index = next_index;
  }

  let content = input[content_start..content_end].to_string();

  if input.get(after_bracket..).is_some_and(|rest| rest.starts_with('(')) {
    let url_start = after_bracket + '('.len_utf8();
    let mut url_index = url_start;
    let url_end;
    let end;
    loop {
      let Some((ch, next_index)) = next_char_at(input, url_index) else {
        return ParseOutcome::NeedMore;
      };
      if ch == ')' {
        url_end = url_index;
        end = next_index;
        break;
      }
      url_index = next_index;
    }

    let url = input[url_start..url_end].to_string();
    return ParseOutcome::Matched {
      end,
      text: content,
      url: Some(url),
    };
  }

  ParseOutcome::Matched {
    end: after_bracket,
    text: content,
    url: None,
  }
}

fn event_type_name(event: &StreamEvent) -> &'static str {
  match event {
    StreamEvent::MessageStart { .. } => "message-start",
    StreamEvent::TextDelta { .. } => "text-delta",
    StreamEvent::ReasoningDelta { .. } => "reasoning-delta",
    StreamEvent::ToolCallDelta { .. } => "tool-call-delta",
    StreamEvent::ToolCall { .. } => "tool-call",
    StreamEvent::ToolResult { .. } => "tool-result",
    StreamEvent::Citation { .. } => "citation",
    StreamEvent::Usage { .. } => "usage",
    StreamEvent::Done { .. } => "done",
    StreamEvent::Error { .. } => "error",
  }
}

#[must_use]
pub fn merge_stream_events(events: &[StreamEvent]) -> Vec<StreamEvent> {
  events.iter().cloned().fold(Vec::new(), |mut output, current| {
    let mut previous = output.last_mut();
    match (&mut previous, &current) {
      (Some(StreamEvent::TextDelta { text: previous_text }), StreamEvent::TextDelta { text }) => {
        previous_text.push_str(text);
      }
      (Some(StreamEvent::ReasoningDelta { text: previous_text }), StreamEvent::ReasoningDelta { text }) => {
        previous_text.push_str(text);
      }
      (_, StreamEvent::ToolResult { call_id, .. }) => {
        if let Some(index) = output.iter().position(|event| {
          matches!(
            event,
            StreamEvent::ToolCall {
              call_id: existing_call_id,
              ..
            } if existing_call_id == call_id
          )
        }) {
          output[index] = current;
        } else {
          output.push(current);
        }
      }
      _ => output.push(current),
    }
    output
  })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::core::{CoreMessage, CoreToolChoice, CoreToolChoiceMode, CoreToolDefinition};

  fn sample_request() -> CoreRequest {
    let mut request = crate::test_support::sample_request();
    request.stream = true;
    request.messages = vec![CoreMessage {
      role: CoreRole::User,
      content: vec![
        CoreContent::Text {
          text: "  hello ".to_string(),
        },
        CoreContent::Text {
          text: " world ".to_string(),
        },
      ],
    }];
    request.tools = vec![CoreToolDefinition {
      name: "doc_read".to_string(),
      description: Some("Read a document".to_string()),
      parameters: json!({
        "type": "object",
        "properties": {
          "docId": { "type": "string", "format": "uuid", "minLength": 1 }
        },
        "additionalProperties": true
      }),
    }];
    request.tool_choice = Some(CoreToolChoice::Mode(CoreToolChoiceMode::Auto));
    request.include = None;
    request.reasoning = None;
    request
  }

  #[test]
  fn should_normalize_messages() {
    let request = sample_request();
    let normalized = normalize_messages(request, &MiddlewareConfig::default());

    assert_eq!(normalized.messages.len(), 1);
    assert_eq!(
      normalized.messages[0].content,
      vec![CoreContent::Text {
        text: "helloworld".to_string()
      }]
    );
  }

  #[test]
  fn should_rewrite_tool_schema() {
    let request = sample_request();
    let config = MiddlewareConfig {
      additional_properties_policy: AdditionalPropertiesPolicy::Forbid,
      property_format_policy: SchemaKeywordPolicy::Drop,
      property_min_length_policy: SchemaKeywordPolicy::Drop,
      array_min_items_policy: SchemaKeywordPolicy::Preserve,
      array_max_items_policy: SchemaKeywordPolicy::Preserve,
      max_tokens_cap: None,
    };

    let rewritten = tool_schema_rewrite(request, &config);
    assert_eq!(
      rewritten.tools[0].parameters,
      json!({
        "type": "object",
        "properties": {
          "docId": { "type": "string" }
        },
        "additionalProperties": false
      })
    );
  }

  #[test]
  fn should_merge_stream_events() {
    let merged = merge_stream_events(&[
      StreamEvent::TextDelta {
        text: "hello".to_string(),
      },
      StreamEvent::TextDelta {
        text: " world".to_string(),
      },
      StreamEvent::ToolCall {
        call_id: "call_1".to_string(),
        name: "doc_read".to_string(),
        arguments: json!({"docId":"a1"}),
        thought: None,
      },
      StreamEvent::ToolResult {
        call_id: "call_1".to_string(),
        output: json!({"ok":true}),
        is_error: None,
      },
    ]);

    assert_eq!(
      merged,
      vec![
        StreamEvent::TextDelta {
          text: "hello world".to_string(),
        },
        StreamEvent::ToolResult {
          call_id: "call_1".to_string(),
          output: json!({"ok":true}),
          is_error: None,
        },
      ]
    );
  }

  #[test]
  fn should_clamp_max_tokens_cases() {
    struct Case {
      name: &'static str,
      request: CoreRequest,
      config: MiddlewareConfig,
      expected_max_tokens: Option<u32>,
    }

    let cases = vec![
      Case {
        name: "model_default_cap",
        request: CoreRequest {
          model: "sonar-pro".to_string(),
          max_tokens: Some(9000),
          ..sample_request()
        },
        config: MiddlewareConfig::default(),
        expected_max_tokens: Some(4096),
      },
      Case {
        name: "config_cap",
        request: CoreRequest {
          model: "gpt-4.1".to_string(),
          max_tokens: Some(3000),
          ..sample_request()
        },
        config: MiddlewareConfig {
          max_tokens_cap: Some(1024),
          ..MiddlewareConfig::default()
        },
        expected_max_tokens: Some(1024),
      },
    ];

    for case in cases {
      let clamped = clamp_max_tokens(case.request, &case.config);
      assert_eq!(clamped.max_tokens, case.expected_max_tokens, "{}", case.name);
    }
  }

  #[test]
  fn should_parse_citations_across_chunks() {
    let mut context = PipelineContext::default();
    let config = MiddlewareConfig::default();

    let first = citation_indexing(
      StreamEvent::TextDelta {
        text: "Use ([AFFiNE](https://affine.pro".to_string(),
      },
      &mut context,
      &config,
    )
    .unwrap();
    assert_eq!(
      first,
      StreamEvent::TextDelta {
        text: "Use ".to_string(),
      }
    );

    let second = citation_indexing(
      StreamEvent::TextDelta {
        text: ")) and [1]".to_string(),
      },
      &mut context,
      &config,
    )
    .unwrap();
    assert_eq!(
      second,
      StreamEvent::TextDelta {
        text: "[^1] and [^1]".to_string(),
      }
    );
    assert_eq!(context.citations, vec!["https://affine.pro".to_string()]);
  }

  #[test]
  fn should_run_stream_middleware_chain() {
    let config = MiddlewareConfig::default();
    let mut context = PipelineContext::default();

    let chain = [stream_event_normalize as StreamMiddleware, citation_indexing];

    let first = run_stream_middleware_chain(
      StreamEvent::TextDelta {
        text: "[1]".to_string(),
      },
      &mut context,
      &config,
      &chain,
    );
    assert!(first.is_empty());

    let second = run_stream_middleware_chain(
      StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      },
      &mut context,
      &config,
      &chain,
    );

    assert_eq!(
      second,
      vec![
        StreamEvent::TextDelta {
          text: "[1]".to_string(),
        },
        StreamEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        },
      ]
    );
  }

  #[test]
  fn should_flush_text_delta_on_newline_boundary() {
    let config = MiddlewareConfig::default();
    let mut context = PipelineContext::default();
    let chain = [stream_event_normalize as StreamMiddleware];

    let first = run_stream_middleware_chain(
      StreamEvent::TextDelta {
        text: "hello\nworld".to_string(),
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      first,
      vec![StreamEvent::TextDelta {
        text: "hello\n".to_string(),
      }]
    );

    let second = run_stream_middleware_chain(
      StreamEvent::TextDelta {
        text: "!\n".to_string(),
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      second,
      vec![StreamEvent::TextDelta {
        text: "world!\n".to_string(),
      }]
    );

    let third = run_stream_middleware_chain(
      StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      third,
      vec![StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      }]
    );
  }

  #[test]
  fn should_flush_text_delta_on_16_char_threshold() {
    let config = MiddlewareConfig::default();
    let mut context = PipelineContext::default();
    let chain = [stream_event_normalize as StreamMiddleware];

    let first = run_stream_middleware_chain(
      StreamEvent::TextDelta {
        text: "abcdefghijklmnopqrst".to_string(),
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      first,
      vec![StreamEvent::TextDelta {
        text: "abcdefghijklmnop".to_string(),
      }]
    );

    let second = run_stream_middleware_chain(
      StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      second,
      vec![
        StreamEvent::TextDelta {
          text: "qrst".to_string(),
        },
        StreamEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        },
      ]
    );
  }

  #[test]
  fn should_flush_text_delta_on_16_char_threshold_with_utf8() {
    let config = MiddlewareConfig::default();
    let mut context = PipelineContext::default();
    let chain = [stream_event_normalize as StreamMiddleware];

    let first = run_stream_middleware_chain(
      StreamEvent::TextDelta {
        text: "你好世界你好世界你好世界你好世界你".to_string(),
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      first,
      vec![StreamEvent::TextDelta {
        text: "你好世界你好世界你好世界你好世界".to_string(),
      }]
    );

    let second = run_stream_middleware_chain(
      StreamEvent::Done {
        finish_reason: Some("stop".to_string()),
        usage: None,
      },
      &mut context,
      &config,
      &chain,
    );
    assert_eq!(
      second,
      vec![
        StreamEvent::TextDelta {
          text: "你".to_string()
        },
        StreamEvent::Done {
          finish_reason: Some("stop".to_string()),
          usage: None,
        },
      ]
    );
  }
}
