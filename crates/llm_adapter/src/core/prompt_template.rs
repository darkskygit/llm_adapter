use serde_json::{Number, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateToken {
  Text(String),
  Variable(String),
  Section { name: String, children: Vec<TemplateToken> },
}

pub fn parse_template(template: &str) -> Result<Vec<TemplateToken>, String> {
  parse_template_until(template, &mut 0, None)
}

pub fn template_uses_key(tokens: &[TemplateToken], key: &str) -> bool {
  tokens.iter().any(|token| match token {
    TemplateToken::Variable(name) => name == key,
    TemplateToken::Section { name, children } => name == key || template_uses_key(children, key),
    TemplateToken::Text(_) => false,
  })
}

pub fn collect_template_keys_in_order(tokens: &[TemplateToken], keys: &mut Vec<String>) {
  for token in tokens {
    match token {
      TemplateToken::Variable(name) => {
        if name != "." && !keys.contains(name) {
          keys.push(name.clone());
        }
      }
      TemplateToken::Section { name, children } => {
        if name != "." && !keys.contains(name) {
          keys.push(name.clone());
        }
        collect_template_keys_in_order(children, keys);
      }
      TemplateToken::Text(_) => {}
    }
  }
}

pub fn render_tokens(tokens: &[TemplateToken], context_stack: &[&Value]) -> String {
  let mut rendered = String::new();

  for token in tokens {
    match token {
      TemplateToken::Text(text) => rendered.push_str(text),
      TemplateToken::Variable(name) => rendered.push_str(&stringify_value(resolve_value(name, context_stack))),
      TemplateToken::Section { name, children } => {
        let Some(value) = resolve_value(name, context_stack) else {
          continue;
        };
        match value {
          Value::Bool(true) => rendered.push_str(&render_tokens(children, context_stack)),
          Value::Array(items) => {
            for item in items {
              let mut next_stack = context_stack.to_vec();
              next_stack.push(item);
              rendered.push_str(&render_tokens(children, &next_stack));
            }
          }
          Value::Object(_) => {
            let mut next_stack = context_stack.to_vec();
            next_stack.push(value);
            rendered.push_str(&render_tokens(children, &next_stack));
          }
          Value::String(text) if !text.is_empty() => {
            let mut next_stack = context_stack.to_vec();
            next_stack.push(value);
            rendered.push_str(&render_tokens(children, &next_stack));
          }
          Value::Number(number) if is_truthy_number(number) => {
            let mut next_stack = context_stack.to_vec();
            next_stack.push(value);
            rendered.push_str(&render_tokens(children, &next_stack));
          }
          _ => {}
        }
      }
    }
  }

  rendered
}

pub fn is_truthy_number(number: &Number) -> bool {
  number
    .as_f64()
    .map(|value| value != 0.0)
    .or_else(|| number.as_i64().map(|value| value != 0))
    .or_else(|| number.as_u64().map(|value| value != 0))
    .unwrap_or(true)
}

pub fn value_to_warning_text(value: &Value) -> String {
  match value {
    Value::String(text) => text.clone(),
    Value::Array(items) => items.first().map(value_to_warning_text).unwrap_or_default(),
    Value::Null => String::new(),
    _ => value.to_string(),
  }
}

fn parse_template_until(
  template: &str,
  cursor: &mut usize,
  closing_section: Option<&str>,
) -> Result<Vec<TemplateToken>, String> {
  let mut tokens = Vec::new();

  while *cursor < template.len() {
    let remainder = &template[*cursor..];
    let Some(open_offset) = remainder.find("{{") else {
      tokens.push(TemplateToken::Text(remainder.to_string()));
      *cursor = template.len();
      break;
    };

    if open_offset > 0 {
      tokens.push(TemplateToken::Text(remainder[..open_offset].to_string()));
    }
    *cursor += open_offset + 2;

    let closing = template[*cursor..]
      .find("}}")
      .ok_or_else(|| "Unclosed mustache tag".to_string())?;
    let raw_tag = template[*cursor..*cursor + closing].trim().to_string();
    *cursor += closing + 2;

    if let Some(section_name) = raw_tag.strip_prefix('#') {
      let section_name = section_name.trim().to_string();
      let children = parse_template_until(template, cursor, Some(&section_name))?;
      tokens.push(TemplateToken::Section {
        name: section_name,
        children,
      });
      continue;
    }

    if let Some(section_name) = raw_tag.strip_prefix('/') {
      let section_name = section_name.trim();
      if closing_section == Some(section_name) {
        return Ok(tokens);
      }
      return Err(format!("Unexpected closing section `{section_name}`"));
    }

    if raw_tag.starts_with('!') {
      continue;
    }

    tokens.push(TemplateToken::Variable(raw_tag));
  }

  if let Some(section_name) = closing_section {
    return Err(format!("Unclosed section `{section_name}`"));
  }

  Ok(tokens)
}

fn resolve_value<'a>(name: &str, context_stack: &'a [&Value]) -> Option<&'a Value> {
  if name == "." {
    return context_stack.last().copied();
  }

  for context in context_stack.iter().rev() {
    if let Some(value) = lookup_path(context, name) {
      return Some(value);
    }
  }

  None
}

fn lookup_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
  let mut current = value;
  for segment in path.split('.') {
    current = current.as_object()?.get(segment)?;
  }
  Some(current)
}

fn stringify_value(value: Option<&Value>) -> String {
  match value {
    Some(Value::String(text)) => text.clone(),
    Some(Value::Number(number)) => number.to_string(),
    Some(Value::Bool(boolean)) => boolean.to_string(),
    Some(Value::Array(items)) => items.iter().map(js_stringify_value).collect::<Vec<_>>().join(","),
    Some(Value::Object(_)) => "[object Object]".to_string(),
    _ => String::new(),
  }
}

fn js_stringify_value(value: &Value) -> String {
  match value {
    Value::String(text) => text.clone(),
    Value::Number(number) => number.to_string(),
    Value::Bool(boolean) => boolean.to_string(),
    Value::Array(items) => items.iter().map(js_stringify_value).collect::<Vec<_>>().join(","),
    Value::Object(_) => "[object Object]".to_string(),
    Value::Null => String::new(),
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{parse_template, render_tokens, template_uses_key};

  #[test]
  fn renders_sections_and_current_item() {
    let tokens = parse_template("{{#links}}- {{.}}\n{{/links}}").unwrap();
    let rendered = render_tokens(&tokens, &[&json!({ "links": ["a", "b"] })]);

    assert_eq!(rendered, "- a\n- b\n");
  }

  #[test]
  fn follows_js_like_stringification() {
    let tokens = parse_template("{{tags}}|{{obj}}").unwrap();
    let rendered = render_tokens(&tokens, &[&json!({ "tags": ["a", "b"], "obj": { "x": 1 } })]);

    assert_eq!(rendered, "a,b|[object Object]");
  }

  #[test]
  fn detects_used_keys() {
    let tokens = parse_template("{{#content}}{{title}}{{/content}}").unwrap();

    assert!(template_uses_key(&tokens, "content"));
    assert!(template_uses_key(&tokens, "title"));
    assert!(!template_uses_key(&tokens, "missing"));
  }
}
