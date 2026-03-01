use serde_json::Value;

pub(crate) fn parse_json_string(text: &str) -> Value {
  serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

pub(crate) fn parse_json(value: Value) -> Value {
  match value {
    Value::String(text) => parse_json_string(&text),
    other => other,
  }
}

pub(crate) fn parse_json_ref(value: &Value) -> Value {
  match value {
    Value::String(text) => parse_json_string(text),
    other => other.clone(),
  }
}

pub(crate) fn stringify_json(value: &Value) -> String {
  match value {
    Value::String(text) => text.clone(),
    _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
  }
}

fn value_as_u64(value: &Value) -> Option<u64> {
  match value {
    Value::Number(number) => number.as_u64(),
    Value::String(text) => text.parse::<u64>().ok(),
    _ => None,
  }
}

pub(crate) fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
  value.get(key).and_then(Value::as_str)
}

pub(crate) fn get_str_or<'a>(value: &'a Value, key: &str, default: &'a str) -> &'a str {
  get_str(value, key).unwrap_or(default)
}

pub(crate) fn get_u64(value: &Value, key: &str) -> Option<u64> {
  value.get(key).and_then(value_as_u64)
}

pub(crate) fn get_u32(value: &Value, key: &str) -> Option<u32> {
  get_u64(value, key).and_then(|raw| u32::try_from(raw).ok())
}

pub(crate) fn get_u32_or(value: &Value, key: &str, default: u32) -> u32 {
  get_u32(value, key).unwrap_or(default)
}

pub(crate) fn get_nested_value<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
  let mut current = value;
  for segment in path {
    current = current.get(*segment)?;
  }
  Some(current)
}

pub(crate) fn get_nested_u64(value: &Value, path: &[&str]) -> Option<u64> {
  get_nested_value(value, path).and_then(value_as_u64)
}

pub(crate) fn get_cached_tokens(value: &Value, detail_keys: &[&str]) -> Option<u32> {
  detail_keys
    .iter()
    .find_map(|key| get_nested_u64(value, &[*key, "cached_tokens"]))
    .and_then(|cached| u32::try_from(cached).ok())
}

pub(crate) fn get_first_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
  keys.iter().find_map(|key| get_str(value, key))
}

pub(crate) fn get_first_str_or<'a>(value: &'a Value, keys: &[&str], default: &'a str) -> &'a str {
  get_first_str(value, keys).unwrap_or(default)
}
