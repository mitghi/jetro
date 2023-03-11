//! Module containing string formater for built-in #format function.

use crate::context::FormatOp;
use dynfmt::{Format, SimpleCurlyFormat};
use serde_json::Value;

pub(crate) trait KeyFormater {
    fn format(&self, format: &str, value: &Value, keys: &Vec<String>) -> Option<String>;
    fn eval(&self, target_format: &FormatOp, value: &Value) -> Option<Value>;
}

struct FormatImpl;

impl FormatImpl {
    fn format(&self, format: &str, value: &Value, keys: &Vec<String>) -> Option<String> {
        let mut values: Vec<&Value> = vec![];
        for key in keys.iter() {
            match value.get(&key) {
                Some(result) => values.push(result),
                _ => {}
            };
        }
        match SimpleCurlyFormat.format(format, values) {
            Ok(result) => Some(result.into()),
            _ => None,
        }
    }

    fn eval(&self, target_format: &FormatOp, value: &Value) -> Option<Value> {
        let FormatOp::FormatString {
            format: ref fmt,
            arguments: ref args,
            ref alias,
        } = target_format;

        let output: Option<String> = self.format(&fmt, &value, args);

        match output {
            Some(output) => match &value {
                Value::Object(ref obj) => {
                    let mut result = obj.clone();
                    result.insert(alias.to_string(), Value::String(output));
                    return Some(Value::Object(result));
                }
                _ => {}
            },
            _ => {}
        };
        return None;
    }
}

impl KeyFormater for FormatImpl {
    fn format(&self, format: &str, value: &Value, keys: &Vec<String>) -> Option<String> {
        self.format(format, value, keys)
    }

    fn eval(&self, target_format: &FormatOp, value: &Value) -> Option<Value> {
        self.eval(&target_format, &value)
    }
}

pub(crate) fn default() -> Box<impl KeyFormater> {
    Box::new(FormatImpl)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_format_impl() {
        let data = serde_json::json!({
            "name": "mr snuggle",
            "alias": "jetro",
        });

        let keys = vec!["name".to_string(), "alias".to_string()];

        let format_impl: Box<dyn KeyFormater> = Box::new(FormatImpl);

        assert_eq!(
            format_impl.format("{}_{}".as_ref(), &data, &keys),
            Some("mr snuggle_jetro".to_string())
        );
    }
}
