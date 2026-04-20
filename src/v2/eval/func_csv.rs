use super::value::Val;
use super::util::val_to_string;

pub fn to_csv(recv: &Val) -> String { csv_impl(recv, ",") }
pub fn to_tsv(recv: &Val) -> String { csv_impl(recv, "\t") }

fn csv_impl(val: &Val, sep: &str) -> String {
    match val {
        Val::Arr(rows) => rows.iter().map(|row| match row {
            Val::Arr(cells) => cells.iter().map(|c| cell(c, sep)).collect::<Vec<_>>().join(sep),
            Val::Obj(m)     => m.values().map(|c| cell(c, sep)).collect::<Vec<_>>().join(sep),
            v               => cell(v, sep),
        }).collect::<Vec<_>>().join("\n"),
        Val::Obj(m) => m.values().map(|c| cell(c, sep)).collect::<Vec<_>>().join(sep),
        v => cell(v, sep),
    }
}

fn cell(v: &Val, sep: &str) -> String {
    match v {
        Val::Str(s) if s.contains(sep) || s.contains('"') || s.contains('\n') => {
            format!("\"{}\"", s.replace('"', "\"\""))
        }
        Val::Str(s) => s.to_string(),
        other       => val_to_string(other),
    }
}
