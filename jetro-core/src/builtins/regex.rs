use crate::context::EvalError;
use crate::value::Val;
use std::sync::Arc;

fn compile_regex_eval(pat: &str) -> Result<Arc<regex::Regex>, EvalError> {
    crate::builtin_helpers::compile_regex(pat).map_err(EvalError)
}

/// Returns `Val::Bool` indicating whether the full string matches `pat`; returns `None` for non-strings.
#[inline]
pub fn re_match_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_apply`]; propagates regex compilation errors as `EvalError`.
#[inline]
pub fn try_re_match_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(Val::Bool(re.is_match(s))))
}

/// Returns the first substring matching `pat`, or `Val::Null` if no match is found.
#[inline]
pub fn re_match_first_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_first_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_first_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_match_first_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(
        re.find(s)
            .map(|m| Val::Str(Arc::from(m.as_str())))
            .unwrap_or(Val::Null),
    ))
}

/// Returns all non-overlapping substrings matching `pat` as a `StrVec`.
#[inline]
pub fn re_match_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_match_all_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_match_all_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_match_all_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out: Vec<Arc<str>> = re
        .find_iter(s)
        .map(|m| Arc::<str>::from(m.as_str()))
        .collect();
    Ok(Some(Val::str_vec(out)))
}

/// Returns capture groups of the first match as an array, or `Val::Null` if no match.
#[inline]
pub fn re_captures_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_captures_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_captures_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    Ok(Some(match re.captures(s) {
        Some(c) => {
            let mut out: Vec<Val> = Vec::with_capacity(c.len());
            for i in 0..c.len() {
                out.push(
                    c.get(i)
                        .map(|m| Val::Str(Arc::from(m.as_str())))
                        .unwrap_or(Val::Null),
                );
            }
            Val::arr(out)
        }
        None => Val::Null,
    }))
}

/// Returns an array of capture-group arrays for every match of `pat` in the string.
#[inline]
pub fn re_captures_all_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_captures_all_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_captures_all_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_captures_all_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let mut all: Vec<Val> = Vec::new();
    for c in re.captures_iter(s) {
        let mut row: Vec<Val> = Vec::with_capacity(c.len());
        for i in 0..c.len() {
            row.push(
                c.get(i)
                    .map(|m| Val::Str(Arc::from(m.as_str())))
                    .unwrap_or(Val::Null),
            );
        }
        all.push(Val::arr(row));
    }
    Ok(Some(Val::arr(all)))
}

/// Replaces the first occurrence of `pat` in the string with `with`.
#[inline]
pub fn re_replace_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_apply(recv, pat, with).ok().flatten()
}

/// Fallible variant of [`re_replace_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_replace_apply(recv: &Val, pat: &str, with: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out = re.replace(s, with);
    Ok(Some(Val::Str(Arc::from(out.as_ref()))))
}

/// Replaces all non-overlapping occurrences of `pat` in the string with `with`.
#[inline]
pub fn re_replace_all_apply(recv: &Val, pat: &str, with: &str) -> Option<Val> {
    try_re_replace_all_apply(recv, pat, with).ok().flatten()
}

/// Fallible variant of [`re_replace_all_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_replace_all_apply(
    recv: &Val,
    pat: &str,
    with: &str,
) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out = re.replace_all(s, with);
    Ok(Some(Val::Str(Arc::from(out.as_ref()))))
}

/// Splits the string on all matches of `pat`, returning a `StrVec` of tokens.
#[inline]
pub fn re_split_apply(recv: &Val, pat: &str) -> Option<Val> {
    try_re_split_apply(recv, pat).ok().flatten()
}

/// Fallible variant of [`re_split_apply`]; propagates regex compilation errors.
#[inline]
pub fn try_re_split_apply(recv: &Val, pat: &str) -> Result<Option<Val>, EvalError> {
    let Some(s) = recv.as_str_ref() else {
        return Ok(None);
    };
    let re = compile_regex_eval(pat)?;
    let out: Vec<Arc<str>> = re.split(s).map(Arc::<str>::from).collect();
    Ok(Some(Val::str_vec(out)))
}

/// Returns `Val::Bool(true)` when the string contains at least one of the `needles`.
#[inline]
pub fn contains_any_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().any(|n| s.contains(n.as_ref()))))
}

/// Returns `Val::Bool(true)` when the string contains every one of the `needles`.
#[inline]
pub fn contains_all_apply(recv: &Val, needles: &[Arc<str>]) -> Option<Val> {
    let s = recv.as_str_ref()?;
    Some(Val::Bool(needles.iter().all(|n| s.contains(n.as_ref()))))
}

