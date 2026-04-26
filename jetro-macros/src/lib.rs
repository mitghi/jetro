//! Compile-time macros for jetro.
//!
//! Re-exported from the main crate under the `macros` feature flag:
//!
//! ```ignore
//! use jetro::prelude::*;
//! use jetro::{jetro, JetroSchema};
//!
//! let titles = jetro!("$.books.map(title)");
//!
//! #[derive(JetroSchema)]
//! #[expr(titles = "$.books.map(title)")]
//! #[expr(count  = "$.books.len()")]
//! struct BookView;
//! ```

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Attribute, DeriveInput, Expr, ExprAssign, ExprLit, ExprPath, Lit, LitStr,
    Meta, Result, Token,
};

// ── Grammar check ─────────────────────────────────────────────────────────────
//
// The proc-macro crate cannot depend on `jetro` (cycle), so we reuse the pest
// grammar directly via a relative path to the main crate.  This parses but
// discards the tree — we only care that the input is grammatically valid at
// compile time; full AST construction still happens at run time.

mod grammar {
    use pest_derive::Parser;
    #[derive(Parser)]
    #[grammar = "./grammar.pest"]
    pub(crate) struct V2Parser;
}
use grammar::{Rule, V2Parser};
use pest::Parser as _;

// ── jetro! ────────────────────────────────────────────────────────────────────

/// `jetro!("expr")` — lightweight compile-time syntax check + `Expr<Value>`
/// constructor.
///
/// Runs a fast lexical validation (balanced delimiters, closed quotes,
/// non-empty body) at compile time; the full grammar check still happens
/// on first evaluation. Reports errors at the exact macro call-site so
/// typos surface in the usual IDE squiggles.
#[proc_macro]
pub fn jetro(input: TokenStream) -> TokenStream {
    let lit = parse_macro_input!(input as LitStr);
    let src = lit.value();

    if let Err(msg) = grammar_check(&src) {
        return syn::Error::new(lit.span(), msg).to_compile_error().into();
    }

    quote! {
        ::jetro::Expr::<::serde_json::Value>::new(#lit)
            .expect("jetro! produced an expression that failed runtime parse — should be unreachable after compile-time check")
    }
    .into()
}

/// Full PEG check.  Returns a concise error string whose first line points at
/// the offending column, so the user sees a real parse error rather than a
/// vague "syntax error" at compile time.
fn grammar_check(src: &str) -> std::result::Result<(), String> {
    if src.trim().is_empty() {
        return Err("jetro!() expression is empty".into());
    }
    V2Parser::parse(Rule::program, src)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

// ── query! ────────────────────────────────────────────────────────────────────
//
// `query!("$.<chain>.filter(<f>).map(<f>).<sum|avg|count>()")` → emits a
// monomorphic inline byte walker specialised to the query's chain + key
// names + sink op.  Generated code reads raw JSON bytes and returns a
// typed scalar (i64 / f64 / u64) without runtime dispatch.
//
// Coverage (Phase 1): single-step source chain `$.<key>`, sink in
// {sum, count, avg}, optional `filter(<field>)` truthy + `map(<field>)`
// path-only.  Shapes outside coverage compile-error with a hint to
// fall back to `Jetro::collect_val("...")`.

#[proc_macro]
pub fn query(input: TokenStream) -> TokenStream {
    let lit = parse_macro_input!(input as LitStr);
    let src = lit.value();

    if let Err(msg) = grammar_check(&src) {
        return syn::Error::new(lit.span(), msg).to_compile_error().into();
    }

    let plan = match query_classify(&src) {
        Ok(p) => p,
        Err(msg) => {
            return syn::Error::new(lit.span(),
                format!("query!: shape unsupported (Phase 1): {} — fall back to Jetro::collect_val", msg))
                .to_compile_error().into();
        }
    };

    emit_query_fn(plan).into()
}

/// Plan for one query — extracted from the source via lightweight parse.
/// The macro doesn't reuse jetro-core's lowering (cycle), so we parse
/// just enough to recognise the supported single-shape vocabulary.
struct QueryPlan {
    chain: Vec<String>,        // e.g. ["data"]
    pred_field: Option<String>, // filter(<field>) — None if no filter
    map_field: Option<String>,  // map(<field>) — None if no map
    op: AggOp,                  // sum / count / avg
}

#[derive(Clone, Copy)]
enum AggOp { Sum, Count, Avg, Min, Max, Len }

/// Recognise `$.<a>.<b>...[.filter(f)][.map(f)].<sum|count|avg>()`.
/// Returns Err on any deviation.  Leaves the heavy lifting (full
/// AST walk) to runtime — this is just a syntactic match.
fn query_classify(src: &str) -> std::result::Result<QueryPlan, String> {
    // Strip `$.` prefix.
    let s = src.trim();
    let s = s.strip_prefix("$.").ok_or("query must start with `$.`")?;
    // Split at `.` honouring `(...)` nesting.
    let mut parts: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => { depth += 1; cur.push(c); }
            ')' => { depth -= 1; cur.push(c); }
            '.' if depth == 0 => {
                if !cur.is_empty() { parts.push(std::mem::take(&mut cur)); }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() { parts.push(cur); }
    // Last part: aggregate `sum() / count() / len() / avg() / min() / max()`.
    let last = parts.last().ok_or("empty query body")?;
    let op = if let Some(_) = last.strip_suffix("()") {
        let name = &last[..last.len() - 2];
        match name {
            "sum" => AggOp::Sum,
            "count" => AggOp::Count,
            "len" => AggOp::Len,
            "avg" => AggOp::Avg,
            "min" => AggOp::Min,
            "max" => AggOp::Max,
            other => return Err(format!("unsupported aggregate: {}()", other)),
        }
    } else {
        return Err("query must terminate in `.sum() / .count() / .len() / .avg() / .min() / .max()`".into());
    };
    parts.pop();

    // Optional `.map(<field>)` (second from end).
    let mut map_field: Option<String> = None;
    if let Some(p) = parts.last() {
        if let Some(inner) = strip_method(p, "map") {
            if !is_simple_ident(&inner) {
                return Err(format!("map(): only path-only `field` ident supported (got `{}`)", inner));
            }
            map_field = Some(inner);
            parts.pop();
        }
    }

    // Optional `.filter(<field>)` (third from end / now second).
    let mut pred_field: Option<String> = None;
    if let Some(p) = parts.last() {
        if let Some(inner) = strip_method(p, "filter") {
            if !is_simple_ident(&inner) {
                return Err(format!("filter(): only path-only `field` ident supported (got `{}`)", inner));
            }
            pred_field = Some(inner);
            parts.pop();
        }
    }

    // Remaining parts are the chain — must all be plain idents.
    for p in &parts {
        if !is_simple_ident(p) {
            return Err(format!("chain step `{}` not a plain identifier", p));
        }
    }

    Ok(QueryPlan {
        chain: parts,
        pred_field,
        map_field,
        op,
    })
}

fn strip_method<'a>(s: &'a str, name: &str) -> Option<String> {
    let prefix = format!("{}(", name);
    if !s.starts_with(&prefix) { return None; }
    if !s.ends_with(')') { return None; }
    Some(s[prefix.len()..s.len() - 1].trim().to_string())
}

fn is_simple_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().enumerate().all(|(i, c)| {
            if i == 0 { c.is_ascii_alphabetic() || c == '_' }
            else      { c.is_ascii_alphanumeric() || c == '_' }
        })
}

/// Emit the specialised byte walker.  Returns a `Box<dyn Fn(&[u8]) -> Option<R>>`
/// for the appropriate result type R based on the op.  Calling overhead
/// minimal — each call is a hand-rolled walker, no Pipeline lower, no
/// VM dispatch.
fn emit_query_fn(plan: QueryPlan) -> TokenStream2 {
    // Build chain step needles `"<key>":` as byte literals.
    let chain_lits: Vec<proc_macro2::Literal> = plan.chain.iter()
        .map(|k| proc_macro2::Literal::byte_string(k.as_bytes()))
        .collect();

    let pred_needle: Option<proc_macro2::Literal> = plan.pred_field.as_ref().map(|f| {
        let mut n = Vec::with_capacity(f.len() + 4);
        n.push(b'"');
        n.extend_from_slice(f.as_bytes());
        n.push(b'"'); n.push(b':');
        proc_macro2::Literal::byte_string(&n)
    });
    let map_needle: Option<proc_macro2::Literal> = plan.map_field.as_ref().map(|f| {
        let mut n = Vec::with_capacity(f.len() + 4);
        n.push(b'"');
        n.extend_from_slice(f.as_bytes());
        n.push(b'"'); n.push(b':');
        proc_macro2::Literal::byte_string(&n)
    });

    // Result type / accumulator selection.
    let (acc_init, acc_push, acc_finalise, ret_ty) = match plan.op {
        AggOp::Sum => (
            quote!{ let mut acc: i64 = 0i64; },
            quote!{ acc = acc.wrapping_add(v); },
            quote!{ Some(acc) },
            quote!{ i64 },
        ),
        AggOp::Count | AggOp::Len => (
            quote!{ let mut acc: u64 = 0u64; },
            quote!{ acc += 1; let _ = v; },
            quote!{ Some(acc) },
            quote!{ u64 },
        ),
        AggOp::Avg => (
            quote!{ let mut sum: f64 = 0.0; let mut n: u64 = 0; },
            quote!{ sum += v as f64; n += 1; },
            quote!{ if n == 0 { None } else { Some(sum / (n as f64)) } },
            quote!{ f64 },
        ),
        AggOp::Min => (
            quote!{ let mut acc: i64 = i64::MAX; let mut seen = false; },
            quote!{ if !seen || v < acc { acc = v; } seen = true; },
            quote!{ if seen { Some(acc) } else { None } },
            quote!{ i64 },
        ),
        AggOp::Max => (
            quote!{ let mut acc: i64 = i64::MIN; let mut seen = false; },
            quote!{ if !seen || v > acc { acc = v; } seen = true; },
            quote!{ if seen { Some(acc) } else { None } },
            quote!{ i64 },
        ),
    };

    // Filter + map needle reads.
    let pred_block = match pred_needle {
        Some(needle) => quote! {
            // Pred-field truthy check via memmem `"<field>":` then
            // bool-like parse.  Falsy on missing.
            const PRED_NEEDLE: &[u8] = #needle;
            let pred_pos = ::memchr::memmem::find(entry_range, PRED_NEEDLE);
            let passes = match pred_pos {
                Some(p) => {
                    let val_start = p + PRED_NEEDLE.len();
                    let val = &entry_range[val_start..];
                    let val = trim_ws_left(val);
                    parse_truthy(val)
                }
                None => false,
            };
            if !passes { i = entry_advance(input, entry_end); continue; }
        },
        None => quote! {},
    };

    let map_block = match map_needle {
        Some(needle) => quote! {
            // Map-field int read via memmem.
            const MAP_NEEDLE: &[u8] = #needle;
            let map_pos = ::memchr::memmem::find(entry_range, MAP_NEEDLE);
            if let Some(p) = map_pos {
                let val_start = p + MAP_NEEDLE.len();
                let val = &entry_range[val_start..];
                let val = trim_ws_left(val);
                if let Some(v) = parse_int(val) {
                    #acc_push
                }
            }
        },
        None => quote! {
            // No map — accumulate entry count or skip.
            let v: i64 = 0;
            #acc_push
        },
    };

    // Generated walker — fully inlined, no runtime dispatch.
    let chain_walk = quote! {
        #(
            const CHAIN_KEY: &[u8] = #chain_lits;
            i = walk_to_field_then(input, i, CHAIN_KEY)?;
            i = trim_ws(input, i);
        )*
        if input.get(i) != Some(&b'[') { return None; }
    };

    quote! {{
        // Anonymous inline fn — monomorphic to this query.  Caller
        // invokes by call-site reference.  Helpers re-defined locally
        // to avoid jetro-core dep cycle (macro crate is leaf).
        #[inline]
        fn _query(input: &[u8]) -> Option<#ret_ty> {
            // Local helpers — reflected from jetro_core::bytescan but
            // inlined here so the macro has zero runtime dep.
            #[inline]
            fn trim_ws(b: &[u8], mut i: usize) -> usize {
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                i
            }
            #[inline]
            fn trim_ws_left(b: &[u8]) -> &[u8] {
                let mut i = 0;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                &b[i..]
            }
            #[inline]
            fn skip_string(b: &[u8], mut i: usize) -> Option<usize> {
                if b.get(i) != Some(&b'"') { return None; }
                i += 1;
                while i < b.len() {
                    match b[i] {
                        b'"' => return Some(i + 1),
                        b'\\' => i += 2,
                        _ => i += 1,
                    }
                }
                None
            }
            fn skip_value(b: &[u8], mut i: usize) -> Option<usize> {
                i = trim_ws(b, i);
                if i >= b.len() { return None; }
                match b[i] {
                    b'"' => skip_string(b, i),
                    b'{' | b'[' => {
                        let opener = b[i];
                        let closer = if opener == b'{' { b'}' } else { b']' };
                        let mut depth: i32 = 1;
                        i += 1;
                        while i < b.len() && depth > 0 {
                            match ::memchr::memchr3(b'"', opener, closer, &b[i..]) {
                                None => return None,
                                Some(off) => {
                                    i += off;
                                    match b[i] {
                                        b'"' => { i = skip_string(b, i)?; }
                                        c if c == opener => { depth += 1; i += 1; }
                                        c if c == closer => { depth -= 1; i += 1; }
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        }
                        if depth == 0 { Some(i) } else { None }
                    }
                    _ => {
                        while i < b.len() && !matches!(b[i],
                            b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                        { i += 1; }
                        Some(i)
                    }
                }
            }
            fn find_field(b: &[u8], obj_start: usize, key: &[u8]) -> Option<usize> {
                let mut i = obj_start;
                if b.get(i) != Some(&b'{') { return None; }
                i += 1;
                loop {
                    i = trim_ws(b, i);
                    if i >= b.len() || b[i] == b'}' { return None; }
                    if b.get(i) != Some(&b'"') { return None; }
                    let key_end = skip_string(b, i)?;
                    let k = &b[i + 1 .. key_end - 1];
                    i = trim_ws(b, key_end);
                    if b.get(i) != Some(&b':') { return None; }
                    i = trim_ws(b, i + 1);
                    if k == key { return Some(i); }
                    i = skip_value(b, i)?;
                    i = trim_ws(b, i);
                    if i < b.len() && b[i] == b',' { i += 1; }
                }
            }
            fn walk_to_field_then(b: &[u8], i: usize, key: &[u8]) -> Option<usize> {
                let i = trim_ws(b, i);
                find_field(b, i, key)
            }
            #[inline]
            fn entry_advance(b: &[u8], next: usize) -> usize {
                let mut i = trim_ws(b, next);
                if i < b.len() && b[i] == b',' { i = trim_ws(b, i + 1); }
                i
            }
            #[inline]
            fn parse_int(b: &[u8]) -> Option<i64> {
                if b.is_empty() { return None; }
                let neg = b[0] == b'-';
                let s = if neg { &b[1..] } else { b };
                let mut acc: i64 = 0;
                let mut any = false;
                for &c in s {
                    if !c.is_ascii_digit() { break; }
                    acc = acc.wrapping_mul(10).wrapping_add((c - b'0') as i64);
                    any = true;
                }
                if !any { return None; }
                Some(if neg { -acc } else { acc })
            }
            #[inline]
            fn parse_truthy(b: &[u8]) -> bool {
                if b.starts_with(b"true")  { return true; }
                if b.starts_with(b"false") { return false; }
                if b.starts_with(b"null")  { return false; }
                if let Some(n) = parse_int(b) { return n != 0; }
                false
            }

            let mut i: usize = 0;
            i = trim_ws(input, i);
            #chain_walk
            // Iterate top-level Array entries.
            i = trim_ws(input, i + 1);
            #acc_init
            while i < input.len() && input[i] != b']' {
                let entry_end = skip_value(input, i)?;
                let entry_range = &input[i + 1 .. entry_end - 1];
                #pred_block
                #map_block
                i = entry_advance(input, entry_end);
            }
            #acc_finalise
        }
        _query as fn(&[u8]) -> Option<#ret_ty>
    }}
}

// ── #[derive(JetroSchema)] ────────────────────────────────────────────────────

/// Derive macro: collects `#[expr(name = "source")]` attributes on the
/// type and emits an `impl JetroSchema` with:
///
/// - `const EXPRS: &'static [(&'static str, &'static str)]` — pairs
/// - `fn exprs() -> &'static [(&'static str, &'static str)]`
/// - `fn names() -> &'static [&'static str]`
///
/// The runtime counterpart trait is defined in the main `jetro` crate so
/// it stays usable without enabling the `macros` feature.
#[proc_macro_derive(JetroSchema, attributes(expr))]
pub fn derive_jetro_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;
    let (impl_g, ty_g, where_g) = input.generics.split_for_impl();

    let pairs = match collect_expr_attrs(&input.attrs) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    if pairs.is_empty() {
        return syn::Error::new_spanned(
            ident,
            "JetroSchema requires at least one #[expr(name = \"src\")] attribute",
        )
        .to_compile_error()
        .into();
    }

    // Compile-time grammar check on every source.
    for (name_lit, src_lit) in &pairs {
        let src = src_lit.value();
        if let Err(msg) = grammar_check(&src) {
            return syn::Error::new(
                src_lit.span(),
                format!("invalid jetro expression for {:?}: {}", name_lit.value(), msg),
            )
            .to_compile_error()
            .into();
        }
    }

    let name_strs: Vec<&LitStr> = pairs.iter().map(|(n, _)| n).collect();
    let src_strs: Vec<&LitStr> = pairs.iter().map(|(_, s)| s).collect();

    let out: TokenStream2 = quote! {
        impl #impl_g ::jetro::JetroSchema for #ident #ty_g #where_g {
            const EXPRS: &'static [(&'static str, &'static str)] = &[
                #( (#name_strs, #src_strs), )*
            ];

            fn exprs() -> &'static [(&'static str, &'static str)] {
                Self::EXPRS
            }

            fn names() -> &'static [&'static str] {
                &[ #( #name_strs, )* ]
            }
        }
    };
    out.into()
}

// Each `#[expr(name = "src")]` line contributes one pair. Multiple
// key/value pairs inside one attribute are allowed — we flatten.
fn collect_expr_attrs(attrs: &[Attribute]) -> Result<Vec<(LitStr, LitStr)>> {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("expr") {
            continue;
        }
        let pairs: ExprAttr = attr.parse_args()?;
        for (n, s) in pairs.0 {
            out.push((n, s));
        }
    }
    Ok(out)
}

struct ExprAttr(Vec<(LitStr, LitStr)>);

impl Parse for ExprAttr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut pairs = Vec::new();
        while !input.is_empty() {
            let assign: ExprAssign = input.parse()?;
            // LHS: bare ident path, e.g. `titles`
            let name = match *assign.left {
                Expr::Path(ExprPath { path, .. }) => path
                    .get_ident()
                    .cloned()
                    .ok_or_else(|| syn::Error::new_spanned(&path, "expected bare identifier"))?,
                other => {
                    return Err(syn::Error::new_spanned(other, "expected `name = \"src\"`"));
                }
            };
            // RHS: string literal.
            let rhs = match *assign.right {
                Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) => s,
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "right-hand side must be a string literal",
                    ));
                }
            };
            pairs.push((LitStr::new(&name.to_string(), name.span()), rhs));

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            } else {
                break;
            }
        }
        Ok(Self(pairs))
    }
}

// Silence unused-import warnings when some syn feature paths are not used.
#[allow(dead_code)]
fn _unused_to_keep_warnings_quiet() {
    let _: Option<Meta> = None;
}
