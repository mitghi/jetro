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
                format!(
                    "invalid jetro expression for {:?}: {}",
                    name_lit.value(),
                    msg
                ),
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
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => s,
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
