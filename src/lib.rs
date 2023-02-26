//! Jetro is a tool for querying and transforming json format.

extern crate pest_derive;

use pest::Parser;
use pest_derive::Parser as Parse;

pub mod parser;
pub mod context;
