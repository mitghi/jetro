//! Convenience re-exports.
//!
//! `use jetro::prelude::*;` pulls in the commonly used types and
//! traits so call-sites don't need a dozen individual `use` lines.
//!
//! ```rust
//! use jetro::prelude::*;
//! use serde_json::json;
//!
//! let j = Jetro::new(json!({"x": 1}));
//! let v = j.collect("$.x")?;
//! # assert_eq!(v, json!(1));
//! # Ok::<(), jetro::Error>(())
//! ```

pub use crate::{Error, Result};
pub use crate::{Jetro, Graph, VM, Expr, Engine};
pub use crate::{Session, SessionBuilder, Catalog, MemCatalog};
pub use crate::{Method, MethodRegistry};
pub use crate::{query, query_with};
pub use crate::db::{Database, ExprBucket, JsonBucket, GraphBucket, BTree};
pub use crate::db::{Bucket, BucketBuilder, GraphQueryBuilder};
pub use crate::db::{Join, JoinBuilder, JoinQuery, JoinedDoc, IntoJoinId};
pub use crate::db::{GraphNode};
pub use crate::db::{Store, BTreeBulk, BTreePrefix, DocIter};
pub use crate::db::{Storage, Tree, FileStorage, MemStorage, BTreeMem};
pub use serde_json::{json, Value};
