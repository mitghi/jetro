use super::stream_plan::RowStreamDirection;
use crate::data::value::Val;
use std::sync::Arc;

pub(super) trait ValueRowSource {
    fn next_row(&mut self) -> Option<Val>;
}

pub(super) enum DocumentRowSource {
    Array {
        rows: Arc<Vec<Val>>,
        next: usize,
        remaining: usize,
        reverse: bool,
    },
    Materialized {
        rows: Vec<Val>,
        next: usize,
    },
}

impl DocumentRowSource {
    pub(super) fn new(root: Val, direction: RowStreamDirection) -> Self {
        match root {
            Val::Arr(rows) => {
                let remaining = rows.len();
                Self::Array {
                    rows,
                    next: 0,
                    remaining,
                    reverse: direction == RowStreamDirection::Reverse,
                }
            }
            root => {
                let mut rows = match root.into_vals() {
                    Ok(rows) => rows,
                    Err(root) => vec![root],
                };
                if direction == RowStreamDirection::Reverse {
                    rows.reverse();
                }
                Self::Materialized { rows, next: 0 }
            }
        }
    }
}

impl ValueRowSource for DocumentRowSource {
    fn next_row(&mut self) -> Option<Val> {
        match self {
            Self::Array {
                rows,
                next,
                remaining,
                reverse,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let idx = if *reverse {
                    *remaining -= 1;
                    *remaining
                } else {
                    let idx = *next;
                    *next += 1;
                    *remaining -= 1;
                    idx
                };
                rows.get(idx).cloned()
            }
            Self::Materialized { rows, next } => {
                let row = rows.get(*next).cloned();
                *next += usize::from(row.is_some());
                row
            }
        }
    }
}
