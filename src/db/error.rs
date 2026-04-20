use std::fmt;

#[derive(Debug)]
pub enum DbError {
    Io(std::io::Error),
    InvalidExpr(String),
    ExprNotFound(String),
    EvalError(String),
    Serialize(String),
    Corrupt(String),
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DbError::Io(e) => write!(f, "io: {e}"),
            DbError::InvalidExpr(e) => write!(f, "invalid expression: {e}"),
            DbError::ExprNotFound(k) => write!(f, "expression not found: {k}"),
            DbError::EvalError(e) => write!(f, "eval: {e}"),
            DbError::Serialize(e) => write!(f, "serialize: {e}"),
            DbError::Corrupt(e) => write!(f, "corrupt: {e}"),
        }
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let DbError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> Self {
        DbError::Io(e)
    }
}
