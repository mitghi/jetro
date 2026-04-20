pub const PAGE_SIZE: usize = 4096;
pub type PageId = u32;
pub const NULL_PAGE: PageId = u32::MAX;

pub const NODE_INTERNAL: u8 = 1;
pub const NODE_LEAF: u8 = 2;

pub const TAG_INLINE: u8 = 0x00;
pub const TAG_OVERFLOW: u8 = 0x01;

/// Values <= this are stored inline in the leaf page.
pub const MAX_INLINE_VAL: usize = 512;

/// Usable bytes in an overflow page (4 bytes reserved for next-page pointer).
pub const OVERFLOW_DATA_SIZE: usize = PAGE_SIZE - 4;
