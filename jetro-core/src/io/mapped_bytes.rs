use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::raw::{c_int, c_void};
#[cfg(unix)]
use std::ptr::NonNull;

pub(super) struct MappedBytes {
    inner: MappedBytesInner,
}

enum MappedBytesInner {
    #[cfg(unix)]
    Mmap(UnixMmap),
    Owned(Vec<u8>),
}

impl MappedBytes {
    pub(super) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self {
                inner: MappedBytesInner::Owned(Vec::new()),
            });
        }

        #[cfg(unix)]
        {
            if let Ok(map) = UnixMmap::map(&file, len) {
                return Ok(Self {
                    inner: MappedBytesInner::Mmap(map),
                });
            }
        }

        let bytes = std::fs::read(path)?;
        Ok(Self {
            inner: MappedBytesInner::Owned(bytes),
        })
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        match &self.inner {
            #[cfg(unix)]
            MappedBytesInner::Mmap(map) => map.as_slice(),
            MappedBytesInner::Owned(bytes) => bytes,
        }
    }
}

#[cfg(unix)]
struct UnixMmap {
    ptr: NonNull<u8>,
    len: usize,
}

#[cfg(unix)]
unsafe impl Send for UnixMmap {}
#[cfg(unix)]
unsafe impl Sync for UnixMmap {}

#[cfg(unix)]
impl UnixMmap {
    fn map(file: &File, len: usize) -> io::Result<Self> {
        const PROT_READ: c_int = 0x1;
        const MAP_PRIVATE: c_int = 0x02;

        extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: c_int,
                flags: c_int,
                fd: c_int,
                offset: isize,
            ) -> *mut c_void;
        }

        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "mmap returned null"))?;
        Ok(Self { ptr, len })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for UnixMmap {
    fn drop(&mut self) {
        extern "C" {
            fn munmap(addr: *mut c_void, len: usize) -> c_int;
        }
        let _ = unsafe { munmap(self.ptr.as_ptr().cast::<c_void>(), self.len) };
    }
}
