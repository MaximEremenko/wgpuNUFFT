//! Error codes and the per-thread last error message.

use std::cell::RefCell;
use std::ffi::CString;

use wgpu_nufft::NufftError;

pub(crate) const SUCCESS: i32 = 0;
pub(crate) const ERROR_INVALID_ARGUMENT: i32 = 1;
pub(crate) const ERROR_PLAN: i32 = 2;
pub(crate) const ERROR_GPU_UNAVAILABLE: i32 = 3;
pub(crate) const ERROR_GPU: i32 = 4;
pub(crate) const ERROR_INTERNAL: i32 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Error {
    InvalidArgument(String),
    Plan(String),
    GpuUnavailable(String),
    Gpu(String),
    Internal(String),
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn code(&self) -> i32 {
        match self {
            Self::InvalidArgument(_) => ERROR_INVALID_ARGUMENT,
            Self::Plan(_) => ERROR_PLAN,
            Self::GpuUnavailable(_) => ERROR_GPU_UNAVAILABLE,
            Self::Gpu(_) => ERROR_GPU,
            Self::Internal(_) => ERROR_INTERNAL,
        }
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(message)
            | Self::Plan(message)
            | Self::GpuUnavailable(message)
            | Self::Gpu(message)
            | Self::Internal(message) => message,
        }
    }
}

impl From<NufftError> for Error {
    fn from(error: NufftError) -> Self {
        match error {
            NufftError::PrecisionUnsupported { .. } => Self::GpuUnavailable(error.to_string()),
            _ => Self::Plan(error.to_string()),
        }
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Records the outcome of an interface call and returns its code.
pub(crate) fn report(result: Result<()>) -> i32 {
    let (code, message) = match result {
        Ok(()) => (SUCCESS, String::new()),
        Err(error) => (error.code(), error.message().to_owned()),
    };
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
    code
}

pub(crate) fn last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Runs an interface call, turning a panic into an internal error so that it
/// never unwinds into the caller's language.
pub(crate) fn call(work: impl FnOnce() -> Result<()>) -> i32 {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|payload| {
            Err(Error::Internal(format!(
                "panic: {}",
                crate::gpu::panic_message(&payload)
            )))
        });
    report(result)
}
