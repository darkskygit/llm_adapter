pub mod backend;
pub mod core;
/// Public middleware hooks intended for external request/stream orchestration.
/// The internal backend dispatch path does not invoke this module directly.
pub mod middleware;
pub mod protocol;
/// Public fallback router helpers for external multi-provider orchestration.
pub mod router;
pub mod stream;

#[cfg(test)]
pub(crate) mod test_support;
