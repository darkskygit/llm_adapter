#[cfg(all(feature = "reqwest-client", feature = "ureq-client"))]
compile_error!("llm_adapter transport features `reqwest-client` and `ureq-client` are mutually exclusive");

#[cfg(not(any(feature = "reqwest-client", feature = "ureq-client")))]
compile_error!("llm_adapter requires exactly one transport feature: `reqwest-client` or `ureq-client`");

mod shared;

#[cfg(feature = "reqwest-client")]
mod reqwest;
#[cfg(feature = "ureq-client")]
mod ureq;

#[cfg(feature = "reqwest-client")]
pub use self::reqwest::ReqwestHttpClient;
#[cfg(feature = "ureq-client")]
pub use self::ureq::UreqHttpClient;

#[cfg(feature = "reqwest-client")]
pub type DefaultHttpClient = ReqwestHttpClient;
#[cfg(feature = "ureq-client")]
pub type DefaultHttpClient = UreqHttpClient;
