mod client;
mod dispatch;
mod request_layer;
mod types;

pub use client::ReqwestHttpClient;
pub use dispatch::{dispatch_request, dispatch_stream, dispatch_stream_with_handler};
pub use types::{
  BackendConfig, BackendError, BackendHttpClient, BackendProtocol, BackendRequestLayer, HttpRequest, HttpResponse,
  HttpStreamResponse,
};

#[cfg(test)]
mod tests;
