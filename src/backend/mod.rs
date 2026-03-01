mod client;
mod dispatch;
mod request_layer;
mod types;

pub use client::ReqwestHttpClient;
pub use dispatch::{
  collect_stream_encoded, collect_stream_events, dispatch_request, dispatch_stream_encoded_with,
  dispatch_stream_events_with,
};
pub use types::{
  BackendConfig, BackendError, BackendHttpClient, BackendProtocol, BackendRequestLayer, HttpRequest, HttpResponse,
  HttpStreamResponse,
};

#[cfg(test)]
mod tests;
