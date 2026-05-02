mod client;
mod dispatch;
mod request_layer;
mod types;

pub use client::DefaultHttpClient;
#[cfg(feature = "reqwest-client")]
pub use client::ReqwestHttpClient;
#[cfg(feature = "ureq-client")]
pub use client::UreqHttpClient;
pub use dispatch::{
  collect_stream_events, dispatch_embedding_request, dispatch_image_request, dispatch_request, dispatch_rerank_request,
  dispatch_stream_events_with, dispatch_structured_request,
};
pub use request_layer::{
  AttachmentReferenceMode, AttachmentReferencePlan, AttachmentReferenceReason, RequestIntent, RequestIntentReasoning,
  ResolvedRequestIntent, resolve_attachment_reference_plan, resolve_request_intent,
};
pub use types::{
  BackendConfig, BackendError, BackendHttpClient, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, HttpBody,
  HttpRequest, HttpResponse, HttpStreamResponse, ImageProtocol, MultipartPart, RerankProtocol, StructuredProtocol,
};

#[cfg(test)]
mod tests;
