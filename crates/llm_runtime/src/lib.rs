mod host;
mod r#loop;
mod replay;
mod round;
mod tool_call;
mod types;

pub use host::{EventSink, ToolExecutor};
pub use r#loop::run_tool_loop;
pub use replay::append_tool_turns;
pub use round::{RoundOutcome, RoundProcessor, RoundProcessorError, StreamRoundRunner, run_stream_round};
pub use tool_call::ToolCallAccumulator;
pub use types::{AccumulatedToolCall, ToolExecutionResult, ToolLoopEvent, ToolResultMessage};
