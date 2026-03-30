use super::SseFrame;

#[derive(Debug, Default)]
pub struct SseFrameDecoder {
  buffer: String,
}

impl SseFrameDecoder {
  pub fn push_chunk(&mut self, chunk: &str) -> Vec<SseFrame> {
    self.buffer.push_str(chunk);

    let mut frames = Vec::new();
    while let Some((index, separator_len)) = find_frame_separator(&self.buffer) {
      let frame_chunk = self.buffer[..index].to_string();
      self.buffer.drain(..index + separator_len);

      if let Some(frame) = parse_frame_block(&frame_chunk) {
        frames.push(frame);
      }
    }

    frames
  }

  pub fn finish(&mut self) -> Vec<SseFrame> {
    if self.buffer.trim().is_empty() {
      self.buffer.clear();
      return Vec::new();
    }

    let remaining = std::mem::take(&mut self.buffer);
    parse_frame_block(remaining.trim()).into_iter().collect()
  }
}

#[must_use]
pub fn parse_sse_frames(raw: &str) -> Vec<SseFrame> {
  let mut decoder = SseFrameDecoder::default();
  let mut frames = decoder.push_chunk(raw);
  frames.extend(decoder.finish());
  frames
}

#[must_use]
pub fn encode_sse_frame(frame: &SseFrame) -> String {
  let mut out = String::new();
  if let Some(event) = &frame.event {
    out.push_str("event: ");
    out.push_str(event);
    out.push('\n');
  }
  for line in frame.data.lines() {
    out.push_str("data: ");
    out.push_str(line);
    out.push('\n');
  }
  out.push('\n');
  out
}

fn find_frame_separator(input: &str) -> Option<(usize, usize)> {
  let lf = input.find("\n\n").map(|index| (index, 2));
  let crlf = input.find("\r\n\r\n").map(|index| (index, 4));
  let cr = input.find("\r\r").map(|index| (index, 2));

  [lf, crlf, cr].into_iter().flatten().min_by_key(|(index, _)| *index)
}

fn parse_frame_block(block: &str) -> Option<SseFrame> {
  let trimmed = block.trim();
  if trimmed.is_empty() {
    return None;
  }

  let mut event = None;
  let mut data_lines = Vec::new();

  for line in trimmed.lines() {
    let line = line.trim_end_matches('\r');
    if let Some(rest) = line.strip_prefix("event:") {
      event = Some(rest.trim().to_string());
    } else if let Some(rest) = line.strip_prefix("data:") {
      data_lines.push(rest.trim_start().to_string());
    }
  }

  if data_lines.is_empty() {
    return None;
  }

  Some(SseFrame {
    event,
    data: data_lines.join("\n"),
  })
}
