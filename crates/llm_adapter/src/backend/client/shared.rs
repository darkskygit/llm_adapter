use std::io::Read;

use crate::backend::BackendError;

pub(super) fn stream_utf8_chunks(
  reader: &mut dyn Read,
  on_chunk: &mut dyn FnMut(&str) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
  let mut buf = [0_u8; 8192];
  let mut pending = Vec::new();

  loop {
    let read = reader
      .read(&mut buf)
      .map_err(|error| BackendError::Http(error.to_string()))?;
    if read == 0 {
      break;
    }

    pending.extend_from_slice(&buf[..read]);
    let mut consumed = 0usize;

    match std::str::from_utf8(&pending) {
      Ok(text) => {
        on_chunk(text)?;
        consumed = pending.len();
      }
      Err(error) => {
        let valid_up_to = error.valid_up_to();
        if valid_up_to > 0 {
          let valid_text =
            std::str::from_utf8(&pending[..valid_up_to]).map_err(|decode| BackendError::Http(decode.to_string()))?;
          on_chunk(valid_text)?;
          consumed = valid_up_to;
        }

        if error.error_len().is_some() {
          let fallback = String::from_utf8_lossy(&pending[consumed..]).to_string();
          on_chunk(&fallback)?;
          consumed = pending.len();
        }
      }
    }

    if consumed > 0 {
      pending.drain(..consumed);
    }
  }

  if !pending.is_empty() {
    let tail = String::from_utf8_lossy(&pending).to_string();
    on_chunk(&tail)?;
  }

  Ok(())
}
