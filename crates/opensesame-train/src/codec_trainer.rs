//! Codec trainer placeholder (Phase J stub).
//!
//! Stage 0 of the OpenSesame training pipeline trains the Mimi codec from
//! scratch. For Phase J we use pretrained Kyutai Mimi weights, so this
//! module is a stub that will be filled in a future phase.

/// Placeholder for future Mimi codec training from scratch.
///
/// For now, we load pretrained Kyutai Mimi weights via
/// `opensesame_mimi::MimiCodec::load_from_safetensors`.
pub struct CodecTrainer;

impl CodecTrainer {
    /// Create a codec trainer stub (not implemented in Phase J).
    pub fn new() -> Self { Self }
}

impl Default for CodecTrainer {
    fn default() -> Self { Self::new() }
}
