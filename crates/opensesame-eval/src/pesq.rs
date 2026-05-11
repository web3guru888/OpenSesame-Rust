//! PESQ stub — proprietary ITU-T P.862 standard.
//!
//! PESQ is not implemented in this crate due to its proprietary nature and
//! patent encumbrances. Use [`crate::visqol::ViSQOL`] as a MOS proxy instead.
//!
//! If you have a PESQ binary available (ITU evaluation license), consider
//! calling it as a subprocess and parsing "MOS-LQO: X.XX" from stdout.

use crate::{EvalError, EvalResult};

/// PESQ metric stub.
pub struct Pesq;

impl Pesq {
    /// Always returns `Err` — PESQ is not implemented.
    ///
    /// Use `ViSQOL::compute` for an open-source MOS proxy.
    pub fn compute(_reference: &[f32], _estimate: &[f32], _sr: u32) -> EvalResult<f32> {
        Err(EvalError::NumericalError(
            "PESQ not implemented: proprietary ITU-T P.862 standard. \
             Use ViSQOL as a MOS proxy instead.",
        ))
    }
}
