//! Checkpoint management for OpenSesame CSM training.
//!
//! Checkpoints are stored as sub-directories of the form
//! `{checkpoint_dir}/step_{N:06}/` with a `meta.json` manifest.
//!
//! # Directory layout
//! ```text
//! checkpoints/
//!   step_000001/
//!     meta.json       ← CheckpointMeta (step, loss, timestamp, config snapshot)
//!   step_005000/
//!     meta.json
//! ```

use std::io;
use std::fs;
use std::path::Path;

/// Metadata stored in each checkpoint directory.
#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    /// Training step at which the checkpoint was saved.
    pub step: usize,
    /// Exponential moving average of training loss at this step.
    pub loss_ema: f32,
    /// Wall-clock timestamp (seconds since epoch) — set to 0 when unavailable.
    pub timestamp: u64,
}

impl CheckpointMeta {
    /// Create a new meta record.
    pub fn new(step: usize, loss_ema: f32) -> Self {
        Self { step, loss_ema, timestamp: 0 }
    }

    /// Serialise to a JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"step":{},"loss_ema":{:.6},"timestamp":{}}}"#,
            self.step, self.loss_ema, self.timestamp
        )
    }

    /// Deserialise from a JSON string using `atlas_json`.
    pub fn from_json(s: &str) -> Result<Self, String> {
        use atlas_json::Json;
        let v = Json::parse(s).map_err(|e| e.to_string())?;
        let step = v.get("step").and_then(|x| x.as_usize())
            .ok_or("missing 'step'")?;
        let loss_ema = v.get("loss_ema").and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .ok_or("missing 'loss_ema'")?;
        let timestamp = v.get("timestamp").and_then(|x| x.as_i64())
            .map(|x| x as u64)
            .unwrap_or(0);
        Ok(Self { step, loss_ema, timestamp })
    }
}

/// High-level checkpoint I/O helpers.
pub struct Checkpoint;

impl Checkpoint {
    /// Returns the sub-directory path for a given step.
    ///
    /// Format: `{dir}/step_{step:06}`.
    pub fn dir_for_step(dir: &str, step: usize) -> String {
        format!("{}/step_{:06}", dir, step)
    }

    /// Write a `meta.json` to `{dir}/step_{step:06}/meta.json`.
    ///
    /// Creates the checkpoint sub-directory if it does not exist.
    pub fn save_manifest(
        dir: &str,
        step: usize,
        loss_ema: f32,
    ) -> Result<(), io::Error> {
        let sub = Self::dir_for_step(dir, step);
        fs::create_dir_all(&sub)?;
        let meta = CheckpointMeta::new(step, loss_ema);
        let path = format!("{}/meta.json", sub);
        fs::write(&path, meta.to_json())?;
        Ok(())
    }

    /// Read and parse `meta.json` from a checkpoint sub-directory path.
    ///
    /// `path` may be either the sub-directory itself or the `meta.json` file.
    pub fn load_manifest(path: &str) -> Result<CheckpointMeta, io::Error> {
        let json_path = if path.ends_with("meta.json") {
            path.to_string()
        } else {
            format!("{}/meta.json", path)
        };
        let contents = fs::read_to_string(&json_path)?;
        CheckpointMeta::from_json(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// List all checkpoint steps found in `dir`, sorted ascending by step.
    ///
    /// Only directories matching `step_NNNNNN` are included.
    pub fn list_checkpoints(dir: &str) -> Vec<(usize, String)> {
        let Ok(entries) = fs::read_dir(dir) else { return Vec::new(); };
        let mut out: Vec<(usize, String)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if !name.starts_with("step_") { return None; }
                let step_str = &name["step_".len()..];
                let step: usize = step_str.parse().ok()?;
                let full = format!("{}/{}", dir, name);
                Some((step, full))
            })
            .collect();
        out.sort_by_key(|(s, _)| *s);
        out
    }

    /// Delete the oldest checkpoints, keeping only the `keep_last_n` most recent.
    ///
    /// Does nothing if there are ≤ `keep_last_n` checkpoints.
    pub fn prune_checkpoints(dir: &str, keep_last_n: usize) {
        let mut ckpts = Self::list_checkpoints(dir);
        // Sort ascending → remove from the front
        while ckpts.len() > keep_last_n {
            let (_, path) = ckpts.remove(0);
            let _ = fs::remove_dir_all(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(suffix: &str) -> String {
        let dir = format!("/tmp/opensesame_ckpt_test_{}", suffix);
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_checkpoint_save_creates_dir() {
        let dir = tmp_dir("save");
        Checkpoint::save_manifest(&dir, 1, 2.345).unwrap();
        let meta_path = format!("{}/step_000001/meta.json", dir);
        assert!(Path::new(&meta_path).exists(),
            "meta.json should exist at {}", meta_path);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_meta_roundtrip() {
        let dir = tmp_dir("roundtrip");
        Checkpoint::save_manifest(&dir, 500, 1.234).unwrap();
        let sub = Checkpoint::dir_for_step(&dir, 500);
        let meta = Checkpoint::load_manifest(&sub).unwrap();
        assert_eq!(meta.step, 500);
        assert!((meta.loss_ema - 1.234).abs() < 1e-4,
            "loss_ema roundtrip: expected 1.234, got {}", meta.loss_ema);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_list_sorted() {
        let dir = tmp_dir("list");
        Checkpoint::save_manifest(&dir, 3000, 0.5).unwrap();
        Checkpoint::save_manifest(&dir, 1000, 0.7).unwrap();
        Checkpoint::save_manifest(&dir, 2000, 0.6).unwrap();
        let list = Checkpoint::list_checkpoints(&dir);
        let steps: Vec<usize> = list.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![1000, 2000, 3000],
            "Checkpoints should be sorted by step ascending");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_prune_keeps_last_n() {
        let dir = tmp_dir("prune");
        for step in [1000, 2000, 3000, 4000, 5000] {
            Checkpoint::save_manifest(&dir, step, 0.5).unwrap();
        }
        Checkpoint::prune_checkpoints(&dir, 3);
        let list = Checkpoint::list_checkpoints(&dir);
        let steps: Vec<usize> = list.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![3000, 4000, 5000],
            "After pruning, only last 3 should remain: {:?}", steps);
        let _ = fs::remove_dir_all(&dir);
    }
}
