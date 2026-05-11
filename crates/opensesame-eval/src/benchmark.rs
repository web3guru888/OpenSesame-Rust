//! Batch evaluation runner and result aggregation.
//!
//! `EvalSuite` combines all Phase-K metrics into a single runner.
//! `BenchmarkResult` holds aggregate statistics.

use std::time::{Duration, Instant};
use crate::{SiSnr, Wer, Stoi, Mcd, ViSQOL};

/// Per-sample evaluation result.
#[derive(Debug, Clone)]
pub struct SingleResult {
    /// SI-SNR in dB (`None` if computation failed).
    pub si_snr_db: Option<f32>,
    /// WER (`None` if no transcripts provided).
    pub wer: Option<f32>,
    /// STOI (`None` if computation failed).
    pub stoi: Option<f32>,
    /// MCD in dB (`None` if computation failed).
    pub mcd_db: Option<f32>,
    /// ViSQOL MOS (`None` if computation failed).
    pub visqol_mos: Option<f32>,
    /// PESQ MOS (always `None` — not implemented).
    pub pesq_mos: Option<f32>,
}

/// Batch evaluation summary statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of evaluated samples.
    pub n_samples: usize,
    /// Mean SI-SNR in dB.
    pub si_snr_mean: f32,
    /// Standard deviation of SI-SNR.
    pub si_snr_std: f32,
    /// Mean WER.
    pub wer_mean: f32,
    /// Mean STOI.
    pub stoi_mean: f32,
    /// Mean MCD in dB.
    pub mcd_mean: f32,
    /// Mean ViSQOL MOS.
    pub visqol_mos_mean: f32,
    /// Standard deviation of ViSQOL MOS.
    pub visqol_mos_std: f32,
    /// Wall-clock time for the whole batch.
    pub wall_time: Duration,
    /// Individual results (one per sample).
    pub individual: Vec<SingleResult>,
}

impl BenchmarkResult {
    /// One-line summary string.
    pub fn summarize(&self) -> String {
        format!(
            "N={} | SI-SNR={:.2}±{:.2}dB | WER={:.3} | STOI={:.4} | \
             MCD={:.2}dB | ViSQOL={:.3}±{:.3} MOS | t={:.1}s",
            self.n_samples,
            self.si_snr_mean,
            self.si_snr_std,
            self.wer_mean,
            self.stoi_mean,
            self.mcd_mean,
            self.visqol_mos_mean,
            self.visqol_mos_std,
            self.wall_time.as_secs_f32(),
        )
    }

    /// Serialise to a compact JSON string (no external crate).
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"n_samples":{n},"si_snr_mean":{si},"wer_mean":{wer},"stoi_mean":{stoi},"mcd_mean":{mcd},"visqol_mos_mean":{mos}}}"#,
            n = self.n_samples,
            si = self.si_snr_mean,
            wer = self.wer_mean,
            stoi = self.stoi_mean,
            mcd = self.mcd_mean,
            mos = self.visqol_mos_mean,
        )
    }
}

/// Flags controlling which metrics are computed.
pub struct EvalSuiteConfig {
    /// Run SI-SNR.
    pub run_sisnr: bool,
    /// Run WER (requires transcripts).
    pub run_wer: bool,
    /// Run STOI.
    pub run_stoi: bool,
    /// Run MCD.
    pub run_mcd: bool,
    /// Run ViSQOL.
    pub run_visqol: bool,
    /// Native sample rate of the input signals.
    pub fs_native: u32,
}

impl Default for EvalSuiteConfig {
    fn default() -> Self {
        Self {
            run_sisnr: true,
            run_wer: false,
            run_stoi: true,
            run_mcd: true,
            run_visqol: true,
            fs_native: 24000,
        }
    }
}

/// Combined evaluation suite.
pub struct EvalSuite {
    config: EvalSuiteConfig,
    stoi: Stoi,
    mcd: Mcd,
    visqol: ViSQOL,
}

impl EvalSuite {
    /// Create an `EvalSuite` with the given configuration.
    pub fn new(config: EvalSuiteConfig) -> Self {
        Self {
            stoi: Stoi::new(),
            mcd: Mcd::new(),
            visqol: ViSQOL::new(),
            config,
        }
    }

    /// Evaluate a single `(reference, estimate)` pair.
    ///
    /// `reference_transcript` and `hypothesis_transcript` are optional; they are
    /// required only when `config.run_wer == true`.
    pub fn evaluate_one(
        &self,
        reference: &[f32],
        estimate: &[f32],
        reference_transcript: Option<&str>,
        hypothesis_transcript: Option<&str>,
    ) -> SingleResult {
        let sr = self.config.fs_native;

        let si_snr_db = if self.config.run_sisnr {
            SiSnr::compute(reference, estimate).ok().and_then(|v| {
                if v.is_finite() { Some(v) } else { Some(100.0) }
            })
        } else {
            None
        };

        let wer = if self.config.run_wer {
            match (reference_transcript, hypothesis_transcript) {
                (Some(r), Some(h)) => {
                    let d = Wer::compute(r, h);
                    if d.wer.is_finite() { Some(d.wer) } else { None }
                }
                _ => None,
            }
        } else {
            None
        };

        let stoi = if self.config.run_stoi {
            self.stoi.compute(reference, estimate, sr).ok()
        } else {
            None
        };

        let mcd_db = if self.config.run_mcd {
            self.mcd.compute(reference, estimate, sr).ok()
        } else {
            None
        };

        let visqol_mos = if self.config.run_visqol {
            self.visqol.compute(reference, estimate, sr).ok()
        } else {
            None
        };

        SingleResult {
            si_snr_db,
            wer,
            stoi,
            mcd_db,
            visqol_mos,
            pesq_mos: None,
        }
    }

    /// Batch evaluation of a list of `(reference, estimate)` pairs.
    ///
    /// Optionally accepts a parallel list of `(ref_text, hyp_text)` pairs for WER.
    pub fn evaluate_batch(
        &self,
        pairs: &[(&[f32], &[f32])],
        transcripts: Option<&[(&str, &str)]>,
    ) -> BenchmarkResult {
        let t0 = Instant::now();
        let individual: Vec<SingleResult> = pairs
            .iter()
            .enumerate()
            .map(|(i, &(r, e))| {
                let (rt, ht) = transcripts
                    .and_then(|t| t.get(i))
                    .map(|&(r, h)| (Some(r), Some(h)))
                    .unwrap_or((None, None));
                self.evaluate_one(r, e, rt, ht)
            })
            .collect();

        let mean_of = |f: &dyn Fn(&SingleResult) -> Option<f32>| -> f32 {
            let vals: Vec<f32> = individual.iter().filter_map(f).collect();
            if vals.is_empty() {
                f32::NAN
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            }
        };
        let std_of = |f: &dyn Fn(&SingleResult) -> Option<f32>, m: f32| -> f32 {
            let vals: Vec<f32> = individual.iter().filter_map(f).collect();
            if vals.len() < 2 {
                return 0.0;
            }
            let v = vals.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / (vals.len() - 1) as f32;
            v.sqrt()
        };

        let si_snr_mean = mean_of(&|r| r.si_snr_db);
        let visqol_mos_mean = mean_of(&|r| r.visqol_mos);

        BenchmarkResult {
            n_samples: individual.len(),
            si_snr_mean,
            si_snr_std: std_of(&|r| r.si_snr_db, si_snr_mean),
            wer_mean: mean_of(&|r| r.wer),
            stoi_mean: mean_of(&|r| r.stoi),
            mcd_mean: mean_of(&|r| r.mcd_db),
            visqol_mos_mean,
            visqol_mos_std: std_of(&|r| r.visqol_mos, visqol_mos_mean),
            wall_time: t0.elapsed(),
            individual,
        }
    }
}
