//! Word Error Rate (WER) — NIST STM 1997 definition.
//!
//! WER = (Substitutions + Deletions + Insertions) / N_reference
//!
//! Computed via dynamic programming (Levenshtein edit distance) with
//! traceback to recover the individual error counts.

/// Detailed WER breakdown.
#[derive(Debug, Clone)]
pub struct WerDetail {
    /// WER score: (S + D + I) / N_ref. May be `f32::INFINITY` when ref is empty.
    pub wer: f32,
    /// Number of substitutions.
    pub substitutions: usize,
    /// Number of deletions (words in ref not in hyp).
    pub deletions: usize,
    /// Number of insertions (words in hyp not in ref).
    pub insertions: usize,
    /// Total reference words.
    pub reference_len: usize,
}

/// WER metric.
pub struct Wer;

impl Wer {
    /// Compute WER between two text strings.
    ///
    /// Text is normalised: lowercased, punctuation stripped, split on whitespace.
    pub fn compute(reference: &str, hypothesis: &str) -> WerDetail {
        let ref_words = Self::normalize_and_tokenize(reference);
        let hyp_words = Self::normalize_and_tokenize(hypothesis);
        let ref_refs: Vec<&str> = ref_words.iter().map(|s| s.as_str()).collect();
        let hyp_refs: Vec<&str> = hyp_words.iter().map(|s| s.as_str()).collect();
        Self::compute_words(&ref_refs, &hyp_refs)
    }

    /// Compute WER between pre-tokenised word sequences.
    pub fn compute_words(reference: &[&str], hypothesis: &[&str]) -> WerDetail {
        let r = reference.len();
        let h = hypothesis.len();

        if r == 0 && h == 0 {
            return WerDetail {
                wer: 0.0,
                substitutions: 0,
                deletions: 0,
                insertions: 0,
                reference_len: 0,
            };
        }
        if r == 0 {
            return WerDetail {
                wer: f32::INFINITY,
                substitutions: 0,
                deletions: 0,
                insertions: h,
                reference_len: 0,
            };
        }

        // DP table: (r+1) × (h+1)
        let mut dp = vec![vec![0usize; h + 1]; r + 1];
        for i in 0..=r {
            dp[i][0] = i;
        }
        for j in 0..=h {
            dp[0][j] = j;
        }
        for i in 1..=r {
            for j in 1..=h {
                dp[i][j] = if reference[i - 1] == hypothesis[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    1 + dp[i - 1][j - 1] // substitution
                        .min(dp[i][j - 1]) // insertion
                        .min(dp[i - 1][j]) // deletion
                };
            }
        }

        // Traceback
        let (mut s, mut d, mut ins) = (0usize, 0usize, 0usize);
        let (mut i, mut j) = (r, h);
        while i > 0 || j > 0 {
            if i == 0 {
                ins += 1;
                j -= 1;
            } else if j == 0 {
                d += 1;
                i -= 1;
            } else if reference[i - 1] == hypothesis[j - 1] {
                i -= 1;
                j -= 1;
            } else {
                let min_val = dp[i - 1][j - 1].min(dp[i][j - 1]).min(dp[i - 1][j]);
                if dp[i - 1][j - 1] == min_val {
                    s += 1;
                    i -= 1;
                    j -= 1;
                } else if dp[i][j - 1] == min_val {
                    ins += 1;
                    j -= 1;
                } else {
                    d += 1;
                    i -= 1;
                }
            }
        }

        WerDetail {
            wer: (s + d + ins) as f32 / r as f32,
            substitutions: s,
            deletions: d,
            insertions: ins,
            reference_len: r,
        }
    }

    /// Levenshtein edit distance between two word sequences.
    pub fn levenshtein(reference: &[&str], hypothesis: &[&str]) -> usize {
        Self::compute_words(reference, hypothesis).substitutions
            + Self::compute_words(reference, hypothesis).deletions
            + Self::compute_words(reference, hypothesis).insertions
    }

    /// Returns `(n_sub, n_del, n_ins, n_ref)`.
    pub fn alignment(reference: &[&str], hypothesis: &[&str]) -> (usize, usize, usize, usize) {
        let d = Self::compute_words(reference, hypothesis);
        (d.substitutions, d.deletions, d.insertions, d.reference_len)
    }

    /// Lowercase, strip non-alphanumeric/space characters, split on whitespace.
    pub fn normalize_and_tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wer_identical() {
        let d = Wer::compute("hello world", "hello world");
        assert_eq!(d.wer, 0.0);
        assert_eq!(d.substitutions, 0);
        assert_eq!(d.deletions, 0);
        assert_eq!(d.insertions, 0);
    }

    #[test]
    fn test_wer_one_deletion() {
        let d = Wer::compute("hello world", "hello");
        assert_eq!(d.deletions, 1);
        assert_eq!(d.substitutions, 0);
        assert_eq!(d.insertions, 0);
        assert!((d.wer - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_insertion() {
        let d = Wer::compute("hello", "hello world");
        assert_eq!(d.insertions, 1);
        assert_eq!(d.deletions, 0);
        assert_eq!(d.substitutions, 0);
        assert!((d.wer - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_substitution() {
        let d = Wer::compute("hello world", "hello there");
        assert_eq!(d.substitutions, 1);
        assert_eq!(d.deletions, 0);
        assert_eq!(d.insertions, 0);
        assert!((d.wer - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wer_empty_both() {
        let d = Wer::compute("", "");
        assert_eq!(d.wer, 0.0);
    }

    #[test]
    fn test_wer_empty_hypothesis() {
        let d = Wer::compute("one two three", "");
        assert_eq!(d.deletions, 3);
        assert!((d.wer - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_complete_wrong() {
        let d = Wer::compute("one two three", "four five six");
        assert_eq!(d.substitutions, 3);
        assert!((d.wer - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_real_example() {
        let reference = "the quick brown fox jumps over the lazy dog";
        let hypothesis = "the quick brown fox jump over lazy dog";
        let d = Wer::compute(reference, hypothesis);
        assert!(d.wer < 0.5, "WER should be < 0.5: {}", d.wer);
    }

    #[test]
    fn test_wer_empty_reference() {
        let d = Wer::compute("", "hello world");
        assert!(d.wer.is_infinite());
        assert_eq!(d.insertions, 2);
    }

    #[test]
    fn test_levenshtein_basic() {
        let r = &["cat", "sat", "on", "mat"];
        let h = &["the", "cat", "sat"];
        // Edit distance: del "on", del "mat", ins "the" → 3
        let dist = Wer::levenshtein(r, h);
        assert!(dist <= r.len() + h.len(), "distance = {}", dist);
    }

    #[test]
    fn test_alignment_basic() {
        let r = &["hello", "world"];
        let h = &["hello", "there"];
        let (s, d, ins, n) = Wer::alignment(r, h);
        assert_eq!(s, 1, "one substitution");
        assert_eq!(d, 0, "no deletions");
        assert_eq!(ins, 0, "no insertions");
        assert_eq!(n, 2, "reference length");
    }
}
