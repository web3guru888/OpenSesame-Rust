//! build.rs — Compile ALL CUDA kernels (ATLAS + OpenSesame audio) if nvcc is available.
//!
//! Zero-dependency GPU strategy:
//!   - `nvcc kernels/*.cu` → compiled into a single static lib
//!   - Link against system `libcuda` and `libcublas` (system libs, not Rust crates)
//!   - Rust calls via `extern "C"` — no cudarc, no tch, no candle
//!
//! Kernels compiled (16 total):
//!   ATLAS (8):       matmul, attention, quant,
//!                    (rmsnorm, rope, silu_mul, adamw, argmax — in attention.cu / matmul.cu)
//!   OpenSesame (8):  conv1d_causal, conv1d_strided, conv1d_transposed, depthwise_conv1d,
//!                    vq_search, ema_update, stft, resample
//!
//! GPU architecture auto-detection order:
//!   1. ATLAS_CUDA_ARCH env var (e.g. "sm_75")
//!   2. Query `nvidia-smi` for the first GPU's compute capability
//!   3. Fall back to sm_75 (T4 / Turing — conservative safe default)
//!
//! If nvcc is not found, CUDA support is silently disabled and
//! atlas-tensor falls back to the pure Rust CPU implementation.

use std::path::{Path, PathBuf};
use std::process::Command;

/// All .cu kernel files to compile (relative to workspace root kernels/).
const ATLAS_KERNELS: &[&str] = &[
    "matmul.cu",
    "attention.cu",
    "quant.cu",
];

const OPENSESAME_KERNELS: &[&str] = &[
    "conv1d_causal.cu",
    "conv1d_strided.cu",
    "conv1d_transposed.cu",
    "depthwise_conv1d.cu",
    "vq_search.cu",
    "ema_update.cu",
    "stft.cu",
    "resample.cu",
];

fn main() {
    // Declare custom cfg flags so rustc's check-cfg lint doesn't warn
    println!("cargo::rustc-check-cfg=cfg(atlas_cuda)");
    println!("cargo::rustc-check-cfg=cfg(atlas_cpu_only)");

    // Emit rerun triggers for all kernels
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ATLAS_CUDA_ARCH");
    for k in ATLAS_KERNELS.iter().chain(OPENSESAME_KERNELS.iter()) {
        println!("cargo:rerun-if-changed=../../kernels/{k}");
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernels_dir = Path::new("../../kernels");

    // ── 1. Locate nvcc ────────────────────────────────────────────────────────
    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        eprintln!("[atlas-tensor/build.rs] nvcc not found — CPU-only build");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    };
    eprintln!("[atlas-tensor/build.rs] nvcc = {}", nvcc.display());

    // ── 2. Determine GPU architecture ─────────────────────────────────────────
    let arch = gpu_arch();
    eprintln!("[atlas-tensor/build.rs] GPU arch = {arch}");

    // ── 3. Compile each .cu → .o ──────────────────────────────────────────────
    let all_kernels: Vec<&str> = ATLAS_KERNELS.iter()
        .chain(OPENSESAME_KERNELS.iter())
        .copied()
        .collect();

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for kernel_file in &all_kernels {
        let cu_path = kernels_dir.join(kernel_file);
        if !cu_path.exists() {
            eprintln!("[atlas-tensor/build.rs] kernel not found: {} — skipping", cu_path.display());
            continue;
        }

        let stem = Path::new(kernel_file).file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));

        let status = Command::new(&nvcc)
            .args([
                "-O3",
                &format!("-arch={arch}"),
                "--compiler-options", "-fPIC",
                "-c", cu_path.to_str().unwrap(),
                "-o", obj.to_str().unwrap(),
            ])
            .status()
            .expect("nvcc invocation failed");

        if !status.success() {
            eprintln!("[atlas-tensor/build.rs] nvcc compilation FAILED for {kernel_file} — falling back to CPU");
            println!("cargo:rustc-cfg=atlas_cpu_only");
            return;
        }
        eprintln!("[atlas-tensor/build.rs] compiled {kernel_file} → {}", obj.display());
        obj_files.push(obj);
    }

    if obj_files.is_empty() {
        eprintln!("[atlas-tensor/build.rs] no kernel objects produced — CPU-only");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    }

    // ── 4. Package all .o → single static lib ────────────────────────────────
    let lib = out_dir.join("libatlas_kernels.a");
    let obj_strs: Vec<&str> = obj_files.iter().map(|p| p.to_str().unwrap()).collect();

    let mut ar_args = vec!["rcs", lib.to_str().unwrap()];
    ar_args.extend_from_slice(&obj_strs);

    let ar_ok = Command::new("ar")
        .args(&ar_args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !ar_ok {
        eprintln!("[atlas-tensor/build.rs] ar failed — CPU-only");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    }

    // ── 5. Tell cargo where to find the static lib and system CUDA libs ───────
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.9/lib64",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.8/lib64",
    ] {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=static=atlas_kernels");
    println!("cargo:rustc-link-lib=cudart");  // CUDA runtime
    println!("cargo:rustc-link-lib=cublas");  // cuBLAS (TF32 tensor cores)
    println!("cargo:rustc-cfg=atlas_cuda");
    eprintln!(
        "[atlas-tensor/build.rs] All {} kernels compiled OK ({arch})",
        obj_files.len()
    );
}

/// Find nvcc: check ATLAS_CUDA_ARCH env hint path, common CUDA install dirs, then PATH.
fn find_nvcc() -> Option<PathBuf> {
    let candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12.2/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
        "nvcc", // already in PATH
    ];
    for c in &candidates {
        let path = PathBuf::from(c);
        let found = if path.is_absolute() {
            path.exists()
        } else {
            Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
        };
        if found {
            return Some(path);
        }
    }
    None
}

/// Determine the best CUDA architecture string.
/// Priority: ATLAS_CUDA_ARCH env → nvidia-smi query → safe default (sm_75).
fn gpu_arch() -> String {
    if let Ok(arch) = std::env::var("ATLAS_CUDA_ARCH") {
        if !arch.is_empty() {
            return arch;
        }
    }

    let smi = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"])
        .output();

    if let Ok(out) = smi {
        if out.status.success() {
            let raw = String::from_utf8_lossy(&out.stdout);
            let cc = raw.trim().replace('.', "");
            if !cc.is_empty() && cc.chars().all(|c| c.is_ascii_digit()) {
                return format!("sm_{cc}");
            }
        }
    }

    "sm_75".to_string()
}
