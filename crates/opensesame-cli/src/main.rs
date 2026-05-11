//! opensesame — Pure Rust Conversational Speech Model CLI
//!
//! Built on ATLAS (Active-inference Training with Learned Adaptive Stigmergy)
//!
//! Usage:
//!   opensesame download  --dataset librispeech --split train-clean-100 --dest ./data
//!   opensesame tokenize  --dataset ./data --codec ./mimi.safetensors --out ./tokens
//!   opensesame train     --stage codec    --config config.toml --checkpoint ./ckpt
//!   opensesame train     --stage backbone --config config.toml --checkpoint ./ckpt
//!   opensesame train     --stage finetune --config config.toml --checkpoint ./ckpt
//!   opensesame eval      --model ./ckpt --testset ./data/test-clean
//!   opensesame infer     --model ./ckpt --input hello.wav --output response.wav
//!   opensesame serve     --model ./ckpt --port 8080
//!   opensesame bench     --model ./ckpt --all
//!   opensesame convert   --from moshi --weights ./moshi.safetensors --out ./ckpt

fn main() {
    println!("opensesame v0.1.0 — Pure Rust CSM on ATLAS");
    println!("Run `opensesame --help` for available commands.");
    // Full CLI implementation: Phase M
}
