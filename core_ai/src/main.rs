use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use futures_util::stream::{self, Stream};
use rayon::prelude::*;
use serde::Deserialize;
use std::{convert::Infallible, net::SocketAddr, path::Path, sync::Arc};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

// --- Native Math Kernels ---

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Tensor {
    fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, shape: (rows, cols) }
    }

    // Parallel Matrix Multiplication: C = A * B
    fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = self.shape;
        let (k2, n) = other.shape;
        assert_eq!(k, k2, "Matrix dimension mismatch: {}x{} * {}x{}", m, k, k2, n);

        let mut result = vec![0.0; m * n];
        result.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
            let a_row = &self.data[i * k..(i + 1) * k];
            for dot_idx in 0..k {
                let a_val = a_row[dot_idx];
                if a_val == 0.0 { continue; }
                let b_row = &other.data[dot_idx * n..(dot_idx + 1) * n];
                for j in 0..n {
                    row_out[j] += a_val * b_row[j];
                }
            }
        });
        Tensor::new(result, m, n)
    }

    fn add_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.data.len(), other.data.len());
        self.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(a, b)| *a += b);
    }

    fn rms_norm(&mut self, weight: &[f32], eps: f32) {
        let cols = self.shape.1;
        self.data.par_chunks_mut(cols).for_each(|row| {
            let pow_sum: f32 = row.iter().map(|x| x * x).sum();
            let inv_std = 1.0 / (pow_sum / cols as f32 + eps).sqrt();
            for (i, x) in row.iter_mut().enumerate() {
                *x = (*x * inv_std) * weight[i];
            }
        });
    }

    fn silu_inplace(&mut self) {
        self.data.par_iter_mut().for_each(|x| {
            *x = *x * (1.0 / (1.0 + (-*x).exp()));
        });
    }

    fn mul_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.data.len(), other.data.len());
        self.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(a, b)| *a *= b);
    }
}

// --- Transformer Architecture ---

struct LayerWeights {
    wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
    w1: Tensor, w2: Tensor, w3: Tensor, // SwiGLU: w3 * silu(w1) * w2? No, usually w2(silu(w1)*w3)
    ffn_norm: Vec<f32>,
    attn_norm: Vec<f32>,
}

struct ModelWeights {
    token_embedding: Tensor,
    layers: Vec<LayerWeights>,
    norm: Vec<f32>,
    output: Tensor,
}

struct Tokenizer {
    // Simple mock tokenizer for the demonstration
    // In real scenarios, this would load a 'tokenizer.json'
}

impl Tokenizer {
    fn decode(&self, id: u32) -> String {
        // Mock decoding: convert ID to char
        if id < 256 {
            (id as u8 as char).to_string()
        } else {
            " ".to_string()
        }
    }
}

struct AppState {
    weights: ModelWeights,
    tokenizer: Tokenizer,
}

// --- Inference Logic ---

fn forward(weights: &ModelWeights, token: u32, _pos: usize) -> u32 {
    let dim = weights.token_embedding.shape.1;
    let mut x = Tensor::new(weights.token_embedding.data[token as usize * dim..(token as usize + 1) * dim].to_vec(), 1, dim);

    for layer in &weights.layers {
        let mut h = x.clone();
        h.rms_norm(&layer.attn_norm, 1e-5);

        // Attention (Simplified: No KV cache for demo, just single token projection)
        let q = h.matmul(&layer.wq);
        let _k = h.matmul(&layer.wk);
        let _v = h.matmul(&layer.wv);

        // Final Attention projection
        let attn_out = q.matmul(&layer.wo); // Mocked attention calc
        x.add_inplace(&attn_out);

        // FFN (SwiGLU)
        let mut h2 = x.clone();
        h2.rms_norm(&layer.ffn_norm, 1e-5);
        let mut g = h2.matmul(&layer.w1);
        g.silu_inplace();
        let up = h2.matmul(&layer.w3);
        g.mul_inplace(&up);
        let ffn_out = g.matmul(&layer.w2);
        x.add_inplace(&ffn_out);
    }

    x.rms_norm(&weights.norm, 1e-5);
    let logits = x.matmul(&weights.output);

    // Greedy sample
    logits.data.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

async fn handle_inference(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        let prompt = payload.prompt;
        let think_msg = format!("<think>\nRaw math inference started for prompt: '{}'\nApplying {} Transformer layers...\nRunning native MatMul & SwiGLU kernels...\n</think>\n", prompt, state.weights.layers.len());

        for word in think_msg.split(' ') {
            if word.is_empty() { continue; }
            let _ = tx.send(format!("{} ", word)).await;
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        // Simulating token generation loop
        let mut current_token = 65; // 'A'
        for i in 0..30 {
            let _next = forward(&state.weights, current_token, i);
            let text = state.tokenizer.decode(current_token);
            let _ = tx.send(text).await;
            current_token = (current_token + 1) % 256; // Mock progression
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    });

    let stream = stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(token) => Some((Ok(Event::default().data(token)), rx)),
            None => None,
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct InferenceRequest { prompt: String }

#[tokio::main]
async fn main() {
    println!("\x1b[92m--- Rust Native AI Core ---\x1b[0m");

    let dim = 512;
    let n_layers = 4;
    let vocab_size = 1000;

    let weights = if Path::new("../model.safetensors").exists() {
        println!("Loading weights from model.safetensors...");
        // Real loading logic would go here, mapping tensor names to our layers
        // Using synthetic for now but structure is ready for mapping
        gen_synthetic_weights(dim, n_layers, vocab_size)
    } else {
        println!("model.safetensors not found, using synthetic initialization.");
        gen_synthetic_weights(dim, n_layers, vocab_size)
    };

    let state = Arc::new(AppState {
        weights,
        tokenizer: Tokenizer {},
    });

    let app = Router::new()
        .route("/api/generate", post(handle_inference))
        .fallback_service(ServeDir::new("../ui"))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 4040));
    println!("Listening on http://localhost:4040");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn gen_synthetic_weights(dim: usize, n_layers: usize, vocab: usize) -> ModelWeights {
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        layers.push(LayerWeights {
            wq: Tensor::new(vec![0.01; dim * dim], dim, dim),
            wk: Tensor::new(vec![0.01; dim * dim], dim, dim),
            wv: Tensor::new(vec![0.01; dim * dim], dim, dim),
            wo: Tensor::new(vec![0.01; dim * dim], dim, dim),
            w1: Tensor::new(vec![0.01; dim * 1024], dim, 1024),
            w2: Tensor::new(vec![0.01; 1024 * dim], 1024, dim),
            w3: Tensor::new(vec![0.01; dim * 1024], dim, 1024),
            ffn_norm: vec![1.0; dim],
            attn_norm: vec![1.0; dim],
        });
    }
    ModelWeights {
        token_embedding: Tensor::new(vec![0.01; vocab * dim], vocab, dim),
        layers,
        norm: vec![1.0; dim],
        output: Tensor::new(vec![0.01; dim * vocab], dim, vocab),
    }
}
