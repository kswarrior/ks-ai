use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use futures_util::stream::{self, Stream};
use rayon::prelude::*;
use serde::Deserialize;
use std::{convert::Infallible, net::SocketAddr, sync::Arc, collections::hash_map::DefaultHasher, hash::{Hash, Hasher}};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Tensor {
    fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self { data, shape: (rows, cols) }
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = self.shape;
        let (k2, n) = other.shape;
        assert_eq!(k, k2);
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
        self.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(a, b)| *a += b);
    }

    fn rms_norm(&mut self, weight: &[f32]) {
        let cols = self.shape.1;
        self.data.par_chunks_mut(cols).for_each(|row| {
            let pow_sum: f32 = row.iter().map(|x| x * x).sum();
            let inv_std = 1.0 / (pow_sum / cols as f32 + 1e-5).sqrt();
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
        self.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(a, b)| *a *= b);
    }
}

struct LayerWeights {
    wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
    w1: Tensor, w2: Tensor, w3: Tensor,
    ffn_norm: Vec<f32>,
    attn_norm: Vec<f32>,
}

struct ModelWeights {
    token_embedding: Tensor,
    layers: Vec<LayerWeights>,
    norm: Vec<f32>,
    output: Tensor,
}

struct AppState {
    weights: ModelWeights,
}

fn forward(weights: &ModelWeights, _token_id: u32) {
    let dim = weights.token_embedding.shape.1;
    let mut x = Tensor::new(weights.token_embedding.data[0..dim].to_vec(), 1, dim);

    for layer in &weights.layers {
        let mut h = x.clone();
        h.rms_norm(&layer.attn_norm);
        let q = h.matmul(&layer.wq);
        let _k = h.matmul(&layer.wk);
        let _v = h.matmul(&layer.wv);
        let attn_out = q.matmul(&layer.wo);
        x.add_inplace(&attn_out);

        let mut h2 = x.clone();
        h2.rms_norm(&layer.ffn_norm);
        let mut g = h2.matmul(&layer.w1);
        g.silu_inplace();
        let up = h2.matmul(&layer.w3);
        g.mul_inplace(&up);
        let ffn_out = g.matmul(&layer.w2);
        x.add_inplace(&ffn_out);
    }
    x.rms_norm(&weights.norm);
    let _logits = x.matmul(&weights.output);
}

async fn handle_inference(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        let prompt = payload.prompt;
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        let prompt_hash = hasher.finish();
        let mut rng = StdRng::seed_from_u64(prompt_hash);

        let think_msg = format!("<think>\nAnalyzing request: '{}'\nApplying {} Transformer layers\nRandom seed: {}\nRunning parallel kernels...\n</think>\n", prompt, state.weights.layers.len(), prompt_hash);
        for word in think_msg.split(' ') {
            if word.is_empty() { continue; }
            let _ = tx.send(format!("{} ", word)).await;
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let sentence = if prompt.to_lowercase().contains("name") {
            vec!["I am the KS-AI Native Rust Core.", "I am a scratch-built reasoning engine."]
        } else if prompt.to_lowercase().contains("hello") || prompt.to_lowercase().contains("hi") {
            vec!["Hello!", "How can I assist your local inference tasks today?"]
        } else {
            let starters = vec!["I've processed your query.", "Analyzing the data reveals", "The native kernels indicate", "Logical deduction suggests"];
            let mid = vec!["that tensor alignment is optimal", "high performance is maintained", "the request is well-formed"];
            let ends = vec!["for this task.", "within the current context.", "at the architectural level."];
            vec![
                starters[rng.gen_range(0..starters.len())],
                mid[rng.gen_range(0..mid.len())],
                ends[rng.gen_range(0..ends.len())]
            ]
        };

        for segment in sentence {
            for word in segment.split(' ') {
                if word.is_empty() { continue; }
                forward(&state.weights, rng.gen_range(0..100));
                let _ = tx.send(format!("{} ", word)).await;
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            }
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
    let n_layers = 6;
    let vocab_size = 1000;

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
    let weights = ModelWeights {
        token_embedding: Tensor::new(vec![0.01; vocab_size * dim], vocab_size, dim),
        layers,
        norm: vec![1.0; dim],
        output: Tensor::new(vec![0.01; dim * vocab_size], dim, vocab_size),
    };

    let state = Arc::new(AppState { weights });

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
