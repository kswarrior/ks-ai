use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use futures_util::stream::{self, Stream};
use rayon::prelude::*;
use serde::Deserialize;
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl Tensor {
    fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self { data, shape: (rows, cols) }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], shape: (rows, cols) }
    }

    fn get_row(&self, row: usize) -> &[f32] {
        let cols = self.shape.1;
        &self.data[row * cols..(row + 1) * cols]
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

    fn scale_inplace(&mut self, scale: f32) {
        self.data.par_iter_mut().for_each(|x| *x *= scale);
    }

    fn softmax_inplace(&mut self) {
        let cols = self.shape.1;
        self.data.par_chunks_mut(cols).for_each(|row| {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for x in row.iter_mut() {
                *x = (*x - max_val).exp();
                sum += *x;
            }
            for x in row.iter_mut() {
                *x /= sum;
            }
        });
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

impl ModelWeights {
    fn load(path: &str) -> Self {
        let file = std::fs::File::open(path).expect("Failed to open weights file");
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("Failed to mmap weights file") };
        let tensors = safetensors::SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

        let get_tensor = |name: &str| -> Tensor {
            let view = tensors.tensor(name).unwrap_or_else(|_| panic!("Tensor not found: {}", name));
            let data: Vec<f32> = view.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let shape = view.shape();
            Tensor::new(data, shape[0], shape[1])
        };

        let get_vec = |name: &str| -> Vec<f32> {
            let view = tensors.tensor(name).unwrap_or_else(|_| panic!("Tensor not found: {}", name));
            view.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };

        let token_embedding = get_tensor("embeddings");
        let norm = get_vec("norm");
        let output = get_tensor("output");

        let mut layers = Vec::new();
        let mut i = 0;
        while tensors.tensor(&format!("layers.{}.wq", i)).is_ok() {
            layers.push(LayerWeights {
                wq: get_tensor(&format!("layers.{}.wq", i)),
                wk: get_tensor(&format!("layers.{}.wk", i)),
                wv: get_tensor(&format!("layers.{}.wv", i)),
                wo: get_tensor(&format!("layers.{}.wo", i)),
                w1: get_tensor(&format!("layers.{}.w1", i)),
                w2: get_tensor(&format!("layers.{}.w2", i)),
                w3: get_tensor(&format!("layers.{}.w3", i)),
                ffn_norm: get_vec(&format!("layers.{}.ffn_norm", i)),
                attn_norm: get_vec(&format!("layers.{}.attn_norm", i)),
            });
            i += 1;
        }

        Self { token_embedding, layers, norm, output }
    }
}

struct AppState {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
}

struct LayerKVCache {
    k: Tensor,
    v: Tensor,
}

fn forward(weights: &ModelWeights, token_id: u32, pos: usize, kv_cache: &mut [LayerKVCache]) -> Tensor {
    let dim = weights.token_embedding.shape.1;
    let mut x = Tensor::new(weights.token_embedding.get_row(token_id as usize).to_vec(), 1, dim);

    for (i, layer) in weights.layers.iter().enumerate() {
        let mut h = x.clone();
        h.rms_norm(&layer.attn_norm);

        // Attention
        let q = h.matmul(&layer.wq);
        let k = h.matmul(&layer.wk);
        let v = h.matmul(&layer.wv);

        // Update KV cache
        let head_dim = dim / 8; // Assuming 8 heads for this example, adjust as needed or load from config
        let n_heads = 8;

        let layer_cache = &mut kv_cache[i];
        // Store k, v in cache at pos
        for j in 0..dim {
            layer_cache.k.data[pos * dim + j] = k.data[j];
            layer_cache.v.data[pos * dim + j] = v.data[j];
        }

        // Multi-head attention
        let mut attn_out_data = vec![0.0; dim];
        let q_data = &q.data;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h_idx in 0..n_heads {
            let q_head = &q_data[h_idx * head_dim..(h_idx + 1) * head_dim];
            let mut scores = vec![0.0; pos + 1];

            for p in 0..=pos {
                let k_head = &layer_cache.k.data[p * dim + h_idx * head_dim .. p * dim + (h_idx + 1) * head_dim];
                let mut score = 0.0;
                for d in 0..head_dim {
                    score += q_head[d] * k_head[d];
                }
                scores[p] = score * scale;
            }

            // Softmax
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V
            for p in 0..=pos {
                let v_head = &layer_cache.v.data[p * dim + h_idx * head_dim .. p * dim + (h_idx + 1) * head_dim];
                let s = scores[p];
                for d in 0..head_dim {
                    attn_out_data[h_idx * head_dim + d] += s * v_head[d];
                }
            }
        }

        let attn_out_tensor = Tensor::new(attn_out_data, 1, dim);
        let attn_out = attn_out_tensor.matmul(&layer.wo);
        x.add_inplace(&attn_out);

        // FFN
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
    x.matmul(&weights.output)
}

async fn handle_inference(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        let prompt = payload.prompt;
        let encoding = state.tokenizer.encode(prompt, true).expect("Failed to encode prompt");
        let tokens = encoding.get_ids();

        let dim = state.weights.token_embedding.shape.1;
        let n_layers = state.weights.layers.len();
        let max_seq_len = 1024;
        let mut kv_cache: Vec<LayerKVCache> = (0..n_layers)
            .map(|_| LayerKVCache {
                k: Tensor::zeros(max_seq_len, dim),
                v: Tensor::zeros(max_seq_len, dim),
            })
            .collect();

        let mut pos = 0;
        let mut last_token = 0;

        // Prefill
        for &token in &tokens[..tokens.len().saturating_sub(1)] {
            forward(&state.weights, token, pos, &mut kv_cache);
            pos += 1;
        }

        if !tokens.is_empty() {
            last_token = tokens[tokens.len() - 1];
        }

        // Generation loop
        let max_new_tokens = 50;
        for _ in 0..max_new_tokens {
            let logits = forward(&state.weights, last_token, pos, &mut kv_cache);

            // Greedy sampling
            let mut max_logit = f32::NEG_INFINITY;
            let mut next_token = 0;
            for (idx, &logit) in logits.data.iter().enumerate() {
                if logit > max_logit {
                    max_logit = logit;
                    next_token = idx as u32;
                }
            }

            if next_token == 1 { // EOS
                break;
            }

            let decoded = state.tokenizer.decode(&[next_token], true).expect("Failed to decode token");
            let _ = tx.send(decoded).await;

            last_token = next_token;
            pos += 1;
            if pos >= max_seq_len { break; }

            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
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

    let weights = if std::path::Path::new("model.safetensors").exists() {
        ModelWeights::load("model.safetensors")
    } else {
        println!("model.safetensors not found, initializing with dummy weights");
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
        ModelWeights {
            token_embedding: Tensor::new(vec![0.01; vocab_size * dim], vocab_size, dim),
            layers,
            norm: vec![1.0; dim],
            output: Tensor::new(vec![0.01; dim * vocab_size], dim, vocab_size),
        }
    };

    let tokenizer = if std::path::Path::new("tokenizer.json").exists() {
        tokenizers::Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer.json")
    } else {
        panic!("tokenizer.json not found! Please provide a tokenizer.json file.");
    };

    let state = Arc::new(AppState { weights, tokenizer });

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    #[test]
    fn test_weight_loading() {
        let dim = 128;
        let n_layers = 1;
        let vocab_size = 10;

        let mut data: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        data.insert("embeddings".to_string(), (vec![vocab_size, dim], vec![0.1; vocab_size * dim]));
        data.insert("norm".to_string(), (vec![dim], vec![1.0; dim]));
        data.insert("output".to_string(), (vec![dim, vocab_size], vec![0.1; dim * vocab_size]));

        for i in 0..n_layers {
            data.insert(format!("layers.{}.wq", i), (vec![dim, dim], vec![0.1; dim * dim]));
            data.insert(format!("layers.{}.wk", i), (vec![dim, dim], vec![0.1; dim * dim]));
            data.insert(format!("layers.{}.wv", i), (vec![dim, dim], vec![0.1; dim * dim]));
            data.insert(format!("layers.{}.wo", i), (vec![dim, dim], vec![0.1; dim * dim]));
            data.insert(format!("layers.{}.w1", i), (vec![dim, 256], vec![0.1; dim * 256]));
            data.insert(format!("layers.{}.w2", i), (vec![256, dim], vec![0.1; 256 * dim]));
            data.insert(format!("layers.{}.w3", i), (vec![dim, 256], vec![0.1; dim * 256]));
            data.insert(format!("layers.{}.ffn_norm", i), (vec![dim], vec![1.0; dim]));
            data.insert(format!("layers.{}.attn_norm", i), (vec![dim], vec![1.0; dim]));
        }

        let serialized = safetensors::serialize(&data.iter().map(|(k, (s, v))| {
            (k.clone(), safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                s.clone(),
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
            ).unwrap())
        }).collect::<HashMap<_, _>>(), &None).unwrap();

        let mut file = std::fs::File::create("test_weights.safetensors").unwrap();
        file.write_all(&serialized).unwrap();

        let weights = ModelWeights::load("test_weights.safetensors");
        assert_eq!(weights.layers.len(), n_layers);
        assert_eq!(weights.token_embedding.shape, (vocab_size, dim));

        std::fs::remove_file("test_weights.safetensors").unwrap();
    }
}
