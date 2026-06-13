#include "inference.h"
#include <llama.h>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <stdexcept>
#include <cstdio>

class InferenceEngine {
public:
    InferenceEngine(const char* model_path, int n_ctx) {
        fprintf(stderr, "[Native] llama_backend_init()\n");
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        fprintf(stderr, "[Native] llama_model_load_from_file(%s)\n", model_path);
        model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            throw std::runtime_error("Failed to load model from file");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx; // Set batch size large enough for the whole context to simplify
        ctx_params.n_ubatch = n_ctx;
        fprintf(stderr, "[Native] llama_init_from_model(n_ctx=%d)\n", n_ctx);
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("Failed to initialize context from model");
        }

        fprintf(stderr, "[Native] Initializing sampler chain\n");
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
    }

    void batch_add(llama_batch & batch, llama_token id, int32_t pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
        batch.token[batch.n_tokens] = id;
        batch.pos[batch.n_tokens] = pos;
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            batch.seq_id[batch.n_tokens][i] = seq_ids[i];
        }
        batch.logits[batch.n_tokens] = logits;
        batch.n_tokens++;
    }

    ~InferenceEngine() {
        fprintf(stderr, "[Native] Cleaning up...\n");
        if (sampler) llama_sampler_free(sampler);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    void infer(const char* prompt, void (*token_callback)(const char*)) {
        std::lock_guard<std::mutex> lock(engine_mutex);

        // Simple sequence removal/clearing
        llama_kv_cache_seq_rm(ctx, 0, -1, -1);

        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tokens = tokenize(vocab, prompt, true);
        if (tokens.empty()) {
            fprintf(stderr, "[Native] Empty prompt tokens\n");
            return;
        }
        fprintf(stderr, "[Native] Prompt tokens: %zu\n", tokens.size());

        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch_add(batch, tokens[i], i, { 0 }, i == tokens.size() - 1);
        }

        fprintf(stderr, "[Native] Decoding prompt...\n");
        int res = llama_decode(ctx, batch);
        if (res != 0) {
            fprintf(stderr, "[Native] llama_decode failed: %d\n", res);
            llama_batch_free(batch);
            return;
        }

        int n_cur = tokens.size();
        int n_decode = 0;
        const int max_tokens = llama_n_ctx(ctx) - tokens.size();

        fprintf(stderr, "[Native] Starting generation loop\n");
        while (n_decode < max_tokens) {
            llama_token next_token = llama_sampler_sample(sampler, ctx, -1);
            llama_sampler_accept(sampler, next_token);

            if (llama_vocab_is_eog(vocab, next_token)) {
                fprintf(stderr, "[Native] EOG reached\n");
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                std::string piece(buf, n);
                token_callback(piece.c_str());
            }

            batch.n_tokens = 0;
            batch_add(batch, next_token, n_cur, { 0 }, true);

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[Native] Loop decode failed\n");
                break;
            }

            n_cur++;
            n_decode++;
            if (n_decode % 10 == 0) fprintf(stderr, ".");
        }
        fprintf(stderr, "\n[Native] Generation done. Decoded %d tokens.\n", n_decode);

        llama_batch_free(batch);
    }

private:
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    struct llama_sampler* sampler = nullptr;
    std::mutex engine_mutex;

    std::vector<llama_token> tokenize(const struct llama_vocab * vocab, const std::string& text, bool add_special) {
        int n_tokens = text.length() + 32;
        std::vector<llama_token> res(n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), res.data(), res.size(), add_special, true);
        if (n_tokens < 0) {
            res.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), res.data(), res.size(), add_special, true);
        } else {
            res.resize(n_tokens);
        }
        return res;
    }
};

extern "C" {

void* init_engine(const char* model_path, int n_ctx) {
    try {
        return new InferenceEngine(model_path, n_ctx);
    } catch (const std::exception& e) {
        fprintf(stderr, "[Native Exception] %s\n", e.what());
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void free_engine(void* engine_ptr) {
    if (engine_ptr) {
        delete static_cast<InferenceEngine*>(engine_ptr);
    }
}

void run_inference(void* engine_ptr, const char* prompt, void (*token_callback)(const char*)) {
    if (engine_ptr && prompt && token_callback) {
        static_cast<InferenceEngine*>(engine_ptr)->infer(prompt, token_callback);
    }
}

}
