#include "inference.h"
#include <llama.h>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <stdexcept>

class InferenceEngine {
public:
    InferenceEngine(const char* model_path, int n_ctx) {
        // Initialize global backend once
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model = llama_load_model_from_file(model_path, model_params);
        if (!model) {
            throw std::runtime_error("Failed to load model");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 512;
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            llama_free_model(model);
            throw std::runtime_error("Failed to create context");
        }

        // Initialize sampler chain
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
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
        if (sampler) llama_sampler_free(sampler);
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
    }

    void infer(const char* prompt, void (*token_callback)(const char*)) {
        std::lock_guard<std::mutex> lock(engine_mutex);

        // Clear context for new generation
        llama_kv_cache_clear(ctx);

        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tokens = tokenize(vocab, prompt, true);

        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch_add(batch, tokens[i], i, { 0 }, i == tokens.size() - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return;
        }

        int n_cur = tokens.size();
        int n_decode = 0;
        const int max_tokens = llama_n_ctx(ctx) - tokens.size();

        while (n_decode < max_tokens) {
            llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, next_token)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
            if (n > 0) {
                buf[n] = '\0';
                token_callback(buf);
            }

            // Prepare next token for decoding
            batch.n_tokens = 0;
            batch_add(batch, next_token, n_cur, { 0 }, true);

            if (llama_decode(ctx, batch) != 0) {
                break;
            }

            n_cur++;
            n_decode++;
        }

        llama_batch_free(batch);
    }

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    std::mutex engine_mutex;

    std::vector<llama_token> tokenize(const struct llama_vocab * vocab, const std::string& text, bool add_special) {
        int n_tokens = text.length() + (add_special ? 1 : 0);
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
        std::cerr << "Engine init error: " << e.what() << std::endl;
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
