#ifndef INFERENCE_H
#define INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the AI engine with the specified model.
 * @param model_path Path to the GGUF model file.
 * @param n_ctx Context window size.
 * @return A pointer to the opaque engine instance, or nullptr on failure.
 */
void* init_engine(const char* model_path, int n_ctx);

/**
 * Frees the AI engine and associated resources.
 * @param engine_ptr Pointer to the engine instance.
 */
void free_engine(void* engine_ptr);

/**
 * Runs inference on the given prompt and streams tokens back via a callback.
 * @param engine_ptr Pointer to the engine instance.
 * @param prompt The input prompt.
 * @param token_callback Function pointer called for each generated token string.
 */
void run_inference(void* engine_ptr, const char* prompt, void (*token_callback)(const char*));

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H
