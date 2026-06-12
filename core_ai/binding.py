import ctypes
import os
import queue
import threading
from typing import Generator

# Define the callback type
TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

class AIEngine:
    def __init__(self, lib_path: str, model_path: str, n_ctx: int = 2048):
        if not os.path.exists(lib_path):
            # Try with lib prefix if not found
            dir_name = os.path.dirname(lib_path)
            base_name = os.path.basename(lib_path)
            if not base_name.startswith('lib'):
                lib_path_alt = os.path.join(dir_name, 'lib' + base_name)
                if os.path.exists(lib_path_alt):
                    lib_path = lib_path_alt

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found at {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # Configure argument and return types
        self.lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.init_engine.restype = ctypes.c_void_p

        self.lib.free_engine.argtypes = [ctypes.c_void_p]
        self.lib.free_engine.restype = None

        self.lib.run_inference.argtypes = [ctypes.c_void_p, ctypes.c_char_p, TOKEN_CALLBACK]
        self.lib.run_inference.restype = None

        self.engine_ptr = self.lib.init_engine(model_path.encode('utf-8'), n_ctx)
        if not self.engine_ptr:
            raise RuntimeError("Failed to initialize native AI engine")

    def __del__(self):
        if hasattr(self, 'engine_ptr') and self.engine_ptr:
            self.lib.free_engine(self.engine_ptr)

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        token_queue = queue.Queue()

        # System prompt for Chain-of-Thought
        system_prompt = (
            "System Prompt: You are an elite, hyper-logical reasoning engine. "
            "Approach every query by first detailing a meticulous, step-by-step psychological analytical "
            "thought process within <think> tags before arriving at your definitive final response."
        )

        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        def callback(token: bytes):
            token_queue.put(token.decode('utf-8'))

        callback_func = TOKEN_CALLBACK(callback)

        def run():
            self.lib.run_inference(self.engine_ptr, full_prompt.encode('utf-8'), callback_func)
            token_queue.put(None) # Signal end of stream

        thread = threading.Thread(target=run)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

if __name__ == "__main__":
    # Example usage (assuming library is compiled)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python binding.py <lib_path> <model_path>")
        sys.exit(1)

    engine = AIEngine(sys.argv[1], sys.argv[2])
    print("Engine initialized. Starting inference...")

    for token in engine.generate_stream("Why is the sky blue?"):
        print(token, end="", flush=True)
    print()
