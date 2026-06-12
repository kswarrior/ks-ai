import ctypes
import os
import queue
import threading
import sys
from pathlib import Path
from typing import Generator, Optional

# Define the callback type
TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

class AIEngine:
    def __init__(self, lib_path: Optional[str] = None, model_path: Optional[str] = None, n_ctx: int = 2048):
        self.root_dir = Path(__file__).parent.parent.absolute()

        # 1. Automatic Library Detection
        if not lib_path:
            lib_name = "libai_core.so" if sys.platform != "win32" else "libai_core.dll"
            search_paths = [
                self.root_dir / lib_name,
                self.root_dir / "core_ai" / lib_name,
                self.root_dir / "core_ai" / "build" / lib_name,
                self.root_dir / "build" / lib_name
            ]
            for path in search_paths:
                if path.exists():
                    lib_path = str(path)
                    break

        if not lib_path or not Path(lib_path).exists():
            self._exit_with_error("C++ Shared Library Binary Missing",
                f"Looked in: {[str(p) for p in search_paths]}\n"
                "Please compile the engine first:\n"
                "  cd core_ai && mkdir -p build && cd build && cmake .. && make")

        # 2. Automatic Model Detection
        if not model_path:
            models_dir = self.root_dir / "models"
            if models_dir.exists():
                gguf_files = list(models_dir.glob("*.gguf"))
                if gguf_files:
                    model_path = str(gguf_files[0])

            # Check root if not found in models/
            if not model_path:
                gguf_files = list(self.root_dir.glob("*.gguf"))
                if gguf_files:
                    model_path = str(gguf_files[0])

        if not model_path or not Path(model_path).exists():
            self._exit_with_error("GGUF Model File Missing",
                f"No .gguf files found in '{self.root_dir}/models/' or root directory.\n"
                "Please place a reasoning model (e.g., phi-4.gguf) in the models/ folder.")

        print(f"\033[94m[AI Engine]\033[0m Loading Library: {lib_path}")
        print(f"\033[94m[AI Engine]\033[0m Loading Model:   {model_path}")

        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_ctypes()
            self.engine_ptr = self.lib.init_engine(model_path.encode('utf-8'), n_ctx)
            if not self.engine_ptr:
                raise RuntimeError("Native init_engine returned null")
        except Exception as e:
            self._exit_with_error("Engine Initialization Failed", str(e))

    def _setup_ctypes(self):
        self.lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.init_engine.restype = ctypes.c_void_p
        self.lib.free_engine.argtypes = [ctypes.c_void_p]
        self.lib.free_engine.restype = None
        self.lib.run_inference.argtypes = [ctypes.c_void_p, ctypes.c_char_p, TOKEN_CALLBACK]
        self.lib.run_inference.restype = None

    def _exit_with_error(self, title: str, details: str):
        print("\n" + "="*60)
        print(f"\033[91mCRITICAL ERROR: {title}\033[0m")
        print("-"*60)
        print(details)
        print("="*60 + "\n")
        sys.exit(1)

    def __del__(self):
        if hasattr(self, 'engine_ptr') and self.engine_ptr:
            self.lib.free_engine(self.engine_ptr)

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        token_queue = queue.Queue()

        system_prompt = (
            "System Prompt: You are an elite, hyper-logical reasoning engine. "
            "Approach every query by first detailing a meticulous, step-by-step psychological analytical "
            "thought process within <think> tags before arriving at your definitive final response."
        )
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        # Buffer to handle multi-byte UTF-8 characters split across tokens
        byte_buffer = bytearray()

        def callback(token: bytes):
            nonlocal byte_buffer
            byte_buffer.extend(token)
            try:
                decoded = byte_buffer.decode('utf-8')
                if decoded:
                    token_queue.put(decoded)
                    byte_buffer.clear()
            except UnicodeDecodeError:
                # Character might be incomplete, wait for next token
                pass

        callback_func = TOKEN_CALLBACK(callback)

        def run():
            try:
                self.lib.run_inference(self.engine_ptr, full_prompt.encode('utf-8'), callback_func)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=run)
        thread.start()

        try:
            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token
        finally:
            thread.join()

if __name__ == "__main__":
    # Test auto-detection
    try:
        engine = AIEngine()
        for token in engine.generate_stream("What is 2+2?"):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"Test failed: {e}")
