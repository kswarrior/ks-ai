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
            msg = f"C++ Shared Library Binary Missing. Looked in: {[str(p) for p in search_paths]}"
            self._exit_with_error("Dependency Error", msg)

        # 2. Automatic Model Detection
        if not model_path:
            models_dir = self.root_dir / "models"
            gguf_files = []
            if models_dir.exists():
                gguf_files.extend(list(models_dir.glob("*.gguf")))
            gguf_files.extend(list(self.root_dir.glob("*.gguf")))

            if gguf_files:
                model_path = str(gguf_files[0])

        if not model_path or not Path(model_path).exists():
            self._exit_with_error("Model Error", "No .gguf files found in models/ or root.")

        print(f"\033[94m[Binding]\033[0m Library: {lib_path}")
        print(f"\033[94m[Binding]\033[0m Model:   {model_path}")

        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_ctypes()
            self.engine_ptr = self.lib.init_engine(model_path.encode('utf-8'), n_ctx)
            if not self.engine_ptr:
                raise RuntimeError("Native engine initialization failed. Check stderr.")
        except Exception as e:
            self._exit_with_error("Init Error", str(e))

    def _setup_ctypes(self):
        self.lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.init_engine.restype = ctypes.c_void_p
        self.lib.free_engine.argtypes = [ctypes.c_void_p]
        self.lib.free_engine.restype = None
        self.lib.run_inference.argtypes = [ctypes.c_void_p, ctypes.c_char_p, TOKEN_CALLBACK]
        self.lib.run_inference.restype = None

    def _exit_with_error(self, title: str, details: str):
        print(f"\n\033[91m{title}:\033[0m {details}")
        sys.exit(1)

    def __del__(self):
        if hasattr(self, 'engine_ptr') and self.engine_ptr:
            self.lib.free_engine(self.engine_ptr)

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        token_queue = queue.Queue()

        # Consistent prompt structure
        system_prompt = "You are a logical reasoning assistant."
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant: <think>\n"

        yield "<think>\n"

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
                pass

        callback_func = TOKEN_CALLBACK(callback)

        def run():
            try:
                self.lib.run_inference(self.engine_ptr, full_prompt.encode('utf-8'), callback_func)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        try:
            while True:
                token = token_queue.get(timeout=30) # 30s timeout
                if token is None:
                    break
                yield token
        except queue.Empty:
            print("\033[93m[Binding]\033[0m Generation timeout.")
        finally:
            # We don't join because it's a daemon thread and might be stuck in native code
            pass

if __name__ == "__main__":
    try:
        engine = AIEngine()
        for token in engine.generate_stream("Hi"):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"Test failed: {e}")
