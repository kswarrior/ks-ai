# AI Core Inference Project

A high-performance, low-latency AI inference system bridging a native C++ engine (using `llama.cpp`) with a concurrent Python wrapper. Optimized for reasoning models like Phi-4 and DeepSeek-R1.

## Project Structure

- `core_ai/`: Native C++ engine and Python bindings.
  - `inference.h/cpp`: C++ inference logic using modern `llama.cpp` samplers.
  - `CMakeLists.txt`: Build configuration.
  - `binding.py`: Python `ctypes` wrapper with streaming support.
- `ui/`: Simple Web UI for testing.
  - `index.html`: Responsive chat interface.
  - `server.py`: Python development server.

## Prerequisites

- CMake (3.10+)
- C++17 compiler (GCC/Clang/MSVC)
- `llama.cpp` shared libraries and headers.
- Python 3.7+

## Building the Native Library

1. Ensure you have `llama.cpp` compiled or installed.
2. Navigate to `core_ai/`:
   ```bash
   cd core_ai
   mkdir build && cd build
   cmake .. -DLLAMA_PATH=/path/to/llama.cpp
   make
   ```
3. This will generate `libai_core.so` (or `ai_core.dll`).

## Running the Web UI

1. Set the environment variables for the shared library and your GGUF model:
   ```bash
   export AI_LIB_PATH=$(pwd)/core_ai/build/libai_core.so
   export AI_MODEL_PATH=/path/to/your/model-q4_k_m.gguf
   ```
2. Run the server:
   ```bash
   python3 ui/server.py
   ```
3. Open `http://localhost:4040` in your browser.

## Features

- **Low Latency**: Native C++ execution with minimal overhead.
- **Streaming**: Token-by-token streaming via Python generators and C callbacks.
- **Chain-of-Thought (CoT)**: Integrated system prompt to force deep reasoning in compatible models.
- **Thread-Safe**: Multiple requests are synchronized at the engine level.

## Recommended Models

- **Phi-4-mini-instruct**: Excellent for lightweight reasoning.
- **DeepSeek-R1-Distill-Qwen-7B**: State-of-the-art reasoning capabilities in a small package.
