#!/bin/bash

# Configuration - Update these or pass as environment variables
MODEL_PATH="${AI_MODEL_PATH:-}"
LIB_PATH="${AI_LIB_PATH:-$(pwd)/core_ai/build/libai_core.so}"

# Check if model path is provided (Optional, AIEngine will auto-detect if not set)
if [ -z "$MODEL_PATH" ]; then
    echo "AI_MODEL_PATH not set. AIEngine will attempt to auto-detect a .gguf model in models/ or the root directory."
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Check if library exists
if [ ! -f "$LIB_PATH" ]; then
    echo "Warning: Shared library not found at $LIB_PATH"
    echo "Did you build the project in core_ai/build?"

    # Try to find it in the current directory if it was built elsewhere
    ALT_LIB=$(find core_ai -name "libai_core.so" -o -name "ai_core.so" | head -n 1)
    if [ -n "$ALT_LIB" ]; then
        echo "Found library at $ALT_LIB. Using it."
        LIB_PATH="$(pwd)/$ALT_LIB"
    else
        exit 1
    fi
fi

echo "Starting AI Core Inference System..."
echo "Model: $MODEL_PATH"
echo "Library: $LIB_PATH"
echo "UI URL: http://localhost:4040"

# Export variables for the Python server
export AI_MODEL_PATH="$MODEL_PATH"
export AI_LIB_PATH="$LIB_PATH"

# Run the UI server
# This will also load the AI engine when the first request comes in via the binding
python3 ui/server.py
