#!/bin/bash
# Download Llama models from HuggingFace Hub
#
# This script wraps the Go download CLI with convenience features.
#
# Prerequisites:
# - For gated models (Llama 2+): Accept license at https://huggingface.co/meta-llama
# - Set HF_TOKEN environment variable or pass --token
#
# Usage:
#   ./download_model.sh [model_alias|repo_id] [output_dir]
#
# Examples:
#   ./download_model.sh                           # Default: tinyllama
#   ./download_model.sh tinyllama                 # TinyLlama 1.1B (public, no token needed)
#   ./download_model.sh llama7b                   # Llama 2 7B (requires HF_TOKEN)
#   ./download_model.sh llama7b-chat ./models     # Llama 2 7B Chat to ./models
#   ./download_model.sh meta-llama/Llama-2-13b-hf # Full repo ID
#
# Supported aliases:
#   tinyllama     -> TinyLlama/TinyLlama-1.1B-Chat-v1.0 (2.2 GB, public)
#   llama7b       -> meta-llama/Llama-2-7b-hf (13 GB, gated)
#   llama7b-chat  -> meta-llama/Llama-2-7b-chat-hf (13 GB, gated)
#   llama13b      -> meta-llama/Llama-2-13b-hf (26 GB, gated)
#   llama13b-chat -> meta-llama/Llama-2-13b-chat-hf (26 GB, gated)

set -e

# Default values
MODEL="${1:-tinyllama}"
OUTPUT="${2:-./models}"

# Script directory for finding Go CLI
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "NeuroGrid Model Downloader"
echo "=========================="
echo ""
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo ""

# Check if this is a gated model that needs HF_TOKEN
check_gated_model() {
    case "$1" in
        llama*|meta-llama/*)
            if [[ -z "$HF_TOKEN" ]]; then
                echo "Warning: $1 is a gated model that requires authentication."
                echo ""
                echo "To download gated models:"
                echo "  1. Create a HuggingFace account at https://huggingface.co"
                echo "  2. Accept the model license at:"
                echo "     https://huggingface.co/meta-llama/Llama-2-7b-hf"
                echo "  3. Create an access token at:"
                echo "     https://huggingface.co/settings/tokens"
                echo "  4. Set the token:"
                echo "     export HF_TOKEN=your_token_here"
                echo ""
                exit 1
            fi
            ;;
    esac
}

# Check for Go CLI binary
find_go_cli() {
    # Check if pre-built binary exists
    if [[ -x "$BUILD_DIR/download" ]]; then
        echo "$BUILD_DIR/download"
        return 0
    fi

    # Check if Go is available to run directly
    if command -v go &> /dev/null; then
        echo "go run $PROJECT_ROOT/cmd/download"
        return 0
    fi

    # Fallback: try to use Python huggingface_hub
    if command -v python3 &> /dev/null && python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "python"
        return 0
    fi

    echo ""
    return 1
}

# Run with Go CLI or fallback
run_download() {
    local cli
    cli=$(find_go_cli)

    if [[ -z "$cli" ]]; then
        echo "Error: No download method available."
        echo ""
        echo "Please install one of:"
        echo "  - Go: https://go.dev/doc/install"
        echo "  - Python huggingface_hub: pip install huggingface_hub"
        echo ""
        echo "Or build the download binary:"
        echo "  make build-download"
        exit 1
    fi

    if [[ "$cli" == "python" ]]; then
        run_python_fallback "$MODEL" "$OUTPUT"
    elif [[ "$cli" == go\ run* ]]; then
        # Run with go run
        $cli --repo "$MODEL" --output "$OUTPUT" ${HF_TOKEN:+--token "$HF_TOKEN"}
    else
        # Run pre-built binary
        "$cli" --repo "$MODEL" --output "$OUTPUT" ${HF_TOKEN:+--token "$HF_TOKEN"}
    fi
}

# Python fallback using huggingface_hub
run_python_fallback() {
    local model="$1"
    local output="$2"

    # Resolve alias to full repo ID
    case "$model" in
        tinyllama)     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
        llama7b)       model="meta-llama/Llama-2-7b-hf" ;;
        llama7b-chat)  model="meta-llama/Llama-2-7b-chat-hf" ;;
        llama13b)      model="meta-llama/Llama-2-13b-hf" ;;
        llama13b-chat) model="meta-llama/Llama-2-13b-chat-hf" ;;
    esac

    echo "Using Python fallback (huggingface_hub)..."
    echo ""

    python3 << EOF
from huggingface_hub import snapshot_download
import os

model_name = "$model"
output_dir = "$output"

print(f"Downloading {model_name}...")

path = snapshot_download(
    repo_id=model_name,
    local_dir=output_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=["*.bin"],  # Skip old pytorch_model.bin format
)

print(f"\nDownloaded to: {path}")

# List downloaded files
print("\nDownloaded files:")
total_size = 0
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f'{subindent}{file} ({size_mb:.1f} MB)')

print(f"\nTotal: {total_size / (1024*1024*1024):.2f} GB")
EOF
}

# Main execution
check_gated_model "$MODEL"
run_download

echo ""
echo "Download complete!"
echo ""
echo "Next steps:"
echo "  - Generate golden data: make golden MODEL=$OUTPUT"
echo "  - Run tests: make test-e2e"
