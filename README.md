# go-do

A thin Go wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp) (`libllama`) that translates natural language tasks into shell commands and executes them.

```
go-do 'List all files in this directory over 1 GB'
go-do 'rsync directory x with directory y and show what changed'
go-do 'Figure out what changed here, write a detailed commit message, then commit and push to origin'
```

## How it works

1. Your task description is sent to a local LLM (via `libllama`).
2. The model generates the shell commands needed to accomplish the task.
3. You review them (or skip review with `--yes`), then `go-do` runs them via `bash`.

## Prerequisites

* A Linux or macOS machine with GCC/Clang, CMake ≥ 3.21, and Go ≥ 1.21.
* A GGUF model file (e.g. [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)).

## Build

```bash
# Clone this repo
git clone https://github.com/devlux76/go-do.git
cd go-do

# Build llama.cpp (auto-detects CUDA / ROCm / Vulkan / Metal / CPU) and go-do
./buildall.sh
```

The script:
- Auto-detects the best GPU backend (CUDA → ROCm → Vulkan → Metal → CPU).
- Builds `llama.cpp` as a shared library and installs it to `/usr/local/lib`.
- Creates `/usr/share/models/` as a shared model directory.
- Compiles the `go-do` binary in the current directory.

You can override the backend with `LLAMA_BACKEND=VULKAN ./buildall.sh`.

## Install a model

Place any GGUF model at `/usr/share/models/default.gguf`, or point to it at runtime:

```bash
cp ~/Downloads/mistral-7b-instruct-v0.3.Q4_K_M.gguf /usr/share/models/default.gguf
# or
export LLAMA_MODEL_PATH=/path/to/your/model.gguf
```

## Usage

```
Usage: go-do [options] <task description>

Options:
  -model string      Path to GGUF model file (default: $LLAMA_MODEL_PATH or /usr/share/models/default.gguf)
  -ctx int           Context window size in tokens (0 = auto-detect)
  -temp float        Sampling temperature — 0 for greedy/deterministic (default 0.7)
  -max-tokens int    Maximum tokens to generate (default 2048)
  -dry-run           Print generated commands without executing them
  -yes               Auto-confirm execution without prompting

Environment variables:
  LLAMA_MODEL_PATH   Path to GGUF model file
  LLAMA_N_CTX        Context window size override
```

### Examples

```bash
# List large files
./go-do 'List all files in this directory over 1 GB'

# Preview without running
./go-do --dry-run 'Figure out what changed here, write a commit message, then commit and push'

# Run without confirmation
./go-do --yes 'Remove all .pyc files recursively from the current directory'

# Use a specific model with a custom context size
./go-do --model /opt/models/llama3.gguf --ctx 8192 'Show disk usage by subdirectory, sorted largest first'
```

## Context window auto-detection

`go-do` automatically selects the context window size by:

1. Reading the context length the model was trained with (`llama_model_n_ctx_train`).
2. Capping it at what available system RAM can comfortably hold (~512 KB per token).
3. Flooring at **2048 tokens** regardless.

Override with `--ctx N` or `LLAMA_N_CTX=N`.
