# Qwen2.5 Migration Guide

This project now uses Qwen2.5 as the primary local LLM for inference tasks.
The goal is better grounding, stronger instruction following, and more stable educational reasoning.

## Architecture

The LLM stack is split into three layers:

- `rag/model_loader.py`: backend selection and model loading.
- `rag/llm.py`: inference facade with timeout handling, logging, and backward-compatible `generate()` calls.
- `rag/prompts.py`: centralized prompt builders for grounded educational workflows.

The rest of the system keeps the current routing/orchestration split:

- `router/intent_classifier.py`
- `router/orchestrator.py`
- `engines/structured_engine.py`
- `engines/semantic_engine.py`
- `engines/topic_graph_engine.py`
- `retrieval/vector_store.py`

## Supported backends

- `transformers` for local Hugging Face Qwen2.5 checkpoints.
- `llama_cpp` for GGUF-quantized Qwen2.5 models.

### Recommended GGUF option

For limited hardware, use a Qwen2.5 7B Instruct GGUF quantization such as:

- `Q4_K_M` for a balanced speed/quality trade-off.
- `Q5_K_M` if you have more RAM/VRAM.
- `Q8_0` only if you want higher quality and have the memory headroom.

If GPU offload is available, set `LLM_GPU_LAYERS` to a positive value.

## Example configuration

```env
# Primary LLM
QWEN_MODEL_PATH=C:\models\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m.gguf
LLM_BACKEND=llama_cpp
LLM_DEVICE=cpu
LLM_CONTEXT_LENGTH=8192
LLM_MAX_NEW_TOKENS=1024
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
LLM_GPU_LAYERS=0
LLM_NUM_THREADS=8
LLM_QUANTIZATION=auto
LLM_ENABLE_STREAMING=1
LLM_TIMEOUT_SECONDS=120
LLM_SYSTEM_PROMPT=You are a grounded educational assistant. Only answer using retrieved course data. If information is missing, explicitly say so.

# Existing retrieval settings still apply
EMBEDDING_MODEL_PATH=./models/minilm
RAG_TOP_K=3
RAG_DEVICE=cpu
```

## Prompt strategy

The prompt builders are intentionally strict:

- Grounded answering: only use retrieved context.
- Retrieval-based summarization: summarize source material, not assistant output.
- Topic reasoning: only trust the topic graph.
- Prerequisite reasoning: refuse if the graph evidence is weak.

Example system prompts used by the model:

- `You are a grounded educational assistant.`
- `Only answer using retrieved course data.`
- `If information is missing, explicitly say so.`

## Migration steps

1. Download or point to a Qwen2.5 Instruct checkpoint.
2. Set `QWEN_MODEL_PATH` and choose `LLM_BACKEND`.
3. For GGUF, install `llama-cpp-python` and use a `.gguf` model file.
4. Keep the embedding model path unchanged for retrieval.
5. Run the focused tests and then the batch stress test.

## Performance notes

- Use GGUF quantization for constrained hardware.
- Prefer `Q4_K_M` for a good quality/speed balance.
- Increase `LLM_NUM_THREADS` on CPU-only systems.
- Set `LLM_GPU_LAYERS` when using llama.cpp with GPU offload.
- Keep `LLM_MAX_NEW_TOKENS` modest for routine Q&A.
- Reduce prompt/context size if retrieval gets too large.

## Debugging

The loader and inference facade now log:

- model load time
- inference latency
- prompt character count
- estimated prompt and response token count
- timeout events

If model loading fails, the code returns a safe grounded fallback instead of fabricating an answer.
