#!/bin/bash
# Start one Qwen3.8-Flash-Next-FP8 SGLang server (TP=2, 32k context) the way the published sweeps
# served it. usage: tools/serve_qwen.sh GPUS PORT [NAME]    e.g. tools/serve_qwen.sh 4,5 8081
# Needs docker + the NVIDIA runtime; HF_HOME (default ~/.cache/huggingface) holds or receives the
# weights. At MEM 0.95 a TP=2 server fills both GPUs: sims need GPUs of their own.
set -e
GPUS=${1:?GPUS, e.g. 4,5}; PORT=${2:?PORT, e.g. 8081}; NAME=${3:-qwen38-$PORT}
HF=${HF_HOME:-$HOME/.cache/huggingface}; IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:dev-qwen38flashnext}
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc host --shm-size 32g -p "$PORT:8000" \
  -v "$HF:$HF:ro" -e HF_HOME="$HF" -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 "$IMAGE" \
  python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-Flash-Next-FP8 --served-model-name qwen \
  --tp 2 --ep 2 --host 0.0.0.0 --port 8000 --mem-fraction-static 0.95 --context-length 32768 \
  --chunked-prefill-size 2048 --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 --reasoning-parser auto --trust-remote-code \
  --max-running-requests 8 --cuda-graph-max-bs 2 --speculative-algorithm NEXTN \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
echo "started $NAME on GPUs $GPUS, http://127.0.0.1:$PORT/v1 (curl it until /v1/models answers)"
