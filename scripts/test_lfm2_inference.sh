#!/bin/bash
# Test LFM2 inference on rtx4090 and capture detailed output
set -e

REMOTE="rtx4090"
REMOTE_DIR="~/Projects/Personal/llm/inference-engine/neurogrid-engine"

echo "=== Starting LFM2 test server on $REMOTE ==="
ssh $REMOTE "bash -s" <<'REMOTE_SCRIPT'
cd ~/Projects/Personal/llm/inference-engine/neurogrid-engine
pkill -f neurogrid 2>/dev/null
sleep 1

# Start with debug logging
LD_LIBRARY_PATH=./build nohup ./build/neurogrid \
  --http-port 8091 --gpu 0 \
  --model ./models/lfm2-1.2b-thinking \
  --model-name lfm2-1.2b-thinking \
  --min-peers 0 --log-level debug \
  </dev/null >/tmp/lfm2_server.log 2>&1 &

sleep 10

# Check server is up
if ! pgrep -f "neurogrid.*8091" >/dev/null; then
    echo "ERROR: Server failed to start"
    tail -20 /tmp/lfm2_server.log
    exit 1
fi

echo "Server running. Sending test query..."

# Send query
RESPONSE=$(curl -s -m 60 -X POST http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2-1.2b-thinking","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"temperature":0.01}')

echo "Response: $RESPONSE"

# Show generation log
echo ""
echo "=== Generation Log ==="
grep "Generate.*Step\|Generate.*token\|logits\|prefill\|forward" /tmp/lfm2_server.log | head -30

# Show first step details
echo ""
echo "=== First Steps Detail ==="
grep "Step 0\|Step 1\|Step 2" /tmp/lfm2_server.log

# Cleanup
pkill -f neurogrid 2>/dev/null
REMOTE_SCRIPT
