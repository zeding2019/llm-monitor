#!/usr/bin/env bash
# 间断发请求给 SGLang。用法:
#   bash curl_sglang_loop.sh <model> [间隔秒] [次数]
# 例:bash curl_sglang_loop.sh Qwen2.5-7B-Instruct 3 20   # 每3秒1个,共20个

URL="${SGLANG_URL:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${1:?用法: bash $0 <model> [间隔秒] [次数]}"
INTERVAL="${2:-3}"
REPEAT="${3:-20}"   # 0 = 无限

echo "→ $URL  每 ${INTERVAL}s 一次,共 $REPEAT 次 (0=无限)"
for ((i=1; REPEAT==0 || i<=REPEAT; i++)); do
  echo "[$(date +%T)] #$i"
  curl -sS "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"第 $i 次测试,请简短回答。\"}],\"max_tokens\":64}" \
    -o /dev/null -w "  HTTP %{http_code} · %{time_total}s\n"
  [ "$REPEAT" -eq 0 ] || [ "$i" -lt "$REPEAT" ] && sleep "$INTERVAL"
done
echo "完成"
