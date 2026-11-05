#!/usr/bin/env bash
# v31.6.1冻结评估入口：一次启动一个套件；自动建manifest但永不覆盖现有实验。
# usage: bash v3161_frozen_eval30/run.sh fresh
# help:  python3 -S v3161_frozen_eval30/eval_v3161.py --help
set -euo pipefail
SUITE="${1:?用法: bash v3161_frozen_eval30/run.sh fresh}"
[[ "$SUITE" == fresh ]] || { echo '[ERROR] 只允许fresh' >&2; exit 2; }
ROOT="$(realpath -m "${ENO_HOME:-/home/dgx/eno-isaac}")"
cd "$ROOT"
SCRIPT="$ROOT/v3161_frozen_eval30/eval_v3161.py"
[[ -s "$SCRIPT" ]] || { echo '[ERROR] 缺少完整包，请先解压在eno-isaac根目录' >&2; exit 2; }
GROUP="$(python3 -S - "$ROOT/v3161_frozen_eval30/seed_plan.json" "$SUITE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf8'))
print(p[sys.argv[2]]['group'])
PY
)"
if [[ ! -e "$ROOT/output/v3161_frozen_eval/$GROUP/manifest.json" ]]; then
  python3 -S "$SCRIPT" prepare "$SUITE" --root "$ROOT"
fi
python3 -S "$SCRIPT" run "$SUITE" --root "$ROOT"
