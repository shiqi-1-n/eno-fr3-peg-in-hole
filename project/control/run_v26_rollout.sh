#!/usr/bin/env bash
# v26 / DGX Spark (aarch64) / IsaacLab 2.3.1
# 任务: 单个独立 PhysX 进程创建随机棒安装+随机孔场景，收集冻结PPO策略的轨迹。
# 注意: 不在真实机器人上运行；训练期间不要把离线真值送入 Actor。
set -euo pipefail
ROOT="$(realpath -m "${ENO_HOME:-$HOME/eno-isaac}")"
IMAGE="${ENO_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.1}"
TAG="${1:?用法: bash scripts/v31_6_1_prelock_eval/run_v26_rollout.sh TAG SEED PPO_CKPT}"
SEED="${2:?缺少SEED}"
CKPT="$(realpath -m "${3:?缺少checkpoint}")"
STEPS="${ENO_V1717_EPISODE_STEPS:-900}"
COARSE="${ENO_V17_COARSE_MODEL:-$ROOT/output/v17_4_bc_models/coarse_bc200_1/bc_actor_best.pt}"
PROFILE="${ENO_V1717_GUARD_PROFILE:-training}"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]{1,50}$ ]] || { echo '非法TAG' >&2;exit 2; }
[[ "$SEED" =~ ^[0-9]{1,10}$ && "$STEPS" =~ ^[0-9]{3,4}$ ]] || exit 2
((STEPS>=300 && STEPS<=2500)) || { echo 'episode步数必须300-2500' >&2;exit 2; }
[[ "$PROFILE" == training || "$PROFILE" == legacy ]] || exit 2
[[ -s "$CKPT" && -s "$COARSE" ]] || { echo 'checkpoint缺失' >&2;exit 2; }
[[ "${ENO_V17_6_1_POSITIVE_PROBE:-0}" == 0 ]] || { echo '禁止positive probe混入训练' >&2;exit 2; }
for dep in 50_phase05b2b_v16_13_online_evidence.py prepare_v17_9_rl.py \
  prepare_v17_6_contact_trial.py prepare_v17_5_full_bc_eval.py \
  prepare_v17_4_spawn_consistent.py prepare_v17_3_coarse.py \
  prepare_v17_2_sensor_capture.py prepare_v17_random_teacher.py \
  prepare_v17_scene_probe.py v17_teacher_extension.txt v17_3_eval_extension.txt \
  v17_5_full_eval_extension.txt v17_6_1_hold_and_probe_extension.txt \
  v31_6_1_prelock_eval/v26_rl_extension.txt v31_6_1_prelock_eval/v26_camera_hook.py v17_17_hole_refine.py v17_3_hole_sensor.py train_v17_3_coarse_bc.py v31_6_1_prelock_eval/v26_ppo.py; do
  [[ -s "$ROOT/scripts/$dep" ]] || { echo "依赖缺失：$ROOT/scripts/$dep" >&2;exit 2; }
done
OUT="$ROOT/output/v26_2_zero_ppo_rollouts/$TAG"
[[ ! -e "$OUT" ]] || { echo "禁止覆盖输出：$OUT" >&2;exit 2; }
mkdir -p "$OUT"
SCRIPT="$ROOT/scripts/v31_6_1_prelock_eval"
TMP="$(mktemp -d "$ROOT/.v17_17_XXXXXXXX")"
NAME="eno-v1717-${TAG:0:13}-$$"
cleanup(){ docker stop -t 5 "$NAME" >/dev/null 2>&1 || true; rm -rf "$TMP"; }
trap cleanup EXIT INT TERM
python3 -S "$ROOT/scripts/prepare_v17_9_rl.py" \
  --source "$ROOT/scripts/50_phase05b2b_v16_13_online_evidence.py" \
  --target "$TMP/rl_base.py" \
  --teacher-extension "$ROOT/scripts/v17_teacher_extension.txt" \
  --eval-extension "$ROOT/scripts/v17_3_eval_extension.txt" \
  --full-extension "$ROOT/scripts/v17_5_full_eval_extension.txt" \
  --contact-extension "$ROOT/scripts/v17_6_1_hold_and_probe_extension.txt" \
  --rl-extension "$SCRIPT/v26_rl_extension.txt"
python3 -S "$SCRIPT/prepare_v26.py" --base "$TMP/rl_base.py" --out "$TMP/rl_generated.py"
[[ -s "$TMP/rl_generated.py" ]] || { echo '[V26_INFRA] 缺少生成文件' >&2; exit 86; }
printf -v CMD '%q ' ./isaaclab.sh -p /eno/v1717/rl_generated.py \
   --headless --enable_cameras --device cuda:0 --mode v17_residual_curriculum --seed "$SEED" \
   --v17-coarse-ckpt /eno/coarse/bc_actor_best.pt --v17-rl-ckpt /eno/rl/policy.pt \
   --v17-coarse-steps "$STEPS" --v17-force-limit-n "${ENO_V1717_LEGACY_FORCE_N:-18}" \
   --output-dir /eno/output/session
set +e
# Docker单例启动开销比较大；不能将这种顺序执行误称为多环境并行训练。
docker run --rm --gpus all --ipc=host --network host --entrypoint bash \
 --name "$NAME" -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
 -e PYTHONPATH=/eno/scripts:/eno/v26 \
 -e ENO_V25_ROT_MAX_DEG="${ENO_V26_ROT_MAX_DEG:-1.0}" \
 -e ENO_V26_CAPTURE_DIR=/eno/output/session/v26_vision \
 -e ENO_V31_SHARED_MOUNT_MODE="delayed" \
 -e ENO_V315_LOCK_VARIANT="${ENO_V315_LOCK_VARIANT:-axis_only}" \
 -e ENO_V315_Z_HOLD_STEPS="${ENO_V315_Z_HOLD_STEPS:-25}" \
 -e ENO_V316_PRELOCK="${ENO_V316_PRELOCK:-1}" \
 -e ENO_V26_CAPTURE_EVERY="${ENO_V26_CAPTURE_EVERY:-5}" \
 -e ENO_V17_RL_DETERMINISTIC="${ENO_V17_RL_DETERMINISTIC:-0}" \
 -e ENO_V1717_GUARD_PROFILE="$PROFILE" \
 -e ENO_V1717_RL_XY_SCALE_M="${ENO_V1717_RL_XY_SCALE_M:-0.00025}" \
 -e ENO_V1717_RL_Z_SCALE_M="${ENO_V1717_RL_Z_SCALE_M:-0.00008}" \
 -v "$ROOT/scripts:/eno/scripts:ro" -v "$SCRIPT:/eno/v26:ro" -v "$TMP:/eno/v1717:ro" \
 -v "$(dirname "$COARSE"):/eno/coarse:ro" \
 -v "$CKPT:/eno/rl/policy.pt:ro" -v "$OUT:/eno/output:rw" \
 "$IMAGE" -lc "test -s /eno/v1717/rl_generated.py || { echo V26_INFRA_MISSING_SCRIPT;exit 86; }; cd /workspace/isaaclab && $CMD" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]};set -e
RESULT="$OUT/session/v17_7_residual_rollout.json"
if [[ -s "$RESULT" ]]; then
  python3 -S - "$RESULT" <<'PY'
import json,sys
p=sys.argv[1];r=json.load(open(p,encoding='utf-8'))
k=('status','schema','seed','policy_updates','steps_total','rl_steps','stop_reason',
   'max_contact_force_N','success','max_true_depth_mm_offline','min_true_radial_mm_offline',
   'guard_profile','visual_rl_observations','vision_ik_steps','student_ik_source','hole_refine','peg_gate_blocked_steps','soft_contact_response_events','free_advance_events')
print('[V17_17_ROLLOUT_SUMMARY] '+json.dumps({x:r.get(x) for x in k},ensure_ascii=False))
PY
else
  echo "未产生rollout JSON，检查 $OUT/run.log" >&2
  exit 5
fi
((rc==0)) || exit "$rc"
