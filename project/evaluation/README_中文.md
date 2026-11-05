# v31.6.1 FR3 单孔冻结随机评估（30 seeds）

**只用于 RGB-D / IK 的几何成功率测试。** 本包不修改 v31.6.1 模型，不训练 PPO，不修改任务成功判定，也不代表通过棒孔物理接触验收。

## 先确认：不是只看预锁定日志

已知定向回归包括 `v3161_p141_gateA`、`v3161_p150_gateA`、`v3161_p125_gateA`、`v3161_p140_gateA`。四个实验必须都存在 `v17_7_residual_rollout.json`，且 `status=COMPLETE`、`success=true`、`stop_reason=success`、`ZERO_PPO_RESIDUAL`、`ppo_action_applied=false`，并包含提前锁定和轴向切换日志。否则本包拒绝创建 fresh 清单。

**重要**：第25步锁定时XY偏差可能仍很大，这只是冻结刚性安装标定，不是允许下插，真正下插依旧受原有门槛约束。该风险还需要在独立随机集上验证。

## 本包包含

- `original_v315_seed_plan.json`：之前 v31.5 预先选好的全部种子（保持原样，便于审计）。
- `seed_plan.json`：沿用**完全相同的30个**fresh随机种子，加入 v31.6.1 的冻结参数、独立组名。
- `eval_v3161.py`：单实例顺序运行、源码与权重SHA256锁定、不可覆盖/不重试、最终聚合审计。
- `run.sh`：只支持 `fresh`，不再重复已知测试种子。
- `test_host.py`：纯宿主机Python标准库模拟测试，不需NumPy。

如果之前已在任何版本使用这30个种子进行算法调参，就不能再把本组称为独立盲测。此包会自动检查原 v31.5 的 `v31_valF_` 目录，但无法发现所有别的自定义实验名，需要人工确认。

## 部署

Windows PowerShell：

```powershell
scp .\v3161_frozen_eval30.zip dgx@192.168.5.12:/home/dgx/eno-isaac/
```

DGX（**先检查已知125、140的完整结果**）：

```bash
cd /home/dgx/eno-isaac
unzip -o v3161_frozen_eval30.zip
python3 -S v3161_frozen_eval30/test_host.py | tail -1
```

若看到 `[V3161_EVAL_HOST_TEST_PASS]` 表示宿主机模拟自测通过，但**不是 Isaac Lab 成功证据**。

准备并核对清单：

```bash
python3 -S v3161_frozen_eval30/eval_v3161.py prepare fresh
cat output/v3161_frozen_eval/v3161_fresh30_20261104/manifest.json | head -40
```

如未完成四条定向回归，prepare会拒绝，而不会擅自运行新种子。不要删除/伪造回归输出；先用新标签完成未完成的旧样本。

确认30种子是全新且未用于调参后，运行：

```bash
bash v3161_frozen_eval30/run.sh fresh
```

任意时刻只读报告：

```bash
python3 -S v3161_frozen_eval30/eval_v3161.py report fresh
```

原始结果目录示例：

```
/home/dgx/eno-isaac/output/v26_2_zero_ppo_rollouts/v3161_valF_660420371/run.log
/home/dgx/eno-isaac/output/v26_2_zero_ppo_rollouts/v3161_valF_660420371/session/v17_7_residual_rollout.json
```

总报告：

```
/home/dgx/eno-isaac/output/v3161_frozen_eval/v3161_fresh30_20261104/report.json
```

失败种子不会自动重试，也不会被删除或替换。`SENSOR_INVALID` 的 `success=null` 不会当作成功，也不会因为从有效轨迹中剔除就虚增比例；汇总以30为分母，同时报告完整性不足。观测达到至少27/30才有≥90%几何成功比例，但统计意义上的真实成功率、接触物理真实性仍需单独验证。

## 明确限制

- `ppo_action_applied=false`, `policy_updates=0`: 不是训练后策略成功率。
- `max_contact_force_N=0`, `contact_feedback_substep_verified=false`: 尚无可信PhysX接触认证。
- `force_partner_filtered_to_hole=false`: 力传感链仍需验证。
- 启动前SHA锁定相关源码、权重；启动中变更任何冻结依赖会拒绝继续。
- 本次只改评估工具，不改变 v31.6.1 控制文件和历史实验。
