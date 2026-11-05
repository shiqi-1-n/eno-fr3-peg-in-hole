# 单孔工程代码导航

本仓库将当前 v31.6.1 相关源码按工程职责整理到 `project/`。**它是原始源码的只读选集，不是已经模块化安装的 Python 包。** 原始相对路径和早期测试版本完整保存在 `archive/legacy_workspace/`；对应源文件见 `project/SOURCE_MAP.json`（含 SHA256）。

| 模块 | 重点文件 | 在当前版本中的角色 |
| --- | --- | --- |
| 仿真/建模 | `project/simulation/50_phase05b2b_v16_13_online_evidence.py` | FR3、棒孔仿真与基础控制代码，历史沿用的主源文件 |
| 仿真/随机化 | `project/simulation/prepare_v17_4_spawn_consistent.py` | 棒/孔场景及随机化改动 |
| 视觉 | `project/perception/v26_camera_hook.py` | 固定 RGB-D 读取和视觉反馈接入 |
| 视觉 | `project/perception/v25_hole.py`, `v25_peg_vision.py` | 孔中心与棒的几何估计 |
| 射线 | `project/perception/v17_3_hole_sensor.py`, `v17_17_hole_refine.py` | 仍在当前版本使用的理想化测距估计；不作为最终合规感知证明 |
| 控制 | `project/control/v26_rl_extension.txt` | 当前阶段切换、BC 粗定位与规则精细控制的主要逻辑 |
| 执行 | `project/control/run_v26_rollout.sh`, `prepare_v26.py` | 当前单孔程序的组装与运行入口 |
| BC | `project/learning/train_v17_3_coarse_bc.py` | 粗定位 BC 训练脚本 |
| 网络加载 | `project/learning/v26_ppo.py` | PPO 网络接口存在；v31.6.1 实际动作残差被清零 |
| 评估 | `project/evaluation/eval_v3161.py`、`seed_plan.json` | 冻结单孔 30 场景的评估配置与口径 |

注意：上述执行代码还会从原 `scripts/` 下加载多个辅助文件。因此，在仓库根目录直接运行 `project/control/run_v26_rollout.sh` **不是受支持的复现方式**。应按照 `docs/REPRODUCE.md` 还原旧工作区并补齐 checkpoint 和环境依赖后运行。

历史完整 BC、PPO 探索、物理接触对照以及六孔试验均在 `archive/legacy_workspace/` 中，不混入当前单孔工程主页。历史演化可查看 `archive/docs/PROGRESS.md`。
