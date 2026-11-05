# 核心代码地图（给第一次查看仓库的人）

这是一个**保留历史版本相对路径**的研究归档，而不是重构完成的单一 Python package。大量历史版本有独立运行入口，不能混用同名模块或把旧版 README 当作最新说明。

| 阅读主题 | 推荐入口 | 说明 |
| --- | --- | --- |
| Isaac Lab 物理环境与历史基础 | `scripts/50_phase05b2b_v16_13_online_evidence.py`、`scripts/` 中 Phase05B 相关文档 | FR3、棒孔场景、接触及控制基础；具体版本需从对应 README 核对 |
| 粗定位 BC 训练 | `scripts/train_v17_3_coarse_bc.py` | 20 维观测到三维动作，属于粗定位训练，不代表精细插入学习 |
| 完整 XYZ BC 训练 | `scripts/train_v17_5_full_bc.py`、`scripts/v17_5_full_eval_extension.txt` | 曾尝试网络负责精细阶段，但历史学生闭环存在 stall |
| RGB-D 全阶段 BC 探索 | `scripts/v19_3/README_中文.md`、`scripts/v19_3/train_v19_bc.py` | 视觉与专家分离的探索；已有专家验证失败，不应宣称全视觉策略已完成 |
| 当前固定相机及视觉估计 | `scripts/v31_6_1_prelock_eval/v26_camera_hook.py`、同目录 `v25_hole.py` 和 `v25_peg_vision.py` | RGB-D 几何估计及审计；要区分视觉在线结果、射线估计和离线真值 |
| 单孔规则/BC 混合控制 | `scripts/v31_6_1_prelock_eval/v26_rl_extension.txt` | v31.6.1 控制主流程。PPO 残差为零，阶段名 `RL` 不表示 PPO 已接管 |
| 单孔冻结评估 | `v3161_frozen_eval30/README_中文.md`、`v3161_frozen_eval30/eval_v3161.py` | 30 种子、冻结配置、摘要；几何成功不等于物理成功 |
| 粗定位 BC 消融 | `bc_ablation_v1/analyze.py`、`evidence/bc_ablation_rollouts/paired_manifest.json` | 配对消融；逐条 rollout 原文件未全部包含在仓库中 |
| 六孔场景 | `v32_six_hole_scene/README_中文.md` | 2×3 阵列与随机化的探索 |
| 六孔指定目标 | `v321_six_hole_target_bc/README_中文.md` | 指定目标与传感器接口探索 |
| 六孔 BC 粗定位 | `v322_six_hole_bc_coarse/README_中文.md` | 当前有安全中止记录，尚未完成六孔闭环 |

## 建议阅读路线

**想看任务与结果**：`README.md` → `docs/EXPERIMENTS.md` → `docs/PROGRESS.md`。

**想看为什么改为分层控制**：完整 BC（v17.5）→ RGB-D 全阶段探索（v19.3）→ v31.6.1 控制与冻结评估 → 粗定位 BC 消融。

**想看六孔**：v32 场景 → v32.1 指定目标 → v32.2 粗定位中止及结果 JSON。不要从六孔失败结论反推单孔 BC 一定无效；那次轨迹在首步被安全机制截停。

## 历史文件说明

- `scripts/` 包含多个版本中同名但不同实现的组件，跨目录引用前必须检查接口版本。
- `docs/SOURCE_MANIFEST.json` 是文件来源与 SHA256 清单，不是推荐运行的脚本列表。
- `docs/HISTORICAL_PACKAGES.json` 只是原 ZIP 清单；原始 ZIP 不随仓库上传。
- `docs/MODEL_CHECKPOINTS.json` 只有权重文件引用和校验值，**不是模型文件本身**。
