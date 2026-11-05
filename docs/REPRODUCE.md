# 复现与源码说明

## 目录原则

- `project/`：当前单孔代码选集，按仿真、视觉、控制、学习、评价分类；每个源文件保留原内容，见 `project/SOURCE_MAP.json`。
- `archive/legacy_workspace/`：历史工作区的原始相对路径，含 `scripts/`、冻结评估目录、多个历史试验版本（包括未完成的六孔）。它是源码归档，不是运行完成证明。
- `evidence/`：少量 JSON 报告。
- `media/demos/`：预留演示视频位置。

## 还原原路径

本次结构化整理改变了 GitHub 中代码所在的路径。为避免破坏历史脚本的依赖关系，**不要直接从 `project/` 启动仿真**。

可以使用：

```bash
python3 -S tools/restore_workspace.py --out /path/to/separate-workspace
```

该命令只把 `archive/legacy_workspace/` 中的源文件重新复制成旧目录结构，**不会下载模型权重、完整数据、实验输出，也不会自动启动 Isaac Sim**。不能将仅还原源码视为已完成复现。

旧运行脚本还依赖 Isaac Lab 2.3.1 容器、DGX 硬件、原模型 checkpoint、绝对路径、个别冻结 SHA 校验和历史输出。评估器中的冻结校验不能为了得到好看的结果而绕过。

## 结果的正确引用方式

v31.6.1 的 27/30 是**几何成功记录**；PPO 残差清零、BC 仅用于粗定位；未完成合规 RGB-D-only 与正式物理成功率验收。更完整的解释见 `docs/RESULTS.md`。

原始代码和部分第三方资源在公开前需单独检查许可；不要上传凭据、隐私资料、未经允许的模型权重。
