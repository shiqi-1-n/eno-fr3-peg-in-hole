#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读还原历史相对路径；不会部署运行环境、下载权重或执行仿真。"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='将 archive/legacy_workspace 还原为独立历史工作区')
    ap.add_argument('--out', type=Path, required=True, help='必须是尚不存在的目录')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'archive' / 'legacy_workspace'
    target = args.out.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit('[ERROR] 未找到 archive/legacy_workspace')
    if target.exists():
        raise SystemExit('[ERROR] 目标路径已存在，拒绝覆盖: '+str(target))
    if target == root or root in target.parents or target in root.parents:
        raise SystemExit('[ERROR] 还原路径不可位于仓库内部或包含仓库')
    # 复制而不是移动；历史原件始终留在 Git 仓库里。
    shutil.copytree(source, target, symlinks=False)
    print('[RESTORED_SOURCE_ONLY]', target)
    print('[NOTE] 缺少模型权重、历史 output 和 Isaac Lab 运行环境；未执行仿真')

if __name__ == '__main__':
    main()
