"""Phase 05B-2A：与 Isaac Lab / USD 无关的几何和随机数工具。

将此文件独立出来的原因：
1. 可以在本地普通 Python 测试圆孔内径、扇区顶点和随机范围；不必启动 GPU 仿真。
2. 这是环境**特权真值**的生成代码，不代表允许传给 policy 神经网络。
3. 所有长度使用国际单位制 m；名字里包含 mm 的量是明确换算后才输出。
"""
from __future__ import annotations
import math
import numpy as np

PEG_DIAMETER_M = 0.020    # 老师指定的圆柱直径：20 mm
HOLE_DIAMETER_M = 0.023   # 老师指定的孔内径：23 mm
PEG_LENGTH_M = 0.040      # 本项目当前工程假设：40 mm
HOLE_DEPTH_M = 0.030      # 本项目当前工程假设：30 mm
SUCCESS_DEPTH_M = 0.015  # 后续验收候选：至少 15 mm（当前尚不验证）
HOLE_BASE_WORLD = (0.50, 0.00, 0.40)  # m：hole 基准孔口坐标
HOLE_RANDOM_HALF_RANGE_M = 0.050      # x/y 各 ±5 cm，覆盖 10cm×10cm
MOUNT_RANDOM_HALF_RANGE_M = 0.005     # link8 局部 x/y 各 ±5 mm
HOLE_OUTER_RADIUS_M = 0.035
WALL_SEGMENTS = 32
CONTACT_OFFSET_M = 0.0004
REST_OFFSET_M = 0.0


def sample_episode(seed: int) -> dict:
    """与 05A 相同的 PCG64 NumPy uniform 抽样顺序，种子可以复现。"""
    rng = np.random.default_rng(int(seed))
    dh = rng.uniform(-HOLE_RANDOM_HALF_RANGE_M, HOLE_RANDOM_HALF_RANGE_M, 2)
    dm = rng.uniform(-MOUNT_RANDOM_HALF_RANGE_M, MOUNT_RANDOM_HALF_RANGE_M, 2)
    return {
        "seed": int(seed),
        "hole_center_world_m": [float(HOLE_BASE_WORLD[0] + dh[0]),
                                 float(HOLE_BASE_WORLD[1] + dh[1]), float(HOLE_BASE_WORLD[2])],
        "hole_xy_offset_m": [float(v) for v in dh],
        "mount_xy_in_link8_m": [float(v) for v in dm],
    }


def wedge_vertices(index: int) -> list[tuple[float, float, float]]:
    """32 段孔壁中的一段：八个凸棱柱顶点，z=0 是孔入口，z=-30mm 是底部。

    每段的内切边对应孔内径 23 mm，而不是错误地把圆内半径作为顶点半径；
    整圈每段各自做凸包，不得对整个空心孔做单个凸包（那会封死孔）。
    """
    if not 0 <= index < WALL_SEGMENTS:
        raise ValueError(f"segment index out of range: {index}")
    a0 = 2 * math.pi * index / WALL_SEGMENTS
    a1 = 2 * math.pi * (index + 1) / WALL_SEGMENTS
    inner_r = (HOLE_DIAMETER_M / 2) / math.cos(math.pi / WALL_SEGMENTS)
    xy = [
        (inner_r * math.cos(a0), inner_r * math.sin(a0)),
        (HOLE_OUTER_RADIUS_M * math.cos(a0), HOLE_OUTER_RADIUS_M * math.sin(a0)),
        (HOLE_OUTER_RADIUS_M * math.cos(a1), HOLE_OUTER_RADIUS_M * math.sin(a1)),
        (inner_r * math.cos(a1), inner_r * math.sin(a1)),
    ]
    return [(x, y, -HOLE_DEPTH_M) for x, y in xy] + [(x, y, 0.) for x, y in xy]


def wall_face_triangles() -> list[tuple[int, int, int]]:
    """与已通过的 01/05A 几何源码保持一致的闭合三角面拓扑。"""
    triangles = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for j in range(4):
        k = (j + 1) % 4
        triangles.extend([(j, k, k + 4), (j, k + 4, j + 4)])
    return triangles


def validate_dimensions() -> None:
    """在任何 USD 创建前调用，避免错误几何进入物理场景。"""
    if not (0 < PEG_DIAMETER_M < HOLE_DIAMETER_M):
        raise ValueError("棒直径必须小于孔直径")
    if not (0 < SUCCESS_DEPTH_M < HOLE_DEPTH_M < PEG_LENGTH_M):
        raise ValueError("棒长、孔深及成功深度不自洽")
    if WALL_SEGMENTS < 16 or HOLE_OUTER_RADIUS_M <= HOLE_DIAMETER_M / 2:
        raise ValueError("孔壁分段或外径设置错误")
    if REST_OFFSET_M >= CONTACT_OFFSET_M:
        raise ValueError("接触偏移必须大于 rest offset")
