# -*- coding: utf-8 -*-
"""Eno / FR3 v16.10：恢复20秒正式时限的评估候选版；继承v16.9配对初态和v16.8重力前馈。

本版本是【单孔、固定孔中心、随机局部初态】的第一阶段 RL 启动实验。
绝不宣称：已经实现完整10x10cm随机孔位、每回合随机棒安装偏差、多孔任务或90%成功率。

关键限制：
- Actor 仅输入关节 q/dq、加入高斯噪声的棒尖到孔中心相对位移、上一动作。
- 精确碰撞/孔位仅存在于环境内的奖励、重置验收、日志之中；策略前向不接收它们。
- 子控制器知道机器人的正运动学（这是机器人模型，不是孔的真值），负责将三维微动作映射为关节参考。
- 关节参考在上一期望关节角的基础上更新，不使用旧的 q_actual+dq 结构。
- 训练起点采用 v15 经证实的约12.26mm孔口上方关节构型；训练不重放164个慢路点。
- 尚不解决官方FR3 fr3_hand_tcp负质量惯量警告和接触事件零回调；正式统计前须另行解决。
- 所有距离SI单位为米，日志展示毫米，所有角度为弧度，速度为弧度/秒。
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
import os
from pathlib import Path

# Isaac Lab 要求 AppLauncher 在导入 omni / pxr 等 Kit/PhysX 依赖之前运行。
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Eno v16.10：20秒标准时限与几何插入候选诊断（只评估）')
parser.add_argument('--seed', type=int, default=20261027)
parser.add_argument('--case',choices=('center','offset6mm'),default='center',help='历史接口占位；v16强制为center')
parser.add_argument('--servo-mode', choices=['active','frozen'], default='active', help='保留原PD+PI；active在路点执行期继续PI积分，frozen冻结80步预热积分')
parser.add_argument('--plan-only', action='store_true', help='只做80步统一预热和FK规划校验，不执行路点')
parser.add_argument('--step-mm', type=float, default=2.5, help='三阶段路点空间间距上限，默认2.5mm')
parser.add_argument('--max-wait-cycles', type=int, default=65, help='单一路点最多等待的20Hz控制周期')
parser.add_argument('--disable-hole-collision', action='store_true', help='训练禁止关闭孔壁碰撞（此参数只用于拒绝错误配置）')
parser.add_argument('--mode', choices=('eval','pulse'), default='pulse', help='pulse只执行安全正负脉冲，不加载PPO；eval加载旧checkpoint运行真实Isaac Lab')
parser.add_argument('--resume', type=Path, default=None, help='从v16/v16.2模型检查点恢复。训练时同时恢复Adam优化器；评估时只恢复模型。')
parser.add_argument('--eval-episodes', type=int, default=20, help='独立评估回合数；采用新的随机重置序列，默认20回合')
parser.add_argument('--max-episode-steps', type=int, choices=(400,), default=400,
                    help='恢复原定20秒正式评估时限：20Hz×400步；本版禁止缩短或加长时限')
parser.add_argument('--audit-episodes', type=int, default=12, help='v16.3 只诊断局部重置次数，不训练；默认12次')
parser.add_argument('--kp', type=float, choices=(400.0,), default=400.0,
                    help='仅诊断：隐式关节驱动位置刚度，单位N*m/rad；不进入正式训练')
parser.add_argument('--kd', type=float, choices=(40.0,), default=40.0,
                    help='仅诊断：关节驱动阻尼，单位N*m*s/rad；与KP配对比较')
parser.add_argument('--bias-scale', type=float, default=0.0, help='零动作对照中关节位置驱动预补偿比例；0=基线，0.5=半量，1=全量。仅诊断，不可用于训练')
parser.add_argument('--gravity-ff-scale', type=float, choices=(0.0,1.0), default=1.0, help='正常控制必须为1；零重力或关闭前馈不能被用于训练；单位Nm')
parser.add_argument('--audit-control-steps', type=int, default=8, help='每次重置后执行零动作的控制周期数，默认8次')
parser.add_argument('--steps', type=int, default=2048, help='本次运行新增环境步数，不含断点里已有的步数；smoke/eval无效')
parser.add_argument('--rollout', type=int, default=128, help='PPO每轮采样环境步数')
parser.add_argument('--epochs', type=int, default=4, help='每批样本的PPO优化轮数')
parser.add_argument('--obs-noise-mm', type=float, default=0.15, help='相对位置估计噪声标准差，毫米')
parser.add_argument('--reset-xy-mm', type=float, default=2.0, help='局部训练初始横向扰动幅度上限，毫米')
parser.add_argument('--output-dir', type=Path, default=Path('/eno/output/phase05b2b_v16_13'))
parser.add_argument('--eno-camera', action='store_true', help='启用真正的Isaac Sim渲染相机；须同时使用AppLauncher的--enable_cameras')
parser.add_argument('--eno-capture-every', type=int, default=5, help='每N个20Hz控制周期抓取一帧RGB；默认5即4FPS实时时间')
parser.add_argument('--eno-record-episodes', type=int, default=1, help='最大记录回合数；限制证据体积和渲染负载')
parser.add_argument('--eno-pulse-steps', type=int, default=12, help='四组受控动作中每组施加多少控制步，默认12=0.6s')
parser.add_argument('--eno-pulse-magnitude', type=float, default=0.35, help='每个方向归一化动作幅度，0.35对应XY每周期0.1225mm目标增量')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.disable_hole_collision:
    parser.error('v16真实训练不允许关闭孔壁碰撞')
if args.case != 'center':
    parser.error('v16局部训练不允许offset6mm负对照')
if args.mode in ('pulse','eval') and args.gravity_ff_scale != 1.0:
    parser.error('v16.8稳定试训与评估必须打开重力前馈；严禁在旧下沉环境里训练')
if args.max_episode_steps != 400:
    parser.error('v16.10正式时限必须400步=20秒')
if args.mode=='eval' and args.resume is None:
    parser.error('eval必须指定--resume检查点；不能评估随机网络')
if not math.isfinite(args.bias_scale) or args.bias_scale != 0.0:
    parser.error('v16.7为避免混杂只允许bias-scale=0；不得继续叠加v16.4偏置')
if (args.kp,args.kd) != (400.,40.):
    parser.error('v16.7只允许原始400/40增益，避免增益和重力前馈同时改变')
if not math.isfinite(args.bias_scale) or not 0.0 <= args.bias_scale <= 1.0:
    parser.error('bias-scale 必须位于[0,1]，仅用于诊断')
if args.audit_episodes < 1 or args.audit_control_steps < 1 or args.audit_control_steps > 30:
    parser.error('audit-episodes应为正数，audit-control-steps必须在1..30内')
if args.eno_capture_every<1 or args.eno_record_episodes<1 or args.eno_pulse_steps<1 or args.eno_pulse_steps>20 or not (0.05<=args.eno_pulse_magnitude<=0.5):
    parser.error('v16.13录像间隔、回合数或脉冲幅度/长度非法')
if args.mode=='pulse' and args.resume is not None:
    parser.error('pulse不需要PPO检查点，拒绝传入--resume，以免混淆策略和控制诊断')
if args.steps < 1 or args.rollout < 1 or args.epochs < 1 or args.eval_episodes < 1:
    parser.error('steps/rollout/epochs/eval-episodes必须为正整数')
if not 0.0 < args.obs_noise_mm <= 2.0 or not 0 <= args.reset_xy_mm <= 4.0:
    parser.error('观测噪声必须为(0,2]mm，初始XY扰动须在[0,4]mm')
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 以下导入均需要 Kit 已启动。
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics  # noqa: E402
import phase05b2a_geometry as geom  # noqa: E402
import phase05b2b_contact_metrics as cm  # noqa: E402
from eno_v1613_live_evidence import LiveEvidence  # noqa: E402

VERSION = '05B2B_V16_13_LIVE_CAUSAL_EVIDENCE_NOT_TRAINING'
FR3_RELATIVE_USD = '/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd'
ASSET_ROOT = get_assets_root_path()
if not ASSET_ROOT:
    raise RuntimeError('找不到官方 Isaac Sim Assets Root；不自动换成 Panda')
FR3_USD = ASSET_ROOT + FR3_RELATIVE_USD
ARM_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]
ENV_ROOT = '/World/envs/env_0'
ROBOT_ROOT = ENV_ROOT + '/Robot'
LINK8_PATH = ROBOT_ROOT + '/fr3_link8'
MOUNT_PATH = LINK8_PATH + '/EnoToolMount'
PEG_PATH = MOUNT_PATH + '/PegCollider'
HOLE_PATH = ENV_ROOT + '/Hole'
WALL_PATH = HOLE_PATH + '/Wall'
SIM_DT = 1.0 / 240.0
DECIMATION = 12
CTRL_DT = SIM_DT * DECIMATION
KP, KD, KI = 400.0, 40.0, 1.0
MAX_BIAS_RAD = 0.12
MAX_BIAS_CHANGE_PER_STEP_RAD = 0.004

# 05A -> 03B 源码核对后的关键数学常数（所有距离单位m、角度rad）：
LEGACY_MAX_XYZ_STEP_M = 0.008
LEGACY_MAX_JOINT_STEP_RAD = 0.035
LEGACY_MAX_AXIS_ROT_RAD = 0.060
LEGACY_IK_DAMPING = 0.075
LEGACY_ANGULAR_WEIGHT_M = 0.15
LEGACY_JOINT_LIMIT_MARGIN_RAD = 0.035
# ---------------------- v7 实验设计与单位 ----------------------
# 本阶段仍是真值专家（仅读取用于控制的真值），不是神经网络/PPO。
# v6所谓“横向对准”实际上同步减小XYZ三个误差，因而提前下降。
# v7将目标拆成三个确定的路点：
#   (A) height_prepare : 保持当前XY，把棒尖调到 world z=0.620m，离孔口220mm。
#   (B) xy_align       : 固定world z=0.620m，将XY移动到孔中心。
#   (C) preinsert      : 只有XY已经达标，才垂直下降至 world z=0.480m。
# 0.620m是目前随机seed下可达性仍须PhysX验证的假设；若不可达，报告会失败。
HEIGHT_PREP_ABOVE_HOLE_M = 0.220
PREINSERT_HEIGHT_ABOVE_HOLE_M = 0.080
STAGE_Z_OVERSHOOT_GUARD_M = 0.025
# 第一阶段和横向阶段保留厘米级安全裕量；绝不在本程序执行实际插孔。
STAGE_TOL_XY_M = 0.010
STAGE_TOL_Z_M = 0.010
STAGE_TOL_TILT_DEG = 3.0
FINAL_TOL_XY_M = 0.005
FINAL_TOL_Z_M = 0.005
FINAL_TOL_TILT_DEG = 1.5
STAGE_MAX_STEP_M = 0.0025
STAGE_MAX_JOINT_RAD = 0.008
FINAL_MAX_STEP_M = 0.0015
FINAL_MAX_JOINT_RAD = 0.005
# 位置任务优先的阻尼，单位与雅可比相匹配。姿态误差每周期限幅：
POSITION_DLS_DAMPING = 0.035
ROTATION_DLS_DAMPING = 0.080
MAX_AXIS_ROT_RAD = 0.025
# 副任务修正如果污染主任务的平移，逐次减半直至满足误差阈值。
MAX_POSITION_LEAK_M = 0.00030
# 每周期实际棒尖不能跳过 15 mm，否则终止，以便检查物理器件。
MAX_MEASURED_TIP_DISPLACEMENT_PER_CYCLE_M = 0.015
HOLD_STEPS = 80
RAMP_STEPS = 20
J1_TARGET_SHIFT_RAD = 0.0135

# 已验证的 04 阶段，在简化法兰工具中关闭原厂夹爪的碰撞和视觉：
# link8 和 PegCollider 必须保持碰撞。子树本身仍含原厂质量/惯量，后续单独处理。
UNUSED_GRIPPER_PARTS = ('fr3_hand', 'fr3_leftfinger', 'fr3_rightfinger', 'fr3_hand_tcp')


def as_finite_list(value):
    """序列化之前确保没有 NumPy 标量或 CUDA Tensor，避免之前 np.bool_ 的错误。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f'JSON指标存在 NaN/Inf：{arr}')
    return arr.tolist()


def apply_collision(prim):
    """PhysX 碰撞体开关。接触 offset 仅为检测距离，不是允许穿透的距离。"""
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(geom.CONTACT_OFFSET_M)
    api.CreateRestOffsetAttr(geom.REST_OFFSET_M)


def make_box(stage, path, center, size_xyz, color):
    """静态长方体；size_xyz 是 x/y/z 的**总长度**，不是半长。"""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size_xyz]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    apply_collision(cube.GetPrim())


def create_hole_in_env(stage, episode):
    """构建 32 段**凸**孔壁和静态孔底；整个 Hole 父节点负责定位。"""
    if stage.GetPrimAtPath(HOLE_PATH).IsValid():
        raise RuntimeError('已经存在 Hole，不得重复覆盖旧孔座')
    hole = UsdGeom.Xform.Define(stage, HOLE_PATH)
    hole.AddTranslateOp().Set(Gf.Vec3d(*episode['hole_center_world_m']))
    UsdGeom.Xform.Define(stage, WALL_PATH)
    triangles = geom.wall_face_triangles()
    for i in range(geom.WALL_SEGMENTS):
        # 用**独立 MeshCollisionAPI(convexHull)**，禁止把整圈空心结构合成一个凸包。
        v = geom.wedge_vertices(i)
        mesh = UsdGeom.Mesh.Define(stage, f'{WALL_PATH}/segment_{i:02d}')
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr([k for face in triangles for k in face])
        xyz = list(zip(*v))
        mesh.CreateExtentAttr([Gf.Vec3f(*(min(t) for t in xyz)),
                               Gf.Vec3f(*(max(t) for t in xyz))])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.55, 0.85)])
        apply_collision(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    plate = 0.020  # m, 孔底板厚度：20mm
    make_box(stage, HOLE_PATH + '/BottomPlate',
             (0, 0, -geom.HOLE_DEPTH_M - plate / 2), (0.14, 0.14, plate), (0.46, 0.46, 0.46))
    # 支撑台顶部接着底板下表面；没有封住上方孔口。
    pedestal_h = geom.HOLE_BASE_WORLD[2] - geom.HOLE_DEPTH_M - plate
    if pedestal_h > 0:
        make_box(stage, HOLE_PATH + '/Pedestal',
                 (0, 0, -geom.HOLE_DEPTH_M - plate - pedestal_h / 2),
                 (0.14, 0.14, pedestal_h), (0.3, 0.32, 0.34))


def make_peg_on_link8(stage, episode):
    """在机器人的 link8 下挂圆柱子碰撞体，而非创建可掉落的独立棒刚体。

    USD 的 link8 预仿真姿态用于定义一个**工具局部安装坐标约定**：局部 -Z
    映射为当前 USD 世界 -Z。Lab reset 的关节姿态可能与该 USD 预设不同，
    因此 reset 后必须报告运行时棒轴的倾角，不能默认它已经竖直。
    """
    link = stage.GetPrimAtPath(LINK8_PATH)
    if not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f'FR3 link8 尚未完整生成：{LINK8_PATH}')
    if link.IsInstance() or link.IsInstanceProxy():
        raise RuntimeError('FR3 link8 是USD实例，不允许直接在其下追加Collider')
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link_mat = xcache.GetLocalToWorldTransform(link)
    inv = link_mat.GetInverse()
    down_local = inv.TransformDir(Gf.Vec3d(0, 0, -1))
    n = math.sqrt(sum(float(v)**2 for v in down_local))
    if not math.isfinite(n) or n < 1e-8:
        raise RuntimeError('无法确定 link8 的 USD 初始安装方向')
    down_local = Gf.Vec3d(*(float(v)/n for v in down_local))
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), down_local).GetQuat()
    mount = UsdGeom.Xform.Define(stage, MOUNT_PATH)
    mount.AddTranslateOp().Set(Gf.Vec3d(*episode['mount_xy_in_link8_m'], 0.0))
    mount.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary())))
    rod = UsdGeom.Cylinder.Define(stage, PEG_PATH)
    rod.CreateRadiusAttr(geom.PEG_DIAMETER_M / 2)
    rod.CreateHeightAttr(geom.PEG_LENGTH_M)
    rod.CreateAxisAttr('Z')
    rod.AddTranslateOp().Set(Gf.Vec3d(0., 0., -geom.PEG_LENGTH_M / 2))
    rod.CreateDisplayColorAttr([Gf.Vec3f(0.98, 0.58, 0.06)])
    apply_collision(rod.GetPrim())
    if rod.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('棒被错误设成了独立刚体，无法随 link8 正确运动')
    # 几何是在 PhysX 开始前定义的：从 USD 坐标链计算固定的 link8 局部棒尖/头端。
    xcache.Clear()
    rod_to_world = xcache.GetLocalToWorldTransform(rod.GetPrim())
    tip_world = rod_to_world.Transform(Gf.Vec3d(0, 0, -geom.PEG_LENGTH_M / 2))
    head_world = rod_to_world.Transform(Gf.Vec3d(0, 0, geom.PEG_LENGTH_M / 2))
    tip_local_point = inv.Transform(tip_world)
    head_local_point = inv.Transform(head_world)
    tip_local = np.array([float(tip_local_point[i]) for i in range(3)], dtype=float)
    head_local = np.array([float(head_local_point[i]) for i in range(3)], dtype=float)
    v = tip_local - head_local
    length = float(np.linalg.norm(v))
    if abs(length - geom.PEG_LENGTH_M) > 1e-5:
        raise RuntimeError(f'棒在link8局部坐标的长度错误：{length}m')
    return tip_local, v / length


def deactivate_factory_gripper_colliders(stage):
    """复用 04 已验证的简化任务约定：关闭原厂夹爪接触，不影响棒和孔。

    注意：并没有改造官方 FR3 的真实 flange、惯量和传感器模型。
    """
    off = []
    for body_name in UNUSED_GRIPPER_PARTS:
        root = stage.GetPrimAtPath(f'{ROBOT_ROOT}/{body_name}')
        if not root.IsValid():
            raise RuntimeError(f'缺失原厂夹爪刚体：{body_name}')
        UsdGeom.Imageable(root).MakeInvisible()
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                if not attr:
                    attr = UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
                attr.Set(False)
                off.append(str(prim.GetPath()))
    for name in ('fr3_leftfinger', 'fr3_rightfinger'):
        if not any('/'+name+'/' in p for p in off):
            raise RuntimeError(f'没有定位到 {name} 的碰撞，禁止盲目继续')
    return off


def inspect_collision_stage(stage, episode, disabled):
    """检查 USD 定义：这是**碰撞体存在性**检查，不是接触物理证明。"""
    wall_paths = []
    for i in range(geom.WALL_SEGMENTS):
        path = f'{WALL_PATH}/segment_{i:02d}'
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f'缺失孔壁碰撞：{path}')
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            raise RuntimeError(f'孔壁不是凸网格碰撞：{path}')
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        if str(mesh_collision.GetApproximationAttr().Get()) != 'convexHull':
            raise RuntimeError(f'孔壁没有使用 convexHull：{path}')
        wall_paths.append(path)
    peg = stage.GetPrimAtPath(PEG_PATH)
    if not peg.IsValid() or not peg.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError('缺失棒碰撞')
    if peg.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('棒错误地成为了单独刚体')
    checked = wall_paths + [PEG_PATH, HOLE_PATH + '/BottomPlate', HOLE_PATH + '/Pedestal']
    for path in checked:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f'缺失静态几何碰撞：{path}')
        enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        if enabled_attr and enabled_attr.Get() is False:
            raise RuntimeError(f'关键棒/孔碰撞已关闭：{path}')
    if PEG_PATH in disabled or any(p in disabled for p in wall_paths):
        raise RuntimeError('错误地关闭棒或孔壁的碰撞')
    hole_origin = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        stage.GetPrimAtPath(HOLE_PATH)).Transform(Gf.Vec3d(0, 0, 0))
    measured = np.asarray(hole_origin, dtype=float)
    target = np.asarray(episode['hole_center_world_m'], dtype=float)
    if not np.allclose(measured, target, atol=1e-7):
        raise RuntimeError(f'随机孔位和USD父节点不一致：{measured} vs {target}')
    return {
        'wall_segment_count': len(wall_paths), 'all_wall_colliders_enabled': True,
        'peg_collider_enabled': True, 'peg_rigidbody_absent': True,
        'hole_origin_world_m': measured.tolist(),
        'gripper_colliders_disabled_count': len(disabled),
        'hole_entrance_z_m': geom.HOLE_BASE_WORLD[2],
        'peg_diameter_mm': geom.PEG_DIAMETER_M * 1000,
        'hole_diameter_mm': geom.HOLE_DIAMETER_M * 1000,
        'hole_depth_mm': geom.HOLE_DEPTH_M * 1000,
        'peg_length_mm': geom.PEG_LENGTH_M * 1000,
    }


def quat_wxyz_to_rot(q):
    """Isaac Lab 2.3.1 body_link_quat_w 顺序为 w/x/y/z；别用 3.0 的文档。"""
    w, x, y, z = [float(v) for v in q]
    s = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(s) or s < 1e-10:
        raise RuntimeError('无效 link8 四元数')
    w, x, y, z = w/s, x/s, y/s, z/s
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ], dtype=float)


def smoothstep(u):
    """关节指令平滑插值，让实验主要验证连接关系而非冲击。"""
    u = min(1., max(0., float(u)))
    return u*u*(3-2*u)


@configclass
class FR3PegHoleSceneCfg(DirectRLEnvCfg):
    """这里只是 Isaac Lab 可运行的场景壳，不是 PPO 可训练的正式环境。"""
    decimation = DECIMATION
    episode_length_s = 300.0  # 44个路点每点最多65周期=143s；避免过程被环境超时自动reset。
    action_space = 7           # 测试专用 7 维占位；正式策略应采用3维法兰增量动作。
    observation_space = 14     # q(7) + dq(7)；**不包含真实孔或棒位置**。
    state_space = 0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=SIM_DT, render_interval=DECIMATION)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3., replicate_physics=True)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=FR3_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., 0.),
            joint_pos={
                'fr3_joint1':0., 'fr3_joint2':-0.3, 'fr3_joint3':0.,
                'fr3_joint4':-1.5, 'fr3_joint5':0., 'fr3_joint6':1.2,
                'fr3_joint7':0.
            },
        ),
        actuators={'all_joints':ImplicitActuatorCfg(
            joint_names_expr=['.*'], stiffness=KP, damping=KD,
        )},
    )


class FR3PegHoleSceneEnv(DirectRLEnv):
    cfg: FR3PegHoleSceneCfg

    def __init__(self, cfg:FR3PegHoleSceneCfg, render_mode=None, **kwargs):
        self.episode_truth = geom.sample_episode(args.seed)  # 构建场景用的特权真值
        self.usd_checks = {}
        self.tip_link_local_m = None
        self.axis_link_local = None
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids, self.arm_names = self.robot.find_joints(ARM_NAMES)
        self.link8_ids, self.link8_names = self.robot.find_bodies('fr3_link8')
        if len(self.arm_ids)!=7 or len(self.link8_ids)!=1 or self.robot.num_joints!=9:
            raise RuntimeError(f'机器人结构异常：{self.robot.joint_names} / {self.robot.data.body_names}')
        self._anchor = self.robot.data.default_joint_pos.clone()
        self._ref = self._anchor.clone()
        self._pd_target = self._anchor.clone()
        self._bias = torch.zeros_like(self._anchor)
        self._step_number = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError('Omni USD stage 尚未建立')
        if int(self.scene.cfg.num_envs)!=1:
            raise RuntimeError('本版未实现多环境随机初始化；严禁把num_envs>1作为已验证')
        # 物理启动前附加碰撞；如果先启动 PhysX 再随意编辑 Collider，物理场景可能未同步。
        create_hole_in_env(stage, self.episode_truth)
        self.tip_link_local_m, self.axis_link_local = make_peg_on_link8(stage, self.episode_truth)
        disabled = deactivate_factory_gripper_colliders(stage)
        self.usd_checks = inspect_collision_stage(stage, self.episode_truth, disabled)
        print('[PHASE05B2A_USD_CHECK] '+json.dumps(self.usd_checks, ensure_ascii=False), flush=True)
        # 环境克隆必须发生在源环境已有完整几何之后。
        self.scene.clone_environments(copy_from_source=False)
        if self.device == 'cpu':
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations['robot']=self.robot
        try:
            self.usd_export_stage=stage
        except Exception:
            pass

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids=self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        root=self.robot.data.default_root_state[env_ids].clone()
        root[:,:3]+=self.scene.env_origins[env_ids]
        q=self.robot.data.default_joint_pos[env_ids].clone()
        dq=self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(q, dq, None, env_ids)
        if hasattr(self, '_anchor'):
            self._anchor[env_ids]=q
            self._ref[env_ids]=q
            self._pd_target[env_ids]=q
            self._bias[env_ids]=0.
            self._step_number=0
        # 本版仅在新仿真进程启动前随机化一次；这里不修改运行中的静态孔USD。

    def _pre_physics_step(self, actions):
        if tuple(actions.shape)!=(self.num_envs,7) or not bool(torch.isfinite(actions).all().item()):
            raise RuntimeError('收到错误维度或非有限的测试动作')
        if bool(torch.any(actions!=0).item()):
            raise RuntimeError('本版只接收零占位动作，实际关节参考由冒烟测试的时间表给出')
        self._step_number+=1
        progress=smoothstep((self._step_number-HOLD_STEPS)/RAMP_STEPS)
        self._ref=self._anchor.clone()
        self._ref[:,self.arm_ids[0]]+=progress*J1_TARGET_SHIFT_RAD
        # 外环只用机器人关节角；不会使用孔位/棒安装真值。
        joint_error=self._ref[:,self.arm_ids]-self.robot.data.joint_pos[:,self.arm_ids]
        step=torch.clamp(KI*CTRL_DT*joint_error,
                         -MAX_BIAS_CHANGE_PER_STEP_RAD,MAX_BIAS_CHANGE_PER_STEP_RAD)
        self._bias[:,self.arm_ids]=torch.clamp(self._bias[:,self.arm_ids]+step,
                                              -MAX_BIAS_RAD,MAX_BIAS_RAD)
        self._pd_target=self._ref.clone()
        self._pd_target[:,self.arm_ids]+=self._bias[:,self.arm_ids]

    def _apply_action(self):
        # 仅设置关节驱动目标；运动期间不 write_joint_state_to_sim() 瞬移机械臂。
        self.robot.set_joint_position_target(self._pd_target)

    def _get_observations(self):
        # 明确观测白名单：真实孔位/安装偏差只能在评估用的函数中访问。
        q=self.robot.data.joint_pos[:,self.arm_ids]
        dq=self.robot.data.joint_vel[:,self.arm_ids]
        return {'policy':torch.cat([q,dq],dim=-1)}

    def _get_rewards(self):
        # 仅 gym 接口占位：不是插孔奖励，也不能训练真正插孔策略。
        return -torch.sum((self.robot.data.joint_pos[:,self.arm_ids]-self._ref[:,self.arm_ids]).square(),dim=-1)

    def _get_dones(self):
        bad=~torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
        timeout=self.episode_length_buf>=self.max_episode_length-1
        return bad,timeout

    def measure_peg_tip_oracle(self):
        """仅供 JSON 验收用真值，不放进 policy 观测；实际来自 PhysX link8 pose。

        这项计算还不能取代真实杆—孔接触报告。
        """
        j=self.link8_ids[0]
        pos=np.asarray(self.robot.data.body_link_pos_w[0,j].detach().cpu(),dtype=float)
        quat=np.asarray(self.robot.data.body_link_quat_w[0,j].detach().cpu(),dtype=float)
        rot=quat_wxyz_to_rot(quat)
        tip=pos+rot@self.tip_link_local_m
        axis=rot@self.axis_link_local
        tilt=math.degrees(math.acos(float(np.clip(np.dot(axis,[0,0,-1]),-1,1))))
        xy=np.asarray(self.episode_truth['hole_center_world_m'][:2])
        radial_mm=float(np.linalg.norm(tip[:2]-xy)*1000)
        depth_mm=float((geom.HOLE_BASE_WORLD[2]-tip[2])*1000)
        return {'link8_world_xyz_m':as_finite_list(pos),
                'link8_world_quat_wxyz':as_finite_list(quat),
                'tip_world_xyz_m':as_finite_list(tip),
                'peg_axis_world_xyz':as_finite_list(axis),
                'tilt_from_world_down_deg':tilt,
                'tip_radial_offset_from_hole_mm':radial_mm,
                'tip_depth_mm_signed':depth_mm}


# =============================================================================
# Phase 05B-2B：基于 2A 场景的物理接触监测 + 真值 IK 正反对照
# =============================================================================
# 前半段的大部分 FR3/棒/孔场景构造代码来自已在 DGX Spark 通过的 05B-2A，
# 以保持几何、资产和关节驱动一致。这部分扩展不修改 05B-2A 原文件。
# 本文件使用准确孔位置和 link8 的真实姿态执行 IK，仅用于 PhysX 物理验收。
# 以后 PPO 策略输入仍必须严格屏蔽这些真值。

class PhysxPegHoleEvents:
    """记录实际物理引擎发出的“棒碰到孔”事件，而非只检查 USD Collider 属性。

    特别提醒：无事件不能证明无碰撞。居中可通行可能本来就不接触孔壁；
    如果 API 未注册成功，应将负对照标记为证据不足，不可以认定通过。
    """
    def __init__(self, stage):
        self.events = 0
        # 所有原始接触事件计数，用于排查严格 PEG_PATH 比对是否过滤了有效事件。
        # 此字段只诊断回调，不参与插孔成功分类。
        self.raw_events = 0
        self.raw_pairs = []
        self.contact_data_count = 0
        self.min_sep_m = None
        self.max_impulse_ns = 0.
        self.pairs = []
        self.errors = []
        self.enabled = False
        self.warning = None
        self._subscription = None
        try:
            link = stage.GetPrimAtPath(LINK8_PATH)
            if not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError('link8 不是有效刚体，无法开启接触报告')
            # 只在 link8 刚体上启用 PhysX 接触通知；不在 PegCollider 上创建第二个刚体。
            api = PhysxSchema.PhysxContactReportAPI.Apply(link)
            api.CreateThresholdAttr(0.0)
            from omni.physx import get_physx_simulation_interface
            self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
            self.enabled = True
        except Exception as ex:
            self.warning = f'{type(ex).__name__}: {ex}'
            print('[05B2B_CONTACT_SETUP_WARNING] '+self.warning, flush=True)

    def _on_contact(self, headers, data):
        try:
            for h in headers:
                a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                self.raw_events += 1
                if [a,b] not in self.raw_pairs and len(self.raw_pairs) < 20:
                    self.raw_pairs.append([a,b])
                is_peg_hole = ((a == PEG_PATH and b.startswith(HOLE_PATH+'/')) or
                               (b == PEG_PATH and a.startswith(HOLE_PATH+'/')))
                if not is_peg_hole:
                    continue
                self.events += 1
                n = int(h.num_contact_data)
                self.contact_data_count += n
                if [a,b] not in self.pairs and len(self.pairs) < 10:
                    self.pairs.append([a,b])
                start = int(h.contact_data_offset)
                for k in range(start,start+n):
                    d = data[k]
                    s = getattr(d,'separation',None)
                    if s is not None and math.isfinite(float(s)):
                        val = float(s)
                        self.min_sep_m = val if self.min_sep_m is None else min(self.min_sep_m,val)
                    imp = getattr(d,'impulse',None)
                    if imp is not None:
                        self.max_impulse_ns = max(self.max_impulse_ns,
                                                  math.sqrt(sum(float(z)**2 for z in imp)))
        except Exception as ex:
            if len(self.errors) < 5:
                self.errors.append(f'{type(ex).__name__}: {ex}')

    def snapshot(self):
        return dict(enabled=self.enabled, initialization_warning=self.warning,
                    callback_errors=self.errors,
                    peg_hole_event_count=self.events,
                    raw_contact_event_count=self.raw_events, raw_contact_pairs=self.raw_pairs,
                    contact_data_count=self.contact_data_count,
                    min_separation_mm=None if self.min_sep_m is None else self.min_sep_m*1000.,
                    max_impulse_Ns=self.max_impulse_ns, pairs=self.pairs,
                    interpretation='未收到事件不代表无碰撞；需用偏心负对照证明物理阻挡')

    def close(self):
        # 订阅生命周期必须比 Isaac Sim 物理场景短，避免关掉app后仍接收回调。
        self._subscription = None


class PhysxContactContrastEnv(FR3PegHoleSceneEnv):
    """复用 2A FR3/真实孔壁，但将测试时序改为外部真值 IK 的关节参考。

    - Gym action 仍是 7 维的零占位（**不是神经网络动作**）。
    - IK 设置 q_ref，外环 PI 只依据关节状态修正目标，机器人由 PD+PhysX 运动。
    - 每个 episode 一个 PhysX 进程；本阶段没有实现随机多回合 reset。
    """
    def _setup_scene(self):
        super()._setup_scene()
        stage = omni.usd.get_context().get_stage()
        # v15b 唯一物理消融：偏心实验的 32 段孔壁碰撞体开/关。
        # 注意：在仿真 play 之前更改 USD Physics CollisionAPI；不改 robot、peg、
        # 关节驱动、IK、随机种子、动作或轨迹。本字段与旧 scene_checks 分开记录，
        # 因为旧 scene_checks 是 super()._setup_scene() 时的先前快照。
        # 严格消融【孔壁】32段，不关闭孔底板/支撑台，减少物理差异。
        hole_paths = [f'{WALL_PATH}/segment_{i:02d}' for i in range(geom.WALL_SEGMENTS)]
        for stationary_path in (HOLE_PATH+'/BottomPlate', HOLE_PATH+'/Pedestal'):
            p=stage.GetPrimAtPath(stationary_path)
            if (not p.IsValid() or not p.GetAttribute('physics:collisionEnabled').IsValid()
                    or not bool(p.GetAttribute('physics:collisionEnabled').Get())):
                raise RuntimeError(f'v15b 底板/支撑台必须仍启用碰撞: {stationary_path}')
        toggled=[]
        for path in hole_paths:
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f'v15b 静态孔碰撞体缺失: {path}')
            attr=prim.GetAttribute('physics:collisionEnabled')
            if not attr.IsValid() or not bool(attr.Get()):
                raise RuntimeError(f'v15b 预消融孔碰撞体状态异常: {path}')
            if args.disable_hole_collision:
                attr.Set(False)
                toggled.append(path)
            now=bool(attr.Get())
            if now == bool(args.disable_hole_collision):
                raise RuntimeError(f'v15b 孔碰撞开关核验失败: {path} = {now}')
        # 不许误关 PegCollider：不然无法将组间差异定位到孔座本身。
        peg_prim=stage.GetPrimAtPath(PEG_PATH)
        peg_enabled=bool(peg_prim.IsValid() and
                         peg_prim.GetAttribute('physics:collisionEnabled').IsValid() and
                         peg_prim.GetAttribute('physics:collisionEnabled').Get())
        if not peg_enabled:
            raise RuntimeError('v15b 棒碰撞体被错误禁用，消融实验必须终止')
        self.collision_ablation = {
            'collision_enabled': not args.disable_hole_collision,
            'hole_collision_prim_count':len(hole_paths),
            'bottom_plate_collision_enabled':True,
            'pedestal_collision_enabled':True,
            'disabled_paths':toggled,
            'confirmed_after_usd_write':True,
            'peg_collision_unchanged':peg_enabled,
            'caution':'关闭孔座碰撞仅用于反事实对照，不可用于真实插孔成功率计算',
        }
        print('[V15B_COLLISION_STATE] '+json.dumps(self.collision_ablation,ensure_ascii=False),flush=True)
        # 在 DirectRLEnv 的模拟器真正 play 之前应用 contact report。
        self.contact_monitor = PhysxPegHoleEvents(stage)

    def _pre_physics_step(self, actions):
        if tuple(actions.shape) != (1,7) or not bool(torch.isfinite(actions).all().item()):
            raise RuntimeError('联系 IK 控制的测试动作必须是形状(1,7)的有限值')
        if bool(torch.any(actions != 0).item()):
            raise RuntimeError('本诊断只接收零占位；动作由外部真值IK专家给出，非策略网络')
        self._step_number += 1
        error = self._ref[:,self.arm_ids] - self.robot.data.joint_pos[:,self.arm_ids]
        incr = torch.clamp(KI*CTRL_DT*error,
                           -MAX_BIAS_CHANGE_PER_STEP_RAD,MAX_BIAS_CHANGE_PER_STEP_RAD)
        # 【v8 唯一伺服消融开关】两组在80个相同的PI预热步后具有相同的补偿状态。
        # frozen: 保留这一恒定补偿量，以免直接关PI导致重力漂移混淆比较。
        # active: 在每周期继续按照原v7的KI=1.0积分规律更新。
        # 两种模式下 Isaac Lab 的ImplicitActuator KP/KD、物理、关节参考全部不变。
        if not getattr(self, '_integral_frozen', False):
            self._bias[:,self.arm_ids] = torch.clamp(self._bias[:,self.arm_ids]+incr,
                                                     -MAX_BIAS_RAD,MAX_BIAS_RAD)
        self._pd_target = self._ref.clone()
        self._pd_target[:,self.arm_ids] += self._bias[:,self.arm_ids]



def _pose_matrix(pos, quat):
    """局部平移+四元数转换 4x4 齐次变换，位置单位 m，四元数采用 wxyz。"""
    p=np.asarray(pos,dtype=float)
    q=np.asarray(quat,dtype=float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise RuntimeError('USD关节local pose含NaN/Inf或者维数不对')
    m=np.eye(4,dtype=float)
    m[:3,:3]=quat_wxyz_to_rot(q)
    m[:3,3]=p
    return m


def _usd_local_pose(joint, i):
    """读取关节在body0/body1上的局部锚点位置和旋转，不依赖显示网格姿态。"""
    pos_attr=joint.GetLocalPos0Attr() if i==0 else joint.GetLocalPos1Attr()
    rot_attr=joint.GetLocalRot0Attr() if i==0 else joint.GetLocalRot1Attr()
    p=pos_attr.Get() if pos_attr else None
    q=rot_attr.Get() if rot_attr else None
    if p is None: p=(0.,0.,0.)
    if q is None: quat=[1.,0.,0.,0.]
    else:
        v=q.GetImaginary()
        quat=[float(q.GetReal()),float(v[0]),float(v[1]),float(v[2])]
    return _pose_matrix([float(v) for v in p],quat)


def _axis_rotation(axis, theta):
    """USD RevoluteJoint 轴的右手系关节旋转，theta单位rad。"""
    a=str(axis).upper()
    if a not in ('X','Y','Z'): raise RuntimeError(f'无法解析USD旋转轴: {axis}')
    v=np.array({'X':[1.,0.,0.],'Y':[0.,1.,0.],'Z':[0.,0.,1.]}[a])
    c,s=math.cos(theta),math.sin(theta)
    r=c*np.eye(3)+(1-c)*np.outer(v,v)+s*np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
    T=np.eye(4);T[:3,:3]=r
    return T


def _unique_rel_target(rel):
    targets=rel.GetTargets() if rel else []
    if len(targets)!=1: return None
    return str(targets[0])


def find_usd_fr3_chain(stage, robot_root, link0_path, link8_path, arm_names):
    """发现USD刚体关节链：只接受 FixedJoint / RevoluteJoint。

    body0/body1 是USD物理连接关系而非视觉物体层级。遍历只取 link0→link8的
    真实关节链，并要求恰好包含7个fr3_joint1...7；对模型版本不符直接报错。
    若USD把某个关节定义成反向关系，则该段变换求逆，不暗中改变运动学定义。
    返回的dump可用于失败时观察确切关节和body路径。
    """
    root=stage.GetPrimAtPath(robot_root)
    if not root.IsValid():raise RuntimeError(f'找不到官方机器人根节点: {robot_root}')
    edges=[]; dump=[]
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim)
        b0=_unique_rel_target(joint.GetBody0Rel())
        b1=_unique_rel_target(joint.GetBody1Rel())
        kind='revolute' if prim.IsA(UsdPhysics.RevoluteJoint) else ('fixed' if prim.IsA(UsdPhysics.FixedJoint) else 'unsupported')
        name=str(prim.GetName())
        axis=str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if kind=='revolute' else None
        item={'name':name,'prim':str(prim.GetPath()),'body0':b0,'body1':b1,'kind':kind,'axis':axis}
        dump.append(item)
        if b0 and b1:
            item['local0']=_usd_local_pose(joint,0)
            item['local1']=_usd_local_pose(joint,1)
            edges.append(item)
    from collections import deque
    adj={}
    for edge in edges:
        adj.setdefault(edge['body0'],[]).append((edge['body1'],edge,1))
        adj.setdefault(edge['body1'],[]).append((edge['body0'],edge,-1))
    # 广度优先搜索：兼容某个joint在USD中被反向写入的情况。
    queue=deque([(link0_path,[])])
    seen={link0_path}
    chain=None
    while queue:
        node,path=queue.popleft()
        if node==link8_path:
            chain=path;break
        for nxt,edge,direction in adj.get(node,[]):
            if nxt not in seen:
                seen.add(nxt);queue.append((nxt,path+[(edge,direction)]))
    if chain is None:
        raise RuntimeError('USD关节链无法从link0连到link8；请检查joint_dump中body路径与链接名称。 joint_dump='+json.dumps(dump,ensure_ascii=False))
    used=[edge['name'] for edge,_ in chain if edge['kind']=='revolute']
    if len(used)!=7 or set(used)!=set(arm_names):
        raise RuntimeError(f'USD关节路径不是标准FR3七关节链: used={used}, expected={arm_names}, dump='+json.dumps(dump,ensure_ascii=False))
    if any(edge['kind']=='unsupported' for edge,_ in chain):
        raise RuntimeError('USD链包含非转动/固定关节，不能假设其位姿')
    return chain,dump


def usd_fk_tip(chain, q_by_name, T_world_link0, tip_link_local, axis_link_local):
    """纯 NumPy FK：不调用仿真器、不改变 q、不运行步进。

    关节闭合关系：T_world_body1 = T_world_body0 * T_local0 * R(q) * inv(T_local1)
    固定关节视为 R=I；USD方向相反时反转整段齐次变换。
    注意 q_ref 是期望关节角，**不是**实际 q_pd_target，更不是 PhysX 终态。
    """
    T=np.array(T_world_link0,dtype=float,copy=True)
    for edge,direction in chain:
        theta=float(q_by_name[edge['name']]) if edge['kind']=='revolute' else 0.
        rel=edge['local0']@(_axis_rotation(edge['axis'],theta) if edge['kind']=='revolute' else np.eye(4))@np.linalg.inv(edge['local1'])
        T=T@(rel if direction==1 else np.linalg.inv(rel))
    tip=T[:3,:3]@np.asarray(tip_link_local,dtype=float)+T[:3,3]
    axis=T[:3,:3]@np.asarray(axis_link_local,dtype=float)
    n=np.linalg.norm(axis)
    if n<1e-8:raise RuntimeError('棒轴长度异常')
    return tip,axis/n,T



# =============================================================================
# v13 ONLY：独立规划 / 验收 / 物理路点跟踪（前面保留原模型控制，旧实验函数不调用）
# =============================================================================
V13_TARGET_HEIGHT_M = 0.620  # 第一阶段安全高度，离孔口0.220m，绝非进入孔内
V13_POSITION_TOL_M = 0.00020       # 规划器每路点3D位置残差上限：0.20 mm
V13_PLAN_XY_TOL_M = 0.00030        # 规划路点相对初始XY漂移上限：0.30 mm
V13_PLAN_TILT_TOL_DEG = 0.50       # 规划路点棒轴偏离竖直的角度上限
V13_AXIS_EQUIV_LEVER_M = 0.15     # 姿态轴XY误差换算为等效平移(m)以进行加权IK
V13_FK_JACOBIAN_EPS_RAD = 1e-5    # 有限差分关节角扰动，rad
V13_LM_DAMPING_M = 0.002          # 加权DLS阻尼；对应加权任务空间单位m
V13_MAX_IK_DELTA_ITER_RAD = 0.014 # 每次LM迭代关节增量最大值
V13_MAX_WAYPOINT_JOINT_DELTA_RAD = 0.025 # 相邻路点关节最大跳跃0.025rad
V13_LIMIT_MARGIN_RAD = 0.050      # 距机械臂软限位安全余量
V13_MAX_PLAN_ITER = 100
V13_EXEC_XY_ERROR_M = 0.0010      # 执行期每个固定路点XY偏差容限：1.0mm
V13_EXEC_Z_ERROR_M = 0.0008       # 执行期Z偏差容限：0.8mm
V13_EXEC_TILT_DEG = 1.0          # 执行期倾角必须小于1度
V13_EXEC_Q_ERROR_RAD = 0.0050     # 执行期实际7关节对当前目标最大误差：0.005rad
V13_EXEC_QVEL_RAD_S = 0.040       # 稳定态关节最大速度0.04rad/s
V13_EXEC_TIP_NET_STEP_M = 0.00020  # 50ms内棒尖净位移<=0.20mm，作为连续稳定证据
V13_PHYSX_QVEL_SAFETY_RAD_S = 0.50  # 原始瞬时速度保留作安全监控，不再以0.04作为唯一稳定门槛
V13_STABLE_CYCLES = 4            # 连续4次(0.20s)满足所有条件才切换下一路点
V13_XY_SAFETY_GUARD_M = 0.025     # 25mm水平偏差早停；旧v9发生75mm漂移
V13_Z_UNDERSHOOT_GUARD_M = 0.012  # 低于本阶段目标高度12mm则立刻早停
V13_TILT_SAFETY_GUARD_DEG = 6.0


# 权重与微分步长沿用v13/v14已通过验证的姿态任务空间约定。
V13_AXIS_EQUIV_LEVER_M = 0.15  # m; 1 rad偏角等效150 mm任务位移
V13_FK_JACOBIAN_EPS_RAD = 1e-5  # rad; 中心差分微扰

def fk_task5(chain, q, Troot, tip_local, axis_local):
    """位置+竖直度5维任务：[x,y,z,0.15*axis_x,0.15*axis_y]。

    对圆柱棒，绕自身轴的yaw不需要约束；只控制棒轴向下的两个倾斜自由度。
    提醒：不能只用3维位置IK，否则我们本地校验出现约20°棒轴倾斜。
    """
    q=np.asarray(q,dtype=float)
    if q.shape!=(7,) or not np.isfinite(q).all():
        raise RuntimeError('规划输入q必须为7个有限弧度值')
    pos,axis,_=usd_fk_tip(chain,dict(zip(ARM_NAMES,q)),Troot,tip_local,axis_local)
    return np.concatenate((pos,V13_AXIS_EQUIV_LEVER_M*axis[:2]))


def fk_jacobian5(chain, q, Troot, tip_local, axis_local):
    """中心差分5x7雅可比，全部来自官方USD正运动学；不依赖接触仿真。"""
    eps=V13_FK_JACOBIAN_EPS_RAD
    columns=[]
    for idx in range(7):
        d=np.zeros(7);d[idx]=eps
        columns.append((fk_task5(chain,q+d,Troot,tip_local,axis_local)
                        -fk_task5(chain,q-d,Troot,tip_local,axis_local))/(2*eps))
    J=np.column_stack(columns)
    if J.shape!=(5,7) or not np.isfinite(J).all():
        raise RuntimeError('规划任务雅可比非法')
    return J




# ============================================================================
# v16：PPO训练专用局部环境（场景与力学仍沿用上面的 v15b 源代码）。
# ============================================================================

# 这个q来自已在本次工程中跑通的v15居中接近路径第192路点，棒尖z≈0.412258m。
# IMPORTANT：只可配合 seed=20261027 的原孔座几何使用；换seed则必须先重做专家规划。
V16_Q_SEED_RAD = np.array([
    0.03790153876812596, 0.007268177014736472, 0.045226080219733004,
    -2.0040962258541968, -0.0003630091164030796,
    2.0113551544561483, -0.00011615613045866232], dtype=np.float64)
V16_HOLE_XY_M = np.array([0.5487093117257903, 0.046225840044624614],dtype=np.float64)
V16_START_Z_M = 0.41225811531904677
# 每个action是[-1,1]的三维局部期望位移，xyz顺序，非力、非关节速度。
V16_ACTION_SCALE_M = np.array([0.00035, 0.00035, 0.00050],dtype=np.float64)
V16_INITIAL_XY_NOISE_M = 0.0020       # 训练起点随机偏差：x/y各±2mm；命令行可更改
V16_INITIAL_Z_NOISE_M = 0.0010        # 训练起点高度微扰±1mm
V16_MAX_TARGET_SHIFT_XY_M = 0.012    # 期望棒尖不得离开局部起点12mm XY
V16_MIN_TARGET_Z_M = 0.379          # 最多向下21mm，超过15mm成功阈值后立即结束
V16_MAX_TARGET_Z_M = 0.425          # 不允许向上走出局部任务
V16_MAX_Q_CMD_ERROR_RAD = 0.11      # 期望与实际任一关节严重不一致时结束该episode
V16_MAX_TILT_DEG = 6.0              # 棒轴倾角保护
V16_SUCCESS_DEPTH_M = 0.015         # 棒尖穿越孔口至少15mm
V16_SUCCESS_RADIAL_M = 0.0010       # 接触成功判定圆心偏差 <=1.0mm
V16_SUCCESS_TILT_DEG = 1.5         # 棒轴倾角 <=1.5度
V16_MAX_EPISODE_STEPS = 400        # 20Hz控制，原定每轮仿真最长20s
V16_OBS_DIM = 20                   # q7+dq7+带噪相对位移3+上一步动作3
V16_ACTION_DIM = 3
V1610_HOLE_RADIUS_M = 0.0115      # 孔半径：23 mm / 2，单位米；用于诊断几何包络，不是成功证明
V1610_PEG_RADIUS_M = 0.0100       # 棒半径：20 mm / 2，单位米
V1610_MAX_GEOM_DEPTH_M = 0.024   # v16保护终止阈值24mm，小于本场景30mm孔深


def v16_signed_insertion_depth_m(hole_mouth_z_m: float, tip_z_m: float) -> float:
    """计算棒尖相对孔口的有符号插入深度（单位：米）。

    该纯数学函数刻意不依赖 Isaac Lab、Torch 和 NumPy，供宿主机直接回归测试。
    坐标系 world Z 向上；故“孔口高度减棒尖高度”才是插入深度。
    例如：hole=0.400m, tip=0.412m -> -0.012m（孔口上方12mm）；
          hole=0.400m, tip=0.382m -> +0.018m（插入18mm）。
    不能取绝对值，也不能反号，否则会污染终止条件、奖励和成功率。
    """
    hole=float(hole_mouth_z_m)
    tip=float(tip_z_m)
    if not math.isfinite(hole) or not math.isfinite(tip):
        raise ValueError('孔口高度和棒尖高度必须为有限数值（m）')
    return hole-tip



def v1610_geometric_insertion_proxy(tip_xyz, axis_down, hole_xyz):
    """纯几何辅助判定，不属于正式成功率；用于复核前几版的1.0mm过严问题。

    输入（均为米）：棒尖世界坐标、沿棒尖方向的【向下】单位轴、孔口中心坐标。
    直径20/23mm；直圆柱插入孔内时不仅棒尖不能撞壁，孔口截面的轴心也必须容纳。
    对于刚性直圆柱，孔内圆柱轴线在Z方向上是线性插值，最远离孔轴的中心点
    出现在该轴段的某个端点，因此检查棒尖与孔口交点两端径向距离是必要的几何筛选。

    注意：这是理想圆柱几何代理。没有检测逐接触点冲量、USD网格穿透或完整机器人状态，
    也没有模拟棒体的倒角/弹性，因此只能称为 `geometric_candidate`。
    """
    # 纯标准库实现：便于在DGX宿主机用 python3 -S 直接核对几何公式。
    p=tuple(float(v) for v in tip_xyz)
    a=tuple(float(v) for v in axis_down)
    hole=tuple(float(v) for v in hole_xyz)
    if any(len(z)!=3 for z in (p,a,hole)) or not all(math.isfinite(v) for z in (p,a,hole) for v in z):
        raise ValueError('棒尖/棒轴/孔中心须为3维有限世界坐标')
    anorm=math.sqrt(sum(v*v for v in a))
    if anorm<=1e-6:
        raise ValueError('棒轴零向量，不允许参与几何插入诊断')
    a=tuple(v/anorm for v in a)
    depth=hole[2]-p[2]
    tip_radial=math.hypot(p[0]-hole[0],p[1]-hole[1])
    theta=math.degrees(math.acos(max(-1.,min(1.,-a[2]))))
    valid_depth=(V16_SUCCESS_DEPTH_M<=depth<=V1610_MAX_GEOM_DEPTH_M)
    if a[2]>=-1e-4 or depth<=0:
        return {'geometric_candidate':False,'depth_mm':depth*1000.,
                'tip_radial_mm':tip_radial*1000.,'mouth_radial_mm':None,
                'max_axis_radial_mm':None,'min_ideal_clearance_mm':None,
                'tilt_deg':theta,'reason':'axis_orientation_or_depth_invalid'}
    # 轴 a 指向棒尖方向；沿反方向从棒尖上溯到孔口：z变化 depth = -a_z*t。
    t=depth/(-a[2])
    mouth_xy=(p[0]-a[0]*t,p[1]-a[1]*t)
    mouth_radial=math.hypot(mouth_xy[0]-hole[0],mouth_xy[1]-hole[1])
    max_center_radial=max(tip_radial,mouth_radial)
    # 理想圆柱截面圆与孔截面圆不重叠的余量；等于0代表恰好接触孔壁。
    clearance=V1610_HOLE_RADIUS_M-V1610_PEG_RADIUS_M-max_center_radial
    candidate=valid_depth and theta<=V16_SUCCESS_TILT_DEG and clearance>=-1e-12
    return {'geometric_candidate':bool(candidate),'depth_mm':depth*1000.,
            'tip_radial_mm':tip_radial*1000.,'mouth_radial_mm':mouth_radial*1000.,
            'max_axis_radial_mm':max_center_radial*1000.,
            'min_ideal_clearance_mm':clearance*1000.,'tilt_deg':theta,
            'reason':'ideal_cylinder_envelope_only_not_physx_certified'}


@configclass
class FR3LocalTrainCfg(FR3PegHoleSceneCfg):
    """严格对应上面的观测和动作，不允许把v15b的7维占位接口拿去训练。"""
    action_space = V16_ACTION_DIM
    observation_space = V16_OBS_DIM
    state_space = 0
    episode_length_s = V16_MAX_EPISODE_STEPS*CTRL_DT


class FR3LocalPPOEnv(PhysxContactContrastEnv):
    """给网络提供真正可执行的连续动作、完整RL奖励和可重置局部初态。"""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._v16_ready = False    # DirectRLEnv初始化过程中可能自动reset；要绕开未完成的FK缓存。
        self._v16_rng = np.random.default_rng(int(args.seed)+216)
        self._v16_successes = 0
        self._v16_finished = 0
        self._v16_failure_categories = {}
        self._v16_last_done = {}
        self._v16_current = None
        # v16.2：独立收集每个真正结束的episode，不能只看每个PPO rollout的loss。
        # 在 _get_dones 标记结束，在下一次 _reset_idx 完成最后一步奖励结算后归档。
        self._v16_completed_records = []
        self._v16_episode_reward_sum = 0.0
        self._v16_episode_steps = 0
        self._v16_pending_end = None
        super().__init__(cfg, render_mode, **kwargs)
        self._v16_last_action = torch.zeros((1,3),device=self.device)

    def prepare_v16(self):
        """必须在Gym创建且环境至少完成1次physics.step之后调用。

        重新抽取v12已验证过的USD FR3关节链，从link0构建数学FK，绝不依赖Panda假机器人。
        为防止使用错seed/资产造成穿模，对已知专家起点进行严格几何检查。
        """
        stage = omni.usd.get_context().get_stage()
        chain, _ = find_usd_fr3_chain(stage,ROBOT_ROOT,ROBOT_ROOT+'/fr3_link0',LINK8_PATH,list(self.arm_names))
        root_ids,_ = self.robot.find_bodies('fr3_link0')
        if len(root_ids)!=1:
            raise RuntimeError('v16无法定位FR3根刚体，拒绝开始训练')
        i=root_ids[0]
        self._v16_root_T = _pose_matrix(self.robot.data.body_link_pos_w[0,i].detach().cpu().numpy(),
                                        self.robot.data.body_link_quat_w[0,i].detach().cpu().numpy())
        self._v16_chain = chain
        hole = np.asarray(self.episode_truth['hole_center_world_m'],dtype=float)
        if np.linalg.norm(hole[:2]-V16_HOLE_XY_M)>0.00001 or abs(hole[2]-geom.HOLE_BASE_WORLD[2])>0.00001:
            raise RuntimeError('本版局部预置q仅适用于20261027孔位；随机孔位不能冒充已实现')
        self._v16_hole=hole
        tip = fk_task5(chain,V16_Q_SEED_RAD,self._v16_root_T,self.tip_link_local_m,self.axis_link_local)[:3]
        errxy=float(np.linalg.norm(tip[:2]-hole[:2]))
        errz=float(abs(tip[2]-V16_START_Z_M))
        print(f'[V16_FK_SEED_CHECK] tip={tip.tolist()} xy_m={errxy:.8f} z_m={errz:.8f}',flush=True)
        if errxy>0.0010 or errz>0.0010:
            raise RuntimeError('已记录的专家q与当前FR3场景FK不符；禁止盲目训练')
        self._v16_start_tip = tip
        self._v16_soft_limits = self.robot.data.soft_joint_pos_limits[0,self.arm_ids].detach().cpu().numpy()
        self._v16_ready=True

    def _v16_random_start_q(self):
        """局部reset：在专家安全起点的任务空间采样小偏差，再用FK的5维阻尼雅可比求关节姿态。

        只改变【机器人初态】，不在运行中的PhysX里移动静态Hole和工具安装USD。
        此设计是v16第一轮训练的明确边界，不是完成10cm随机孔位/安装误差。
        """
        for _ in range(12):
            dx=float(self._v16_rng.uniform(-args.reset_xy_mm,args.reset_xy_mm))/1000.
            dy=float(self._v16_rng.uniform(-args.reset_xy_mm,args.reset_xy_mm))/1000.
            dz=float(self._v16_rng.uniform(-1.,1.))/1000.
            J=fk_jacobian5(self._v16_chain,V16_Q_SEED_RAD,self._v16_root_T,
                           self.tip_link_local_m,self.axis_link_local)
            task_delta=np.array([dx,dy,dz,0.,0.])
            lam=0.002
            dq=J.T@np.linalg.solve(J@J.T + lam*lam*np.eye(5),task_delta)
            q=V16_Q_SEED_RAD+dq
            if not np.isfinite(q).all() or np.max(np.abs(dq))>0.05:
                continue
            if np.any(q<=self._v16_soft_limits[:,0]+0.02) or np.any(q>=self._v16_soft_limits[:,1]-0.02):
                continue
            xyz=fk_task5(self._v16_chain,q,self._v16_root_T,self.tip_link_local_m,self.axis_link_local)[:3]
            if xyz[2]<self._v16_hole[2]+0.006:
                continue
            if np.linalg.norm(xyz[:2]-self._v16_hole[:2])>0.006:
                continue
            return q,xyz
        raise RuntimeError('v16局部起点采样12次均不满足安全条件；请检查FR3限位/FK')

    def _reset_idx(self, env_ids):
        # DirectRLEnv完成episode后会调用这里。在reset清空缓存之前，归档刚结束的回合。
        # _get_dones 提供终态成功/失败、深度、径向偏差；_get_rewards 记录真实奖励和步数。
        # 这样即使一个PPO rollout跨越多个episode，也可正确计算逐回合指标。
        if getattr(self, '_v16_ready', False) and self._v16_pending_end is not None:
            record = dict(self._v16_pending_end)
            record['episode_return'] = float(self._v16_episode_reward_sum)
            record['actual_control_steps'] = int(self._v16_episode_steps)
            self._v16_completed_records.append(record)
            print('[V162_EPISODE] ' + json.dumps(record, ensure_ascii=False), flush=True)
            self._v16_pending_end = None
        self._v16_episode_reward_sum = 0.0
        self._v16_episode_steps = 0
        # 官方 DirectRLEnv要求先清理场景/episode统计，再进行articulation reset。
        super()._reset_idx(env_ids)
        if not self._v16_ready:
            return
        if env_ids is None:
            env_ids=self.robot._ALL_INDICES
        if len(env_ids)!=1 or int(env_ids[0])!=0:
            raise RuntimeError('本版只支持单环境episode随机重置')
        q,xyz=self._v16_random_start_q()
        full=self.robot.data.default_joint_pos[env_ids].clone()
        full[:,self.arm_ids]=torch.as_tensor(q,dtype=full.dtype,device=self.device).unsqueeze(0)
        vel=torch.zeros_like(full)
        self.robot.write_joint_state_to_sim(full,vel,None,env_ids)
        self._ref=full.clone()
        self._pd_target=full.clone()
        self._bias=torch.zeros_like(full)
        # v16.7: 保留已关闭的v16.4位置偏置试验代码，但强制bias-scale=0。
        # 基于v16.3的关节静态偏移估计，低于相同构型下的摩擦/接触/重力完整动力学模型。
        #      joint2：无动作时关节角上升约 +0.0129rad，故对位置目标预置 -0.0129rad。
        #      joint4：无动作时关节角下降约 -0.0110rad，故对位置目标预置 +0.0110rad。
        # 单位弧度；实验对照的0/0.5/1比例，均通过同一场景与随机数种子独立运行。
        # 这里只设置 PhysX 位置伺服目标，不瞬移关节；绝不能把这个补偿冒充真实控制性能。
        j2=self.arm_ids[1]
        j4=self.arm_ids[3]
        self._bias[:,j2] = -0.0129*args.bias_scale
        self._bias[:,j4] = +0.0110*args.bias_scale
        self._pd_target[:,self.arm_ids] += self._bias[:,self.arm_ids]
        self._v16_last_action=torch.zeros((1,3),device=self.device)
        self._v16_previous_depth_m=float(self._v16_hole[2]-xyz[2])
        self._v16_last_done={}
        self._v16_current=None
        # 允许改变的是episode初始关节状态，不在每个物理step内瞬移机械臂。

    def _v16_true_state(self):
        """特权状态仅供环境reward/done和带噪观测模拟，不直接返回给Actor。

        【重要修复 v16.1】插入深度统一遵守有符号约定：
          depth_m = z_hole_mouth - z_peg_tip
          * < 0：棒尖在孔口上方，尚未插入。例如 0.412m => -12mm。
          * = 0：棒尖与孔口等高。
          * > 0：棒尖在孔口以下。例如 0.382m => +18mm。
        上一版误写为 -relative[2]，导致起点被误报为“已插入”并阻断PPO。
        reward、success、超深保护和 reset gate 必须使用同一约定。
        """
        s=self.measure_peg_tip_oracle()
        tip=np.asarray(s['tip_world_xyz_m'],dtype=float)
        relative=self._v16_hole-tip
        radial=float(np.linalg.norm(relative[:2]))
        depth=v16_signed_insertion_depth_m(self._v16_hole[2],tip[2])
        if abs(depth-float(relative[2]))>1e-12:
            raise RuntimeError('v16.1插入深度符号或相对位置定义出现内部不一致')
        return {'tip':tip,'relative':relative,'radial':radial,'depth':depth,
                'tilt_deg':float(s['tilt_from_world_down_deg'])}

    def _pre_physics_step(self, actions):
        """Network action xyz微位移 -> 利用机器人自己已知的FK映射为关节期望轨迹。

        与v9关键区别：新目标是在上一次q_ref基础上推进，绝不用q_actual+dq继承底层跟踪误差。
        此低层控制器【不读取孔中心】，只知道机器人模型和局部动作。到孔中心方向必须由Actor学习。
        """
        if tuple(actions.shape)!=(1,3) or not bool(torch.isfinite(actions).all().item()):
            raise RuntimeError('v16训练动作必须是(1,3)有限张量')
        if not self._v16_ready:
            # 仅用于prepare之前让Isaac Sim执行少数步更新关节缓存。
            self._pd_target=self._anchor.clone()
            return
        u=np.clip(actions[0].detach().cpu().numpy(),-1.,1.)
        self._v16_last_action=torch.as_tensor(u,dtype=torch.float32,device=self.device).reshape(1,3)
        q=self._ref[0,self.arm_ids].detach().cpu().numpy().astype(float)
        cur=fk_task5(self._v16_chain,q,self._v16_root_T,self.tip_link_local_m,self.axis_link_local)
        dx=u*V16_ACTION_SCALE_M
        wanted=cur.copy()
        wanted[:3]+=dx
        # 工作空间限位以专家起点为中心，不使用孔的绝对真值做隐形自动对准。
        wanted[:2]=np.clip(wanted[:2],self._v16_start_tip[:2]-V16_MAX_TARGET_SHIFT_XY_M,
                             self._v16_start_tip[:2]+V16_MAX_TARGET_SHIFT_XY_M)
        wanted[2]=np.clip(wanted[2],V16_MIN_TARGET_Z_M,V16_MAX_TARGET_Z_M)
        wanted[3:]=0.0             # 棒轴保持竖直，已证实3D位置IK可能导致约20°倾斜。
        J=fk_jacobian5(self._v16_chain,q,self._v16_root_T,self.tip_link_local_m,self.axis_link_local)
        task_error=wanted-cur
        lam=0.002
        dq=J.T@np.linalg.solve(J@J.T + lam*lam*np.eye(5),task_error)
        dq=np.clip(dq,-0.008,0.008)
        q_next=np.clip(q+dq,self._v16_soft_limits[:,0]+0.02,self._v16_soft_limits[:,1]-0.02)
        self._ref[0,self.arm_ids]=torch.as_tensor(q_next,dtype=self._ref.dtype,device=self.device)
        error=self._ref[:,self.arm_ids]-self.robot.data.joint_pos[:,self.arm_ids]
        step=torch.clamp(KI*CTRL_DT*error,-0.003,0.003)
        self._bias[:,self.arm_ids]=torch.clamp(self._bias[:,self.arm_ids]+step,-0.025,0.025)
        self._pd_target=self._ref.clone()
        self._pd_target[:,self.arm_ids]+=self._bias[:,self.arm_ids]

    def _get_observations(self):
        q=self.robot.data.joint_pos[:,self.arm_ids]
        vel=self.robot.data.joint_vel[:,self.arm_ids]
        if not self._v16_ready:
            pad=torch.zeros((1,6),device=self.device)
            return {'policy':torch.cat([q/3.0,vel/3.0,pad],dim=-1).to(torch.float32)}
        # 模拟视觉/位姿估计器：每个维度重新抽样带噪相对位置，绝不输出未加噪声的oracle值。
        true=self._v16_true_state()
        measured=true['relative'] + self._v16_rng.normal(0.,args.obs_noise_mm/1000.,size=3)
        measured=np.clip(measured/0.02,-3.,3.)  # 把20mm范围归一化到约1；不是毫米原值。
        sensor=torch.as_tensor(measured,dtype=torch.float32,device=self.device).reshape(1,3)
        obs=torch.cat([q/3.0,vel/3.0,sensor,self._v16_last_action],dim=-1).float()
        if tuple(obs.shape)!=(1,V16_OBS_DIM) or not bool(torch.isfinite(obs).all().item()):
            raise RuntimeError('v16观察不满足20维有限数值白名单')
        return {'policy':obs}

    def _get_dones(self):
        if not self._v16_ready:
            zeros=torch.zeros(1,dtype=torch.bool,device=self.device)
            return zeros,zeros
        s=self._v16_true_state()
        self._v16_current=s
        success=(s['depth']>=V16_SUCCESS_DEPTH_M and s['radial']<=V16_SUCCESS_RADIAL_M
                 and s['tilt_deg']<=V16_SUCCESS_TILT_DEG)
        joint_actual=self.robot.data.joint_pos[0,self.arm_ids]
        qerror=float(torch.max(torch.abs(self._ref[0,self.arm_ids]-joint_actual)).item())
        oob=(s['radial']>=0.012 or s['tilt_deg']>V16_MAX_TILT_DEG or qerror>V16_MAX_Q_CMD_ERROR_RAD
             or s['depth']>0.024 or not np.isfinite(s['tip']).all())
        # 旧v16.9使用>=max_episode_length-1，导致240步实际在239步结束。
        # v16.10使用>=max_episode_length，保证400×0.05=20.00秒，而非19.95秒。
        timeout=bool(self.episode_length_buf[0]>=self.max_episode_length)
        ended=success or oob
        if success or oob or timeout:
            self._v16_finished+=1
            if success:self._v16_successes+=1
            reason='success' if success else ('out_of_bounds' if oob else 'timeout')
            self._v16_failure_categories[reason]=self._v16_failure_categories.get(reason,0)+1
            self._v16_last_done={'reason':reason,'depth_mm':s['depth']*1000.,
                                 'radial_mm':s['radial']*1000.,'tilt_deg':s['tilt_deg'],
                                 'steps':int(self.episode_length_buf[0])}
            # 记录在reset前，绝不让下一回合覆盖本回合终态。
            self._v16_pending_end={'episode':int(self._v16_finished), **self._v16_last_done}
            if self._v16_finished<=3 or self._v16_finished%10==0:
                print('[V16_EPISODE] '+json.dumps({'episode':self._v16_finished,**self._v16_last_done},ensure_ascii=False),flush=True)
        return (torch.tensor([ended],dtype=torch.bool,device=self.device),
                torch.tensor([timeout],dtype=torch.bool,device=self.device))

    def _get_rewards(self):
        if not self._v16_ready:
            return torch.zeros(1,device=self.device)
        s=self._v16_current if self._v16_current is not None else self._v16_true_state()
        depth=s['depth']
        depth_gain_mm=(depth-self._v16_previous_depth_m)*1000.
        self._v16_previous_depth_m=depth
        # Dense奖励用于帮助首次PPO启动；不是声称最优奖励设计：
        # 进入孔前：精度更好、在保持对准时下降能够获得奖励；无效动作产生小惩罚。
        # 插入孔内：轴线偏差仍严格要求<1mm，不奖励偏心穿透。
        r=1.5*max(-2.0,min(2.0,depth_gain_mm))
        r-=0.025*(s['radial']*1000.)
        r-=0.0025*abs(float(self._v16_hole[2]-0.016)-s['tip'][2])*1000.
        r-=0.01*float(torch.sum(self._v16_last_action.square()).item())
        if s['depth']>=V16_SUCCESS_DEPTH_M and s['radial']<=V16_SUCCESS_RADIAL_M and s['tilt_deg']<=V16_SUCCESS_TILT_DEG:
            r+=30.0
        if s['radial']>=0.012 or s['tilt_deg']>V16_MAX_TILT_DEG or s['depth']>0.024:
            r-=15.0
        # v16.2：只做统计，不把累计奖励写入Actor观测。
        self._v16_episode_reward_sum += float(r)
        self._v16_episode_steps += 1
        return torch.tensor([r],dtype=torch.float32,device=self.device)


# ============================================================================
# 小型原生PyTorch PPO：不依赖rsl_rl、skrl或额外pip安装，降低第一次训练环境门槛。
# ============================================================================
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """共享输入、分离actor/critic的两层MLP；参数通过实际反向传播更新。"""
    def __init__(self,obs_dim=V16_OBS_DIM,action_dim=V16_ACTION_DIM):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),
                                 nn.Linear(128,action_dim),nn.Tanh())
        self.critic=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((action_dim,),-0.9))
    def distribution(self,obs):
        return Normal(self.actor(obs),self.log_std.exp().expand(obs.shape[0],-1))
    def value(self,obs):return self.critic(obs).squeeze(-1)
    def evaluate(self,obs,action):
        d=self.distribution(obs)
        return d.log_prob(action).sum(-1),d.entropy().sum(-1),self.value(obs)


def ppo_update(agent,opt,buffer,last_value,gamma=0.99,lam=0.95):
    """GAE + clipped policy objective + value loss，实际optimizer.step更新权重。

    单环境版本，用CPU/PyTorch向量做GAE；done掩码避免不同episode互相传播回报。
    注：时间截断处这里也截断优势，属于第一个可运行PPO最小实现，非最终高效训练器。
    """
    obs=torch.cat(buffer['obs'],dim=0)
    act=torch.cat(buffer['action'],dim=0)
    oldlp=torch.cat(buffer['logp'],dim=0).detach()
    rewards=torch.cat(buffer['reward'],dim=0).detach()
    done=torch.cat(buffer['done'],dim=0).detach()
    values=torch.cat(buffer['value'],dim=0).detach()
    adv=torch.zeros_like(rewards)
    gae=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        nextv=last_value if t==len(rewards)-1 else values[t+1:t+2]
        mask=1.-done[t:t+1]
        delta=rewards[t:t+1]+gamma*nextv*mask-values[t:t+1]
        gae=delta+gamma*lam*mask*gae
        adv[t:t+1]=gae
    target=(adv+values).detach()
    normalized=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
    info={}
    for _ in range(args.epochs):
        lp,entropy,val=agent.evaluate(obs,act)
        ratio=(lp-oldlp).exp()
        pg=-torch.minimum(ratio*normalized,torch.clamp(ratio,0.8,1.2)*normalized).mean()
        vl=(val-target).square().mean()
        ent=entropy.mean()
        loss=pg+0.5*vl-0.005*ent
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError('PPO loss产生NaN/Inf，立即停止')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(),1.0)
        opt.step()
        info={'policy_loss':float(pg.item()),'value_loss':float(vl.item()),
              'entropy':float(ent.item()),'loss':float(loss.item())}
    return info


def cuda_pytorch_preflight(device):
    """DGX Spark GB10的CUDA架构兼容性必须先做真实计算+反向传播，不能只看is_available。"""
    torch.manual_seed(args.seed)
    m=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,3)).to(device)
    x=torch.randn(32,16,device=device)
    y=m(x).square().mean()
    y.backward()
    torch.cuda.synchronize() if str(device).startswith('cuda') else None
    if not bool(torch.isfinite(y).item()) or any(p.grad is None for p in m.parameters()):
        raise RuntimeError('CUDA/PyTorch前向或反向计算失败')
    return {'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'device':str(device),'gpu_name':torch.cuda.get_device_name(0) if str(device).startswith('cuda') else 'CPU',
            'loss':float(y.item()),'backward':True}


def _v162_load_checkpoint(path, network, optimizer=None):
    """严格加载真实v16.1/v16.2 checkpoint，避免维度不一致或错把随机网络当恢复成功。

    * network.state_dict 严格加载；20维观测/3维动作必须完全一致。
    * 训练恢复时，优化器Adam的动量/step同步恢复。若缺失优化器则直接拒绝训练。
    * checkpoint文件只含模型与优化器，不含模拟器刚体状态；下次episode会重新采样。
    * 不允许任意文件反序列化：PyTorch使用 weights_only=True。
    """
    if path is None or not path.is_file():
        raise FileNotFoundError('找不到恢复权重文件：'+str(path))
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict):
        raise ValueError('checkpoint顶层必须为dict')
    if int(checkpoint.get('obs_dim',-1))!=V16_OBS_DIM or int(checkpoint.get('action_dim',-1))!=V16_ACTION_DIM:
        raise ValueError('checkpoint网络输入输出维度不符，拒绝加载')
    if 'model' not in checkpoint:
        raise ValueError('checkpoint中缺少model')
    network.load_state_dict(checkpoint['model'],strict=True)
    if optimizer is not None:
        if 'optimizer' not in checkpoint:
            raise ValueError('不能断点续训：checkpoint中缺少Adam优化器状态')
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Spark GPU续训：Adam一阶/二阶矩必须与参数在同一设备。
        # 注意非capturable Adam的step标量应留在CPU，不能粗暴把所有状态都迁至CUDA。
        # optim.load_state_dict通常会自行处理，但此处显式保证旧v16保存的状态兼容。
        device=next(network.parameters()).device
        capturable=bool(optimizer.param_groups[0].get('capturable',False))
        for states in optimizer.state.values():
            for key,value in list(states.items()):
                if torch.is_tensor(value):
                    target=device if (key!='step' or capturable) else torch.device('cpu')
                    states[key]=value.to(target)
    step=int(checkpoint.get('steps',-1)); update=int(checkpoint.get('update',-1))
    if step<0 or update<0:raise ValueError('checkpoint中steps/update非法')
    return checkpoint,{'resume_path':str(path),'steps_before':step,'updates_before':update,
                       'optimizer_restored':optimizer is not None}


def _v162_save_checkpoint(output_dir,network,optimizer,steps,updates,episodes,successes):
    """每轮更新后原子写入：断电时不会把上一份有效checkpoint写坏。"""
    output_dir.mkdir(parents=True,exist_ok=True)
    dest=output_dir/'ppo_v16_2_latest.pt'
    tmp=output_dir/'ppo_v16_2_latest.pt.tmp'
    torch.save({'model':network.state_dict(),'optimizer':optimizer.state_dict(),
                'steps':int(steps),'update':int(updates),'seed':int(args.seed),
                'obs_dim':V16_OBS_DIM,'action_dim':V16_ACTION_DIM,
                'completed_episodes':int(episodes),'successes':int(successes),
                'version':VERSION},tmp)
    os.replace(tmp,dest)
    return dest


def _v162_episode_metrics(records):
    """汇总真正已终止的episode，明确成功率分母，不把未结束回合算失败。"""
    n=len(records)
    success=sum(r.get('reason')=='success' for r in records)
    def mean(key):
        return float(sum(float(r.get(key,0.0)) for r in records)/n) if n else None
    reasons={}
    for r in records:reasons[r['reason']]=reasons.get(r['reason'],0)+1
    return {'completed_episodes':n,'successes':success,
            'success_rate':float(success/n) if n else None,
            'mean_episode_return':mean('episode_return'),
            'mean_terminal_depth_mm':mean('depth_mm'),
            'mean_terminal_radial_mm':mean('radial_mm'),
            'mean_min_radial_mm':mean('min_radial_mm'),
            'mean_max_depth_mm':mean('max_depth_mm'),
            'mean_last20_depth_progress_mm':mean('last20_depth_progress_mm'),
            'mean_last20_radial_improvement_mm':mean('last20_radial_improvement_mm'),
            'failure_reasons':reasons}


def train_loop(env,network,output_dir):
    """v16.2真正PPO训练：可恢复权重+Adam，每个episode单独统计，继续保存可恢复checkpoint。

    --steps 只表示【本次额外采样】多少步。已有256步+本次2048步=累计2304步。
    注意：仿真重置、随机数、rollout缓冲不会从旧checkpoint重建，因此是episode级续训，
    并非严格逐位复现一次从未中断的长跑；它保留了已学模型和Adam动量。
    """
    obs,_=env.reset(seed=args.seed)
    state=obs['policy']
    opt=torch.optim.Adam(network.parameters(),lr=3e-4)
    total_before=0; updates_before=0; episodes_before=0; successes_before=0
    if args.resume is not None:
        checkpoint,restore=_v162_load_checkpoint(args.resume,network,opt)
        total_before=restore['steps_before'];updates_before=restore['updates_before']
        episodes_before=int(checkpoint.get('completed_episodes',0))
        successes_before=int(checkpoint.get('successes',0))
        # v16.1早期checkpoint没有这两个计数时，从相邻历史的最后一轮补齐。
        oldhist=args.resume.with_name('train_history_v16.json')
        if oldhist.is_file() and 'completed_episodes' not in checkpoint:
            try:
                last=json.loads(oldhist.read_text(encoding='utf8'))[-1]
                episodes_before=int(last.get('episodes',0))
                successes_before=int(last.get('successes',0))
            except (ValueError,IndexError,TypeError,KeyError):
                pass
        print('[V162_RESUMED] '+json.dumps(restore,ensure_ascii=False),flush=True)
    else:
        restore={'resume_path':None,'steps_before':0,'updates_before':0,
                 'optimizer_restored':False}
        print('[V162_FRESH_TRAINING] 从随机网络开始训练',flush=True)
    # 同一输出目录反复续训时，保留此前v16.2每轮更新历史；
    # 旧v16.1历史仍留在原来的train目录，不改写原文件。
    history_file=output_dir/'train_history_v16_2.json'
    # v16.9: 如果新实验目录还没有历史，则尝试从本次指定Checkpoint的兄弟文件继承曲线。
    # 这不会修改原有历史；旧记录没有新增对中指标时就保持缺失，不擅自填0。
    source_history=history_file if history_file.is_file() else (
        args.resume.with_name('train_history_v16_2.json') if args.resume is not None else None)
    if source_history is not None and source_history.is_file():
        try:
            old_history=json.loads(source_history.read_text(encoding='utf8'))
            history=old_history if isinstance(old_history,list) else []
            history=[row for row in history if int(row.get('update',-1))<=updates_before]
        except (ValueError,TypeError,AttributeError):
            history=[]
    else:
        history=[]
    run_steps=0; update_id=updates_before
    first_param=next(network.parameters()).detach().clone()
    while run_steps<args.steps:
        buffer={k:[] for k in ('obs','action','logp','reward','done','value')}
        batch_steps=min(args.rollout,args.steps-run_steps)
        for _ in range(batch_steps):
            with torch.no_grad():
                d=network.distribution(state)
                raw=d.sample()
                logp=d.log_prob(raw).sum(-1)
                value=network.value(state)
            # 与v16.1保持动作采样语义一致，确保可以直接恢复旧checkpoint。
            next_obs,reward,term,trunc,_=env.step(raw.clamp(-1.,1.))
            ending=(term|trunc).to(torch.float32)
            for k,v in (('obs',state),('action',raw),('logp',logp),('reward',reward),('done',ending),('value',value)):
                buffer[k].append(v.detach().clone())
            state=next_obs['policy'].detach()
            run_steps+=1
        with torch.no_grad():last=network.value(state)
        update_info=ppo_update(network,opt,buffer,last)
        update_id+=1
        metrics=_v162_episode_metrics(env.unwrapped._v16_completed_records)
        item={'update':update_id,'steps':total_before+run_steps,
              'steps_this_run':run_steps,'episodes_this_run':metrics['completed_episodes'],
              'successes_this_run':metrics['successes'],**update_info,
              'mean_episode_return_this_run':metrics['mean_episode_return'],
              'mean_terminal_depth_mm_this_run':metrics['mean_terminal_depth_mm'],
              'mean_terminal_radial_mm_this_run':metrics['mean_terminal_radial_mm'],
              'mean_min_radial_mm_this_run':metrics['mean_min_radial_mm'],
              'mean_max_depth_mm_this_run':metrics['mean_max_depth_mm'],
              'mean_last20_depth_progress_mm_this_run':metrics['mean_last20_depth_progress_mm'],
              'mean_last20_radial_improvement_mm_this_run':metrics['mean_last20_radial_improvement_mm'],
              'failure_reasons_this_run':metrics['failure_reasons']}
        print('[V162_PPO_UPDATE] '+json.dumps(item,ensure_ascii=False),flush=True)
        history.append(item)
        _v162_save_checkpoint(output_dir,network,opt,total_before+run_steps,update_id,
                              episodes_before+metrics['completed_episodes'],
                              successes_before+metrics['successes'])
        # 日志存于新目录；不覆盖v16.1已训练好的.pt及历史。
        history_file.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf8')
        (output_dir/'episode_records_v16_2.json').write_text(
            json.dumps(env.unwrapped._v16_completed_records,ensure_ascii=False,indent=2),encoding='utf8')
    changed=float(torch.max(torch.abs(next(network.parameters()).detach()-first_param)).item())
    if changed<=0 or not math.isfinite(changed):
        raise RuntimeError('PPO确实运行了但本次没有参数更新，拒绝标记成功')
    metrics=_v162_episode_metrics(env.unwrapped._v16_completed_records)
    return {'status':'PPO_UPDATES_COMPLETED','new_environment_steps':run_steps,
            'cumulative_steps':total_before+run_steps,'new_updates':update_id-updates_before,
            'cumulative_updates':update_id,'resume':restore,
            'max_parameter_change_this_run':changed,'episodes_this_run':metrics,
            'checkpoint':'ppo_v16_2_latest.pt',
            'limitations':'单孔固定真值场景的局部任务。没有验证10x10cm孔位/安装偏差/90%成功率'}


def evaluate_loop(env,network,output_dir):
    """独立评估：仅加载已训练网络，采用动作分布均值，不调用optimizer.step。

    孔位置仍固定，随机初始状态换用评估专用RNG流。成功率分母为已完成episode数。
    这是固定孔位【局部】泛化检验，不能被写成完整老师任务的90%指标。
    """
    if args.resume is None:
        raise RuntimeError('eval模式必须提供--resume权重路径；拒绝评估随机初始化网络')
    _,info=_v162_load_checkpoint(args.resume,network,None)
    network.eval()
    # 仅改变局部初始随机位移和观测噪声的抽样序列，孔真值与机器人仍然不变。
    u=env.unwrapped
    u._v169_eval_pairing=True
    u._v169_eval_reset_index=0
    u._v169_episode_traces.clear()
    u._v16_completed_records.clear()
    u._v16_finished=0
    u._v16_successes=0
    u._v16_pending_end=None
    # 首次评估reset会创建第一个独立种子。之后每次自动重置各用下一个独立种子。

    print('[V162_EVAL_START] '+json.dumps({'episodes':args.eval_episodes,**info},ensure_ascii=False),flush=True)
    obs,_=env.reset(seed=args.seed)
    for step in range(int(args.eval_episodes)*(args.max_episode_steps+2)+10):
        with torch.no_grad():
            action=network.distribution(obs['policy']).mean.clamp(-1.,1.)
        obs,_,terminated,truncated,_=env.step(action)
        if len(env.unwrapped._v16_completed_records)>=args.eval_episodes:
            break
    records=env.unwrapped._v16_completed_records[:args.eval_episodes]
    if len(records)!=args.eval_episodes:
        raise RuntimeError(f'评估计划{args.eval_episodes}回合，实际只有{len(records)}回合')
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'eval_episodes_v16_2.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    traces=u._v169_episode_traces[:args.eval_episodes]
    if len(traces)!=args.eval_episodes:
        raise RuntimeError('配对评估轨迹数与完成回合数不匹配，拒绝报告成功率')
    (output_dir/'eval_traces_v16_10.json').write_text(
        json.dumps(traces,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    result=_v162_episode_metrics(records)
    result.update({'status':'EVALUATION_COMPLETED','episodes_requested':args.eval_episodes,
                   'policy_mode':'deterministic_mean','checkpoint':str(args.resume),
                   'optimizer_step_called':False,
                   'evaluation_reset_stream':'per_episode:seed+1000216+1000003*i',
                   'max_episode_steps_configured':int(args.max_episode_steps),
                   'max_episode_steps_actual':int(u.max_episode_length),
                   'episode_time_seconds':float(args.max_episode_steps*CTRL_DT),
                   'trace_file':'eval_traces_v16_10.json',
                   'official_evaluation_horizon_s':20.0,
                   'legacy_strict_success_count':sum(r['reason']=='success' for r in records),
                   'ideal_geometry_candidate_count':sum(bool(r['ever_geometric_candidate']) for r in records),
                   'ideal_geometry_candidate_rate_not_official':sum(bool(r['ever_geometric_candidate']) for r in records)/len(records),
                   'physical_contact_collision_certified_count':None,
                   'official_physical_success_rate':None,
                   'physical_success_not_certified_reason':'需要单独核验PhysX接触、穿透和有效孔内几何；理想包络不可当正式成功',
                   'near_miss_12mm_count':sum(r['max_depth_mm']>=12.0 for r in records),
                   'near_miss_15mm_count':sum(r['max_depth_mm']>=15.0 for r in records),
                   'reached_mouth_region_count':sum(r['min_radial_near_mouth_mm'] is not None for r in records),
                   'aligned_at_mouth_under_1p5mm_count':sum(
                       r['min_radial_near_mouth_mm'] is not None and r['min_radial_near_mouth_mm']<=1.5 for r in records),
                   'last20_depth_stall_under_0p5mm_count':sum(abs(r['last20_depth_progress_mm'])<0.5 for r in records),
                   'limitations':'20秒时限的固定孔位、局部随机起点评估；旧1mm规则仅是严格指标，新增理想圆柱几何候选不等于PhysX物理成功，也不等于完整90%验收'})
    print('[V162_EVAL_RESULT] '+json.dumps(result,ensure_ascii=False),flush=True)
    return result

# =============================================================================
# v16.7: PhysX 动力学重力补偿（仅诊断），绝不直接注入 PPO Actor 观测。
# =============================================================================
def v167_extract_joint_gravity_forces(raw, joint_count):
    """PhysX 返回的重力补偿力矩拆包；严格核对形状避免浮动基座的6维根力混进关节。

    固定基座: [N, D]；自由基座: [N, D+6]，前6列是根的空间力。
    在 Isaac Sim 5.1 不同包装层中该差异曾导致下标偏移，故绝不能直接 [:, arm_ids]。
    最后输出单位N*m，长度D；只供诊断前馈，不暴露给强化学习 Actor。
    """
    import numpy as _np
    a=_np.asarray(raw, dtype=float)
    if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] not in (joint_count,joint_count+6):
        raise RuntimeError(f'PhysX重力力矩形状异常: {a.shape}，期望(1,{joint_count})或(1,{joint_count+6})；拒绝前馈')
    if not _np.isfinite(a).all():
        raise RuntimeError('PhysX重力力矩中存在NaN/Inf，拒绝前馈')
    return a[0,6:].copy() if a.shape[1]==joint_count+6 else a[0].copy()


class FR3V167GravityFeedforwardEnv(FR3LocalPPOEnv):
    """保持原始隐式PD增益和关节力矩上限，仅将关节静态重力广义力作为前馈。

    在 _apply_action 阶段和 set_joint_position_target 并行设置 effort target；
    v16.8用于在正常重力条件下进行可控的PPO试训。前馈仅为底层控制器，绝不进入Actor观测。
    """
    def __init__(self,cfg,render_mode=None,**kwargs):
        self._v167_ff_scale=float(args.gravity_ff_scale)
        self._v167_last_tau_gravity=None
        self._v167_last_tau_command=None
        self._v167_last_clipped_count=0
        self._v167_readback=None
        super().__init__(cfg,render_mode,**kwargs)

    def _v167_gravity_effort(self):
        view=self.robot.root_physx_view
        method=getattr(view,'get_gravity_compensation_forces',None)
        if not callable(method):
            raise RuntimeError('本机PhysX ArticulationView不提供get_gravity_compensation_forces；不得猜测重力力矩')
        raw=method()
        # 兼容torch.Tensor、warp array (numpy())、NumPy数组。显式转CPU以便校验。
        if isinstance(raw,torch.Tensor):
            raw=raw.detach().cpu().numpy()
        elif hasattr(raw,'numpy'):
            raw=raw.numpy()
        forces=v167_extract_joint_gravity_forces(raw,self.robot.num_joints)
        if len(self.robot.joint_names)!=len(forces):
            raise RuntimeError('PhysX DOF个数和Articulation joint_names不一致，拒绝错误索引的前馈')
        # Lab/PhysX关节顺序可能不同。仅当能证明一致才允许力矩真正下发。
        names=getattr(view,'dof_names',None)
        if names is not None:
            normalized=[str(x) for x in names]
            if normalized != [str(x) for x in self.robot.joint_names]:
                raise RuntimeError(f'PhysX/Isaac Lab关节顺序不同，拒绝错误前馈: physx={normalized}, lab={self.robot.joint_names}')
        else:
            # 基于同一仿真ArticulationView，但没有可读名字时必须显式记下不确定性。
            # 固定此程序FR3恰有9关节，臂7+手指2，并检查各维力矩合理。
            if self.robot.num_joints != 9:
                raise RuntimeError('无法核对PhysX关节顺序且DOF不是预期9，拒绝前馈')
        return forces

    def _apply_action(self):
        super()._apply_action()
        if self._v167_ff_scale == 0.0:
            # 显式写0 effort确保ON/OFF只差重力前馈；不遗留上个episode的力矩命令。
            self._v167_last_tau_gravity=[0.0]*self.robot.num_joints
            self._v167_last_tau_command=[0.0]*self.robot.num_joints
            self._v167_last_clipped_count=0
            full=torch.zeros((1,self.robot.num_joints),dtype=torch.float32,device=self.device)
            self.robot.set_joint_effort_target(full)
            return
        tau_g=self._v167_gravity_effort()
        limits=self.robot.data.joint_effort_limits[0].detach().cpu().numpy().astype(float)
        if limits.shape != tau_g.shape or not np.isfinite(limits).all() or np.any(limits<=0):
            raise RuntimeError('力矩上限shape/数值异常，拒绝未限定的前馈')
        # 绝不提高制造商所设的joint effort限制：如第5~7关节12 Nm。
        tau_cmd=np.clip(self._v167_ff_scale*tau_g,-limits,limits)
        self._v167_last_tau_gravity=tau_g.tolist()
        self._v167_last_tau_command=tau_cmd.tolist()
        self._v167_last_clipped_count=int(np.count_nonzero(np.abs(tau_g)>limits+1e-6))
        full=torch.as_tensor(tau_cmd,dtype=torch.float32,device=self.device).reshape(1,-1)
        self.robot.set_joint_effort_target(full)
        # 数据缓冲的读回是指令而非PhysX内部求解的真实关节力矩。
        readback=getattr(self.robot.data,'joint_effort_target',None)
        if readback is None:
            raise RuntimeError('无法读回Lab joint_effort_target，无法确认前馈命令进入指令缓冲')
        r=readback[0].detach().cpu().numpy().astype(float)
        if r.shape != tau_cmd.shape or not np.allclose(r,tau_cmd,atol=1e-4,rtol=0):
            raise RuntimeError(f'关节力矩下发后读回不一致: maxdiff={np.max(np.abs(r-tau_cmd)) if r.shape==tau_cmd.shape else "shape"}')
        self._v167_readback=r.tolist()

# =============================================================================
# v16.9：分离“每个episode起点的随机流”和“episode内观测噪声”的随机消耗。
# 这样不同时间上限即使在第1回合结束步数不同，第N回合仍用【完全相同】随机初始q。
# 保留物理质量、重力、PD、网络、奖励原样；只增加统计和评估的受控随机机制。
# =============================================================================
class FR3V169TimedPPOEnv(FR3V167GravityFeedforwardEnv):
    """在v16.8环境外增加观测轨迹日志，不影响实际Actor输入和物理指令。"""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Isaac Lab超类初始化过程可能自动调用_reset_idx，故字段必须提前建立。
        self._v169_eval_pairing = False
        self._v169_eval_reset_index = 0
        self._v169_episode_traces = []
        self._v169_current_trace = None
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        ready=bool(getattr(self, '_v16_ready', False))
        eval_pair=ready and bool(getattr(self, '_v169_eval_pairing', False))
        eval_seed=None
        if eval_pair:
            # 这里重新建立【当前episode内部】独立RNG；用于局部起点和每步模拟观测噪声。
            # 无论前一回合执行119还是239步，都不影响下一回合的起点。
            eval_seed=int(args.seed)+1000216+1000003*self._v169_eval_reset_index
            self._v16_rng=np.random.default_rng(eval_seed)
            self._v169_eval_reset_index+=1
        super()._reset_idx(env_ids)
        if ready:
            # 在write_joint_state_to_sim后，读取用于定义初态的实际q；
            # 它不交给Actor作为额外信息，仅用于对照归档。
            q=self.robot.data.joint_pos[0,self.arm_ids].detach().cpu().numpy().astype(float)
            tip=fk_task5(self._v16_chain,q,self._v16_root_T,self.tip_link_local_m,
                         self.axis_link_local)[:3]
            self._v169_current_trace={
                'episode':int(self._v16_finished)+1,
                'eval_init_seed':eval_seed,
                'initial_q_rad':q.tolist(),
                'initial_tip_xyz_m':tip.tolist(),
                'initial_radial_mm':float(np.linalg.norm(self._v16_hole[:2]-tip[:2])*1000.),
                'initial_depth_mm':float((self._v16_hole[2]-tip[2])*1000.),
                'samples':[],
            }

    def _get_dones(self):
        terminated,truncated=super()._get_dones()
        if not bool(getattr(self, '_v16_ready', False)):
            return terminated,truncated
        s=self._v16_current
        trace=self._v169_current_trace
        if s is None or trace is None:
            return terminated,truncated
        # v16.10：在真实 PhysX 更新后读取 link8 的棒轴方向，只做【离线几何候选统计】。
        # 仍然使用 v16.9 原来的 <=1mm 成功终止条件，避免在尚未核验孔壁接触前
        # 静默改变策略的执行轨迹、奖励或模型有效性。
        oracle=self.measure_peg_tip_oracle()
        geometry=v1610_geometric_insertion_proxy(s['tip'],oracle['peg_axis_world_xyz'],self._v16_hole)
        # 仅记录刚刚实际执行过的物理状态；step由Isaac Lab计数器给出。
        trace['samples'].append({
            'geometry_candidate':geometry['geometric_candidate'],
            'geometry_min_clearance_mm':geometry['min_ideal_clearance_mm'],
            'geometry_mouth_axis_radial_mm':geometry['mouth_radial_mm'],
            'step':int(self.episode_length_buf[0]),
            'depth_mm':float(s['depth']*1000.),
            'radial_mm':float(s['radial']*1000.),
            'tilt_deg':float(s['tilt_deg']),
            'tip_xyz_m':s['tip'].tolist(),
            'action_xyz_normalized':self._v16_last_action[0].detach().cpu().tolist(),
        })
        if bool((terminated|truncated).item()):
            samples=trace['samples']
            last20=samples[-20:]
            # 进度为时间窗口两端的净变化；正值表示向下更深、径向更接近孔心。
            # 与终态接近不同，时长变更时仍应判断最后阶段是否真的有进展。
            near_mouth=[x for x in samples if -1.0 <= x['depth_mm'] <= 2.0]
            aligned=[x for x in samples if x['radial_mm']<=1.5 and x['tilt_deg']<=1.5]
            extras={
                # 这些两个带条件的辅助指标有时不存在，所以用None而不是伪造0mm。
                'min_radial_near_mouth_mm':min((x['radial_mm'] for x in near_mouth),default=None),
                'max_depth_while_aligned_mm':max((x['depth_mm'] for x in aligned),default=None),
                'min_radial_mm':min(x['radial_mm'] for x in samples),
                'max_depth_mm':max(x['depth_mm'] for x in samples),
                'last20_depth_progress_mm':last20[-1]['depth_mm']-last20[0]['depth_mm'],
                'last20_radial_improvement_mm':last20[0]['radial_mm']-last20[-1]['radial_mm'],
                'initial_radial_mm':trace['initial_radial_mm'],
                'initial_depth_mm':trace['initial_depth_mm'],
                'eval_init_seed':trace['eval_init_seed'],
                'initial_q_rad':trace['initial_q_rad'],
                'max_abs_tilt_deg':max(abs(x['tilt_deg']) for x in samples),
                # 检查实际发生过的每一步，而非用全程最大深度和全程最小偏差拼凑“假成功”。
                'ever_geometric_candidate':any(x['geometry_candidate'] for x in samples),
                'first_geometry_candidate_step':next((x['step'] for x in samples if x['geometry_candidate']),None),
                'best_geometric_clearance_mm':max((x['geometry_min_clearance_mm'] for x in samples
                                                  if x['geometry_min_clearance_mm'] is not None),default=None),
                'geometry_not_collision_certified':True,
            }
            if self._v16_pending_end is None:
                raise RuntimeError('v16.9: 结束回合却无v16.2 pending记录，拒绝丢失统计')
            self._v16_pending_end.update(extras)
            trace.update({'reason':self._v16_pending_end['reason'],
                          'metrics':extras,'steps_actual':len(samples)})
            self._v169_episode_traces.append(trace)
        return terminated,truncated

# ===========================================================================
# v16.3｜无 PPO 更新的重置动力学诊断：不得用于训练成功率
# ===========================================================================
def _v163_snapshot(unwrapped):
    """独立记录 PhysX link8 的棒尖与【当前实际关节角】FK 棒尖。

    此处读取特权物理状态只用于调试，绝不提供给 Actor；diagnostic 模式
    不创建网络，不运行 optimizer.step()。单位：XYZ米，偏差/深度毫米。
    重置后未physics.step时 link8 物理缓存可能仍是旧帧，因此必须同时
    记录FK和PhysX，不能单独凭缓存判断机械臂已经移动或穿模。
    """
    q=unwrapped.robot.data.joint_pos[0,unwrapped.arm_ids].detach().cpu().numpy().astype(float)
    qref=unwrapped._ref[0,unwrapped.arm_ids].detach().cpu().numpy().astype(float)
    target=unwrapped._pd_target[0,unwrapped.arm_ids].detach().cpu().numpy().astype(float)
    fk=fk_task5(unwrapped._v16_chain,q,unwrapped._v16_root_T,
                unwrapped.tip_link_local_m,unwrapped.axis_link_local)
    phys=unwrapped._v16_true_state()
    xyz=np.asarray(phys['tip'],dtype=float)
    return {'q_actual_rad':q.tolist(),
            'max_qref_error_rad':float(np.max(np.abs(q-qref))),
            'max_pd_target_error_rad':float(np.max(np.abs(q-target))),
            'joint_position_target_rad': target.tolist(),
            'joint_reference_rad': qref.tolist(),
            'bias_arm_rad':unwrapped._bias[0,unwrapped.arm_ids].detach().cpu().numpy().astype(float).tolist(),
            'fk_tip_m':fk[:3].tolist(), 'physx_tip_m':xyz.tolist(),
            'fk_physx_error_mm':float(np.linalg.norm(fk[:3]-xyz)*1000),
            'fk_depth_mm':float((unwrapped._v16_hole[2]-fk[2])*1000),
            'physx_depth_mm':float(phys['depth']*1000),
            'radial_mm':float(phys['radial']*1000),
            'tilt_deg':float(phys['tilt_deg'])}


def v167_reset_audit(env,output_dir):
    """每回合显式reset，然后只执行0动作；不更新任何神经网络。

    如果报告显示3个50ms周期下降29mm，要先查FK角度变化/关节驱动，
    而不是直接把30mm算作策略主动插入。探针只使用已有碰撞开启场景。
    """
    u=env.unwrapped
    records=[]; unusual=0
    for episode in range(1,args.audit_episodes+1):
        # 关键改动：每一指定episode有自己的随机流，不受上一回合自动reset次数影响。
        # 因而ON/OFF组的第5~8回合也可以一一配对，避免v16.5/v16.6随机流错位。
        u._v16_rng=np.random.default_rng(int(args.seed)+216+episode*10007)
        env.reset(seed=args.seed)
        before=_v163_snapshot(u)
        before['gravity_ff_scale']=float(args.gravity_ff_scale)
        start=before['fk_tip_m']
        samples=[]; ended=None
        for t in range(1,args.audit_control_steps+1):
            zeros=torch.zeros((1,3),dtype=torch.float32,device=u.device)
            _,reward,terminated,truncated,_=env.step(zeros)
            # done时Isaac Lab可能已经自动重置，当前的物理快照属于下一回合；
            # 这时只记录_get_dones在重置前保存的终态，不假装这是正常跟踪样本。
            if bool((terminated|truncated).item()):
                ended=dict(u._v16_completed_records[-1]) if u._v16_completed_records else {'reason':'unknown_auto_reset'}
                samples.append({'cycle':t,'automatic_reset_occurred':True,
                                'done_before_reset':ended})
                break
            now=_v163_snapshot(u)
            now['gravity_tau_requested_Nm']=u._v167_last_tau_gravity
            now['feedforward_tau_commanded_Nm']=u._v167_last_tau_command
            now['feedforward_joint_clipped_count']=u._v167_last_clipped_count
            now['tau_readback_Nm']=u._v167_readback if args.gravity_ff_scale else [0.0]*u.robot.num_joints
            now['cycle']=t
            now['change_from_initial_fk_mm']=float(np.linalg.norm(np.asarray(now['fk_tip_m'])-start)*1000)
            now['initial_fk_z_change_mm']=float((start[2]-now['fk_tip_m'][2])*1000)
            now['zero_action']=True
            now['anomalies']=[]
            if now['change_from_initial_fk_mm']>5.0:
                now['anomalies'].append('零动作FK位移超过5mm')
            if now['fk_physx_error_mm']>2.0:
                now['anomalies'].append('FK与PhysX棒尖差异超过2mm')
            if now['physx_depth_mm']>0.0:
                now['anomalies'].append('零动作时棒尖进入孔口以下')
            if now['radial_mm']>=12.0 or now['tilt_deg']>6.0:
                now['anomalies'].append('零动作发生横向/姿态越界')
            unusual+=bool(now['anomalies'])
            samples.append(now)
        entry={'episode':episode,'before_first_physics_step':before,
               'samples':samples,'ended':ended}
        records.append(entry)
        # 控制台提供紧凑摘要，完整各关节数据保存在JSON中。
        last=next((p for p in reversed(samples) if 'physx_depth_mm' in p),None)
        print('[V167_RESET_EPISODE] '+json.dumps({
            'episode':episode,'reset_fk_depth_mm':before['fk_depth_mm'],
            'reset_physx_depth_mm':before['physx_depth_mm'],
            'last_depth_mm':last['physx_depth_mm'] if last else None,
            'max_zero_action_fk_shift_mm':max((p.get('change_from_initial_fk_mm',0) for p in samples),default=0),
            'ended':ended,'anomaly_samples':sum(bool(p.get('anomalies')) for p in samples)
        },ensure_ascii=False),flush=True)
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'reset_audit_v16_8.json'
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    first=[next((v for v in x['samples'] if 'change_from_initial_fk_mm' in v),None) for x in records]
    first=[x for x in first if x is not None]
    completed=sum(not any(s.get('automatic_reset_occurred') for s in e['samples']) for e in records)
    auto_resets=len(records)-completed
    def mean(key):
        return sum(float(x[key]) for x in first)/len(first) if first else None
    max_shift=max((float(s['change_from_initial_fk_mm']) for e in records for s in e['samples'] if 'change_from_initial_fk_mm' in s),default=None)
    max_tilt=max((float(s['tilt_deg']) for e in records for s in e['samples'] if 'tilt_deg' in s),default=None)
    # 只有各个重置和全周期均安全，才能称候选补偿通过。
    passed=(len(first)==len(records) and auto_resets==0 and unusual==0 and
            max_shift is not None and max_shift<2.0 and max_tilt is not None and max_tilt<1.0)
    return {'status':'RESET_AUDIT_COMPLETED','quality':'STABLE_CANDIDATE' if passed else 'ANOMALIES_DETECTED',
            'first_step_mean_shift_mm':mean('change_from_initial_fk_mm'),
            'first_step_mean_down_mm':mean('initial_fk_z_change_mm'),
            'max_zero_action_shift_mm':max_shift,
            'max_zero_action_tilt_deg':max_tilt,
            'episodes_without_auto_reset':completed,
            'early_auto_reset_count':auto_resets,
            'episodes':len(records),'zero_action_samples_flagged':unusual,
            'diagnostic_file':path.name,
            'kp':float(args.kp), 'kd':float(args.kd),
            'gravity_ff_scale':float(args.gravity_ff_scale),
            'bias_scale':float(args.bias_scale),
            'gravity_z_m_s2':-9.81,
            'limitations':'仅比较正常重力下的重力补偿力矩前馈，日志是指令读回而非求解器实际力矩；尚未修复TCP惯量警告、训练收敛或多孔/随机孔位要求'}



# ============================================================================
# v16.13：在线真实物理读取、受控四方向动作；不改旧IK、控制器、奖励、PPO权重。
# ============================================================================
class FR3V1613OnlineEvidenceEnv(FR3V169TimedPPOEnv):
    """在真实 PhysX 运动更新后、Isaac Lab 自动reset之前采证。

    DirectRLEnv.step 的概念顺序：pre_physics_step -> 控制子步 -> _get_dones ->
    _get_rewards -> 自动reset。把采证钩子放在 _get_dones 之后才能确保拍到
    原回合刚执行完的一帧，而不是自动reset后的新初态。特权真值仅归档，不进Actor。
    """
    def __init__(self,cfg,render_mode=None,**kwargs):
        self._eno_live_recorder=None
        self._eno_record_max=0
        self._eno_recorded_ticks=0
        self._eno_pending_capture=None
        super().__init__(cfg,render_mode,**kwargs)

    def _get_dones(self):
        terminated,truncated=super()._get_dones()
        recorder=self._eno_live_recorder
        if recorder is None or not getattr(self,'_v16_ready',False):
            return terminated,truncated
        trace=getattr(self,'_v169_current_trace',None)
        ep=int(trace['episode']) if isinstance(trace,dict) and 'episode' in trace else None
        if ep is None or ep>self._eno_record_max:
            return terminated,truncated
        step=int(self.episode_length_buf[0].item())
        act=self._v16_last_action[0].detach().cpu().tolist()
        # 不是离线重建：这些值来自当前正在执行 env.step 的物理Articulation。
        # 实际机器人/棒/孔USD在同一个Sim Stage中；摄像头不能成为Actor观察。
        do_render=bool(args.eno_camera and step % args.eno_capture_every==0)
        # 只在PhysX回调内记录状态，不在 _get_dones() 中调用 renderer 或 Replicator。
        # Replicator的主动采集在外层 step() 返回后触发，避免更新循环重入。
        if do_render and not bool((terminated | truncated).any().item()):
            self._eno_pending_capture=(ep,step)
        extra={'step_readback':step,'pulse_case':getattr(self,'_eno_pulse_case',None),
               'source_after_physics':True,
               'joint7_order':list(self.arm_names),'arm_joint_indices':[int(i) for i in self.arm_ids],
               'joint7_rad_actual':[float(x) for x in self.robot.data.joint_pos[0,self.arm_ids].detach().cpu().tolist()],
               'terminated_on_this_step':bool(terminated[0].item()),
               'truncated_on_this_step':bool(truncated[0].item()),
               'contact_snapshot_cumulative_not_certification': (
                    self.contact_monitor.snapshot() if getattr(self,'contact_monitor',None) is not None else None)}
        recorder.sample_live(robot=self.robot,episode=ep,control_step=step,
                             action_xyz_normalized=act,hole_center_m=self._v16_hole.tolist(),
                             physics_step_completed=True,render_step_completed=False,**extra)
        self._eno_recorded_ticks+=1
        return terminated,truncated

    def step(self, action):
        """外层IsaacLab step执行完毕后采集视频；本函数不更新PPO。

        特别强调：subframe渲染仍可能耗时，加入前后日志便于定位。
        只在非terminal状态抓图，以免把自动reset后的画面误标为上一回合终态。
        """
        self._eno_pending_capture=None
        result=super().step(action)
        info=self._eno_pending_capture
        self._eno_pending_capture=None
        if info is not None and self._eno_live_recorder is not None:
            ep, tick=info
            print(f'[V1613_CAPTURE_START] episode={ep} step={tick} backend=Replicator',flush=True)
            import time as _eno_time
            start=_eno_time.monotonic()
            self._eno_live_recorder.capture_post_step(ep,tick)
            print(f'[V1613_CAPTURE_DONE] episode={ep} step={tick} seconds={_eno_time.monotonic()-start:.2f}',flush=True)
        return result


def v1613_pulse_test(env, out_dir):
    """从配对同初态施加+X/-X/+Y/-Y，检查实际关节+棒尖运动。

    限于初态孔口上方，不尝试进入孔内；每次使用同一eval reset seed。
    数值不是假定动作=位移，而是记录实时PhysX返回的状态。
    """
    u=env.unwrapped
    u._v169_eval_pairing=True
    u._v169_episode_traces.clear()
    experiments=[('x_plus',[args.eno_pulse_magnitude,0.,0.]),
                 ('x_minus',[-args.eno_pulse_magnitude,0.,0.]),
                 ('y_plus',[0.,args.eno_pulse_magnitude,0.]),
                 ('y_minus',[0.,-args.eno_pulse_magnitude,0.])]
    results=[]
    for name,vec in experiments:
        # 指定相同初态随机流; 不依赖前一次运行消费了多少随机数。
        u._v169_eval_reset_index=0
        u._eno_pulse_case=name
        obs,_=env.reset(seed=args.seed)
        q0=u.robot.data.joint_pos[0,u.arm_ids].detach().cpu().numpy().astype(float)
        p0=fk_task5(u._v16_chain,q0,u._v16_root_T,u.tip_link_local_m,u.axis_link_local)[:3].copy()
        start_depth=float(u._v16_true_state()['depth']*1000.)
        # 前两步保持零动作以检查物理稳定和消除重置缓存影响。之后测量基线。
        zero=torch.zeros((1,3),device=u.device)
        for _ in range(2):
            obs,_,term,trunc,_=env.step(zero)
            if bool((term|trunc).any().item()):
                raise RuntimeError('脉冲实验零动作预热竟触发结束，请先检查reset/动力学')
        p1=u._v16_true_state()['tip'].copy()
        q1=u.robot.data.joint_pos[0,u.arm_ids].detach().cpu().numpy().astype(float)
        max_step=0
        action=torch.as_tensor([vec],dtype=torch.float32,device=u.device)
        for k in range(args.eno_pulse_steps):
            obs,_,term,trunc,_=env.step(action)
            max_step=k+1
            if bool((term|trunc).any().item()):
                raise RuntimeError(f'{name}: 脉冲第{k+1}步自动终止；停止实验防止误判')
        p2=u._v16_true_state()['tip'].copy()
        q2=u.robot.data.joint_pos[0,u.arm_ids].detach().cpu().numpy().astype(float)
        results.append({'case':name,'input_action_xyz':vec,
                        'initial_q7_rad':q0.tolist(),'initial_tip_xyz_m_FK_from_reset_q_not_Physx':p0.tolist(),
                        'baseline_tip_xyz_m':p1.tolist(),
                        'final_tip_xyz_m':p2.tolist(),
                        'baseline_drift_xyz_mm':((p1-p0)*1000).tolist(),
                        'actual_pulse_delta_xyz_mm':((p2-p1)*1000).tolist(),
                        'initial_radial_mm':float(np.linalg.norm(u._v16_hole[:2]-p1[:2])*1000),
                        'final_radial_mm':float(np.linalg.norm(u._v16_hole[:2]-p2[:2])*1000),
                        'command_sign_matches_actual_axis':bool((p2-p1)[0 if name.startswith('x') else 1]*
                                      (vec[0] if name.startswith('x') else vec[1])>0),
                        'start_depth_mm':start_depth,
                        'final_depth_mm':float(u._v16_true_state()['depth']*1000.),
                        'q7_after_baseline':q1.tolist(),'q7_after_pulse':q2.tolist(),
                        'physics_steps_pulse':max_step})
        print('[V1613_PULSE] '+json.dumps(results[-1],ensure_ascii=False),flush=True)
    if not np.allclose(results[0]['initial_q7_rad'],results[1]['initial_q7_rad'],atol=0,rtol=0) or not np.allclose(results[2]['initial_q7_rad'],results[3]['initial_q7_rad'],atol=0,rtol=0):
        raise RuntimeError('脉冲实验配对初态不一致；拒绝错误归因')
    report={'version':VERSION,'status':'PULSE_COMPLETED','not_policy_training':True,
            'paired_seed':int(args.seed)+1000216,
            'all_pairs_share_identical_initial_q':True,
            'pulse_control_steps':int(args.eno_pulse_steps),
            'action_magnitude':args.eno_pulse_magnitude,
            'all_axis_signs_match':all(r['command_sign_matches_actual_axis'] for r in results),
            'results':results,'physical_success_rate':None,
            'note':'小幅无接触方向测试，不能声称插孔成功；实际位移必须结合预热漂移与初态几何解释'}
    (out_dir/'pulse_v16_13.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report

def main():
    # 保护既有实验：直接新建训练目录，避免覆盖之前已存在的模型。
    if args.mode == 'train' and (args.output_dir/'ppo_v16_2_latest.pt').exists():
        raise RuntimeError('输出目录已有checkpoint；请使用新的输出目录并显式--resume，不允许覆盖旧PPO文件')
    # 新版本强制独立目录：防止误覆盖旧评估报告及录像。
    if args.output_dir.exists():
        raise RuntimeError('v16.13证据目录已存在，请修改--output-dir；不会覆盖历史数据')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'version':VERSION,'parent_source':'43_phase05b2b_v16_10_official20s_eval.py hotfix1', 'purpose':'ONLINE_PULSE_AND_RGB_EVIDENCE_NOT_TRAINING', 'status':'NOT_STARTED','seed':args.seed,'mode':args.mode,
            'diagnostic_bias_scale':float(args.bias_scale),
            'gravity_z_m_s2':-9.81, 'kp':float(args.kp), 'kd':float(args.kd),
            'gravity_ff_scale':float(args.gravity_ff_scale),
            'max_episode_steps_configured':int(args.max_episode_steps),
            'max_episode_seconds_configured':float(args.max_episode_steps*CTRL_DT),
            'diagnostic_only_or_pilot':'OFFICIAL_TIME_20S_BUT_LOCAL_PROXY_NOT_FINAL_90PCT',
            'limits':{'hole_fixed':True,'mount_perturbation_per_episode':False,'num_envs':1,
                      'actor_sees_exact_pose':False,'observation_noise_mm':args.obs_noise_mm,
                      'reset_xy_mm':args.reset_xy_mm,'resume':str(args.resume) if args.resume else None,
                      'success_depth_mm':V16_SUCCESS_DEPTH_M*1000,
                      'success_radial_mm':V16_SUCCESS_RADIAL_M*1000,
                      'episode_max_steps':int(args.max_episode_steps),
                      'episode_duration_s':float(args.max_episode_steps*CTRL_DT),
                      'legacy_strict_radial_mm':1.0,
                      'ideal_geometric_radial_clearance_mm':1.5,
                      'official_physical_success_unverified':True},
            'cuda_preflight':None,'result':None,'error':None}
    env=None
    recorder=None
    try:
        cfg=FR3LocalTrainCfg();cfg.scene.num_envs=1;cfg.seed=args.seed
        cfg.sim.device=args.device or 'cuda:0'
        # v16.8: 正常重力(-9.81)和已验证的原始400/40隐式PD不变。
        # 训练/评估时必须施加静态重力补偿；这是环境底层控制器而非Agent动作。
        cfg.sim.gravity=(0.,0.,-9.81)
        # v16.9: Isaac Lab的真实时间上限来自cfg.episode_length_s，而不是独立的for循环。
        # 必须在gym.make之前设置。v16.10已修正旧版的-1提前截止，20秒上限对应第400个控制周期。
        cfg.episode_length_s = args.max_episode_steps * CTRL_DT
        cfg.robot_cfg.actuators['all_joints'].stiffness=float(args.kp)
        cfg.robot_cfg.actuators['all_joints'].damping=float(args.kd)
        report['cuda_preflight']=cuda_pytorch_preflight(cfg.sim.device)
        print('[V16_CUDA_PREFLIGHT] '+json.dumps(report['cuda_preflight'],ensure_ascii=False),flush=True)
        name='Eno-FR3-Live-Evidence-v1613'
        gym.register(id=name,entry_point=f'{__name__}:FR3V1613OnlineEvidenceEnv',disable_env_checker=True)
        env=gym.make(name,cfg=cfg)
        u=env.unwrapped
        report['max_episode_steps_physics_readback']=int(u.max_episode_length)
        if abs(int(u.max_episode_length)-int(args.max_episode_steps))>1:
            raise RuntimeError(f'IsaacLab环境真实回合上限未生效: {u.max_episode_length} vs {args.max_episode_steps}')
        # 关键验证：不能只看config里宣称的刚度；必须核实Isaac Lab/PhysX读取到的实际值。
        # ArticulationData的关节刚度/阻尼为solver读数，单位分别Nm/rad、Nm*s/rad。
        stiffness=u.robot.data.joint_stiffness[0,u.arm_ids].detach().cpu().numpy().astype(float)
        damping=u.robot.data.joint_damping[0,u.arm_ids].detach().cpu().numpy().astype(float)
        report['actual_drive_gains']={
            'kp_arm':stiffness.tolist(), 'kd_arm':damping.tolist(),
            'kp_match':bool(np.allclose(stiffness,args.kp,rtol=0,atol=1e-4)),
            'kd_match':bool(np.allclose(damping,args.kd,rtol=0,atol=1e-4)),
        }
        effort=getattr(u.robot.data,'joint_effort_limits',None)
        if effort is not None:
            report['actual_drive_gains']['effort_limit_arm_Nm']=effort[0,u.arm_ids].detach().cpu().numpy().astype(float).tolist()
        else:
            report['actual_drive_gains']['effort_limit_arm_Nm']=None
            report['actual_drive_gains']['effort_limit_note']='当前ArticulationData未公开joint_effort_limits，需要额外运行时接口核查'
        if not report['actual_drive_gains']['kp_match'] or not report['actual_drive_gains']['kd_match']:
            raise RuntimeError('实际关节增益与预注册配置不一致；拒绝把此组当有效对照')
        print('[V167_ACTUATOR_GAINS] '+json.dumps(report['actual_drive_gains'],ensure_ascii=False),flush=True)
        env.reset(seed=args.seed)
        zeros=torch.zeros((1,3),device=u.device)
        for _ in range(3):
            env.step(zeros)
        u.prepare_v16()
        obs,_=env.reset(seed=args.seed)
        if args.mode=='reset-audit':
            # 本版仅允许eval，分支不可达。
            report['result']=v167_reset_audit(env,args.output_dir)
            report['status']=report['result']['status']
            report['scene_checks']=u.usd_checks
            print('[V167_RESULT] '+json.dumps(report['result'],ensure_ascii=False),flush=True)
            return 0
        # 第一次真实physics step后检查：此前v12已证明离线FK与PhysX在此FR3模型应达毫米以下精度。
        env.step(zeros)
        phys=u._v16_true_state()
        expected=fk_task5(u._v16_chain,u.robot.data.joint_pos[0,u.arm_ids].detach().cpu().numpy(),
                          u._v16_root_T,u.tip_link_local_m,u.axis_link_local)[:3]
        physical_fk_mm=float(np.linalg.norm(expected-phys['tip'])*1000.)
        # 保留真实测量值和各检查布尔量。旧版把三种错误合并在一条信息中，
        # 遇到深度符号错误时无法定位。这里先写入报告，再独立拒绝不安全状态。
        reasons=[]
        if not math.isfinite(physical_fk_mm) or physical_fk_mm>2.0:
            reasons.append('FK与PhysX偏差超2mm')
        if not math.isfinite(phys['depth']) or phys['depth']>0.0:
            reasons.append('起始棒尖已位于孔口以下（插入深度>0mm）')
        if not math.isfinite(phys['radial']) or phys['radial']>0.006:
            reasons.append('起始径向偏移超6mm')
        if not math.isfinite(phys['tilt_deg']) or phys['tilt_deg']>3.0:
            reasons.append('起始棒轴倾角超3度')
        report['initial_reset_check']={
            'fk_error_mm':physical_fk_mm,
            'depth_mm':float(phys['depth']*1000.),
            'tip_clearance_above_mouth_mm':float(-phys['depth']*1000.),
            'radial_mm':float(phys['radial']*1000.),
            'tilt_deg':float(phys['tilt_deg']),
            'safe_to_train':len(reasons)==0,
            'rejection_reasons':reasons,
        }
        print('[V16_INITIAL_RESET_CHECK] '+json.dumps(report['initial_reset_check'],ensure_ascii=False),flush=True)
        if reasons:
            raise RuntimeError('v16初始状态安全检查失败：'+'；'.join(reasons))
        report['fk_vs_physx_mm']=physical_fk_mm
        report['scene_checks']=u.usd_checks
        # Hotfix 5 独立重渲染要求不在物理进程启动相机。
        # 为避免回到 Hotfix4 的同步采图死锁，严格拒绝两种模式同时启用。
        if os.environ.get('ENO_V1613_RECORD_REPLAY')=='1' and args.eno_camera:
            raise RuntimeError('Hotfix5: 物理状态记录与在线相机模式不能同时启用')
        recorder=LiveEvidence(out_dir=args.output_dir/'live',
                              record_every=1,capture_every=args.eno_capture_every,
                              robot_prim_path=ROBOT_ROOT,peg_prim_path=PEG_PATH,
                              hole_wall_prefix=WALL_PATH+'/segment_',
                              tool_tip_local_link8_m=u.tip_link_local_m)
        scene_audit=recorder.inspect_stage()
        report['current_live_stage_audit']=scene_audit
        if recorder.errors:
            raise RuntimeError('在线USD检查失败：'+'; '.join(recorder.errors))
        if args.eno_camera:
            print('[V1613_SETUP_CAMERA_BEGIN]',flush=True)
            recorder.setup_camera(u._v16_hole.tolist())
            # 不在主程序启动阶段执行同步render预热：此前Replicator版本在此处疑似停滞。
            # 真正抓帧移至环境外层step之后。禁止在_get_dones回调里重入仿真渲染。
            print('[V1613_SETUP_CAMERA_END] prewarm_render_skipped=true',flush=True)
        u._eno_live_recorder=recorder
        u._eno_record_max=args.eno_record_episodes
        report['online_capture_enabled']=bool(args.eno_camera)
        if args.mode=='pulse':
            report['result']=v1613_pulse_test(env,args.output_dir)
        elif args.mode=='reset-audit':
            report['result']=v167_reset_audit(env,args.output_dir)
        elif args.mode=='smoke':
            # 真实仿真步进和奖励调用检查，不改变策略网络权重。
            obs,_=env.reset(seed=args.seed)
            total_reward=0.;done_count=0
            for _ in range(16):
                act=torch.as_tensor([[0.,0.,-0.5]],device=u.device)
                obs,reward,terminated,truncated,_=env.step(act)
                total_reward+=float(reward.item())
                done_count+=int((terminated|truncated).item())
                if not bool(torch.isfinite(obs['policy']).all().item()):
                    raise RuntimeError('冒烟测试观测出现NaN/Inf')
            report['result']={'status':'SMOKE_PASSED','steps':16,'reward_sum':total_reward,
                              'observation_dim':int(obs['policy'].shape[-1]),'done_count':done_count}
        else:
            network=ActorCritic().to(u.device)
            report['result']=evaluate_loop(env,network,args.output_dir)
        report['status']=report['result']['status']
        report['physics_simulation_steps_captured_online']=int(u._eno_recorded_ticks)
        if recorder is not None and recorder.errors:
            report['online_evidence_warnings']=list(recorder.errors)
        if args.eno_camera and (recorder is None or recorder.frame_counter<2):
            raise RuntimeError('启用了在线相机，却没有保存至少2帧有效画面；不得宣称完成可视化证据')
        print('[V16_RESULT] '+json.dumps(report['result'],ensure_ascii=False),flush=True)
    except Exception as exc:
        report['status']='ERROR'
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
        print('[V167_ERROR] '+report['error'],flush=True)
    finally:
        if recorder is not None:
            try:
                live_summary=recorder.close()
                report['online_evidence_summary_file']='live/summary_v16_13_live.json'
                report['online_evidence_level']=live_summary.get('evidence_level')
                report['rendered_png_count']=live_summary.get('isaac_sim_rgb_frames_saved',0)
                if args.eno_camera and report['rendered_png_count']<2:
                    report['status']='ERROR'
                    report['error']=report.get('error') or '录制失败：少于2帧Isaac Sim RGB，不把离线图片冒充录像'
            except Exception as exc:
                report['status']='ERROR';report['error']=f'证据记录器关闭失败: {exc}'
        try:
            (args.output_dir/'summary_v16_13.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        except Exception as exc:
            print('[V167_SAVE_ERROR] '+str(exc),flush=True)
        if env is not None:
            try:env.close()
            except Exception:pass
        try:simulation_app.close()
        except Exception:pass
    return 0 if report['status'] in ('EVALUATION_COMPLETED','PULSE_COMPLETED') else 1


if __name__=='__main__':
    raise SystemExit(main())
