#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17.4 生成器：在 v17.3.1 原有专家/传感器/BC 工程上，修复真实碰撞体随机化。

根因：v17.3.1 reset() 更新了 PhysX Hole 根刚体的 root pose，但孔壁 scene query
仍留在 Stage 初始创建的位置。把“RigidObject pose 读回”当成“孔壁真移动”是不成立的。

临时且工程上可行的方案：每次独立 Isaac Sim 进程在 PhysX 启动**之前**一次性
采样 10x10cm 孔位 + ±5mm 实体安装扰动，并在那个位置创建整个孔的碰撞体。
运行期间不再要求复合孔壁移动。每个新进程就是一个新随机训练/示范 episode。
这牺牲了并行效率，但修复了场景物理一致性，且可以直接复用 v17.3 的非特权观测。

用户须知：这是修复候选，只有 DGX 现场射线审核与实际运动通过后才接受为数据。
本文件仅做离线源代码生成；永不覆盖 v16 原件。注释使用中文说明物理意义、约束和单位。
"""
from pathlib import Path
import ast
import argparse
import py_compile
import tempfile

from prepare_v17_3_coarse import build as build_v173


def once(code: str, old: str, new: str, purpose: str) -> str:
    found = code.count(old)
    if found != 1:
        raise RuntimeError(f"{purpose}: 模板锚点应恰好出现1次，实际{found}次；停止生成而不是冒险修改错误代码")
    return code.replace(old, new, 1)


def build(source: Path, target: Path, teacher_ext: Path, eval_ext: Path):
    if target.exists():
        raise FileExistsError(f"禁止覆盖已生成源码: {target}")
    with tempfile.TemporaryDirectory(prefix='v17_4_') as d:
        stage = Path(d) / 'base_v1731.py'
        build_v173(source, stage, teacher_ext, eval_ext)
        code = stage.read_text(encoding='utf-8')

    # 在旧装配创建“真实碰撞壁”之前随机化孔中心。
    # 这样USD Hole Xform、子级32块convexCollider以及初始PhysX几何来自相同位置。
    # 与上一版 reset() 仅写 kinematic root pose 完全不同。
    anchor = "        # 【物理安装真实扰动】：在make_peg_on_link8创建棒与collider【之前】修改安装参数。"
    start = '''        # v17.4【真实物理随机单孔】：每新仿真进程仅采样一次，单位为米。
        # 与教师的实际评估目标使用同一个 episode_truth，必须先于 create_hole_in_env()。
        # 注意不能把此真值输入学生策略：这里只用于创建真实的环境资产。
        center_xy=np.asarray(V16_HOLE_XY_M,dtype=np.float64)
        spawn_offset_xy=self._v17_rng.uniform(-.05,.05,size=2)  # 中心±50mm=10x10cm
        spawn_xyz=np.asarray([center_xy[0]+spawn_offset_xy[0],
                              center_xy[1]+spawn_offset_xy[1],
                              float(geom.HOLE_BASE_WORLD[2])],dtype=np.float64)
        self.episode_truth['hole_center_world_m']=spawn_xyz.tolist()
        self._v17_spawn_hole_world=spawn_xyz.copy()
        self._v17_last_target=spawn_xyz.copy()
        print('[V17_4_HOLE_SPAWN_BEFORE_PHYSX] '+json.dumps({
            'hole_world_m':spawn_xyz.tolist(),
            'spawn_offset_xy_mm':(spawn_offset_xy*1000).tolist(),
            'mode':'per_process_real_collider_spawn_NOT_per_reset',
        },ensure_ascii=False),flush=True)
'''
    code = once(code, anchor, start+anchor, '注入启动前随机真实孔壁')

    # 旧 reset 随机移动复合刚体的行为被禁用：根 pose 虽能更新，子碰撞仍在旧位置。
    # 由于一轨一进程采集，按“进程随机”仍可产生真实随机孔位轨迹。
    a = code.index("    def _reset_idx(self, env_ids):\n        # 清理机器人状态并调用Isaac Lab原始reset。")
    b = code.index("\ndef v17_scene_smoke():", a)
    new_reset = '''    def _reset_idx(self, env_ids):
        """v17.4 每回合仅重置机械臂和【同一个已建模孔位】，绝不虚构碰撞体瞬移。

        重要：现在每个 Isaac Sim 进程只支持一条随机孔专家轨迹。若想同进程多回合
        重新随机孔位，必须重新做真实可移动碰撞组件并独立通过查询验证。
        """
        super()._reset_idx(env_ids)  # 重置FR3关节、控制器引用等
        if not hasattr(self,'hole') or not self.hole.is_initialized:
            return  # 创建场景的内部提前reset无需处理
        if env_ids is None:
            env_ids=self.robot._ALL_INDICES
        if len(env_ids)!=1 or int(env_ids[0])!=0:
            raise RuntimeError('v17.4当前只支持单环境，不能伪装成多环境并行训练')
        target=self._v17_spawn_hole_world.copy()  # 本进程内固定，和真实碰撞几何一致
        pose=self.hole.data.default_root_state[env_ids,:7].clone()
        pose[0,:3]=torch.as_tensor(target,dtype=pose.dtype,device=pose.device)
        pose[0,3:]=torch.as_tensor([1.,0.,0.,0.],dtype=pose.dtype,device=pose.device)
        self.hole.write_root_pose_to_sim(pose,env_ids)
        self.hole.write_root_velocity_to_sim(
            torch.zeros((1,6),dtype=pose.dtype,device=pose.device),env_ids)
        self._v17_last_target=target
        self.episode_truth['hole_center_world_m']=target.tolist()
        self._v17_reset_count+=1
'''
    code = code[:a] + new_reset + code[b:]

    # 报告必须正确：孔位不是“每次 env.reset 随机”而是“每次新进程随机”。
    code=once(code,"'per_episode_hole_randomized': True", "'per_episode_hole_randomized': False", '场景随机化粒度')
    code=once(code,"'hole_randomization_granularity':'each_episode_physx_rigidbody'", "'hole_randomization_granularity':'once_per_new_isaac_sim_process_real_colliders'", '专家摘要随机化粒度')

    # v17.3.1已有射线审计，但初始与reset位置将重合；增加**远场负对照**。
    # 避免“场景中的孔其实固定在另一处，但有射线打到别的物体”造成虚假通过。
    anchor = "    depth_quality=depth_sensor.calibrate_once(nominal_world,nominal_tip_local,link8_rot,q0)"
    gate = '''    # v17.4【物理碰撞壁定位审计】只用于离线验收，不参与学生策略输入。
    # 在实际生成的孔心 +20mm圆环上应命中孔口；距孔心150mm应无孔口命中。
    # 这样验证的是真实PhysX射线查询到的碰撞表面，不是只读取RigidObject root pose。
    near_hits=_v17_3_audit_top_ring(hole[:2])
    far_xy=np.asarray(hole[:2],dtype=np.float64)+np.array([.150,0.],dtype=np.float64)
    far_hits=_v17_3_audit_top_ring(far_xy)
    position_audit={'collider_spawn_xy_m':hole[:2].tolist(),
                    'near_ring_hits_of_16':int(near_hits),
                    'far_ring_hits_of_16':int(far_hits),
                    'far_probe_dx_mm':150.,
                    'criterion':'near>=12 and far==0',
                    'passed':bool(near_hits>=12 and far_hits==0)}
    print('[V17_4_COLLIDER_SPAWN_AUDIT] '+json.dumps(position_audit,ensure_ascii=False),flush=True)
    if not position_audit['passed']:
        raise RuntimeError('V17_4_PHYSICS_COLLIDER_FAIL: 启动前创建的随机孔壁位置异常，禁止采集BC轨迹')
'''
    code=once(code,anchor,gate+anchor,'为教师加入孔壁原位+负对照审计')
    # 两条独立证据：仿真真值读取只用于审计，测量值只能来自raycast。
    code=once(code,
        "    depth_quality['offline_hole_error_mm']=hole_measure_error_mm",
        "    depth_quality['offline_hole_error_mm']=hole_measure_error_mm\n"
        "    depth_quality['v17_4_collider_audit']=position_audit",'记录孔壁审计')

    # 新采集轨迹带有不可伪造的质量元信息；旧v17.1/3.1轨迹即使有同schema也不得混入。
    code=once(code,
        "        'student_observation_schema':FEATURE_SCHEMA,",
        "        'student_observation_schema':FEATURE_SCHEMA,\n"
        "        'v17_4_spawn_collider_audit':position_audit,",
        '将物理位置核验写进专家summary')

    # 除教师外，在线学生评估必须独立进行质量检查，防止网络使用错误射线孔位。
    anchor="        quality=sensor.calibrate_once(f0[:3],nominal_local,rot0,q0)"
    gate_eval='''        # 仅用于评估开始前的诊断/拒收，禁止把这个真值误差传给策略网络。
        measured_center=np.asarray(quality['hole_detected_xy_m'],dtype=float)
        _offline_err=float(np.linalg.norm(measured_center-np.asarray(u._v16_hole[:2]))*1000.)
        if _offline_err>5.:
            raise RuntimeError(f'V17_4_EVAL_BAD_SENSOR: measured hole error {_offline_err:.3f}mm (>5)')
'''
    code=once(code,anchor,anchor+'\n'+gate_eval.rstrip('\n'),'学生在线评估质量门禁')

    # 当前 v17_scene_smoke 曾把根位姿读回视作物理位移成功；停用旧模式，防止混淆。
    code=once(code,"    return v17_scene_smoke()",
              "    raise RuntimeError('v17.4已禁用旧v17_scene物理移动冒烟测试：此版为每进程真实孔位生成')",'停用失效根位姿测试')

    ast.parse(code,filename=str(target))
    target.write_text(code,encoding='utf8')
    py_compile.compile(str(target),doraise=True)
    print('[V17_4_SPAWN_CONSISTENT_SOURCE_GENERATED] '+str(target),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='在v17.3.1基础上生成可验收的每进程真实随机孔物理场景')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--teacher-extension',type=Path,required=True)
    p.add_argument('--eval-extension',type=Path,required=True)
    a=p.parse_args()
    build(a.source,a.target,a.teacher_extension,a.eval_extension)
