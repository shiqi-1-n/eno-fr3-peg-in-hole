#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v26：将合法视觉估计接入残差 PPO 的状态与底层 IK。

本模块最重要的安全边界：
- 相机RGB-D+本体关节/FK生成的**安装估计**可以用于控制；
- 仿真初始化时记录的真实tip_link_local_m / axis_link_local 只供
  场景物理、特权奖励和成功判定；决不能进入学生的IK计算。
- 在调用原来的 `_pre_physics_step` 时，短暂将工具参数切成学生估计，
  调用返回立即恢复原值。这避免重写关节PD+IK并保留reward的真值语义。
  这个动态隔离必须继续在DGX的真实执行轨迹里检查，不可替代代码审计。
"""
from __future__ import annotations
import numpy as np

class V26VisualUnavailable(RuntimeError):
    """摄像头观察或估计失效，拒绝将此轨迹提供给PPO更新。"""


def _rot_from_wxyz(wxyz):
    """机器人本体link8四元数转世界旋转；这是关节/本体信息，非随机棒真值。"""
    a=np.asarray(wxyz,dtype=float).reshape(4)
    a=a/max(1e-12,float(np.linalg.norm(a)))
    w,x,y,z=a
    return np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=float)


def v26_latest_visual(u, *, strict=True):
    """利用最近一次RGB-D安装估计和**当前**本体FK得到20Hz相对位姿。

    注意这里不是简单重复使用上一帧的世界坐标：机械臂每个控制步都在动，
    必须传播法兰本体位置和旋转。世界孔中心来自先前RGB-D跟踪记录，
    不允许来自 `u._v16_hole` / `episode_truth` / `measure_peg_tip_oracle`。
    """
    v=getattr(u,'_v26_latest_visual',None)
    if v is None:
        if strict:raise V26VisualUnavailable('还没有来自RGB-D的合法棒孔相对位姿')
        return None,None
    # 这些变量由相机跟踪器处理，输入只来自图像/深度及合法本体关节状态。
    state=getattr(u,'_v23_visual_state',None)
    mount=getattr(state,'mount',None)
    if mount is None or mount.estimate_params is None:
        if strict:raise V26VisualUnavailable('尚无RGB-D估计的工具安装位姿')
        return None,None
    if mount.last_observed_step is None:
        if strict: raise V26VisualUnavailable('没有任何真实RGB-D棒安装拟合，禁止使用名义/真值冒充')
        return None,None
    now=int(getattr(u,'_v21_step_count',0))
    age=now-int(mount.last_observed_step)
    if age<0:raise V26VisualUnavailable('时间戳倒退：疑似reset后复用上次标定')
    locked=mount.locked_mount_params is not None
    # 首次锁定之前仍限制过旧RGB-D测量；锁定后本次episode中刚性安装常量
    # 不会因入孔遮挡而过期。绝不通过把60改成600、或者读仿真真值实现兜底。
    if age>60 and not locked:
        if strict:raise V26VisualUnavailable(f'插入前视觉棒安装标定过期{age}步，未完成可信锁定')
        return None,None
    q=dict(v['quality'])
    hole=q.get('hole') or {}
    hole_xy=np.asarray(hole.get('center_xy_m',[]),dtype=float)
    if locked:
        hole_xy=np.asarray(mount.locked_hole_xy,dtype=float)
    if hole_xy.shape!=(2,) or not np.isfinite(hole_xy).all():
        raise V26VisualUnavailable('视觉跟踪没有合法孔中心')
    link=u.link8_ids[0]
    pos=u.robot.data.body_link_pos_w[0,link].detach().cpu().numpy()
    quat=u.robot.data.body_link_quat_w[0,link].detach().cpu().numpy()
    rot=_rot_from_wxyz(quat)
    params_used=(mount.locked_mount_params if locked else mount.estimate_params)
    local_tip,local_axis=mount._model(params_used)
    tip=pos+rot@local_tip
    axis=rot@local_axis
    axis=axis/np.linalg.norm(axis)
    hole_z=float(state.mouth_z) # 固定桌面几何标定常数，不是随机孔的XY真值
    rel=np.r_[np.asarray([hole_xy[0],hole_xy[1],hole_z])-tip,axis[:2]]
    if rel.shape!=(5,) or not np.isfinite(rel).all() or np.linalg.norm(rel[3:5])>.5:
        raise V26VisualUnavailable('20Hz RGB-D+FK五维估计非有限或轴向无效')
    if q.get('privileged_mount_pose_used') is not False:
        raise V26VisualUnavailable('视觉模块未明确排除真实安装位姿')
    # 只有真正的RGB-D联合拟合近孔口时才能锁定，不能基于上一帧的FK传播结果锁定。
    # 这些判断全部使用学生可得的视觉相对位置及历史；不使用offline truth。
    if not locked:
        real_frame=(age<=5 and bool(q.get('raw_peg_visible_this_step'))
                    and q.get('sensor_estimate_status')=='fresh_rgbd_plus_fk')
        enough_history=len(mount.history)>=3
        aligned=(float(np.linalg.norm(rel[:2]))<=.0012 and
                 -.008<=float(rel[2])<=.010 and
                 float(np.linalg.norm(rel[3:5]))<=np.sin(np.deg2rad(1.0)))
        # v31.6: 双视角可能在接近孔口过程中被遮挡。
        # 区分两个概念：①先冻结本次刚性安装标定；②允许真正下插。
        # 这里提前做的只有①，RL阶段的径向/姿态/跟踪安全门保持原样。
        # 仅使用RGB-D历史、拟合质量以及法兰FK和静态孔视觉测量。
        # mount.history记录真实视觉拟合使用的每帧相机点云，不是仿真真值。
        _v316_prelock = False
        _v316_state = {}
        if __import__('os').environ.get('ENO_V316_PRELOCK', '1') == '1':
            _v316_cam_last = {}
            for _v316_t, _v316_groups in mount.history:
                for _v316_name, _v316_pts in _v316_groups:
                    if _v316_name in ('peg_front', 'peg_left') and len(_v316_pts) >= 25:
                        _v316_cam_last[_v316_name] = max(
                            _v316_cam_last.get(_v316_name, -1000000), int(_v316_t))
            _v316_two_real_views = all(
                now-_v316_cam_last.get(_v316_name, -1000000) <= 35
                for _v316_name in ('peg_front', 'peg_left'))
            _v316_cand = getattr(mount, '_v314_pending_params', None)
            _v316_cand_step = getattr(mount, '_v314_pending_at_step', None)
            _v316_quality = getattr(mount, '_v314_pending_quality', None) or {}
            _v316_cand_good = False
            _v316_shift_mm = None
            _v316_axis_deg = None
            if _v316_cand is not None and _v316_cand_step is not None:
                _v316_cp = np.asarray(_v316_cand, dtype=float)
                if (_v316_cp.shape == (4,) and np.isfinite(_v316_cp).all() and
                        0 <= now-int(_v316_cand_step) <= 5):
                    _v316_old_tip, _v316_old_ax = mount._model(mount.estimate_params)
                    _v316_new_tip, _v316_new_ax = mount._model(_v316_cp)
                    _v316_shift_mm = float(np.linalg.norm(_v316_new_tip-_v316_old_tip)*1000.)
                    _v316_axis_deg = float(np.degrees(np.arccos(
                        np.clip(float(_v316_new_ax@_v316_old_ax), -1., 1.))))
                    _v316_med = _v316_quality.get('mount_surface_median_mm')
                    _v316_cand_good = (
                        _v316_med is not None and np.isfinite(float(_v316_med))
                        and float(_v316_med) <= .30
                        and _v316_shift_mm <= 4. and _v316_axis_deg <= 4.)
            # 至少三次不同RGB-D采样时刻的候选在法兰局部坐标系保持稳定。
            # 注意：只记录学生可得的候选，禁止使用offline真实棒轴验证。
            _v316_trace = list(getattr(mount, '_v316_candidate_trace', []))
            if (_v316_cand_good and real_frame and _v316_cand_step == now and
                    (not _v316_trace or int(_v316_trace[-1][0]) < now)):
                _v316_trace.append((int(now), np.asarray(_v316_cand, dtype=float).copy()))
                _v316_trace = _v316_trace[-8:]
                mount._v316_candidate_trace = _v316_trace
            _v316_stable = False
            _v316_stable_tip_mm = None
            _v316_stable_axis_deg = None
            if (len(_v316_trace) >= 3 and
                    0 <= now-int(_v316_trace[-3][0]) <= 15):
                _v316_last_three = [_v316_p for _, _v316_p in _v316_trace[-3:]]
                _v316_tips = [mount._model(_v316_p)[0] for _v316_p in _v316_last_three]
                _v316_axes = [mount._model(_v316_p)[1] for _v316_p in _v316_last_three]
                _v316_stable_tip_mm = max(float(np.linalg.norm(
                    _v316_tt-_v316_tips[-1])*1000.) for _v316_tt in _v316_tips)
                _v316_stable_axis_deg = max(float(np.degrees(np.arccos(np.clip(
                    float(_v316_aa@_v316_axes[-1]), -1., 1.)))) for _v316_aa in _v316_axes)
                _v316_stable = (_v316_stable_tip_mm <= 1.5 and
                                 _v316_stable_axis_deg <= .6)
            _v316_radial_mm = float(np.linalg.norm(rel[:2])*1000.)
            _v316_depth_mm = float(rel[2]*1000.)
            # v31.6.1：这是「固定刚性工具标定」，不是「判断已对准孔口」。
            # XY偏差可为数毫米，不能用 v31.6 的 <=4mm 对中条件拒绝稳定的标定。
            # 但只能在孔口上方8~12mm的安全预接近高度锁定，不能在临孔时突然调姿。
            # 仍需要真正的双侧视观测、近期/稳定的拟合和世界棒轴合理性。
            # 注意：完全不修改 RL 下插门控、成功判定或仿真接触模型。
            _v316_axis_world_deg = float(np.degrees(np.arcsin(np.clip(
                float(np.linalg.norm(rel[3:5])), 0., 1.))))
            _v316_height_ok = bool(-12.0 <= _v316_depth_mm <= -8.0)
            _v316_axis_ok = bool(_v316_axis_world_deg <= 8.0)
            _v316_prelock = bool(
                real_frame and len(mount.history) >= 5 and _v316_two_real_views
                and _v316_cand_good and _v316_stable and _v316_height_ok
                and _v316_axis_ok)
            _v316_state = {
                'step': now, 'radial_mm': round(_v316_radial_mm, 3),
                'depth_mm': round(_v316_depth_mm, 3),
                'front_last': _v316_cam_last.get('peg_front'),
                'left_last': _v316_cam_last.get('peg_left'),
                'candidate_step': _v316_cand_step,
                'candidate_med_mm': _v316_quality.get('mount_surface_median_mm'),
                'candidate_tip_delta_mm': _v316_shift_mm,
                'candidate_axis_delta_deg': _v316_axis_deg,
                'stable_fit_last3': _v316_stable,
                'stable_tip_spread_mm': _v316_stable_tip_mm,
                'stable_axis_spread_deg': _v316_stable_axis_deg,
                'raw_fresh': bool(real_frame),
                'two_recent_side_views': bool(_v316_two_real_views),
                'history_sufficient': bool(len(mount.history)>=5),
                'candidate_good': bool(_v316_cand_good),
                'height_safe_8_to_12mm': _v316_height_ok,
                'world_axis_deg': round(_v316_axis_world_deg, 4),
                'world_axis_ok': _v316_axis_ok,
                'xy_alignment_not_required_for_calibration': True,
                'prelock_ready': _v316_prelock,
                'truth_used': False}
            # 每25步输出一次诊断；即使早锁条件不满足也方便辨别原因。
            if now%25 == 0 and -20. <= _v316_depth_mm <= 2.:
                print('[V316_PRELOCK_GATE] '+__import__('json').dumps(
                    _v316_state, ensure_ascii=False), flush=True)
        if real_frame and enough_history and (aligned or _v316_prelock):
            if _v316_prelock and not aligned:
                print('[V316_EARLY_CALIBRATION_LOCK] '+__import__('json').dumps(
                    _v316_state, ensure_ascii=False), flush=True)
            mount.locked_mount_params=np.asarray(mount.estimate_params,dtype=float).copy()
            # v31.5: 在基线视觉锁定之后，把“真实棒尖位置的估计”和“棒轴估计”
            # 分开做受控消融。始终不读取真实棒/孔的随机位姿。
            # axis_only 只修正斜率[2:4]，旧的棒尖局部位置[0:2]严格不变；
            # full 是原v31.4对照；baseline原样。绝不能将这几种模式混合算成功率。
            import os as _v315_os
            _variant = _v315_os.environ.get('ENO_V315_LOCK_VARIANT', 'axis_only')
            if _v315_os.environ.get('ENO_V31_SHARED_MOUNT_MODE') == 'delayed' and _variant != 'baseline':
                cand = getattr(mount, '_v314_pending_params', None)
                t = getattr(mount, '_v314_pending_at_step', None)
                reason = None
                if cand is None or t is None or now-int(t)>5 or int(t)>now:
                    reason = '候选超过5步未刷新；保持原安装标定'
                else:
                    cand = np.asarray(cand, dtype=float)
                    original_params = np.asarray(mount.estimate_params, dtype=float).copy()
                    if (cand.shape != (4,) or original_params.shape != (4,)
                            or not np.isfinite(cand).all() or not np.isfinite(original_params).all()):
                        reason = '候选/原参数非有限或形状不正确'
                    else:
                        old_tip,old_axis = mount._model(original_params)
                        candidate_tip,candidate_axis = mount._model(cand)
                        raw_tip_mm=float(np.linalg.norm(candidate_tip-old_tip)*1000.)
                        raw_axis_deg=float(np.degrees(np.arccos(np.clip(float(old_axis@candidate_axis),-1.,1.))))
                        if raw_tip_mm>4. or raw_axis_deg>4.:
                            reason = f'候选超限: full_tip_mm={raw_tip_mm:.4f}, full_axis_deg={raw_axis_deg:.4f}'
                        else:
                            applied=original_params.copy()
                            if _variant=='axis_only':
                                applied[2:4]=cand[2:4]
                            elif _variant=='full':
                                applied=cand.copy()
                            else:
                                raise RuntimeError('ENO_V315_LOCK_VARIANT非法：'+str(_variant))
                            new_tip,new_axis=mount._model(applied)
                            actual_tip_mm=float(np.linalg.norm(new_tip-old_tip)*1000.)
                            actual_axis_deg=float(np.degrees(np.arccos(np.clip(float(old_axis@new_axis),-1.,1.))))
                            if _variant=='axis_only' and actual_tip_mm>1e-6:
                                raise RuntimeError('v31.5 axis_only意外改变局部棒尖位置，拒绝继续')
                            mount.locked_mount_params=applied.copy()
                            # 只在实验性新参数启用时，短暂阻止Z负向动作。
                            # 单次episode控制频率20Hz，25步对应1.25秒；
                            # 这不是成功率技巧，只是让IK的姿态修正先稳定下来。
                            _cooldown=int(_v315_os.environ.get('ENO_V315_Z_HOLD_STEPS','25'))
                            if not 0<=_cooldown<=60:
                                raise RuntimeError('ENO_V315_Z_HOLD_STEPS只能是0..60')
                            u._v315_z_hold_until=int(now+_cooldown)
                            u._v315_z_hold_logged=False
                            print('[V315_LOCK_SWITCH] '+__import__('json').dumps({
                                'variant':_variant,'step':int(now),'last_candidate_step':int(t),
                                'old_to_candidate_tip_mm':round(raw_tip_mm,5),
                                'old_to_candidate_axis_deg':round(raw_axis_deg,5),
                                'applied_tip_jump_mm':round(actual_tip_mm,8),
                                'applied_axis_change_deg':round(actual_axis_deg,5),
                                'z_hold_steps':_cooldown,'truth_used':False,
                            },ensure_ascii=False),flush=True)
                if reason is not None:
                    print('[V315_LOCK_REJECT] '+reason,flush=True)
            elif _variant=='baseline':
                print('[V315_LOCK_BASELINE] 无候选参数切换',flush=True)
            # 同一控制步内的观测和IK必须使用**同一份**锁定后的工具参数。
            # 原v31.4先计算rel再锁定、却没有立即刷新rel；会造成首步门控与IK不同步。
            # 这里只使用同一帧孔视觉测量、法兰FK和新锁定参数，绝无仿真真值。
            _v315_tool_tip,_v315_tool_axis=mount._model(mount.locked_mount_params)
            _v315_world_tip=pos+rot@_v315_tool_tip
            _v315_world_axis=rot@_v315_tool_axis
            rel=np.r_[np.asarray([hole_xy[0],hole_xy[1],hole_z])-_v315_world_tip,
                      _v315_world_axis[:2]]
            if rel.shape!=(5,) or not np.isfinite(rel).all():
                raise V26VisualUnavailable('v31.5锁定后同一步相对姿态非法')
            mount.locked_hole_xy=np.asarray(hole_xy,dtype=float).copy()
            mount.locked_at_step=now
            locked=True
            print('[V261_VISUAL_MOUNT_LOCK] '
                  f'step={now} last_rgbd_step={mount.last_observed_step} '
                  f'radial_est_mm={np.linalg.norm(rel[:2])*1000:.3f} '
                  f'depth_est_mm={rel[2]*1000:.3f} '
                  f'mount_fit_frames={len(mount.history)} '
                  'source=RGBD+FR3_FK (NOT simulator pose)',flush=True)
    q['mount_freshness_control_steps']=age
    q['mount_locked_before_occlusion']=locked
    q['mount_locked_at_step']=mount.locked_at_step
    q['mount_tracking_mode']='rigid_mount_fk' if locked else 'active_rgbd_joint_fit'
    q['policy_fused_at_current_joint_state']=True
    return rel.copy(),q


def install_v26_student_ik(cls):
    """在继承原始IK的类上安装最小包装器，**不覆盖测量真值的方法**。"""
    original=cls._pre_physics_step
    ns=original.__globals__
    # 若原方法为继承方法，从父类解析函数globals。必须先找到原FK函数。
    if 'fk_task5' not in ns:
        for ancestor in cls.__mro__:
            fn=ancestor.__dict__.get('_pre_physics_step')
            if fn is not None and 'fk_task5' in fn.__globals__:
                ns=fn.__globals__;break
    if 'fk_task5' not in ns:
        raise RuntimeError('无法定位真实低层IK的FK函数，拒绝安装v26')
    fk=ns['fk_task5'];qseed=ns['V16_Q_SEED_RAD']

    def student_prephysics(self,actions):
        if not getattr(self,'_v16_ready',False):
            return original(self,actions)
        nom_tip=np.asarray(getattr(self,'_v23_nominal_tip_local'),dtype=float)
        nom_axis=np.asarray(getattr(self,'_v23_nominal_axis_local'),dtype=float)
        if nom_tip.shape!=(3,) or nom_axis.shape!=(3,):
            raise V26VisualUnavailable('名义工具几何未正确创建')
        # 首先仅用名义模型保证粗定位；一旦RGB-D完成安装估计，所有阶段用它控制。
        tool_tip=nom_tip.copy();tool_axis=nom_axis.copy()
        model=getattr(getattr(self,'_v23_visual_state',None),'mount',None)
        if model is not None and model.estimate_params is not None:
            rel,quality=v26_latest_visual(self,strict=False)
            if rel is not None:
                # 锁定后学生IK与PPO必须使用相同的安装位姿快照。
                mounting=(model.locked_mount_params if model.locked_mount_params is not None
                          else model.estimate_params)
                tool_tip,tool_axis=model._model(mounting)
                tool_tip=np.asarray(tool_tip,dtype=float).copy()
                tool_axis=np.asarray(tool_axis,dtype=float).copy()
        if not (np.isfinite(tool_tip).all() and np.isfinite(tool_axis).all()):
            raise V26VisualUnavailable('学生IK工具几何出现非有限数值')
        # 工作空间限位中心同样不能沿用真实tip的起始坐标。
        student_start=fk(self._v16_chain,qseed,self._v16_root_T,nom_tip,nom_axis)[:3]
        real_tip=self.tip_link_local_m
        real_axis=self.axis_link_local
        previous_start=self._v16_start_tip
        self.tip_link_local_m=tool_tip
        self.axis_link_local=tool_axis
        self._v16_start_tip=student_start
        # 实际计数：供最终报告确认至少有一段PPO物理动作走过视觉安装IK。
        if model is not None and model.estimate_params is not None and rel is not None:
            self._v26_vision_ik_steps=getattr(self,'_v26_vision_ik_steps',0)+1
        else:
            self._v26_nominal_ik_steps=getattr(self,'_v26_nominal_ik_steps',0)+1
        # v31.5: 姿态切换后允许视觉伺服继续调XY与棒轴姿态，
        # 但在短暂的settle窗口里拒绝负Z动作，防止正在修姿时直接下插。
        # “动作被限制”将单独记录；本版本为离线对照，不可混入PPO训练更新。
        _v315_until=getattr(self,'_v315_z_hold_until',None)
        if (_v315_until is not None and model is not None
                and model.locked_mount_params is not None):
            _v315_now=int(getattr(self,'_v21_step_count',-1))
            if _v315_now < int(_v315_until):
                if not getattr(self,'_v315_z_hold_logged',False):
                    print('[V315_Z_HOLD_START] '
                          f'step={_v315_now} until={_v315_until} student_only=true',flush=True)
                    self._v315_z_hold_logged=True
                # torch Tensor直接用张量自带clamp_min，无需额外导入torch。
                actions=actions.clone()
                actions[:,2]=actions[:,2].clamp_min(0.)
            elif getattr(self,'_v315_z_hold_logged',False):
                print('[V315_Z_HOLD_END] '+f'step={_v315_now}',flush=True)
                self._v315_z_hold_until=None
                self._v315_z_hold_logged=False
        try:
            return original(self,actions)
        finally:
            # 真值留给 env reward / oracle；绝不能在Actor输入或学生IK中使用。
            self.tip_link_local_m=real_tip
            self.axis_link_local=real_axis
            self._v16_start_tip=previous_start

    if getattr(cls,'_v26_ik_patch_applied',False):
        raise RuntimeError('v26 IK patch禁止重复安装')
    cls._pre_physics_step=student_prephysics
    cls._v26_ik_patch_applied=True
    print('[V26_STUDENT_IK_GUARD] PPO动作仅按名义/视觉工具参数进行FK+IK，oracle真值恢复仅供奖励',flush=True)


def v26_install_nominal_control_start(u):
    """prepare_v16完成后仅记录控制改造状态；不使用初始化的特权工具坐标。"""
    if not getattr(u,'_v16_ready',False):
        raise RuntimeError('必须先运行原prepare_v16()')
    if not hasattr(u,'_v23_nominal_tip_local') or not hasattr(u,'_v23_visual_state'):
        raise V26VisualUnavailable('没有v25相机安装与名义工具模型，拒绝启动训练')
    print('[V26_STUDENT_CONTROL_START] nominal tool before first visual; later uses observed mount',flush=True)
