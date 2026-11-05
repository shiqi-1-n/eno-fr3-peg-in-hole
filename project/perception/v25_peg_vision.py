#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 棒安装位姿估计：多相机、多时刻、法兰局部坐标系联合圆柱拟合。

已知量：机器人本体 link8 的关节正运动学（位置+旋转）、名义40mm工具轴和半径10mm、
         固定相机RGB-D及内外参。未知量：本次安装平移/倾斜（从观测自己估计）。
严格禁止：实际挂载 tip_local / 实际 mount_perturbation / 真实孔位 / 离线误差进入本模块。

关键点：每个真实侧面点先转到法兰局部系，使每一时刻相同的安装参数保持不变。
不再用世界系中逐帧棒尖的平均值冒充本体安装标定。单场景内刚性安装不变是必要前提。
柱面拟合偏差仍可能来自视角/标定，不能以低残差替代真值精度验收。
"""
from __future__ import annotations
from collections import deque
import numpy as np
from scipy.ndimage import binary_erosion,label
from scipy.optimize import least_squares
from v21_rgbd_vision import _backproject,_cam_calib,diagnostic_masks,VisualInvalid
from v25_hole import JointStaticHoleTracker

PEG_RADIUS_M=.010
PEG_LENGTH_M=.040


def unit(v):
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    if n<1.e-7 or not np.isfinite(n):raise VisualInvalid('归一化轴向失败')
    return v/n


def basis_from_axis(a0):
    a0=unit(a0)
    # 只构造法兰局部两条正交横向方向，不使用真实安装扰动方向。
    base=np.array([1.,0.,0.]) if abs(a0[0])<.9 else np.array([0.,1.,0.])
    e1=unit(base-a0*np.dot(base,a0))
    e2=unit(np.cross(a0,e1))
    return e1,e2


class JointFlangeMount:
    """保存最多12个采集时刻，每时刻两个侧视独立点集；所有点已经变到法兰局部系。"""
    def __init__(self,history_len=12,max_age_steps=80):
        self.history=deque(maxlen=history_len)
        self.nominal_local_tip=None
        self.nominal_local_axis=None
        self.estimate_params=None
        self.last_observed_step=None
        self.max_age_steps=max_age_steps
        self.last_quality={}
        # v26.1: 一次任务中刚性棒安装固定；已通过视觉标定并接近孔口后可锁定。
        # 该变量只能由v26_runtime基于视觉观测设置，不能来自仿真真实安装姿态。
        self.locked_mount_params=None
        self.locked_hole_xy=None
        self.locked_at_step=None

    def _collect_points(self,packets,flange_pos,flange_R,nom_tip,nom_axis):
        group=[];stats={}
        for name,pkt in packets.items():
            rgb,depth,*_=pkt
            orange,_,_=diagnostic_masks(rgb,depth)
            if not orange.any():
                stats[name]={'raw_pixels':0,'selected':0};continue
            K,cpos,R=_cam_calib(pkt)
            v,u=np.nonzero(orange)
            xyz=_backproject(orange,depth,K,cpos,R)
            along=(xyz-nom_tip)@(-nom_axis)
            radial=np.linalg.norm((xyz-nom_tip)-np.outer(along,-nom_axis),axis=1)
            keep=(along>-.013)&(along<PEG_LENGTH_M+.013)&(radial<.033)
            region=np.zeros_like(orange,bool)
            region[v[keep],u[keep]]=True
            lab,n=label(region,structure=np.ones((3,3),np.uint8))
            if n:
                counts=np.bincount(lab.ravel());counts[0]=0
                region=(lab==np.argmax(counts))
                if region.sum()>=30:
                    inner=binary_erosion(region,iterations=2)
                    if inner.sum()>=25:region=inner
                    pworld=_backproject(region,depth,K,cpos,R)
                    # 对齐：world_column = flange_R @ local_column + flange_pos。
                    plocal=(pworld-flange_pos)@flange_R
                    if len(plocal)>240:plocal=plocal[np.linspace(0,len(plocal)-1,240,dtype=int)]
                    if len(plocal)>=25:
                        group.append((name,plocal))
                        stats[name]={'raw_pixels':int(orange.sum()),'selected':len(plocal)}
                        continue
            stats[name]={'raw_pixels':int(orange.sum()),'selected':0}
        return group,stats

    def _model(self,x):
        a0=self.nominal_local_axis;e1,e2=basis_from_axis(a0)
        tip=self.nominal_local_tip+e1*x[0]+e2*x[1]
        axis=unit(a0+e1*x[2]+e2*x[3])
        return tip,axis

    def _fit(self,hist):
        clouds=[pts for _,views in hist for _,pts in views]
        if len(clouds)<2:raise VisualInvalid('棒有效视角数据少于2组，拒绝姿态标定')
        x0=np.zeros(4) if self.estimate_params is None else self.estimate_params
        # 每个相机/时刻等量采样，不让某一相机因距离近而吞掉另一相机信息。
        pts=np.concatenate(clouds,axis=0)
        def surface(x):
            tip,ax=self._model(x)
            d=pts-tip
            return np.linalg.norm(d-np.outer(d@ax,ax),axis=1)-PEG_RADIUS_M
        def residual(x):
            # 很弱的几何先验：约束只表达任务中小扰动的有限范围，不包含实际随机真值。
            return np.r_[surface(x),x[0]*.025,x[1]*.025,x[2]*.003,x[3]*.003]
        sol=least_squares(residual,x0,bounds=([-.014,-.014,-.1,-.1],[.014,.014,.1,.1]),
                          loss='soft_l1',f_scale=.0007,max_nfev=70)
        raw=surface(sol.x)
        med=float(np.median(np.abs(raw))*1000)
        p90=float(np.quantile(np.abs(raw),.9)*1000)
        if med>1.2 or p90>2.4:raise VisualInvalid(f'棒联合拟合残差过大 med={med:.3f}mm p90={p90:.3f}mm')
        # 双视角是几何角度可观测性要求；不将单视角柱面低残差当成角度精度保证。
        views={name for _,groups in hist for name,_ in groups}
        if len(views)<2:raise VisualInvalid('缺少至少两个独立棒侧视方向')
        return sol.x,{'mount_joint_frames':len(hist),'mount_joint_points':len(pts),
                      'mount_joint_views':sorted(views),'mount_surface_median_mm':med,
                      'mount_surface_p90_mm':p90,'optimizer_success':bool(sol.success)}

    def observe(self,packets,nom_tip,nom_axis,flange_pos,flange_R,step):
        flange_pos=np.asarray(flange_pos,float);flange_R=np.asarray(flange_R,float)
        nom_tip=np.asarray(nom_tip,float);nom_axis=unit(nom_axis)
        if flange_pos.shape!=(3,) or flange_R.shape!=(3,3):raise VisualInvalid('法兰FK位姿尺寸错误')
        if not np.isfinite(flange_R).all() or abs(np.linalg.det(flange_R)-1)>.01:
            raise VisualInvalid('法兰FK旋转矩阵非正交')
        local_tip=flange_R.T@(nom_tip-flange_pos)
        local_axis=unit(flange_R.T@nom_axis)
        if self.nominal_local_tip is None:
            self.nominal_local_tip=local_tip
            self.nominal_local_axis=local_axis
        elif np.linalg.norm(local_tip-self.nominal_local_tip)>.0005 or np.linalg.norm(local_axis-self.nominal_local_axis)>.005:
            raise VisualInvalid('名义工具模型在任务中改变，拒绝混合不同安装定义')
        # 插入前成功锁定后，本次任务内棒刚性固定在法兰上：
        # 后续相机被孔座遮挡不等于安装标定失效，不再对仅剩少量可见点反复重拟合。
        # 注意：这是单次仿真episode的模型假设，不可照搬到可能松动的真实抓取场景。
        raw_accepted=False
        if self.locked_mount_params is not None:
            stats={'mount':'locked_from_prior_rgbd; no new peg fit required'}
            err='棒安装标定已在插入前锁定，当前以法兰关节姿态传播'
        else:
            group,stats=self._collect_points(packets,flange_pos,flange_R,nom_tip,nom_axis)
            err=None
            if len(group)>=1 and sum(len(a) for _,a in group)>=90:
                candidate=list(self.history)+[(int(step),group)]
                try:
                    params,quality=self._fit(candidate)
                    self.history.append((int(step),group))
                    self.estimate_params=params
                    self.last_observed_step=int(step)
                    self.last_quality=quality
                    raw_accepted=True
                except (VisualInvalid,ValueError) as ex:err=str(ex)
            else:err='棒的当前相机ROI深度点不足'
        if self.estimate_params is None or self.last_observed_step is None:
            raise VisualInvalid('还没有可信的棒安装视觉估计: '+str(err))
        age=int(step)-self.last_observed_step
        if age<0 or (self.locked_mount_params is None and age>self.max_age_steps):
            raise VisualInvalid(f'插入前棒安装视觉标定超过有效时间: {age}步')
        params_used=(self.locked_mount_params if self.locked_mount_params is not None
                     else self.estimate_params)
        tip_local,axis_local=self._model(params_used)
        tip_world=flange_pos+flange_R@tip_local
        axis_world=unit(flange_R@axis_local)
        return tip_world,axis_world,{'fresh_rgbd':raw_accepted,'age_steps':age,
            'current_peg_views':stats,'current_error':err,'fit_quality':self.last_quality,
            'estimated_mount_xy_and_tilt_local':self.estimate_params.tolist(),
            'source':'rendered_rgbd_plus_robot_fk','privileged_pose_used':False,
            'mount_locked_before_occlusion':self.locked_mount_params is not None}


class VisionState:
    def __init__(self,mouth_z=.4):
        self.hole=JointStaticHoleTracker(mouth_z_m=mouth_z,preferred_camera='hole')
        self.mount=JointFlangeMount()
        self.mouth_z=float(mouth_z)

    def observe(self,packets,nominal_tip,nominal_axis,flange_pos,flange_R,step):
        hole=self.hole.observe(packets)
        if hole['status']=='uninitialized':raise VisualInvalid('孔口未能从相机定位: '+hole.get('reason',''))
        # 第一帧发生在原专家完成reset之前：只为静态孔口建立视觉初始化，
        # 不能拿暂态机器人姿态拟合本次棒安装参数。
        if int(step) < 5:raise VisualInvalid('专家reset前的第一帧不用于棒安装估计')
        tip,axis,peg=self.mount.observe(packets,nominal_tip,nominal_axis,flange_pos,flange_R,step)
        h=np.array([*hole['center_xy_m'],self.mouth_z],float)
        rel=np.r_[h-tip,axis[:2]]
        if not np.isfinite(rel).all():raise VisualInvalid('相对5D测量非有限数')
        return rel,{'hole':hole,'peg_tip_estimated_m':tip.tolist(),
                'axis_world':axis.tolist(),'sensor_estimate_status':
                'fresh_rgbd_plus_fk' if peg['fresh_rgbd'] else 'propagated_from_prior_rgbd_plus_fk',
                'raw_peg_visible_this_step':peg['fresh_rgbd'],
                'mount_freshness_control_steps':peg['age_steps'],
                'mount_calibration_frames':len(self.mount.history),
                'peg_source_error_if_propagated':peg['current_error'],
                'peg_raw_fit_quality':peg,'sensors_used':['rendered_rgbd','robot_flange_proprioceptive_fk'],
                'privileged_mount_pose_used':False}


# -*- coding: utf-8 -*-
"""v31 候选：在原 JointFlangeMount._fit 后安装的可切换扩展。

模式：shadow（旧算法全程控制）/ delayed（旧算法锁定后才启用候选）/ off（原样）。
真实的安装参数只有4个：平移2、轴向斜率2。其他参数是相机观测偏置，
仅用于拟合时吸收跨相机坐标不一致，绝不存入 tip/axis 或用于奖励真值。
本候选把偏置近似看成法兰局部系短时固定：相机误差实际可能属于世界系，
因此只作为小规模回归候选，不应直接宣称可迁移到真实硬件或最终验收。
"""
import os as _v31_os
import json as _v31_json

_V31_OLD_FIT = JointFlangeMount._fit
_V31_MODE = _v31_os.environ.get('ENO_V31_SHARED_MOUNT_MODE', 'shadow').strip().lower()
if _V31_MODE not in ('off', 'shadow', 'delayed'):
    raise RuntimeError('ENO_V31_SHARED_MOUNT_MODE 只能是 off/shadow/delayed')


def _v31_fit_candidate(self, hist):
    # 只使用原 RGB-D + FR3 FK 形成的法兰局部点云；没有任何物理场景真值。
    views = [(name, np.asarray(pts, dtype=float))
             for _, groups in hist for name, pts in groups if len(pts) >= 25]
    names = [n for n in ('peg_front', 'peg_left', 'hole') if any(v == n for v, _ in views)]
    if len(names) < 2:
        raise VisualInvalid('v31 至少需要两个相机方向的有效观测')
    if len(views) < 2 or sum(len(p) for _, p in views) < 90:
        raise VisualInvalid('v31 有效圆柱点云不足')
    # 通过零均值约束消除“真实安装平移”和“相机偏置”之间的规范自由度。
    # 第一个相机偏置相对于其他相机的差值作为优化变量；所有相机偏置和为0。
    nv = len(names)
    def decode(x):
        raw = np.zeros((nv, 2), dtype=float)
        if nv > 1:
            raw[1:] = np.asarray(x[4:], dtype=float).reshape(nv - 1, 2)
        corrected = raw - np.mean(raw, axis=0, keepdims=True)
        return {name: corrected[i] for i, name in enumerate(names)}

    e1, e2 = basis_from_axis(self.nominal_local_axis)
    per = {n: np.concatenate([p for k, p in views if k == n], axis=0) for n in names}
    def surf(x):
        tip, ax = self._model(x[:4])
        offsets = decode(x)
        errors = []
        for name in names:
            shifted_tip = tip + e1 * offsets[name][0] + e2 * offsets[name][1]
            pts = per[name]
            d = pts - shifted_tip
            errors.append(np.linalg.norm(d - np.outer(d @ ax, ax), axis=1) - PEG_RADIUS_M)
        return np.concatenate(errors)

    def residual(x):
        # 与旧解相同的 mount 弱先验；另加很弱的相机偏置先验。
        # .02 × 1mm = .02mm，远小于通常的点云表面残差。
        return np.r_[surf(x), x[0] * .025, x[1] * .025,
                     x[2] * .003, x[3] * .003, x[4:] * .02]

    bounds_lo = np.array([-.014, -.014, -.1, -.1] + [-.002] * (2 * (nv - 1)))
    bounds_hi = -bounds_lo
    old = np.asarray(self.estimate_params if self.estimate_params is not None else np.zeros(4), dtype=float)
    prior_bias = getattr(self, '_v31_bias_prior', {})
    starts = []
    warm = np.r_[old, np.zeros(2 * (nv - 1))]
    for i, name in enumerate(names[1:]):
        if name in prior_bias and names[0] in prior_bias:
            warm[4+2*i:6+2*i] = np.asarray(prior_bias[name]) - np.asarray(prior_bias[names[0]])
    starts += [('warm', warm), ('reset_bias', np.r_[old, np.zeros(2 * (nv-1))])]
    starts += [('zero', np.zeros(len(bounds_lo)))]
    candidates = []
    for label, init in starts:
        init = np.minimum(np.maximum(init, bounds_lo + 1e-7), bounds_hi - 1e-7)
        try:
            sol = least_squares(residual, init,
                    bounds=(bounds_lo, bounds_hi),
                    loss='soft_l1', f_scale=.0007,
                    max_nfev=100, x_scale='jac')
            if np.isfinite(sol.cost) and np.isfinite(sol.x).all() and sol.success:
                candidates.append((float(sol.cost), label, sol))
        except (ValueError, FloatingPointError):
            continue
    if not candidates:
        raise VisualInvalid('v31 共享轴向优化没有收敛的有限解')
    cost, label, sol = min(candidates, key=lambda item: item[0])
    geom_error = np.abs(surf(sol.x))
    med = float(np.median(geom_error) * 1000)
    p90 = float(np.quantile(geom_error, .9) * 1000)
    if med > 1.2 or p90 > 2.4:
        raise VisualInvalid(f'v31 棒拟合残差过大 med={med:.3f}mm p90={p90:.3f}mm')
    cam_offset = decode(sol.x)
    max_offset = max(float(np.linalg.norm(v)) for v in cam_offset.values())
    if max_offset > .0025:
        raise VisualInvalid('v31 相机偏置超出2.5mm可信范围')
    tip, ax = self._model(sol.x[:4])
    # 仅保留相机偏置作为下一次拟合的数值初值；控制使用前4个真实安装参数。
    quality = {
        'mount_joint_frames': len(hist),
        'mount_joint_points': int(sum(len(p) for p in per.values())),
        'mount_joint_views': list(names),
        'mount_surface_median_mm': med,
        'mount_surface_p90_mm': p90,
        'optimizer_success': bool(sol.success),
        'v31_diagnostic_mode': _V31_MODE,
        'v31_best_initialization': label,
        'v31_shared_axis_tilt_local_deg': float(np.degrees(np.arccos(np.clip(np.dot(ax, self.nominal_local_axis), -1, 1)))),
        'v31_nuisance_offsets_local_mm': {k: [round(float(t*1000), 5) for t in v] for k, v in cam_offset.items()},
        'v31_nuisance_max_local_mm': round(max_offset*1000, 5),
        'v31_fit_cost': cost,
    }
    return np.asarray(sol.x[:4], dtype=float).copy(), quality, cam_offset


def _v31_fit_switch(self, hist):
    if _V31_MODE == 'off':
        return _V31_OLD_FIT(self, hist)
    old_params = old_quality = None
    old_exc = None
    try:
        old_params, old_quality = _V31_OLD_FIT(self, hist)
    except (VisualInvalid, ValueError) as exc:
        old_exc = str(exc)
    new_params = new_quality = cam_offset = None
    new_exc = None
    try:
        new_params, new_quality, cam_offset = _v31_fit_candidate(self, hist)
    except (VisualInvalid, ValueError) as exc:
        new_exc = str(exc)
    step = int(hist[-1][0]) if hist else -1
    if step % 25 == 0 or step < 15 or (new_exc and _V31_MODE == 'active'):
        def angle(x):
            if x is None: return None
            a = self._model(x)[1]
            return round(float(np.degrees(np.arccos(np.clip(float(a @ self.nominal_local_axis), -1, 1)))), 5)
        print('[V31_MOUNT_COMPARE] ' + _v31_json.dumps({
            'step': step, 'mode': _V31_MODE,
            'old_local_tilt_deg': angle(old_params),
            'new_local_tilt_deg': angle(new_params),
            'old_med_mm': (old_quality or {}).get('mount_surface_median_mm'),
            'new_med_mm': (new_quality or {}).get('mount_surface_median_mm'),
            'offsets': (new_quality or {}).get('v31_nuisance_offsets_local_mm'),
            'old_error': old_exc, 'new_error': new_exc,
            'truth_used': False,
        }, ensure_ascii=False), flush=True)
    # v31.4: 只缓存候选，锁定前不得让它影响IK、机械臂接近路径或孔位锁定判据。
    if new_params is not None:
        self._v314_pending_params = np.asarray(new_params, dtype=float).copy()
        self._v314_pending_at_step = step
        self._v314_pending_quality = dict(new_quality)
        # 保存相机偏置仅用作下一帧优化初值（不加入学生工具位姿）。
        self._v31_bias_prior = {n: np.array(v, dtype=float).copy() for n,v in cam_offset.items()}
    else:
        # 失去双视角时，之前的候选仍在，但锁定时必须检查新鲜度。
        pass
    if _V31_MODE in ('shadow', 'delayed'):
        if old_params is None:
            raise VisualInvalid('原模型无法完成拟合: ' + str(old_exc))
        # 严格保持基线实际控制的安装参数。
        return old_params, old_quality
    if new_params is None:
        raise VisualInvalid('v31 主动模式拒绝不可信拟合: ' + str(new_exc))
    self._v31_bias_prior = {n: np.array(v, dtype=float).copy() for n, v in cam_offset.items()}
    return new_params, new_quality

JointFlangeMount._fit = _v31_fit_switch
print('[V31_SHARED_MOUNT_INSTALLED] mode=' + _V31_MODE + ' student=RGBD+FK only', flush=True)
