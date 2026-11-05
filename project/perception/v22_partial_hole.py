#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v22：仅处理已有RGB-D中的静态孔口；不修改原专家，也不查询物体真实位姿。

问题：旧版v21的 `binary_fill_holes()` 必须看到完整蓝色圆环才能估计孔中心；
在机械臂挡住一部分孔口时，即使可见几千个像素，也会直接判无效。

解法：在本次物理场景中孔位在一条trajectory内是静止的，先用有完整孔口的
一帧估计外圆中心和半径，然后在以后各帧仅从靠近预计外圆位置的**实际可见
蓝色边界**做鲁棒拟合。遮挡严重时保留上次观测、显式标记 cached。

注意：
- 旧先验来自相机先前帧，不是 PhysX 的随机真值。
- 本算法只解决孔的中心；**不能**推断被遮住的棒尖或棒轴。
- 这份代码首先是离线诊断工具，不能当作已可用于PPO的五维观测。
- 随机换孔位置时必须创建新tracker，不能跨episode沿用旧位置。
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from v21_rgbd_vision import (_cam_calib, _backproject, _largest_region,
                              diagnostic_masks, _estimate_hole, VisualInvalid)


class StaticHoleTracker:
    def __init__(self, mouth_z_m=0.4, preferred_camera='detail'):
        self.mouth_z_m=float(mouth_z_m)
        self.preferred_camera=preferred_camera
        self.center_xy_m=None
        self.outer_radius_m=None
        self.observations=0

    def reset(self):
        """每个独立随机孔场景需调用；旧v17的同进程reset不移动实体孔。"""
        self.center_xy_m=None
        self.outer_radius_m=None
        self.observations=0

    def initialize(self, packets):
        """只凭真正可见的完整孔口初始化，不用目标孔真值。"""
        order=[self.preferred_camera]+[x for x in packets if x!=self.preferred_camera]
        problems=[]
        for name in order:
            if name not in packets: continue
            try:
                xy,info=_estimate_hole({name:packets[name]},self.mouth_z_m)
                radius_mm=info['hole_views'][name]['outer_radius_mm']
                if not (20.0<=radius_mm<=60.0):
                    raise VisualInvalid('初始外圆直径不可信')
                self.center_xy_m=np.asarray(xy,dtype=float)
                self.outer_radius_m=float(radius_mm)/1000.0
                self.observations=1
                return dict(status='measured_initial', camera=name,
                            center_xy_m=self.center_xy_m.tolist(),
                            outer_radius_mm=radius_mm, observations=self.observations)
            except VisualInvalid as e:
                problems.append(f'{name}: {e}')
        return dict(status='uninitialized', reason='; '.join(problems))

    def observe(self, packets):
        """返回三种状态：新的可见弧拟合 / 复用同一场景的旧视觉坐标 / 未初始化。

        只有测量通过几何质量检查才更新缓存。不可以把 cached 误报成新传感器测量。
        """
        if self.center_xy_m is None:
            return self.initialize(packets)
        issues=[]
        for name in [self.preferred_camera]+[x for x in packets if x!=self.preferred_camera]:
            if name not in packets: continue
            try:
                p=packets[name]
                rgb,depth,K,pos,quat=p
                _,blue,_=diagnostic_masks(rgb,depth)
                blue=_largest_region(blue,80)
                K,pos,R=_cam_calib(p)
                yy,xx=np.nonzero(blue)
                xyz=_backproject(blue,depth,K,pos,R)
                close=np.abs(xyz[:,2]-self.mouth_z_m)<0.0015
                if int(close.sum())<80: raise VisualInvalid('孔上表面可见点少于80')
                top=np.zeros_like(blue,dtype=bool)
                top[yy[close],xx[close]]=True
                # 允许孔表面部分遮挡；只使用真实可见区域边缘，不填出假圆环。
                border=top&~binary_erosion(top)
                border_xyz=_backproject(border,depth,K,pos,R)
                xy=border_xyz[:,:2]
                rad=np.linalg.norm(xy-self.center_xy_m,axis=1)
                # 边缘可能包含遮挡造成的假边，按上一帧**视觉**外圆先验过滤。
                selected=xy[np.abs(rad-self.outer_radius_m)<0.0035]
                if len(selected)<45: raise VisualInvalid('可见外圆边界少于45点')
                angles=np.arctan2(selected[:,1]-self.center_xy_m[1],
                                  selected[:,0]-self.center_xy_m[0])
                bins,_=np.histogram(angles,bins=24,range=(-np.pi,np.pi))
                covered=int(np.sum(bins>=2))
                if covered<7: raise VisualInvalid('外圆可见角度覆盖不足')
                # 固定从初始可见表面测得的外半径：仅估计孔中心XY。
                sol=least_squares(lambda c:np.linalg.norm(selected-c,axis=1)-self.outer_radius_m,
                                  self.center_xy_m,loss='soft_l1',f_scale=0.0008,max_nfev=100)
                median_resid_mm=float(np.median(np.abs(sol.fun))*1000)
                displacement_mm=float(np.linalg.norm(sol.x-self.center_xy_m)*1000)
                if median_resid_mm>1.0: raise VisualInvalid('外圆拟合残差大于1mm')
                if displacement_mm>2.0: raise VisualInvalid('静态孔心跳变超过2mm')
                self.center_xy_m=np.asarray(sol.x,dtype=float)
                self.observations+=1
                return dict(status='measured_partial_arc', camera=name,
                            center_xy_m=self.center_xy_m.tolist(),
                            outer_radius_mm=self.outer_radius_m*1000,
                            edge_points=int(len(selected)),angular_bins_of_24=covered,
                            residual_median_mm=median_resid_mm,
                            shift_from_last_mm=displacement_mm,
                            observations=self.observations)
            except (VisualInvalid,ValueError) as e:
                issues.append(f'{name}: {e}')
        return dict(status='cached_static_hole', center_xy_m=self.center_xy_m.tolist(),
                    outer_radius_mm=self.outer_radius_m*1000,
                    observations=self.observations,
                    warning='本帧没有有效新测量，复用本episode较早的相机估计；不是当前图像重新测量',
                    reasons=issues)
