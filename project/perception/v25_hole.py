#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 静态孔中心：真实 RGB-D 外弧点在多帧中联合拟合圆心和半径。

信息边界：只使用 RGB-D、固定相机标定和已知桌面孔口 Z 高度。
绝不读取随机孔中心真值；独立的离线审计工具才允许读取真值。

与 v22 的关键差异：v22 把第 1 帧外圆半径当作永久固定值，
这里将每帧真实可见的外弧点放到同一个窗口，对 (cx,cy,r) 共同优化。
已知圆柱孔的半径 11.5 mm 不是蓝色孔座外圆半径，因此不能直接用 11.5 mm 约束外圆。
"""
from __future__ import annotations
from collections import deque
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from v21_rgbd_vision import _cam_calib,_backproject,_largest_region,diagnostic_masks,VisualInvalid
from v22_partial_hole import StaticHoleTracker

class JointStaticHoleTracker(StaticHoleTracker):
    def __init__(self, mouth_z_m=0.4, preferred_camera='hole', history_len=13):
        super().__init__(mouth_z_m=mouth_z_m, preferred_camera=preferred_camera)
        self.history=deque(maxlen=history_len)
        self.first_radius=None
        self.last_joint_info={}

    def reset(self):
        super().reset()
        self.history.clear()
        self.first_radius=None
        self.last_joint_info={}

    def _arc_points(self,packet):
        rgb,depth,K,pos,quat=packet
        _,blue,_=diagnostic_masks(rgb,depth)
        blue=_largest_region(blue,80)
        K,pos,R=_cam_calib(packet)
        yy,xx=np.nonzero(blue)
        xyz=_backproject(blue,depth,K,pos,R)
        close=np.abs(xyz[:,2]-self.mouth_z_m)<.0015
        if int(close.sum())<80: raise VisualInvalid('孔口上表面有效深度少于80点')
        top=np.zeros_like(blue,bool)
        top[yy[close],xx[close]]=True
        border=top & ~binary_erosion(top)
        xy=_backproject(border,depth,K,pos,R)[:,:2]
        rad=np.linalg.norm(xy-self.center_xy_m,axis=1)
        # ROI 由早期相机测得的孔中心和外半径定义，绝不查询目标孔真值。
        sel=xy[np.abs(rad-self.outer_radius_m)<.0035]
        if len(sel)<45:raise VisualInvalid('实测外弧点不足45')
        angles=np.arctan2(sel[:,1]-self.center_xy_m[1],sel[:,0]-self.center_xy_m[0])
        bins,_=np.histogram(angles,bins=24,range=(-np.pi,np.pi))
        coverage=int(np.sum(bins>=2))
        if coverage<7:raise VisualInvalid('外弧可见角度不足7/24格')
        # 对每帧限制采样数量，避免大分辨率图像人为增加某一帧的权重。
        if len(sel)>250:sel=sel[np.linspace(0,len(sel)-1,250,dtype=int)]
        return sel,coverage

    def observe(self,packets):
        if self.center_xy_m is None:
            info=super().initialize(packets)
            if info.get('status')=='measured_initial':
                self.first_radius=float(self.outer_radius_m)
                # 初始化也必须来自真实相机；初始帧仅用于建立 ROI，后续可重新拟合。
            return info
        issues=[]
        for name in [self.preferred_camera]+[x for x in packets if x!=self.preferred_camera]:
            if name not in packets:continue
            try:
                sel,covered=self._arc_points(packets[name])
                history_new=list(self.history)+[(sel,name)]
                pts=np.concatenate([x[0] for x in history_new],axis=0)
                x0=np.r_[self.center_xy_m,self.outer_radius_m]
                # 初始视觉半径仅限制合理搜索范围，不在优化中固定。
                rlo=max(.020,float(self.first_radius)-.005)
                rhi=min(.060,float(self.first_radius)+.005)
                def residual(x):
                    return np.linalg.norm(pts-x[:2],axis=1)-x[2]
                fitted=least_squares(residual,x0,loss='soft_l1',f_scale=.0008,
                    bounds=([x0[0]-.010,x0[1]-.010,rlo],
                            [x0[0]+.010,x0[1]+.010,rhi]),max_nfev=90)
                med=float(np.median(np.abs(fitted.fun))*1000.)
                move=float(np.linalg.norm(fitted.x[:2]-self.center_xy_m)*1000.)
                if med>1.0:raise VisualInvalid('联合外圆拟合残差超过1mm')
                if move>2.0:raise VisualInvalid('静态孔心单帧跳变超过2mm')
                if not fitted.success and fitted.cost>1e-3:raise VisualInvalid('孔口拟合未收敛')
                self.history.append((sel,name))
                self.center_xy_m=np.asarray(fitted.x[:2],float)
                self.outer_radius_m=float(fitted.x[2])
                self.observations+=1
                info={'status':'joint_multiframe_arc','camera':name,
                    'center_xy_m':self.center_xy_m.tolist(),
                    'outer_radius_mm':self.outer_radius_m*1000,
                    'edge_points_current_frame':len(sel),'joint_frames':len(self.history),
                    'joint_points':len(pts),'angle_bins_current_frame':covered,
                    'residual_median_mm':med,'shift_from_last_mm':move,
                    'observations':self.observations}
                self.last_joint_info=info
                return info
            except (VisualInvalid,ValueError) as ex:
                issues.append(f'{name}: {ex}')
        return {'status':'cached_static_hole','center_xy_m':self.center_xy_m.tolist(),
                'outer_radius_mm':self.outer_radius_m*1000,
                'joint_frames':len(self.history),'reasons':issues,
                'warning':'本帧无可信新弧段，沿用当前episode先前RGB-D定位，非该帧真值'}
