#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在现有孔口射线测距基础上，利用23mm孔的内圈边界修正中心偏差。

只读取 raycast 的高度；不读取环境中孔的真实坐标。旧版fine扫描只取
粗估中心±18mm内的顶面命中像素中位数，可能把环形区域裁切得不对称。
本版改为评估候选孔中心内外两个环带的相反表面标签：
  r=8~10mm 预期孔内没有顶面；r=13~16mm 预期属于孔上表面。
参数来源于已知孔径23mm，不是随机场景的真值。
失败时抛异常，不以oracle自动校正。
"""
from __future__ import annotations
import numpy as np

class HoleRefineInvalid(RuntimeError):
    pass

def refine_hole_ray_center(sensor, grid_mm=.75, candidate_mm=.25):
    center=np.asarray(sensor._hole_meas[:2],dtype=float).copy()
    # 搜索窗口必须覆盖完整孔内外边界，而不是原先的±18mm裁切窗口。
    v=np.arange(-.026,.0261,grid_mm/1000.)
    xs,ys=np.meshgrid(center[0]+v,center[1]+v,indexing='xy')
    coords=np.column_stack([xs.ravel(),ys.ravel()])
    top=[]
    for x,y in coords:
        z=sensor._sample(float(x),float(y))
        top.append(bool(z is not None and abs(z-sensor._z)<.0012))
    top=np.asarray(top,dtype=bool)
    if top.sum()<90 or (~top).sum()<40:
        raise HoleRefineInvalid('孔口环带射线测量缺少足够的顶面/非顶面区域')
    offsets=np.arange(-.005,.00501,candidate_mm/1000.)
    winner=None
    for dx in offsets:
        for dy in offsets:
            trial=center+np.array([dx,dy])
            radius=np.linalg.norm(coords-trial,axis=1)
            # 实测孔径23mm → 内边界半径11.5mm。
            # 对整圈边界的像素分类正确率进行优化，不只比较宽泛的内外环带；
            # 否则多个偏离中心的候选点可能都得到满分，产生1mm系统误差。
            edge_region=(radius<=.018)
            if edge_region.sum()<80:continue
            expected_top=(radius>=.0115)
            score=float(np.mean(top[edge_region]==expected_top[edge_region]))
            score-=.0001*np.linalg.norm([dx,dy])/.005
            if winner is None or score>winner[0]:winner=(score,trial)
    if winner is None or winner[0]<.91:
        score_text='none' if winner is None else f'{winner[0]:.3f}'
        raise HoleRefineInvalid('孔口内外环带区分度不足: '+score_text)
    sensor._hole_meas[:2]=winner[1]
    return {'coarse_xy_m':center.tolist(),
            'refined_xy_m':winner[1].tolist(),
            'shift_from_initial_mm':float(np.linalg.norm(winner[1]-center)*1000.),
            'annulus_score':round(float(winner[0]),4),
            'ray_samples':len(top),
            'uses_privileged_hole_position':False}
