#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17.3 非特权【仅孔位】测距。无棒尖测距、无真实随机安装偏差。

传感器：固定世界坐标系上方的 PhysX 垂直射线阵列，像极简深度传感器。
输入只包括：射线命中位置/距离、已知标定区域、已知孔口高度、关节q/dq、
名义工具FK和历史动作；绝不查询 scene._v16_hole、真实peg tip、mount_delta。

边界：这不是已验证的 RGB-D 相机、不是接触/力控制，也不具有未知安装偏差的观测能力。
仅供大范围粗定位 BC：离孔仍>~10 mm时学会寻找大致方向，禁止声称最终插入成功。
"""
from __future__ import annotations
import numpy as np

FEATURE_DIM = 20
FEATURE_SCHEMA = 'v17_3_hole_ray_q7_dq7_nominal_relative3_prev_action3'


def _grid(cx, cy, span, step):
    n = round(2 * span / step) + 1
    values = np.linspace(-span, span, n, dtype=np.float64)
    for dx in values:
        for dy in values:
            yield float(cx + dx), float(cy + dy)


class V17RayDepthEstimator:
    """保留与 v17.2 教师采集相同的构造/方法接口，以便复用已验证的控制链。"""
    def __init__(self, hole_workspace_center_xy, hole_mouth_z, rng_seed=0,
                 grid_mm=2.0, measurement_sigma_mm=0.15):
        import carb
        from omni.physx import get_physx_scene_query_interface
        self._query = get_physx_scene_query_interface()
        self._carb = carb
        self._center = np.asarray(hole_workspace_center_xy, dtype=np.float64)
        self._z = float(hole_mouth_z)
        self._step = float(grid_mm) / 1000.
        self._sigma = float(measurement_sigma_mm) / 1000.
        self._rng = np.random.default_rng(int(rng_seed) + 937)
        self._hole_meas = None

    def _sample(self, x, y):
        # 仅使用射线返回的命中距离；没有访问collider所在prim的pose或path。
        origin_z, max_length = self._z + .09, .14
        result = self._query.raycast_closest(
            self._carb.Float3(float(x), float(y), origin_z),
            self._carb.Float3(0., 0., -1.), max_length)
        if not bool(result['hit']):
            return None
        distance = float(result['distance'])
        if not np.isfinite(distance) or not (0. <= distance <= max_length):
            return None
        return origin_z - distance

    def calibrate_once(self, nominal_tip_w, nominal_tip_local, link8_R_w, q7,
                       rng_peg_search_mm=20):
        """从孔口上表面射线命中点估计XY。参数nominal_*只校验形状，不用于猜孔真值。

        不能在拟合失败时以随机初始化的孔中心作为传感结果；直接报错。
        这一步只涉及孔测距，因此不会触发之前失败的peg_bottom查询。
        """
        if self._hole_meas is not None:
            raise RuntimeError('每条轨迹只允许校准一次；重新随机孔位必须重新采集测距')
        nominal_tip_local = np.asarray(nominal_tip_local, dtype=np.float64)
        if nominal_tip_local.shape != (3,) or np.any(np.abs(nominal_tip_local[:2]) > 1e-10):
            raise RuntimeError('名义工具局部坐标不能带有真实棒安装XY偏差')
        # 整个工作区孔中心可能偏移±50mm，孔座的外缘还有约30mm半径。
        # 旧版±70mm覆盖不足，改成±90mm，仍只查询距离，不读取真值。
        samples, raw_hits = [], 0
        for x,y in _grid(self._center[0], self._center[1], .090, self._step):
            z = self._sample(x,y)
            if z is None: continue
            raw_hits += 1
            if abs(z-self._z) <= .0012:
                samples.append([x,y])
        print('[V17_3_HOLE_RAY_AUDIT] '+str({
            'raw_hits': raw_hits, 'hole_top_hits': len(samples),
            'mount_observed': False}), flush=True)
        if len(samples) < 25:
            raise RuntimeError(f'孔测距有效点不足 {len(samples)}/25，不能使用真值补救')
        center = np.median(np.asarray(samples, dtype=np.float64), axis=0)
        radius = float(np.median(np.linalg.norm(np.asarray(samples)-center,axis=1)))
        if not (.009 <= radius <= .030):
            raise RuntimeError(f'孔顶面形状尺寸异常，proxy radius={radius*1000.:.2f}mm')
        fine = []
        for x,y in _grid(center[0],center[1], .018, .00075):
            z=self._sample(x,y)
            if z is not None and abs(z-self._z)<=.0012:
                fine.append([x,y])
        if len(fine) >= 40:
            center=np.median(np.asarray(fine,dtype=np.float64),axis=0)
        if not np.isfinite(center).all() or np.max(np.abs(center-self._center))>.075:
            raise RuntimeError('孔中心测量不合理，禁止oracle兜底')
        self._hole_meas = np.array([center[0],center[1],self._z], dtype=np.float64)
        info = {'type':'physx_ray_hole_top_only_not_rgb_camera',
                'hole_top_surface_pixels':len(samples),
                'hole_fine_pixels':len(fine),
                'hole_ring_radius_proxy_mm':radius*1000.,
                'hole_detected_xy_m':center.tolist(),
                'mount_offset_unobserved':True,
                'no_scene_ground_truth_access':True,
                'feature_schema':FEATURE_SCHEMA}
        print('[V17_3_HOLE_MEASURED] '+str(info), flush=True)
        return info

    def student_observation(self, q7,dq7, nominal_tip_world_from_fk,
                            link8_R_w,last_action3):
        """产生20D观测：q7/3, dq7/3, 测量孔-名义tip(3)/0.05, 上一步动作(3)。
        注意：真实棒尖和真实安装偏差不在此函数的参数里；禁止oracle/真值加噪替代。
        """
        if self._hole_meas is None:
            raise RuntimeError('必须先用真实射线测得孔位置')
        q=np.asarray(q7,dtype=np.float32)
        v=np.asarray(dq7,dtype=np.float32)
        tip=np.asarray(nominal_tip_world_from_fk,dtype=np.float64)
        prev=np.asarray(last_action3,dtype=np.float32)
        if q.shape!=(7,) or v.shape!=(7,) or tip.shape!=(3,) or prev.shape!=(3,):
            raise RuntimeError('学生观测q/dq/tip/prev维度错误')
        # 噪声加在测量孔中心基础上，不是加在真实孔位或真实棒尖上。
        relative=self._hole_meas-tip+self._rng.normal(0.,self._sigma,size=3)
        x=np.concatenate([q/3.,v/3.,np.clip(relative/.05,-3.,3.),prev]).astype(np.float32)
        if x.shape!=(20,) or not np.isfinite(x).all():
            raise RuntimeError('学生观测必须是20D有限向量')
        return x
