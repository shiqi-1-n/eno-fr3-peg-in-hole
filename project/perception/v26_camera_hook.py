# -*- coding: utf-8 -*-
"""v21: 不接管专家的纯旁路 RGB-D 记录器。

关键保证：
1. 仅对已有 FR3 仿真环境 _setup_scene() 增加固定相机，专家控制不变。
2. 对环境 step() 的返回值和实际执行动作完全不修改。
3. 相机/几何估计异常只进入独立的视觉错误日志，不改专家运动。
4. RGB-D 学生观测保存到 student_camera_*.jsonl，与离线 oracle 验收分文件。
5. 首次验证严格复用 v17.4 原始 XY 安装扰动，不擅自声明旋转已覆盖；
   待视觉有效再基于原专家增加旋转扰动适配。

注意：这属于 Isaac Lab 2.3.1/Python 运行时扩展；必须在原始专家生成代码
的类定义完成后、main() 开始前安装（prepare_v21.py 负责）。
"""
from __future__ import annotations
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def _look_at(pos, target):
    """OpenGL 相机的局部 -Z 指向目标、+Y 向上；四元数顺序 wxyz。"""
    p = np.asarray(pos, dtype=float)
    f = np.asarray(target, dtype=float) - p
    norm = float(np.linalg.norm(f))
    if norm < .05:
        raise ValueError('相机与目标过近')
    f /= norm
    z = -f
    up = np.array([0., 0., 1.])
    if abs(float(np.dot(z, up))) > .98:
        up = np.array([0., 1., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def install_v26_camera_hook(base_env_cls, step_owner_cls, sim_utils, workspace_xy, env_root):
    """在原专家创建环境之前安装旁路相机；不修改任何专家决策函数。"""
    from isaaclab.sensors import Camera, CameraCfg
    from v21_rgbd_vision import diagnostic_masks, VisualInvalid
    from v25_peg_vision import VisionState

    root = Path(os.environ.get('ENO_V26_CAPTURE_DIR', '/eno/output/session/v26_vision')).resolve()
    interval = int(os.environ.get('ENO_V26_CAPTURE_EVERY', '5'))
    if not 1 <= interval <= 200:
        raise RuntimeError('ENO_V26_CAPTURE_EVERY 必须在1..200控制步')
    workspace_xy = tuple(float(x) for x in workspace_xy)
    if len(workspace_xy) != 2:
        raise RuntimeError('固定工作区坐标配置错误')

    original_setup = base_env_cls._setup_scene
    original_step = step_owner_cls.step
    # 这3台都是物理固定安装的 RGB-D，安装位置与随机孔位/真实棒位姿无关。
    # hole：保留v21的孔口视角，继续复用v22圆弧跟踪；
    # peg_left / peg_front：新增交叉侧视，避免机械臂末端从上方遮住40mm圆柱棒。
    config = {
        'hole': {
            'pos': (workspace_xy[0] + .26, workspace_xy[1] + .27, .90),
            'focal': 16.0, 'size': (960, 720),
            'look': (workspace_xy[0] - .06, workspace_xy[1], .52),
        },
        'peg_left': {
            'pos': (workspace_xy[0] - .28, workspace_xy[1] + .025, .63),
            'focal': 11.0, 'size': (960, 720),
            'look': (workspace_xy[0] - .05, workspace_xy[1], .43),
        },
        'peg_front': {
            'pos': (workspace_xy[0] - .025, workspace_xy[1] - .28, .62),
            'focal': 11.0, 'size': (960, 720),
            'look': (workspace_xy[0] - .05, workspace_xy[1], .43),
        },
    }
    mouth_z = float(os.environ.get('ENO_V21_KNOWN_MOUTH_Z_M', '.4'))

    def setup_with_passive_camera(self):
        result = original_setup(self)
        # 用已知机器人工具零偏装配模型建立名义ROI，不读取本场景真实mount_xy。
        # USD在安装随机棒之前就有link8姿态；固定down_local与随机安装无关。
        from pxr import Usd, UsdGeom, Gf
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        link=stage.GetPrimAtPath('/World/envs/env_0/Robot/fr3_link8')
        if not link.IsValid():raise RuntimeError('v23: 缺少FR3 link8 USD节点')
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        inv=cache.GetLocalToWorldTransform(link).GetInverse()
        down_local=inv.TransformDir(Gf.Vec3d(0.,0.,-1.))
        d=np.array([float(down_local[i]) for i in range(3)],dtype=float)
        d=d/np.linalg.norm(d)
        self._v23_nominal_tip_local=d*0.040  # 单位m，已知棒长，不包含随机XY/旋转
        self._v23_nominal_axis_local=d
        self._v23_visual_state=VisionState(mouth_z=mouth_z)
        self._v21_camera_error = None
        self._v21_cameras = {}
        self._v21_cam_calib = {}
        self._v21_step_count = 0
        self._v26_latest_visual = None
        for name, settings in config.items():
            try:
                pos = settings['pos']
                quat = _look_at(pos, settings['look'])
                cfg = CameraCfg(
                    prim_path=f'{env_root}/V21Camera_{name}',
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=settings['focal'], horizontal_aperture=20.955,
                        clipping_range=(.06, 3.0)),
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=quat, convention='opengl'),
                    width=settings['size'][0], height=settings['size'][1],
                    update_period=0.0, data_types=['rgb', 'distance_to_image_plane'],
                )
                camera = Camera(cfg)
                self.scene.sensors[f'v26_camera_{name}'] = camera
                self._v21_cameras[name] = camera
                self._v21_cam_calib[name] = (np.asarray(pos), np.asarray(quat))
                print('[V25_CAMERA_CREATED] ' + json.dumps({
                    'camera': name, 'fixed_world_pos_m': pos,
                    'fixed_target_m': settings['look'], 'focal_length_mm': settings['focal'],
                }, ensure_ascii=False), flush=True)
            except Exception as exc:
                # 独立视觉失败与专家失败，不能替专家选择动作或改变初始化。
                self._v21_camera_error = f'{name}: {type(exc).__name__}: {exc}'
                _append_jsonl(root/'vision_errors.jsonl',
                              {'stage':'setup','camera':name,'error':self._v21_camera_error})
                print('[V21_VISION_SETUP_WARN] '+self._v21_camera_error, flush=True)
        return result

    def record_after_step(self):
        # 视觉只观察步骤执行后的实际画面；不注入专家控制循环。
        self._v21_step_count += 1
        idx = self._v21_step_count
        if idx != 1 and idx % interval != 0:
            return
        if not self._v21_cameras:
            _append_jsonl(root/'vision_errors.jsonl', {'step':idx, 'error':'未创建RGB-D相机'})
            return
        frames = {}
        packets = {}
        for name, cam in self._v21_cameras.items():
            try:
                raw = cam.data.output
                rgb = raw['rgb'][0].detach().cpu().numpy()
                depth = raw['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()
                K = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
                p, q = self._v21_cam_calib[name]
                # 保存原始画面，让用户真正看到机器人和孔；记录与学生观察同源。
                stem = f'{name}_{idx:05d}'
                rgb8 = np.asarray(rgb[...,:3])
                if rgb8.dtype != np.uint8:
                    rgb8 = np.clip(rgb8, 0, 255).astype(np.uint8)
                raw_dir = root/'frames'
                raw_dir.mkdir(parents=True, exist_ok=True)
                if idx == 1 or idx % 25 == 0:
                    Image.fromarray(rgb8).save(raw_dir/(stem+'.jpg'), quality=85)
                if idx == 1 or idx % 50 == 0:
                    np.savez_compressed(raw_dir/(stem+'.npz'),
                        rgb=rgb8,depth=depth,K=K,cam_pos=p,cam_quat=q)
                peg, hole, stats = diagnostic_masks(rgb8, depth)
                overlay = np.zeros_like(rgb8)
                overlay[peg]=(255,120,0)
                overlay[hole]=(20,100,255)
                if idx == 1 or idx % 50 == 0:
                    Image.fromarray(overlay).save(raw_dir/(stem+'_masks.png'))
                frames[name] = stats
                packets[name] = (rgb8,depth,K,p,q)
            except Exception as exc:
                _append_jsonl(root/'vision_errors.jsonl',
                              {'step':idx,'camera':name,'error':f'{type(exc).__name__}: {exc}'})
        # 学生观测：只允许使用真实渲染图像、深度、相机内外参和固定几何常数。
        rec = {'step':idx, 'sensor':'actual_rendered_rgbd',
               'camera_stats':frames, 'valid':False}
        if len(packets)>=2:
            try:
                # link8法兰位置是机器人本体运动学信息，不是棒的真实安装偏差。
                # 仅构造名义空间ROI，不能代替深度相机对棒的实际测量。
                linkid=self.link8_ids[0]
                link_pos=self.robot.data.body_link_pos_w[0,linkid].detach().cpu().numpy()
                wxyz=self.robot.data.body_link_quat_w[0,linkid].detach().cpu().numpy()
                link_rot=Rotation.from_quat([wxyz[1],wxyz[2],wxyz[3],wxyz[0]]).as_matrix()
                nominal_tip=link_pos+link_rot@self._v23_nominal_tip_local
                nominal_axis=link_rot@self._v23_nominal_axis_local
                rel,quality=self._v23_visual_state.observe(packets,nominal_tip,nominal_axis,link_pos,link_rot,idx)
                self._v26_latest_visual = {'relative_pose5_estimated':np.asarray(rel,dtype=float).copy(),
                    'quality':quality, 'step':int(idx)}
                rec.update({'valid':True,'relative_pose5_estimated':rel.tolist(),
                            'fit_quality':quality,
                            'sensor_estimate_status':quality['sensor_estimate_status'],
                            'raw_peg_visible_this_step':quality['raw_peg_visible_this_step'],
                            'mount_freshness_control_steps':quality['mount_freshness_control_steps']})
            except (VisualInvalid,ValueError,RuntimeError) as exc:
                rec['error'] = f'{type(exc).__name__}: {exc}'
        else:
            rec['error'] = '有效相机少于2台，不回填真值'
        _append_jsonl(root/'student_camera_observations.jsonl', rec)
        print('[V25_VISION_SAMPLE] '+json.dumps({
            'step':idx,'valid':rec['valid'],
            'error':str(rec.get('error',''))[:240],
            'camera_pixel_counts':{k:(v.get('peg_mask_pixels'),v.get('hole_mask_pixels'))
                                   for k,v in frames.items()},
        },ensure_ascii=False),flush=True)
        # 真值审计和学生传感数据**独立文件**；视觉估计函数本身从不接收真值。
        # 即使视觉失效，也记录真实距离段，避免只统计有效帧造成选择性偏差。
        try:
            oracle = self.measure_peg_tip_oracle()
            true_tip = np.asarray(oracle['tip_world_xyz_m'], dtype=float)
            true_axis = np.asarray(oracle['peg_axis_world_xyz'], dtype=float)
            true_axis /= np.linalg.norm(true_axis)
            true_hole = np.asarray(self._v16_hole, dtype=float)
            true_rel = true_hole - true_tip
            true_radial_mm = float(np.linalg.norm(true_rel[:2])*1000)
            true_height_above_mm = float((true_tip[2]-true_hole[2])*1000)
            # 独立统计粗定位与近孔口（不把真值用于检测器，也不用于专家动作）
            bin_name = 'near_mouth' if abs(true_height_above_mm) <= 50. and true_radial_mm <= 10. else 'far_from_mouth'
            audit = {
                'step': idx, 'phase_bin_offline': bin_name,
                'camera_valid': bool(rec['valid']),
                'true_radial_mm_offline': true_radial_mm,
                'true_height_above_mm_offline': true_height_above_mm,
                'rotation_perturbation_tested': float(os.environ.get('ENO_V25_ROT_MAX_DEG','0'))>0.0,
            }
            if rec['valid']:
                pred = np.asarray(rec['relative_pose5_estimated'], dtype=float)
                audit['xy_error_mm'] = float(np.linalg.norm(pred[:2]-true_rel[:2])*1000)
                audit['z_error_mm'] = float(abs(pred[2]-true_rel[2])*1000)
                pred_axis = np.array([pred[3], pred[4],
                    -np.sqrt(max(0., 1. - pred[3]**2 - pred[4]**2))])
                pred_axis /= np.linalg.norm(pred_axis)
                audit['axis_angle_error_deg'] = float(math.degrees(math.acos(
                    np.clip(float(np.dot(true_axis, pred_axis)), -1., 1.))))
                audit['estimated_xy_m'] = pred[:2].tolist()
                audit['true_xy_m_offline'] = true_rel[:2].tolist()
            else:
                audit['invalid_reason'] = rec.get('error', '未知')
            _append_jsonl(root/'OFFLINE_ONLY_pose_errors.jsonl', audit)
        except Exception as exc:
            _append_jsonl(root/'OFFLINE_ONLY_pose_errors.jsonl', {
                'step': idx, 'truth_error': f'{type(exc).__name__}: {exc}'})

    def step_with_passive_camera(self, actions):
        # 原始动作、原始仿真step、原始返回值：一行不修改。
        result = original_step(self, actions)
        try:
            if hasattr(self, '_v21_cameras'):
                record_after_step(self)
        except Exception as exc:
            _append_jsonl(root/'vision_errors.jsonl',{
                'step':int(getattr(self,'_v21_step_count',-1)),
                'error':f'{type(exc).__name__}: {exc}',
                'traceback':traceback.format_exc(limit=4),
            })
            print('[V21_VISION_CAPTURE_WARN] '+str(exc),flush=True)
        return result

    # 明确验证没有重复注入、没有覆盖教师函数。
    if getattr(base_env_cls, '_v23_hook_installed', False):
        raise RuntimeError('重复安装v21旁路相机钩子')
    base_env_cls._setup_scene = setup_with_passive_camera
    step_owner_cls.step = step_with_passive_camera
    base_env_cls._v23_hook_installed = True
    print('[V26_RGBD_INSTALLED] cameras + timestamped sensor observations',flush=True)
