#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在原 v17 RL 生成结果的 main() 前注入视觉钩子 + 真实旋转场景。

本模块仅改变传感器和环境随机棒角度，不复制原专家，且禁止覆盖已生成代码。
"""
import argparse,ast
from pathlib import Path
ROTATION_WRAPPER = "\n# v25: 在物理场景创建之前对真实棒的安装施加受控X/Y旋转扰动。\n# 这里只是场景域随机化；原v17.4专家控制函数不变，仍从旧IK接口执行。\nimport os as _v25_os\nimport numpy as _v25_np\n_V25_ROT_MAX_DEG = float(_v25_os.environ.get('ENO_V25_ROT_MAX_DEG', '1.0'))\nif not (0.0 <= _V25_ROT_MAX_DEG <= 3.0):\n    raise RuntimeError('ENO_V25_ROT_MAX_DEG必须为0..3度，非法则停止而非默默忽略')\n_v25_original_make_peg = make_peg_on_link8\ndef make_peg_on_link8(stage, episode):\n    tip,axis = _v25_original_make_peg(stage, episode)\n    if _V25_ROT_MAX_DEG==0.0:\n        print('[V25_MOUNT_ROTATION] none', flush=True)\n        return tip,axis\n    from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom\n    rng=_v25_np.random.default_rng(int(args.seed)+232309)\n    rx,ry=[float(x) for x in rng.uniform(-_V25_ROT_MAX_DEG,_V25_ROT_MAX_DEG,size=2)]\n    mount=stage.GetPrimAtPath(MOUNT_PATH)\n    rod=stage.GetPrimAtPath(PEG_PATH)\n    link=stage.GetPrimAtPath(LINK8_PATH)\n    if not mount.IsValid() or not rod.IsValid() or not link.IsValid():\n        raise RuntimeError('v25 mount/peg/link8 USD无效')\n    ops=_UsdGeom.Xformable(mount).GetOrderedXformOps()\n    orient_ops=[op for op in ops if op.GetOpType()==_UsdGeom.XformOp.TypeOrient]\n    if len(orient_ops)!=1:raise RuntimeError('v25未找到唯一安装旋转op')\n    op=orient_ops[0];old=op.Get()\n    oldq=_Gf.Quatd(float(old.GetReal()),_Gf.Vec3d(*[float(t) for t in old.GetImaginary()]))\n    baseline=_Gf.Rotation(oldq)\n    perturb=(_Gf.Rotation(_Gf.Vec3d(1.,0.,0.),rx)\n            * _Gf.Rotation(_Gf.Vec3d(0.,1.,0.),ry))\n    nq=(baseline*perturb).GetQuat()\n    op.Set(_Gf.Quatf(float(nq.GetReal()), _Gf.Vec3f(*[float(t) for t in nq.GetImaginary()])))\n    xcache=_UsdGeom.XformCache(_Usd.TimeCode.Default())\n    inv=xcache.GetLocalToWorldTransform(link).GetInverse()\n    T=xcache.GetLocalToWorldTransform(rod)\n    zlen=float(geom.PEG_LENGTH_M)\n    tw=T.Transform(_Gf.Vec3d(0.,0.,-zlen*.5))\n    hw=T.Transform(_Gf.Vec3d(0.,0., zlen*.5))\n    tl=inv.Transform(tw);hl=inv.Transform(hw)\n    new_tip=_v25_np.array([float(tl[i]) for i in range(3)])\n    vec=new_tip-_v25_np.array([float(hl[i]) for i in range(3)])\n    new_axis=vec/_v25_np.linalg.norm(vec)\n    print('[V25_MOUNT_ROTATION] '+__import__('json').dumps({'seed':int(args.seed),\n        'roll_deg_scene_truth_only':rx,'pitch_deg_scene_truth_only':ry,\n        'scope':'USD geometry and teacher ONLY; excluded from student observations'}), flush=True)\n    return new_tip,new_axis\n"
HOOK = """
# v26: RGB-D视觉采样，学生控制仅使用视觉估计的工具安装参数。
from v26_camera_hook import install_v26_camera_hook
from v26_runtime import install_v26_student_ik
install_v26_camera_hook(FR3PegHoleSceneEnv, DirectRLEnv, sim_utils,
                        V16_HOLE_XY_M, ENV_ROOT)
install_v26_student_ik(V176ContactEnv)
"""
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('禁止覆盖生成代码:'+str(a.out))
    code=a.base.read_text(encoding='utf8')
    anchor='\ndef main():\n'
    if code.count(anchor)!=1:raise RuntimeError('原RL生成代码main()定位不唯一')
    new=code.replace(anchor,'\n'+ROTATION_WRAPPER+'\n'+HOOK+anchor,1)
    ast.parse(new,filename=str(a.out))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(new,encoding='utf8')
    print('[V26_GENERATED] '+str(a.out),flush=True)
if __name__=='__main__':main()
