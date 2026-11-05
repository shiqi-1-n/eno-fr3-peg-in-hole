#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在已验证的 v17.6.1 源码上生成RL rollout；基础项目与权重保持不变。"""
from __future__ import annotations
import ast,argparse,py_compile,tempfile
from pathlib import Path
from prepare_v17_6_contact_trial import build as build_v176

def once(s,a,b,label):
    n=s.count(a)
    if n!=1:raise RuntimeError(f'{label}: 找到{n}个锚点，要求恰好1个，拒绝错版修改')
    return s.replace(a,b,1)

def build(src,dst,teacher,eval_ext,full_ext,contact_ext,rl_ext):
    if dst.exists():raise FileExistsError('拒绝覆盖旧生成文件: '+str(dst))
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp)/'v176.py'
        build_v176(src,base,teacher,eval_ext,full_ext,contact_ext)
        s=base.read_text(encoding='utf-8')
    s=once(s,"'v17_full_bc_eval','v17_contact_trial'), default=",
           "'v17_full_bc_eval','v17_contact_trial','v17_residual_curriculum'), default=",'RL CLI模式')
    s=once(s,"parser.add_argument('--v17-force-limit-n',type=float,default=30.0,help='试插力保护阈值，单位N')",
        "parser.add_argument('--v17-force-limit-n',type=float,default=30.0,help='试插力保护阈值，单位N')\n"
        "parser.add_argument('--v17-rl-ckpt',type=Path,default=None,help='残差PPO当前冻结网络')",'RL checkpoint参数')
    s=once(s,'\ndef main():\n','\n'+rl_ext.read_text(encoding='utf-8')+'\n\ndef main():\n','插入RL rollout')
    s=once(s,"    if args.mode=='v17_contact_trial':\n        return v17_6_contact_trial_main()",
           "    if args.mode=='v17_contact_trial':\n        return v17_6_contact_trial_main()\n"
           "    if args.mode=='v17_residual_curriculum':\n        return v17_9_residual_rollout_main()",'RL分发')
    ast.parse(s,filename=str(dst))
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(s,encoding='utf-8')
    py_compile.compile(str(dst),doraise=True)
    print('[V17_9_RL_GENERATED]',dst,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','target','teacher-extension','eval-extension','full-extension','contact-extension','rl-extension'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    build(a.source,a.target,a.teacher_extension,a.eval_extension,
          a.full_extension,a.contact_extension,a.rl_extension)
