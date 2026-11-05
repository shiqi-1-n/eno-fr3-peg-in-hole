#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v31.6.1 冻结评估驱动：只编排既有runner，不修改IsaacLab/PPO/环境/成功判据。

命令：
  python3 -S v3161_frozen_eval30/eval_v3161.py prepare fresh
  python3 -S v3161_frozen_eval30/eval_v3161.py run fresh
  python3 -S v3161_frozen_eval30/eval_v3161.py report fresh

重要约束：
- 所有种子在 seed_plan.json 中预注册，不因失败筛选/替换。
- 先做历史两条视觉失效种子专项测试，不将其算作独立盲测。
- 已存在任何同名仿真输出，均拒绝覆盖。不自动重新跑失败案例。
- 模型状态必须是 v31.6.1 prelock axis_only / ZERO_PPO_RESIDUAL，不能称之为训练后的RL。
- 几何成功仅按原结果JSON success字段，不能用max_depth超阈值冒充成功。
- 真实接触、传感力学的可信度另行审计；这是当前版本的已知验收缺口。
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Any, List, Optional, Tuple

HERE=Path(__file__).resolve().parent
ROOT_DEFAULT=Path('/home/dgx/eno-isaac')
CONFIG_FILES=(
    'scripts/v31_6_1_prelock_eval/run_v26_rollout.sh',
    'scripts/v31_6_1_prelock_eval/v25_peg_vision.py',
    'scripts/v31_6_1_prelock_eval/v26_runtime.py',
    'scripts/v31_6_1_prelock_eval/v26_rl_extension.txt',
    'scripts/50_phase05b2b_v16_13_online_evidence.py',
    'output/v26_1_visual_ppo_models/v261_first2/round_000.pt',
    'output/v17_4_bc_models/coarse_bc200_1/bc_actor_best.pt',
    # v31.6.1 runner会调用的上游代码一并SHA冻结，避免只冻结视觉但场景悄悄变化。
    'scripts/prepare_v17_9_rl.py',
    'scripts/prepare_v17_6_contact_trial.py',
    'scripts/prepare_v17_5_full_bc_eval.py',
    'scripts/prepare_v17_4_spawn_consistent.py',
    'scripts/prepare_v17_3_coarse.py',
    'scripts/prepare_v17_2_sensor_capture.py',
    'scripts/prepare_v17_random_teacher.py',
    'scripts/prepare_v17_scene_probe.py',
    'scripts/v17_teacher_extension.txt',
    'scripts/v17_3_eval_extension.txt',
    'scripts/v17_5_full_eval_extension.txt',
    'scripts/v17_6_1_hold_and_probe_extension.txt',
    'scripts/v17_17_hole_refine.py',
    'scripts/v17_3_hole_sensor.py',
    'scripts/train_v17_3_coarse_bc.py',
    'scripts/v31_6_1_prelock_eval/v26_camera_hook.py',
    'scripts/v31_6_1_prelock_eval/v26_ppo.py',
    'scripts/v31_6_1_prelock_eval/prepare_v26.py',
)


def json_load(path:Path)->Any:
    with path.open('r',encoding='utf8') as fp: return json.load(fp)


def write_json_new(path:Path, obj:Any)->None:
    # 'x' 原子地拒绝覆盖已有文件。不存在“静默重新开始一轮评估”。
    with path.open('x',encoding='utf8') as fp:
        json.dump(obj,fp,ensure_ascii=False,indent=2,allow_nan=False)
        fp.write('\n')


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()


def sha_sources(root:Path)->Dict[str,str]:
    # 启动前检查所有关键依赖是否存在；独立目录不需要重新安装。
    missing=[s for s in CONFIG_FILES if not (root/s).is_file()]
    if missing: raise RuntimeError('找不到v31.6.1源文件/权重。请先安装v31.6.1：\n'+'\n'.join(missing))
    return {s:sha(root/s) for s in CONFIG_FILES}


def plan(): return json_load(HERE/'seed_plan.json')


def safe_plan(p:dict, suite:str)->Tuple[str,List[int],Dict[str,str]]:
    group=p[suite]['group']; seeds=p[suite]['seeds']; env=p['frozen_config']
    if not re.fullmatch(r'[A-Za-z0-9_-]{5,55}',group): raise ValueError('group不安全')
    if not isinstance(seeds,list) or len(seeds)!=(2 if suite=='pilot' else 30):
        raise ValueError('预注册seed个数异常')
    if not all(type(i) is int and 100000000<=i<=999999999 for i in seeds):
        raise ValueError('种子必须是9位整数')
    if len(set(seeds))!=len(seeds):raise ValueError('重复seed')
    if set(p['pilot']['seeds'])&set(p['fresh']['seeds']):raise ValueError('pilot和fresh有重叠')
    original=json_load(HERE/'original_v315_seed_plan.json')
    if seeds != original[suite]['seeds']:raise ValueError('随机种子与原v31.5事先固定的清单不一致')
    if p.get('source_plan_sha256')!=sha(HERE/'original_v315_seed_plan.json'):raise ValueError('原始清单SHA不匹配')
    if env.get('ENO_V315_LOCK_VARIANT')!='axis_only':raise ValueError('错误的变体')
    if env.get('ENO_V316_PRELOCK')!='1':raise ValueError('预锁定开关必须为1')
    if env.get('ENO_V1717_EPISODE_STEPS')!='1400':raise ValueError('错误的步数')
    if env.get('ENO_V17_COARSE_MODEL')!='/home/dgx/eno-isaac/output/v17_4_bc_models/coarse_bc200_1/bc_actor_best.pt':raise ValueError('粗定位BC权重路径不一致')
    return group,seeds,env


def tag_for(suite:str,seed:int)->str:
    return ('v3161_valP_' if suite=='pilot' else 'v3161_valF_')+str(seed)


def group_dir(root:Path,group:str)->Path:
    return root/'output'/'v3161_frozen_eval'/group


REGRESSION_TAGS = {
    202611125:'v3161_p125_gateA',
    202611140:'v3161_p140_gateA',
    202611141:'v3161_p141_gateA',
    202611150:'v3161_p150_gateA',
}

def regression_gate(root:Path)->None:
    """避免未经成功回归就开始盲测；只读历史结果，不重新运行。"""
    problems=[]
    for seed,tag in REGRESSION_TAGS.items():
        fp=root/'output'/'v26_2_zero_ppo_rollouts'/tag/'session'/'v17_7_residual_rollout.json'
        log=root/'output'/'v26_2_zero_ppo_rollouts'/tag/'run.log'
        if not fp.exists():
            problems.append(f'{seed}: 缺少结果JSON {fp}')
            continue
        try:
            outcome=json_load(fp)
            info=event_info(log)
        except (OSError,ValueError) as e:
            problems.append(f'{seed}: 读取失败 {e}')
            continue
        if not (outcome.get('seed')==seed and outcome.get('status')=='COMPLETE'
                and outcome.get('success') is True and outcome.get('stop_reason')=='success'
                and outcome.get('policy_ablation')=='ZERO_PPO_RESIDUAL'
                and outcome.get('ppo_action_applied') is False
                and outcome.get('policy_updates')==0
                and info['switch_count']==1 and info['prelock_count']==1
                and info['lock_seen']):
            problems.append(f'{seed}: 几何成功/zeroPPO/提前锁定/切换证据不全')
        else:
            print(f'[V3161_REGRESSION_PASS] seed={seed} steps={outcome.get("steps_total")} tag={tag}')
    if problems:
        raise RuntimeError('历史回归尚未完成，不能启动新的盲测：\n'+'\n'.join(problems))

def prepare(root:Path,suite:str,p:dict)->Path:
    group,seeds,config=safe_plan(p,suite)
    regression_gate(root)
    g=group_dir(root,group)
    if g.exists():
        raise RuntimeError('同名评估组已经存在，拒绝覆盖：'+str(g)+'\n继续原组请用 run；只读汇总请用 report。')
    # 先查已有轨迹标签，避免不属于当前组的历史记录被当作本轮样本。
    out=root/'output'/'v26_2_zero_ppo_rollouts'
    exists=[tag_for(suite,s) for s in seeds if (out/tag_for(suite,s)).exists()]
    if suite=='fresh':
        known=[f'v31_valF_{i}' for i in seeds if (out/f'v31_valF_{i}').exists()]
        if known:raise RuntimeError('旧v31.5已测试以下预选种子，不能称为新盲测：'+', '.join(known))
    if exists:raise RuntimeError('预注册标签已存在，拒绝启动：'+', '.join(exists))
    hashes=sha_sources(root)
    g.mkdir(parents=True,exist_ok=False)
    manifest={
        'schema':'v3161_frozen_eval_01',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'suite':suite,
        'group':group,
        'plan_sha256':sha(HERE/'seed_plan.json'),
        'seed_plan':p['plan_version'],
        'seeds':seeds,
        'tags':[tag_for(suite,s) for s in seeds],
        'frozen_config':config,
        'source_sha256':hashes,
        'official_test':suite=='fresh',
        'notes_cn':'pilot不合并入盲测；运行期间不允许更改源码、模型权重、控制参数或成功阈值。',
    }
    write_json_new(g/'manifest.json',manifest)
    print('[V3161_EVAL_PREPARED]',group,'n=',len(seeds))
    print('[V3161_EVAL_MANIFEST]',g/'manifest.json')
    print('[V3161_EVAL_SEED_PLAN_SHA256]',manifest['plan_sha256'])
    return g


def check_manifest(root:Path,suite:str,p:dict,verify_source=True)->Tuple[Path,dict]:
    group,seeds,config=safe_plan(p,suite)
    g=group_dir(root,group)
    mf=g/'manifest.json'
    if not mf.exists():raise RuntimeError('缺少manifest，请先执行 prepare '+suite)
    m=json_load(mf)
    if (m.get('suite')!=suite or m.get('seeds')!=seeds
            or m.get('frozen_config')!=config
            or m.get('tags')!=[tag_for(suite,s) for s in seeds]
            or m.get('plan_sha256')!=sha(HERE/'seed_plan.json')):
        raise RuntimeError('manifest与原始种子/配置不一致，拒绝继续')
    if verify_source:
        current=sha_sources(root)
        changed=[key for key in current if current[key]!=m['source_sha256'].get(key)]
        if changed:
            raise RuntimeError('冻结后的源码/权重已变化！拒绝继续：\n'+'\n'.join(changed))
    return g,m


def read_outcome(root:Path,tag:str)->Optional[dict]:
    fp=root/'output'/'v26_2_zero_ppo_rollouts'/tag/'session'/'v17_7_residual_rollout.json'
    if not fp.is_file():return None
    try:return json_load(fp)
    except (OSError,json.JSONDecodeError):return None


def event_info(logpath:Path)->dict:
    res={'switch_count':0,'lock_reject_count':0,'lock_baseline_count':0,
         'last_switch':None,'lock_seen':False,'sensor_error_hint':False,'prelock_count':0}
    if not logpath.is_file():return res
    # 以run.log而非终端转储为准。只读日志，不把物理真值用于决策。
    with logpath.open('r',encoding='utf8',errors='replace') as f:
        for line in f:
            if '[V315_LOCK_SWITCH]' in line:
                res['switch_count']+=1
                try:res['last_switch']=json.loads(line.split('[V315_LOCK_SWITCH]',1)[1].strip())
                except json.JSONDecodeError: pass
            if '[V315_LOCK_REJECT]' in line:res['lock_reject_count']+=1
            if '[V315_LOCK_BASELINE]' in line:res['lock_baseline_count']+=1
            if '[V261_VISUAL_MOUNT_LOCK]' in line:res['lock_seen']=True
            if '[V316_EARLY_CALIBRATION_LOCK]' in line:res['prelock_count']+=1
            if ('SENSOR_INVALID' in line or 'VisualInvalid:' in line):res['sensor_error_hint']=True
    return res


def result_row(root:Path,suite:str,seed:int)->dict:
    tag=tag_for(suite,seed)
    base=root/'output'/'v26_2_zero_ppo_rollouts'/tag
    o=read_outcome(root,tag)
    loginfo=event_info(base/'run.log')
    row={'seed':seed,'tag':tag,'output_exists':base.exists(),
         'result_exists':o is not None, 'status':'MISSING','success':None,
         'steps_total':None,'stop_reason':None,'max_true_depth_mm_offline':None,
         'min_true_radial_mm_offline':None,'max_contact_force_N':None,
         'contact_feedback_substep_verified':None,
         'policy_ablation':None,'ppo_action_applied':None,'policy_updates':None,
         'peg_gate_blocked_steps':None,'mount_lock_step':None,
         'trustworthy_record':False,'integrity_errors':[],**loginfo}
    if not base.exists():return row
    if o is None:
        row['status']='NO_RESULT_JSON';return row
    for key in ('status','success','steps_total','stop_reason','max_true_depth_mm_offline',
                'min_true_radial_mm_offline','max_contact_force_N',
                'contact_feedback_substep_verified','policy_ablation','ppo_action_applied',
                'policy_updates','peg_gate_blocked_steps','mount_lock_step'):
        row[key]=o.get(key)
    errs=[]
    if o.get('status')!='COMPLETE':errs.append('status不是COMPLETE')
    if o.get('seed')!=seed:errs.append('seed与预注册不一致')
    if type(o.get('success')) is not bool:errs.append('success不是布尔值')
    if o.get('success') is True and o.get('stop_reason')!='success':
        errs.append('success=true但stop_reason不为success')
    if o.get('policy_ablation')!='ZERO_PPO_RESIDUAL':errs.append('策略消融标记不一致')
    if o.get('ppo_action_applied') is not False:errs.append('PPO动作标记不一致')
    if o.get('policy_updates')!=0:errs.append('存在策略更新，已不属于本对照')
    if loginfo['switch_count']>1:errs.append('重复执行安装参数切换')
    if loginfo['prelock_count']>1:errs.append('重复提前锁定')
    sw=loginfo['last_switch']
    if sw is not None:
        if sw.get('variant')!='axis_only':errs.append('在线切换不是axis_only')
        if sw.get('truth_used') is not False:errs.append('在线切换报告使用了真值')
        if sw.get('applied_tip_jump_mm') is None or abs(sw['applied_tip_jump_mm'])>0.000001:
            errs.append('棒尖参数跳变不为零')
    if o.get('success') and not loginfo['lock_seen']:
        errs.append('成功轨迹未找到视觉锁定日志')
    row['integrity_errors']=errs
    row['trustworthy_record']=not errs
    return row


def wilson(success:int,n:int)->Optional[List[float]]:
    if n<1:return None
    z=1.959963984540054
    p=success/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z/den*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round(max(0,mid-half),5),round(min(1,mid+half),5)]


def report(root:Path,suite:str,p:dict)->dict:
    g,m=check_manifest(root,suite,p,verify_source=False)
    rows=[result_row(root,suite,seed) for seed in m['seeds']]
    n=len(rows)
    valid=[r for r in rows if r['trustworthy_record']]
    success=[r for r in valid if r['success'] is True]
    missing=[r for r in rows if not r['result_exists']]
    errors=[r for r in rows if r['integrity_errors']]
    contact_verified=[r for r in rows if r['contact_feedback_substep_verified'] is True]
    all_done=(len(valid)==n and len(missing)==0 and len(errors)==0)
    out={
        'schema':'v3161_frozen_report_01','suite':suite,'group':m['group'],
        'is_independent_fresh_suite':suite=='fresh',
        'planned_n':n,'valid_completed_n':len(valid),'valid_geometry_success_n':len(success),
        'valid_geometry_failure_n':len(valid)-len(success),
        'missing_or_invalid_json_n':len(missing),'integrity_error_n':len(errors),
        'all_completed_and_consistent':all_done,
        'conservative_success_fraction':round(len(success)/n,6),
        'complete_only_fraction_NOT_OFFICIAL':round(len(success)/len(valid),6) if valid else None,
        'wilson95_for_conservative_rate':wilson(len(success),n),
        'meets_observed_90pct_if_complete':all_done and len(success)/n>=.9,
        'physx_contact_verified_episodes_n':len(contact_verified),
        'full_teacher_task_accepted':False,
        'trained_policy_participated':False,
        'interpretation_cn':('这是正式盲测几何统计，不是证明真实物理成功率≥90%。'
                             if suite=='fresh' else '这是历史问题专项回归，禁止混入fresh盲测统计。'),
        'records':rows,
    }
    fp=g/'report.json'
    # 报告可以重复生成覆盖（不会改变任何原始实验；只读原始JSON）。
    with fp.open('w',encoding='utf8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('[V3161_EVAL_REPORT] '+json.dumps({k:v for k,v in out.items() if k not in ('records','interpretation_cn')},ensure_ascii=False))
    print('[V3161_EVAL_REPORT_FILE]',fp)
    for r in rows:
        print('[V3161_EVAL_CASE] '+json.dumps({k:r[k] for k in (
            'seed','status','success','steps_total','stop_reason','switch_count',
            'lock_reject_count','prelock_count','peg_gate_blocked_steps','max_true_depth_mm_offline',
            'max_contact_force_N','trustworthy_record','integrity_errors')},ensure_ascii=False))
    if suite=='fresh' and not all_done:
        print('[V3161_EVAL_CAUTION] 部分轨迹未完成或完整性审计不通过，不能宣称本轮正式成功率已确认。')
    if not contact_verified:
        print('[V3161_EVAL_CAUTION] 未发现已验证的PhysX接触；即使全部几何成功，也不能称为真实物理插入。')
    return out


def append_event(g:Path,event:dict)->None:
    with (g/'process_events.jsonl').open('a',encoding='utf8') as fp:
        fp.write(json.dumps(event,ensure_ascii=False)+'\n')
        fp.flush();os.fsync(fp.fileno())


def run(root:Path,suite:str,p:dict)->None:
    g,m=check_manifest(root,suite,p,verify_source=True)
    regression_gate(root)
    runner=root/'scripts/v31_6_1_prelock_eval/run_v26_rollout.sh'
    checkpoint=root/'output/v26_1_visual_ppo_models/v261_first2/round_000.pt'
    env=os.environ.copy()
    # 明确覆盖关键环境变量，避免本机遗留训练/测试export影响统计口径。
    env.update(m['frozen_config'])
    env['ENO_HOME']=str(root)
    print('[V3161_EVAL_RUN] suite=',suite,'group=',m['group'],'number=',len(m['seeds']),flush=True)
    print('[V3161_EVAL_RUN] 每条一个独立物理进程；不覆盖、不重试同名结果。',flush=True)
    for index,seed in enumerate(m['seeds'],1):
        # 每条开始前重复核对源代码，防止中途修改制造不可复现的混合报告。
        check_manifest(root,suite,p,verify_source=True)
        tag=tag_for(suite,seed)
        out=root/'output'/'v26_2_zero_ppo_rollouts'/tag
        if out.exists():
            existing=result_row(root,suite,seed)
            mark='COMPLETE' if existing['trustworthy_record'] else 'INCOMPLETE/INVALID'
            print(f'[V3161_EVAL_SKIP_EXISTING] {index}/{len(m["seeds"])} {seed} {mark} ; 原始目录禁止覆写',flush=True)
            continue
        print(f'[V3161_EVAL_EPISODE_START] {index}/{len(m["seeds"])} seed={seed} tag={tag}',flush=True)
        ev={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'event':'START','suite':suite,'seed':seed,'tag':tag}
        append_event(g,ev)
        command=['bash',str(runner),tag,str(seed),str(checkpoint)]
        with (g/'launcher.log').open('a',encoding='utf8') as log:
            log.write('\n[V3161_EVAL_EPISODE_START] '+json.dumps(ev,ensure_ascii=False)+'\n');log.flush()
            proc=subprocess.run(command,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT)
        row=result_row(root,suite,seed)
        final={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
               'event':'EXIT','suite':suite,'seed':seed,'tag':tag,
               'returncode':proc.returncode,'result_exists':row['result_exists'],
               'status':row['status'],'success':row['success'],
               'trustworthy_record':row['trustworthy_record']}
        append_event(g,final)
        print('[V3161_EVAL_EPISODE_END] '+json.dumps(final,ensure_ascii=False),flush=True)
        if proc.returncode!=0 or not row['trustworthy_record']:
            print('[V3161_EVAL_RUN_STOP] 发生程序错误/结果无效；已保留现场并停止本组。'
                  '请先 report 检查；严禁删除或重跑同名轨迹以提升成绩。',flush=True)
            break
    final_report=report(root,suite,p)
    return final_report["all_completed_and_consistent"]


def main():
    parser=argparse.ArgumentParser(description='FR3 v31.6.1 预注册、冻结、无重试单孔评估')
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('suite',choices=['fresh'])
    parser.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    a=parser.parse_args();root=a.root.resolve();p=plan()
    try:
        if a.command=='prepare':prepare(root,a.suite,p)
        elif a.command=='run':
            ok=run(root,a.suite,p)
            if not ok: return 4
        else:report(root,a.suite,p)
    except (OSError,ValueError,RuntimeError,KeyError) as err:
        print('[V3161_EVAL_ERROR]',str(err),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
