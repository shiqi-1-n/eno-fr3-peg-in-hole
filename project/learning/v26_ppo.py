#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v26：棒/孔射线相对位置 + 对中优先 PPO 更新器。可以单独运行，不需要启动 PhysX。

一条训练 trajectory 对应一个真实独立 Isaac Sim 进程/一个孔位+一组真实挂载偏差。
Actor 始终只接收观测；真实棒尖真值仅在仿真环境侧构造 reward/完成判据。

使用 Tanh-squashed Gaussian；在采样和 PPO 更新时对数概率计算保持一致。
PPO 使用广义优势估计 GAE、clip surrogate、value loss、梯度裁剪。
注意：这只是可验证的训练实现，不能由权重文件存在推断插孔成功。
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

OBS_DIM = 37
ACTION_DIM = 3
SCHEMA = 'v26_1_obs37_rgbd_relpose5_fk_locked_v1'


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),
                                   nn.Linear(128,128),nn.Tanh(),nn.Linear(128,ACTION_DIM))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),
                                    nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.2))
        # 初始动作均值=0：PPO 先从已验证的低速基线的「小残差」探索，而不是随机大跳。
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, x):
        mu = self.actor(x)
        std = self.log_std.clamp(-2.7,-0.4).exp().expand_as(mu)
        return mu, std, self.critic(x).squeeze(-1)

    def act(self, obs, deterministic=False, action_mask=None):
        """返回[-1,1]^3残差以及**真正可影响物理的动作维度**的logp。

        力超限但仍在低阈值保护范围时，安全层会暂时禁止XY动作。
        对于被安全层屏蔽的维度，PPO不得错误地给它分配动作优势（credit assignment）。
        注意：mask只由当前可观测接触力和名义深度决定，绝非特权真值。
        """
        x = torch.as_tensor(obs,dtype=torch.float32).reshape(1,OBS_DIM)
        mask=torch.ones((1,ACTION_DIM),dtype=torch.float32) if action_mask is None else \
            torch.as_tensor(action_mask,dtype=torch.float32).reshape(1,ACTION_DIM)
        with torch.no_grad():
            mu,std,value=self.forward(x)
            dist=Normal(mu,std)
            pre=mu if deterministic else dist.sample()
            a=torch.tanh(pre)
            per_dim=dist.log_prob(pre)-torch.log(1-a.square()+1e-6)
            logp=(per_dim*mask).sum(-1)
        return a[0].tolist(),float(logp[0]),float(value[0])

    def score_actions(self,x,action,action_mask=None):
        mu,std,val=self.forward(x)
        clipped=action.clamp(-0.999999,0.999999)
        z=torch.atanh(clipped)
        dist=Normal(mu,std)
        mask=torch.ones_like(action) if action_mask is None else action_mask
        per_dim=dist.log_prob(z)-torch.log(1-clipped.square()+1e-6)
        logp=(per_dim*mask).sum(-1)
        entropy=(dist.entropy()*mask).sum(-1)  # 高斯熵近似；不表示被安全屏蔽维度的探索熵。
        return logp,entropy,val


def save_ckpt(path,net,opt,updates,episodes):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save({'schema':SCHEMA,'input_dim':OBS_DIM,'action_dim':ACTION_DIM,
                'model_state_dict':net.state_dict(), 'optimizer':opt.state_dict(),
                'updates':int(updates),'episodes':int(episodes)},tmp)
    os.replace(tmp,path)


def load_ckpt(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    if c.get('schema')!=SCHEMA or c.get('input_dim')!=OBS_DIM or c.get('action_dim')!=ACTION_DIM:
        raise ValueError('v17.13 PPO checkpoint架构不匹配，拒绝误用v17.4/5 BC权重')
    net=ActorCritic();net.load_state_dict(c['model_state_dict']);net.eval()
    return net,c


def update(ckpt_path,paths,save_path,epochs=4,minibatch=64,seed=0):
    """每轮只使用同一 frozen checkpoint 采集的 on-policy rollouts，拒绝离策略混训。"""
    net,c=load_ckpt(ckpt_path);net.train()
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    opt.load_state_dict(c['optimizer'])
    path_objs=[Path(p) for p in paths]
    if not path_objs or any(not p.is_file() for p in path_objs):
        raise RuntimeError('缺少完整的rollout JSON，禁止更新')
    transitions=[]; advantages=[]; returns=[]; stats=[]
    for path in path_objs:
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('schema')!=SCHEMA or r.get('status')!='COMPLETE' or r.get('deterministic_eval',False):
            raise RuntimeError(f'拒绝状态异常或schema错误轨迹: {path}')
        if r.get('policy_updates')!=c['updates']:
            raise RuntimeError(f'轨迹并非由同一策略版本产生: {path} rollout={r.get("policy_updates")} model={c["updates"]}')
        steps=r['steps'];T=len(steps)
        if T<2:raise RuntimeError(f'轨迹太短，拒绝更新: {path}')
        adv=np.zeros(T,np.float32)
        last=float(r.get('bootstrap_value',0))
        gae=0.
        for i in reversed(range(T)):
            s=steps[i]; terminal=bool(s['done'])
            nextval=0. if terminal else (float(steps[i+1]['value']) if i+1<T else last)
            delta=float(s['reward'])+0.99*nextval*(not terminal)-float(s['value'])
            gae=delta+0.99*0.95*(not terminal)*gae
            adv[i]=gae
        for i,s in enumerate(steps):
            transitions.append(s);advantages.append(float(adv[i]));returns.append(float(adv[i]+s['value']))
        stats.append({'seed':r.get('seed'),'steps':T,'episode_reward':r.get('episode_reward'),
                      'success':r.get('success'),'stop_reason':r.get('stop_reason')})
    x=torch.tensor([s['obs'] for s in transitions],dtype=torch.float32)
    a=torch.tensor([s['action'] for s in transitions],dtype=torch.float32)
    oldlogp=torch.tensor([s['logp'] for s in transitions],dtype=torch.float32)
    masks=torch.tensor([s['action_mask'] for s in transitions],dtype=torch.float32)
    if masks.shape!=a.shape or not torch.all((masks==0)|(masks==1)) or (masks.sum(1)<1).any():
        raise RuntimeError('动作安全mask格式错误，拒绝更新')
    with torch.no_grad():
        lp0,_,_=net.score_actions(x,a,masks)
        err=float((lp0-oldlogp).abs().max())
        if err>1e-3:raise RuntimeError(f'on-policy动作logp与当前checkpoint不匹配 max_abs={err:.6g}')
    ret=torch.tensor(returns,dtype=torch.float32)
    adv=torch.tensor(advantages,dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(ret).all() or not torch.isfinite(oldlogp).all():
        raise RuntimeError('训练数据包含NaN/Inf，拒绝更新')
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    torch.manual_seed(seed)
    diagnostics=[]
    for epoch in range(epochs):
        for idx in torch.randperm(len(x)).split(minibatch):
            lp,entropy,value=net.score_actions(x[idx],a[idx],masks[idx])
            ratio=torch.exp(lp-oldlogp[idx])
            clipratio=ratio.clamp(.8,1.2)
            policy_loss=-torch.minimum(ratio*adv[idx],clipratio*adv[idx]).mean()
            value_loss=.5*(value-ret[idx]).square().mean()
            loss=policy_loss+.5*value_loss-.002*entropy.mean()
            if not torch.isfinite(loss):raise RuntimeError('PPO loss非有限数值')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
            diagnostics.append([float(policy_loss.detach()),float(value_loss.detach()),
                               float(entropy.mean().detach()),float(((ratio-1).abs()>.2).float().mean().detach())])
    save_ckpt(save_path,net,opt,c['updates']+1,c['episodes']+len(paths))
    d=np.asarray(diagnostics)
    summary={'schema':SCHEMA,'updates':c['updates']+1,'episodes_total':c['episodes']+len(paths),
             'new_episodes':len(paths),'timesteps':len(transitions),
             'mean_episode_reward':float(np.mean([z['episode_reward'] for z in stats])),
             'successes':sum(bool(z['success']) for z in stats),
             'policy_loss':float(d[:,0].mean()),'value_loss':float(d[:,1].mean()),
             'gaussian_entropy_approx':float(d[:,2].mean()),'clip_fraction':float(d[:,3].mean()),
             'episodes':stats}
    print('[V17_17_PPO_UPDATE_PASS] '+json.dumps(summary,ensure_ascii=False),flush=True)
    # 训练长程日志与checkpoint配对；PPO更新后写入，便于断点恢复与绘制学习曲线。
    sp=Path(save_path).with_suffix('.summary.json')
    sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init');sp.add_argument('--out',type=Path,required=True);sp.add_argument('--seed',type=int,default=0)
    sp.add_argument('--warmstart-from',type=Path,default=None,help='可选: 从v17.7/v17.8/v17.9同结构权重迁移网络参数（不继承优化器/轮次）')
    sp=sub.add_parser('update');sp.add_argument('--checkpoint',type=Path,required=True)
    sp.add_argument('--rollouts',type=Path,nargs='+',required=True)
    sp.add_argument('--out',type=Path,required=True);sp.add_argument('--seed',type=int,default=0)
    sp.add_argument('--epochs',type=int,default=4)
    sp.add_argument('--minibatch',type=int,default=64)
    sp=sub.add_parser('inspect');sp.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='init':
        if a.out.exists():raise FileExistsError('拒绝覆盖现有权重: '+str(a.out))
        torch.manual_seed(a.seed)
        net=ActorCritic()
        if a.warmstart_from is not None:
            raise ValueError('v26观测语义已改变，禁止从旧版PPO warmstart；请全新init')
            # 这只迁移同为37维的Actor/Critic参数。训练任务奖励和动作尺度已经变化，
            # 所以重置PPO优化器和训练轮数；绝不把旧轨迹当新任务的on-policy数据。
            old=torch.load(a.warmstart_from,map_location='cpu',weights_only=True)
            if old.get('schema') not in ('v17_7b_obs37_q_dq_ray_prev_contact_feedback_phase_v1',
                                         'v17_7c_obs37_near_mouth_reward_v2',
                                         'v17_8_obs37_training_first_v1',
                                         'v17_9_obs37_contact_learning_v1'):
                raise ValueError('迁移只支持v17.7-B/C的37D PPO checkpoint')
            net.load_state_dict(old['model_state_dict'],strict=True)
            print('[V17_7C_WARMSTART_PASS] 已迁移37D模型参数；优化器及训练次数重新初始化',flush=True)
        opt=torch.optim.Adam(net.parameters(),lr=3e-4)
        save_ckpt(a.out,net,opt,0,0)
        print('[V17_17_PPO_INIT_PASS]',str(a.out))
    elif a.cmd=='inspect':
        _,c=load_ckpt(a.checkpoint)
        print('[V17_17_PPO_CHECKPOINT]',{'updates':c['updates'],'episodes':c['episodes'],'schema':c['schema']})
    else:
        if a.out.exists():raise FileExistsError('拒绝覆盖已有更新权重: '+str(a.out))
        update(a.checkpoint,a.rollouts,a.out,a.epochs,a.minibatch,a.seed)

if __name__=='__main__':main()
