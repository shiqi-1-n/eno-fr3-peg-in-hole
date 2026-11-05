#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17.3 粗定位MLP行为克隆：20D孔口测距+名义工具模型 -> 3D局部XYZ归一化动作。

全部训练样本是同步的学生可用观测o_t、专家动作a_t，不读取真值。
checkpoint包含训练集的均值方差，以便未来在线执行严格一致预处理。
这是监督学习权重；成功率必须通过闭环PhysX环境测，不可用离线MSE替代。
"""
import argparse,json,random,os,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset


class BCActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),
                               nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3),nn.Tanh())
    def forward(self,x):return self.net(x)


def train(src,out,epochs,batch_size,lr,seed):
    if out.exists():raise FileExistsError('拒绝覆盖训练结果:'+str(out))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # 模型很小，默认CPU避免DGX GB10旧torch内核架构兼容问题；可自行启用CUDA。
    device='cuda' if os.environ.get('ENO_BC_CUDA','0')=='1' and torch.cuda.is_available() else 'cpu'
    z=np.load(src,allow_pickle=False)
    xtr=np.asarray(z['x_train'],dtype=np.float32);xv=np.asarray(z['x_val'],dtype=np.float32)
    ytr=np.asarray(z['y_train'],dtype=np.float32);yv=np.asarray(z['y_val'],dtype=np.float32)
    if not (xtr.shape[1:]==(20,) and xv.shape[1:]==(20,) and ytr.shape[1:]==(3,) and yv.shape[1:]==(3,)):
        raise RuntimeError('BC数据维度不符合20->3')
    mean=xtr.mean(axis=0);std=np.maximum(xtr.std(axis=0),.02)
    xt=torch.from_numpy((xtr-mean)/std);xvt=torch.from_numpy((xv-mean)/std)
    yt=torch.from_numpy(ytr);yvt=torch.from_numpy(yv)
    loader=DataLoader(TensorDataset(xt,yt),batch_size=batch_size,shuffle=True)
    actor=BCActor().to(device)
    optimizer=torch.optim.Adam(actor.parameters(),lr=lr,weight_decay=1e-5)
    early_best=float('inf');best_state=None;history=[]
    for epoch in range(1,epochs+1):
        actor.train();running=[]
        for xs,ys in loader:
            xs=xs.to(device);ys=ys.to(device)
            pred=actor(xs)
            loss=torch.mean((pred-ys).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),2.)
            optimizer.step();running.append(float(loss.item()))
        actor.eval()
        with torch.no_grad():
            vv=torch.mean((actor(xvt.to(device))-yvt.to(device)).square()).item()
        history.append({'epoch':epoch,'train_mse':float(np.mean(running)),'val_mse':vv})
        if vv<early_best:
            early_best=vv
            best_state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        if epoch%10==0 or epoch==1 or epoch==epochs:
            print(f'[BC] epoch={epoch}/{epochs} train_mse={np.mean(running):.6f} val_mse={vv:.6f}',flush=True)
    if best_state is None:raise RuntimeError('无有效BC权重')
    out.mkdir(parents=True,exist_ok=False)
    torch.save({'model_state_dict':best_state,'observation_mean':torch.from_numpy(mean.astype(np.float32)),
                'observation_std':torch.from_numpy(std.astype(np.float32)),'feature_schema':'v17_3_hole_ray_q7_dq7_nominal_relative3_prev_action3',
                'action_dim':3,'input_dim':20,'training_episodes':len(z['train_seeds']),
                'validation_episodes':len(z['val_seeds']),'seed':seed,'val_mse':early_best},out/'bc_actor_best.pt')
    (out/'train_history.json').write_text(json.dumps({'device':device,'best_val_mse':early_best,'history':history,
      'caution':'离线MSE不是任何形式的机器人成功率，必须做闭环PhysX评估'},ensure_ascii=False,indent=2),encoding='utf8')
    print('[V17_3_COARSE_BC_TRAIN_PASS] best_val_mse=',early_best,' checkpoint=',out/'bc_actor_best.pt',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=20261111);a=p.parse_args()
    if not 1<=a.epochs<=10000:raise SystemExit('epochs非法')
    train(a.dataset,a.output,a.epochs,a.batch_size,a.lr,a.seed)
