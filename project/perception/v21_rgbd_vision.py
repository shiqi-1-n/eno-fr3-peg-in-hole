#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 RGB-D视觉几何（精度验收版）（学生唯一的棒孔位姿来源）。

输入：两台已固定标定相机的RGB、光轴深度、K、相机外参。
输出：棒实际尖端相对孔口的位置差（XYZ）和棒轴XY方向（5D）。

严格禁止：真实棒孔pose、随机安装扰动、USD物体路径、episode_truth作为估计输入。
已知的棒直径20mm、孔口高度和相机标定属于物理设备参数，不是随机真值。

v19.2错误：把多个橙色连通区域合并，76像素的另一处机器人橙色表面
与棒本体相距0.76米，导致PCA棒轴偏转和500mm级错误。

修复：只选最大连续且深度有效的橙色区域；从实际深度恢复圆柱侧表面，
拟合棒轴；在已知孔口Z平面利用蓝色环形上表面恢复内/外圆，拟合孔中心。
提供拟合残差、双视角一致性等质量指标。未通过DGX的毫米级真值验收之前，
不能声称可用于正式精细插入BC。本模块在独立数据上离线运行，不启动PhysX。
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_erosion
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PEG_RADIUS_M = .010  # 已知实体半径，不是pose真值
HOLE_INNER_R_M = .0115  # 已知孔半径

class VisualInvalid(RuntimeError):
    pass


def _as_array(rgb, depth):
    rgb=np.asarray(rgb)
    depth=np.asarray(depth).squeeze()
    if rgb.ndim!=3 or rgb.shape[2]<3 or depth.shape!=rgb.shape[:2]:
        raise VisualInvalid('RGB和深度形状不匹配')
    if np.max(rgb)<=1.1:rgb=(rgb*255.).clip(0,255).astype(np.uint8)
    return rgb,depth


def diagnostic_masks(rgb, depth):
    """颜色阈值只是可见表面检测，不是语义标签或物体位姿接口。"""
    rgb,depth=_as_array(rgb,depth)
    r,g,b=(rgb[...,i].astype(float) for i in range(3))
    valid=np.isfinite(depth)&(depth>.08)&(depth<2.5)
    peg=(r>75)&(r>g*1.25)&(g>b*1.25)&valid
    hole=(b>55)&(b>r*1.28)&(b>g*1.08)&valid
    return peg,hole,{'peg_mask_pixels':int(peg.sum()),'hole_mask_pixels':int(hole.sum()),
        'finite_depth_pixels':int(valid.sum()),'rgb_shape':list(rgb.shape),
        'rgb_range':[int(rgb.min()),int(rgb.max())]}


def _largest_region(mask, min_pixels):
    """8邻域选择最大的连续物体区域：避免把另一只机械臂上的橙色像素当棒。"""
    lab,count=label(mask,structure=np.ones((3,3),np.uint8))
    if count==0:raise VisualInvalid('当前视角没有目标颜色连通区域')
    counts=np.bincount(lab.ravel());counts[0]=0
    idx=int(np.argmax(counts))
    if counts[idx]<min_pixels:raise VisualInvalid(f'目标颜色最大连续区域不足 {min_pixels} 像素，实际={counts[idx]}')
    return lab==idx


def _cam_calib(packet):
    _,_,K,pos,quat=packet
    K=np.asarray(K,float);pos=np.asarray(pos,float);quat=np.asarray(quat,float)
    if K.shape!=(3,3) or pos.shape!=(3,) or quat.shape!=(4,) or min(K[0,0],K[1,1])<=0:
        raise VisualInvalid('相机标定内外参非法')
    R=Rotation.from_quat((quat[1],quat[2],quat[3],quat[0])).as_matrix()
    return K,pos,R


def _pixel_rays(u, v, K, R):
    # Isaac RGB图像行向下，OpenGL相机局部右X、上Y、光轴-Z。
    local=np.stack(((u-K[0,2])/K[0,0],-(v-K[1,2])/K[1,1],-np.ones_like(u)),axis=1)
    return local@R.T


def _backproject(mask, depth, K, pos, R):
    v,u=np.nonzero(mask);d=depth[v,u]
    if len(v)==0:raise VisualInvalid('没有可反投影的像素')
    rays=_pixel_rays(u,v,K,R)
    # distance_to_image_plane为光轴-Z深度，而不是单位射线欧氏距离。
    return pos+rays*d[:,None]


def _project_pixels_to_z(mask, z, K, pos, R):
    """用可见边界像素与已知孔口平面求交；禁止使用目标孔的真实XY。"""
    v,u=np.nonzero(mask)
    rays=_pixel_rays(u,v,K,R)
    ok=np.abs(rays[:,2])>1e-5
    rays=rays[ok]
    scale=(z-pos[2])/rays[:,2]
    ok2=(scale>.01)&(scale<3.)
    return pos+rays[ok2]*scale[ok2,None]


def _estimate_hole(camera_packets, mouth_z):
    """对孔口蓝色环的上平面进行内/外边缘圆拟合，避免把蓝色实体表面中位数当孔中心。

    圆的直径不强行固定，分别拟合内、外边缘，并要求中心一致。即使只看到孔壁
    上表面的一部分，也不会从随机孔真实pose补全信息。
    """
    candidates=[];qualities={}
    for name, packet in camera_packets.items():
        rgb,depth=_as_array(packet[0],packet[1]);K,pos,R=_cam_calib(packet)
        _,holemask,_=diagnostic_masks(rgb,depth)
        try:
            holemask=_largest_region(holemask,80)
            pts=_backproject(holemask,depth,K,pos,R)
            v,u=np.nonzero(holemask)
            # 只有靠近孔口平面的蓝色表面才形成有实体厚度的环。
            top=np.zeros_like(holemask)
            close=np.abs(pts[:,2]-mouth_z)<.0015
            top[v[close],u[close]]=True
            top=_largest_region(top,50)
            filled=binary_fill_holes(top)
            inner=filled&~top
            inner=_largest_region(inner,12)
            rims=[]
            for region, radius_range in ((inner,(.007,.018)),(filled,(.020,.060))):
                edge=region&~binary_erosion(region)
                world=_project_pixels_to_z(edge,mouth_z,K,pos,R)
                if len(world)<18:raise VisualInvalid('孔口环边缘可见像素不足')
                xy=world[:,:2]
                center0=np.median(xy,axis=0)
                radius0=np.median(np.linalg.norm(xy-center0,axis=1))
                fit=least_squares(lambda x:np.linalg.norm(xy-x[:2],axis=1)-x[2],
                    np.r_[center0,radius0],loss='soft_l1',f_scale=.001,max_nfev=80)
                if not radius_range[0]<fit.x[2]<radius_range[1]:
                    raise VisualInvalid(f'孔口边缘半径异常: {fit.x[2]*1000:.1f}mm')
                if np.median(np.abs(fit.fun))>.002:
                    raise VisualInvalid('孔口边缘圆拟合残差超过2mm')
                rims.append(fit.x)
            if np.linalg.norm(rims[0][:2]-rims[1][:2])>.003:
                raise VisualInvalid('孔口内外环圆心不一致')
            xy=(rims[0][:2]+rims[1][:2])*.5
            candidates.append((name,xy))
            qualities[name]={'inner_radius_mm':float(rims[0][2]*1000),
                'outer_radius_mm':float(rims[1][2]*1000),'center_m':xy.tolist()}
        except VisualInvalid as exc:
            qualities[name]={'invalid':str(exc)}
    if not candidates:raise VisualInvalid('两相机均不能形成可信孔口圆环:'+str(qualities))
    centers=np.asarray([c[1] for c in candidates])
    disagreement=float(np.max(np.linalg.norm(centers-centers.mean(axis=0),axis=1))*2) if len(centers)>1 else 0.
    if disagreement>.006:
        raise VisualInvalid(f'双相机孔中心差异={disagreement*1000:.2f}mm，拒绝融合')
    return np.median(centers,axis=0),{'hole_views':qualities,'hole_crossview_disagreement_mm':disagreement*1000}


def _estimate_peg(camera_packets):
    rods=[];full_rods=[];qualities={}
    for name,packet in camera_packets.items():
        rgb,depth=_as_array(packet[0],packet[1]);K,pos,R=_cam_calib(packet)
        pegmask,_,_=diagnostic_masks(rgb,depth)
        try:
            main=_largest_region(pegmask,100)
            # 深度和颜色轮廓边缘的混合像素常来自不同表面，不能当作圆柱侧面
            # 来拟合轴线。内缩像素只用于拟合轴线；棒尖候选仍来自完整可见区域。
            # 必须继续用离线真值审计实际末端误差，残差小≠实际tip准确。
            inner=binary_erosion(main,iterations=5)
            if int(inner.sum())<100:
                raise VisualInvalid('当前棒可见内表面过窄，不能可靠恢复圆柱轴线')
            pts=_backproject(inner,depth,K,pos,R)
            full_pts=_backproject(main,depth,K,pos,R)
            median_z=np.median(pts[:,2]);keep=np.abs(pts[:,2]-median_z)<.07
            pts=pts[keep]
            full_pts=full_pts[np.abs(full_pts[:,2]-median_z)<.07]
            if len(pts)<100 or len(full_pts)<100:
                raise VisualInvalid('棒侧面或末端可见的有效深度点不足')
            rods.append(pts)
            full_rods.append(full_pts)
            qualities[name]={'selected_pixels':int(main.sum()),
                'interior_pixels_for_axis':int(inner.sum()),
                'axis_fit_points':int(len(pts)),
                'visible_points_for_tip':int(len(full_pts))}
        except VisualInvalid as exc:
            qualities[name]={'invalid':str(exc)}
    if not rods:raise VisualInvalid('两相机均无可用的棒表面连通区域:'+str(qualities))
    pts=np.concatenate(rods)
    pts_all=np.concatenate(full_rods)
    if len(pts)>12000:pts=pts[np.linspace(0,len(pts)-1,12000,dtype=int)]
    center=np.median(pts,axis=0)
    _,_,V=np.linalg.svd(pts-center,full_matrices=False)
    axis0=V[0]
    if axis0[2]>0:axis0=-axis0
    if -axis0[2]<.55:raise VisualInvalid('棒表面主轴倾斜超过初始可辨认范围')
    def residual(x):
        a=np.array([x[2],x[3],-1.]);a/=np.linalg.norm(a)
        d=pts-np.array([x[0],x[1],center[2]])
        return np.linalg.norm(d-np.outer(d@a,a),axis=1)-PEG_RADIUS_M
    guess=np.r_[center[:2],axis0[0]/(-axis0[2]),axis0[1]/(-axis0[2])]
    fit=least_squares(residual,guess,loss='soft_l1',f_scale=.002,max_nfev=80)
    a=np.array([fit.x[2],fit.x[3],-1.]);a/=np.linalg.norm(a)
    if -a[2]<.70:raise VisualInvalid('棒轴线圆柱拟合朝向不合理')
    residual_mm=float(np.median(abs(residual(fit.x)))*1000)
    if residual_mm>1.0:raise VisualInvalid(f'棒侧面拟合残差{residual_mm:.2f}mm > 1.0mm：目前无法证明亚毫米精度，不输出给PPO')
    axis_point=np.array([fit.x[0],fit.x[1],center[2]])
    # 用完整可见圆柱点的远端估计tip，但它仍可能因端面遮挡存在系统偏差。
    # 该估计必须接受OFFLINE_ONLY真实tip误差检验，不能凭柱面残差放行。
    along=(pts_all-axis_point)@a
    # v19.2错误：用上端点+40mm猜测棒尖，即使只看见棒的一部分也会导致偏差。
    # 对目前的几何采用可见最前端沿轴投影的99.5%位置；这是可见表面近似，
    # 不保证底端全可见。部署前必须用离线真值校验，绝不悄悄回填真值。
    tip=axis_point+float(np.quantile(along,.995))*a
    return tip,a,{'peg_views':qualities,'peg_points':int(len(pts)),
         'peg_fit_residual_mm':residual_mm,
         'tip_method':'visible_endpoint_quantile_not_ground_truth',
         'peg_tip_estimated_m':tip.tolist(),
         'peg_axis_world':a.tolist()}


def estimate(camera_packets,hole_mouth_height_m=.4):
    if not camera_packets:raise VisualInvalid('没有RGB-D相机帧')
    hole_xy,hq=_estimate_hole(camera_packets,float(hole_mouth_height_m))
    tip,axis,pq=_estimate_peg(camera_packets)
    rel=np.array([hole_xy[0]-tip[0],hole_xy[1]-tip[1],
         float(hole_mouth_height_m)-tip[2],axis[0],axis[1]],float)
    if not np.isfinite(rel).all():raise VisualInvalid('视觉5维相对位姿不是有限值')
    # 这只是相机几何算法内部的拟合置信度，不能代替与物理真值的独立误差检验。
    # 不能仅因为返回了5维数值，就称其可以实现0.5mm插入控制。
    return rel,{**hq,**pq}
