"""Live carried-body center from calibrated RGB-D and robot proprioception."""
import numpy as np


def carried_center(observations,eef,grasp_to_base,extent,settings):
    g=np.asarray(settings);eef=np.asarray(eef);cloud=[]
    for sensor in observations:
        points=sensor['world'].reshape(-1,3)
        keep=np.linalg.norm(points[:,:2]-eef[:2],axis=1)<g[4]*max(extent[:2])
        keep&=(points[:,2]>eef[2]-g[3]*grasp_to_base)&(points[:,2]<eef[2]-g[2]*grasp_to_base)
        cloud.extend(points[keep])
    detail=dict(points=len(cloud),applied=False)
    if len(cloud)<g[6]:return None,detail
    points=np.unique(np.floor(np.asarray(cloud)/g[8]).astype(int),axis=0)*g[8]+g[8]/2
    bounds=np.quantile(points[:,:2],[g[5],1-g[5]],axis=0)
    center=bounds.mean(0)
    detail.update(voxels=len(points),observed_center=center.tolist(),bounds=bounds.tolist(),eef_to_center=(center-eef[:2]).tolist())
    return center,detail
