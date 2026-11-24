"""Refine a compartment center from observed wall bands, without object state."""
import numpy as np

def observed_back_center(observations,lower,upper,center,slot,settings):
    lo=np.asarray(lower);hi=np.asarray(upper);center=np.asarray(center).copy();slot=np.asarray(slot)
    g=np.asarray(settings);axis=int(np.argmax(abs(slot[:2])));other=1-axis;cloud=[]
    for sensor in observations:
        points=sensor['world'].reshape(-1,3)
        keep=np.all((points[:,:2]>=lo[:2]-2*g[0])&(points[:,:2]<=hi[:2]+2*g[0]),axis=1)
        keep&=abs(points[:,other]-center[other])<(hi[other]-lo[other])*g[1]
        keep&=(points[:,2]>lo[2]+g[2])&(points[:,2]<hi[2]+g[3])
        cloud.extend(points[keep])
    detail=dict(applied=False,axis=axis,points=len(cloud))
    if not cloud:return center,detail
    points=np.unique(np.floor(np.asarray(cloud)/g[0]).astype(int),axis=0)*g[0]+g[0]/2
    edges=np.arange(lo[axis]-3*g[0],hi[axis]+4*g[0],g[0]);counts,_=np.histogram(points[:,axis],edges)
    occupied=np.flatnonzero(counts>=g[4]);groups=np.split(occupied,np.flatnonzero(np.diff(occupied)>1)+1)
    bands=[(float(edges[group[0]]),float(edges[group[-1]+1])) for group in groups if len(group)]
    detail['wall_bands']=bands
    if len(bands)<g[7]:return center,detail
    left,right=bands[:2] if slot[axis]<0 else bands[-2:]
    inside=[left[1]+g[5],right[0]-g[5]]
    if inside[1]-inside[0]<g[6]:return center,detail
    old=float(center[axis]);center[axis]=sum(inside)/2
    detail.update(applied=True,old_center=old,new_center=float(center[axis]),interior=inside)
    return center,detail
