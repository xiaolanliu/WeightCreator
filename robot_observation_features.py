"""Visible candidate overlap with the proprioceptively located robot hand."""
import numpy as np


def robot_overlap(sensor,candidates,boxes,margin):
    x,y,z,w=np.asarray(sensor['eef_quat'],dtype=float)/np.linalg.norm(sensor['eef_quat'])
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    boxes=np.asarray(boxes);overlap=[]
    for candidate in candidates:
        x0,y0,x1,y1=candidate['box'];points=sensor['world'][y0:y1+1,x0:x1+1].reshape(-1,3)
        points=points[np.all((points>=np.asarray(candidate['lower'])-.003)&(points<=np.asarray(candidate['upper'])+.003),axis=1)]
        local=(points-sensor['eef_pos'])@rotation
        inside=np.any(np.all((local[:,None]>=boxes[None,:,:3]-margin)&(local[:,None]<=boxes[None,:,3:]+margin),axis=2),axis=1)
        overlap.append(float(inside.mean()) if len(inside) else 0.)
    return overlap
