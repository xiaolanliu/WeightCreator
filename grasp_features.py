"""Observable grasp-axis features from an RGB-D proposal, without object state."""
import numpy as np


def source_axes(sensor, candidate):
    x0,y0,x1,y1=candidate['box']
    points=sensor['world'][y0:y1+1,x0:x1+1].reshape(-1,3)
    lo=np.asarray(candidate['lower'])-.003
    hi=np.asarray(candidate['upper'])+.003
    points=points[np.all((points>=lo)&(points<=hi),axis=1)]
    if len(points)<20:
        extent=np.asarray(candidate['upper'])[:2]-np.asarray(candidate['lower'])[:2]
        axis=np.eye(2)[int(np.argmin(extent))]
        return dict(minor_axis=axis.tolist(),major_width=float(extent.max()),aspect=float(extent.max()/max(extent.min(),1e-4)),points=len(points))
    xy=points[:,:2]-np.median(points[:,:2],axis=0)
    values,vectors=np.linalg.eigh(xy.T@xy/len(xy))
    aspect=float(np.sqrt(values[1]/max(values[0],1e-8)))
    projection=xy@vectors[:,1]
    width=float(np.quantile(projection,.98)-np.quantile(projection,.02))
    minor_projection=xy@vectors[:,0]
    minor_width=float(np.quantile(minor_projection,.98)-np.quantile(minor_projection,.02))
    return dict(minor_axis=vectors[:,0].tolist(),major_width=width,minor_width=minor_width,aspect=aspect,points=len(points))


def palm_clearance_features(sensor,source,geometry,jaw_axis,offset_fractions,palm_bounds):
    """Fixed candidate actions evaluated against visible points; no weight search."""
    jaw=np.asarray(jaw_axis);jaw=jaw/max(np.linalg.norm(jaw),1e-6)
    tangent=np.array([-jaw[1],jaw[0]])
    center=np.asarray(source['position']).copy();center[2]=source['upper'][2]-.018
    shifts=np.asarray(offset_fractions)[:,None]*geometry['major_width']*tangent[None,:]
    points=sensor['world'].reshape(-1,3)
    points=points[(abs(points[:,0]-center[0])<.25)&(abs(points[:,1]-center[1])<.25)]
    points=points[(points[:,2]>center[2]+palm_bounds[2])&(points[:,2]<center[2]+palm_bounds[3])]
    # Density-independent 4mm voxels keep the score consistent across camera sizes.
    points=np.unique(np.floor(points/.004).astype(int),axis=0).astype(float)*.004+.002
    features=[]
    for shift in shifts:
        delta=points[:,:2]-center[:2]-shift
        collisions=np.count_nonzero((abs(delta@jaw)<palm_bounds[0])&(abs(delta@tangent)<palm_bounds[1]))
        features.append([float(collisions),float(np.linalg.norm(shift))])
    return np.column_stack([shifts,np.zeros(len(shifts))]),np.asarray(features)


def bowl_clearance_features(sensor,source,target,reference,head,target_sensor=None):
    """Fixed authored rim actions scored at source and destination; no fitting."""
    from scipy.spatial.transform import Rotation
    target_sensor=sensor if target_sensor is None else target_sensor
    extent=np.asarray(source['upper'])-source['lower'];base=head.grasp_offset[0].numpy();default=base+head.grasp_offset[1].numpy()*extent
    bounds=head.palm_bounds.numpy();source_z=float(head.condition[2]);target_z=float(head.condition[5]+head.target_weight[5,2,6])
    boxes=head.bowl_collision_boxes.numpy() if hasattr(head,'bowl_collision_boxes') else None
    clouds=[]
    for index,(observed,z,center) in enumerate([(sensor,source_z,source['position']),(target_sensor,target_z,target['position'])]):
        points=observed['world'].reshape(-1,3)
        zlo,zhi=(bounds[2],bounds[3]) if boxes is None else (-.05,.18)
        if hasattr(head,'bowl_sweep_height'):zhi+=float(head.bowl_sweep_height)
        keep=(np.linalg.norm(points[:,:2]-np.asarray(center)[:2],axis=1)<.3)&(points[:,2]>z+zlo)&(points[:,2]<z+zhi)
        if index==0 and hasattr(head,'source_obstacle_margin'):
            margin=float(head.source_obstacle_margin)
            intended_contact=np.all((points>=np.asarray(source['lower'])-margin)&(points<=np.asarray(source['upper'])+margin),axis=1)
            keep &= ~intended_contact
        points=np.unique(np.floor(points[keep]/.004).astype(int),axis=0).astype(float)*.004+.002;clouds.append(points)
    shifts=[];poses=[];features=[];details=[]
    for dx,dy,yaw in head.bowl_candidates.tolist():
        delta=base+np.array([dx*extent[0],dy*extent[1],0.]);shift=delta-default
        pose=Rotation.from_euler('z',yaw).as_matrix()@reference;jaw=(pose@head.closing_axis.numpy())[:2];jaw/=max(np.linalg.norm(jaw),1e-6);tangent=np.array([-jaw[1],jaw[0]])
        centers=[np.asarray(source['position'])[:2]+delta[:2],np.asarray(target['position'])[:2]+(head.carry_offset.numpy()@delta)[:2]]
        counts=[]
        for index,(points,center,z) in enumerate(zip(clouds,centers,[source_z,target_z])):
            if boxes is None:
                relative=points[:,:2]-center;collision=(abs(relative@jaw)<bounds[0])&(abs(relative@tangent)<bounds[1])
            else:
                active_boxes=boxes.copy()
                if index==1 and hasattr(head,'bowl_destination_closure'):
                    jaw_shift=-np.sign(active_boxes[1:,1]+active_boxes[1:,4])*float(head.bowl_destination_closure)
                    active_boxes[1:,1]+=jaw_shift;active_boxes[1:,4]+=jaw_shift
                relative=(points-np.array([*center,z]))@pose
                collision=np.any(np.all((relative[:,None,:]>=active_boxes[None,:,:3])&(relative[:,None,:]<=active_boxes[None,:,3:]),axis=2),axis=1)
                if hasattr(head,'bowl_sweep_height'):
                    for fraction in (.5,1.):
                        relative=(points-np.array([*center,z+fraction*float(head.bowl_sweep_height)]))@pose
                        collision|=np.any(np.all((relative[:,None,:]>=active_boxes[None,:,:3])&(relative[:,None,:]<=active_boxes[None,:,3:]),axis=2),axis=1)
            counts.append(int(np.count_nonzero(collision)))
        feature=[sum(counts),float(np.linalg.norm(shift[:2]))]
        view=np.asarray(sensor['camera_position'])[:2]-np.asarray(source['position'])[:2];view/=max(np.linalg.norm(view),1e-6)
        radial=delta[:2]-base[:2];alignment=float(view@radial/max(np.linalg.norm(radial),1e-6))
        if hasattr(head,'bowl_view_score'):feature.append(alignment)
        shifts.append(shift);poses.append(pose);features.append(feature);details.append(dict(source_voxels=counts[0],target_voxels=counts[1],rim_delta=delta.tolist(),yaw=yaw,view_alignment=alignment))
    return np.asarray(shifts),poses,np.asarray(features),details
